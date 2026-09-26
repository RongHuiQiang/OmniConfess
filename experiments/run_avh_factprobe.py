#!/usr/bin/env python3
"""
FactProbe experiment for AVHBench.

Two-step correction:
  Step 1 — Probe: ask the model what the correct value is, given evidence.
  Step 2 — Replace: rewrite the answer substituting only the flagged span.

Output: outputs/lv2_avh_factprobe/audio_only_n50.jsonl
"""
import argparse
import json
import os
import re
import sys
import importlib.util
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# scripts/ lives in the OUTER OmniConfess; the actual Python package lives
# one level deeper at OmniConfess/OmniConfess/.
OUTER_ROOT = Path(__file__).resolve().parent.parent   # .../OmniConfess
INNER_ROOT = OUTER_ROOT / "OmniConfess"               # .../OmniConfess/OmniConfess

# Sanity check: model_utils must be in INNER_ROOT
assert (INNER_ROOT / "methods/shared/model_utils.py").exists(), \
    f"model_utils not found under {INNER_ROOT}"

# Add both to sys.path so local imports work
sys.path.insert(0, str(INNER_ROOT))
sys.path.insert(0, str(OUTER_ROOT))

# Data files live in the OUTER tree
AVH_JSON   = OUTER_ROOT / "OmniHalluBench/avhbench/avhbench.json"
PROBE_PATH = OUTER_ROOT / "outputs/cf_probe/avh_probe_bigN_shard0of1.jsonl"
OUT_DIR    = OUTER_ROOT / "outputs/lv2_avh_factprobe"
OUT_FILE   = OUT_DIR / "audio_only_n50.jsonl"
SUMMARY_F  = OUT_DIR / "SUMMARY_factprobe.md"

# model_utils and localizer modules live in INNER_ROOT
MU_PATH    = INNER_ROOT / "methods/shared/model_utils.py"

# ---------------------------------------------------------------------------
# Helpers — mirror run_avh_lv2.py exactly
# ---------------------------------------------------------------------------

def _load_module_direct(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_avh_probe(probe_path, avhbench_data, signal_mode="audio_only"):
    """Load & deduplicate probe, attach question, compute delta_img_tok."""
    uid_to_q = {}
    for item in avhbench_data:
        uid = item["uid"]  # e.g. "avhbench_000074"
        num = str(int(uid.split("_")[-1]))
        uid_to_q[num] = item["question"]

    seen   = {}
    records = []
    with open(probe_path) as f:
        for line in f:
            rec = json.loads(line)
            rid = rec["id"]
            if rid in seen:
                continue
            seen[rid] = True
            records.append(rec)

    result = []
    for rec in records:
        d_v = rec.get("delta_video", 0.0)
        d_a = rec.get("delta_audio", 0.0)
        if signal_mode == "combined":
            shim = d_v + d_a
        elif signal_mode == "video_only":
            shim = d_v
        else:  # audio_only
            shim = d_a
        rec["delta_img_tok"] = shim
        rec["question"]      = uid_to_q.get(rec["id"], "")
        result.append(rec)
    return result


def _infer_fact_type(span_text: str) -> str:
    span_lower = span_text.lower()
    color_words = {"red","blue","green","yellow","white","black","orange",
                   "purple","pink","brown","gray","grey","silver","gold",
                   "dark","light","bright","colored","colour","color"}
    if any(w in span_lower for w in color_words):
        return "color"
    sound_words = {"sound","noise","bark","meow","chirp","beep","ring",
                   "loud","quiet","silent","music","song","voice","speak",
                   "say","talk","audio","hear"}
    if any(w in span_lower for w in sound_words):
        return "sound"
    action_words = {"run","walk","jump","sit","stand","eat","drink","hold",
                    "carry","move","dance","swim","fly","drive","ride","throw",
                    "catch","wave","clap","fall","climb","play"}
    if any(w in span_lower for w in action_words):
        return "action"
    count_words  = {"one","two","three","four","five","six","seven","eight",
                    "nine","ten","1","2","3","4","5","6","7","8","9","0",
                    "single","pair","couple","several","many","few"}
    if any(w in span_lower for w in count_words):
        return "count or number"
    return "fact"


def _token_f1(pred: str, ref: str) -> float:
    pred_toks = set(pred.lower().split())
    ref_toks  = set(ref.lower().split())
    if not pred_toks or not ref_toks:
        return 0.0
    common = pred_toks & ref_toks
    if not common:
        return 0.0
    p = len(common) / len(pred_toks)
    r = len(common) / len(ref_toks)
    return 2 * p * r / (p + r)


def _content_word_overlap(a: str, b: str) -> float:
    STOP = {"a","an","the","is","are","was","were","it","its","in","on","at",
            "to","of","and","or","not","no","do","does","did","be","been",
            "being","have","has","had","this","that","these","those","i","you",
            "he","she","we","they","my","your","his","her","our","their"}
    wa = {w for w in a.lower().split() if w not in STOP}
    wb = {w for w in b.lower().split() if w not in STOP}
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / min(len(wa), len(wb))


def _trigram_repeat(text: str) -> bool:
    words = text.split()
    if len(words) < 6:
        return False
    tris = [tuple(words[i:i+3]) for i in range(len(words)-2)]
    return len(tris) != len(set(tris))


def _lv2_accept_decision(repaired: str, base_pred: str,
                          verify_status: str = "") -> bool:
    if not repaired.strip():
        return False
    if _content_word_overlap(repaired, base_pred) < 0.25:
        return False
    if _trigram_repeat(repaired):
        return False
    if len(repaired) < 0.5 * max(len(base_pred), 1):
        return False
    if repaired.strip() in base_pred or base_pred.strip().startswith(repaired.strip()):
        return False
    REFUSAL = {"cannot","can't","unable","don't know","no information",
               "not enough","insufficient","unclear"}
    base_has_refusal = any(w in base_pred.lower() for w in REFUSAL)
    rep_has_refusal  = any(w in repaired.lower()  for w in REFUSAL)
    vs_lower = verify_status.lower()
    contradicted = any(x in vs_lower for x in
                       ("unsupported","contradicted","unsupported_by_evidence"))
    if base_has_refusal and not rep_has_refusal and not contradicted:
        return False
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit",         type=int,  default=50)
    parser.add_argument("--max-new-tokens",type=int,  default=96)
    parser.add_argument("--probe-path",    default=str(PROBE_PATH))
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ data
    print("[factprobe] Loading AVHBench …")
    with open(AVH_JSON) as f:
        avhbench_data = json.load(f)

    probe_records = load_avh_probe(args.probe_path, avhbench_data,
                                   signal_mode="audio_only")
    probe_records = sorted(probe_records, key=lambda r: int(r["id"]))
    probe_records = [r for r in probe_records if int(r["id"]) < args.limit]
    print(f"[factprobe] {len(probe_records)} probe records (limit={args.limit})")

    # ------------------------------------------------------------------ localizer + verifier
    import localizer_v2    as lv2_mod
    import evidence_verifier as ev_mod

    # ------------------------------------------------------------------ model
    print("[factprobe] Loading model …")
    # model_utils uses relative imports, must be imported as a package member
    from methods.shared import model_utils as _mu
    from methods.shared.prompts import get_system_prompt
    from methods.shared.data_loader import load_dataset, load_media_for_item
    build_inputs                         = _mu.build_inputs
    make_messages                        = _mu.make_messages
    build_user_content_with_media_tokens = _mu.build_user_content_with_media_tokens
    prepare_for_thinker                  = _mu.prepare_for_thinker
    decode_outputs                       = _mu.decode_outputs

    # warm CUDA
    _ = torch.zeros(1, device="cuda")

    shm_path = Path("/dev/shm/Qwen2.5-Omni-7B")
    if shm_path.exists():
        print(f"[factprobe] Using shm model at {shm_path}")
        processor, model = _mu.load_model(str(shm_path))
    else:
        processor, model = _mu.load_model()

    sys_prompt = get_system_prompt("avhbench", "v1")

    # Load AVHBench items via data_loader for media resolution
    AVH_DATA      = str(AVH_JSON)
    AVH_DATA_ROOT = str(OUTER_ROOT / "OmniHalluBench/avhbench")
    avh_items     = load_dataset(AVH_DATA, AVH_DATA_ROOT)
    avh_by_uid_num = {}
    for it in avh_items:
        uid = it.get("uid", "")
        try:
            num = str(int(uid.split("_")[-1]))
            avh_by_uid_num[num] = it
        except (ValueError, IndexError):
            pass

    # ------------------------------------------------------------------ resume
    done_ids = set()
    if OUT_FILE.exists():
        with open(OUT_FILE) as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["id"])
                except Exception:
                    pass
        print(f"[factprobe] Resuming — {len(done_ids)} already done")

    out_fh = open(OUT_FILE, "a")

    # ------------------------------------------------------------------ loop
    rows = []
    for rec in probe_records:
        rid = rec["id"]
        if rid in done_ids:
            # load existing row for summary
            continue

        question  = rec.get("question", "")
        base_pred = "".join(rec["tokens"])
        gt        = rec.get("gt", "")

        # build cache record
        cache_record = {
            "delta_img_tok":    rec["delta_img_tok"],
            "delta_video":      rec.get("delta_video", 0.0),
            "delta_audio":      rec.get("delta_audio", 0.0),
            "delta_text_tok":   rec.get("delta_text", 0.0),
            "classes":          rec.get("classes", []),
            "lp_full":          rec.get("lp_full", []),
            "tokens":           rec["tokens"],
            "id":               rid,
        }

        # localize
        lv2_result = lv2_mod.localize(cache_record)

        # load media  (mirrors run_avh_lv2.py media loading)
        uid    = f"avhbench_{int(rid):06d}"
        # build generate closure first (needs media handles)
        data_item = next(
            (it for it in avhbench_data
             if it["uid"] == uid), None
        )

        _gen_state = {}
        if data_item is not None:
            try:
                pv, av, iv = load_media_for_item(data_item, str(AVH_JSON),
                                                  str(OUTER_ROOT / "OmniHalluBench/avhbench"))
            except Exception:
                pv, av, iv = None, None, None
            _gen_state = {
                "pv":  pv,
                "av":  av,
                "iv":  iv,
                "sys": sys_prompt,
            }
        else:
            _gen_state = {"pv": None, "av": None, "iv": None,
                          "sys": sys_prompt}

        def _lv2_generate(prompt_text, max_new_tokens=args.max_new_tokens,
                          _gs=_gen_state):
            uc  = build_user_content_with_media_tokens(
                      _gs["pv"], _gs["av"], _gs["iv"], prompt_text)
            msgs = make_messages(_gs["sys"], uc)
            inp  = build_inputs(processor, model, msgs,
                                _gs["pv"], _gs["av"], _gs["iv"], "haloquest")
            inp  = prepare_for_thinker(inp, model)
            with torch.inference_mode():
                oids = model.thinker.generate(
                    **inp, max_new_tokens=max_new_tokens, do_sample=False)
            return decode_outputs(processor, oids,
                                  inp["input_ids"].shape[1])[0].strip()

        # verify
        ev_mod.verify_candidates(lv2_result, rec, _lv2_generate)

        rev = lv2_result.revision_candidates
        if not rev:
            row = dict(
                id=rid, question=question, gt=gt, base_pred=base_pred,
                status="no_candidates",
                probed_correction=None,
                candidate_text=None,   candidate_f1=None,
                repaired_pred=base_pred, repaired_f1=_token_f1(base_pred, gt),
                accepted=False, scene_ok=None,
                step1_obs=None,
                base_f1=_token_f1(base_pred, gt),
            )
            out_fh.write(json.dumps(row) + "\n")
            out_fh.flush()
            rows.append(row)
            continue

        cand      = rev[0]
        span_text = cand.span_text
        step1_obs = getattr(cand, "step1_observation", "") or ""
        vs        = getattr(cand, "verify_status", "") or ""

        fact_type = _infer_fact_type(span_text)

        # ---- Step 1: Fact Probe ----------------------------------------
        probe_prompt = (
            f"Given this question about a video/audio clip: {question}\n\n"
            f'The original answer states: "{span_text}"\n\n'
            f"Evidence from the clip shows: {step1_obs}\n\n"
            f"What is the correct {fact_type} in this clip? "
            f"Answer in 3-10 words only.\n"
            f"Correct value:"
        )
        probed_correction = _lv2_generate(probe_prompt, max_new_tokens=32)
        # trim to first line / sentence
        probed_correction = probed_correction.split("\n")[0].strip()
        probed_correction = re.split(r"[.!?]", probed_correction)[0].strip()

        # ---- Step 2: Targeted Replacement ------------------------------
        replace_prompt = (
            f"Original answer: {base_pred}\n"
            f'The claim "{span_text}" is incorrect.\n'
            f"Correct value based on evidence: {probed_correction}\n\n"
            f'Rewrite the answer replacing only "{span_text}" with the correct value.\n'
            f"Keep all other words unchanged.\n"
            f"Corrected answer:"
        )
        candidate_text = _lv2_generate(replace_prompt,
                                        max_new_tokens=args.max_new_tokens)
        candidate_text = candidate_text.split("\n")[0].strip()

        candidate_f1 = _token_f1(candidate_text, gt)
        base_f1      = _token_f1(base_pred, gt)

        accepted = _lv2_accept_decision(candidate_text, base_pred,
                                         verify_status=vs)

        # scene check: content word overlap ≥ 0.25 (already in accept decision,
        # but expose it separately)
        scene_ok = _content_word_overlap(candidate_text, base_pred) >= 0.25

        repaired_pred = candidate_text if accepted else base_pred
        repaired_f1   = _token_f1(repaired_pred, gt)

        obs_snippet = step1_obs[:120] if step1_obs else None

        row = dict(
            id=rid,
            question=question,
            gt=gt,
            base_pred=base_pred,
            status="processed",
            probed_correction=probed_correction,
            candidate_text=candidate_text,
            candidate_f1=candidate_f1,
            repaired_pred=repaired_pred,
            repaired_f1=repaired_f1,
            accepted=accepted,
            scene_ok=scene_ok,
            step1_obs=obs_snippet,
            base_f1=base_f1,
        )
        out_fh.write(json.dumps(row) + "\n")
        out_fh.flush()
        rows.append(row)
        print(f"  [{rid}] base_f1={base_f1:.3f}  cand_f1={candidate_f1:.3f} "
              f"  accepted={accepted}  probe='{probed_correction[:50]}'")

    out_fh.close()

    # ------------------------------------------------------------------ reload all rows for summary
    all_rows = []
    with open(OUT_FILE) as f:
        for line in f:
            try:
                all_rows.append(json.loads(line))
            except Exception:
                pass

    processed = [r for r in all_rows if r.get("status") == "processed"]
    n_total   = len(all_rows)
    n_proc    = len(processed)

    def safe_mean(vals):
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else 0.0

    base_f1_mean    = safe_mean([r["base_f1"]     for r in all_rows])
    repaired_f1_mean= safe_mean([r["repaired_f1"] for r in all_rows])
    cand_f1_mean    = safe_mean([r["candidate_f1"] for r in processed])
    delta_f1        = repaired_f1_mean - base_f1_mean
    oracle_f1_mean  = safe_mean([max(r["base_f1"], r.get("candidate_f1") or 0.0)
                                  for r in all_rows])
    n_accepted      = sum(1 for r in all_rows if r.get("accepted"))

    probe_lengths = [len(r["probed_correction"].split())
                     for r in processed if r.get("probed_correction")]
    avg_probe_len = sum(probe_lengths) / len(probe_lengths) if probe_lengths else 0.0

    # top-5 most-improved
    deltas = [(r["repaired_f1"] - r["base_f1"], r)
              for r in processed
              if r.get("accepted") and r.get("probed_correction")]
    deltas.sort(key=lambda x: -x[0])
    top5   = deltas[:5]

    summary_lines = [
        "# FactProbe Summary — AVHBench audio_only n=50",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| base_f1_mean      | {base_f1_mean:.4f} |",
        f"| repaired_f1_mean  | {repaired_f1_mean:.4f} |",
        f"| delta_f1          | {delta_f1:+.4f} |",
        f"| oracle_f1_mean    | {oracle_f1_mean:.4f} |",
        f"| cand_f1_mean (processed) | {cand_f1_mean:.4f} |",
        f"| n_total           | {n_total} |",
        f"| n_processed       | {n_proc} |",
        f"| n_accepted        | {n_accepted} |",
        f"| avg_probed_len (words) | {avg_probe_len:.1f} |",
        "",
        "## v4 baseline (for reference)",
        "oracle_f1 = 19.58",
        "",
        "## Top-5 most improved samples",
        "",
    ]
    for imp, r in top5:
        summary_lines += [
            f"### id={r['id']}  Δ={imp:+.4f}",
            f"- question:  {r['question'][:100]}",
            f"- base_pred: {r['base_pred'][:100]}",
            f"- span:      {r.get('candidate_text','')[:80]}",
            f"- probed_correction: **{r['probed_correction']}**",
            f"- repaired:  {r['repaired_pred'][:100]}",
            f"- gt:        {r['gt'][:80]}",
            "",
        ]

    SUMMARY_F.write_text("\n".join(summary_lines))
    print(f"\n[factprobe] Done. base={base_f1_mean:.4f} repaired={repaired_f1_mean:.4f} "
          f"oracle={oracle_f1_mean:.4f} delta={delta_f1:+.4f} "
          f"accepted={n_accepted}/{n_total}")
    print(f"[factprobe] Summary → {SUMMARY_F}")

    # feishu
    msg = (f"FactProbe完成: oracle={oracle_f1_mean:.2f} (v4基线19.58), "
           f"delta={delta_f1:+.2f}, "
           f"平均probed_correction长度={avg_probe_len:.1f}词")
    os.system(f'/home/tsq/.claude/feishu/feishu_send.sh "{msg}"')


if __name__ == "__main__":
    main()
