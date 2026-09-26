#!/usr/bin/env python3
"""
FactProbe v2 — Span-aware probe + evidence-grounded accept gate.

Changes vs v1 (run_avh_factprobe.py):
  1. All span/claim fields saved to output row:
       span_text, claim, verify_status, fact_type, span_kind
  2. Targeted probe prompt: evaluates the SPECIFIC span claim, not open-ended.
       "The answer says '{span_text}'. Based on the video/audio, is this correct?
        If wrong, give ONLY the corrected value in ≤5 words. If unsure, say UNSURE."
  3. Evidence-grounded accept gate (no GT, no Token-F1):
       Re-verify: ask model "Does the corrected answer align better with the
       evidence than the original?" → accept only when YES.
  4. Explicit localization_failed path: when no revision_candidates, mark
       localization_failed=True and use independent fallback (no span-targeted probe).
  5. Detailed per-sample log: base_pred, span_text, claim, verify_status, fact_type,
       probe_q, probe_answer, recheck_answer, recheck_verdict, accepted, candidate_text.

Usage:
  CUDA_VISIBLE_DEVICES=0 MMHALU_MODEL_ROOT=/tmp HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  PYTHONPATH=/dev/shm/site-packages LD_LIBRARY_PATH=/dev/shm/so_mirror \
    /home/tsq/.local/bin/python3.13 scripts/run_avh_factprobe_v2.py \
    --limit 50 --arm audio_only \
    --out outputs/lv2_avh_factprobe_v2/audio_only_n50.jsonl
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
OUTER_ROOT = Path(__file__).resolve().parent.parent   # .../OmniConfess
INNER_ROOT = OUTER_ROOT / "OmniConfess"               # .../OmniConfess/OmniConfess

assert (INNER_ROOT / "methods/shared/model_utils.py").exists(), \
    f"model_utils not found under {INNER_ROOT}"

sys.path.insert(0, str(INNER_ROOT))
sys.path.insert(0, str(OUTER_ROOT))

AVH_JSON   = OUTER_ROOT / "OmniHalluBench/avhbench/avhbench.json"
PROBE_PATH = OUTER_ROOT / "outputs/cf_probe/avh_probe_bigN_shard0of1.jsonl"
OUT_BASE   = OUTER_ROOT / "outputs/lv2_avh_factprobe_v2"

MU_PATH    = INNER_ROOT / "methods/shared/model_utils.py"

# ---------------------------------------------------------------------------
# Helpers — mirror run_avh_lv2.py / run_avh_factprobe.py exactly
# ---------------------------------------------------------------------------

def _load_module_direct(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_avh_probe(probe_path, avhbench_data, signal_mode="audio_only"):
    """Load & deduplicate probe, attach question+gt, compute delta_img_tok shim."""
    import numpy as np

    uid_to_q  = {}
    uid_to_gt = {}
    for item in avhbench_data:
        uid = item["uid"]
        num = str(int(uid.split("_")[-1]))
        uid_to_q[num]  = item.get("question", "")
        uid_to_gt[num] = item.get("answer", "")

    seen    = {}
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
        d_v = np.asarray(rec.get("delta_video", [0.0]*len(rec["tokens"])), dtype=np.float64)
        d_a = np.asarray(rec.get("delta_audio", [0.0]*len(rec["tokens"])), dtype=np.float64)
        if signal_mode == "combined":
            shim = d_v + d_a
        elif signal_mode == "video_only":
            shim = list(d_v)
        else:  # audio_only
            shim = list(d_a)
        rec["delta_img_tok"] = shim
        rec["question"]      = uid_to_q.get(rec["id"],  "")
        rec["gt"]            = uid_to_gt.get(rec["id"], "")
        result.append(rec)
    return result


def _token_f1(pred: str, ref: str) -> float:
    """Token-level F1 (offline evaluation only — never used in accept gate)."""
    pred_toks = pred.lower().split()
    ref_toks  = ref.lower().split()
    if not pred_toks or not ref_toks:
        return 0.0
    common = set(pred_toks) & set(ref_toks)
    if not common:
        return 0.0
    p = len([t for t in pred_toks if t in common]) / len(pred_toks)
    r = len([t for t in ref_toks  if t in common]) / len(ref_toks)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


# ---------------------------------------------------------------------------
# Probe & accept prompts — v2 targeted design
# ---------------------------------------------------------------------------

def _build_verdict_prompt(span_text: str, claim: str,
                           base_pred: str, question: str,
                           step1_obs: str) -> str:
    """
    Stage 1 — binary verdict: is the claim correct or wrong?
    Separates the YES/NO judgment from the corrected-value generation.
    Model answers YES (correct), NO (wrong), or UNSURE.
    """
    obs_line = (f"Evidence from the clip: {step1_obs}\n\n"
                if step1_obs and step1_obs.strip()
                else "")
    eval_target = claim if (claim and claim != span_text) else span_text
    return (
        f"Question about this video/audio clip: {question}\n\n"
        f"The original answer is: \"{base_pred}\"\n\n"
        f"{obs_line}"
        f"Focus on this specific claim from the answer: \"{eval_target}\"\n\n"
        f"Based on what you see and hear in the clip, is this claim factually correct?\n"
        f"Answer with exactly one word:\n"
        f"  YES  — the claim is correct\n"
        f"  NO   — the claim is wrong\n"
        f"  UNSURE — not enough information\n\n"
        f"Verdict (YES / NO / UNSURE):"
    )


def _build_correction_prompt(span_text: str, claim: str,
                              base_pred: str, question: str,
                              step1_obs: str) -> str:
    """
    Stage 2 — corrected value: only called when Stage 1 returned NO.
    Ask for the specific corrected value (free short answer, ≤8 words).
    """
    obs_line = (f"Evidence from the clip: {step1_obs}\n\n"
                if step1_obs and step1_obs.strip()
                else "")
    eval_target = claim if (claim and claim != span_text) else span_text
    return (
        f"Question about this video/audio clip: {question}\n\n"
        f"The original answer is: \"{base_pred}\"\n\n"
        f"{obs_line}"
        f"The claim \"{eval_target}\" is WRONG.\n"
        f"What is the correct value? Answer in 1-8 words only.\n"
        f"Do NOT repeat the question or explain. Give ONLY the corrected value.\n\n"
        f"Corrected value:"
    )


def _parse_verdict(raw: str) -> str:
    """Parse Stage 1 verdict. Returns 'YES', 'NO', or 'UNSURE'."""
    s = raw.strip().split("\n")[0].strip().upper()
    s = re.split(r"[.!,;:]", s)[0].strip()
    if s.startswith("NO"):
        return "NO"
    if s.startswith("YES"):
        return "YES"
    if "UNSURE" in s or "NOT SURE" in s or "CANNOT" in s or "CAN'T" in s:
        return "UNSURE"
    # Fallback: scan tokens
    for tok in s.split():
        tok = tok.rstrip(".,;:")
        if tok == "NO":
            return "NO"
        if tok == "YES":
            return "YES"
    return "UNSURE"


def _parse_correction(raw: str) -> str | None:
    """Parse Stage 2 correction. Returns corrected value or None if uninformative."""
    s = raw.strip().split("\n")[0].strip()
    s = re.split(r"[.!]", s)[0].strip()
    # Reject if model hedges or outputs something too long
    if not s:
        return None
    upper = s.upper()
    if any(w in upper for w in ("UNSURE", "CANNOT", "DON'T KNOW", "NOT SURE",
                                  "UNCLEAR", "I'M NOT", "THE CLIP")):
        return None
    words = s.split()
    if len(words) > 8:
        return None   # too descriptive → uninformative
    return s


def _build_replace_prompt(span_text: str, corrected_value: str,
                           base_pred: str) -> str:
    """Step 2: minimal replacement — substitute span with corrected value."""
    return (
        f"Original answer: \"{base_pred}\"\n\n"
        f"The claim \"{span_text}\" has been identified as incorrect.\n"
        f"Corrected value: \"{corrected_value}\"\n\n"
        f"Task: Rewrite the answer replacing ONLY \"{span_text}\" with "
        f"\"{corrected_value}\". Keep all other words unchanged.\n"
        f"If you are not confident, return the original answer unchanged.\n"
        f"Output ONLY the rewritten answer, no explanation.\n\n"
        f"Rewritten answer:"
    )


def _build_recheck_prompt(base_pred: str, candidate: str,
                           question: str, step1_obs: str) -> str:
    """
    Evidence-grounded accept gate: does candidate align BETTER with evidence?

    No GT, no Token-F1. Pure evidence comparison.
    """
    obs_line = (f"Evidence from the clip: {step1_obs}\n\n"
                if step1_obs and step1_obs.strip()
                else "")
    return (
        f"Question: {question}\n\n"
        f"{obs_line}"
        f"Original answer: \"{base_pred}\"\n"
        f"Revised answer:  \"{candidate}\"\n\n"
        f"Does the REVISED answer align BETTER with the video/audio evidence "
        f"than the original?\n"
        f"Answer YES only if the revision corrects a specific factual error that "
        f"contradicts what you see or hear.\n"
        f"Answer NO if the revision introduces new errors, changes correct facts, "
        f"or if you cannot determine.\n\n"
        f"VERDICT: YES or NO\n"
        f"Reason (one sentence):"
    )


def _parse_recheck_verdict(raw: str) -> str:
    """Return 'YES' or 'NO' from a recheck response."""
    upper = raw.strip().upper()
    m = re.search(r'VERDICT\s*:\s*(YES|NO)', upper)
    if m:
        return m.group(1)
    # First meaningful word
    for token in upper.split():
        tok = token.rstrip(".,;:")
        if tok in ("YES", "NO"):
            return tok
    return "NO"  # conservative default


def _infer_fact_type(span_text: str) -> str:
    """Mirror of run_avh_factprobe._infer_fact_type — inlined to avoid path dep."""
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


def _build_fallback_probe_prompt(base_pred: str, question: str,
                                  step1_obs: str) -> str:
    """
    Fallback path when localization failed (no revision_candidates).
    More general than targeted probe — does NOT anchor to a specific span.
    """
    obs_line = (f"Evidence from the clip: {step1_obs}\n\n"
                if step1_obs and step1_obs.strip()
                else "")
    return (
        f"Question: {question}\n\n"
        f"Original answer: \"{base_pred}\"\n\n"
        f"{obs_line}"
        f"Does the original answer contain any factual error that contradicts "
        f"what you see or hear in the clip?\n"
        f"If YES: rewrite the answer fixing ONLY the error. "
        f"Keep all correct parts unchanged.\n"
        f"If NO or UNSURE: return the original answer unchanged.\n\n"
        f"Corrected answer:"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit",     type=int, default=50)
    ap.add_argument("--arm",       choices=["audio_only","combined","video_only"],
                    default="audio_only")
    ap.add_argument("--out",       default=None)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--probe-max-tokens", type=int, default=32)
    ap.add_argument("--recheck-max-tokens", type=int, default=48)
    args = ap.parse_args()

    OUT_DIR = OUT_BASE
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else \
        OUT_DIR / f"{args.arm}_n{args.limit}.jsonl"
    summary_path = out_path.with_suffix(".summary.md")

    print(f"[factprobe_v2] arm={args.arm}  limit={args.limit}  out={out_path}")

    # ------------------------------------------------------------------ data
    print("[factprobe_v2] Loading AVHBench …")
    with open(AVH_JSON) as f:
        avhbench_data = json.load(f)

    probe_records = load_avh_probe(PROBE_PATH, avhbench_data,
                                   signal_mode=args.arm)
    probe_records = sorted(probe_records, key=lambda r: int(r["id"]))
    probe_records = [r for r in probe_records if int(r["id"]) < args.limit]
    print(f"[factprobe_v2] {len(probe_records)} probe records (limit={args.limit})")

    # ------------------------------------------------------------------ localizer + verifier
    print("[factprobe_v2] Importing localizer/verifier …")
    import localizer_v2    as lv2_mod
    import evidence_verifier as ev_mod

    # ------------------------------------------------------------------ model
    print("[factprobe_v2] Loading model …")
    from methods.shared import model_utils as _mu
    from methods.shared.prompts import get_system_prompt
    from methods.shared.data_loader import load_dataset, load_media_for_item

    build_user_content_with_media_tokens = _mu.build_user_content_with_media_tokens
    build_inputs          = _mu.build_inputs
    make_messages         = _mu.make_messages
    prepare_for_thinker   = _mu.prepare_for_thinker
    decode_outputs        = _mu.decode_outputs
    sys_prompt            = get_system_prompt("avhbench", "v1")

    shm_path = Path("/dev/shm/Qwen2.5-Omni-7B")
    if shm_path.exists():
        print(f"[factprobe_v2] Using shm model at {shm_path}")
        processor, model = _mu.load_model(str(shm_path))
    else:
        processor, model = _mu.load_model()
    model.eval()

    # Load AVHBench items for media resolution
    AVH_JSON_STR  = str(AVH_JSON)
    AVH_DATA_ROOT = str(AVH_JSON.parent)
    avh_data_list = load_dataset(AVH_JSON_STR, AVH_DATA_ROOT)
    avh_items_by_uid = {item["uid"]: item for item in avh_data_list}

    # ------------------------------------------------------------------ resume
    done_ids = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    done_ids.add(str(json.loads(line)["id"]))
                except Exception:
                    pass
        print(f"[factprobe_v2] Resuming — {len(done_ids)} already done")

    out_fh = open(out_path, "a")
    rows   = []

    # ------------------------------------------------------------------ main loop
    for rec in probe_records:
        rid = str(rec["id"])
        if rid in done_ids:
            continue

        question  = rec.get("question", "")
        base_pred = "".join(rec["tokens"])
        gt        = rec.get("gt", "")

        # build cache record (delta_img_tok already added by load_avh_probe)
        cache_record = {
            "delta_img_tok":  rec["delta_img_tok"],
            "delta_video":    rec.get("delta_video", [0.0]*len(rec["tokens"])),
            "delta_audio":    rec.get("delta_audio", [0.0]*len(rec["tokens"])),
            "delta_text_tok": [0.0]*len(rec["tokens"]),
            "classes":        rec.get("classes", []),
            "lp_full":        rec.get("lp_full", []),
            "tokens":         rec["tokens"],
            "id":             rid,
            "question":       question,
        }

        # localize
        lv2_result = lv2_mod.localize(cache_record)

        # load media
        uid       = f"avhbench_{int(rid):06d}"
        data_item = avh_items_by_uid.get(uid)

        _gen_state = {}
        if data_item is not None:
            try:
                pv, av, iv = load_media_for_item(data_item, AVH_JSON_STR, AVH_DATA_ROOT)
            except Exception:
                pv, av, iv = None, None, None
            _gen_state = {"pv": pv, "av": av, "iv": iv, "sys": sys_prompt}
        else:
            _gen_state = {"pv": None, "av": None, "iv": None, "sys": sys_prompt}

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

        # verify candidates (fills verify_status, step1_observation, decision)
        ev_mod.verify_candidates(lv2_result, rec, _lv2_generate)

        rev = lv2_result.revision_candidates

        # ------------------------------------------------------------------
        # CASE 1: Localization failed — no revision candidates
        # ------------------------------------------------------------------
        if not rev:
            # Fallback: generic probe without span anchor
            n_all = len(lv2_result.candidates)
            # Try a brief fallback observation via the first candidate if available
            fallback_obs = ""
            if lv2_result.candidates:
                c0 = lv2_result.candidates[0]
                fallback_obs = getattr(c0, "step1_observation", "") or ""

            fallback_prompt = _build_fallback_probe_prompt(
                base_pred, question, fallback_obs)
            fallback_cand = _lv2_generate(fallback_prompt, args.max_new_tokens)
            fallback_cand = fallback_cand.split("\n")[0].strip()

            # Accept gate for fallback: simple recheck
            fb_recheck = "NO"
            fb_accepted = False
            if fallback_cand and fallback_cand.strip() != base_pred.strip():
                rc_prompt  = _build_recheck_prompt(base_pred, fallback_cand,
                                                   question, fallback_obs)
                rc_raw     = _lv2_generate(rc_prompt, args.recheck_max_tokens)
                fb_recheck = _parse_recheck_verdict(rc_raw)
                fb_accepted = (fb_recheck == "YES")

            repaired = fallback_cand if fb_accepted else base_pred
            row = dict(
                id=rid, question=question, gt=gt, base_pred=base_pred,
                status="localization_failed",
                localization_failed=True,
                n_all_candidates=n_all,
                # span fields — all None (no span identified)
                span_text=None, claim=None, verify_status=None,
                span_kind=None, fact_type=None,
                step1_obs=None,
                # probe
                probe_q=None, probe_answer=None,
                probe_is_unsure=None, probe_corrected_value=None,
                # replace
                candidate_text=fallback_cand,
                # recheck
                recheck_verdict=fb_recheck,
                accepted=fb_accepted,
                # output
                repaired_pred=repaired,
                # offline eval only
                base_f1=_token_f1(base_pred, gt),
                repaired_f1=_token_f1(repaired, gt),
                candidate_f1=_token_f1(fallback_cand, gt),
            )
            out_fh.write(json.dumps(row) + "\n")
            out_fh.flush()
            rows.append(row)
            print(f"  [{rid}] localization_failed  fallback_accepted={fb_accepted}")
            continue

        # ------------------------------------------------------------------
        # CASE 2: Has revision candidate — span-targeted probe
        # ------------------------------------------------------------------
        cand       = rev[0]
        span_text  = cand.span_text or ""
        claim      = cand.claim     or span_text
        span_kind  = cand.span_kind
        verify_status = cand.verify_status
        step1_obs  = getattr(cand, "step1_observation", "") or ""

        # Infer fact_type (kept for logging compatibility with v1)
        fact_type = _infer_fact_type(span_text)

        # -- Step 1a: Verdict probe (YES / NO / UNSURE) --
        verdict_prompt = _build_verdict_prompt(
            span_text, claim, base_pred, question, step1_obs)
        verdict_raw = _lv2_generate(verdict_prompt, args.probe_max_tokens)
        verdict = _parse_verdict(verdict_raw)

        if verdict != "NO":
            # Span is correct or model is unsure → no edit
            is_unsure = (verdict == "UNSURE")
            row = dict(
                id=rid, question=question, gt=gt, base_pred=base_pred,
                status="no_edit",
                localization_failed=False,
                n_all_candidates=len(lv2_result.candidates),
                span_text=span_text, claim=claim,
                verify_status=verify_status, span_kind=span_kind,
                fact_type=fact_type,
                step1_obs=step1_obs[:120] if step1_obs else None,
                probe_q=verdict_prompt[:200],
                probe_answer=verdict_raw.strip().split("\n")[0].strip(),
                probe_verdict=verdict,
                probe_is_unsure=is_unsure,
                probe_corrected_value=None,
                candidate_text=None,
                recheck_verdict=None,
                accepted=False,
                repaired_pred=base_pred,
                base_f1=_token_f1(base_pred, gt),
                repaired_f1=_token_f1(base_pred, gt),
                candidate_f1=None,
            )
            out_fh.write(json.dumps(row) + "\n")
            out_fh.flush()
            rows.append(row)
            status_tag = "probe_unsure" if is_unsure else "probe_correct"
            print(f"  [{rid}] {status_tag}  span='{span_text[:30]}'  verdict={verdict}")
            continue

        # -- Step 1b: Correction call (verdict was NO) --
        correction_prompt = _build_correction_prompt(
            span_text, claim, base_pred, question, step1_obs)
        correction_raw = _lv2_generate(correction_prompt, args.probe_max_tokens)
        corrected_value = _parse_correction(correction_raw)

        if corrected_value is None:
            # Correction was uninformative despite NO verdict → skip
            row = dict(
                id=rid, question=question, gt=gt, base_pred=base_pred,
                status="no_edit",
                localization_failed=False,
                n_all_candidates=len(lv2_result.candidates),
                span_text=span_text, claim=claim,
                verify_status=verify_status, span_kind=span_kind,
                fact_type=fact_type,
                step1_obs=step1_obs[:120] if step1_obs else None,
                probe_q=correction_prompt[:200],
                probe_answer=correction_raw.strip().split("\n")[0].strip(),
                probe_verdict=verdict,
                probe_is_unsure=True,
                probe_corrected_value=None,
                candidate_text=None,
                recheck_verdict=None,
                accepted=False,
                repaired_pred=base_pred,
                base_f1=_token_f1(base_pred, gt),
                repaired_f1=_token_f1(base_pred, gt),
                candidate_f1=None,
            )
            out_fh.write(json.dumps(row) + "\n")
            out_fh.flush()
            rows.append(row)
            print(f"  [{rid}] probe_correction_uninformative  span='{span_text[:30]}'  raw='{correction_raw[:40]}'")
            continue

        # -- Step 2: Targeted replacement --
        replace_prompt  = _build_replace_prompt(span_text, corrected_value, base_pred)
        candidate_text  = _lv2_generate(replace_prompt, args.max_new_tokens)
        candidate_text  = candidate_text.split("\n")[0].strip()

        # -- Step 3: Evidence-grounded accept gate (no GT/F1) --
        recheck_verdict = "NO"
        accepted        = False
        if candidate_text and candidate_text.strip() != base_pred.strip():
            rc_prompt       = _build_recheck_prompt(
                                  base_pred, candidate_text, question, step1_obs)
            rc_raw          = _lv2_generate(rc_prompt, args.recheck_max_tokens)
            recheck_verdict = _parse_recheck_verdict(rc_raw)
            accepted        = (recheck_verdict == "YES")

        repaired = candidate_text if accepted else base_pred

        # Offline eval fields (never used in accept gate)
        base_f1      = _token_f1(base_pred, gt)
        candidate_f1 = _token_f1(candidate_text, gt) if candidate_text else base_f1
        repaired_f1  = _token_f1(repaired, gt)

        row = dict(
            id=rid, question=question, gt=gt, base_pred=base_pred,
            status="processed",
            localization_failed=False,
            n_all_candidates=len(lv2_result.candidates),
            span_text=span_text,
            claim=claim,
            verify_status=verify_status,
            span_kind=span_kind,
            fact_type=fact_type,
            step1_obs=step1_obs[:120] if step1_obs else None,
            probe_q=verdict_prompt[:200],
            probe_answer=verdict_raw.strip().split("\n")[0].strip(),
            probe_verdict=verdict,
            probe_is_unsure=False,
            probe_corrected_value=corrected_value,
            candidate_text=candidate_text,
            recheck_verdict=recheck_verdict,
            accepted=accepted,
            repaired_pred=repaired,
            base_f1=base_f1,
            repaired_f1=repaired_f1,
            candidate_f1=candidate_f1,
        )
        out_fh.write(json.dumps(row) + "\n")
        out_fh.flush()
        rows.append(row)
        print(f"  [{rid}] span='{span_text[:28]}'  probe='{corrected_value[:30]}'  "
              f"recheck={recheck_verdict}  accepted={accepted}  "
              f"base_f1={base_f1:.3f}  cand_f1={candidate_f1:.3f}")

    out_fh.close()

    # ------------------------------------------------------------------ summary
    all_rows = []
    with open(out_path) as f:
        for line in f:
            try:
                all_rows.append(json.loads(line))
            except Exception:
                pass

    n_total    = len(all_rows)
    n_loc_fail = sum(1 for r in all_rows if r.get("localization_failed"))
    n_no_edit  = sum(1 for r in all_rows if r.get("status") == "no_edit")
    n_proc     = sum(1 for r in all_rows if r.get("status") == "processed")
    n_accepted = sum(1 for r in all_rows if r.get("accepted"))

    base_f1_mean     = sum(r.get("base_f1", 0)     for r in all_rows) / max(n_total, 1) * 100
    repaired_f1_mean = sum(r.get("repaired_f1", r.get("base_f1", 0)) for r in all_rows) / max(n_total, 1) * 100
    oracle_f1_mean   = sum(max(r.get("candidate_f1") or r.get("base_f1", 0),
                               r.get("base_f1", 0))
                           for r in all_rows) / max(n_total, 1) * 100

    n_improved = sum(1 for r in all_rows
                     if r.get("candidate_f1") is not None
                     and r["candidate_f1"] > r.get("base_f1", 0))
    n_degraded = sum(1 for r in all_rows
                     if r.get("accepted") and r.get("candidate_f1") is not None
                     and r["candidate_f1"] < r.get("base_f1", 0))

    # Probe quality analysis
    n_echo = 0
    for r in all_rows:
        probe = r.get("probe_answer", "") or ""
        obs   = r.get("step1_obs", "") or ""
        if probe and obs:
            pw = set(probe.lower().split())
            ow = set(obs.lower().split())
            if pw and ow and len(pw & ow) / len(pw) > 0.5:
                n_echo += 1

    delta_f1 = repaired_f1_mean - base_f1_mean

    summary = f"""# FactProbe v2 Summary

## Configuration
- arm: {args.arm}
- limit: {args.limit}
- targeted probe prompt: YES (v2)
- evidence-grounded accept gate: YES (recheck vs base)

## Key Results
| Metric | Value |
|---|---|
| n_total | {n_total} |
| n_localization_failed | {n_loc_fail} |
| n_no_edit (probe_correct/unsure) | {n_no_edit} |
| n_processed (span-targeted) | {n_proc} |
| n_accepted | {n_accepted} |
| base_f1_mean | {base_f1_mean:.2f} |
| repaired_f1_mean | {repaired_f1_mean:.2f} |
| delta_f1 | {delta_f1:+.2f} |
| oracle_f1_mean | {oracle_f1_mean:.2f} |
| n_candidate_improved (oracle) | {n_improved} |
| n_accepted_degraded | {n_degraded} |

## Probe Echo Analysis
| Metric | Value |
|---|---|
| n_echo (probe >50% overlap with step1_obs) | {n_echo} |
| echo_rate | {n_echo/max(n_total,1)*100:.1f}% |

## Comparison with v1 FactProbe
| Version | base_f1 | repaired_f1 | delta | oracle |
|---|---|---|---|---|
| v1 (open-ended probe) | 27.05 | 25.58 | -1.47 | 28.54 |
| v2 (targeted probe) | {base_f1_mean:.2f} | {repaired_f1_mean:.2f} | {delta_f1:+.2f} | {oracle_f1_mean:.2f} |

## Hypothesis
Did targeted span-probe reduce echo rate and improve accept precision?
- v1 echo rate: 84%
- v2 echo rate: {n_echo/max(n_total,1)*100:.0f}%
- v1 bad_accepted: 18/37
- v2 bad_accepted: {n_degraded}/{n_accepted}
"""

    print(summary)
    with open(summary_path, "w") as f:
        f.write(summary)
    print(f"[factprobe_v2] Summary written to {summary_path}")


if __name__ == "__main__":
    main()
