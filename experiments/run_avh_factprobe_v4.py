#!/usr/bin/env python3
"""
FactProbe v4 — Raw-Evidence Recheck (symmetric, self-reference-free).

Key changes vs v3:
  - step1_obs NEVER passed to verifier (cuts self-reference chain).
  - Symmetric dual-probe: ask separately whether the CLIP supports the BASE
    claim and whether it supports the CANDIDATE claim.
  - Accept only when candidate=SUPPORTED AND base=CONTRADICTED.
  - Evidence-dependence check: run the same prompts text-only (no media).
    If verdict is identical without media, flag as non-evidence-based.
  - Frozen v1 candidate pool: no new generation.

Stop condition (per user spec):
  If this gate still cannot discriminate positive from negative v1 candidates,
  stop tuning the gate and pivot to Candidate Generation research.

Usage:
  CUDA_VISIBLE_DEVICES=0 MMHALU_MODEL_ROOT=/tmp HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 PYTHONPATH=/dev/shm/site-packages \
  LD_LIBRARY_PATH=/dev/shm/so_mirror \
  /home/tsq/.local/bin/python3.13 scripts/run_avh_factprobe_v4.py \
    --arm audio_only --limit 50
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
OUTER_ROOT = Path(__file__).resolve().parent.parent
INNER_ROOT = OUTER_ROOT / "OmniConfess"

assert (INNER_ROOT / "methods/shared/model_utils.py").exists(), \
    f"model_utils not found under {INNER_ROOT}"

sys.path.insert(0, str(INNER_ROOT))
sys.path.insert(0, str(OUTER_ROOT))

AVH_JSON   = OUTER_ROOT / "OmniHalluBench/avhbench/avhbench.json"
V1_OUT     = OUTER_ROOT / "outputs/lv2_avh_factprobe/audio_only_n50.jsonl"
OUT_DIR    = OUTER_ROOT / "outputs/lv2_avh_factprobe_v4"
OUT_FILE   = OUT_DIR / "audio_only_n50.jsonl"
SUMMARY_F  = OUT_DIR / "SUMMARY_factprobe_v4.md"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


STOP = {"a","an","the","is","are","was","were","it","its","in","on","at",
        "to","of","and","or","not","no","do","does","did","be","been",
        "being","have","has","had","this","that","i","you","he","she",
        "we","they","my","your","his","her","our","their"}


def _content_word_overlap(a: str, b: str) -> float:
    wa = {w.strip('.,!?;:"\'') for w in a.lower().split()
          if w.strip('.,!?;:"\'') not in STOP}
    wb = {w.strip('.,!?;:"\'') for w in b.lower().split()
          if w.strip('.,!?;:"\'') not in STOP}
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / min(len(wa), len(wb))


# ---------------------------------------------------------------------------
# Deterministic pre-filter (same as v3 — catches obvious junk before GPU)
# ---------------------------------------------------------------------------

_ENTITY_NOUNS = {
    "bird","cat","dog","person","man","woman","people","animal",
    "helicopter","car","truck","vehicle","boat","train","plane","aircraft",
    "toilet","pirate","cage","fire","smoke","tree","guitar","conductor",
    "orchestra","microphone","sign","crowd","clock","baby","seagull","pigeon",
}
_ACTIVITY_NEG_RE = re.compile(
    r"no visible (acceleration|motion|movement|flying|running|playing)"
    r"|no\s+\w+(\s+or\s+\w+(\s+\w+)?)?\s+(is|are)\s+(evident|audible|visible|present)"
    r"|(stationary|no visible motion|not moving|engine running but)"
    r"|no (indication|sign) of (playing|distortion|movement|motion|acceleration|grazing)",
    re.I,
)


def _is_entity_negation(candidate: str) -> bool:
    cl = candidate.lower().strip()
    for noun in _ENTITY_NOUNS:
        if re.search(
            rf"\bno\b[^.]*\b{re.escape(noun)}\b[^.]*\b(visible|present|specified|found|detected|shown)",
            cl,
        ):
            return True
    m = re.match(r"^no\s+(\w+)", cl)
    if m and m.group(1) in _ENTITY_NOUNS:
        return True
    for noun in _ENTITY_NOUNS:
        if re.search(rf"there is no\b[^.]*\b{re.escape(noun)}\b", cl):
            return True
    return False


def deterministic_gate(candidate: str, base_pred: str) -> tuple[bool, list[str]]:
    reasons = []
    # G1: LaTeX format injection
    if any(x in candidate for x in [r'\boxed', r'\frac', r'\text', r'\begin', r'\end']):
        reasons.append("format_injection")
    # G2: entity negation (not activity negation)
    if _is_entity_negation(candidate) and not _ACTIVITY_NEG_RE.search(candidate):
        reasons.append("entity_negation")
    # G3: length destruction
    base_len = len(base_pred.split())
    cand_len = len(candidate.split())
    if base_len > 0 and cand_len < 0.65 * base_len:
        reasons.append("too_short")
    # G4: scene substitution
    if _content_word_overlap(candidate, base_pred) < 0.50:
        reasons.append("low_overlap")
    # G5: trivial no-op
    c_clean = re.sub(r"^(a |an |the )", "", candidate.lower()).rstrip(".")
    b_clean = re.sub(r"^(a |an |the )", "", base_pred.lower()).rstrip(".")
    if c_clean == b_clean:
        reasons.append("no_op")
    return len(reasons) == 0, reasons


# ---------------------------------------------------------------------------
# Fact difference extraction (CPU)
# ---------------------------------------------------------------------------

_FACT_DIMS = [
    ("subject",  r"^([A-Z][^,\.]+?)(?:,|\s+(?:is|are|was|were|flies|runs|sits|stands|speaks|moves|drives|walks|plays|hovers|perches))"),
    ("location", r"\b(?:in|on|at|near|beside|inside|outside|through|behind|in front of)\s+([^,\.]+)"),
    ("action",   r"\b(accelerates?|flies?|runs?|sits?|stands?|speaks?|moves?|drives?|walks?|plays?|hovers?|perches?|grazes?|barks?|lands?|leads?|conducting|signing|feeding|flushes?|refills?)\b"),
    ("sound",    r"\b(barks?|whistles?|thumping|buzzing|whirring?|squealing?|ringing?|snoring?|meow(?:ing)?|baa(?:ing)?|chirp(?:ing)?)\b"),
    ("object",   r"\b(helicopter|bird|cat|dog|car|truck|boat|cage|guitar|microphone|blanket|thermometer|clock|barrel|ladder)\b"),
]


def extract_diff(base_pred: str, candidate: str) -> list[dict]:
    diffs = []
    for dim, pat in _FACT_DIMS:
        m_base = re.search(pat, base_pred, re.I)
        m_cand = re.search(pat, candidate, re.I)
        bv = m_base.group(1).strip().lower() if m_base else None
        cv = m_cand.group(1).strip().lower() if m_cand else None
        if bv and cv and bv != cv:
            diffs.append({"dim": dim, "base_val": bv, "cand_val": cv})
    if not diffs:
        base_words = set(base_pred.lower().split()) - STOP
        cand_words = set(candidate.lower().split()) - STOP
        removed = sorted(base_words - cand_words)[:4]
        added   = sorted(cand_words - base_words)[:4]
        if removed or added:
            diffs.append({
                "dim":      "general",
                "base_val": " ".join(removed),
                "cand_val": " ".join(added),
            })
    return diffs


# ---------------------------------------------------------------------------
# Symmetric raw-evidence probes (NO step1_obs)
# ---------------------------------------------------------------------------

def build_claim_probe(question: str, claim_text: str, arm: str) -> str:
    """
    Ask the model whether a SPECIFIC CLAIM is supported by the raw clip.
    No step1_obs, no prior model description — only the question and the claim.
    """
    if arm == "audio_only":
        channel_note = "Focus only on the AUDIO channel (sounds, voices, audio cues)."
    elif arm == "video_only":
        channel_note = "Focus only on the VISUAL channel (what appears on screen)."
    else:
        channel_note = "Use both audio and visual evidence."

    return (
        f"Question about this clip: {question}\n\n"
        f"Claim to verify: \"{claim_text}\"\n\n"
        f"{channel_note}\n\n"
        f"Based ONLY on what you directly perceive in the clip right now, "
        f"does the evidence SUPPORT or CONTRADICT this claim?\n"
        f"If the clip does not contain enough information to judge, say INSUFFICIENT.\n\n"
        f"Reply with exactly one word: SUPPORTED, CONTRADICTED, or INSUFFICIENT.\n\n"
        f"Verdict:"
    )


def build_text_only_probe(question: str, claim_text: str) -> str:
    """Same probe but without any media — used as ablation control."""
    return (
        f"Question about a clip: {question}\n\n"
        f"Claim: \"{claim_text}\"\n\n"
        f"Without any clip to watch, based only on the question wording and common sense, "
        f"can you judge this claim?\n"
        f"Reply with exactly one word: SUPPORTED, CONTRADICTED, or INSUFFICIENT.\n\n"
        f"Verdict:"
    )


def parse_verdict(raw: str) -> str:
    r = raw.strip().upper()
    for tok in r.split():
        if tok in ("SUPPORTED", "CONTRADICTED", "INSUFFICIENT"):
            return tok
    if "SUPPORT" in r:
        return "SUPPORTED"
    if "CONTRADICT" in r:
        return "CONTRADICTED"
    return "INSUFFICIENT"


# ---------------------------------------------------------------------------
# Accept gate
# ---------------------------------------------------------------------------

def raw_evidence_gate(
    base_verdict: str,
    cand_verdict: str,
    base_verdict_textonly: str,
    cand_verdict_textonly: str,
    det_passed: bool,
    det_reasons: list[str],
) -> tuple[bool, str]:
    """
    Accept iff:
      1. Deterministic gate passes
      2. candidate claim is SUPPORTED by evidence
      3. base claim is NOT SUPPORTED (CONTRADICTED or INSUFFICIENT)
      4. At least one of the verdicts differs from the text-only ablation
         (i.e., media actually changed the answer)
    """
    if not det_passed:
        return False, f"det_reject:{','.join(det_reasons)}"

    # Core logic: candidate must be supported, base must not be
    if cand_verdict != "SUPPORTED":
        return False, f"cand_not_supported:{cand_verdict}"
    if base_verdict == "SUPPORTED":
        return False, "base_also_supported"

    # Evidence-dependence check: both verdicts match text-only → not using clip
    if (base_verdict == base_verdict_textonly and
            cand_verdict == cand_verdict_textonly):
        return False, "evidence_independent"

    return True, "evidence_grounded"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm",   default="audio_only",
                        choices=["audio_only", "video_only", "combined"])
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--v1-pool", default=str(V1_OUT))
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ load frozen pool
    print("[v4] Loading frozen v1 candidate pool …")
    v1_rows = []
    with open(args.v1_pool) as f:
        for line in f:
            try:
                v1_rows.append(json.loads(line))
            except Exception:
                pass
    v1_rows = [r for r in v1_rows if int(r["id"]) < args.limit]
    print(f"[v4] {len(v1_rows)} rows (limit={args.limit})")

    # ------------------------------------------------------------------ load AVHBench + model
    print("[v4] Loading AVHBench + model …")
    with open(AVH_JSON) as f:
        avhbench_data = json.load(f)

    from methods.shared import model_utils as _mu
    from methods.shared.prompts import get_system_prompt
    from methods.shared.data_loader import load_dataset, load_media_for_item

    build_inputs                         = _mu.build_inputs
    make_messages                        = _mu.make_messages
    build_user_content_with_media_tokens = _mu.build_user_content_with_media_tokens
    prepare_for_thinker                  = _mu.prepare_for_thinker
    decode_outputs                       = _mu.decode_outputs

    _ = torch.zeros(1, device="cuda")
    shm_path = Path("/dev/shm/Qwen2.5-Omni-7B")
    if shm_path.exists():
        print(f"[v4] Using shm model at {shm_path}")
        processor, model = _mu.load_model(str(shm_path))
    else:
        processor, model = _mu.load_model()

    sys_prompt = get_system_prompt("avhbench", "v1")

    AVH_DATA      = str(AVH_JSON)
    AVH_DATA_ROOT = str(OUTER_ROOT / "OmniHalluBench/avhbench")
    avh_items     = load_dataset(AVH_DATA, AVH_DATA_ROOT)
    avh_by_uid    = {}
    for it in avh_items:
        uid = it.get("uid", "")
        try:
            num = str(int(uid.split("_")[-1]))
            avh_by_uid[num] = it
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
        print(f"[v4] Resuming — {len(done_ids)} already done")

    out_fh = open(OUT_FILE, "a")

    # ------------------------------------------------------------------ per-sample loop
    for v1_row in v1_rows:
        rid       = v1_row["id"]
        if rid in done_ids:
            continue

        base_pred    = v1_row["base_pred"]
        candidate    = v1_row.get("candidate_text") or ""
        question     = v1_row.get("question", "")
        gt           = v1_row.get("gt", "")
        base_f1      = v1_row.get("base_f1") or _token_f1(base_pred, gt)
        candidate_f1 = v1_row.get("candidate_f1")

        # ---- no candidate: carry forward ----
        if not candidate or v1_row.get("status") == "no_candidates":
            row = dict(
                id=rid, question=question, gt=gt,
                base_pred=base_pred, status="no_candidate",
                candidate_text=None, candidate_f1=None,
                det_passed=None, det_reasons=[],
                changed_facts=[],
                base_claim=None, cand_claim=None,
                base_verdict=None, cand_verdict=None,
                base_verdict_textonly=None, cand_verdict_textonly=None,
                accepted=False, accept_reason="no_candidate",
                repaired_pred=base_pred,
                base_f1=base_f1, repaired_f1=base_f1,
            )
            out_fh.write(json.dumps(row) + "\n")
            out_fh.flush()
            continue

        # ---- Step 1: deterministic gate ----
        det_passed, det_reasons = deterministic_gate(candidate, base_pred)

        # ---- Step 2: extract diffs ----
        diffs = extract_diff(base_pred, candidate)

        base_verdict           = None
        cand_verdict           = None
        base_verdict_textonly  = None
        cand_verdict_textonly  = None
        base_claim             = None
        cand_claim             = None

        if det_passed and candidate.strip() != base_pred.strip():
            # Choose the most salient changed fact for probing
            # Prefer object/subject/action dim over general
            key_diff = None
            for d in diffs:
                if d["dim"] in ("object", "subject", "action"):
                    key_diff = d
                    break
            if key_diff is None and diffs:
                key_diff = diffs[0]

            if key_diff:
                # Build claim strings from the diff
                base_claim = f"The {key_diff['dim']} is \"{key_diff['base_val']}\""
                cand_claim = f"The {key_diff['dim']} is \"{key_diff['cand_val']}\""
            else:
                # Fallback: use full sentences
                base_claim = base_pred
                cand_claim = candidate

            # Load media for this sample
            uid       = f"avhbench_{int(rid):06d}"
            data_item = next((it for it in avhbench_data if it["uid"] == uid), None)
            pv, av, iv = None, None, None
            if data_item is not None:
                try:
                    pv, av, iv = load_media_for_item(data_item, AVH_DATA, AVH_DATA_ROOT)
                except Exception:
                    pass

            def _gen_with_media(prompt_text, _pv=pv, _av=av, _iv=iv,
                                 max_new_tokens=args.max_new_tokens):
                uc   = build_user_content_with_media_tokens(_pv, _av, _iv, prompt_text)
                msgs = make_messages(sys_prompt, uc)
                inp  = build_inputs(processor, model, msgs, _pv, _av, _iv, "haloquest")
                inp  = prepare_for_thinker(inp, model)
                with torch.inference_mode():
                    oids = model.thinker.generate(
                        **inp, max_new_tokens=max_new_tokens, do_sample=False)
                return decode_outputs(processor, oids,
                                      inp["input_ids"].shape[1])[0].strip()

            def _gen_text_only(prompt_text,
                               max_new_tokens=args.max_new_tokens):
                # No media — pass None for all media args
                uc   = build_user_content_with_media_tokens(None, None, None, prompt_text)
                msgs = make_messages(sys_prompt, uc)
                inp  = build_inputs(processor, model, msgs,
                                    None, None, None, "haloquest")
                inp  = prepare_for_thinker(inp, model)
                with torch.inference_mode():
                    oids = model.thinker.generate(
                        **inp, max_new_tokens=max_new_tokens, do_sample=False)
                return decode_outputs(processor, oids,
                                      inp["input_ids"].shape[1])[0].strip()

            # ---- Step 3: symmetric raw-evidence probes ----
            base_probe = build_claim_probe(question, base_claim, args.arm)
            cand_probe = build_claim_probe(question, cand_claim, args.arm)

            base_raw  = _gen_with_media(base_probe)
            cand_raw  = _gen_with_media(cand_probe)
            base_verdict = parse_verdict(base_raw)
            cand_verdict = parse_verdict(cand_raw)

            # ---- Step 4: text-only ablation ----
            base_to_raw = _gen_text_only(build_text_only_probe(question, base_claim))
            cand_to_raw = _gen_text_only(build_text_only_probe(question, cand_claim))
            base_verdict_textonly = parse_verdict(base_to_raw)
            cand_verdict_textonly = parse_verdict(cand_to_raw)

        # ---- Step 5: accept decision ----
        accepted, accept_reason = raw_evidence_gate(
            base_verdict or "INSUFFICIENT",
            cand_verdict or "INSUFFICIENT",
            base_verdict_textonly or "INSUFFICIENT",
            cand_verdict_textonly or "INSUFFICIENT",
            det_passed,
            det_reasons,
        )

        repaired_pred = candidate if accepted else base_pred
        repaired_f1   = _token_f1(repaired_pred, gt)

        row = dict(
            id=rid, question=question, gt=gt,
            base_pred=base_pred, status="processed",
            candidate_text=candidate, candidate_f1=candidate_f1,
            det_passed=det_passed, det_reasons=det_reasons,
            changed_facts=diffs[:3],
            base_claim=base_claim, cand_claim=cand_claim,
            base_verdict=base_verdict, cand_verdict=cand_verdict,
            base_verdict_textonly=base_verdict_textonly,
            cand_verdict_textonly=cand_verdict_textonly,
            accepted=accepted, accept_reason=accept_reason,
            repaired_pred=repaired_pred,
            base_f1=base_f1, repaired_f1=repaired_f1,
        )
        out_fh.write(json.dumps(row) + "\n")
        out_fh.flush()
        print(
            f"  [{rid:3s}] det={'P' if det_passed else 'F'}  "
            f"base={base_verdict or '-':14s}  cand={cand_verdict or '-':14s}  "
            f"to_base={base_verdict_textonly or '-':14s}  to_cand={cand_verdict_textonly or '-':14s}  "
            f"acc={accepted}  reason={accept_reason}"
        )

    out_fh.close()

    # ------------------------------------------------------------------ summary
    all_rows = []
    with open(OUT_FILE) as f:
        for line in f:
            try:
                all_rows.append(json.loads(line))
            except Exception:
                pass

    def safe_mean(vals):
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else 0.0

    base_f1_mean     = safe_mean([r["base_f1"]     for r in all_rows])
    repaired_f1_mean = safe_mean([r["repaired_f1"] for r in all_rows])
    delta_f1         = repaired_f1_mean - base_f1_mean
    oracle_f1        = safe_mean([
        max(r["base_f1"], r["candidate_f1"] if r.get("candidate_f1") else 0.0)
        for r in all_rows])

    n_accepted    = sum(1 for r in all_rows if r.get("accepted"))
    n_det_reject  = sum(1 for r in all_rows if r.get("det_passed") is False)
    n_ev_ind      = sum(1 for r in all_rows
                        if r.get("accept_reason") == "evidence_independent")
    n_cand_sup    = sum(1 for r in all_rows
                        if r.get("cand_verdict") == "SUPPORTED")
    n_base_contra = sum(1 for r in all_rows
                        if r.get("base_verdict") == "CONTRADICTED")

    # Offline analysis vs v1 known labels
    v1_good_ids = {"0","7","11","14","15","23","24","25","28","30","32","45","46","49"}
    v1_bad_ids  = {"1","3","6","8","9","12","13","16","20","22","26","27","31",
                   "34","35","37","39","40","42","43","44","47","48"}
    n_pos_acc   = sum(1 for r in all_rows if r.get("accepted") and r["id"] in v1_good_ids)
    n_neg_acc   = sum(1 for r in all_rows if r.get("accepted") and r["id"] in v1_bad_ids)
    n_false_rej = sum(1 for r in all_rows if not r.get("accepted") and r["id"] in v1_good_ids)

    # Verdict distribution for v1-good and v1-bad
    def verdict_dist(rows_subset):
        bc = {}
        cc = {}
        for r in rows_subset:
            bv = r.get("base_verdict") or "NO_PROBE"
            cv = r.get("cand_verdict") or "NO_PROBE"
            bc[bv] = bc.get(bv, 0) + 1
            cc[cv] = cc.get(cv, 0) + 1
        return bc, cc

    good_rows = [r for r in all_rows if r["id"] in v1_good_ids]
    bad_rows  = [r for r in all_rows if r["id"] in v1_bad_ids]
    good_bv, good_cv = verdict_dist(good_rows)
    bad_bv,  bad_cv  = verdict_dist(bad_rows)

    lines = [
        "# FactProbe v4 Summary — Raw-Evidence Recheck (symmetric, self-reference-free)",
        "",
        "## Key results (same n=50 frozen pool as v1)",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| base_f1_mean         | {base_f1_mean:.4f} ({base_f1_mean*100:.2f}%) |",
        f"| repaired_f1_mean     | {repaired_f1_mean:.4f} ({repaired_f1_mean*100:.2f}%) |",
        f"| delta_f1             | {delta_f1:+.4f} ({delta_f1*100:+.2f}%) |",
        f"| candidate_oracle_f1  | {oracle_f1:.4f} ({oracle_f1*100:.2f}%) |",
        f"| n_accepted           | {n_accepted}/50 |",
        f"| n_det_rejected       | {n_det_reject} |",
        f"| n_evidence_indep     | {n_ev_ind} (flagged, not accepted) |",
        f"| n_cand_SUPPORTED     | {n_cand_sup} |",
        f"| n_base_CONTRADICTED  | {n_base_contra} |",
        "",
        "## Gate analysis vs v1 known outcomes (GT used for labelling only)",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| n_positive_accept (v1 good accepted) | {n_pos_acc}/14 |",
        f"| n_negative_accept (v1 bad accepted)  | {n_neg_acc}/23 |",
        f"| n_false_reject    (v1 good rejected) | {n_false_rej}/14 |",
        "",
        "## Verdict distributions",
        "",
        "### v1-good candidates (14 rows)",
        f"base_verdict: {good_bv}",
        f"cand_verdict: {good_cv}",
        "",
        "### v1-bad candidates (23 rows)",
        f"base_verdict: {bad_bv}",
        f"cand_verdict: {bad_cv}",
        "",
        "## Comparison table",
        "",
        "| Version | base_f1 | repaired_f1 | delta | oracle | n_accepted |",
        "|---|---|---|---|---|---|",
        "| v1 (open-ended probe)      | 27.05% | 25.58% | -1.47% | 28.54% | 37/50 |",
        "| v3 (confession-gate)       | 27.05% | 26.94% | -0.11% | 28.54% |  2/50 |",
        f"| v4 (raw-evidence recheck)  | {base_f1_mean*100:.2f}% | "
        f"{repaired_f1_mean*100:.2f}% | {delta_f1*100:+.2f}% | "
        f"{oracle_f1*100:.2f}% | {n_accepted}/50 |",
        "",
        "## Stop-condition check",
        "",
        "If n_positive_accept < 5 AND n_negative_accept > n_positive_accept:",
        "→ gate cannot discriminate; pivot to Candidate Generation research.",
        f"  n_positive_accept={n_pos_acc}  n_negative_accept={n_neg_acc}",
        f"  → {'STOP: pivot to candidate generation' if n_pos_acc < 5 and n_neg_acc >= n_pos_acc else 'CONTINUE: gate showing signal'}",
    ]

    SUMMARY_F.write_text("\n".join(lines))
    print(f"\n[v4] Done. base={base_f1_mean:.4f} repaired={repaired_f1_mean:.4f} "
          f"oracle={oracle_f1:.4f} delta={delta_f1:+.4f} accepted={n_accepted}/50")
    print(f"[v4] pos_accept={n_pos_acc}  neg_accept={n_neg_acc}  "
          f"false_rej={n_false_rej}  ev_indep={n_ev_ind}")
    print(f"[v4] Summary → {SUMMARY_F}")

    msg = (f"FactProbe v4完成: delta={delta_f1*100:+.2f}% "
           f"accepted={n_accepted}/50 "
           f"pos={n_pos_acc} neg={n_neg_acc} false_rej={n_false_rej} "
           f"ev_indep={n_ev_ind}")
    os.system(f'/home/tsq/.claude/feishu/feishu_send.sh "{msg}"')


if __name__ == "__main__":
    main()
