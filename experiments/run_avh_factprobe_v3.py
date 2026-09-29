#!/usr/bin/env python3
"""
FactProbe v3 — Description-Level Correction with Confession-Guided Verification.

Architecture (frozen candidate pool):
  Step 0  — Load frozen v1 candidate pool (no new generation).
  Step 1  — Deterministic gate: entity-negation, format-injection,
             length-destruction, low-overlap.  Activity-negation exceptions
             (no visible acceleration / no distortion evident) are preserved.
  Step 2  — Candidate Difference Extraction (CPU): extract changed facts
             (subject, action, object, location, sound-source).
  Step 3  — Evidence-Grounded Recheck (GPU): for each changed fact ask the
             model to compare base_pred vs candidate against raw clip evidence.
             Audio-only tasks → check audio channel; hybrid → check both.
  Step 4  — Confession-Guided Accept Gate:
               ACCEPT  if recheck supports candidate AND deterministic gate passes
               REJECT  if recheck finds no evidence support or base is better
               HOLD    if evidence is insufficient → fall back to base

Output: outputs/lv2_avh_factprobe_v3/audio_only_n50.jsonl

Report metrics (no GT/F1 in online routing):
  n_positive_accept, n_negative_accept, n_false_reject_vs_v1,
  candidate_oracle, delta_f1

Usage:
  CUDA_VISIBLE_DEVICES=0 MMHALU_MODEL_ROOT=/tmp HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 PYTHONPATH=/dev/shm/site-packages \
  LD_LIBRARY_PATH=/dev/shm/so_mirror \
  /home/tsq/.local/bin/python3.13 scripts/run_avh_factprobe_v3.py \
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

AVH_JSON   = OUTER_ROOT / "OmniHalluBench/avhbench/annotations/avhbench.json"
PROBE_PATH = OUTER_ROOT / "outputs/cf_probe/avh_probe_bigN_shard0of1.jsonl"
V1_OUT     = OUTER_ROOT / "outputs/lv2_avh_factprobe/audio_only_n50.jsonl"
OUT_DIR    = OUTER_ROOT / "outputs/lv2_avh_factprobe_v3"
OUT_FILE   = OUT_DIR / "audio_only_n50.jsonl"
SUMMARY_F  = OUT_DIR / "SUMMARY_factprobe_v3.md"

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
# Step 1 — Deterministic gate
# ---------------------------------------------------------------------------

# Concrete nouns whose negation ("no X visible") = entity negation → REJECT
_ENTITY_NOUNS = {
    "bird","cat","dog","person","man","woman","people","animal",
    "helicopter","car","truck","vehicle","boat","train","plane","aircraft",
    "toilet","toilet visible","pirate","cage","fire","smoke","water","tree",
    "guitar","conductor","orchestra","microphone","sign","text","crowd",
    "clock","baby","fire truck","seagull","pigeon",
}

# Patterns for activity/state negation → ALLOW (these are valid corrections)
_ACTIVITY_NEG_PATTERNS = [
    r"no visible (acceleration|motion|movement|flying|running|playing|playing sound)",
    r"no (distortion|playing sound|sound) (is )?(evident|audible|visible|present)",
    r"(stationary|no visible motion|not moving|engine running but)",
    r"no (indication|sign) of (playing|distortion|movement|motion|acceleration|grazing)",
]
_ACTIVITY_NEG_RE = re.compile("|".join(_ACTIVITY_NEG_PATTERNS), re.I)


def _is_entity_negation(candidate: str) -> bool:
    """Returns True if candidate makes an entity-existence negation claim."""
    cand_lower = candidate.lower().strip()
    # Pattern: "No [noun] visible/present/specified" at start or as main clause
    for noun in _ENTITY_NOUNS:
        pat = rf"\bno\b[^.]*\b{re.escape(noun)}\b[^.]*\b(visible|present|specified|found|detected|shown)"
        if re.search(pat, cand_lower):
            return True
    # Starts with "No [noun]" as main subject (not activity)
    m = re.match(r"^no\s+(\w+)", cand_lower)
    if m:
        first_word = m.group(1)
        if first_word in _ENTITY_NOUNS:
            return True
    # "There is no [noun]" / "no [noun] is visible"
    for noun in _ENTITY_NOUNS:
        if re.search(rf"there is no\b[^.]*\b{re.escape(noun)}\b", cand_lower):
            return True
    return False


def _is_activity_negation(candidate: str) -> bool:
    """Returns True if candidate negates an activity/state (not an entity)."""
    return bool(_ACTIVITY_NEG_RE.search(candidate))


def deterministic_gate(candidate: str, base_pred: str) -> tuple[bool, list[str]]:
    """
    Returns (passes, reasons_for_rejection).
    REJECT if any hard pattern fires (and it's not an activity-negation exception).
    """
    reasons = []
    cand_lower = candidate.lower().strip()

    # G1: LaTeX format injection
    if any(x in candidate for x in [r'\boxed', r'\frac', r'\text', '\\\\boxed',
                                      r'\begin', r'\end']):
        reasons.append("format_injection")

    # G2: Entity negation (unless it's actually activity negation)
    if _is_entity_negation(candidate) and not _is_activity_negation(candidate):
        reasons.append("entity_negation")

    # G3: Length destruction (candidate < 65% of base word count)
    base_len = len(base_pred.split())
    cand_len = len(candidate.split())
    if base_len > 0 and cand_len < 0.65 * base_len:
        reasons.append("too_short")

    # G4: Content overlap too low — scene substitution
    ov = _content_word_overlap(candidate, base_pred)
    if ov < 0.50:
        reasons.append("low_overlap")

    # G5: Trivial no-op (stripping article / minor punctuation only)
    c_clean = re.sub(r"^(a |an |the )", "", cand_lower).rstrip(".")
    b_clean = re.sub(r"^(a |an |the )", "", base_pred.lower()).rstrip(".")
    if c_clean == b_clean:
        reasons.append("no_op")

    return len(reasons) == 0, reasons


# ---------------------------------------------------------------------------
# Step 2 — Candidate Difference Extraction (CPU, no model)
# ---------------------------------------------------------------------------

_FACT_DIMS = [
    ("subject",  r"^([A-Z][^,\.]+?)(?:,|\s+(?:is|are|was|were|flies|runs|sits|stands|speaks|moves|drives|walks|plays))"),
    ("location", r"\b(?:in|on|at|near|beside|inside|outside|through|behind|in front of)\s+([^,\.]+)"),
    ("action",   r"\b((?:accelerates?|flies?|runs?|sits?|stands?|speaks?|moves?|drives?|walks?|plays?|hovers?|perches?|graze|barks?|lands?|leads?|conduct(?:ing|s)?|signing|feeding|flushes?|refills?))\b"),
    ("sound",    r"\b((?:barks?|whistles?|thumping|buzzing|whirring?|squealing?|ringing?|snoring?|meow(?:ing)?|baa(?:ing)?|chirp(?:ing)?))\b"),
]


def extract_diff(base_pred: str, candidate: str) -> list[dict]:
    """
    Return list of {dim, base_val, cand_val} for facts that changed.
    Simple regex-based; not exhaustive but sufficient for recheck prompts.
    """
    diffs = []
    for dim, pat in _FACT_DIMS:
        m_base = re.search(pat, base_pred, re.I)
        m_cand = re.search(pat, candidate, re.I)
        bv = m_base.group(1).strip().lower() if m_base else None
        cv = m_cand.group(1).strip().lower() if m_cand else None
        if bv and cv and bv != cv:
            diffs.append({"dim": dim, "base_val": bv, "cand_val": cv})
    # Fallback: whole-sentence diff if nothing was caught
    if not diffs:
        base_words = set(base_pred.lower().split()) - STOP
        cand_words = set(candidate.lower().split()) - STOP
        removed = base_words - cand_words
        added   = cand_words - base_words
        if removed or added:
            diffs.append({
                "dim": "general",
                "base_val": " ".join(sorted(removed)[:5]),
                "cand_val": " ".join(sorted(added)[:5]),
            })
    return diffs


# ---------------------------------------------------------------------------
# Step 3 — Evidence-Grounded Recheck (GPU)
# ---------------------------------------------------------------------------

def build_recheck_prompt(question: str, base_pred: str, candidate: str,
                          step1_obs: str, arm: str, diffs: list[dict]) -> str:
    """
    Ask the model to compare base_pred vs candidate against raw clip evidence.
    No GT, no F1.  arm determines which evidence channel to focus on.
    """
    if arm == "audio_only":
        channel = "audio evidence (sounds, voices, audio cues)"
    elif arm == "video_only":
        channel = "visual evidence (what is seen in the video frames)"
    else:
        channel = "both audio and visual evidence"

    diff_lines = ""
    for d in diffs[:3]:  # cap at 3 facts
        diff_lines += (f"  • {d['dim']}: original says \"{d['base_val']}\", "
                       f"candidate says \"{d['cand_val']}\"\n")
    if not diff_lines:
        diff_lines = "  (general rephrasing)\n"

    prompt = (
        f"You are verifying a factual correction for a video/audio clip.\n\n"
        f"Question asked: {question}\n\n"
        f"Original answer: {base_pred}\n\n"
        f"Proposed correction: {candidate}\n\n"
        f"Observed evidence from the clip: {step1_obs}\n\n"
        f"The following facts were changed:\n{diff_lines}\n"
        f"Based ONLY on the {channel} in this clip, which answer is more accurate?\n"
        f"Consider whether the evidence SUPPORTS, CONTRADICTS, or is INSUFFICIENT "
        f"to judge the changed facts.\n\n"
        f"Reply with exactly one of:\n"
        f"CANDIDATE_BETTER — the correction is supported by evidence\n"
        f"BASE_BETTER      — the original is more accurate\n"
        f"INSUFFICIENT     — not enough evidence to decide\n\n"
        f"Verdict:"
    )
    return prompt


def parse_recheck_verdict(raw: str) -> str:
    """Extract CANDIDATE_BETTER / BASE_BETTER / INSUFFICIENT from model output."""
    r = raw.upper()
    if "CANDIDATE_BETTER" in r or "CANDIDATE" in r and "BETTER" in r:
        return "CANDIDATE_BETTER"
    if "BASE_BETTER" in r or "BASE" in r and "BETTER" in r:
        return "BASE_BETTER"
    return "INSUFFICIENT"


# ---------------------------------------------------------------------------
# Step 4 — Confession-Guided Accept Gate
# ---------------------------------------------------------------------------

def confession_gate(det_passed: bool, det_reasons: list[str],
                    recheck_verdict: str) -> tuple[bool, str]:
    """
    Combine deterministic gate + evidence recheck verdict.
    Returns (accepted, reason_string).

    Accept only if:
      - Deterministic gate passes AND
      - Recheck says CANDIDATE_BETTER

    Reject if:
      - Deterministic gate fails, OR
      - Recheck says BASE_BETTER or INSUFFICIENT
    """
    if not det_passed:
        return False, f"det_reject:{','.join(det_reasons)}"
    if recheck_verdict == "CANDIDATE_BETTER":
        return True, "evidence_supported"
    if recheck_verdict == "BASE_BETTER":
        return False, "base_better"
    # INSUFFICIENT → preserve base
    return False, "evidence_insufficient"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm",   default="audio_only",
                        choices=["audio_only","video_only","combined"])
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--v1-pool", default=str(V1_OUT))
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ load frozen pool
    print("[v3] Loading frozen v1 candidate pool …")
    v1_rows = []
    with open(args.v1_pool) as f:
        for line in f:
            try:
                v1_rows.append(json.loads(line))
            except Exception:
                pass
    v1_rows = [r for r in v1_rows
               if int(r["id"]) < args.limit]
    print(f"[v3] {len(v1_rows)} rows loaded (limit={args.limit})")

    # ------------------------------------------------------------------ load AVHBench + media helpers
    print("[v3] Loading AVHBench data + model …")
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
        print(f"[v3] Using shm model at {shm_path}")
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
        print(f"[v3] Resuming — {len(done_ids)} already done")

    out_fh = open(OUT_FILE, "a")

    # ------------------------------------------------------------------ per-sample loop
    for v1_row in v1_rows:
        rid = v1_row["id"]
        if rid in done_ids:
            continue

        base_pred     = v1_row["base_pred"]
        candidate     = v1_row.get("candidate_text") or ""
        step1_obs     = v1_row.get("step1_obs") or ""
        question      = v1_row.get("question", "")
        gt            = v1_row.get("gt", "")
        base_f1       = v1_row.get("base_f1") or _token_f1(base_pred, gt)
        candidate_f1  = v1_row.get("candidate_f1")  # offline eval only

        # ---- rows with no candidate: carry forward as no_edit ----
        if not candidate or v1_row.get("status") == "no_candidates":
            row = dict(
                id=rid, question=question, gt=gt,
                base_pred=base_pred, status="no_candidate",
                candidate_text=None, candidate_f1=None,
                det_passed=None, det_reasons=[],
                changed_facts=[], recheck_verdict=None,
                accepted=False, accept_reason="no_candidate",
                repaired_pred=base_pred,
                base_f1=base_f1,
                repaired_f1=base_f1,
            )
            out_fh.write(json.dumps(row) + "\n")
            out_fh.flush()
            continue

        # ---- Step 1: deterministic gate ----
        det_passed, det_reasons = deterministic_gate(candidate, base_pred)

        # ---- Step 2: diff extraction ----
        diffs = extract_diff(base_pred, candidate)

        # ---- Step 3: evidence recheck (only if det gate passes) ----
        recheck_verdict = "SKIP"
        if det_passed and candidate.strip() != base_pred.strip():
            # build generate closure for this sample
            uid        = f"avhbench_{int(rid):06d}"
            data_item  = next((it for it in avhbench_data if it["uid"] == uid), None)
            pv, av, iv = None, None, None
            if data_item is not None:
                try:
                    pv, av, iv = load_media_for_item(
                        data_item, AVH_DATA, AVH_DATA_ROOT)
                except Exception:
                    pass

            def _generate(prompt_text, _pv=pv, _av=av, _iv=iv,
                           max_new_tokens=args.max_new_tokens):
                uc   = build_user_content_with_media_tokens(_pv, _av, _iv, prompt_text)
                msgs = make_messages(sys_prompt, uc)
                inp  = build_inputs(processor, model, msgs,
                                    _pv, _av, _iv, "haloquest")
                inp  = prepare_for_thinker(inp, model)
                with torch.inference_mode():
                    oids = model.thinker.generate(
                        **inp, max_new_tokens=max_new_tokens, do_sample=False)
                return decode_outputs(processor, oids,
                                      inp["input_ids"].shape[1])[0].strip()

            recheck_prompt  = build_recheck_prompt(
                question, base_pred, candidate, step1_obs, args.arm, diffs)
            recheck_raw     = _generate(recheck_prompt)
            recheck_verdict = parse_recheck_verdict(recheck_raw)
        elif not det_passed:
            recheck_verdict = "SKIPPED_DET_FAIL"

        # ---- Step 4: confession-guided accept ----
        accepted, accept_reason = confession_gate(det_passed, det_reasons,
                                                  recheck_verdict)

        repaired_pred = candidate if accepted else base_pred
        repaired_f1   = _token_f1(repaired_pred, gt)

        row = dict(
            id=rid, question=question, gt=gt,
            base_pred=base_pred, status="processed",
            candidate_text=candidate, candidate_f1=candidate_f1,
            det_passed=det_passed, det_reasons=det_reasons,
            changed_facts=diffs[:3],
            recheck_verdict=recheck_verdict,
            accepted=accepted, accept_reason=accept_reason,
            repaired_pred=repaired_pred,
            base_f1=base_f1,
            repaired_f1=repaired_f1,
        )
        out_fh.write(json.dumps(row) + "\n")
        out_fh.flush()
        print(f"  [{rid}] det={'PASS' if det_passed else 'FAIL':4s}  "
              f"recheck={recheck_verdict:20s}  "
              f"accepted={accepted}  "
              f"cand_f1={candidate_f1 if candidate_f1 is not None else '?'}")

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
    oracle_f1_mean   = safe_mean([
        max(r["base_f1"], r["candidate_f1"] if r.get("candidate_f1") else 0.0)
        for r in all_rows])

    n_total       = len(all_rows)
    n_accepted    = sum(1 for r in all_rows if r.get("accepted"))
    n_det_reject  = sum(1 for r in all_rows
                        if r.get("det_passed") is False)
    n_ev_reject   = sum(1 for r in all_rows
                        if r.get("recheck_verdict") in ("BASE_BETTER","INSUFFICIENT")
                        and r.get("det_passed") is True)
    n_ev_support  = sum(1 for r in all_rows
                        if r.get("recheck_verdict") == "CANDIDATE_BETTER")

    # Compare vs v1 known good/bad accepts (offline — uses GT for analysis only)
    # v1 good accept ids (delta > 0): 0,7,11,14,15,23,24,25,28,30,32,45,46,49
    v1_good_ids = {"0","7","11","14","15","23","24","25","28","30","32","45","46","49"}
    v1_bad_ids  = {"1","3","6","8","9","12","13","16","20","22","26","27","31",
                   "34","35","37","39","40","42","43","44","47","48"}
    n_pos_accept    = sum(1 for r in all_rows
                          if r.get("accepted") and r["id"] in v1_good_ids)
    n_neg_accept    = sum(1 for r in all_rows
                          if r.get("accepted") and r["id"] in v1_bad_ids)
    n_false_reject  = sum(1 for r in all_rows
                          if not r.get("accepted") and r["id"] in v1_good_ids)

    lines = [
        "# FactProbe v3 Summary — Description-Level + Confession-Guided Verification",
        "",
        "## Evaluation (same n=50 pool as v1, base_f1 reference = 27.05%)",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| n_total               | {n_total} |",
        f"| base_f1_mean          | {base_f1_mean:.4f} |",
        f"| repaired_f1_mean      | {repaired_f1_mean:.4f} |",
        f"| delta_f1              | {delta_f1:+.4f} |",
        f"| candidate_oracle_f1   | {oracle_f1_mean:.4f} |",
        f"| n_accepted            | {n_accepted} |",
        f"| n_det_rejected        | {n_det_reject} |",
        f"| n_ev_rejected         | {n_ev_reject} |",
        f"| n_evidence_supported  | {n_ev_support} |",
        "",
        "## Gate analysis vs v1 known outcomes (offline, GT used for analysis only)",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| n_positive_accept (v1 good ids accepted) | {n_pos_accept} / {len(v1_good_ids)} |",
        f"| n_negative_accept (v1 bad ids accepted)  | {n_neg_accept} / {len(v1_bad_ids)} |",
        f"| n_false_reject    (v1 good ids rejected) | {n_false_reject} / {len(v1_good_ids)} |",
        "",
        "## v1 comparison",
        "",
        "| Version | base_f1 | repaired_f1 | delta | oracle |",
        "|---|---|---|---|---|",
        "| v1 (open-ended probe) | 27.05 | 25.58 | -1.47 | 28.54 |",
        f"| v3 (confession-guided) | {base_f1_mean:.2f} | {repaired_f1_mean:.2f} "
        f"| {delta_f1:+.2f} | {oracle_f1_mean:.2f} |",
        "",
        "## Recheck verdict breakdown",
        "",
    ]
    verdict_counts = {}
    for r in all_rows:
        v = r.get("recheck_verdict", "NO_CANDIDATE")
        verdict_counts[v] = verdict_counts.get(v, 0) + 1
    for v, c in sorted(verdict_counts.items()):
        lines.append(f"- {v}: {c}")

    SUMMARY_F.write_text("\n".join(lines))
    print(f"\n[v3] Done. base={base_f1_mean:.4f} repaired={repaired_f1_mean:.4f} "
          f"oracle={oracle_f1_mean:.4f} delta={delta_f1:+.4f} "
          f"accepted={n_accepted}/{n_total}")
    print(f"[v3] n_positive_accept={n_pos_accept}  "
          f"n_negative_accept={n_neg_accept}  "
          f"n_false_reject={n_false_reject}")
    print(f"[v3] Summary → {SUMMARY_F}")

    msg = (f"FactProbe v3完成: delta={delta_f1:+.2f} "
           f"accepted={n_accepted}/{n_total} "
           f"pos_acc={n_pos_accept} neg_acc={n_neg_accept} "
           f"false_rej={n_false_reject}")
    os.system(f'/home/tsq/.claude/feishu/feishu_send.sh "{msg}"')


if __name__ == "__main__":
    main()
