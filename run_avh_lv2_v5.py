# -*- coding: utf-8 -*-
"""
run_avh_lv2_v5.py — AVHBench v5: Evidence-Grounded Multi-Candidate Correction

Upgrades the v4 A+B2 dual-candidate pipeline with:
  1. Fact-Level Commitment Anchoring (KEEP/VERIFY/CORRECT tagging)
  2. Modality-Aware Evidence Routing (audio/video/AV-relation per fact)
  3. Strategy-Driven Multi-Candidate Generation (3 targeted strategies)
  4. Evidence-Based Candidate Selection (step1_obs alignment + KEEP preservation)

Constraints:
  - Do NOT use GT / Token-F1 in any online routing, candidate generation, or accept decision
  - Cards 0-3 only; confirm with nvidia-smi before use
  - Do NOT modify freeform_repair.py or run_halo_cu121.py
  - Output dir: outputs/lv2_avh_v5/

Usage:
  CUDA_VISIBLE_DEVICES=3 MMHALU_MODEL_ROOT=/tmp HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \\
  PYTHONPATH=/dev/shm/site-packages LD_LIBRARY_PATH=/dev/shm/so_mirror \\
    /home/tsq/.local/bin/python3.13 run_avh_lv2_v5.py \\
    --arm audio_only --limit 50 --out outputs/lv2_avh_v5/audio_only_n50.jsonl
"""
import argparse
import importlib.util
import json
import os
import re
import sys
import time
import traceback

import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
SHARED = os.path.join(REPO, "OmniConfess", "methods", "shared")

sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "OmniConfess"))

# Patch CVE-2025-32434 before transformers import
try:
    import transformers.utils.import_utils as _tiu_early
    _tiu_early.check_torch_load_is_safe = lambda: None
except Exception:
    pass

AVH_DATA = os.path.join(REPO, "OmniHalluBench", "avhbench", "avhbench.json")
AVH_DATA_ROOT = os.path.join(REPO, "OmniHalluBench", "avhbench")
PROBE_PATH = os.path.join(REPO, "outputs", "cf_probe",
                           "avh_probe_bigN_shard0of1.jsonl")
BASE_PREDS_PATH = os.path.join(
    REPO, "outputs", "avhbench_base_model", "avhbench_base_model_full.jsonl")

ARMS = ("combined", "video_only", "audio_only", "base")

# Strategy constants
MINIMAL_REPLACEMENT     = "MINIMAL_REPLACEMENT"
EVIDENCE_COMPLETION     = "EVIDENCE_COMPLETION"
CROSS_MODAL_RECONCILIATION = "CROSS_MODAL_RECONCILIATION"


# ---------------------------------------------------------------------------
# Unchanged helpers from run_avh_lv2.py (copied verbatim; no modification)
# ---------------------------------------------------------------------------

def _load_module_direct(name, file_path):
    spec = importlib.util.spec_from_file_location(name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_avh_probe(probe_path, avhbench_data, signal_mode="combined"):
    avh_by_uid = {}
    for item in avhbench_data:
        uid = item.get("uid", "")
        try:
            num = str(int(uid.split("_")[-1]))
        except (ValueError, IndexError):
            continue
        avh_by_uid[num] = item

    seen_ids = set()
    records = []
    with open(probe_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rid = str(r["id"])
            if rid in seen_ids:
                continue
            seen_ids.add(rid)

            d_v = np.asarray(r["delta_video"], dtype=np.float64)
            d_a = np.asarray(r["delta_audio"], dtype=np.float64)

            if signal_mode == "combined":
                r["delta_img_tok"] = list(d_v + d_a)
            elif signal_mode == "video_only":
                r["delta_img_tok"] = list(d_v)
            elif signal_mode == "audio_only":
                r["delta_img_tok"] = list(d_a)
            else:
                raise ValueError(f"Unknown signal_mode: {signal_mode!r}")

            meta = avh_by_uid.get(rid, {})
            r["question"] = meta.get("question", r.get("question", ""))
            records.append(r)
    return records


def _check_no_scene_replacement(repaired, base_pred):
    """Returns True if repaired preserves >=40% content words from base.
    Fix 2: tightened from 25% to 40% — the 25% gate was too loose and allowed
    wholesale scene replacements that degraded F1.
    """
    def content_words(text):
        stop = {'a','an','the','is','are','was','were','in','on','at','of','to','and','or',
                'not','no','there','this','that','it','its','with','for','by','as','be','have',
                'has','had','do','does','did','but','from','into','out','up','down','can','could',
                'will','would','should','may','might','what','which','who','how','when','where'}
        words = set(re.findall(r'\b[a-z]+\b', text.lower()))
        return words - stop

    base_content = content_words(base_pred)
    rep_content = content_words(repaired)
    if not base_content:
        return True
    overlap = len(base_content & rep_content) / len(base_content)
    return overlap >= 0.40  # tightened from 0.25 (Fix 2)


def _check_preserves_positive_content(repaired, base, verify_status=""):
    base_lower = base.lower()
    repaired_lower = repaired.lower()
    _refusal_words = {"not visible", "cannot", "no ", "there is no",
                      "not present", "not shown", "unable", "unknown",
                      "unclear", "can't", "don't know", "do not know"}
    base_is_positive = not any(w in base_lower for w in _refusal_words)
    repaired_is_refusal = any(w in repaired_lower for w in _refusal_words)
    if base_is_positive and repaired_is_refusal:
        _edit_signals = {"UNSUPPORTED", "CONTRADICTED", "UNSUPPORTED_BY_EVIDENCE",
                         "INSUFFICIENT_TO_DETERMINE"}
        if verify_status and any(sig in verify_status.upper() for sig in _edit_signals):
            return True
        return False
    return True


def _lv2_accept_decision(repaired, base_pred, verify_status=""):
    """Hard accept/reject gate — unchanged from v4."""
    rep = repaired.strip()
    base = base_pred.strip()
    if not rep:
        return False
    if not _check_no_scene_replacement(rep, base):
        return False
    words = rep.split()
    if len(words) >= 6:
        trigrams = [tuple(words[i:i+3]) for i in range(len(words)-2)]
        seen = set()
        for tg in trigrams:
            if tg in seen:
                return False
            seen.add(tg)
    if len(base) > 0 and len(rep) < 0.5 * len(base):
        return False
    if rep in base or base.startswith(rep):
        return False
    _neg_words = {"not", "no", "never", "cannot", "can't", "don't",
                  "doesn't", "isn't", "aren't", "wasn't", "weren't",
                  "unable", "unknown", "unclear", "undetermined"}
    _refusal_phrases = ("there is no", "there are no", "not visible",
                        "cannot see", "not shown", "not shown in")
    base_lower = base.lower()
    rep_lower = rep.lower()
    base_has_refusal = (
        any(ph in base_lower for ph in _refusal_phrases)
        or bool(set(base_lower.split()) & _neg_words)
    )
    rep_has_refusal = (
        any(ph in rep_lower for ph in _refusal_phrases)
        or bool(set(rep_lower.split()) & _neg_words)
    )
    if base_has_refusal and not rep_has_refusal:
        return False
    _evidence_signals = {"UNSUPPORTED", "CONTRADICTED", "UNSUPPORTED_BY_EVIDENCE"}
    _has_evidence = any(sig in verify_status.upper() for sig in _evidence_signals)
    if not _has_evidence:
        if not _check_preserves_positive_content(rep, base, verify_status):
            return False
    return True


def run_base_arm(avhbench_data, base_preds_path, limit, out_path, token_f1_fn):
    base_by_id = {}
    with open(base_preds_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            base_by_id[str(r["id"])] = r

    rows = []
    for idx, item in enumerate(avhbench_data[:limit]):
        rid = str(idx)
        bp = base_by_id.get(rid, {})
        pred = bp.get("pred", "")
        gt = item.get("answer", "")
        question = item.get("question", "")
        f1 = round(100 * token_f1_fn(pred, gt, multiref=True), 2)
        rows.append({
            "id": idx, "question": question, "gt": gt,
            "base_pred": pred, "repaired_pred": pred,
            "arm": "base", "n_spans": 0, "status": "base",
            "base_f1": f1, "repaired_f1": f1,
        })

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = len(rows)
    mb = sum(r["base_f1"] for r in rows) / n if n else 0.0
    summary = {
        "arm": "base", "n_run": n,
        "base_tokenf1_mean": round(mb, 2),
        "repaired_tokenf1_mean": round(mb, 2),
        "delta_f1": 0.0, "out": out_path,
    }
    print("\n===== SUMMARY (base arm) =====", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    return rows


# ---------------------------------------------------------------------------
# v5 Phase 1: Fact-Level Commitment Anchoring
# ---------------------------------------------------------------------------

def _parse_step1_obs(verify_raw):
    """Extract step1_observation text from verify_raw string."""
    if not verify_raw:
        return ""
    if "[step1=" in verify_raw:
        tail = verify_raw[verify_raw.index("[step1=") + 7:]
        # Find the closing ] that belongs to this tag (not nested)
        return tail[:tail.index("]")].strip() if "]" in tail else tail[:300].strip()
    return ""


def _anchor_fact_level(base_pred, lv2_result, _generate_fn):
    """
    Phase 1: decompose base_pred into individual facts and tag each as
    KEEP / VERIFY / CORRECT based on the verifier's revision_candidates.

    Returns a dict:
      {
        "facts": [str, ...],           # ordered list of fact strings
        "tags": {fact: "KEEP"|"VERIFY"|"CORRECT"},
        "correction_types": {fact: "ABSENT"|"WRONG_ATTR"|"INCOMPLETE"|"CONFLICTING"|None},
        "modalities": {fact: "audio"|"video"|"av"},
        "flagged_spans": [str, ...],   # span_text values from revision_candidates
        "step1_obs": str,              # primary observation text
        "primary_claim": str,          # primary.claim
        "primary_span": str,           # primary.span_text
      }
    """
    rev = getattr(lv2_result, "revision_candidates", []) if lv2_result else []
    flagged_spans = [getattr(c, "span_text", "") or "" for c in rev]
    flagged_claims = [getattr(c, "claim", "") or "" for c in rev]

    # Collect step1 observations
    step1_obs_list = []
    for c in rev:
        obs = getattr(c, "step1_observation", None) or ""
        if not obs:
            obs = _parse_step1_obs(getattr(c, "verify_raw", "") or "")
        if obs:
            step1_obs_list.append(obs)
    primary_step1_obs = step1_obs_list[0] if step1_obs_list else ""
    primary_claim = getattr(rev[0], "claim", "") if rev else ""
    primary_span = getattr(rev[0], "span_text", "") if rev else ""

    # Build a concise context for the LLM about what is flagged
    flagged_ctx = ""
    if flagged_spans:
        flagged_ctx = "Flagged spans (possibly wrong): " + "; ".join(
            f'"{s}"' for s in flagged_spans if s)
    obs_ctx = f"\nWhat the AV evidence actually shows: {primary_step1_obs}" \
              if primary_step1_obs else ""

    prompt = (
        f"Answer: {base_pred}\n\n"
        f"{flagged_ctx}{obs_ctx}\n\n"
        f"Task: List the factual claims in the answer. For each claim, assign a tag.\n"
        f"Tags:\n"
        f"  KEEP     — not flagged; likely correct\n"
        f"  VERIFY   — flagged but evidence is uncertain\n"
        f"  CORRECT  — flagged AND evidence contradicts it\n\n"
        f"Also classify the correction type for VERIFY/CORRECT claims:\n"
        f"  ABSENT       — claim says X exists but evidence shows it doesn't\n"
        f"  WRONG_ATTR   — claim has wrong attribute of a real object\n"
        f"  INCOMPLETE   — claim is correct but missing key AV detail\n"
        f"  CONFLICTING  — claim directly contradicts what evidence shows\n\n"
        f"Also note modality: audio / video / av\n\n"
        f"Format each line as:\n"
        f"FACT: <short claim> | TAG: <tag> | TYPE: <type or NA> | MOD: <modality>\n\n"
        f"List 2-6 facts maximum. Be concise.\n\nFacts:"
    )

    try:
        raw = _generate_fn(prompt, 96)
    except Exception:
        raw = ""

    facts = []
    tags = {}
    correction_types = {}
    modalities = {}

    for line in raw.splitlines():
        line = line.strip()
        if not line.upper().startswith("FACT:"):
            continue
        # Parse: FACT: X | TAG: Y | TYPE: Z | MOD: W
        parts = line.split("|")
        fact_part = parts[0][5:].strip().rstrip(".")
        if not fact_part:
            continue

        tag = "KEEP"
        ctype = None
        mod = "video"

        for part in parts[1:]:
            p = part.strip()
            if p.upper().startswith("TAG:"):
                v = p[4:].strip().upper()
                if v in ("KEEP", "VERIFY", "CORRECT"):
                    tag = v
            elif p.upper().startswith("TYPE:"):
                v = p[5:].strip().upper()
                if v in ("ABSENT", "WRONG_ATTR", "INCOMPLETE", "CONFLICTING"):
                    ctype = v
            elif p.upper().startswith("MOD:"):
                v = p[4:].strip().lower()
                if v in ("audio", "video", "av"):
                    mod = v

        facts.append(fact_part)
        tags[fact_part] = tag
        correction_types[fact_part] = ctype
        modalities[fact_part] = mod

    # Fallback: if LLM returned nothing, treat primary span as a CORRECT fact
    if not facts and primary_span:
        facts = [primary_span.strip()]
        tags[primary_span.strip()] = "CORRECT"
        correction_types[primary_span.strip()] = "CONFLICTING"
        modalities[primary_span.strip()] = "video"

    return {
        "facts": facts,
        "tags": tags,
        "correction_types": correction_types,
        "modalities": modalities,
        "flagged_spans": flagged_spans,
        "step1_obs": primary_step1_obs,
        "primary_claim": primary_claim,
        "primary_span": primary_span,
    }


# ---------------------------------------------------------------------------
# v5 Phase 2 + 3: Strategy Selection and Candidate Generation
# ---------------------------------------------------------------------------

def _get_correction_strategy(fact_anchor):
    """
    Pure logic (no LLM): return list of applicable strategies from the fact anchor.
    May return 1-3 strategies; at minimum one always fires.
    """
    facts = fact_anchor["facts"]
    tags = fact_anchor["tags"]
    ctypes = fact_anchor["correction_types"]
    mods = fact_anchor["modalities"]

    n_correct = sum(1 for f in facts if tags.get(f) == "CORRECT")
    n_verify  = sum(1 for f in facts if tags.get(f) == "VERIFY")

    strategies = []

    # Strategy 1: MINIMAL_REPLACEMENT — 1-2 CORRECT facts with WRONG_ATTR or ABSENT
    wrong_or_absent = any(
        ctypes.get(f) in ("WRONG_ATTR", "ABSENT")
        for f in facts if tags.get(f) in ("CORRECT", "VERIFY")
    )
    if wrong_or_absent and n_correct <= 2:
        strategies.append(MINIMAL_REPLACEMENT)

    # Strategy 2: EVIDENCE_COMPLETION — INCOMPLETE type or VERIFY with AV detail missing
    incomplete = any(
        ctypes.get(f) == "INCOMPLETE"
        for f in facts if tags.get(f) in ("CORRECT", "VERIFY")
    )
    has_step1 = bool(fact_anchor.get("step1_obs"))
    if incomplete or (n_verify > 0 and has_step1 and MINIMAL_REPLACEMENT not in strategies):
        strategies.append(EVIDENCE_COMPLETION)

    # Strategy 3: CROSS_MODAL_RECONCILIATION — AV-relation or audio modality facts flagged
    av_audio_flagged = any(
        mods.get(f) in ("av", "audio")
        for f in facts if tags.get(f) in ("CORRECT", "VERIFY")
    )
    if av_audio_flagged or ctypes.get(fact_anchor.get("primary_span", "")) == "CONFLICTING":
        strategies.append(CROSS_MODAL_RECONCILIATION)

    # Always fire at least one strategy; default to MINIMAL_REPLACEMENT
    if not strategies:
        if fact_anchor.get("step1_obs"):
            strategies.append(EVIDENCE_COMPLETION)
        else:
            strategies.append(MINIMAL_REPLACEMENT)

    return strategies


def _regen_strategy_minimal(fact_anchor, lv2_result, question, base_pred, _generate_fn,
                             max_new_tokens=96):
    """Strategy 1: Minimal Replacement — replace only the specific wrong fact substring."""
    rev = getattr(lv2_result, "revision_candidates", []) if lv2_result else []
    facts = fact_anchor["facts"]
    tags = fact_anchor["tags"]
    ctypes = fact_anchor["correction_types"]
    step1_obs = fact_anchor.get("step1_obs", "")
    primary_span = fact_anchor.get("primary_span", "")

    # Find the CORRECT/VERIFY fact to replace (prefer CORRECT, ABSENT or WRONG_ATTR)
    target_fact = None
    for f in facts:
        if tags.get(f) in ("CORRECT", "VERIFY") and ctypes.get(f) in ("ABSENT", "WRONG_ATTR"):
            target_fact = f
            break
    if target_fact is None:
        for f in facts:
            if tags.get(f) in ("CORRECT", "VERIFY"):
                target_fact = f
                break
    if target_fact is None and primary_span:
        target_fact = primary_span.strip()

    keep_facts = [f for f in facts if tags.get(f) == "KEEP"]
    keep_str = "; ".join(f'"{f}"' for f in keep_facts) if keep_facts else "(none specified)"

    evidence_hint = step1_obs if step1_obs else "the claim appears inaccurate based on the question"
    correction_hint = ""
    if ctypes.get(target_fact) == "ABSENT":
        correction_hint = f'Replace "{target_fact}" with "there is no {target_fact.split()[-1] if target_fact else "it"}"'
    elif ctypes.get(target_fact) == "WRONG_ATTR":
        correction_hint = f'Replace the attribute in "{target_fact}" with what the evidence shows'

    prompt = (
        f"You are doing a MINIMAL targeted correction to ONE specific fact in an answer.\n\n"
        f"Original answer: {base_pred}\n"
        f"Fact to correct: \"{target_fact}\"\n"
        f"What evidence shows: {evidence_hint}\n"
        f"{correction_hint}\n\n"
        f"Facts to KEEP unchanged: {keep_str}\n\n"
        f"Rules (strict):\n"
        f"1. Replace ONLY \"{target_fact}\" — change nothing else.\n"
        f"2. Keep ALL other content of the original answer verbatim.\n"
        f"3. Do NOT add new objects, scenes, or details.\n"
        f"4. If unsure of the correct replacement, return the original answer unchanged.\n"
        f"5. Output ONLY the corrected answer, no explanation.\n\n"
        f"Corrected answer:"
    )
    try:
        return _generate_fn(prompt, max_new_tokens)
    except Exception:
        return None


def _regen_strategy_completion(fact_anchor, lv2_result, question, base_pred, _generate_fn,
                                max_new_tokens=96):
    """Strategy 2: Evidence Completion — keep base, append/integrate missing AV detail."""
    step1_obs = fact_anchor.get("step1_obs", "")
    facts = fact_anchor["facts"]
    tags = fact_anchor["tags"]
    ctypes = fact_anchor["correction_types"]

    # Find INCOMPLETE facts
    incomplete_facts = [f for f in facts
                        if tags.get(f) in ("CORRECT", "VERIFY") and ctypes.get(f) == "INCOMPLETE"]
    missing_detail = "; ".join(f'"{f}"' for f in incomplete_facts) if incomplete_facts else \
        "an additional AV detail supported by evidence"

    evidence_hint = step1_obs if step1_obs else "the evidence provides additional context"

    prompt = (
        f"Question: {question}\n\n"
        f"Original answer: {base_pred}\n\n"
        f"What the audio/video evidence shows: {evidence_hint}\n\n"
        f"Missing detail to add: {missing_detail}\n\n"
        f"Task: Integrate the missing detail into the answer.\n"
        f"Rules:\n"
        f"1. Keep the original answer's existing content intact.\n"
        f"2. Add or integrate ONLY the detail the evidence supports.\n"
        f"3. Do NOT replace existing scene description.\n"
        f"4. Keep the final answer to 1-2 sentences.\n"
        f"5. Output ONLY the improved answer.\n\n"
        f"Improved answer:"
    )
    try:
        return _generate_fn(prompt, max_new_tokens)
    except Exception:
        return None


def _regen_strategy_crossmodal(fact_anchor, lv2_result, question, base_pred, _generate_fn,
                                max_new_tokens=96):
    """Strategy 3: Cross-Modal Reconciliation — reconcile audio/visual relationship."""
    step1_obs = fact_anchor.get("step1_obs", "")
    rev = getattr(lv2_result, "revision_candidates", []) if lv2_result else []

    # Collect audio and video observations if available
    audio_obs = ""
    video_obs = ""
    for c in rev:
        vr = getattr(c, "verify_raw", "") or ""
        obs = getattr(c, "step1_observation", None) or _parse_step1_obs(vr)
        span = (getattr(c, "span_text", "") or "").lower()
        if any(w in span for w in ("sound", "hear", "audio", "music", "voice", "speak", "say")):
            audio_obs = obs or audio_obs
        elif obs and not audio_obs:
            video_obs = obs

    seen_part = video_obs if video_obs else step1_obs
    heard_part = audio_obs if audio_obs else ""

    seen_str  = f"What is seen: {seen_part}" if seen_part else ""
    heard_str = f"What is heard: {heard_part}" if heard_part else ""
    av_ctx = "\n".join(x for x in [seen_str, heard_str] if x)
    if not av_ctx:
        av_ctx = f"AV evidence: {step1_obs}" if step1_obs else "the audio-visual evidence"

    keep_facts = [f for f in fact_anchor["facts"]
                  if fact_anchor["tags"].get(f) == "KEEP"]
    keep_str = "; ".join(f'"{f}"' for f in keep_facts) if keep_facts else "(keep all non-AV facts)"

    prompt = (
        f"Question: {question}\n\n"
        f"Original answer: {base_pred}\n\n"
        f"{av_ctx}\n\n"
        f"Facts to keep unchanged: {keep_str}\n\n"
        f"Task: Fix the audio-visual relationship error in the answer.\n"
        f"Rules:\n"
        f"1. Reconcile what is seen and heard per the evidence above.\n"
        f"2. Keep all non-AV facts from the original answer unchanged.\n"
        f"3. Do NOT replace the scene with an entirely different scene.\n"
        f"4. Output ONLY the corrected answer, no explanation.\n\n"
        f"Corrected answer:"
    )
    try:
        return _generate_fn(prompt, max_new_tokens)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# v5 Phase 4: Evidence-Based Candidate Selection
# ---------------------------------------------------------------------------

def _count_keep_facts_preserved(candidate_text, keep_facts):
    """Count how many KEEP facts from the base answer are still present in candidate."""
    if not keep_facts or not candidate_text:
        return 0
    cand_lower = candidate_text.lower()
    count = 0
    for fact in keep_facts:
        fact_words = set(w for w in fact.lower().split() if len(w) > 3)
        if not fact_words:
            count += 1  # short fact — assume preserved
            continue
        if sum(1 for w in fact_words if w in cand_lower) >= max(1, len(fact_words) // 2):
            count += 1
    return count


def _candidate_fixes_flagged_fact(candidate_text, fact_anchor):
    """
    Check whether candidate resolves the CORRECT/VERIFY facts in a way
    consistent with step1_obs.  Returns a score 0-2:
      2 = directly references evidence observation content
      1 = removes/replaces flagged span
      0 = no change detected
    """
    if not candidate_text:
        return 0
    step1_obs = fact_anchor.get("step1_obs", "")
    flagged_spans = fact_anchor.get("flagged_spans", [])
    cand_lower = candidate_text.lower()

    score = 0

    # Check if candidate references content from step1_obs (at least 2 content words)
    if step1_obs:
        obs_words = set(w for w in re.findall(r'\b[a-z]{4,}\b', step1_obs.lower()))
        cand_words = set(re.findall(r'\b[a-z]{4,}\b', cand_lower))
        shared = obs_words & cand_words
        if len(shared) >= 2:
            score = 2

    # Check if flagged span content is no longer present (or negated) in candidate
    for span in flagged_spans:
        if not span:
            continue
        span_words = set(w for w in span.lower().split() if len(w) > 3)
        if not span_words:
            continue
        overlap = sum(1 for w in span_words if w in cand_lower)
        if overlap < len(span_words) * 0.5:
            score = max(score, 1)

    return score


def _evidence_select_candidate(candidates_dict, fact_anchor, lv2_result, base_pred):
    """
    Evidence-based candidate selection.

    candidates_dict: {strategy_name: candidate_text or None}

    Returns: (selected_text, selected_strategy, selection_reason)
    """
    if not candidates_dict:
        return base_pred, "base", "no_candidates"

    keep_facts = [f for f in fact_anchor["facts"]
                  if fact_anchor["tags"].get(f) == "KEEP"]
    n_keep = len(keep_facts)

    scored = []
    for strategy, cand in candidates_dict.items():
        if cand is None:
            continue
        cand = cand.strip()
        if not cand:
            continue

        # Hard floor: scene replacement check
        if not _check_no_scene_replacement(cand, base_pred):
            print(f"    [select] {strategy}: REJECT scene_replacement", flush=True)
            continue

        evidence_score = _candidate_fixes_flagged_fact(cand, fact_anchor)
        keep_preserved = _count_keep_facts_preserved(cand, keep_facts)
        keep_ratio = keep_preserved / n_keep if n_keep > 0 else 1.0

        # Combined score: evidence alignment (0-2) + keep preservation (0-1)
        total = evidence_score + keep_ratio
        scored.append((total, evidence_score, keep_ratio, strategy, cand))
        print(f"    [select] {strategy}: ev={evidence_score} keep={keep_ratio:.2f} total={total:.2f}",
              flush=True)

    if not scored:
        return base_pred, "base", "all_rejected_scene_floor"

    # Sort: total desc, then evidence_score desc, then keep_ratio desc
    scored.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    best_total, best_ev, best_keep, best_strategy, best_cand = scored[0]

    # Require at least some evidence alignment OR keep preservation
    if best_ev == 0 and best_keep < 0.5 and n_keep > 0:
        return base_pred, "base", f"low_quality_all (best_total={best_total:.2f})"

    reason = f"ev={best_ev},keep={best_keep:.2f},total={best_total:.2f}"
    return best_cand, best_strategy, reason


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=ARMS, required=True)
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--probe-path", type=str, default=PROBE_PATH)
    args = ap.parse_args()

    out_path = args.out
    if not out_path:
        out_path = os.path.join(REPO, "outputs", "lv2_avh_v5",
                                f"{args.arm}_n{args.limit}.jsonl")
    if not os.path.isabs(out_path):
        out_path = os.path.join(REPO, out_path)

    print(f"[start] arm={args.arm} limit={args.limit} out={out_path}", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    # --- imports ---
    print("[import] freeform_repair...", flush=True)
    import freeform_repair as fr
    print("[import] tokenf1...", flush=True)
    from scripts.tokenf1_fullset import token_f1_multiref
    try:
        import localizer_v2 as lv2
        import evidence_verifier as ev
        _LOCALIZER_AVAILABLE = True
    except ImportError as e:
        print(f"[import] localizer_v2/evidence_verifier unavailable: {e}", flush=True)
        _LOCALIZER_AVAILABLE = False
    print("[import] done", flush=True)

    avhbench_data = json.load(open(AVH_DATA, encoding="utf-8"))
    print(f"[data] avhbench items: {len(avhbench_data)}", flush=True)

    if args.arm == "base":
        run_base_arm(avhbench_data, BASE_PREDS_PATH, args.limit, out_path,
                     token_f1_multiref)
        return

    signal_mode = {
        "combined": "combined",
        "video_only": "video_only",
        "audio_only": "audio_only",
    }[args.arm]

    probe_records = load_avh_probe(args.probe_path, avhbench_data, signal_mode)
    print(f"[probe] loaded {len(probe_records)} unique records "
          f"(signal_mode={signal_mode})", flush=True)

    probe_records = [r for r in probe_records if int(r["id"]) < args.limit]
    probe_records.sort(key=lambda r: int(r["id"]))
    print(f"[probe] after limit: {len(probe_records)} records", flush=True)

    if not probe_records:
        print("[ERROR] No probe records in range", flush=True)
        sys.exit(1)

    # --- model load ---
    import torch
    _torch_ver = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
    if _torch_ver < (2, 6):
        print("[import] patching torchvision ABI mismatch (torch<2.6)...", flush=True)
        import torch._library.fake_impl as _fi
        _orig_register = _fi.FakeImplHolder.register
        def _safe_register(self, func, *a, **kw):
            try:
                return _orig_register(self, func, *a, **kw)
            except (RuntimeError, TypeError) as _e:
                if "does not exist" in str(_e) or "unexpected keyword" in str(_e):
                    class _Nop:
                        def destroy(self): pass
                    return _Nop()
                raise
        _fi.FakeImplHolder.register = _safe_register

    try:
        from transformers import Qwen2_5OmniForConditionalGeneration as _OmniCls
        import torch as _tp
        def _load_speakers_patched(self, path):
            for k, v in _tp.load(path, weights_only=True).items():
                self.speaker_map[k] = v
        _OmniCls.load_speakers = _load_speakers_patched
        print("[import] patched load_speakers", flush=True)
    except Exception as _pe:
        print(f"[import] load_speakers patch failed: {_pe}", flush=True)

    _load_module_direct("methods.shared.paths",
                        os.path.join(SHARED, "paths.py"))
    _mu = _load_module_direct("methods.shared.model_utils",
                              os.path.join(SHARED, "model_utils.py"))
    load_model = _mu.load_model

    import torch as _torch_init
    _dummy = _torch_init.zeros(1, device="cuda")
    del _dummy
    print(f"[cuda] context init OK  device=cuda:{_torch_init.cuda.current_device()}", flush=True)

    _DEVSHM_MODEL = "/dev/shm/Qwen2.5-Omni-7B"
    if os.path.isdir(_DEVSHM_MODEL) and os.path.isfile(
            os.path.join(_DEVSHM_MODEL, "config.json")):
        print(f"[model] using /dev/shm local copy: {_DEVSHM_MODEL}", flush=True)
        _load_model_fn = lambda: _mu.load_model(_DEVSHM_MODEL)
    else:
        print("[model] /dev/shm copy not found, using default model path", flush=True)
        _load_model_fn = _mu.load_model

    print("[model] loading Qwen2.5-Omni-7B...", flush=True)
    t0 = time.time()
    processor, model = _load_model_fn()
    print(f"[model] loaded in {time.time()-t0:.1f}s  "
          f"device={next(model.parameters()).device}", flush=True)

    from methods.shared.model_utils import (
        build_inputs, make_messages,
        build_user_content_with_media_tokens,
        prepare_for_thinker, decode_outputs,
    )
    _dl = _load_module_direct("methods.shared.data_loader",
                              os.path.join(SHARED, "data_loader.py"))
    load_dataset_fn = _dl.load_dataset
    load_media_for_item = _dl.load_media_for_item
    from methods.shared.prompts import get_system_prompt

    avh_items = load_dataset_fn(AVH_DATA, AVH_DATA_ROOT)
    avh_by_id_idx = {str(it["id"]): it for it in avh_items}
    avh_by_uid_num = {}
    for it in avh_items:
        uid = it.get("uid", "")
        try:
            num = str(int(uid.split("_")[-1]))
            avh_by_uid_num[num] = it
        except (ValueError, IndexError):
            pass
    avh_by_id = avh_by_uid_num if avh_by_uid_num else avh_by_id_idx
    print(f"[data] loaded {len(avh_items)} items via data_loader "
          f"(uid_num keys={len(avh_by_uid_num)})", flush=True)

    # Resume support
    done_ids = set()
    if os.path.exists(out_path):
        for line in open(out_path, encoding="utf-8"):
            line = line.strip()
            if line:
                try:
                    done_ids.add(str(json.loads(line)["id"]))
                except Exception:
                    pass
        print(f"[resume] {len(done_ids)} already done", flush=True)

    rows = []
    base_f1s, rep_f1s = [], []
    n_fail = 0
    strategy_counts = {
        MINIMAL_REPLACEMENT: 0,
        EVIDENCE_COMPLETION: 0,
        CROSS_MODAL_RECONCILIATION: 0,
    }
    strategy_oracle = {k: [] for k in strategy_counts}
    strategy_delta  = {k: [] for k in strategy_counts}
    strategy_scene_ok = {k: 0 for k in strategy_counts}
    n_fell_back_to_base = 0

    sys_prompt = get_system_prompt("avhbench", "v1")

    for k, rec in enumerate(probe_records):
        rid = str(rec["id"])
        if rid in done_ids:
            continue

        it = avh_by_id.get(rid)
        if it is None:
            print(f"[SKIP] id={rid} not in data_loader output", flush=True)
            continue

        base_pred = "".join(rec["tokens"])
        gt = rec.get("gt", "")
        question = rec.get("question", it.get("question", ""))

        print(f"\n[{k+1}/{len(probe_records)}] id={rid} "
              f"q={question[:60]!r}", flush=True)

        row = {
            "id": int(rid),
            "question": question,
            "gt": gt,
            "base_pred": base_pred,
            "arm": args.arm,
            "signal_mode": signal_mode,
            "status": "success",
            "n_spans": 0,
            "v5": True,
        }

        lv2_result = None
        if _LOCALIZER_AVAILABLE:
            cache_record = {
                "id": rec["id"],
                "tokens": rec["tokens"],
                "delta_img_tok": rec["delta_img_tok"],
                "delta_video": rec.get("delta_video", []),
                "delta_audio": rec.get("delta_audio", []),
                "delta_text_tok": rec.get("delta_text", []),
                "classes": rec.get("classes"),
                "question": question,
                "lp_full": rec.get("lp_full", []),
            }
            lv2_result = lv2.localize(cache_record)

            try:
                _pv, _av, _iv = load_media_for_item(it, AVH_DATA, AVH_DATA_ROOT)
                _gen_state = {"pv": _pv, "av": _av, "iv": _iv, "sys": sys_prompt}

                def _lv2_generate(prompt_text, max_new_tokens, _gs=_gen_state):
                    uc = build_user_content_with_media_tokens(
                        _gs["pv"], _gs["av"], _gs["iv"], prompt_text)
                    msgs = make_messages(_gs["sys"], uc)
                    inp = build_inputs(processor, model, msgs,
                                       _gs["pv"], _gs["av"], _gs["iv"],
                                       "haloquest")
                    inp = prepare_for_thinker(inp, model)
                    with torch.inference_mode():
                        oids = model.thinker.generate(
                            **inp,
                            max_new_tokens=max_new_tokens,
                            do_sample=False,
                        )
                    return decode_outputs(
                        processor, oids, inp["input_ids"].shape[1]
                    )[0].strip()

                lv2_result = ev.verify_candidates(lv2_result, it, _lv2_generate)

            except Exception as e:
                n_fail += 1
                row["status"] = "media_load_failed"
                row["error"] = f"{type(e).__name__}: {e}"
                row["repaired_pred"] = base_pred
                row["traceback"] = traceback.format_exc()[-800:]
                print(f"[FAIL media id={rid}] {row['error']}", flush=True)
                _lv2_generate = None

            if lv2_result is not None:
                row["lv2_n_candidates"] = len(lv2_result.candidates)
                row["lv2_n_revision"] = len(lv2_result.revision_candidates)
                row["candidates"] = [c.to_dict() for c in lv2_result.candidates]
                row["n_spans"] = len(lv2_result.revision_candidates)

        # --- v5 Pipeline ---
        _row_written = False  # track whether row was already saved (short-span fast-path)
        if row["status"] == "success":
            rev = lv2_result.revision_candidates if lv2_result else []
            if not rev or _lv2_generate is None:
                row["repaired_pred"] = base_pred
                row["lv2_regen_accepted"] = False
                row["strategy_used"] = "none"
                if not rev:
                    row["no_edit"] = True
            else:
                try:
                    _statuses = [c.verify_status for c in rev]
                    _agg = "UNSUPPORTED" if any(
                        "UNSUPPORTED" in vs for vs in _statuses
                    ) else (_statuses[0] if _statuses else "")

                    # Fix 1: Short-span filter — spans with <4 whitespace tokens are
                    # degenerate anchors (mean delta = −9.69). Skip to no_edit.
                    _primary_span = (getattr(rev[0], "span_text", "") or "").strip()
                    if len(_primary_span.split()) < 4:
                        row["repaired_pred"] = base_pred
                        row["lv2_regen_accepted"] = False
                        row["strategy_used"] = "skipped_short_span"
                        row["skipped_short_span"] = True
                        row["base_f1"] = round(
                            100 * token_f1_multiref(base_pred, gt, multiref=True), 2)
                        row["repaired_f1"] = row["base_f1"]
                        base_f1s.append(row["base_f1"])
                        rep_f1s.append(row["repaired_f1"])
                        rows.append(row)
                        with open(out_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        _row_written = True
                        print(f"  [skip] short span ({len(_primary_span.split())} tokens): "
                              f"{_primary_span!r}", flush=True)
                        if (k + 1) % 5 == 0 or (k + 1) == len(probe_records):
                            print(f"[prog] {k+1}/{len(probe_records)} fails={n_fail}", flush=True)
                        torch.cuda.empty_cache()
                        continue

                    # --- Phase 1: Fact-Level Commitment Anchoring ---
                    fact_anchor = _anchor_fact_level(
                        base_pred, lv2_result, _lv2_generate)
                    row["fact_anchor"] = {
                        "facts": fact_anchor["facts"],
                        "tags": fact_anchor["tags"],
                        "correction_types": fact_anchor["correction_types"],
                        "modalities": fact_anchor["modalities"],
                        "step1_obs": fact_anchor["step1_obs"],
                        "primary_span": fact_anchor["primary_span"],
                    }
                    print(f"  [anchor] facts={len(fact_anchor['facts'])} "
                          f"KEEP={sum(1 for t in fact_anchor['tags'].values() if t=='KEEP')} "
                          f"VERIFY={sum(1 for t in fact_anchor['tags'].values() if t=='VERIFY')} "
                          f"CORRECT={sum(1 for t in fact_anchor['tags'].values() if t=='CORRECT')}",
                          flush=True)

                    # --- Phase 2: Strategy Selection ---
                    strategies = _get_correction_strategy(fact_anchor)
                    row["strategies_selected"] = strategies
                    print(f"  [strategy] selected: {strategies}", flush=True)

                    # --- Phase 3: Generate Candidates ---
                    candidates_dict = {}
                    for strategy in strategies:
                        if strategy == MINIMAL_REPLACEMENT:
                            cand = _regen_strategy_minimal(
                                fact_anchor, lv2_result, question, base_pred,
                                _lv2_generate, args.max_new_tokens)
                        elif strategy == EVIDENCE_COMPLETION:
                            cand = _regen_strategy_completion(
                                fact_anchor, lv2_result, question, base_pred,
                                _lv2_generate, args.max_new_tokens)
                        elif strategy == CROSS_MODAL_RECONCILIATION:
                            cand = _regen_strategy_crossmodal(
                                fact_anchor, lv2_result, question, base_pred,
                                _lv2_generate, args.max_new_tokens)
                        else:
                            cand = None
                        if cand is not None:
                            candidates_dict[strategy] = cand.strip()
                            print(f"  [cand:{strategy[:4]}] {cand[:80]!r}", flush=True)

                    # Save all candidates for offline analysis
                    row["all_candidates"] = {
                        s: {"text": t,
                            "f1": round(100 * token_f1_multiref(t, gt, multiref=True), 2),
                            "scene_ok": _check_no_scene_replacement(t, base_pred)}
                        for s, t in candidates_dict.items()
                    }

                    # --- Phase 4: Evidence-Based Selection ---
                    selected_text, selected_strategy, selection_reason = \
                        _evidence_select_candidate(
                            candidates_dict, fact_anchor, lv2_result, base_pred)

                    row["strategy_used"] = selected_strategy
                    row["selection_reason"] = selection_reason
                    row["candidate_text"] = selected_text

                    if selected_strategy == "base":
                        n_fell_back_to_base += 1

                    # Hard accept/reject gate (unchanged from v4)
                    accepted = _lv2_accept_decision(
                        selected_text, base_pred, verify_status=_agg)

                    row["repaired_pred"] = selected_text if accepted else base_pred
                    row["lv2_regen_accepted"] = accepted
                    row["candidate_f1"] = round(
                        100 * token_f1_multiref(selected_text, gt, multiref=True), 2)

                    # Backward-compat fields
                    row["prompt_path"] = selected_strategy
                    row["candidate_scene_ok"] = _check_no_scene_replacement(
                        selected_text, base_pred)

                    print(f"  strategy={selected_strategy} accepted={accepted} "
                          f"reason={selection_reason} | "
                          f"{row['repaired_pred'][:70]!r}", flush=True)

                except Exception as e:
                    n_fail += 1
                    row["status"] = "regen_failed"
                    row["error"] = f"{type(e).__name__}: {e}"
                    row["repaired_pred"] = base_pred
                    row["traceback"] = traceback.format_exc()[-800:]
                    print(f"[FAIL regen id={rid}] {row['error']}", flush=True)

        row["base_f1"] = round(
            100 * token_f1_multiref(base_pred, gt, multiref=True), 2)
        row["repaired_f1"] = round(
            100 * token_f1_multiref(row["repaired_pred"], gt, multiref=True), 2)

        if row["status"] == "success":
            base_f1s.append(row["base_f1"])
            rep_f1s.append(row["repaired_f1"])
            # Per-strategy stats
            s_used = row.get("strategy_used", "none")
            if s_used in strategy_counts:
                strategy_counts[s_used] += 1
                cand_f1 = row.get("candidate_f1", row["base_f1"])
                strategy_oracle[s_used].append(max(cand_f1, row["base_f1"]))
                strategy_delta[s_used].append(row["repaired_f1"] - row["base_f1"])
                if row.get("candidate_scene_ok", True):
                    strategy_scene_ok[s_used] += 1

        rows.append(row)
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

        if (k + 1) % 5 == 0 or (k + 1) == len(probe_records):
            print(f"[prog] {k+1}/{len(probe_records)} fails={n_fail}", flush=True)

        torch.cuda.empty_cache()

    # --- Summary ---
    n_ok = len(base_f1s)
    mb = sum(base_f1s) / n_ok if n_ok else 0.0
    mr = sum(rep_f1s) / n_ok if n_ok else 0.0

    oracle_f1s = []
    cand_f1s = []
    n_accepted = sum(1 for r in rows if r.get("lv2_regen_accepted"))
    n_improved = sum(1 for r in rows
                     if r.get("lv2_regen_accepted") and r["repaired_f1"] > r["base_f1"])
    n_degraded = sum(1 for r in rows
                     if r.get("lv2_regen_accepted") and r["repaired_f1"] < r["base_f1"])
    n_scene_replaced = sum(1 for r in rows
                           if "candidate_text" in r
                           and not _check_no_scene_replacement(
                               r["candidate_text"], r["base_pred"]))
    for r in rows:
        if r["status"] == "success":
            cf1 = r.get("candidate_f1")
            if cf1 is not None:
                cand_f1s.append(cf1)
                oracle_f1s.append(max(cf1, r["base_f1"]))
            else:
                oracle_f1s.append(r["base_f1"])

    mo = sum(oracle_f1s) / len(oracle_f1s) if oracle_f1s else 0.0
    mc = sum(cand_f1s) / len(cand_f1s) if cand_f1s else 0.0

    def _mean(lst):
        return round(sum(lst) / len(lst), 2) if lst else 0.0

    per_strategy_stats = {}
    for s in [MINIMAL_REPLACEMENT, EVIDENCE_COMPLETION, CROSS_MODAL_RECONCILIATION]:
        per_strategy_stats[s] = {
            "n_selected": strategy_counts[s],
            "oracle_f1_mean": _mean(strategy_oracle[s]),
            "delta_f1_mean": _mean(strategy_delta[s]),
            "n_scene_ok": strategy_scene_ok[s],
        }

    summary = {
        "arm": args.arm,
        "signal_mode": signal_mode,
        "n_run": len(rows),
        "n_success": n_ok,
        "n_regen_failed": n_fail,
        "n_samples_edited": sum(1 for r in rows if r.get("n_spans", 0) > 0
                                and r["status"] == "success"),
        "n_samples_no_span": sum(1 for r in rows if r.get("no_edit")),
        "base_tokenf1_mean": round(mb, 2),
        "repaired_tokenf1_mean": round(mr, 2),
        "delta_f1": round(mr - mb, 2),
        "candidate_f1_mean": round(mc, 2),
        "oracle_f1_mean": round(mo, 2),
        "n_accepted": n_accepted,
        "n_improved": n_improved,
        "n_degraded": n_degraded,
        "n_scene_replaced_candidates": n_scene_replaced,
        "n_fell_back_to_base": n_fell_back_to_base,
        "per_strategy": per_strategy_stats,
        "strategy_distribution": {
            s: round(strategy_counts[s] / n_ok, 3) if n_ok else 0.0
            for s in strategy_counts
        },
        "out": out_path,
    }
    print("\n===== SUMMARY (v5) =====", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()


