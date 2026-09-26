# -*- coding: utf-8 -*-
"""
run_avh_lv2.py — AVHBench lv2 base-aware regen driver (multi-arm)

Adapts run_halo_cu121.py's lv2 pipeline to AVHBench (audio-visual, 645 items).

Key differences from HaloQuest:
  - Probe cache has delta_video + delta_audio (not delta_img_tok)
  - Shim: delta_img_tok = delta_video + delta_audio (combined AV relevance)
    per freeform_repair.load_avhbench convention; originals preserved as-is
  - Three signal variants: combined / video_only / audio_only
  - GT field is "answer" in avhbench.json (joined by numeric id)
  - question is absent in probe records; joined from avhbench.json by id

Arms:
  combined   — delta_video + delta_audio  (main, mirrors HaloQuest arm_A)
  video_only — delta_video only
  audio_only — delta_audio only
  base       — no regen (baseline; reads avhbench_base_model_full.jsonl)

Usage:
  CUDA_VISIBLE_DEVICES=2 MMHALU_MODEL_ROOT=/slow_share/tsq/MODELS HF_HUB_OFFLINE=1 \\
  PYTHONPATH=/dev/shm/site-packages \\
  LD_LIBRARY_PATH=/dev/shm/so_mirror \\
    /home/tsq/.local/bin/python3.13 run_avh_lv2.py \\
    --arm combined --limit 5 --out outputs/lv2_avh/smoke_n5.jsonl

Constraints: cards 0-3 only. No modification of existing files.
"""
import argparse
import importlib.util
import json
import os
import sys
import time
import traceback

import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
SHARED = os.path.join(REPO, "OmniConfess", "methods", "shared")

sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "OmniConfess"))

# Patch CVE-2025-32434 torch.load safety check before transformers import
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


# ---------------------------------------------------------------------------
# Copied verbatim from run_halo_cu121.py (no modification of existing files)
# ---------------------------------------------------------------------------

def _extract_facts_from_pred(base_pred, _generate_fn, max_new_tokens=48):
    """Phase-0: extract factual commitments from base_pred as FACT: lines."""
    prompt = (
        f"List the specific factual claims in this answer as FACT lines.\n"
        f"Answer: {base_pred}\n\n"
        f"Rules:\n"
        f"- Each line must start with 'FACT:'\n"
        f"- Each FACT is a short noun phrase or action phrase (3-8 words)\n"
        f"- Focus on: objects, colors, sounds, actions, scenes, attributes\n"
        f"- Do NOT paraphrase or interpret; copy the actual words used\n"
        f"- List 2-5 facts maximum\n\n"
        f"Facts:"
    )
    raw = _generate_fn(prompt, max_new_tokens)
    facts = []
    for line in raw.splitlines():
        line = line.strip()
        if line.upper().startswith("FACT:"):
            fact = line[5:].strip().rstrip(".")
            if fact:
                facts.append(fact)
    return facts


def _match_fact_to_span(facts, span_text):
    """Find the fact most closely related to span_text; return None if no match."""
    if not facts or not span_text:
        return None
    span_lower = span_text.lower().strip()
    # Exact substring match first
    for fact in facts:
        fact_lower = fact.lower()
        if fact_lower in span_lower or span_lower in fact_lower:
            return fact
    # Token overlap: any word from span appears in fact
    span_words = set(span_lower.split())
    best_fact = None
    best_overlap = 0
    for fact in facts:
        fact_words = set(fact.lower().split())
        overlap = len(span_words & fact_words)
        if overlap > best_overlap:
            best_overlap = overlap
            best_fact = fact
    # Require at least one meaningful shared word (length > 3)
    if best_overlap > 0:
        span_meaningful = {w for w in span_words if len(w) > 3}
        best_words = set(best_fact.lower().split()) if best_fact else set()
        if span_meaningful & best_words:
            return best_fact
    return None


def _regen_prompt_A(lv2_result, question, base_pred, _generate_fn,
                     max_new_tokens=96):
    """Prompt A (control): evidence-grounded sentence-level targeted fix.
    Requires step1_observation; returns None when observation is absent."""
    rev = lv2_result.revision_candidates
    if not rev:
        return None
    primary = rev[0]
    step1_obs = getattr(primary, "step1_observation", None) or ""
    if not step1_obs:
        return None
    prompt = (
        f"Question: {question}\n\n"
        f"Original Answer: {base_pred}\n\n"
        f"What the video/audio actually shows:\n{step1_obs}\n\n"
        f"Specific issue: The claim \"{primary.claim}\" is not supported.\n\n"
        f"Task: Write ONE corrected sentence.\n"
        f"Rules:\n"
        f"- Keep ALL parts of the original answer that are NOT about \"{primary.span_text}\"\n"
        f"- Only change the specific unsupported claim\n"
        f"- Do not introduce new objects, scenes, or details not in the original answer\n"
        f"- If the object is absent: replace that part with 'There is no [object]'\n"
        f"- If unsure: keep the original answer unchanged\n\n"
        f"Corrected Answer (change only the unsupported part, keep everything else):"
    )
    return _generate_fn(prompt, max_new_tokens)


def _regen_prompt_B2(lv2_result, base_pred, _generate_fn,
                      max_new_tokens=96, target_fact=None):
    """Prompt B2 (experimental): minimal fact replacement.
    Fires whenever target_fact is identified; uses step1_obs as evidence hint when present.
    Returns None when target_fact is absent."""
    rev = lv2_result.revision_candidates
    if not rev or not target_fact:
        return None
    primary = rev[0]
    step1_obs = getattr(primary, "step1_observation", None) or ""
    span_text = getattr(primary, "span_text", "") or ""
    evidence_hint = (step1_obs if step1_obs
                     else "the claim may be inaccurate based on the question context")
    prompt = (
        f"You are correcting ONE specific factual claim in an answer.\n\n"
        f"Original answer: {base_pred}\n"
        f"Claim to correct: {target_fact}  "
        f"(this appears in the answer as part of: \"{span_text}\")\n"
        f"What the evidence suggests: {evidence_hint}\n\n"
        f"Rules (strict):\n"
        f"1. Keep the entire original answer EXCEPT the claim above.\n"
        f"2. Replace ONLY the factual claim \"{target_fact}\" with a corrected version.\n"
        f"3. If you are not confident what the correct fact is, "
        f"return the original answer unchanged.\n"
        f"4. Do NOT add new objects, scenes, or entities not in the original.\n"
        f"5. Output ONLY the corrected answer, no explanation.\n\n"
        f"Corrected answer:"
    )
    return _generate_fn(prompt, max_new_tokens)


def _regen_prompt_B_fallback(lv2_result, question, base_pred, _generate_fn,
                               max_new_tokens=96):
    """Prompt B fallback: used only when neither A nor B2 can fire."""
    rev = lv2_result.revision_candidates
    if not rev:
        return None
    primary = rev[0]
    prompt = (
        f"Question: {question}\n\n"
        f"Original Answer: {base_pred}\n\n"
        f"Verification finding: The specific claim \"{primary.span_text}\" "
        f"was marked as {primary.verify_status}.\n\n"
        f"Task: Write ONE corrected sentence.\n"
        f"Critical rules:\n"
        f"- This is a TARGETED correction, not a rewrite\n"
        f"- Keep the scene, setting, and all other facts from the original answer\n"
        f"- Only address the specific claim about \"{primary.span_text}\"\n"
        f"- Do NOT replace the original scene with a different scene\n"
        f"- Do NOT add new objects or details\n"
        f"- If you cannot correct just this one part, return the original answer unchanged\n\n"
        f"Corrected Answer (minimal change only):"
    )
    return _generate_fn(prompt, max_new_tokens)


# Keep old name as shim for any external callers (not used in main loop below).
def _lv2_evidence_regen(lv2_result, question, base_pred, _generate_fn,
                         max_new_tokens=96):
    cand = _regen_prompt_A(lv2_result, question, base_pred, _generate_fn, max_new_tokens)
    if cand is None:
        cand = _regen_prompt_B_fallback(lv2_result, question, base_pred,
                                         _generate_fn, max_new_tokens)
    return cand if cand is not None else base_pred


def _check_no_scene_replacement(repaired, base_pred):
    """Returns True if repaired is safe (no wholesale scene replacement).
    Rejects candidates where less than 25% of base content words are preserved.
    """
    import re
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
    return overlap >= 0.25  # at least 25% content word overlap


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
        if verify_status and any(sig in verify_status.upper()
                                  for sig in _edit_signals):
            return True
        return False
    return True


def _lv2_accept_decision(repaired, base_pred, verify_status=""):
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


# ---------------------------------------------------------------------------
# Module loader (same pattern as run_halo_cu121.py)
# ---------------------------------------------------------------------------

def _load_module_direct(name, file_path):
    spec = importlib.util.spec_from_file_location(name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# AVHBench-specific probe loading with shim
# ---------------------------------------------------------------------------

def load_avh_probe(probe_path, avhbench_data, signal_mode="combined"):
    """
    Load avh probe records, attach shim delta_img_tok, join question+gt from
    avhbench_data.

    signal_mode:
        "combined"   — delta_img_tok = delta_video + delta_audio  (main arm)
        "video_only" — delta_img_tok = delta_video
        "audio_only" — delta_img_tok = delta_audio
    """
    # Build avhbench lookup by uid numeric suffix (avhbench_000074 -> "74")
    # probe record r["id"] is a uid number like 74, NOT a list index
    avh_by_uid = {}
    for item in avhbench_data:
        uid = item.get("uid", "")
        # uid format: "avhbench_000074" -> numeric suffix "74"
        try:
            num = str(int(uid.split("_")[-1]))
        except (ValueError, IndexError):
            continue
        avh_by_uid[num] = item

    # For each probe id keep only the FIRST occurrence (bigN has repeated runs)
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

            # gt is already stored correctly in the probe cache; trust it.
            # Join question from avhbench.json via uid numeric lookup.
            meta = avh_by_uid.get(rid, {})
            r["question"] = meta.get("question", r.get("question", ""))
            # r["gt"] is already correct from probe creation — do NOT overwrite it

            records.append(r)
    return records


# ---------------------------------------------------------------------------
# Base arm (no regen; scores avhbench_base_model_full.jsonl directly)
# ---------------------------------------------------------------------------

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
            "id": idx,
            "question": question,
            "gt": gt,
            "base_pred": pred,
            "repaired_pred": pred,
            "arm": "base",
            "n_spans": 0,
            "status": "base",
            "base_f1": f1,
            "repaired_f1": f1,
        })

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = len(rows)
    mb = sum(r["base_f1"] for r in rows) / n if n else 0.0
    summary = {
        "arm": "base",
        "n_run": n,
        "base_tokenf1_mean": round(mb, 2),
        "repaired_tokenf1_mean": round(mb, 2),
        "delta_f1": 0.0,
        "out": out_path,
    }
    print("\n===== SUMMARY (base arm) =====", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=ARMS, required=True,
                    help="Which arm to run")
    ap.add_argument("--limit", type=int, default=50,
                    help="Number of items (from item 0)")
    ap.add_argument("--out", type=str, default="",
                    help="Output .jsonl path (default: outputs/lv2_avh/<arm>_n<limit>.jsonl)")
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--probe-path", type=str, default=PROBE_PATH)
    args = ap.parse_args()

    out_path = args.out
    if not out_path:
        out_path = os.path.join(REPO, "outputs", "lv2_avh",
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

    # --- load dataset ---
    avhbench_data = json.load(open(AVH_DATA, encoding="utf-8"))
    print(f"[data] avhbench items: {len(avhbench_data)}", flush=True)

    # --- base arm: no GPU needed ---
    if args.arm == "base":
        run_base_arm(avhbench_data, BASE_PREDS_PATH, args.limit, out_path,
                     token_f1_multiref)
        return

    # --- lv2 arms: need probe cache + model ---
    signal_mode = {
        "combined": "combined",
        "video_only": "video_only",
        "audio_only": "audio_only",
    }[args.arm]

    probe_records = load_avh_probe(args.probe_path, avhbench_data, signal_mode)
    print(f"[probe] loaded {len(probe_records)} unique records "
          f"(signal_mode={signal_mode})", flush=True)

    # Limit to first N items (items 0..limit-1 by probe id order)
    probe_records = [r for r in probe_records if int(r["id"]) < args.limit]
    probe_records.sort(key=lambda r: int(r["id"]))
    print(f"[probe] after limit: {len(probe_records)} records", flush=True)

    if not probe_records:
        print("[ERROR] No probe records in range — check probe path and limit", flush=True)
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
        print("[import] patched load_speakers (CVE-2025-32434 bypass)", flush=True)
    except Exception as _pe:
        print(f"[import] load_speakers patch failed: {_pe}", flush=True)

    _load_module_direct("methods.shared.paths",
                        os.path.join(SHARED, "paths.py"))
    _mu = _load_module_direct("methods.shared.model_utils",
                              os.path.join(SHARED, "model_utils.py"))
    load_model = _mu.load_model
    # Force CUDA context init before load_model so device_map="auto" sees the GPU.
    # Without this, device_map="auto" places the model on CPU (0 GPU MiB visible),
    # causing processor() calls to hang on large video+audio tensors.
    import torch as _torch_init
    _dummy = _torch_init.zeros(1, device="cuda")
    del _dummy
    print(f"[cuda] context init OK  device=cuda:{_torch_init.cuda.current_device()}", flush=True)

    # Prefer /dev/shm copy of model weights to avoid NFS mmap stall
    # (rpc_wait_bit_killable on /slow_share). Fall back to model_utils default.
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

    # Shared model utilities needed for the generate closure
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

    # Load avhbench items via shared data_loader (normalized schema)
    avh_items = load_dataset_fn(AVH_DATA, AVH_DATA_ROOT)
    # data_loader assigns list-index as 'id' for avhbench items (no numeric 'id' field
    # in avhbench.json). Probe record ids are uid numerics like 74, 1, 79 — extracted
    # from the uid string "avhbench_000074". Build both lookups so we can resolve either.
    avh_by_id_idx = {str(it["id"]): it for it in avh_items}       # list-index keyed
    avh_by_uid_num = {}                                            # uid-numeric keyed
    for it in avh_items:
        uid = it.get("uid", "")
        try:
            num = str(int(uid.split("_")[-1]))
            avh_by_uid_num[num] = it
        except (ValueError, IndexError):
            pass
    # Prefer uid-numeric lookup (matches probe record ids); fall back to list-index
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
        gt = rec.get("gt", "")   # probe already has gt
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
        }

        lv2_result = None
        if _LOCALIZER_AVAILABLE:
            cache_record = {
                "id": rec["id"],
                "tokens": rec["tokens"],
                "delta_img_tok": rec["delta_img_tok"],   # shim already applied
                "delta_video": rec.get("delta_video", []),
                "delta_audio": rec.get("delta_audio", []),
                "delta_text_tok": rec.get("delta_text", []),
                "classes": rec.get("classes"),
                "question": question,
                "lp_full": rec.get("lp_full", []),
            }
            lv2_result = lv2.localize(cache_record)

            # Build generate closure for this item
            try:
                _pv, _av, _iv = load_media_for_item(it, AVH_DATA, AVH_DATA_ROOT)
                _gen_state = {"pv": _pv, "av": _av, "iv": _iv, "sys": sys_prompt}

                def _lv2_generate(prompt_text, max_new_tokens, _gs=_gen_state):
                    uc = build_user_content_with_media_tokens(
                        _gs["pv"], _gs["av"], _gs["iv"], prompt_text)
                    msgs = make_messages(_gs["sys"], uc)
                    inp = build_inputs(processor, model, msgs,
                                       _gs["pv"], _gs["av"], _gs["iv"],
                                       "haloquest")  # triggers max_pixels=224*224
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

                # Steps 4-5: verify candidates
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

        # Regen
        if row["status"] == "success":
            rev = lv2_result.revision_candidates if lv2_result else []
            if not rev or _lv2_generate is None:
                row["repaired_pred"] = base_pred
                row["lv2_regen_accepted"] = False
                if not rev:
                    row["no_edit"] = True
            else:
                try:
                    _primary = rev[0]
                    _step1_obs = getattr(_primary, "step1_observation", None) or ""
                    _span_text = getattr(_primary, "span_text", "") or ""
                    _statuses = [c.verify_status for c in rev]
                    _agg = "UNSUPPORTED" if any(
                        "UNSUPPORTED" in vs for vs in _statuses
                    ) else (_statuses[0] if _statuses else "")

                    # --- Phase-0 fact extraction (always attempt) ---
                    try:
                        _facts = _extract_facts_from_pred(
                            base_pred, _lv2_generate, max_new_tokens=48)
                        _tgt_fact = _match_fact_to_span(_facts, _span_text)
                    except Exception:
                        _facts = []
                        _tgt_fact = None
                    row["phase0_facts"] = _facts
                    row["phase0_target_fact"] = _tgt_fact

                    # --- Generate Prompt A candidate (when step1_obs present) ---
                    cand_A = _regen_prompt_A(
                        lv2_result, question, base_pred,
                        _lv2_generate, args.max_new_tokens)

                    # --- Generate Prompt B2 candidate (when target_fact found) ---
                    cand_B2 = _regen_prompt_B2(
                        lv2_result, base_pred,
                        _lv2_generate, args.max_new_tokens,
                        target_fact=_tgt_fact)

                    # --- Decide which path fired and pick the candidate ---
                    if cand_A is not None and cand_B2 is not None:
                        _prompt_path = "A+B2"  # both fired — record both, pick A as primary
                        repaired = cand_A
                    elif cand_B2 is not None:
                        _prompt_path = "B2"
                        repaired = cand_B2
                    elif cand_A is not None:
                        _prompt_path = "A"
                        repaired = cand_A
                    else:
                        _prompt_path = "B_fallback"
                        repaired = _regen_prompt_B_fallback(
                            lv2_result, question, base_pred,
                            _lv2_generate, args.max_new_tokens) or base_pred

                    # --- Evaluate and record ALL candidates for per-path comparison ---
                    accepted = _lv2_accept_decision(repaired, base_pred,
                                                    verify_status=_agg)
                    row["candidate_text"] = repaired
                    row["candidate_f1"] = round(
                        100 * token_f1_multiref(repaired, gt, multiref=True), 2)

                    # Store per-path candidates for offline analysis
                    if cand_A is not None:
                        row["candidate_A"] = cand_A
                        row["candidate_A_f1"] = round(
                            100 * token_f1_multiref(cand_A, gt, multiref=True), 2)
                        row["candidate_A_accepted"] = _lv2_accept_decision(
                            cand_A, base_pred, verify_status=_agg)
                        row["candidate_A_scene_ok"] = _check_no_scene_replacement(
                            cand_A, base_pred)
                    if cand_B2 is not None:
                        row["candidate_B2"] = cand_B2
                        row["candidate_B2_f1"] = round(
                            100 * token_f1_multiref(cand_B2, gt, multiref=True), 2)
                        row["candidate_B2_accepted"] = _lv2_accept_decision(
                            cand_B2, base_pred, verify_status=_agg)
                        row["candidate_B2_scene_ok"] = _check_no_scene_replacement(
                            cand_B2, base_pred)

                    row["repaired_pred"] = repaired if accepted else base_pred
                    row["lv2_regen_accepted"] = accepted
                    row["prompt_path"] = _prompt_path
                    print(f"  prompt={_prompt_path} accepted={accepted} | "
                          f"A={'Y' if cand_A else 'N'} "
                          f"B2={'Y' if cand_B2 else 'N'} | "
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

        rows.append(row)
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

        if (k + 1) % 5 == 0 or (k + 1) == len(probe_records):
            print(f"[prog] {k+1}/{len(probe_records)} fails={n_fail}", flush=True)

        torch.cuda.empty_cache()

    n_ok = len(base_f1s)
    mb = sum(base_f1s) / n_ok if n_ok else 0.0
    mr = sum(rep_f1s) / n_ok if n_ok else 0.0
    n_edited = sum(1 for r in rows if r.get("n_spans", 0) > 0
                   and r["status"] == "success")

    # Candidate Oracle F1: max(candidate_f1, base_f1) per row — upper bound
    oracle_f1s = []
    cand_f1s = []
    n_accepted = sum(1 for r in rows if r.get("lv2_regen_accepted"))
    n_improved = sum(1 for r in rows
                     if r.get("lv2_regen_accepted")
                     and r["repaired_f1"] > r["base_f1"])
    n_degraded = sum(1 for r in rows
                     if r.get("lv2_regen_accepted")
                     and r["repaired_f1"] < r["base_f1"])
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

    # Prompt path counts (A+B2 = both fired simultaneously)
    n_path_A        = sum(1 for r in rows if r.get("prompt_path") == "A")
    n_path_B2       = sum(1 for r in rows if r.get("prompt_path") == "B2")
    n_path_AB2      = sum(1 for r in rows if r.get("prompt_path") == "A+B2")
    n_path_B_fallback = sum(1 for r in rows if r.get("prompt_path") == "B_fallback")
    n_with_path = n_path_A + n_path_B2 + n_path_AB2 + n_path_B_fallback
    # Rows where B2 candidate exists (path==B2 or path==A+B2)
    n_B2_fired = n_path_B2 + n_path_AB2

    # Per-path candidate oracle F1 for offline comparison
    def _path_stats(rows_with_key, f1_key, base_key="base_f1", scene_key=None):
        f1s = [r[f1_key] for r in rows_with_key if f1_key in r]
        base = [r[base_key] for r in rows_with_key if f1_key in r]
        oracle = [max(f, b) for f, b in zip(f1s, base)]
        improved = sum(1 for f, b in zip(f1s, base) if f > b)
        degraded = sum(1 for f, b in zip(f1s, base) if f < b)
        accepted = sum(1 for r in rows_with_key if r.get(f1_key.replace("_f1","_accepted")))
        scene_bad = 0
        if scene_key:
            scene_bad = sum(1 for r in rows_with_key
                            if scene_key in r and not r[scene_key])
        return {
            "n": len(f1s),
            "cand_f1_mean": round(sum(f1s)/len(f1s), 2) if f1s else 0.0,
            "oracle_f1_mean": round(sum(oracle)/len(oracle), 2) if oracle else 0.0,
            "n_improved": improved,
            "n_degraded": degraded,
            "n_accepted": accepted,
            "n_scene_replaced": scene_bad,
        }

    rows_has_A  = [r for r in rows if "candidate_A"  in r]
    rows_has_B2 = [r for r in rows if "candidate_B2" in r]
    stats_A  = _path_stats(rows_has_A,  "candidate_A_f1",  scene_key="candidate_A_scene_ok")
    stats_B2 = _path_stats(rows_has_B2, "candidate_B2_f1", scene_key="candidate_B2_scene_ok")

    mo = sum(oracle_f1s) / len(oracle_f1s) if oracle_f1s else 0.0
    mc = sum(cand_f1s) / len(cand_f1s) if cand_f1s else 0.0

    summary = {
        "arm": args.arm,
        "signal_mode": signal_mode,
        "n_run": len(rows),
        "n_success": n_ok,
        "n_regen_failed": n_fail,
        "n_samples_edited": n_edited,
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
        "prompt_path_A": n_path_A,
        "prompt_path_B2": n_path_B2,
        "prompt_path_A+B2": n_path_AB2,
        "prompt_path_B_fallback": n_path_B_fallback,
        "prompt_B2_fired": n_B2_fired,
        "prompt_B2_hit_rate": round(n_B2_fired / n_with_path, 3) if n_with_path else 0.0,
        "per_path_A":  stats_A,
        "per_path_B2": stats_B2,
        "out": out_path,
    }
    print("\n===== SUMMARY =====", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
