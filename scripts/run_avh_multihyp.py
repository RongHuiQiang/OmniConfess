# -*- coding: utf-8 -*-
"""
scripts/run_avh_multihyp.py — Multi-hypothesis temperature sampling for AVHBench

Hypothesis: temperature=0.7 sampling produces candidates with higher Token-F1
than greedy (temp=0) candidates, i.e. candidate_oracle_f1_mean > 19.58.

Setup:
  - signal_mode = audio_only (same as the hypothesis baseline)
  - limit = 50
  - 3 candidates per sample with temperature=0.7, top_p=0.9
  - Candidate selection: scene preservation + coherence + evidence alignment (NOT Token-F1)
  - Output: outputs/lv2_avh_multihyp/audio_only_temp07_n50.jsonl

Usage:
  CUDA_VISIBLE_DEVICES=3 MMHALU_MODEL_ROOT=/tmp HF_HUB_OFFLINE=1 \\
  TRANSFORMERS_OFFLINE=1 \\
  PYTHONPATH=/dev/shm/site-packages \\
  LD_LIBRARY_PATH=/dev/shm/so_mirror \\
    /home/tsq/.local/bin/python3.13 scripts/run_avh_multihyp.py
"""
import importlib.util
import json
import os
import re
import sys
import time
import traceback

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHARED = os.path.join(REPO, "OmniConfess", "methods", "shared")

sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "OmniConfess"))

# Patch CVE-2025-32434 torch.load safety check before transformers import
try:
    import transformers.utils.import_utils as _tiu_early
    _tiu_early.check_torch_load_is_safe = lambda: None
except Exception:
    pass

AVH_DATA      = os.path.join(REPO, "OmniHalluBench", "avhbench", "avhbench.json")
AVH_DATA_ROOT = os.path.join(REPO, "OmniHalluBench", "avhbench")
PROBE_PATH    = os.path.join(REPO, "outputs", "cf_probe",
                              "avh_probe_bigN_shard0of1.jsonl")
OUT_DIR       = os.path.join(REPO, "outputs", "lv2_avh_multihyp")
OUT_PATH      = os.path.join(OUT_DIR, "audio_only_temp07_n50.jsonl")
SUMMARY_PATH  = os.path.join(OUT_DIR, "SUMMARY_multihyp.md")

SIGNAL_MODE   = "audio_only"
LIMIT         = 50
N_CANDIDATES  = 3
TEMPERATURE   = 0.7
TOP_P         = 0.9
MAX_NEW_TOKENS = 96

# ---------------------------------------------------------------------------
# Import helper
# ---------------------------------------------------------------------------

def _load_module_direct(name, file_path):
    spec = importlib.util.spec_from_file_location(name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Inline copies of _check_no_scene_replacement and _lv2_accept_decision
# (copied verbatim from run_avh_lv2.py to avoid import side-effects)
# ---------------------------------------------------------------------------

def _content_words(text):
    stop = {
        'a','an','the','is','are','was','were','in','on','at','of','to','and','or',
        'not','no','there','this','that','it','its','with','for','by','as','be','have',
        'has','had','do','does','did','but','from','into','out','up','down','can','could',
        'will','would','should','may','might','what','which','who','how','when','where'
    }
    words = set(re.findall(r'\b[a-z]+\b', text.lower()))
    return words - stop


def _check_no_scene_replacement(repaired, base_pred):
    """Returns True if repaired is safe (no wholesale scene replacement).
    Rejects candidates where less than 25% of base content words are preserved.
    """
    base_content = _content_words(base_pred)
    rep_content  = _content_words(repaired)
    if not base_content:
        return True
    overlap = len(base_content & rep_content) / len(base_content)
    return overlap >= 0.25


def _check_preserves_positive_content(repaired, base, verify_status=""):
    base_lower     = base.lower()
    repaired_lower = repaired.lower()
    _refusal_words = {
        "not visible", "cannot", "no ", "there is no",
        "not present", "not shown", "unable", "unknown",
        "unclear", "can't", "don't know", "do not know"
    }
    base_is_positive   = not any(w in base_lower for w in _refusal_words)
    repaired_is_refusal = any(w in repaired_lower for w in _refusal_words)
    if base_is_positive and repaired_is_refusal:
        _edit_signals = {"UNSUPPORTED", "CONTRADICTED", "UNSUPPORTED_BY_EVIDENCE",
                         "INSUFFICIENT_TO_DETERMINE"}
        if verify_status and any(sig in verify_status.upper() for sig in _edit_signals):
            return True
        return False
    return True


def _lv2_accept_decision(repaired, base_pred, verify_status=""):
    rep  = repaired.strip()
    base = base_pred.strip()
    if not rep:
        return False
    if not _check_no_scene_replacement(rep, base):
        return False
    words = rep.split()
    if len(words) >= 6:
        trigrams = [tuple(words[i:i+3]) for i in range(len(words) - 2)]
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
    rep_lower  = rep.lower()
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
# Coherence heuristic
# ---------------------------------------------------------------------------

def _check_coherence(candidate, base_pred):
    """Length ratio check: 0.5 < len(cand)/len(base) < 2.0"""
    if not base_pred:
        return True
    ratio = len(candidate) / len(base_pred)
    return 0.5 < ratio < 2.0


# ---------------------------------------------------------------------------
# Evidence alignment score (word overlap with step1_obs, NOT Token-F1)
# ---------------------------------------------------------------------------

def _evidence_alignment_score(candidate, step1_obs):
    """Count content-word overlap between candidate and step1_obs."""
    cand_words = _content_words(candidate)
    obs_words  = _content_words(step1_obs)
    if not obs_words:
        return 0
    return len(cand_words & obs_words)


# ---------------------------------------------------------------------------
# Probe loading (audio_only shim — mirrors run_avh_lv2.py exactly)
# ---------------------------------------------------------------------------

def load_avh_probe_audio_only(probe_path, avhbench_data):
    avh_by_uid = {}
    for item in avhbench_data:
        uid = item.get("uid", "")
        try:
            num = str(int(uid.split("_")[-1]))
        except (ValueError, IndexError):
            continue
        avh_by_uid[num] = item

    seen_ids = set()
    records  = []
    with open(probe_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r   = json.loads(line)
            rid = str(r["id"])
            if rid in seen_ids:
                continue
            seen_ids.add(rid)

            d_a = np.asarray(r["delta_audio"], dtype=np.float64)
            r["delta_img_tok"] = list(d_a)  # audio_only shim

            meta          = avh_by_uid.get(rid, {})
            r["question"] = meta.get("question", r.get("question", ""))

            records.append(r)
    return records


# ---------------------------------------------------------------------------
# Prompt A builder (evidence-grounded, same as run_avh_lv2.py)
# ---------------------------------------------------------------------------

def _build_prompt_A(lv2_result, question, base_pred):
    """Returns the Prompt A text, or None if step1_obs is absent."""
    rev = lv2_result.revision_candidates
    if not rev:
        return None
    primary  = rev[0]
    step1_obs = getattr(primary, "step1_observation", None) or ""
    if not step1_obs:
        return None
    return (
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # --- imports ---
    print("[import] freeform_repair...", flush=True)
    import freeform_repair as fr  # noqa: F401 (side-effects may be needed)
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

    # --- load probe ---
    probe_records = load_avh_probe_audio_only(PROBE_PATH, avhbench_data)
    print(f"[probe] loaded {len(probe_records)} unique records (signal_mode={SIGNAL_MODE})",
          flush=True)

    probe_records = [r for r in probe_records if int(r["id"]) < LIMIT]
    probe_records.sort(key=lambda r: int(r["id"]))
    print(f"[probe] after limit={LIMIT}: {len(probe_records)} records", flush=True)

    if not probe_records:
        print("[ERROR] No probe records — check probe path and limit.", flush=True)
        sys.exit(1)

    # --- torch / model setup (mirrors run_avh_lv2.py exactly) ---
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

    # Force CUDA context init so device_map="auto" sees the GPU
    import torch as _torch_init
    _dummy = _torch_init.zeros(1, device="cuda")
    del _dummy
    print(f"[cuda] context init OK  device=cuda:{_torch_init.cuda.current_device()}",
          flush=True)

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
    load_dataset_fn   = _dl.load_dataset
    load_media_for_item = _dl.load_media_for_item
    from methods.shared.prompts import get_system_prompt

    # Build data_loader item lookup (uid-numeric keyed, matches probe ids)
    avh_items = load_dataset_fn(AVH_DATA, AVH_DATA_ROOT)
    avh_by_uid_num = {}
    for it in avh_items:
        uid = it.get("uid", "")
        try:
            num = str(int(uid.split("_")[-1]))
            avh_by_uid_num[num] = it
        except (ValueError, IndexError):
            pass
    avh_by_id_idx = {str(it["id"]): it for it in avh_items}
    avh_by_id = avh_by_uid_num if avh_by_uid_num else avh_by_id_idx
    print(f"[data] loaded {len(avh_items)} items via data_loader "
          f"(uid_num keys={len(avh_by_uid_num)})", flush=True)

    # Resume support
    done_ids = set()
    if os.path.exists(OUT_PATH):
        for line in open(OUT_PATH, encoding="utf-8"):
            line = line.strip()
            if line:
                try:
                    done_ids.add(str(json.loads(line)["id"]))
                except Exception:
                    pass
        print(f"[resume] {len(done_ids)} already done", flush=True)

    sys_prompt = get_system_prompt("avhbench", "v1")

    rows   = []
    n_fail = 0

    for k, rec in enumerate(probe_records):
        rid = str(rec["id"])
        if rid in done_ids:
            continue

        it = avh_by_id.get(rid)
        if it is None:
            print(f"[SKIP] id={rid} not in data_loader output", flush=True)
            continue

        base_pred = "".join(rec["tokens"])
        gt        = rec.get("gt", "")
        question  = rec.get("question", it.get("question", ""))

        print(f"\n[{k+1}/{len(probe_records)}] id={rid} "
              f"q={question[:60]!r}", flush=True)

        row = {
            "id":        int(rid),
            "question":  question,
            "gt":        gt,
            "base_pred": base_pred,
            "signal_mode": SIGNAL_MODE,
        }

        # --- lv2 localization ---
        lv2_result     = None
        _lv2_generate  = None
        step1_obs      = ""
        verify_status  = ""

        if _LOCALIZER_AVAILABLE:
            cache_record = {
                "id":             rec["id"],
                "tokens":         rec["tokens"],
                "delta_img_tok":  rec["delta_img_tok"],
                "delta_video":    rec.get("delta_video", []),
                "delta_audio":    rec.get("delta_audio", []),
                "delta_text_tok": rec.get("delta_text", []),
                "classes":        rec.get("classes"),
                "question":       question,
                "lp_full":        rec.get("lp_full", []),
            }
            lv2_result = lv2.localize(cache_record)

            try:
                _pv, _av, _iv = load_media_for_item(it, AVH_DATA, AVH_DATA_ROOT)
                _gen_state = {"pv": _pv, "av": _av, "iv": _iv, "sys": sys_prompt}

                # Greedy generate closure (for evidence_verifier)
                def _lv2_generate_greedy(prompt_text, max_new_tokens,
                                         _gs=_gen_state):
                    uc   = build_user_content_with_media_tokens(
                        _gs["pv"], _gs["av"], _gs["iv"], prompt_text)
                    msgs = make_messages(_gs["sys"], uc)
                    inp  = build_inputs(processor, model, msgs,
                                        _gs["pv"], _gs["av"], _gs["iv"],
                                        "haloquest")
                    inp  = prepare_for_thinker(inp, model)
                    with torch.inference_mode():
                        oids = model.thinker.generate(
                            **inp,
                            max_new_tokens=max_new_tokens,
                            do_sample=False,
                        )
                    return decode_outputs(
                        processor, oids, inp["input_ids"].shape[1]
                    )[0].strip()

                _lv2_generate = _lv2_generate_greedy

                # Sampled generate closure (temperature=0.7)
                def _lv2_generate_sampled(prompt_text, max_new_tokens,
                                          _gs=_gen_state):
                    uc   = build_user_content_with_media_tokens(
                        _gs["pv"], _gs["av"], _gs["iv"], prompt_text)
                    msgs = make_messages(_gs["sys"], uc)
                    inp  = build_inputs(processor, model, msgs,
                                        _gs["pv"], _gs["av"], _gs["iv"],
                                        "haloquest")
                    inp  = prepare_for_thinker(inp, model)
                    with torch.inference_mode():
                        oids = model.thinker.generate(
                            **inp,
                            max_new_tokens=max_new_tokens,
                            do_sample=True,
                            temperature=TEMPERATURE,
                            top_p=TOP_P,
                        )
                    return decode_outputs(
                        processor, oids, inp["input_ids"].shape[1]
                    )[0].strip()

                # Run evidence_verifier with greedy closure (same as run_avh_lv2.py)
                lv2_result = ev.verify_candidates(lv2_result, it, _lv2_generate)

            except Exception as e:
                n_fail += 1
                row["status"]      = "media_load_failed"
                row["error"]       = f"{type(e).__name__}: {e}"
                row["traceback"]   = traceback.format_exc()[-800:]
                row["repaired_pred"] = base_pred
                row["repaired_f1"] = round(
                    100 * token_f1_multiref(base_pred, gt, multiref=True), 2)
                row["base_f1"]     = row["repaired_f1"]
                print(f"[FAIL media id={rid}] {row['error']}", flush=True)
                rows.append(row)
                with open(OUT_PATH, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                torch.cuda.empty_cache()
                continue

            if lv2_result is not None:
                rev = lv2_result.revision_candidates
                if rev:
                    primary      = rev[0]
                    step1_obs    = getattr(primary, "step1_observation", None) or ""
                    statuses     = [c.verify_status for c in rev]
                    verify_status = ("UNSUPPORTED" if any(
                        "UNSUPPORTED" in vs for vs in statuses
                    ) else (statuses[0] if statuses else ""))
                row["lv2_n_candidates"] = len(lv2_result.candidates)
                row["lv2_n_revision"]   = len(lv2_result.revision_candidates)
                row["n_spans"]          = len(lv2_result.revision_candidates)

        # --- Multi-hypothesis generation ---
        has_revision = (
            lv2_result is not None
            and lv2_result.revision_candidates
            and _lv2_generate_sampled is not None
        )

        if not has_revision:
            # No span to fix: keep base
            row["no_edit"]           = True
            row["selected_candidate_idx"] = -1
            row["selection_reason"]  = "fallback_base"
            row["repaired_pred"]     = base_pred
            row["lv2_regen_accepted"] = False
        else:
            prompt_text = _build_prompt_A(lv2_result, question, base_pred)
            if prompt_text is None:
                # Prompt A requires step1_obs; fall back to base
                row["selected_candidate_idx"] = -1
                row["selection_reason"]  = "fallback_base"
                row["repaired_pred"]     = base_pred
                row["lv2_regen_accepted"] = False
            else:
                # Generate N_CANDIDATES with temperature sampling
                candidates = []
                for ci in range(N_CANDIDATES):
                    try:
                        cand = _lv2_generate_sampled(prompt_text, MAX_NEW_TOKENS)
                    except Exception as eg:
                        print(f"  [gen error cand={ci}] {eg}", flush=True)
                        cand = base_pred
                    candidates.append(cand)
                    print(f"  cand[{ci}]: {cand[:70]!r}", flush=True)

                # Score each candidate
                cand_scene_ok  = []
                cand_coherence = []
                cand_f1_list   = []
                for ci, cand in enumerate(candidates):
                    sc = _check_no_scene_replacement(cand, base_pred)
                    co = _check_coherence(cand, base_pred)
                    cf = round(100 * token_f1_multiref(cand, gt, multiref=True), 2)
                    cand_scene_ok.append(sc)
                    cand_coherence.append(co)
                    cand_f1_list.append(cf)
                    print(f"  cand[{ci}] scene_ok={sc} coherence={co} f1={cf:.2f}",
                          flush=True)

                # Record all candidates
                row["candidates_text"]     = candidates
                row["cand_scene_ok"]       = cand_scene_ok
                row["cand_coherence"]      = cand_coherence
                row["cand_f1"]             = cand_f1_list
                row["candidate_oracle_f1"] = max(cand_f1_list)

                # Select: among passing candidates, pick highest evidence alignment
                passing_indices = [
                    ci for ci in range(N_CANDIDATES)
                    if cand_scene_ok[ci] and cand_coherence[ci]
                ]

                if passing_indices:
                    best_idx = max(
                        passing_indices,
                        key=lambda ci: _evidence_alignment_score(
                            candidates[ci], step1_obs)
                    )
                    selected_cand    = candidates[best_idx]
                    selection_reason = "evidence_aligned"
                    row["selected_candidate_idx"] = best_idx
                else:
                    # All candidates failed checks — check if any pass scene only
                    scene_ok_indices = [ci for ci in range(N_CANDIDATES)
                                        if cand_scene_ok[ci]]
                    if scene_ok_indices:
                        best_idx = max(
                            scene_ok_indices,
                            key=lambda ci: _evidence_alignment_score(
                                candidates[ci], step1_obs)
                        )
                        selected_cand    = candidates[best_idx]
                        selection_reason = "fallback_scene"
                        row["selected_candidate_idx"] = best_idx
                    else:
                        selected_cand    = base_pred
                        selection_reason = "fallback_base"
                        row["selected_candidate_idx"] = -1

                row["selection_reason"] = selection_reason

                # Apply accept decision
                accepted = _lv2_accept_decision(
                    selected_cand, base_pred, verify_status=verify_status)
                row["lv2_regen_accepted"] = accepted
                row["repaired_pred"]      = selected_cand if accepted else base_pred
                print(f"  selected={row['selected_candidate_idx']} "
                      f"reason={selection_reason} accepted={accepted}", flush=True)

        row["base_f1"] = round(
            100 * token_f1_multiref(base_pred, gt, multiref=True), 2)
        row["repaired_f1"] = round(
            100 * token_f1_multiref(row["repaired_pred"], gt, multiref=True), 2)

        rows.append(row)
        with open(OUT_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

        if (k + 1) % 5 == 0 or (k + 1) == len(probe_records):
            print(f"[prog] {k+1}/{len(probe_records)} fails={n_fail}", flush=True)

        torch.cuda.empty_cache()

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    success_rows = [r for r in rows if "base_f1" in r and "repaired_f1" in r]
    n = len(success_rows)

    base_f1_mean    = sum(r["base_f1"]    for r in success_rows) / n if n else 0.0
    repaired_f1_mean = sum(r["repaired_f1"] for r in success_rows) / n if n else 0.0
    delta_f1        = repaired_f1_mean - base_f1_mean

    has_oracle = [r for r in success_rows if "candidate_oracle_f1" in r]
    oracle_f1_mean = (
        sum(r["candidate_oracle_f1"] for r in has_oracle) / len(has_oracle)
        if has_oracle else 0.0
    )

    n_accepted      = sum(1 for r in rows if r.get("lv2_regen_accepted"))
    n_no_edit       = sum(1 for r in rows if r.get("no_edit"))
    n_fallback_base = sum(1 for r in rows if r.get("selection_reason") == "fallback_base")
    n_ev_aligned    = sum(1 for r in rows if r.get("selection_reason") == "evidence_aligned")
    n_fb_scene      = sum(1 for r in rows if r.get("selection_reason") == "fallback_scene")

    # Per-candidate scene_ok counts
    per_cand_scene = [0] * N_CANDIDATES
    per_cand_n     = [0] * N_CANDIDATES
    for r in rows:
        so = r.get("cand_scene_ok", [])
        for ci in range(min(N_CANDIDATES, len(so))):
            per_cand_n[ci]     += 1
            per_cand_scene[ci] += int(so[ci])

    summary = {
        "n_run":                  len(rows),
        "n_success":              n,
        "n_fail":                 n_fail,
        "n_no_edit":              n_no_edit,
        "signal_mode":            SIGNAL_MODE,
        "temperature":            TEMPERATURE,
        "top_p":                  TOP_P,
        "n_candidates":           N_CANDIDATES,
        "base_f1_mean":           round(base_f1_mean, 2),
        "repaired_f1_mean":       round(repaired_f1_mean, 2),
        "delta_f1":               round(delta_f1, 2),
        "candidate_oracle_f1_mean": round(oracle_f1_mean, 2),
        "v4_baseline_oracle":     19.58,
        "oracle_beats_v4":        oracle_f1_mean > 19.58,
        "n_accepted":             n_accepted,
        "n_selection_evidence_aligned": n_ev_aligned,
        "n_selection_fallback_scene":   n_fb_scene,
        "n_selection_fallback_base":    n_fallback_base,
        "per_candidate_scene_ok": {
            f"cand_{ci}": f"{per_cand_scene[ci]}/{per_cand_n[ci]}"
            for ci in range(N_CANDIDATES)
        },
        "out": OUT_PATH,
    }

    print("\n===== SUMMARY =====", flush=True)
    print(json.dumps(summary, indent=2), flush=True)

    # Count how often B-type candidates beat A-type (here: how often any non-cand0 wins)
    n_noncand0_wins = sum(
        1 for r in rows
        if r.get("selected_candidate_idx", 0) > 0
    )

    # Write markdown summary
    md = f"""# MultiHyp temp=0.7 Summary

## Configuration
- signal_mode: {SIGNAL_MODE}
- temperature: {TEMPERATURE}, top_p: {TOP_P}
- n_candidates: {N_CANDIDATES}
- limit: {LIMIT}
- output: {OUT_PATH}

## Key Results
| Metric | Value |
|---|---|
| n_run | {len(rows)} |
| base_f1_mean | {base_f1_mean:.2f} |
| repaired_f1_mean | {repaired_f1_mean:.2f} |
| delta_f1 | {delta_f1:+.2f} |
| candidate_oracle_f1_mean | {oracle_f1_mean:.2f} |
| v4_baseline_oracle | 19.58 |
| oracle_beats_v4 | {oracle_f1_mean > 19.58} |
| n_accepted | {n_accepted} |
| n_no_edit (no span) | {n_no_edit} |

## Selection Breakdown
| Reason | Count |
|---|---|
| evidence_aligned | {n_ev_aligned} |
| fallback_scene | {n_fb_scene} |
| fallback_base | {n_fallback_base} |

## Per-Candidate Scene OK
{chr(10).join(f'- cand_{ci}: {per_cand_scene[ci]}/{per_cand_n[ci]}' for ci in range(N_CANDIDATES))}

## Hypothesis Answer
Does temp=0.7 sampling produce candidates with higher oracle Token-F1 than greedy (v4 oracle=19.58)?

**candidate_oracle_f1_mean = {oracle_f1_mean:.2f}** vs v4_baseline_oracle = 19.58

{"**YES** — diversity in generation helps; oracle exceeds v4 baseline." if oracle_f1_mean > 19.58 else "**NO** — diversity in generation does NOT help; oracle does not exceed v4 baseline. Bottleneck is not diversity."}

Non-cand0 wins (cand 1 or 2 selected over cand 0): {n_noncand0_wins}
"""

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"[summary] written to {SUMMARY_PATH}", flush=True)

    # Send feishu notification
    oracle_str = f"{oracle_f1_mean:.2f}"
    delta_str  = f"{delta_f1:+.2f}"
    feishu_msg = (
        f"MultiHyp temp=0.7 完成: oracle={oracle_str} (v4基线19.58), "
        f"delta={delta_str}, 3候选中非cand0赢共{n_noncand0_wins}次"
    )
    os.system(f'/home/tsq/.claude/feishu/feishu_send.sh "{feishu_msg}"')


if __name__ == "__main__":
    main()
