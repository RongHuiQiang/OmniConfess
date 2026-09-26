# -*- coding: utf-8 -*-
"""
timing_benchmark.py — per-sample inference timing for OmniConfess pipeline

Measures wall-clock time for each stage of the Champion arm:
  T0: pipeline start
  T1: Anchoring done  (lv2.localize — CPU, reads probe cache)
  T2: Interrogation done (ev.verify_candidates — GPU, or skipped)
  T3: Correction done (_lv2_evidence_regen — GPU)
  T4: Accept decision done (_lv2_accept_decision — CPU)

torch.cuda.synchronize() is called at every boundary.

Usage (HaloQuest, Champion arm, n=20):
  CUDA_VISIBLE_DEVICES=0 MMHALU_MODEL_ROOT=/tmp HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  PYTHONPATH=/tmp/pip-target-b9a06nc9/lib/python \
  LD_LIBRARY_PATH=/dev/shm/so_mirror:/tmp/pip-target-b9a06nc9/lib/python/torch/lib \
    /slow_share/tsq/venvs/omniconfess/bin/python3 \
      scripts/timing_benchmark.py --dataset haloquest --n 20 \
        --skip-verify

Usage (AVHBench, combined arm, n=20):
  ... python3 scripts/timing_benchmark.py --dataset avhbench --n 20
"""
import argparse
import importlib.util
import json
import os
import sys
import time
import traceback

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHARED = os.path.join(REPO, "OmniConfess", "methods", "shared")

sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "OmniConfess"))

# Patch CVE-2025-32434 early
try:
    import transformers.utils.import_utils as _tiu_early
    _tiu_early.check_torch_load_is_safe = lambda: None
except Exception:
    pass

HALO_DATA_PATH = os.path.join(REPO, "OmniHalluBench", "haloquest", "haloquest.json")
HALO_DATA_ROOT = os.path.join(REPO, "OmniHalluBench", "haloquest")
AVH_DATA_PATH  = os.path.join(REPO, "OmniHalluBench", "avhbench", "avhbench.json")
AVH_DATA_ROOT  = os.path.join(REPO, "OmniHalluBench", "avhbench")
AVH_PROBE_PATH = os.path.join(REPO, "outputs", "cf_probe",
                               "avh_probe_bigN_shard0of1.jsonl")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_module_direct(name, file_path):
    spec = importlib.util.spec_from_file_location(name, file_path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _sync_time():
    """CUDA sync + high-res wall clock. Falls back gracefully if no CUDA."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass
    return time.perf_counter()


def _probe_cache_info(path, label):
    """Return dict with cache metadata; does not load full records."""
    if not os.path.exists(path):
        return {"exists": False, "path": path, "label": label}
    stat   = os.stat(path)
    n_items = 0
    seen   = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rid = json.loads(line).get("id")
                if rid not in seen:
                    seen.add(rid)
                    n_items += 1
            except Exception:
                pass
    return {
        "exists":     True,
        "label":      label,
        "path":       path,
        "n_items":    n_items,
        "size_bytes": stat.st_size,
        "size_mb":    round(stat.st_size / 1e6, 2),
        "note":       "Cached Inference — probe cache already exists, no rebuild needed",
    }


def _percentile(vals, p):
    if not vals:
        return 0.0
    sv = sorted(vals)
    idx = (len(sv) - 1) * p / 100
    lo, hi = int(idx), min(int(idx) + 1, len(sv) - 1)
    return sv[lo] + (sv[hi] - sv[lo]) * (idx - lo)


def _stats(vals):
    if not vals:
        return {"n": 0, "mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "n":      len(vals),
        "mean":   round(sum(vals) / len(vals), 3),
        "median": round(_percentile(vals, 50), 3),
        "p95":    round(_percentile(vals, 95), 3),
        "max":    round(max(vals), 3),
    }


def _print_summary(rows, dataset_label, cache_info):
    print(f"\n{'='*64}", flush=True)
    print(f"  TIMING SUMMARY — {dataset_label}  (Cached Inference)", flush=True)
    print(f"{'='*64}", flush=True)

    print(f"\nCache info:", flush=True)
    print(f"  path    : {cache_info['path']}", flush=True)
    print(f"  items   : {cache_info.get('n_items', 'N/A')}", flush=True)
    print(f"  size    : {cache_info.get('size_mb', 'N/A')} MB", flush=True)
    print(f"  status  : {cache_info.get('note', '')}", flush=True)

    ok = [r for r in rows if r["success"]]
    print(f"\nSamples: {len(rows)} total, {len(ok)} success", flush=True)

    for key, label in [
        ("time_total_sec",       "total       "),
        ("time_anchor_sec",      "anchor      "),
        ("time_interrogate_sec", "interrogate "),
        ("time_correct_sec",     "correct     "),
        ("time_decide_sec",      "decide      "),
    ]:
        vals = [r[key] for r in ok]
        s = _stats(vals)
        print(f"  {label}  mean={s['mean']:.3f}s  median={s['median']:.3f}s  "
              f"p95={s['p95']:.3f}s  max={s['max']:.3f}s", flush=True)

    # token throughput
    tot_toks  = [r["output_tokens_total"]      for r in ok]
    corr_toks = [r["output_tokens_correct"]    for r in ok]
    int_toks  = [r["output_tokens_interrogate"] for r in ok]
    tps_tot   = [r["tok_per_sec_total"]   for r in ok if r["tok_per_sec_total"]   > 0]
    tps_corr  = [r["tok_per_sec_correct"] for r in ok if r["tok_per_sec_correct"] > 0]
    print(f"\n  output_tokens  total={sum(tot_toks)}  "
          f"interrogate={sum(int_toks)}  correct={sum(corr_toks)}", flush=True)
    if tps_tot:
        s = _stats(tps_tot)
        print(f"  tok/s (total)    mean={s['mean']:.1f}  median={s['median']:.1f}  "
              f"p95={s['p95']:.1f}  max={s['max']:.1f}", flush=True)
    if tps_corr:
        s = _stats(tps_corr)
        print(f"  tok/s (correct)  mean={s['mean']:.1f}  median={s['median']:.1f}  "
              f"p95={s['p95']:.1f}  max={s['max']:.1f}", flush=True)

    n_cached = sum(1 for r in ok if r["is_cached_anchor"])
    print(f"\n  Cached anchor (no GPU): {n_cached}/{len(ok)}", flush=True)
    print(f"{'='*64}\n", flush=True)

    return {k: _stats([r[k] for r in ok]) for k in [
        "time_total_sec", "time_anchor_sec", "time_interrogate_sec",
        "time_correct_sec", "time_decide_sec"
    ]}


# ---------------------------------------------------------------------------
# HaloQuest timing
# ---------------------------------------------------------------------------

def run_haloquest(n, skip_verify, out_path, start_idx=0):
    print(f"\n[timing] HaloQuest  n={n}  skip_verify={skip_verify}", flush=True)

    # Cache info (always pre-existing for Champion arm)
    halo_probe_path = os.path.join(REPO, "outputs", "cf_probe",
                                    "haloquest_textcf_tokens.jsonl")
    cache_info = _probe_cache_info(halo_probe_path, "haloquest_textcf_tokens")
    print(f"[cache] {cache_info}", flush=True)

    # Imports
    import freeform_repair as fr
    try:
        import localizer_v2 as lv2
        import evidence_verifier as ev
        _LOC_OK = True
    except ImportError as e:
        print(f"[import] localizer_v2/evidence_verifier not found: {e}", flush=True)
        _LOC_OK = False

    # Patch torchvision ABI and load model
    import torch
    _tv = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
    if _tv < (2, 6):
        import torch._library.fake_impl as _fi
        _orig = _fi.FakeImplHolder.register
        def _safe(self, func, *a, **kw):
            try:
                return _orig(self, func, *a, **kw)
            except (RuntimeError, TypeError) as e:
                if "does not exist" in str(e) or "unexpected keyword" in str(e):
                    class _N:
                        def destroy(self): pass
                    return _N()
                raise
        _fi.FakeImplHolder.register = _safe

    try:
        from transformers import Qwen2_5OmniForConditionalGeneration as _OC
        def _lsp(self, path):
            for k, v in torch.load(path, weights_only=True).items():
                self.speaker_map[k] = v
        _OC.load_speakers = _lsp
    except Exception:
        pass

    _load_module_direct("methods.shared.paths", os.path.join(SHARED, "paths.py"))
    _mu = _load_module_direct("methods.shared.model_utils",
                               os.path.join(SHARED, "model_utils.py"))

    _DEVSHM = "/dev/shm/Qwen2.5-Omni-7B"
    _init = torch.zeros(1, device="cuda"); del _init

    t_load_start = time.perf_counter()
    if os.path.isdir(_DEVSHM) and os.path.isfile(os.path.join(_DEVSHM, "config.json")):
        print(f"[model] loading from /dev/shm ...", flush=True)
        processor, model = _mu.load_model(_DEVSHM)
    else:
        print("[model] loading from default path ...", flush=True)
        processor, model = _mu.load_model()
    t_load_end = time.perf_counter()
    print(f"[model] loaded in {t_load_end - t_load_start:.1f}s", flush=True)

    from methods.shared.model_utils import (
        build_inputs, make_messages,
        build_user_content_with_media_tokens,
        prepare_for_thinker, decode_outputs,
    )
    _dl = _load_module_direct("methods.shared.data_loader",
                               os.path.join(SHARED, "data_loader.py"))
    load_dataset     = _dl.load_dataset
    load_media_for_item = _dl.load_media_for_item
    from methods.shared.prompts import get_system_prompt

    items = load_dataset(HALO_DATA_PATH, HALO_DATA_ROOT)
    by_id = {str(it["id"]): it for it in items}
    cache = fr.load_haloquest()
    joined = []
    for s in cache:
        it = by_id.get(str(s["id"]))
        if it is not None:
            joined.append((s, it))
    joined = joined[start_idx:start_idx + n]
    print(f"[data] running {len(joined)} samples (start_idx={start_idx})", flush=True)

    sys_prompt = get_system_prompt("haloquest", "v1")
    rows = []

    for k, (s, it) in enumerate(joined):
        sid = str(s["id"])
        row = {
            "sample_id":                 sid,
            "success":                   False,
            "time_total_sec":            0.0,
            "time_anchor_sec":           0.0,
            "time_interrogate_sec":      0.0,
            "time_correct_sec":          0.0,
            "time_decide_sec":           0.0,
            "is_cached_anchor":          True,
            "dataset":                   "haloquest",
            "output_tokens_interrogate": 0,
            "output_tokens_correct":     0,
            "output_tokens_total":       0,
            "tok_per_sec_correct":       0.0,
            "tok_per_sec_total":         0.0,
        }

        try:
            base_pred = fr.tokens_to_text(s["tokens"])
            question  = it.get("question", "")

            # T0: pipeline start
            T0 = _sync_time()

            # ---- ANCHOR (lv2.localize — CPU, from probe cache) ----
            cache_record = {
                "id":            s["id"],
                "tokens":        s["tokens"],
                "delta_img_tok": s.get("delta_img_tok", []),
                "delta_text_tok":s.get("delta_text_tok", []),
                "classes":       s.get("classes"),
                "question":      question,
                "lp_full":       s.get("lp_full", []),
            }
            lv2_result = lv2.localize(cache_record) if _LOC_OK else None
            T1 = _sync_time()  # Anchoring done

            row["is_cached_anchor"] = True  # localize reads only pre-computed delta_img_tok

            # ---- Build generate closure ----
            _pv, _av, _iv = load_media_for_item(it, HALO_DATA_PATH, HALO_DATA_ROOT)
            _gs = {"pv": _pv, "av": _av, "iv": _iv, "sys": sys_prompt}
            # phase flag: True = inside interrogation, False = inside correction
            _phase = {"interrogate": True}
            _tok_counts = {"interrogate": 0, "correct": 0}

            def _generate(prompt_text, max_new_tokens, _g=_gs):
                uc   = build_user_content_with_media_tokens(
                    _g["pv"], _g["av"], _g["iv"], prompt_text)
                msgs = make_messages(_g["sys"], uc)
                inp  = build_inputs(processor, model, msgs,
                                    _g["pv"], _g["av"], _g["iv"], "haloquest")
                inp  = prepare_for_thinker(inp, model)
                with torch.inference_mode():
                    oids = model.thinker.generate(
                        **inp, max_new_tokens=max_new_tokens, do_sample=False)
                n_out = int(oids.shape[1]) - int(inp["input_ids"].shape[1])
                phase_key = "interrogate" if _phase["interrogate"] else "correct"
                _tok_counts[phase_key] += max(n_out, 0)
                return decode_outputs(
                    processor, oids, inp["input_ids"].shape[1])[0].strip()

            # ---- INTERROGATION (verify_candidates or skipped) ----
            _phase["interrogate"] = True
            if lv2_result is not None and _LOC_OK:
                if skip_verify:
                    for c in lv2_result.candidates:
                        c.verify_status = "UNSUPPORTED"
                        c.verify_raw    = "skip_verify"
                        c.decision      = "REVISION_CANDIDATE"
                else:
                    lv2_result = ev.verify_candidates(lv2_result, it, _generate)
            _phase["interrogate"] = False
            T2 = _sync_time()  # Interrogation done

            # ---- CORRECTION (_lv2_evidence_regen) ----
            repaired   = base_pred
            rev        = lv2_result.revision_candidates if lv2_result else []
            agg_status = ""
            if rev:
                # inline _lv2_evidence_regen (same logic as run_halo_cu121.py)
                primary   = rev[0]
                step1_obs = getattr(primary, "step1_observation", None) or ""
                issue_lines = []
                for c in rev:
                    obs  = getattr(c, "step1_observation", None) or ""
                    desc = (f'The claim "{c.claim}" — visual observation: {obs}'
                            if obs else
                            f'The claim "{c.claim}" was not supported (status: {c.verify_status})')
                    issue_lines.append(f"  - {desc}")
                issue_block = "\n".join(issue_lines)
                if step1_obs:
                    prompt = (
                        f"Question: {question}\n\nOriginal Answer: {base_pred}\n\n"
                        f"Visual Evidence (what the image actually shows):\n{step1_obs}\n\n"
                        f"Issue Found:\n{issue_block}\n\n"
                        f"Instructions:\n"
                        f"- If the object/attribute is absent: say 'There is no {{object}}' clearly\n"
                        f"- Write ONE complete natural sentence\n"
                        f"- Do NOT add information not visible in the image\n\n"
                        f"Corrected Answer:"
                    )
                else:
                    prompt = (
                        f"[Original Answer]: {base_pred}\n\n"
                        f"[Verification Results]:\n{issue_block}\n\n"
                        f"[Question]: {question}\n\n"
                        f"Write ONE complete, natural sentence that corrects the original answer.\n\n"
                        f"Corrected answer:"
                    )
                repaired = _generate(prompt, 96)

                _statuses = [c.verify_status for c in rev]
                agg_status = ("UNSUPPORTED" if any("UNSUPPORTED" in vs for vs in _statuses)
                              else (_statuses[0] if _statuses else ""))
            T3 = _sync_time()  # Correction done

            # ---- DECIDE (_lv2_accept_decision — CPU) ----
            accepted = False
            if rev and repaired != base_pred:
                rep = repaired.strip()
                base = base_pred.strip()
                # 3-gram check
                words = rep.split()
                ok_ngram = True
                if len(words) >= 6:
                    seen_tg = set()
                    for i in range(len(words) - 2):
                        tg = tuple(words[i:i+3])
                        if tg in seen_tg:
                            ok_ngram = False; break
                        seen_tg.add(tg)
                accepted = (
                    bool(rep) and ok_ngram
                    and not (len(base) > 0 and len(rep) < 0.5 * len(base))
                    and rep not in base
                    and not base.startswith(rep)
                )
            T4 = _sync_time()  # Decide done

            t_correct = T3 - T2
            t_total   = T4 - T0
            n_int     = _tok_counts["interrogate"]
            n_cor     = _tok_counts["correct"]
            n_tot     = n_int + n_cor
            row["time_total_sec"]            = round(t_total, 4)
            row["time_anchor_sec"]           = round(T1 - T0, 4)
            row["time_interrogate_sec"]      = round(T2 - T1, 4)
            row["time_correct_sec"]          = round(t_correct, 4)
            row["time_decide_sec"]           = round(T4 - T3, 4)
            row["output_tokens_interrogate"] = n_int
            row["output_tokens_correct"]     = n_cor
            row["output_tokens_total"]       = n_tot
            row["tok_per_sec_correct"]       = round(n_cor / t_correct, 2) if t_correct > 0 else 0.0
            row["tok_per_sec_total"]         = round(n_tot / t_total,   2) if t_total   > 0 else 0.0
            row["success"]                   = True

            print(f"[{k+1}/{len(joined)}] id={sid}  "
                  f"total={row['time_total_sec']:.2f}s  "
                  f"anchor={row['time_anchor_sec']:.3f}s  "
                  f"interrog={row['time_interrogate_sec']:.2f}s  "
                  f"correct={row['time_correct_sec']:.2f}s  "
                  f"decide={row['time_decide_sec']:.4f}s  "
                  f"toks(int={n_int},cor={n_cor})  "
                  f"tok/s(cor={row['tok_per_sec_correct']:.1f})  "
                  f"accepted={accepted}", flush=True)

            torch.cuda.empty_cache()

        except Exception as e:
            row["error"] = f"{type(e).__name__}: {e}"
            row["traceback"] = traceback.format_exc()[-600:]
            print(f"[FAIL id={sid}] {row['error']}", flush=True)

        rows.append(row)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[out] wrote {len(rows)} rows to {out_path}", flush=True)

    stats = _print_summary(rows, "HaloQuest", cache_info)
    return rows, stats


# ---------------------------------------------------------------------------
# AVHBench timing
# ---------------------------------------------------------------------------

def _load_avh_probe(probe_path, avhbench_data, signal_mode="combined"):
    avh_by_uid = {}
    for item in avhbench_data:
        uid = item.get("uid", "")
        try:
            num = str(int(uid.split("_")[-1]))
        except (ValueError, IndexError):
            continue
        avh_by_uid[num] = item

    seen = set()
    records = []
    with open(probe_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r   = json.loads(line)
            rid = str(r["id"])
            if rid in seen:
                continue
            seen.add(rid)
            d_v = np.asarray(r["delta_video"], dtype=np.float64)
            d_a = np.asarray(r["delta_audio"], dtype=np.float64)
            if signal_mode == "combined":
                r["delta_img_tok"] = list(d_v + d_a)
            elif signal_mode == "video_only":
                r["delta_img_tok"] = list(d_v)
            else:
                r["delta_img_tok"] = list(d_a)
            meta = avh_by_uid.get(rid, {})
            r["question"] = meta.get("question", r.get("question", ""))
            records.append(r)
    return records


def run_avhbench(n, out_path, start_idx=0):
    print(f"\n[timing] AVHBench  n={n}  arm=combined", flush=True)

    cache_info = _probe_cache_info(AVH_PROBE_PATH, "avh_probe_bigN_shard0of1")
    print(f"[cache] {cache_info}", flush=True)

    import freeform_repair as fr  # noqa: F401 (needed for sys.path side effect)
    try:
        import localizer_v2 as lv2
        import evidence_verifier as ev
        _LOC_OK = True
    except ImportError as e:
        print(f"[import] localizer_v2/evidence_verifier not found: {e}", flush=True)
        _LOC_OK = False

    import torch
    _tv = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
    if _tv < (2, 6):
        import torch._library.fake_impl as _fi
        _orig = _fi.FakeImplHolder.register
        def _safe(self, func, *a, **kw):
            try:
                return _orig(self, func, *a, **kw)
            except (RuntimeError, TypeError) as e:
                if "does not exist" in str(e) or "unexpected keyword" in str(e):
                    class _N:
                        def destroy(self): pass
                    return _N()
                raise
        _fi.FakeImplHolder.register = _safe

    try:
        from transformers import Qwen2_5OmniForConditionalGeneration as _OC
        def _lsp(self, path):
            for k, v in torch.load(path, weights_only=True).items():
                self.speaker_map[k] = v
        _OC.load_speakers = _lsp
    except Exception:
        pass

    _load_module_direct("methods.shared.paths", os.path.join(SHARED, "paths.py"))
    _mu = _load_module_direct("methods.shared.model_utils",
                               os.path.join(SHARED, "model_utils.py"))

    _DEVSHM = "/dev/shm/Qwen2.5-Omni-7B"
    _init = torch.zeros(1, device="cuda"); del _init

    t_load_start = time.perf_counter()
    if os.path.isdir(_DEVSHM) and os.path.isfile(os.path.join(_DEVSHM, "config.json")):
        processor, model = _mu.load_model(_DEVSHM)
    else:
        processor, model = _mu.load_model()
    t_load_end = time.perf_counter()
    print(f"[model] loaded in {t_load_end - t_load_start:.1f}s", flush=True)

    from methods.shared.model_utils import (
        build_inputs, make_messages,
        build_user_content_with_media_tokens,
        prepare_for_thinker, decode_outputs,
    )
    _dl = _load_module_direct("methods.shared.data_loader",
                               os.path.join(SHARED, "data_loader.py"))
    load_media_for_item = _dl.load_media_for_item
    from methods.shared.prompts import get_system_prompt

    avhbench_data = json.load(open(AVH_DATA_PATH, encoding="utf-8"))
    probe_records = _load_avh_probe(AVH_PROBE_PATH, avhbench_data, "combined")
    probe_records.sort(key=lambda r: int(r["id"]))
    probe_records = probe_records[start_idx:start_idx + n]
    print(f"[data] {len(probe_records)} probe records (start_idx={start_idx})", flush=True)

    # data_loader normalized items
    avh_items = _dl.load_dataset(AVH_DATA_PATH, AVH_DATA_ROOT)
    avh_by_uid = {}
    for it in avh_items:
        uid = it.get("uid", "")
        try:
            num = str(int(uid.split("_")[-1]))
            avh_by_uid[num] = it
        except (ValueError, IndexError):
            pass

    sys_prompt = get_system_prompt("avhbench", "v1")
    rows = []

    for k, rec in enumerate(probe_records):
        rid = str(rec["id"])
        it  = avh_by_uid.get(rid)
        if it is None:
            print(f"[SKIP] id={rid} not in data_loader", flush=True)
            continue

        base_pred = "".join(rec["tokens"])
        question  = rec.get("question", it.get("question", ""))

        row = {
            "sample_id":                 rid,
            "success":                   False,
            "time_total_sec":            0.0,
            "time_anchor_sec":           0.0,
            "time_interrogate_sec":      0.0,
            "time_correct_sec":          0.0,
            "time_decide_sec":           0.0,
            "is_cached_anchor":          True,
            "dataset":                   "avhbench",
            "output_tokens_interrogate": 0,
            "output_tokens_correct":     0,
            "output_tokens_total":       0,
            "tok_per_sec_correct":       0.0,
            "tok_per_sec_total":         0.0,
        }

        try:
            # T0
            T0 = _sync_time()

            # ---- ANCHOR ----
            if _LOC_OK:
                cache_record = {
                    "id":            rec["id"],
                    "tokens":        rec["tokens"],
                    "delta_img_tok": rec["delta_img_tok"],
                    "delta_video":   rec.get("delta_video", []),
                    "delta_audio":   rec.get("delta_audio", []),
                    "delta_text_tok":rec.get("delta_text", []),
                    "classes":       rec.get("classes"),
                    "question":      question,
                    "lp_full":       rec.get("lp_full", []),
                }
                lv2_result = lv2.localize(cache_record)
            else:
                lv2_result = None
            T1 = _sync_time()

            row["is_cached_anchor"] = True

            # ---- Build generate closure ----
            _pv, _av, _iv = load_media_for_item(it, AVH_DATA_PATH, AVH_DATA_ROOT)
            _gs = {"pv": _pv, "av": _av, "iv": _iv, "sys": sys_prompt}
            _phase = {"interrogate": True}
            _tok_counts = {"interrogate": 0, "correct": 0}

            def _generate(prompt_text, max_new_tokens, _g=_gs):
                uc   = build_user_content_with_media_tokens(
                    _g["pv"], _g["av"], _g["iv"], prompt_text)
                msgs = make_messages(_g["sys"], uc)
                inp  = build_inputs(processor, model, msgs,
                                    _g["pv"], _g["av"], _g["iv"], "haloquest")
                inp  = prepare_for_thinker(inp, model)
                with torch.inference_mode():
                    oids = model.thinker.generate(
                        **inp, max_new_tokens=max_new_tokens, do_sample=False)
                n_out = int(oids.shape[1]) - int(inp["input_ids"].shape[1])
                phase_key = "interrogate" if _phase["interrogate"] else "correct"
                _tok_counts[phase_key] += max(n_out, 0)
                return decode_outputs(
                    processor, oids, inp["input_ids"].shape[1])[0].strip()

            # ---- INTERROGATION (verify — AVH always verifies) ----
            _phase["interrogate"] = True
            if lv2_result is not None and _LOC_OK:
                lv2_result = ev.verify_candidates(lv2_result, it, _generate)
            _phase["interrogate"] = False
            T2 = _sync_time()

            # ---- CORRECTION ----
            repaired   = base_pred
            rev        = lv2_result.revision_candidates if lv2_result else []
            agg_status = ""
            if rev:
                primary   = rev[0]
                step1_obs = getattr(primary, "step1_observation", None) or ""
                if step1_obs:
                    prompt = (
                        f"Question: {question}\n\nOriginal Answer: {base_pred}\n\n"
                        f"What the video/audio actually shows:\n{step1_obs}\n\n"
                        f"Specific issue: The claim \"{primary.claim}\" is not supported.\n\n"
                        f"Task: Write ONE corrected sentence.\n"
                        f"Rules:\n"
                        f"- Keep ALL parts of the original answer NOT about \"{primary.span_text}\"\n"
                        f"- Only change the specific unsupported claim\n"
                        f"- If the object is absent: say 'There is no [object]'\n\n"
                        f"Corrected Answer:"
                    )
                    repaired = _generate(prompt, 96)
                else:
                    issue_lines = []
                    for c in rev:
                        obs  = getattr(c, "step1_observation", None) or ""
                        desc = (f'The claim "{c.claim}" — visual observation: {obs}'
                                if obs else
                                f'The claim "{c.claim}" status: {c.verify_status}')
                        issue_lines.append(f"  - {desc}")
                    issue_block = "\n".join(issue_lines)
                    prompt = (
                        f"Question: {question}\n\nOriginal Answer: {base_pred}\n\n"
                        f"Verification finding: The specific claim \"{primary.span_text}\" "
                        f"was marked as {primary.verify_status}.\n\n"
                        f"Task: Write ONE corrected sentence.\n"
                        f"- Keep the scene and all other facts from the original answer\n"
                        f"- Only address the specific claim about \"{primary.span_text}\"\n\n"
                        f"Corrected Answer:"
                    )
                    repaired = _generate(prompt, 96)
                _statuses  = [c.verify_status for c in rev]
                agg_status = ("UNSUPPORTED" if any("UNSUPPORTED" in vs for vs in _statuses)
                              else (_statuses[0] if _statuses else ""))
            T3 = _sync_time()

            # ---- DECIDE ----
            accepted = False
            if rev and repaired != base_pred:
                rep  = repaired.strip()
                base = base_pred.strip()
                words = rep.split()
                ok_ngram = True
                if len(words) >= 6:
                    seen_tg = set()
                    for i in range(len(words) - 2):
                        tg = tuple(words[i:i+3])
                        if tg in seen_tg:
                            ok_ngram = False; break
                        seen_tg.add(tg)
                # content-word overlap guard (AVH-specific)
                import re as _re
                _stop = {'a','an','the','is','are','was','were','in','on','at','of','to',
                         'and','or','not','no','there','this','that','it','its','with','for',
                         'by','as','be','have','has','had','do','does','did','but','from'}
                def _cw(t):
                    return set(_re.findall(r'\b[a-z]+\b', t.lower())) - _stop
                bc, rc = _cw(base), _cw(rep)
                overlap_ok = (len(bc & rc) / len(bc) >= 0.25) if bc else True
                accepted = (
                    bool(rep) and ok_ngram and overlap_ok
                    and not (len(base) > 0 and len(rep) < 0.5 * len(base))
                    and rep not in base and not base.startswith(rep)
                )
            T4 = _sync_time()

            t_correct = T3 - T2
            t_total   = T4 - T0
            n_int     = _tok_counts["interrogate"]
            n_cor     = _tok_counts["correct"]
            n_tot     = n_int + n_cor
            row["time_total_sec"]            = round(t_total, 4)
            row["time_anchor_sec"]           = round(T1 - T0, 4)
            row["time_interrogate_sec"]      = round(T2 - T1, 4)
            row["time_correct_sec"]          = round(t_correct, 4)
            row["time_decide_sec"]           = round(T4 - T3, 4)
            row["output_tokens_interrogate"] = n_int
            row["output_tokens_correct"]     = n_cor
            row["output_tokens_total"]       = n_tot
            row["tok_per_sec_correct"]       = round(n_cor / t_correct, 2) if t_correct > 0 else 0.0
            row["tok_per_sec_total"]         = round(n_tot / t_total,   2) if t_total   > 0 else 0.0
            row["success"]                   = True

            print(f"[{k+1}/{len(probe_records)}] id={rid}  "
                  f"total={row['time_total_sec']:.2f}s  "
                  f"anchor={row['time_anchor_sec']:.3f}s  "
                  f"interrog={row['time_interrogate_sec']:.2f}s  "
                  f"correct={row['time_correct_sec']:.2f}s  "
                  f"decide={row['time_decide_sec']:.4f}s  "
                  f"toks(int={n_int},cor={n_cor})  "
                  f"tok/s(cor={row['tok_per_sec_correct']:.1f})  "
                  f"accepted={accepted}", flush=True)

            torch.cuda.empty_cache()

        except Exception as e:
            row["error"]     = f"{type(e).__name__}: {e}"
            row["traceback"] = traceback.format_exc()[-600:]
            print(f"[FAIL id={rid}] {row['error']}", flush=True)

        rows.append(row)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[out] wrote {len(rows)} rows to {out_path}", flush=True)

    stats = _print_summary(rows, "AVHBench", cache_info)
    return rows, stats


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["haloquest", "avhbench", "both"],
                    default="both")
    ap.add_argument("--n", type=int, default=10,
                    help="Number of samples per dataset")
    ap.add_argument("--skip-verify", action="store_true",
                    help="HaloQuest Champion arm: skip ev.verify_candidates "
                         "(--lv2-skip-verify equivalent)")
    ap.add_argument("--halo-out", type=str,
                    default=os.path.join(REPO, "outputs", "timing",
                                         "haloquest_timing_n10.jsonl"))
    ap.add_argument("--avh-out", type=str,
                    default=os.path.join(REPO, "outputs", "timing",
                                         "avhbench_timing_n10.jsonl"))
    ap.add_argument("--start-idx", type=int, default=0,
                    help="Start offset into the dataset (for parallel sharding)")
    args = ap.parse_args()

    results = {}

    if args.dataset in ("haloquest", "both"):
        rows_h, stats_h = run_haloquest(args.n, args.skip_verify, args.halo_out,
                                         start_idx=args.start_idx)
        results["haloquest"] = stats_h

    if args.dataset in ("avhbench", "both"):
        rows_a, stats_a = run_avhbench(args.n, args.avh_out,
                                        start_idx=args.start_idx)
        results["avhbench"] = stats_a

    # Final combined summary line (for feishu notification)
    halo_mean = (results["haloquest"]["time_total_sec"]["mean"]
                 if "haloquest" in results else None)
    avh_mean  = (results["avhbench"]["time_total_sec"]["mean"]
                 if "avhbench" in results else None)
    print(
        f"\n[DONE] HaloQuest mean={halo_mean}s  AVHBench mean={avh_mean}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
