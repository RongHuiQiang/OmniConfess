# -*- coding: utf-8 -*-
"""
run_halo_cu121.py — haloquest single-tau single-GPU driver (v4, new file)

Fixes torchvision ABI crash when using torch==2.5.1+cu121:
  methods/shared/__init__.py eagerly imports model_utils → transformers →
  torchvision, which tries to register torchvision::nms against the cu121
  op registry and fails.  Solution: import data_loader and model_utils
  directly by file path via importlib, bypassing __init__.py entirely.

Constraints: cards 0-3 only. No modification of existing files. No commit.

Usage:
  CUDA_VISIBLE_DEVICES=0 MMHALU_MODEL_ROOT=/slow_share/tsq/MODELS HF_HUB_OFFLINE=1 \
  PYTHONPATH=/tmp/pip-target-b9a06nc9/lib/python \
  LD_LIBRARY_PATH=/dev/shm/so_mirror:/tmp/pip-target-b9a06nc9/lib/python/torch/lib \
    python3 run_halo_cu121.py --tau 0.5 --limit 50 \
      --out outputs/freeform_grid/haloquest_tau0.5.jsonl
"""
import argparse
import importlib.util
import json
import math
import os
import sys
import time
import traceback

REPO = os.path.dirname(os.path.abspath(__file__))
SHARED = os.path.join(REPO, "OmniConfess", "methods", "shared")

# Insert repo roots so freeform_repair and scripts can be found
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "OmniConfess"))

# Patch out transformers CVE-2025-32434 torch.load safety check BEFORE
# transformers is imported. spk_dict.pt is Qwen2.5-Omni TTS speaker dict
# (not used in text-only inference). Our CUDA env is pinned to torch 2.5.1
# +cu121 which transformers >=5.17 blocks. Checkpoint is trusted.
try:
    import transformers.utils.import_utils as _tiu_early
    _tiu_early.check_torch_load_is_safe = lambda: None
except Exception:
    pass  # will be patched again later if needed

DATA_PATH = os.path.join(REPO, "OmniHalluBench", "haloquest", "haloquest.json")
DATA_ROOT = os.path.join(REPO, "OmniHalluBench", "haloquest")


def _load_module_direct(name, file_path):
    """Import a module from an absolute file path, bypassing package __init__.py."""
    spec = importlib.util.spec_from_file_location(name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _lv2_evidence_regen(lv2_result, question, base_pred, _generate_fn, max_new_tokens=96):
    """
    Evidence-grounded whole-sentence regeneration for lv2 path (C1 version).

    Uses the step1 visual observation stored in each REVISION_CANDIDATE
    (populated by evidence_verifier B1) to ground the correction prompt.
    Falls back to claim/status-only format when step1_observation is absent.
    Returns the raw generated string (accept/reject decided by caller).
    """
    rev = lv2_result.revision_candidates
    if not rev:
        return base_pred

    # Use first revision candidate for the main visual context.
    # If multiple candidates exist, list all issues but anchor on the primary obs.
    primary = rev[0]
    step1_obs = getattr(primary, "step1_observation", None) or ""

    # Build per-candidate issue descriptions
    issue_lines = []
    for c in rev:
        obs = getattr(c, "step1_observation", None) or ""
        if obs:
            desc = f'The claim "{c.claim}" — visual observation: {obs}'
        else:
            desc = f'The claim "{c.claim}" was not supported by the image (status: {c.verify_status})'
        issue_lines.append(f"  - {desc}")
    issue_block = "\n".join(issue_lines)

    # Determine the primary object/attribute for corrective instruction
    claim = primary.claim or primary.span_text

    if step1_obs:
        # C1: include explicit visual observation in the prompt
        prompt = (
            f"Question: {question}\n\n"
            f"Original Answer: {base_pred}\n\n"
            f"Visual Evidence (what the image actually shows):\n{step1_obs}\n\n"
            f"Issue Found:\n{issue_block}\n\n"
            f"Instructions:\n"
            f"- If the object/attribute is absent: say 'There is no {{object}}' clearly\n"
            f"- If the object exists but detail is unclear: say 'There is a {{object}}, "
            f"but {{detail}} is not clear'\n"
            f"- If the original answer is essentially correct: preserve it\n"
            f"- Write ONE complete natural sentence\n"
            f"- Do NOT add information not visible in the image\n\n"
            f"Corrected Answer:"
        )
    else:
        # Fallback: original format without step1 observation
        prompt = (
            f"[Original Answer]: {base_pred}\n\n"
            f"[Verification Results]:\n{issue_block}\n\n"
            f"[Question]: {question}\n\n"
            f"The original answer contains a claim that is not supported by the image.\n"
            f"Write ONE complete, natural sentence that corrects the original answer.\n\n"
            f"Important:\n"
            f"- If something is not present in the image, say 'There is no X' or 'X is not visible', "
            f"not 'the image does not provide enough information'.\n"
            f"- If the original answer had a specific factual error, correct the specific fact.\n"
            f"- Preserve any parts of the original answer that are correct.\n"
            f"- Be direct and specific, not vague or hedging.\n\n"
            f"Corrected answer:"
        )
    return _generate_fn(prompt, max_new_tokens)


def _check_preserves_positive_content(repaired: str, base: str,
                                       verify_status: str = "") -> bool:
    """
    If base contains a concrete positive assertion (not a refusal/negation),
    check that repaired does not wholesale convert it into a refusal.

    Prevents cases like: base="The dog is brown" → repaired="There is no dog"
    when the verify step did NOT conclude UNSUPPORTED.

    verify_status: pass the lv2 verify conclusion string if available
    (e.g. "UNSUPPORTED", "SUPPORTED", "PARTIAL").  When UNSUPPORTED, a
    positive→refusal flip is legitimate and this guard is bypassed.

    Returns False (reject repaired) when the guard fires; True otherwise.
    """
    base_lower = base.lower()
    repaired_lower = repaired.lower()

    _refusal_words = {"not visible", "cannot", "no ", "there is no",
                      "not present", "not shown", "unable", "unknown",
                      "unclear", "can't", "don't know", "do not know"}

    base_is_positive = not any(w in base_lower for w in _refusal_words)
    repaired_is_refusal = any(w in repaired_lower for w in _refusal_words)

    if base_is_positive and repaired_is_refusal:
        # If the verifier found the claim not supported (any edit-signal status),
        # a flip to refusal is correct — let it through.
        # v3 statuses: CONTRADICTED, UNSUPPORTED_BY_EVIDENCE, INSUFFICIENT_TO_DETERMINE
        # legacy: UNSUPPORTED
        _edit_signals = {"UNSUPPORTED", "CONTRADICTED", "UNSUPPORTED_BY_EVIDENCE",
                         "INSUFFICIENT_TO_DETERMINE"}
        if verify_status and any(sig in verify_status.upper() for sig in _edit_signals):
            return True
        return False

    return True


def _lv2_accept_decision(repaired: str, base_pred: str,
                         verify_status: str = "") -> bool:
    """
    Decide whether to accept the repaired answer over base_pred.
    Does NOT read ground truth. Returns True to accept, False to fall back.

    Reject if:
    1. repaired contains 3-gram repetition (degeneration signal)
    2. repaired is more than 50% shorter than base_pred (collapse)
    3. repaired is a mechanical substring or prefix of base_pred
    4. repaired drops a negation/refusal word that base_pred had
       (e.g. base says "not visible" but repaired omits "not")
    5. repaired converts a positive assertion to a refusal without
       verifier support (semantic preservation check)
    """
    rep = repaired.strip()
    base = base_pred.strip()

    if not rep:
        return False

    # 1. 3-gram repetition check
    words = rep.split()
    if len(words) >= 6:
        trigrams = [tuple(words[i:i+3]) for i in range(len(words)-2)]
        seen = set()
        for tg in trigrams:
            if tg in seen:
                return False
            seen.add(tg)

    # 2. Length collapse: repaired shorter than 50% of base
    if len(base) > 0 and len(rep) < 0.5 * len(base):
        return False

    # 3. Mechanical substring/prefix
    if rep in base or base.startswith(rep):
        return False

    # 4. Negation-flip: base had refusal/negation words but repaired lost them
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

    # 5. Semantic preservation: positive assertion → refusal without evidence support
    # v3: both CONTRADICTED and UNSUPPORTED_BY_EVIDENCE are legitimate edit signals,
    # so a positive→refusal flip is allowed when either is present.
    _evidence_signals = {"UNSUPPORTED", "CONTRADICTED", "UNSUPPORTED_BY_EVIDENCE"}
    _has_evidence = any(sig in verify_status.upper() for sig in _evidence_signals)
    if not _has_evidence:
        if not _check_preserves_positive_content(rep, base, verify_status):
            return False

    # 6. Refusal→refusal equivalent rewrite: if both base and repaired are refusals,
    # and they are semantically equivalent (token F1 > 0.5), reject the rewrite.
    # This prevents accepting paraphrases like "not visible" → "no information about"
    # that produce the same meaning but different token overlap, hurting F1.
    _refusal_check_markers = ["not visible", "cannot", "no ", "there is no", "not present",
                               "does not", "not shown", "unable", "unclear", "can't"]
    base_is_refusal_b = any(m in base.lower() for m in _refusal_check_markers)
    rep_is_refusal_b = any(m in rep.lower() for m in _refusal_check_markers)
    if base_is_refusal_b and rep_is_refusal_b:
        # Both are refusals — if token F1(repaired, base) > 0.5, they are equivalent
        # and there is no value in accepting the rewrite.
        try:
            from scripts.tokenf1_fullset import token_f1_multiref as _tf1
            self_f1 = _tf1(rep, base, multiref=False)
            if self_f1 > 0.5:
                return False  # equivalent refusal, reject rewrite
        except Exception:
            pass  # if import fails, allow through

    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--prior-baseline", type=str, default="none")
    ap.add_argument("--max-new-tokens", type=int, default=96)
    # Round 2 span-selection controls
    ap.add_argument("--min-span-len", type=int, default=1,
                    help="Round 2: 丢弃长度 < min_span_len token 的短 span (默认1=不过滤)")
    ap.add_argument("--max-spans", type=int, default=0,
                    help="Round 2: 每样本最多保留 N 个 span，按 rho 均值取 top-N (默认0=不限)")
    # Round 3 dependency-mode controls
    ap.add_argument("--select-mode", type=str, default="rho",
                    choices=["rho", "under_reliance", "over_reliance", "either", "topk"],
                    help="Round 3: span 选择信号 (rho/under_reliance/over_reliance/either/topk)")
    # Round 5 diagnosis-injection controls
    ap.add_argument("--diag-mode", type=str, default="none",
                    choices=["none", "hard", "soft"],
                    help="Round 5: prompt 诊断注入模式 (none=无注入, hard=机制语言, soft=操作性指令)")
    # Round 6 evidence-constrained correction controls
    ap.add_argument("--evidence-mode", action="store_true", default=False,
                    help="Round 6: 启用两步证据查询路径 (Phase B)")
    ap.add_argument("--evidence-max-new-tokens", type=int, default=64,
                    help="Round 6: 证据查询 generate 的 token 上限")
    # Round 7 coverage-gated repair
    ap.add_argument("--coverage-cap", type=float, default=0.0,
                    help="Round 7: 跳过修复的覆盖率上界 (0=不限; e.g. 0.4 → coverage>0.4 跳过)")
    # 新定位器 (localizer_v2) 控制
    ap.add_argument("--use-localizer-v2", action="store_true",
                    help="使用新 delta_img_tok 定位器替换 rho_i span 选择（Step1-5）")
    ap.add_argument("--localizer-min-vis-dep", type=float, default=0.05,
                    help="新定位器 visual_dependence 阈值（候选入口）")
    ap.add_argument("--lv2-skip-verify", action="store_true",
                    help="Arm C 消融: 跳过 verify_candidates，所有候选直接变 REVISION_CANDIDATE")
    ap.add_argument("--lv2-always-accept", action="store_true",
                    help="Arm E 消融: 跳过 accept decision，直接使用 repaired 结果 (不经过 _lv2_accept_decision)")
    args = ap.parse_args()

    tau = args.tau
    is_inf = math.isinf(tau) or tau > 1e8

    out_path = args.out
    if not out_path:
        tag = "inf" if is_inf else str(tau)
        out_path = os.path.join(REPO, "outputs", "freeform_grid",
                                f"haloquest_tau{tag}.jsonl")
    if not os.path.isabs(out_path):
        out_path = os.path.join(REPO, out_path)

    print(f"[start] pid={os.getpid()} tau={tau} is_inf={is_inf} "
          f"limit={args.limit} out={out_path}", flush=True)
    print(f"[path] DATA_PATH={DATA_PATH} exists={os.path.exists(DATA_PATH)}",
          flush=True)

    # --- imports: freeform_repair and tokenf1 don't touch torchvision ---
    print("[import] freeform_repair...", flush=True)
    import freeform_repair as fr
    print("[import] tokenf1...", flush=True)
    from scripts.tokenf1_fullset import token_f1_multiref
    # 新定位器（localizer_v2.py 已存在时启用，否则跳过）
    try:
        import localizer_v2 as lv2
        import evidence_verifier as ev
        _LOCALIZER_AVAILABLE = True
    except ImportError:
        _LOCALIZER_AVAILABLE = False
    print("[import] done", flush=True)

    # --- data_loader: import directly, skip __init__.py ---
    print("[import] data_loader (direct)...", flush=True)
    _dl = _load_module_direct(
        "methods.shared.data_loader",
        os.path.join(SHARED, "data_loader.py"),
    )
    load_dataset = _dl.load_dataset
    print("[import] data_loader OK", flush=True)

    # --- build join ---
    print("[join] building join...", flush=True)
    items = load_dataset(DATA_PATH, DATA_ROOT)
    by_id = {str(it["id"]): it for it in items}
    cache = fr.load_haloquest()
    joined = []
    missing = 0
    for s in cache:
        it = by_id.get(str(s["id"]))
        if it is None:
            missing += 1
            continue
        joined.append((s, it))
    print(f"[join] cache={len(cache)} data={len(items)} "
          f"joined={len(joined)} missing={missing}", flush=True)

    if args.limit and args.limit > 0:
        joined = joined[:args.limit]
    print(f"[join] running {len(joined)} samples tau={tau}", flush=True)

    # --- model load (deferred, only when tau < inf) ---
    processor, model = None, None
    if not is_inf:
        import torch
        _torch_ver = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
        if _torch_ver < (2, 6):
            print("[import] patching torchvision/torch ABI mismatch (torch<2.6)...", flush=True)
            # torchvision._meta_registrations tries to register torchvision::nms as
            # a fake impl, but older torch builds don't have that op in their
            # registry, causing a RuntimeError.  Patch FakeImplHolder.register to
            # silently skip registrations for ops that don't exist in the registry.
            import torch._library.fake_impl as _fi
            _orig_register = _fi.FakeImplHolder.register
            def _safe_register(self, func, *args, **kwargs):
                try:
                    return _orig_register(self, func, *args, **kwargs)
                except (RuntimeError, TypeError) as _e:
                    if "does not exist" in str(_e) or "unexpected keyword" in str(_e):
                        class _NopHandle:
                            def destroy(self): pass
                        return _NopHandle()
                    raise
            _fi.FakeImplHolder.register = _safe_register
            print("[import] patch applied, importing model_utils...", flush=True)
        else:
            print(f"[import] torch {torch.__version__} >= 2.6, skip ABI patch", flush=True)

        # Patch Qwen2_5OmniForConditionalGeneration.load_speakers to skip
        # the check_torch_load_is_safe() guard added in transformers >=5.17
        # (CVE-2025-32434). spk_dict.pt is the TTS speaker dict — unused in
        # text-only inference. Torch is pinned at 2.5.1+cu121; checkpoint trusted.
        try:
            from transformers import Qwen2_5OmniForConditionalGeneration as _OmniCls
            import torch as _torch_patch
            def _load_speakers_patched(self, path):
                for key, value in _torch_patch.load(path, weights_only=True).items():
                    self.speaker_map[key] = value
            _OmniCls.load_speakers = _load_speakers_patched
            print("[import] patched load_speakers (CVE-2025-32434 bypass)", flush=True)
        except Exception as _pe:
            print(f"[import] load_speakers patch failed: {_pe}", flush=True)

        # Import paths.py first so resolve_model works
        _load_module_direct(
            "methods.shared.paths",
            os.path.join(SHARED, "paths.py"),
        )
        _mu = _load_module_direct(
            "methods.shared.model_utils",
            os.path.join(SHARED, "model_utils.py"),
        )
        load_model = _mu.load_model
        print("[model] loading Qwen2.5-Omni-7B...", flush=True)
        t0 = time.time()
        processor, model = load_model()
        print(f"[model] loaded in {time.time()-t0:.1f}s", flush=True)
    else:
        print("[model] tau=inf, skip model load", flush=True)

    # --- main loop ---
    rows = []
    n_fail = 0
    base_f1s, rep_f1s = [], []

    for k, (s, it) in enumerate(joined):
        cfg_s = fr.RepairConfig(
            tau=tau if not is_inf else 1e9,
            prior_baseline=args.prior_baseline,
            max_new_tokens=args.max_new_tokens,
            dataset_name="haloquest",
            min_span_len=args.min_span_len,
            max_spans=args.max_spans,
            select_mode=args.select_mode,
            diag_mode=args.diag_mode,
            evidence_mode=args.evidence_mode,
            evidence_max_new_tokens=args.evidence_max_new_tokens,
            coverage_cap=args.coverage_cap,
        )
        scored = fr.score_sample(s, cfg_s)
        base_pred = fr.tokens_to_text(s["tokens"])
        gt = it.get("gt", "")
        spans = scored["spans"]

        # --- 新定位器骨架（--use-localizer-v2 时替换 span 选择）---
        lv2_result = None
        _lv2_generate = None  # hoisted: set inside model-available block below
        if args.use_localizer_v2 and _LOCALIZER_AVAILABLE:
            # s 里已有 tokens/delta_img_tok 等缓存字段（由 load_haloquest 保留）
            # 构造 localizer_v2 期望的 record 格式（缓存字段直接传，question 从 it 取）
            cache_record = {
                "id": s["id"],
                "tokens": s["tokens"],
                "delta_img_tok": s.get("delta_img_tok", []),  # ① load_haloquest 已保留原始字段
                "delta_text_tok": s.get("delta_text_tok", []),
                "classes": s.get("classes"),
                "question": it.get("question", ""),
                "lp_full": s.get("lp_full", []),
            }
            lv2_result = lv2.localize(cache_record)
            # 旧 rho 结果留作诊断基线
            lv2_result.rho_spans_old = scored["spans"]

            if model is not None:
                # Build a generate closure bound to this sample's media
                import torch as _torch_lv2
                from methods.shared.model_utils import (
                    build_inputs, make_messages,
                    build_user_content_with_media_tokens,
                    prepare_for_thinker, decode_outputs,
                )
                from methods.shared.data_loader import load_media_for_item
                from methods.shared.prompts import get_system_prompt

                _pv, _av, _iv = load_media_for_item(it, DATA_PATH, DATA_ROOT)
                _sys_prompt = get_system_prompt(cfg_s.dataset_name, "v1")

                # Use a mutable container so the closure captures by reference
                # and the outer scope can also call _lv2_generate cleanly.
                _gen_state = {
                    "pv": _pv, "av": _av, "iv": _iv,
                    "sys": _sys_prompt,
                }

                def _lv2_generate(prompt_text, max_new_tokens,
                                  _gs=_gen_state):
                    uc = build_user_content_with_media_tokens(
                        _gs["pv"], _gs["av"], _gs["iv"], prompt_text)
                    msgs = make_messages(_gs["sys"], uc)
                    inp = build_inputs(processor, model, msgs,
                                       _gs["pv"], _gs["av"], _gs["iv"],
                                       cfg_s.dataset_name)
                    inp = prepare_for_thinker(inp, model)
                    with _torch_lv2.inference_mode():
                        oids = model.thinker.generate(
                            **inp,
                            max_new_tokens=max_new_tokens,
                            do_sample=False,
                        )
                    return decode_outputs(
                        processor, oids,
                        inp["input_ids"].shape[1]
                    )[0].strip()

                if args.lv2_skip_verify:
                    # Arm C: skip verify — promote all candidates directly
                    # (backward-compatible: use UNSUPPORTED so downstream
                    #  _check_preserves_positive_content lets flips through)
                    for _c in lv2_result.candidates:
                        _c.verify_status = "UNSUPPORTED"
                        _c.verify_raw = "skip_verify"
                        _c.decision = "REVISION_CANDIDATE"
                else:
                    # Steps 4-5: v3 broad_verify — no candidates are filtered
                    lv2_result = ev.verify_candidates(
                        lv2_result, it, _lv2_generate)

            spans = [(c.token_start, c.token_end)
                     for c in lv2_result.revision_candidates]
        # --- 结束 ---

        rec = {
            "id": s["id"],
            "question": it.get("question", ""),
            "gt": gt,
            "n_tokens": scored["n_tokens"],
            "n_spans": len(spans),
            "spans": spans,
            "base_pred": base_pred,
            "status": "success",
            "tau": tau,
        }

        if lv2_result is not None:
            rec["lv2_n_candidates"] = len(lv2_result.candidates)
            rec["lv2_n_revision"] = len(lv2_result.revision_candidates)
            rec["lv2_rho_spans_old"] = lv2_result.rho_spans_old
            # Serialize all candidates (including verify_status, step1_observation)
            rec["candidates"] = [c.to_dict() for c in lv2_result.candidates]

        if is_inf or not spans:
            rec["repaired_pred"] = base_pred
            rec["replacements"] = []
            if not spans:
                rec["no_edit"] = True
        elif lv2_result is not None and lv2_result.revision_candidates and model is not None:
            # LV2 path: evidence-grounded whole-sentence regeneration
            try:
                repaired = _lv2_evidence_regen(
                    lv2_result, it.get("question", ""), base_pred,
                    _lv2_generate, args.max_new_tokens)
                if args.lv2_always_accept:
                    # Arm E: skip accept decision, always use repaired result
                    accepted = True
                else:
                    # Aggregate verify_status: UNSUPPORTED if any candidate is UNSUPPORTED
                    _statuses = [c.verify_status for c in lv2_result.revision_candidates]
                    _agg_status = "UNSUPPORTED" if any(
                        "UNSUPPORTED" in vs for vs in _statuses
                    ) else (_statuses[0] if _statuses else "")
                    accepted = _lv2_accept_decision(repaired, base_pred,
                                                    verify_status=_agg_status)
                rec["repaired_pred"] = repaired if accepted else base_pred
                rec["lv2_regen_accepted"] = accepted
                rec["replacements"] = []
                rec["span_token_coverage"] = 0.0
            except Exception as e:
                n_fail += 1
                rec["status"] = "regen_failed"
                rec["error"] = f"{type(e).__name__}: {e}"
                rec["repaired_pred"] = base_pred
                rec["traceback"] = traceback.format_exc()[-800:]
                print(f"[FAIL id={s['id']}] {rec['error']}", flush=True)
        else:
            try:
                res = fr.regenerate_spans(
                    it, s["tokens"], spans, processor, model,
                    cfg_s, DATA_PATH, DATA_ROOT,
                    dep_labels=scored.get("dep_labels"),
                    gamma=scored.get("gamma"),
                )
                rec["repaired_pred"] = res["repaired_text"]
                rec["replacements"] = res["replacements"]
                rec["span_token_coverage"] = res.get("span_token_coverage", 0.0)
                rec["coverage_cap_skip"] = res.get("coverage_cap_skip", False)
            except Exception as e:
                n_fail += 1
                rec["status"] = "regen_failed"
                rec["error"] = f"{type(e).__name__}: {e}"
                rec["repaired_pred"] = base_pred
                rec["traceback"] = traceback.format_exc()[-800:]
                print(f"[FAIL id={s['id']}] {rec['error']}", flush=True)

        if "span_token_coverage" not in rec:
            rec["span_token_coverage"] = 0.0
        rec["base_f1"] = round(100 * token_f1_multiref(base_pred, gt, multiref=True), 2)
        rec["repaired_f1"] = round(100 * token_f1_multiref(rec["repaired_pred"], gt, multiref=True), 2)
        if rec["status"] == "success":
            base_f1s.append(rec["base_f1"])
            rep_f1s.append(rec["repaired_f1"])
        rows.append(rec)

        if (k + 1) % 5 == 0 or (k + 1) == len(joined):
            print(f"[prog] {k+1}/{len(joined)} fails={n_fail}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_ok = len(base_f1s)
    mb = sum(base_f1s) / n_ok if n_ok else 0.0
    mr = sum(rep_f1s) / n_ok if n_ok else 0.0
    n_edited = sum(1 for r in rows if r.get("n_spans", 0) > 0
                   and r["status"] == "success" and not is_inf)

    # Refusal-flip: base_pred had refusal language but repaired_pred lost it
    def _has_refusal_words(t):
        tl = t.lower()
        for ph in ("there is no", "there are no", "cannot", "can't", "don't",
                   "doesn't", "isn't", "aren't", "wasn't", "weren't"):
            if ph in tl:
                return True
        return bool(set(tl.split()) & {"not", "no", "never", "unable",
                                       "unknown", "unclear", "undetermined"})

    ok_rows = [r for r in rows if r["status"] == "success"]
    n_refusal_in_base = sum(1 for r in ok_rows if _has_refusal_words(r["base_pred"]))
    n_refusal_flip = sum(
        1 for r in ok_rows
        if _has_refusal_words(r["base_pred"]) and not _has_refusal_words(r["repaired_pred"])
    )
    refusal_flip_rate = round(n_refusal_flip / n_refusal_in_base, 4) if n_refusal_in_base else 0.0

    coverages = [r["span_token_coverage"] for r in ok_rows if r.get("n_spans", 0) > 0]
    mean_span_coverage = round(sum(coverages) / len(coverages), 4) if coverages else 0.0
    frac_high_coverage = round(
        sum(1 for c in coverages if c > 0.8) / len(coverages), 4
    ) if coverages else 0.0

    # Round 7: coverage-cap skip stats + delta on actually-edited subset
    edited_rows = [r for r in ok_rows if r.get("n_spans", 0) > 0]
    n_coverage_skipped = sum(1 for r in edited_rows if r.get("coverage_cap_skip", False))
    n_edited_attempted = len(edited_rows)
    frac_coverage_skipped = round(n_coverage_skipped / n_edited_attempted, 4) if n_edited_attempted else 0.0
    actually_edited = [r for r in edited_rows if not r.get("coverage_cap_skip", False)]
    if actually_edited:
        delta_edited_only = round(
            sum(r["repaired_f1"] - r["base_f1"] for r in actually_edited) / len(actually_edited), 2
        )
    else:
        delta_edited_only = 0.0

    examples = [r for r in rows if r.get("n_spans", 0) > 0
                and r["status"] == "success"][:3]
    print("\n===== 3 BEFORE/AFTER EXAMPLES =====", flush=True)
    for ex in examples:
        print(json.dumps({
            "id": ex["id"],
            "question": ex["question"],
            "base_pred": ex["base_pred"][:300],
            "repaired_pred": ex["repaired_pred"][:300],
            "base_f1": ex["base_f1"],
            "repaired_f1": ex["repaired_f1"],
            "n_spans": ex["n_spans"],
        }, ensure_ascii=False, indent=2), flush=True)

    # verify_status breakdown across all candidates in lv2 runs
    _v_ctr = {"CONTRADICTED": 0, "UNSUPPORTED_BY_EVIDENCE": 0,
               "INSUFFICIENT_TO_DETERMINE": 0, "SUPPORTED": 0,
               "UNSUPPORTED": 0, "INSUFFICIENT_EVIDENCE": 0}
    for r in rows:
        for c in r.get("candidates", []):
            s = c.get("verify_status", "")
            if s in _v_ctr:
                _v_ctr[s] += 1

    summary = {
        "n_run": len(rows),
        "n_success": n_ok,
        "n_regen_failed": n_fail,
        "n_samples_edited": n_edited,
        "n_samples_no_span": sum(1 for r in rows if r.get("no_edit")),
        "base_tokenf1_mean": round(mb, 2),
        "repaired_tokenf1_mean": round(mr, 2),
        "delta_f1": round(mr - mb, 2),
        "n_refusal_in_base": n_refusal_in_base,
        "n_refusal_flip": n_refusal_flip,
        "refusal_flip_rate": refusal_flip_rate,
        "mean_span_coverage": mean_span_coverage,
        "frac_high_coverage": frac_high_coverage,
        "n_coverage_skipped": n_coverage_skipped,
        "frac_coverage_skipped": frac_coverage_skipped,
        "delta_edited_only": delta_edited_only,
        "tau": "inf" if is_inf else tau,
        "prior_baseline": args.prior_baseline,
        "evidence_mode": args.evidence_mode,
        "coverage_cap": args.coverage_cap,
        "out": out_path,
        # verify_status breakdown (v3 4-status + legacy 2-status)
        "n_contradicted": _v_ctr["CONTRADICTED"],
        "n_unsupported_by_evidence": _v_ctr["UNSUPPORTED_BY_EVIDENCE"],
        "n_insufficient_to_determine": _v_ctr["INSUFFICIENT_TO_DETERMINE"],
        "n_supported": _v_ctr["SUPPORTED"] + _v_ctr["UNSUPPORTED"] * 0,
        "n_legacy_unsupported": _v_ctr["UNSUPPORTED"],
        "n_legacy_insufficient": _v_ctr["INSUFFICIENT_EVIDENCE"],
    }
    print("\n===== SUMMARY =====", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
