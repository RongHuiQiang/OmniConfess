# -*- coding: utf-8 -*-
"""
base_cad.py -- CAD (Context-Aware Decoding) baseline on Qwen2.5-Omni-7B.

Reference
---------
Weijia Shi, Xiaochuang Han, Mike Lewis, Yulia Tsvetkov, Luke Zettlemoyer,
Scott Wen-tau Yih. "Trusting Your Evidence: Hallucinate Less with
Context-aware Decoding." NAACL 2024 (Short Papers), pp. 783-791.
arXiv:2305.14739.  Official code: github.com/xhan77/context-aware-decoding

Core formula (paper Eq. 3, in logit space)
------------------------------------------
  logit_CAD(y_t) = (1+alpha)*logit(y_t | c, x, y_<t) - alpha*logit(y_t | x, y_<t)

  c is the evidence/context, x the query, and alpha >= 0 controls the contrast
  strength. alpha = 0 degenerates to ordinary greedy decoding.

Hyperparameter provenance
-------------------------
  alpha = 0.5
    From Shi et al., NAACL 2024, whose main experiments use alpha = 0.5
    ("we set alpha = 0.5" in Section 3); this is also the default in the
    official repository and in the published reproductions. The default here
    is the same 0.5 and nothing was tuned.

  The paper combines the two branches linearly in logit space, without first
  applying log_softmax, and this implementation follows the paper. That differs
  from the log_softmax formulation used by VCD and AVCD, which is why base_cad.py
  does not share a formula implementation with base_vcd.py. (Since argmax is
  insensitive to subtracting a constant from either branch, the logit-space and
  log-softmax-space combinations differ only when the two partition functions are
  unequal; the CAD paper's form is used here.)

What counts as "evidence" on each dataset
-----------------------------------------
  CAD is originally a text-only method in which c is a retrieved document. This
  benchmark includes multimodal subsets, so evidence is defined uniformly as the
  grounding evidence, determined automatically under --cad_context auto:

    ragtruth / pubmedqa (text only)   c = the passage
        The no-evidence branch keeps the same template with the passage blanked,
        which preserves prompt alignment. This is the standard CAD ablation:
        remove c only, leave the instruction untouched. This is CAD's native
        setting and matches the paper exactly.

    cmm / phd / haloquest / avhbench (multimodal)   c = the visual/audio input
        The no-evidence branch drops all media tokens and keeps only the system
        prompt and the question. This extends the original paper, which does not
        cover the multimodal case; the evidence type actually used is recorded in
        the cad_context field of each result.

Degenerate behaviour
--------------------
  If a sample has neither a passage nor any media, which should not occur but the
  code must handle, the two branches are identical and CAD reduces exactly to
  greedy decoding. In that case the second forward pass is skipped and the output
  is marked has_cad = False, cad_context = "none",
  degraded_reason = "no evidence to ablate".
  The text-only subsets (pubmedqa, ragtruth) never take this path: they have a
  passage, so CAD is fully active on them. This is unlike VCD and AVCD, which do
  degenerate on text-only data.

Usage:
    python base_cad.py \
        --datasets ragtruth --data_path ./OmniHalluBench/ragtruth/ragtruth.json \
        --output_dir ./outputs --file_name base_cad_ragtruth_v0 \
        --prompt_version v0 --max_samples 2
"""
import os, sys, json, time, argparse, gc
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from shared import (
    load_dataset, load_media_for_item,
    load_model, build_inputs, prepare_for_thinker,
    build_user_content_with_media_tokens, make_messages,
    get_system_prompt, get_user_prompt,
)
from shared.eval_utils import parse_pred

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# --- CAD hyperparameters (values recommended by the paper; do not tune) ---
CAD_ALPHA = 0.5

TEXT_ONLY_DS = {"halueval", "ragtruth", "pubmedqa", "summedits"}

VALID_KEYS = {
    "input_ids", "attention_mask",
    "pixel_values_videos", "video_grid_thw", "video_second_per_grid",
    "pixel_values_images", "pixel_values", "image_grid_thw",
    "feature_attention_mask", "input_features",
}


# --- Shared: single-branch prefill (captures the mRoPE position_ids) ---

def _clean_to_device(d, device):
    out = {k: v for k, v in d.items() if k in VALID_KEYS}
    for k, v in out.items():
        if isinstance(v, torch.Tensor):
            out[k] = (v.to(dtype=torch.bfloat16, device=device)
                      if torch.is_floating_point(v) else v.to(device))
    return out


def prefill_branch(model, inputs, device):
    """
    Prefill one branch and return (logits_last, past_kv, attn_mask, pos_ids_next).

    Qwen2.5-Omni uses mRoPE, so during decoding the position_ids must be passed
    explicitly and incremented step by step; otherwise position information is
    scrambled and the output degenerates into repetition. A forward_pre_hook
    captures the position_ids actually used during prefill, and the last column
    plus one becomes the decoding start. The same pattern is used in base_vcd.py
    and base_dola.py.
    """
    clean = _clean_to_device(inputs, device)
    captured = {}

    def _pos_hook(module, args, kwargs):
        if kwargs.get("position_ids") is not None and "position_ids" not in captured:
            captured["position_ids"] = kwargs["position_ids"].detach().clone()

    h = model.thinker.model.register_forward_pre_hook(_pos_hook, with_kwargs=True)
    try:
        with torch.inference_mode():
            out = model.thinker(**clean, past_key_values=None, use_cache=True)
    finally:
        h.remove()

    pos_prefill = captured.get("position_ids")
    pos_next = pos_prefill[..., -1:] + 1 if pos_prefill is not None else None
    return out.logits[:, -1, :], out.past_key_values, clean.get("attention_mask"), pos_next


def _decode_step(model, token_id_tensor, attn_mask, past_kv, pos_ids, device):
    """Single decoding step; returns (logits_last, new_past_kv, new_attn_mask)."""
    attn_mask = torch.cat(
        [attn_mask, torch.ones(1, 1, device=device, dtype=torch.long)], dim=-1)
    kw = dict(input_ids=token_id_tensor, attention_mask=attn_mask,
              past_key_values=past_kv, use_cache=True)
    if pos_ids is not None:
        kw["position_ids"] = pos_ids
    out = model.thinker(**kw)
    return out.logits[:, -1, :], out.past_key_values, attn_mask


# --- CAD step-by-step generation ---

def cad_generate(model, inputs_ctx, inputs_noctx, tokenizer,
                 max_new_tokens=256, alpha=CAD_ALPHA, record_process=True):
    """
    CAD contrastive decoding.
      inputs_ctx   : input with evidence (context)
      inputs_noctx : input without evidence; None degenerates to plain greedy
    Returns (response_str, generated_ids, logp_list).

    When record_process is on, logp_list holds the log-probability of each
    generated token under the distribution actually used for selection, i.e.
    log_softmax of the combined CAD logits (phi-Decoding trajectory style, cf.
    base_model.py); with inputs_noctx=None this is the plain greedy model
    distribution.

    Note that the two branches have different sequence lengths, since removing the
    passage or the media makes one shorter, so each maintains its own
    attention_mask and position_ids rather than sharing them as base_vcd.py does.
    """
    device = next(model.parameters()).device
    eos_id = tokenizer.eos_token_id

    logits_c, pkv_c, attn_c, pos_c = prefill_branch(model, inputs_ctx, device)
    if inputs_noctx is not None:
        logits_n, pkv_n, attn_n, pos_n = prefill_branch(model, inputs_noctx, device)
    else:
        logits_n = pkv_n = attn_n = pos_n = None

    def _combine(lc, ln):
        if ln is None:
            return lc
        return (1.0 + alpha) * lc - alpha * ln

    generated = []
    logp_list = []
    with torch.inference_mode():
        comb = _combine(logits_c, logits_n)
        cur = comb.argmax(dim=-1, keepdim=True)
        tok = cur.item()
        generated.append(tok)
        if record_process:
            lp = F.log_softmax(comb.float(), dim=-1)[0, tok]
            logp_list.append(round(lp.item(), 4))

        for _ in range(1, max_new_tokens):
            if tok == eos_id:
                break
            if len(generated) >= 5 and len(set(generated[-5:])) == 1:
                break

            logits_c, pkv_c, attn_c = _decode_step(
                model, cur, attn_c, pkv_c, pos_c, device)
            if pos_c is not None:
                pos_c = pos_c + 1

            if pkv_n is not None:
                logits_n, pkv_n, attn_n = _decode_step(
                    model, cur, attn_n, pkv_n, pos_n, device)
                if pos_n is not None:
                    pos_n = pos_n + 1

            comb = _combine(logits_c, logits_n)
            cur = comb.argmax(dim=-1, keepdim=True)
            tok = cur.item()
            generated.append(tok)
            if record_process:
                lp = F.log_softmax(comb.float(), dim=-1)[0, tok]
                logp_list.append(round(lp.item(), 4))

    return (tokenizer.decode(generated, skip_special_tokens=True).strip(),
            generated, logp_list)


# --- Build the no-evidence input ---

def build_noctx_inputs(processor, model, item, args, system_prompt,
                       pixel_values, audio_values, image_values, cad_context):
    """
    Remove the kind of evidence selected by cad_context and return
    (clean_inputs or None, actual_context_kind). None means there is no evidence
    to remove, so CAD degenerates to greedy.
    """
    has_media   = any(v is not None for v in (pixel_values, audio_values, image_values))
    has_passage = bool(item.get("passage"))

    kind = cad_context
    if kind == "auto":
        kind = "passage" if has_passage else ("media" if has_media else "none")

    if kind == "none" or (kind == "passage" and not has_passage) \
                      or (kind == "media" and not has_media):
        return None, "none"

    if kind == "passage":
        user_prompt = get_user_prompt(
            item["question"], args.datasets, args.prompt_version, passage="")
        user_content = build_user_content_with_media_tokens(
            pixel_values, audio_values, image_values, user_prompt)
        messages = make_messages(system_prompt, user_content)
        raw = build_inputs(processor, model, messages,
                           pixel_values, audio_values, image_values, args.datasets)
    else:  # media
        user_prompt = get_user_prompt(
            item["question"], args.datasets, args.prompt_version,
            passage=item.get("passage"))
        user_content = build_user_content_with_media_tokens(
            None, None, None, user_prompt)
        messages = make_messages(system_prompt, user_content)
        raw = build_inputs(processor, model, messages,
                           None, None, None, args.datasets)

    return prepare_for_thinker(raw, model), kind


# --- Main ---

def attach_trajectory(result, gen_ids, logp_list, input_len):
    """Append phi-Decoding style per-token trajectory fields (see base_model.py)."""
    result["record_process"] = True
    if gen_ids:
        cum = float(sum(logp_list)) if logp_list else 0.0
        result["token_ids"] = list(gen_ids)
        result["logp_list"] = list(logp_list)
        result["cumulative_logprob"] = round(cum, 4)
        result["norm_logp"] = round(cum / (len(gen_ids) + 1e-8), 4)
        result["input_len"] = int(input_len) if input_len is not None else None
        result["output_tokens"] = len(gen_ids)
    else:
        for k in ("token_ids", "logp_list", "cumulative_logprob",
                  "norm_logp", "input_len"):
            result[k] = None
        result["output_tokens"] = 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets",       type=str, required=True)
    parser.add_argument("--data_path",      type=str, required=True)
    parser.add_argument("--data_root",      type=str, default="./OmniHalluBench/")
    parser.add_argument("--output_dir",     type=str, default="./results")
    parser.add_argument("--file_name",      type=str, default="base_cad_out")
    parser.add_argument("--time_path",      type=str, default="./results/time")
    parser.add_argument("--max_samples",    type=int, default=-1)
    parser.add_argument("--start_index",    type=int, default=0,
                        help="Start from sample start_index (0-based); combine with --max_samples to take a range. "
                             "It also makes the fallback idx global for datasets without an id field "
                             "(cmm/phd/avhbench), so it does not collide with an earlier run's 0..199.")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--alpha",          type=float, default=CAD_ALPHA,
                        help="CAD contrast weight (paper default 0.5, Shi et al. NAACL 2024)")
    parser.add_argument("--cad_context",    type=str, default="auto",
                        choices=["auto", "passage", "media"],
                        help="which evidence to ablate for the no-context branch")
    parser.add_argument("--prompt_version", type=str, default="v0")
    parser.add_argument("--record_process", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Record per-token trajectory (token ids, per-token "
                             "logprobs, cumulative/norm logp) alongside the response")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.time_path,  exist_ok=True)

    output_path = os.path.join(args.output_dir, args.file_name + ".jsonl")
    error_path  = os.path.join(args.output_dir, f"ERROR-{args.file_name}.log")

    done_ids = set()
    if os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line.strip())["id"])
                except Exception:
                    pass
        print(f"[resume] skipping {len(done_ids)} done samples", flush=True)

    print("[model] loading Qwen2.5-Omni...", flush=True)
    processor, model = load_model()
    tokenizer = processor.tokenizer

    print(f"[cad] alpha={args.alpha}  cad_context={args.cad_context}", flush=True)

    data = load_dataset(args.data_path, args.data_root)
    if args.start_index > 0:
        data = data[args.start_index:]
    if args.max_samples > 0:
        data = data[: args.max_samples]
    total = len(data)
    print(f"[data] {total} samples | dataset={args.datasets}", flush=True)

    system_prompt = get_system_prompt(args.datasets, args.prompt_version)
    start = time.time()

    with open(output_path, "a", encoding="utf-8") as res_f:
        for idx, item in enumerate(data, start=args.start_index):
            item_id = item.get("id", idx)
            if item_id in done_ids:
                continue

            t0 = time.time()
            print(f"\n[{idx+1}/{args.start_index+total}] id={item_id}", flush=True)

            try:
                pixel_values, audio_values, image_values = load_media_for_item(
                    item, args.data_path, args.data_root)

                # -- Branch with evidence --
                user_prompt = get_user_prompt(
                    item["question"], args.datasets, args.prompt_version,
                    passage=item.get("passage"))
                user_content = build_user_content_with_media_tokens(
                    pixel_values, audio_values, image_values, user_prompt)
                messages = make_messages(system_prompt, user_content)
                raw_inputs = build_inputs(
                    processor, model, messages,
                    pixel_values, audio_values, image_values, args.datasets)
                clean_ctx = prepare_for_thinker(raw_inputs, model)

                # -- Branch without evidence --
                clean_noctx, ctx_kind = build_noctx_inputs(
                    processor, model, item, args, system_prompt,
                    pixel_values, audio_values, image_values, args.cad_context)

                degraded_reason = None
                if clean_noctx is None:
                    degraded_reason = "no evidence to ablate"
                    print("  [degraded] no evidence → plain greedy", flush=True)
                else:
                    print(f"  [cad] context={ctx_kind} | "
                          f"len_ctx={clean_ctx['input_ids'].shape[1]} "
                          f"len_noctx={clean_noctx['input_ids'].shape[1]}", flush=True)

                response, gen_ids, logp_list = cad_generate(
                    model, clean_ctx, clean_noctx, tokenizer,
                    max_new_tokens=args.max_new_tokens, alpha=args.alpha,
                    record_process=args.record_process)
                input_len = clean_ctx["input_ids"].shape[1]

                elapsed = time.time() - t0
                pred_clean = parse_pred(response, args.datasets)
                result = {
                    "id":           item_id,
                    "pred":         pred_clean,
                    "raw_output":   response,
                    "gt":           item.get("gt", item.get("answer", item.get("label", ""))),
                    "question":     item.get("question", item.get("input", "")),
                    "group_key":    item.get("group_key", item.get("sub_category", "Default")),
                    "sub_category": item.get("group_key", item.get("sub_category", "Default")),
                    "dataset":      args.datasets,
                    "method":       "cad",
                    "has_cad":      clean_noctx is not None,
                    "cad_context":  ctx_kind,
                    "time_sec":     round(elapsed, 3),
                    "status":       "success",
                }
                if degraded_reason:
                    result["degraded_reason"] = degraded_reason
                if args.record_process:
                    attach_trajectory(result, gen_ids, logp_list, input_len)
                res_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                res_f.flush()
                print(f"  pred: {pred_clean!r}  raw: {response[:60]!r}  "
                      f"gt: {result['gt']!r}  ({elapsed:.1f}s)", flush=True)

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); gc.collect()
                print(f"  [OOM] id={item_id}", flush=True)
                with open(error_path, "a") as ef:
                    ef.write(f"{time.strftime('%H:%M:%S')} | id={item_id} | OOM\n")
                _err = {"id": item_id, "status": "OOM", "error": "CUDA OOM"}
                if args.record_process:
                    attach_trajectory(_err, None, None, None)
                res_f.write(json.dumps(_err, ensure_ascii=False) + "\n")
                res_f.flush()

            except Exception as e:
                import traceback
                torch.cuda.empty_cache(); gc.collect()
                print(f"  [ERR] id={item_id}: {repr(e)}", flush=True)
                with open(error_path, "a") as ef:
                    ef.write(f"{time.strftime('%H:%M:%S')} | id={item_id} | {repr(e)}\n")
                    ef.write(traceback.format_exc() + "\n")
                _err = {"id": item_id, "status": "failed", "error": repr(e)[:200]}
                if args.record_process:
                    attach_trajectory(_err, None, None, None)
                res_f.write(json.dumps(_err, ensure_ascii=False) + "\n")
                res_f.flush()

            finally:
                gc.collect(); torch.cuda.empty_cache()

    elapsed_total = time.time() - start
    with open(os.path.join(args.time_path, f"{args.file_name}.txt"), "w") as tf:
        tf.write(f"time: {elapsed_total:.1f}s\n")
        tf.write(f"total_samples: {total}\n")
        tf.write(f"alpha: {args.alpha}\n")
        tf.write(f"cad_context: {args.cad_context}\n")
    print(f"\n[done] {total} samples in {elapsed_total/60:.1f} min. "
          f"Results: {output_path}", flush=True)


if __name__ == "__main__":
    main()
