# -*- coding: utf-8 -*-
"""
base_vcd.py -- VCD (Visual Contrastive Decoding) baseline on Qwen2.5-Omni.

Core formula (VCD, CVPR 2024):
  next_token_logits = (1+a) * log_softmax(logit_full) - a * log_softmax(logit_distorted)

Implementation notes:
  - the distorted image/video adds Gaussian noise (std = 0.1) to pixel_values
  - a step-by-step forward loop maintains two KV caches, full and distorted
  - text-only samples have no visual contrast and degenerate to plain greedy
  - the interface matches base_model.py across all datasets

Usage:
    python base_vcd.py \
        --datasets phd --data_path ./OmniHalluBench/phd/phd.json \
        --output_dir ./outputs --file_name base_vcd_phd_v0 \
        --max_samples 20
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

# --- VCD parameters ---
VCD_ALPHA    = 0.1
NOISE_STD    = 0.1
TEXT_ONLY_DS = {"halueval", "ragtruth", "pubmedqa"}


# --- Build the distorted input ---

def make_distorted_inputs(inputs):
    """
    Add Gaussian noise to the visual tensors, leave every other key untouched, and
    return the distorted input dict. Handles the pixel_values,
    pixel_values_images and pixel_values_videos keys. Returns None for text-only
    samples.
    """
    VISUAL_KEYS = {"pixel_values", "pixel_values_images", "pixel_values_videos"}
    has_visual = any(k in inputs for k in VISUAL_KEYS)
    if not has_visual:
        return None

    dist = {}
    for k, v in inputs.items():
        if not isinstance(v, torch.Tensor):
            dist[k] = v
        elif k in VISUAL_KEYS:
            noise   = torch.randn_like(v, dtype=v.dtype) * NOISE_STD
            dist[k] = torch.clamp(v + noise, -10.0, 10.0)
        else:
            dist[k] = v.clone()
    return dist


# --- VCD step-by-step generation ---

def vcd_generate(model, inputs_f, inputs_d, tokenizer,
                 max_new_tokens=256, alpha=VCD_ALPHA, record_process=True):
    """
    VCD contrastive decoding.
    - inputs_f: the full input with the original visual data
    - inputs_d: the distorted input with noised visual data; None degenerates to
      plain greedy
    Returns (response_str, generated_ids_list, logp_list).

    When record_process is on, logp_list holds the log-probability of each
    generated token under the distribution actually used for selection, i.e.
    log_softmax of the contrastive VCD logits (phi-Decoding trajectory style,
    cf. base_model.py). On text-only samples the contrast term vanishes, so
    this equals the plain greedy model distribution.

    Qwen2.5-Omni uses mRoPE, so decoding must pass position_ids explicitly and
    increment them step by step, otherwise position information is scrambled and
    the output degenerates into repetition. Concretely:
    1. a forward_pre_hook captures the position_ids used during prefill
    2. the first decoding step takes the last prefill position plus one, and every
       later step adds one
    3. the attention_mask grows by one each step and must not be ones(1, 1)
    4. generation stops early after five identical tokens in a row
    """
    device = next(model.parameters()).device

    VALID = {"input_ids", "attention_mask",
             "pixel_values_videos", "video_grid_thw", "video_second_per_grid",
             "pixel_values_images", "pixel_values", "image_grid_thw",
             "feature_attention_mask", "input_features"}

    def _clean(d):
        out = {k: v for k, v in d.items() if k in VALID}
        for k, v in out.items():
            if isinstance(v, torch.Tensor):
                out[k] = (v.to(dtype=torch.bfloat16, device=device)
                          if torch.is_floating_point(v) else v.to(device))
        return out

    cf = _clean(inputs_f)
    cd = _clean(inputs_d) if inputs_d is not None else None

    eos_id = tokenizer.eos_token_id
    attn_f = cf.get("attention_mask")
    attn_d = cd.get("attention_mask") if cd is not None else None

    # -- Prefill: capture position_ids (required by mRoPE) --
    captured = {}

    def _pos_hook(module, args, kwargs):
        if 'position_ids' in kwargs and kwargs['position_ids'] is not None:
            if 'position_ids' not in captured:
                captured['position_ids'] = kwargs['position_ids'].detach().clone()

    h = model.thinker.model.register_forward_pre_hook(_pos_hook, with_kwargs=True)
    try:
        with torch.inference_mode():
            out_f = model.thinker(**cf, past_key_values=None, use_cache=True)
            if cd is not None:
                out_d = model.thinker(**cd, past_key_values=None, use_cache=True)
    finally:
        h.remove()

    pkv_f = out_f.past_key_values
    pkv_d = out_d.past_key_values if cd is not None else None

    pos_ids_prefill = captured.get('position_ids')
    pos_ids_decode  = pos_ids_prefill[..., -1:] + 1 if pos_ids_prefill is not None else None

    # -- Decoding loop --
    generated = []
    logp_list = []

    with torch.inference_mode():
        logits_f = out_f.logits[:, -1, :]
        if cd is not None:
            logits_d = out_d.logits[:, -1, :]
            vcd_logits = ((1.0 + alpha) * F.log_softmax(logits_f, dim=-1)
                          -       alpha  * F.log_softmax(logits_d, dim=-1))
        else:
            vcd_logits = logits_f

        curr_id  = vcd_logits.argmax(dim=-1, keepdim=True)
        token_id = curr_id.item()
        generated.append(token_id)
        if record_process:
            lp = F.log_softmax(vcd_logits.float(), dim=-1)[0, token_id]
            logp_list.append(round(lp.item(), 4))

        for step in range(1, max_new_tokens):
            if token_id == eos_id:
                break
            if len(generated) >= 5 and len(set(generated[-5:])) == 1:
                break

            # Grow attention mask
            new_col = torch.ones(1, 1, device=device, dtype=torch.long)
            attn_f  = torch.cat([attn_f, new_col], dim=-1)

            decode_kw = dict(
                input_ids=curr_id,
                attention_mask=attn_f,
                past_key_values=pkv_f,
                use_cache=True,
            )
            if pos_ids_decode is not None:
                decode_kw['position_ids'] = pos_ids_decode

            out_f = model.thinker(**decode_kw)
            pkv_f = out_f.past_key_values

            if cd is not None:
                attn_d = torch.cat([attn_d, new_col], dim=-1)
                decode_kw_d = dict(
                    input_ids=curr_id,
                    attention_mask=attn_d,
                    past_key_values=pkv_d,
                    use_cache=True,
                )
                if pos_ids_decode is not None:
                    decode_kw_d['position_ids'] = pos_ids_decode
                out_d = model.thinker(**decode_kw_d)
                pkv_d = out_d.past_key_values

            if pos_ids_decode is not None:
                pos_ids_decode = pos_ids_decode + 1

            logits_f = out_f.logits[:, -1, :]
            if cd is not None:
                logits_d = out_d.logits[:, -1, :]
                vcd_logits = ((1.0 + alpha) * F.log_softmax(logits_f, dim=-1)
                              -       alpha  * F.log_softmax(logits_d, dim=-1))
            else:
                vcd_logits = logits_f

            curr_id  = vcd_logits.argmax(dim=-1, keepdim=True)
            token_id = curr_id.item()
            generated.append(token_id)
            if record_process:
                lp = F.log_softmax(vcd_logits.float(), dim=-1)[0, token_id]
                logp_list.append(round(lp.item(), 4))

    response = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return response, generated, logp_list


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
    parser.add_argument("--file_name",      type=str, default="base_vcd_out")
    parser.add_argument("--time_path",      type=str, default="./results/time")
    parser.add_argument("--max_samples",    type=int, default=-1)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--alpha",          type=float, default=VCD_ALPHA,
                        help="VCD contrast weight (default 0.1)")
    parser.add_argument("--noise_std",      type=float, default=NOISE_STD,
                        help="Gaussian noise std on pixel_values (default 0.1)")
    parser.add_argument("--prompt_version", type=str, default="v1")
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

    print(f"[vcd] alpha={args.alpha}, noise_std={args.noise_std}", flush=True)

    data = load_dataset(args.data_path, args.data_root)
    if args.max_samples > 0:
        data = data[: args.max_samples]
    total = len(data)
    print(f"[data] {total} samples | dataset={args.datasets}", flush=True)

    system_prompt = get_system_prompt(args.datasets, args.prompt_version)
    is_text_only  = args.datasets.lower() in TEXT_ONLY_DS
    start = time.time()

    with open(output_path, "a", encoding="utf-8") as res_f:
        for idx, item in enumerate(data):
            item_id = item.get("id", idx)
            if item_id in done_ids:
                continue

            t0 = time.time()
            print(f"\n[{idx+1}/{total}] id={item_id}", flush=True)

            try:
                pixel_values, audio_values, image_values = load_media_for_item(
                    item, args.data_path, args.data_root)

                user_prompt = get_user_prompt(
                    item["question"], args.datasets, args.prompt_version,
                    passage=item.get("passage"))
                user_content = build_user_content_with_media_tokens(
                    pixel_values, audio_values, image_values, user_prompt)
                messages = make_messages(system_prompt, user_content)

                raw_inputs = build_inputs(
                    processor, model, messages,
                    pixel_values, audio_values, image_values, args.datasets)
                clean_f = prepare_for_thinker(raw_inputs, model)

                if is_text_only:
                    clean_d = None
                    print(f"  [text-only] VCD skipped (greedy)", flush=True)
                else:
                    clean_d = make_distorted_inputs(clean_f)
                    if clean_d is None:
                        print(f"  [no visual] VCD skipped (greedy)", flush=True)

                response, gen_ids, logp_list = vcd_generate(
                    model, clean_f, clean_d, tokenizer,
                    max_new_tokens=args.max_new_tokens,
                    alpha=args.alpha,
                    record_process=args.record_process)
                input_len = clean_f["input_ids"].shape[1]

                elapsed = time.time() - t0
                # Extract clean pred from raw output (Yes/No for judgment tasks)
                pred_clean = parse_pred(response, args.datasets)
                result  = {
                    "id":           item_id,
                    "pred":         pred_clean,
                    "raw_output":   response,
                    "gt":           item.get("answer", item.get("gt", item.get("label", ""))),
                    "question":     item.get("question", item.get("input", "")),
                    "group_key":    item.get("sub_category", item.get("category", "")),
                    "sub_category": item.get("sub_category", item.get("category", "")),
                    "dataset":      args.datasets,
                    "method":       "vcd",
                    "has_vcd":      clean_d is not None,
                    "time_sec":     round(elapsed, 3),
                    "status":       "success",
                }
                if args.record_process:
                    attach_trajectory(result, gen_ids, logp_list, input_len)
                res_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                res_f.flush()
                print(f"  pred: {pred_clean!r}  raw: {response[:60]!r}  gt: {result['gt']!r}  ({elapsed:.1f}s)", flush=True)

            except torch.cuda.OutOfMemoryError as oom:
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
        tf.write(f"noise_std: {args.noise_std}\n")
    print(f"\n[done] {total} samples in {elapsed_total/60:.1f} min. Results: {output_path}", flush=True)


if __name__ == "__main__":
    main()
