# -*- coding: utf-8 -*-
"""
base_dola.py — DoLa (Decoding by Contrasting Layers) baseline on Qwen2.5-Omni.

Core idea (DoLa, ICLR 2024):
  next_token_logits = log_softmax(mature_logits) - log_softmax(premature_logits)
  where mature = final model logits, premature = early layer logits (after norm+lm_head projection)

Implementation details:
  - Extract premature layer (L8, ~29% depth) intermediate logits via DualLayerAuditor hook
  - Use step-by-step forward pass (not model.generate) for DoLa contrastive decoding
  - Supports all multimodal and text-only datasets

Usage:
    CUDA_VISIBLE_DEVICES=1 python base_dola.py \
        --datasets phd --data_path ./OmniHalluBench/PhD/all_phd.json \
        --output_dir ./results --file_name base_dola_phd_v0 \
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
    make_result, save_result,
)
from shared.eval_utils import parse_pred

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


PREMATURE_LAYER = 14



class DoLaAuditor:
    """ Qwen thinker  L{premature_layer}  logits norm+lm_head """

    def __init__(self, model, premature_layer=PREMATURE_LAYER):
        self.model         = model
        self.premature_layer = premature_layer
        self.premature_logits = None
        self._handle      = None

    def _make_hook(self):
        def _hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            with torch.no_grad():
                normed  = self.model.thinker.model.norm(hidden)
                logits  = self.model.thinker.lm_head(normed)
                self.premature_logits = logits[:, -1, :].clone()
        return _hook

    def register(self):
        layer = self.model.thinker.model.layers[self.premature_layer]
        self._handle = layer.register_forward_hook(self._make_hook())

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        self.premature_logits = None



def dola_generate(model, auditor, inputs, tokenizer,
                  max_new_tokens=256, alpha=1.0):
    """
    DoLa contrastive decoding generation.
    Returns (response_str, generated_ids_list).
    """
    VALID_KEYS = {
        "input_ids", "attention_mask",
        "pixel_values_videos", "video_grid_thw", "video_second_per_grid",
        "pixel_values_images", "pixel_values", "image_grid_thw",
        "feature_attention_mask", "input_features",
    }
    clean = {k: v for k, v in inputs.items() if k in VALID_KEYS}

    device = next(model.parameters()).device
    for k, v in clean.items():
        if isinstance(v, torch.Tensor):
            clean[k] = (v.to(dtype=torch.bfloat16, device=device)
                        if torch.is_floating_point(v) else v.to(device))

    eos_id         = tokenizer.eos_token_id
    accu_attn_mask = clean.get("attention_mask")

    captured = {}

    def _pos_hook(module, args, kwargs):
        if 'position_ids' in kwargs and kwargs['position_ids'] is not None:
            if 'position_ids' not in captured:
                captured['position_ids'] = kwargs['position_ids'].detach().clone()

    h = model.thinker.model.register_forward_pre_hook(_pos_hook, with_kwargs=True)
    auditor.register()
    try:
        with torch.inference_mode():
            out = model.thinker(**clean, past_key_values=None, use_cache=True)
    finally:
        h.remove()

    past_kv = out.past_key_values

    pos_ids_prefill = captured.get('position_ids')
    pos_ids_decode  = pos_ids_prefill[..., -1:] + 1 if pos_ids_prefill is not None else None

    generated_ids = []

    try:
        with torch.inference_mode():
            mature_logits    = out.logits[:, -1, :]
            premature_logits = auditor.premature_logits

            if premature_logits is not None:
                dola_logits = (F.log_softmax(mature_logits, dim=-1)
                               - alpha * F.log_softmax(premature_logits, dim=-1))
            else:
                dola_logits = mature_logits

            next_token = dola_logits.argmax(dim=-1, keepdim=True)
            token_id   = next_token.item()
            generated_ids.append(token_id)

            for step in range(1, max_new_tokens):
                if token_id == eos_id:
                    break
                if len(generated_ids) >= 5 and len(set(generated_ids[-5:])) == 1:
                    break

                new_col = torch.ones(1, 1, device=device, dtype=torch.long)
                accu_attn_mask = torch.cat([accu_attn_mask, new_col], dim=-1)

                decode_kw = dict(
                    input_ids=next_token,
                    attention_mask=accu_attn_mask,
                    past_key_values=past_kv,
                    use_cache=True,
                )
                if pos_ids_decode is not None:
                    decode_kw['position_ids'] = pos_ids_decode

                out = model.thinker(**decode_kw)
                past_kv = out.past_key_values

                if pos_ids_decode is not None:
                    pos_ids_decode = pos_ids_decode + 1

                mature_logits    = out.logits[:, -1, :]
                premature_logits = auditor.premature_logits

                if premature_logits is not None:
                    dola_logits = (F.log_softmax(mature_logits, dim=-1)
                                   - alpha * F.log_softmax(premature_logits, dim=-1))
                else:
                    dola_logits = mature_logits

                next_token = dola_logits.argmax(dim=-1, keepdim=True)
                token_id   = next_token.item()
                generated_ids.append(token_id)

    finally:
        auditor.remove()

    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return response, generated_ids



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets",       type=str, required=True)
    parser.add_argument("--data_path",      type=str, required=True)
    parser.add_argument("--data_root",      type=str, default="./OmniHalluBench/")
    parser.add_argument("--output_dir",     type=str, default="./results")
    parser.add_argument("--file_name",      type=str, default="base_dola_out")
    parser.add_argument("--time_path",      type=str, default="./results/time")
    parser.add_argument("--max_samples",    type=int, default=-1)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--alpha",          type=float, default=0.1,
                        help="DoLa contrast weight (default 0.1; 1.0 causes garbled output)")
    parser.add_argument("--premature_layer",type=int, default=PREMATURE_LAYER,
                        help="Which layer index as premature (default 8)")
    parser.add_argument("--prompt_version", type=str, default="v1")
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

    auditor = DoLaAuditor(model, premature_layer=args.premature_layer)
    print(f"[dola] premature_layer=L{args.premature_layer}, alpha={args.alpha}", flush=True)

    data = load_dataset(args.data_path, args.data_root)
    if args.max_samples > 0:
        data = data[: args.max_samples]
    total = len(data)
    print(f"[data] {total} samples, dataset={args.datasets}", flush=True)

    system_prompt = get_system_prompt(args.datasets, args.prompt_version)
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
                clean = prepare_for_thinker(raw_inputs, model)

                response, gen_ids = dola_generate(
                    model, auditor, clean, tokenizer,
                    max_new_tokens=args.max_new_tokens,
                    alpha=args.alpha)

                elapsed = time.time() - t0
                # Extract clean pred from raw output (Yes/No for judgment tasks)
                pred_clean = parse_pred(response, args.datasets)
                result = {
                    "id":           item_id,
                    "pred":         pred_clean,
                    "raw_output":   response,
                    "gt":           item.get("answer", item.get("gt", item.get("label", ""))),
                    "question":     item.get("question", item.get("input", "")),
                    "group_key":    item.get("sub_category", item.get("category", "")),
                    "sub_category": item.get("sub_category", item.get("category", "")),
                    "dataset":      args.datasets,
                    "method":       "dola",
                    "time_sec":     round(elapsed, 3),
                    "status":       "success",
                }
                res_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                res_f.flush()
                print(f"  pred: {pred_clean!r}  raw: {response[:60]!r}  gt: {result['gt']!r}  ({elapsed:.1f}s)", flush=True)

            except torch.cuda.OutOfMemoryError as oom:
                torch.cuda.empty_cache(); gc.collect()
                print(f"  [OOM] id={item_id}", flush=True)
                with open(error_path, "a") as ef:
                    ef.write(f"{time.strftime('%H:%M:%S')} | id={item_id} | OOM\n")
                res_f.write(json.dumps({"id": item_id, "status": "OOM", "error": "CUDA OOM"}, ensure_ascii=False) + "\n")
                res_f.flush()

            except Exception as e:
                import traceback
                print(f"  [ERR] id={item_id}: {repr(e)}", flush=True)
                with open(error_path, "a") as ef:
                    ef.write(f"{time.strftime('%H:%M:%S')} | id={item_id} | {repr(e)}\n")
                    ef.write(traceback.format_exc() + "\n")
                res_f.write(json.dumps({"id": item_id, "status": "failed", "error": repr(e)[:200]}, ensure_ascii=False) + "\n")
                res_f.flush()

            finally:
                gc.collect()
                torch.cuda.empty_cache()

    elapsed_total = time.time() - start
    with open(os.path.join(args.time_path, f"{args.file_name}.txt"), "w") as tf:
        tf.write(f"time: {elapsed_total:.1f}s\n")
        tf.write(f"total_samples: {total}\n")
        tf.write(f"premature_layer: {args.premature_layer}\n")
        tf.write(f"alpha: {args.alpha}\n")
    print(f"\n[done] {total} samples in {elapsed_total/60:.1f} min. Results: {output_path}", flush=True)


if __name__ == "__main__":
    main()
