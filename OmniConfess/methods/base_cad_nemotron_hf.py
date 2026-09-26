# -*- coding: utf-8 -*-
"""
base_cad_nemotron_hf.py — CAD (Context-Aware Decoding) baseline on
NVIDIA Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16, transformers
trust_remote_code path.

Port of methods/base_cad.py (Qwen2.5-Omni) to the Nemotron HF runtime.

Core (Shi et al., NAACL 2024, Eq.3, logit space):
  logit_CAD(y_t) = (1+alpha)*logit(y|c,x,y<t) - alpha*logit(y|x,y<t)
  alpha=0.5 (paper default, do not tune). alpha=0 == plain greedy.

Evidence c (matches the Qwen file's --cad_context auto):
  * ragtruth / pubmedqa (text): c = passage; no-ctx branch blanks the passage
    (same template, passage=""), CAD's native text setting.
  * cmm / phd / haloquest / avhbench (multimodal): c = the media; no-ctx branch
    drops all media tokens (system + question only).
  * neither passage nor media => branches identical => degenerate to greedy
    (has_cad=False).

Nemotron-specific adaptation (vs the Qwen file):
  * submodule model.language_model (NemotronHForCausalLM); NO model.thinker.
  * NO mRoPE position_ids hook. Two INDEPENDENT NemotronHHybridDynamicCache
    instances (one per branch), each stepped once per token; the backbone
    auto-creates and advances cache_position. Mamba recurrent state advances
    exactly once per branch per step (the correctness-critical invariant).
  * media fusion via nemotron_fuse.fuse_media for whichever branch carries media.

The two branches have different lengths (blanked passage / dropped media), so
each owns its own attention_mask and cache — combined only at the logit level.

Usage:
  CUDA_VISIBLE_DEVICES=7 NEMO_KEEP_CACHE=1 \
    /slow_share/tsq/venvs/nemotron-hf/bin/python base_cad_nemotron_hf.py \
      --datasets pubmedqa --data_path .../pubmedqa.json \
      --output_dir ./outputs/pubmedqa_nemotron_cad --alpha 0.5 --dry_run
"""
import os, sys, time, json, argparse, random, gc
import numpy as np
import torch
import torch.nn.functional as F

REPO = os.environ.get('NEMO_REPO', '/home/tsq/rhq/omniconfess/OmniConfess')
sys.path.insert(0, os.path.join(REPO, 'OmniConfess', 'methods'))
from shared import (
    load_dataset, load_media_for_item, resolve_media_path,
    get_system_prompt, get_user_prompt, make_result, save_result, parse_pred,
)
from shared.data_loader import load_image

import base_model_nemotron_hf as base
from nemotron_fuse import fuse_media

DEFAULT_MODEL = '/slow_share/tsq/MODELS/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16'
CAD_ALPHA = 0.5
TEXT_ONLY_DS = {'halueval', 'ragtruth', 'pubmedqa', 'summedits'}


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def build_branch_inputs(model, processor, tokenizer, item, args, system_prompt,
                        videos, audio, images, drop_passage=False, drop_media=False):
    """Return fused (inputs_embeds, input_ids, attention_mask) for one branch."""
    has_video = videos is not None and not drop_media
    has_audio = audio is not None and not drop_media
    has_image = images is not None and not drop_media
    passage = '' if drop_passage else item.get('passage')
    user_prompt = get_user_prompt(item['question'], args.datasets, args.prompt_version,
                                  passage=passage)
    media_prefix = ''
    if has_video: media_prefix += '<video>\n'
    if has_audio: media_prefix += '<so_embedding>\n'
    if has_image: media_prefix += '<image>\n'
    user_text = (media_prefix + user_prompt) if media_prefix else user_prompt
    messages = [{'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_text}]
    try:
        text = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                             tokenize=False, enable_thinking=args.enable_thinking)
    except TypeError:
        text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    proc_kwargs = dict(text=[text], return_tensors='pt')
    if has_image: proc_kwargs['images'] = images
    if has_video: proc_kwargs['videos'] = videos
    if has_audio: proc_kwargs['audio'] = audio
    inputs = processor(**proc_kwargs)
    DEVICE = next(model.parameters()).device
    inputs = {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in inputs.items()}
    for mk in ('num_patches', 'num_tokens', 'imgs_sizes'):
        inputs.pop(mk, None)
    # inference_mode: the C-RADIO video ViT inside fuse_media would otherwise
    # build an autograd graph (~2x memory) and OOM on 64-frame cmm videos.
    with torch.inference_mode():
        ie, iid, am = fuse_media(model, inputs['input_ids'], inputs.get('attention_mask'),
                                 pixel_values=inputs.get('pixel_values'),
                                 pixel_values_videos=inputs.get('pixel_values_videos'),
                                 sound_clips=inputs.get('sound_clips'))
    return ie, iid, am


def cad_generate(model, tokenizer, ie_c, am_c, ie_n, am_n,
                 max_new_tokens=256, alpha=CAD_ALPHA, record_process=True):
    """Dual-branch CAD manual decode. ie_n None => plain greedy.
    Returns (response, generated_ids, logp_list)."""
    lm = model.language_model
    device = ie_c.device
    eos_id = tokenizer.eos_token_id
    embed = lm.get_input_embeddings()

    def _combine(lc, ln):
        return lc if ln is None else (1.0 + alpha) * lc - alpha * ln

    generated, logp_list = [], []
    with torch.inference_mode():
        out_c = lm(inputs_embeds=ie_c, attention_mask=am_c, past_key_values=None, use_cache=True, logits_to_keep=1)
        pkv_c = out_c.past_key_values
        lc = out_c.logits[:, -1, :].float()
        if ie_n is not None:
            out_n = lm(inputs_embeds=ie_n, attention_mask=am_n, past_key_values=None, use_cache=True, logits_to_keep=1)
            pkv_n = out_n.past_key_values
            ln = out_n.logits[:, -1, :].float()
        else:
            pkv_n = None; ln = None

        comb = _combine(lc, ln)
        nxt = comb.argmax(dim=-1, keepdim=True); tok = nxt.item(); generated.append(tok)
        if record_process:
            logp_list.append(round(F.log_softmax(comb, dim=-1)[0, tok].item(), 4))

        for _ in range(1, max_new_tokens):
            if tok == eos_id:
                break
            if len(generated) >= 5 and len(set(generated[-5:])) == 1:
                break
            step_embed = embed(nxt)
            if am_c is not None:
                am_c = torch.cat([am_c, torch.ones(am_c.shape[0], 1, device=device, dtype=am_c.dtype)], dim=-1)
            out_c = lm(inputs_embeds=step_embed, attention_mask=am_c, past_key_values=pkv_c, use_cache=True, logits_to_keep=1)
            pkv_c = out_c.past_key_values
            lc = out_c.logits[:, -1, :].float()
            if pkv_n is not None:
                if am_n is not None:
                    am_n = torch.cat([am_n, torch.ones(am_n.shape[0], 1, device=device, dtype=am_n.dtype)], dim=-1)
                out_n = lm(inputs_embeds=step_embed, attention_mask=am_n, past_key_values=pkv_n, use_cache=True, logits_to_keep=1)
                pkv_n = out_n.past_key_values
                ln = out_n.logits[:, -1, :].float()
            comb = _combine(lc, ln)
            nxt = comb.argmax(dim=-1, keepdim=True); tok = nxt.item(); generated.append(tok)
            if record_process:
                logp_list.append(round(F.log_softmax(comb, dim=-1)[0, tok].item(), 4))

    response = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return response, generated, logp_list


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--datasets', type=str, required=True)
    p.add_argument('--data_path', type=str, required=True)
    p.add_argument('--data_root', type=str, default=None)
    p.add_argument('--output_dir', type=str, default='./outputs')
    p.add_argument('--file_name', type=str, default='base_cad_nemotron')
    p.add_argument('--time_path', type=str, default='./outputs/time')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max_new_tokens', type=int, default=256)
    p.add_argument('--alpha', type=float, default=CAD_ALPHA)
    p.add_argument('--cad_context', type=str, default='auto', choices=['auto','passage','media'])
    p.add_argument('--prompt_version', type=str, default='v1', choices=['v0','v1','v2','v3'])
    p.add_argument('--model_path', type=str, default=os.environ.get('NEMOTRON_MODEL_PATH', DEFAULT_MODEL))
    p.add_argument('--enable_thinking', action='store_true')
    p.add_argument('--record_process', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--dry_run', action='store_true')
    args = p.parse_args()
    if not args.data_root:
        args.data_root = os.path.join(REPO, 'OmniHalluBench', args.datasets) + os.sep

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True); os.makedirs(args.time_path, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.file_name + '.jsonl')

    done_ids = set()
    if os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                try: done_ids.add(json.loads(line.strip())['id'])
                except Exception: pass
        print(f'[resume] skipping {len(done_ids)} done', flush=True)

    from transformers import AutoModel, AutoProcessor
    print(f'[model] loading {args.model_path} (bf16, eager)...', flush=True)
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True,
                                      torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
                                      attn_implementation='eager')
    DEVICE = os.environ.get('NEMO_DEVICE', 'cuda:0')
    model.to(DEVICE).eval()
    base._repair_buffers(model, args.model_path, DEVICE)
    # Wrapper stays eager (no sdpa declared); force inner LLM attention to sdpa
    # so long multimodal prefills don't materialize [B,H,S,S]. CAD reads no attn.
    try:
        model.language_model.config._attn_implementation = 'sdpa'
        for _blk in model.language_model.backbone.layers:
            if getattr(_blk, 'block_type', None) == 'attention':
                _blk.mixer.config._attn_implementation = 'sdpa'
        print('[attn] inner LLM attention -> sdpa (wrapper eager)', flush=True)
    except Exception as _e:
        print(f'[attn] sdpa-on-LLM failed: {_e}', flush=True)
    if getattr(model, 'video_pruning_rate', 0):
        print(f'[repair] video_pruning_rate {model.video_pruning_rate} -> 0.0', flush=True)
        model.video_pruning_rate = 0.0
    print(f'[model] loaded in {time.time()-t0:.0f}s', flush=True)
    tokenizer = processor.tokenizer
    print(f'[cad] alpha={args.alpha} cad_context={args.cad_context}', flush=True)

    data = load_dataset(args.data_path, args.data_root)
    if args.dry_run:
        data = data[:3]; print(f'[dry_run] {len(data)} samples', flush=True)
    system_prompt = get_system_prompt(args.datasets, args.prompt_version)
    all_out = 0; start = time.time()

    for idx, item in enumerate(data):
        if item['id'] in done_ids:
            continue
        t0 = time.time()
        print(f'\n[{idx+1}/{len(data)}] id={item["id"]}', flush=True)

        json_dir = os.path.dirname(args.data_path)
        videos = images = audio = None
        if item.get('video_path'):
            vp = resolve_media_path(item['video_path'], json_dir, args.data_root)
            videos = base.load_video_frames_native(vp)
        if item.get('audio_path'):
            ap = resolve_media_path(item['audio_path'], json_dir, args.data_root)
            audio = base.load_audio_wave(ap)
        if item.get('image_path'):
            ip = resolve_media_path(item['image_path'], json_dir, args.data_root)
            images = load_image(ip)

        has_media = any(v is not None for v in (videos, audio, images))
        has_passage = bool(item.get('passage'))
        kind = args.cad_context
        if kind == 'auto':
            kind = 'passage' if has_passage else ('media' if has_media else 'none')

        response = ''; gen_ids = None; logp_list = None; input_len = 0
        ctx_kind = kind; degraded = None
        try:
            # context branch (full evidence)
            ie_c, iid_c, am_c = build_branch_inputs(
                model, processor, tokenizer, item, args, system_prompt,
                videos, audio, images)
            input_len = int(iid_c.shape[1])
            # no-context branch
            if kind == 'passage' and has_passage:
                ie_n, _, am_n = build_branch_inputs(
                    model, processor, tokenizer, item, args, system_prompt,
                    videos, audio, images, drop_passage=True)
            elif kind == 'media' and has_media:
                ie_n, _, am_n = build_branch_inputs(
                    model, processor, tokenizer, item, args, system_prompt,
                    videos, audio, images, drop_media=True)
            else:
                ie_n = am_n = None; ctx_kind = 'none'; degraded = 'no evidence to ablate'
                print('  [degraded] no evidence -> plain greedy', flush=True)
            if ie_n is not None:
                print(f'  [cad] context={ctx_kind} len_c={ie_c.shape[1]} len_n={ie_n.shape[1]}', flush=True)
            response, gen_ids, logp_list = cad_generate(
                model, tokenizer, ie_c, am_c, ie_n, am_n,
                max_new_tokens=args.max_new_tokens, alpha=args.alpha,
                record_process=args.record_process)
            all_out += len(gen_ids)
            print(f'  response[:120]: {response[:120]!r} tokens={len(gen_ids)}', flush=True)
        except Exception as e:
            import traceback; print(f'  [ERROR] {e}', flush=True); traceback.print_exc()
        elapsed = time.time() - t0
        # Dual-branch manual decode over heavy media accumulates fragmentation;
        # always free per-sample (warm-claim reservation only mattered at load).
        gc.collect(); torch.cuda.empty_cache()

        result = make_result(item, response, args.datasets, 'base_cad_nemotron', time_sec=elapsed)
        result['pred'] = parse_pred(response, args.datasets)
        result['has_cad'] = (ie_n is not None) if 'ie_n' in dir() else False
        result['cad_context'] = ctx_kind
        if degraded: result['degraded_reason'] = degraded
        result['nemotron'] = {'runtime': 'transformers-5.6.2-trust-remote-code',
                              'enable_thinking': bool(args.enable_thinking),
                              'alpha': args.alpha}
        if args.record_process:
            result['record_process'] = True
            if gen_ids is not None and logp_list is not None:
                cum = float(np.sum(logp_list)) if logp_list else 0.0
                result['token_ids'] = list(gen_ids); result['logp_list'] = logp_list
                result['cumulative_logprob'] = round(cum, 4)
                result['norm_logp'] = round(cum / (len(gen_ids) + 1e-8), 4)
                result['input_len'] = input_len; result['output_tokens'] = int(len(gen_ids))
            else:
                for k in ('token_ids','logp_list','cumulative_logprob','norm_logp'): result[k] = None
                result['input_len'] = input_len; result['output_tokens'] = 0
        save_result(result, output_path)

    total = time.time() - start
    with open(os.path.join(args.time_path, args.file_name + '.txt'), 'w') as f:
        f.write(f'total_time: {total:.2f}s\ntotal_output_tokens: {all_out}\nalpha: {args.alpha}\n')
    print(f'\n[done] total={total:.1f}s | output={output_path}', flush=True)


if __name__ == '__main__':
    main()
