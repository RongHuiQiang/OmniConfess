# -*- coding: utf-8 -*-
"""
base_vcd_nemotron_hf.py — VCD (Visual Contrastive Decoding) baseline on
NVIDIA Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16, transformers
trust_remote_code path.

Port of methods/base_vcd.py (Qwen2.5-Omni) to the Nemotron HF runtime.

Core (Leng et al., CVPR 2024, log-softmax space):
  vcd_logits = (1+alpha)*log_softmax(logit_orig) - alpha*log_softmax(logit_distorted)
  distorted = same inputs with Gaussian noise on the visual pixels
             (randn * NOISE_STD, clamp [-10,10]). alpha=0.1, noise_std=0.1
             (Qwen file defaults). Text-only samples degenerate to greedy.

Nemotron-specific: shares the CAD dual-HybridDynamicCache manual-step machinery
(inner-LLM sdpa + logits_to_keep=1 + ViT inference_mode fusion; see
base_cad_nemotron_hf.py / nemotron_fuse.py for the OOM-avoidance provenance).
The only differences vs CAD are (a) the second branch is the NOISED-media branch
(not the media-dropped branch), and (b) the combine is in log-softmax space
(VCD) rather than raw logit space (CAD).

VCD needs visual pixels, so it is only meaningful on multimodal cells
(phd image, cmm video+audio); text-only (pubmedqa/ragtruth) degenerates to
greedy and is skipped per the coordinator's Q2.5 convention.

Usage:
  CUDA_VISIBLE_DEVICES=7 \
    /slow_share/tsq/venvs/nemotron-hf/bin/python base_vcd_nemotron_hf.py \
      --datasets phd --data_path .../phd.json \
      --output_dir ./outputs/phd_nemotron_vcd --alpha 0.1 --noise_std 0.1
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
VCD_ALPHA = 0.1
NOISE_STD = 0.1
VISUAL_KEYS = {'pixel_values', 'pixel_values_videos'}


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def build_proc_inputs(model, processor, tokenizer, item, args, system_prompt,
                      videos, audio, images):
    """Return the raw processor `inputs` dict (device-placed, meta-stripped)."""
    has_video = videos is not None
    has_audio = audio is not None
    has_image = images is not None
    user_prompt = get_user_prompt(item['question'], args.datasets, args.prompt_version,
                                  passage=item.get('passage'))
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
    return inputs


def distort_visual(inputs, noise_std=NOISE_STD):
    """Return a copy of `inputs` with Gaussian noise on the visual pixel tensors.
    Returns None if there is no visual key (text-only)."""
    has_visual = any(k in inputs and inputs[k] is not None for k in VISUAL_KEYS)
    if not has_visual:
        return None
    dist = {}
    for k, v in inputs.items():
        if not torch.is_tensor(v):
            dist[k] = v
        elif k in VISUAL_KEYS:
            noise = torch.randn_like(v, dtype=v.dtype) * noise_std
            dist[k] = torch.clamp(v + noise, -10.0, 10.0)
        else:
            dist[k] = v
    return dist


def fuse(model, inputs):
    with torch.inference_mode():
        return fuse_media(model, inputs['input_ids'], inputs.get('attention_mask'),
                          pixel_values=inputs.get('pixel_values'),
                          pixel_values_videos=inputs.get('pixel_values_videos'),
                          sound_clips=inputs.get('sound_clips'))


def vcd_generate(model, tokenizer, ie_o, am_o, ie_d, am_d,
                 max_new_tokens=256, alpha=VCD_ALPHA, record_process=True):
    """Dual-branch VCD manual decode (log-softmax contrast). ie_d None => greedy."""
    lm = model.language_model
    device = ie_o.device
    eos_id = tokenizer.eos_token_id
    embed = lm.get_input_embeddings()

    def _combine(lo, ld):
        if ld is None:
            return lo
        return (1.0 + alpha) * F.log_softmax(lo, dim=-1) - alpha * F.log_softmax(ld, dim=-1)

    generated, logp_list = [], []
    with torch.inference_mode():
        out_o = lm(inputs_embeds=ie_o, attention_mask=am_o, past_key_values=None, use_cache=True, logits_to_keep=1)
        pkv_o = out_o.past_key_values
        lo = out_o.logits[:, -1, :].float()
        if ie_d is not None:
            out_d = lm(inputs_embeds=ie_d, attention_mask=am_d, past_key_values=None, use_cache=True, logits_to_keep=1)
            pkv_d = out_d.past_key_values
            ld = out_d.logits[:, -1, :].float()
        else:
            pkv_d = None; ld = None

        comb = _combine(lo, ld)
        nxt = comb.argmax(dim=-1, keepdim=True); tok = nxt.item(); generated.append(tok)
        if record_process:
            logp_list.append(round(F.log_softmax(comb, dim=-1)[0, tok].item(), 4))

        for _ in range(1, max_new_tokens):
            if tok == eos_id:
                break
            if len(generated) >= 5 and len(set(generated[-5:])) == 1:
                break
            step_embed = embed(nxt)
            if am_o is not None:
                am_o = torch.cat([am_o, torch.ones(am_o.shape[0], 1, device=device, dtype=am_o.dtype)], dim=-1)
            out_o = lm(inputs_embeds=step_embed, attention_mask=am_o, past_key_values=pkv_o, use_cache=True, logits_to_keep=1)
            pkv_o = out_o.past_key_values
            lo = out_o.logits[:, -1, :].float()
            if pkv_d is not None:
                if am_d is not None:
                    am_d = torch.cat([am_d, torch.ones(am_d.shape[0], 1, device=device, dtype=am_d.dtype)], dim=-1)
                out_d = lm(inputs_embeds=step_embed, attention_mask=am_d, past_key_values=pkv_d, use_cache=True, logits_to_keep=1)
                pkv_d = out_d.past_key_values
                ld = out_d.logits[:, -1, :].float()
            comb = _combine(lo, ld)
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
    p.add_argument('--file_name', type=str, default='base_vcd_nemotron')
    p.add_argument('--time_path', type=str, default='./outputs/time')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max_new_tokens', type=int, default=256)
    p.add_argument('--alpha', type=float, default=VCD_ALPHA)
    p.add_argument('--noise_std', type=float, default=NOISE_STD)
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
    print(f'[vcd] alpha={args.alpha} noise_std={args.noise_std}', flush=True)

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

        response = ''; gen_ids = None; logp_list = None; input_len = 0
        has_vcd = False; degraded = None
        try:
            inputs = build_proc_inputs(model, processor, tokenizer, item, args,
                                       system_prompt, videos, audio, images)
            input_len = int(inputs['input_ids'].shape[1])
            ie_o, iid_o, am_o = fuse(model, inputs)
            dist = distort_visual(inputs, args.noise_std)
            if dist is not None:
                ie_d, _, am_d = fuse(model, dist)
                has_vcd = True
                print(f'  [vcd] noised-visual branch active len={ie_o.shape[1]}', flush=True)
            else:
                ie_d = am_d = None; degraded = 'text-only: no visual to distort'
                print('  [degraded] no visual -> plain greedy', flush=True)
            response, gen_ids, logp_list = vcd_generate(
                model, tokenizer, ie_o, am_o, ie_d, am_d,
                max_new_tokens=args.max_new_tokens, alpha=args.alpha,
                record_process=args.record_process)
            del ie_o, ie_d, inputs, dist
            all_out += len(gen_ids)
            print(f'  response[:120]: {response[:120]!r} tokens={len(gen_ids)}', flush=True)
        except Exception as e:
            import traceback; print(f'  [ERROR] {e}', flush=True); traceback.print_exc()
        elapsed = time.time() - t0
        gc.collect(); torch.cuda.empty_cache()

        result = make_result(item, response, args.datasets, 'base_vcd_nemotron', time_sec=elapsed)
        result['pred'] = parse_pred(response, args.datasets)
        result['has_vcd'] = has_vcd
        if degraded: result['degraded_reason'] = degraded
        result['nemotron'] = {'runtime': 'transformers-5.6.2-trust-remote-code',
                              'enable_thinking': bool(args.enable_thinking),
                              'alpha': args.alpha, 'noise_std': args.noise_std}
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
        f.write(f'total_time: {total:.2f}s\ntotal_output_tokens: {all_out}\n')
        f.write(f'alpha: {args.alpha}\nnoise_std: {args.noise_std}\n')
    print(f'\n[done] total={total:.1f}s | output={output_path}', flush=True)


if __name__ == '__main__':
    main()
