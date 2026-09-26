# -*- coding: utf-8 -*-
"""
base_dola_nemotron_hf.py — DoLa (Decoding by Contrasting Layers) baseline on
NVIDIA Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16, transformers
trust_remote_code path.

Port of methods/base_dola.py (Qwen2.5-Omni) to the Nemotron HF runtime.

Core (DoLa, ICLR 2024):
  dola_logits = log_softmax(mature_logits) - alpha * log_softmax(premature_logits)
  premature = an intermediate layer's residual, projected through the final
  RMSNorm (backbone.norm_f) + lm_head.

Nemotron-specific adaptation (vs the Qwen file):
  * submodule path: model.language_model (NemotronHForCausalLM), whose LLM stack
    is model.language_model.backbone.layers[k], final norm .backbone.norm_f,
    projection .lm_head. There is NO model.thinker.
  * NO mRoPE position_ids hook: Mamba2 layers advance via cache_position (auto),
    only the 6 attention layers use partial rotary internally; manual stepping
    passes inputs_embeds + the returned NemotronHHybridDynamicCache and lets the
    backbone compute cache_position.
  * premature layer must be a residual point in the 52-layer hybrid stack. The
    hidden state after ANY block is a full-width residual, so any index works;
    the DEFAULT is a Mamba (M) layer (L28) to avoid landing exactly on an
    attention layer (Qwen's equi-scaled L26 == attn index). Sweep --premature_layer
    and --alpha on a dev slice first (garbled output => wrong layer/alpha, drop it,
    do NOT run full).
  * media fusion (image/video/sound -> inputs_embeds) is replicated via
    nemotron_fuse.fuse_media, since manual stepping bypasses the wrapper generate
    that normally fuses. Text-only cells (pubmedqa/ragtruth) skip fusion.

Usage:
  CUDA_VISIBLE_DEVICES=7 NEMO_KEEP_CACHE=1 \
    /slow_share/tsq/venvs/nemotron-hf/bin/python base_dola_nemotron_hf.py \
      --datasets pubmedqa --data_path .../pubmedqa.json \
      --output_dir ./outputs/pubmedqa_nemotron_dola \
      --premature_layer 28 --alpha 0.1 --dry_run
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
PREMATURE_LAYER = 28   # a Mamba (M) residual layer; sweep on dev before full


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


class DoLaAuditor:
    """Capture the premature layer's residual, project via backbone.norm_f + lm_head."""
    def __init__(self, model, premature_layer=PREMATURE_LAYER):
        self.lm = model.language_model
        self.premature_layer = premature_layer
        self.premature_logits = None
        self._handle = None

    def _hook(self, module, inp, output):
        hidden = output[0] if isinstance(output, tuple) else output
        with torch.no_grad():
            # project ONLY the last position (full-seq lm_head over a long video
            # prompt would allocate [B, N, 131072] and OOM on cmm/haloquest)
            last = self.lm.backbone.norm_f(hidden[:, -1:, :])
            logits = self.lm.lm_head(last)
            self.premature_logits = logits[:, -1, :].float().clone()

    def register(self):
        layer = self.lm.backbone.layers[self.premature_layer]
        self._handle = layer.register_forward_hook(self._hook)

    def remove(self):
        if self._handle is not None:
            self._handle.remove(); self._handle = None
        self.premature_logits = None


def dola_generate(model, auditor, inputs_embeds, attention_mask, tokenizer,
                  max_new_tokens=256, alpha=0.1, record_process=True):
    """Manual step-by-step DoLa decode over the NemotronH hybrid cache.

    Returns (response, generated_ids, logp_list). No F.log_softmax over the
    contrast (already in log-ratio space); argmax taken directly, matching the
    Qwen file. Early stop after 5 identical tokens.
    """
    lm = model.language_model
    device = inputs_embeds.device
    eos_id = tokenizer.eos_token_id
    embed = lm.get_input_embeddings()

    auditor.register()
    generated, logp_list = [], []
    try:
        with torch.inference_mode():
            out = lm(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                     past_key_values=None, use_cache=True, logits_to_keep=1)
            past_kv = out.past_key_values
            mature = out.logits[:, -1, :].float()
            prem = auditor.premature_logits
            dl = (F.log_softmax(mature, dim=-1) - alpha * F.log_softmax(prem, dim=-1)) \
                 if prem is not None else mature
            nxt = dl.argmax(dim=-1, keepdim=True)
            tok = nxt.item(); generated.append(tok)
            if record_process:
                logp_list.append(round(F.log_softmax(dl.float(), dim=-1)[0, tok].item(), 4))

            cur_mask = attention_mask
            for _ in range(1, max_new_tokens):
                if tok == eos_id:
                    break
                if len(generated) >= 5 and len(set(generated[-5:])) == 1:
                    break
                if cur_mask is not None:
                    cur_mask = torch.cat(
                        [cur_mask, torch.ones(cur_mask.shape[0], 1, device=device, dtype=cur_mask.dtype)], dim=-1)
                step_embed = embed(nxt)
                out = lm(inputs_embeds=step_embed, attention_mask=cur_mask,
                         past_key_values=past_kv, use_cache=True, logits_to_keep=1)
                past_kv = out.past_key_values
                mature = out.logits[:, -1, :].float()
                prem = auditor.premature_logits
                dl = (F.log_softmax(mature, dim=-1) - alpha * F.log_softmax(prem, dim=-1)) \
                     if prem is not None else mature
                nxt = dl.argmax(dim=-1, keepdim=True)
                tok = nxt.item(); generated.append(tok)
                if record_process:
                    logp_list.append(round(F.log_softmax(dl.float(), dim=-1)[0, tok].item(), 4))
    finally:
        auditor.remove()

    response = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return response, generated, logp_list


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--datasets', type=str, required=True)
    p.add_argument('--data_path', type=str, required=True)
    p.add_argument('--data_root', type=str, default=None)
    p.add_argument('--output_dir', type=str, default='./outputs')
    p.add_argument('--file_name', type=str, default='base_dola_nemotron')
    p.add_argument('--time_path', type=str, default='./outputs/time')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max_new_tokens', type=int, default=256)
    p.add_argument('--alpha', type=float, default=0.1,
                   help='DoLa contrast weight (Qwen file default 0.1; 1.0 garbles)')
    p.add_argument('--premature_layer', type=int, default=PREMATURE_LAYER,
                   help='premature residual layer (default 28 = Mamba layer)')
    p.add_argument('--prompt_version', type=str, default='v1', choices=['v0','v1','v2','v3'])
    p.add_argument('--model_path', type=str,
                   default=os.environ.get('NEMOTRON_MODEL_PATH', DEFAULT_MODEL))
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
    model = AutoModel.from_pretrained(
        args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, attn_implementation='eager')
    DEVICE = os.environ.get('NEMO_DEVICE', 'cuda:0')
    model.to(DEVICE).eval()
    base._repair_buffers(model, args.model_path, DEVICE)
    # Wrapper class doesn't declare sdpa, so load stays eager; but force the
    # inner LLM's 6 attention layers to sdpa (read at forward time) so the
    # 3286-token cmm prefill doesn't materialize a full [B,H,S,S] eager matrix
    # (cmm OOM cause in the manual-step path). DoLa reads no attn weights.
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

    # sanity: premature layer type
    blk = model.language_model.backbone.layers[args.premature_layer].block_type
    print(f'[dola] premature_layer=L{args.premature_layer} ({blk}) alpha={args.alpha}', flush=True)
    auditor = DoLaAuditor(model, premature_layer=args.premature_layer)

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
        has_video = has_audio = has_image = False
        if item.get('video_path'):
            vp = resolve_media_path(item['video_path'], json_dir, args.data_root)
            videos = base.load_video_frames_native(vp); has_video = videos is not None
        if item.get('audio_path'):
            ap = resolve_media_path(item['audio_path'], json_dir, args.data_root)
            audio = base.load_audio_wave(ap); has_audio = audio is not None
        if item.get('image_path'):
            ip = resolve_media_path(item['image_path'], json_dir, args.data_root)
            images = load_image(ip); has_image = images is not None

        user_prompt = get_user_prompt(item['question'], args.datasets,
                                      args.prompt_version, passage=item.get('passage'))
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

        response = ''; gen_ids = None; logp_list = None; input_len = 0
        try:
            inputs = processor(**proc_kwargs)
            inputs = {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in inputs.items()}
            for mk in ('num_patches', 'num_tokens', 'imgs_sizes'):
                inputs.pop(mk, None)
            input_ids = inputs['input_ids']
            attn_mask = inputs.get('attention_mask')
            input_len = int(input_ids.shape[1])
            # fuse_media runs the C-RADIO video ViT; without inference_mode it
            # builds an autograd graph (~2x memory) and OOMs on 64-frame cmm
            # videos. The base runner is safe because generate() is @no_grad.
            with torch.inference_mode():
                inputs_embeds, input_ids, attn_mask = fuse_media(
                    model, input_ids, attn_mask,
                    pixel_values=inputs.get('pixel_values'),
                    pixel_values_videos=inputs.get('pixel_values_videos'),
                    sound_clips=inputs.get('sound_clips'))
            response, gen_ids, logp_list = dola_generate(
                model, auditor, inputs_embeds, attn_mask, tokenizer,
                max_new_tokens=args.max_new_tokens, alpha=args.alpha,
                record_process=args.record_process)
            del inputs_embeds, inputs
            all_out += len(gen_ids)
            print(f'  response[:120]: {response[:120]!r}  tokens={len(gen_ids)}', flush=True)
        except Exception as e:
            import traceback; print(f'  [ERROR] {e}', flush=True); traceback.print_exc()
        elapsed = time.time() - t0
        # Manual-step decode over heavy media (cmm video+audio) accumulates
        # fragmentation; the warm-claim reservation only mattered at load time,
        # so we always free per-sample here (unlike the black-box base/SC path).
        gc.collect(); torch.cuda.empty_cache()

        result = make_result(item, response, args.datasets, 'base_dola_nemotron', time_sec=elapsed)
        result['pred'] = parse_pred(response, args.datasets)
        result['nemotron'] = {'runtime': 'transformers-5.6.2-trust-remote-code',
                              'enable_thinking': bool(args.enable_thinking),
                              'media': {'video': has_video, 'audio': has_audio, 'image': has_image},
                              'premature_layer': args.premature_layer, 'alpha': args.alpha}
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
        f.write(f'premature_layer: {args.premature_layer}\nalpha: {args.alpha}\n')
    print(f'\n[done] total={total:.1f}s | output={output_path}', flush=True)


if __name__ == '__main__':
    main()
