#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""delta_vs_attention_phd.py -- δ-attention consistency on PhD.

Question: do tokens with high counterfactual δ also get high last-layer
attention attribution (mass on image tokens)?  If the two signals disagree,
attribution is contaminated by intermediate tokens and the counterfactual is
the more faithful signal (FlashTrace-style argument, §4.4 ammunition).

Per sample (teacher-forced on the greedy trajectory):
  δ_t   = lp_yes_cf(prefix=full_prompt + gen[:t]) - lp_yes_full(same prefix)
          (same Yes-logprob contrast as cf_confess_phd's decision-token Δ_cf)
  attn_t= last-layer attention mass the query at position (prompt_len+t) places
          on image tokens (mean over heads), eager attention row capture
          (monkeypatch borrowed from methods/base_avcd.py's _avcd_attn_forward)

Aggregation:
  - per-sample IoU of top-10% δ tokens vs top-10% attn tokens
  - per-sample Spearman rank corr(δ_t, attn_t)
  - split by greedy-pred correct vs wrong

Usage:
  CUDA_VISIBLE_DEVICES=4 python delta_vs_attention_phd.py \
      --data_path OmniHalluBench/phd/phd.json --n 120 \
      --output_dir outputs/problem_structure_fig \
      --file_name delta_vs_attention_phd
"""
import os, sys, time, json, argparse, math

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, 'OmniConfess', 'methods'))

import numpy as np
import torch
import torch.nn.functional as F

from shared import (
    load_dataset, load_media_for_item, load_model, build_inputs,
    prepare_for_thinker, build_user_content_with_media_tokens, make_messages,
    get_system_prompt, get_user_prompt,
)
from cf_confess_phd import YES_ID, NO_ID, split_question

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from transformers.models.qwen2_5_omni import modeling_qwen2_5_omni as _qomni

STATE = {"active": False, "row": None}
ORIG_FWD = _qomni.Qwen2_5OmniAttention.forward


def _row_attn_forward(self, hidden_states, attention_mask=None, position_ids=None,
                      past_key_values=None, past_key_value=None, output_attentions=False,
                      use_cache=False, cache_position=None, position_embeddings=None, **kwargs):
    """Eager attention (copy of base_avcd._avcd_attn_forward, recording branch only).

    When STATE['active'] and this is the LAST decoder layer, stash the final
    query row's attention distribution (mean over heads) into STATE['row'].
    """
    bsz, q_len, _ = hidden_states.size()
    query_states = self.q_proj(hidden_states).view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    cos, sin = position_embeddings
    query_states, key_states = _qomni.apply_multimodal_rotary_pos_emb(
        query_states, key_states, cos, sin, self.rope_scaling["mrope_section"])
    _pkv = past_key_values if past_key_values is not None else past_key_value
    if _pkv is not None:
        key_states, value_states = _pkv.update(
            key_states, value_states, self.layer_idx,
            {"sin": sin, "cos": cos, "cache_position": cache_position})
    key_states = _qomni.repeat_kv(key_states, self.num_key_value_groups)
    value_states = _qomni.repeat_kv(value_states, self.num_key_value_groups)
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
    if attention_mask is None and q_len > 1 and key_states.shape[-2] == q_len:
        min_v = torch.finfo(attn_weights.dtype).min
        causal = torch.full((q_len, q_len), min_v, device=attn_weights.device,
                            dtype=attn_weights.dtype)
        causal = torch.triu(causal, diagonal=1)
        attn_weights = attn_weights + causal[None, None, :, :]
    elif attention_mask is not None:
        attn_weights = attn_weights + attention_mask[:, :, :, : key_states.shape[-2]]
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    if STATE["active"] and q_len > 1:
        STATE["row"] = attn_weights[:, :, -1, :].float().mean(dim=1)[0].detach().cpu()
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
    attn_output = self.o_proj(attn_output)
    return attn_output, (attn_weights if output_attentions else None)


def set_seed(seed):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


@torch.inference_mode()
def forward_last(model, clean, prefix_ids, capture_attn=False):
    """Teacher-forced forward of clean + prefix; return (lp_yes, attn_row or None)."""
    ids = torch.cat([clean['input_ids'][0],
                     torch.tensor(prefix_ids, dtype=torch.long,
                                  device=clean['input_ids'].device)],
                    dim=0).unsqueeze(0)
    kw = dict(input_ids=ids, attention_mask=torch.ones_like(ids))
    for k in ('pixel_values_images', 'pixel_values', 'image_grid_thw',
              'pixel_values_videos', 'video_grid_thw',
              'video_second_per_grid', 'input_features', 'feature_attention_mask'):
        if k in clean:
            kw[k] = clean[k]
    if capture_attn:
        STATE["active"] = True; STATE["row"] = None
    logits = model.thinker(**kw).logits[0, -1].float()
    row = STATE["row"] if capture_attn else None
    STATE["active"] = False
    lp = torch.log_softmax(logits, dim=-1)
    return lp[YES_ID].item(), row


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return None
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def topk_iou(a, b, frac=0.10):
    a, b = np.asarray(a, float), np.asarray(b, float)
    k = max(1, int(round(len(a) * frac)))
    sa = set(np.argsort(a)[::-1][:k].tolist())
    sb = set(np.argsort(b)[::-1][:k].tolist())
    return len(sa & sb) / len(sa | sb)


def pred_yes_no(text):
    t = text.strip().lower()
    if t.startswith('yes'): return 'yes'
    if t.startswith('no'): return 'no'
    if '\\boxed{yes' in t: return 'yes'
    if '\\boxed{no' in t: return 'no'
    if 'yes' in t.split('.')[0]: return 'yes'
    if 'no' in t.split('.')[0]: return 'no'
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', default='phd')
    ap.add_argument('--data_path', required=True)
    ap.add_argument('--data_root', default='./OmniHalluBench/')
    ap.add_argument('--output_dir', default='outputs/problem_structure_fig')
    ap.add_argument('--file_name', default='delta_vs_attention_phd')
    ap.add_argument('--n', type=int, default=120)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--max_new_tokens', type=int, default=64)
    ap.add_argument('--prompt_version', default='v1')
    ap.add_argument('--min_gen_tokens', type=int, default=10)
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, args.file_name + '.jsonl')
    done_ids = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            try: done_ids.add(json.loads(line)['id'])
            except Exception: pass
        print(f'[resume] skip {len(done_ids)}', flush=True)

    print('[model] loading Qwen2.5-Omni...', flush=True)
    processor, model = load_model()
    tokenizer = processor.tokenizer
    cls = type(model.thinker.model.layers[0].self_attn).__name__
    print(f'[attn class] {cls}', flush=True)

    img_pad_id = None
    for _t in ('<|IMAGE|>', '<|image_pad|>', '<|vision_pad|>'):
        _i = tokenizer.convert_tokens_to_ids(_t)
        if _i is not None and _i >= 0 and tokenizer.convert_ids_to_tokens(_i) == _t:
            img_pad_id = _i
            break
    print(f'[image_pad id] {img_pad_id}', flush=True)

    _qomni.Qwen2_5OmniAttention.forward = _row_attn_forward

    data = load_dataset(args.data_path, args.data_root)[:args.n]
    system_prompt = get_system_prompt(args.datasets, args.prompt_version)

    for idx, item in enumerate(data):
        if item['id'] in done_ids:
            continue
        t0 = time.time()
        try:
            pv, av, iv = load_media_for_item(item, args.data_path, args.data_root)
            q_full, q_only = split_question(item['question'])
            u_full = get_user_prompt(q_full, args.datasets, args.prompt_version)
            uc_full = build_user_content_with_media_tokens(pv, av, iv, u_full)
            clean_f = prepare_for_thinker(
                build_inputs(processor, model,
                             make_messages(system_prompt, uc_full), pv, av, iv,
                             args.datasets), model)
            u_cf = get_user_prompt(q_only, args.datasets, args.prompt_version)
            uc_cf = build_user_content_with_media_tokens(pv, av, iv, u_cf)
            clean_cf = prepare_for_thinker(
                build_inputs(processor, model,
                             make_messages(system_prompt, uc_cf), pv, av, iv,
                             args.datasets), model)
            input_len = clean_f['input_ids'].shape[1]
            _ids0 = clean_f['input_ids'][0]
            if not torch.is_tensor(_ids0):
                _ids0 = torch.tensor(_ids0)
            if _ids0.dim() == 0:
                _ids0 = clean_f['input_ids'].reshape(-1)
            full_ids_list = _ids0.tolist()
            img_mask_full = (_ids0 == img_pad_id).cpu()
            print(f'[dbg] input_ids type={type(clean_f["input_ids"])} shape='
                  f'{getattr(clean_f["input_ids"], "shape", None)} row0={type(clean_f["input_ids"][0])} '
                  f'img_pad_id={img_pad_id!r} n_img={int(img_mask_full.sum())}', flush=True)
            n_img = int(img_mask_full.sum())

            # greedy trajectory (no intervention)
            with torch.inference_mode():
                out = model.thinker.generate(
                    **clean_f, max_new_tokens=args.max_new_tokens, do_sample=False,
                    pad_token_id=tokenizer.eos_token_id)
            _seq = out.sequences if hasattr(out, 'sequences') else out
            gen = _seq[0][input_len:].cpu().tolist()
            # cut at eos
            eos = tokenizer.eos_token_id
            if eos in gen:
                gen = gen[:gen.index(eos)]
            response = tokenizer.decode(gen, skip_special_tokens=True).strip()
            pred = pred_yes_no(response)
            gt = str(item.get('gt', item.get('label', ''))).strip().lower()
            correct = None if (pred is None or not gt) else int(pred == gt)

            deltas, attns = [], []
            L = len(full_ids_list)
            for t in range(len(gen)):
                prefix = gen[:t]
                ids_f = full_ids_list + prefix
                ly_full, row = forward_last(model, clean_f, prefix, capture_attn=True)
                ly_cf, _ = forward_last(model, clean_cf, prefix)
                deltas.append(ly_cf - ly_full)
                # row covers ids_f; image span lives in [0, L)
                if row is not None and len(row) == len(ids_f):
                    attns.append(float(row[:L][img_mask_full].sum()))
                else:
                    attns.append(None)
            ok = [t for t in range(len(gen)) if attns[t] is not None]
            deltas_ok = [deltas[t] for t in ok]
            attns_ok = [attns[t] for t in ok]

            rec = dict(id=item['id'], gt=gt, pred=pred, correct=correct,
                       n_gen=len(gen), n_img_tokens=n_img, prompt_len=L,
                       response=response[:120])
            if len(ok) >= args.min_gen_tokens:
                rec.update(
                    iou_top10=topk_iou(deltas_ok, attns_ok, 0.10),
                    iou_top20=topk_iou(deltas_ok, attns_ok, 0.20),
                    spearman=spearman(deltas_ok, attns_ok),
                    delta_mean=float(np.mean(deltas_ok)),
                    delta_absmean=float(np.mean(np.abs(deltas_ok))),
                    attn_mean=float(np.mean(attns_ok)),
                    deltas=[round(d, 4) for d in deltas_ok],
                    attns=[round(a, 6) for a in attns_ok],
                )
            else:
                rec['skipped_short'] = True
            with open(out_path, 'a') as f:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            print(f"[{idx+1}/{len(data)}] id={item['id']} ngen={len(gen)} pred={pred} "
                  f"correct={correct} iou={rec.get('iou_top10')} rho={rec.get('spearman')} "
                  f"| {time.time()-t0:.1f}s", flush=True)
            del clean_f, clean_cf
            torch.cuda.empty_cache()
        except Exception as e:
            import traceback; traceback.print_exc()
            with open(out_path, 'a') as f:
                f.write(json.dumps({'id': item['id'], 'error': str(e)}) + '\n')

    _qomni.Qwen2_5OmniAttention.forward = ORIG_FWD
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
