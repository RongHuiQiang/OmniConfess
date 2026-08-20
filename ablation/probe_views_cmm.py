#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A1 view-ablation probe for CMM: the double-masked (text-only) view.

Adds the missing 'nomedia' view (question only, both audio and video
dropped) at the same decision token as the shipped cmm probe, so the A1
cmm view-ablation table can include the double-mask arm offline.
"""
import os, sys, json, time, argparse, random

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'OmniConfess', 'methods'))
os.chdir(ROOT)

import torch
from shared import (
    load_dataset, load_media_for_item, load_model,
    build_user_content_with_media_tokens, make_messages,
    get_system_prompt, get_user_prompt, build_inputs, prepare_for_thinker,
)

YES_ID, NO_ID = 9454, 2753


def load_aligned(scored_path, data_dir):
    rows = [json.loads(l) for l in open(scored_path)]
    data = load_dataset(data_dir, './OmniHalluBench/')
    by_q = {}
    for it in data:
        by_q.setdefault(it['question'], []).append(it)
    aligned = []
    for r in rows:
        if not r.get('token_ids'):
            continue
        items = by_q.get(r['question'])
        if not items:
            continue
        aligned.append((r, items[0]))
    return aligned


def find_decision_index(tokenizer, token_ids):
    ids = list(token_ids)
    if ids and ids[0] in (YES_ID, NO_ID):
        return 0, ids[0]
    for i in range(len(ids) - 2, -1, -1):
        if tokenizer.decode([ids[i]]).strip() == 'boxed':
            if tokenizer.decode([ids[i + 1]]).strip() == '{':
                return i + 2, ids[i + 2] if i + 2 < len(ids) else None
    return None, None


@torch.inference_mode()
def decision_logp(model, clean, prefix_ids):
    ids = torch.cat([clean['input_ids'][0],
                     torch.tensor(prefix_ids, dtype=torch.long,
                                  device=clean['input_ids'].device)], dim=0
                    ).unsqueeze(0)
    kw = dict(input_ids=ids, attention_mask=torch.ones_like(ids))
    for k in ('pixel_values_images', 'pixel_values', 'image_grid_thw',
              'pixel_values_videos', 'video_grid_thw', 'video_second_per_grid',
              'input_features', 'feature_attention_mask'):
        if k in clean:
            kw[k] = clean[k]
    logits = model.thinker(**kw).logits[0, -1].float()
    lp = torch.log_softmax(logits, dim=-1)
    return lp[YES_ID].item(), lp[NO_ID].item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scored', default='outputs/cmm_base_model/cmm_base_model_full_single_scored.jsonl')
    ap.add_argument('--data_path', default='./OmniHalluBench/cmm/cmm.json')
    ap.add_argument('--out', default='outputs/cf_probe/cmm_probe_nomedia.jsonl')
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    aligned = load_aligned(args.scored, args.data_path)
    if args.limit:
        aligned = aligned[:args.limit]
    print(f'[probe] {len(aligned)} aligned rows', flush=True)
    processor, model = load_model()
    tokenizer = processor.tokenizer
    system_prompt = get_system_prompt('cmm', 'v1')

    t0 = time.time()
    with open(args.out, 'w') as fout:
        for n, (r, item) in enumerate(aligned):
            try:
                prefix_len, dec_tok = find_decision_index(tokenizer, r['token_ids'])
                if prefix_len is None:
                    continue
                prefix = r['token_ids'][:prefix_len]
                u = get_user_prompt(item['question'], 'cmm', 'v1')
                uc = build_user_content_with_media_tokens(None, None, None, u)
                clean = prepare_for_thinker(
                    build_inputs(processor, model,
                                 make_messages(system_prompt, uc),
                                 None, None, None, 'cmm'), model)
                ly, ln = decision_logp(model, clean, prefix)
                rec = dict(id=r['id'], q=r['question'][:60], gt=r['gt'],
                           pred=r['pred'], dec_index=prefix_len, dec_token=dec_tok,
                           lp_yes_nomedia=round(ly, 4), lp_no_nomedia=round(ln, 4),
                           margin_nomedia=round(ly - ln, 4))
                fout.write(json.dumps(rec) + '\n')
                fout.flush()
                del clean
                torch.cuda.empty_cache()
                if (n + 1) % 50 == 0:
                    print(f'  [{n+1}/{len(aligned)}] {time.time()-t0:.0f}s', flush=True)
            except Exception as e:
                print(f'  [ERROR] {r["id"]}: {e}', flush=True)
                torch.cuda.empty_cache()
    print(f'[done] {time.time()-t0:.0f}s -> {args.out}', flush=True)


if __name__ == '__main__':
    main()
