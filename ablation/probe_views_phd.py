#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A1 view-ablation probe: the double-masked (question-only) view for PhD.

Measures, at the SAME decision token as the existing full probe
(outputs/cf_probe/phd_probe_full.jsonl, whose full/noimg/noctx views are
already on disk), the teacher-forced lp_yes / lp_no of the view

    qonly : question sentence only (marker dropped, context dropped, IMAGE
            dropped)   -> "double mask" arm of the A1 view ablation

Output rows carry (id, gt, pred, dec_index) + lp_yes_qonly/lp_no_qonly so the
offline arm simulation can join them to the existing probe rows.
"""
import os
import sys
import time
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'OmniConfess', 'methods'))
os.chdir(ROOT)
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
from shared import (
    load_dataset, load_media_for_item, load_model, build_inputs,
    prepare_for_thinker, build_user_content_with_media_tokens, make_messages,
    get_system_prompt, get_user_prompt,
)

MARKER = ("In case there is an inconsistency between the context and the image "
          "content, you should follow the image.")
YES_ID, NO_ID = 9454, 2753


def split_question(q):
    idx = q.find(MARKER)
    if idx < 0:
        return q, q
    return q, q[idx + len(MARKER):].lstrip()


def find_decision_index(tokenizer, token_ids):
    ids = list(token_ids)
    if ids and ids[0] in (YES_ID, NO_ID):
        return 0, ids[0]
    for i in range(len(ids) - 2, -1, -1):
        if tokenizer.decode([ids[i]]).strip() == 'boxed':
            if tokenizer.decode([ids[i + 1]]).strip() == '{':
                return i + 2, ids[i + 2] if i + 2 < len(ids) else None
    for i in range(len(ids) - 1, 0, -1):
        if tokenizer.decode(ids[:i]).endswith('\\boxed{'):
            return i, ids[i]
    return None, None


@torch.inference_mode()
def decision_logp(model, clean, prefix_ids):
    ids = torch.cat([clean['input_ids'][0],
                     torch.tensor(prefix_ids, dtype=torch.long,
                                  device=clean['input_ids'].device)], dim=0
                    ).unsqueeze(0)
    attn = torch.ones_like(ids)
    kw = dict(input_ids=ids, attention_mask=attn)
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
    ap.add_argument('--scored', default='outputs/phd_base_model/phd_base_model_full_single_scored.jsonl')
    ap.add_argument('--data_path', default='./OmniHalluBench/phd/phd.json')
    ap.add_argument('--data_root', default='./OmniHalluBench/')
    ap.add_argument('--out', default='outputs/cf_probe/phd_probe_qonly.jsonl')
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    import json
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rows = [json.loads(l) for l in open(args.scored)]
    data = load_dataset(args.data_path, args.data_root)
    by_q = {}
    for it in data:
        by_q.setdefault(it['question'], []).append(it)

    sample = [r for r in rows if r.get('token_ids')]
    if args.limit:
        sample = sample[:args.limit]

    print('[model] loading...', flush=True)
    processor, model = load_model()
    tokenizer = processor.tokenizer
    system_prompt = get_system_prompt('phd', 'v1')

    t0 = time.time()
    with open(args.out, 'w') as fout:
        for n, r in enumerate(sample):
            items = by_q.get(r['question'])
            if not items:
                continue
            item = items[0]
            pv, av, iv = load_media_for_item(item, args.data_path, args.data_root)
            prefix_len, dec_tok = find_decision_index(tokenizer, r['token_ids'])
            if prefix_len is None:
                continue
            prefix = r['token_ids'][:prefix_len]
            _, q_only = split_question(r['question'])
            u_q = get_user_prompt(q_only, 'phd', 'v1')
            uc_q = build_user_content_with_media_tokens(None, None, None, u_q)
            clean_q = prepare_for_thinker(
                build_inputs(processor, model,
                             make_messages(system_prompt, uc_q),
                             None, None, None, 'phd'), model)
            ly, ln = decision_logp(model, clean_q, prefix)
            rec = dict(id=r['id'], gt=r['gt'], pred=r['pred'],
                       dec_index=prefix_len, dec_token=dec_tok,
                       q=r['question'][:80],
                       lp_yes_qonly=round(ly, 4), lp_no_qonly=round(ln, 4),
                       margin_qonly=round(ly - ln, 4))
            fout.write(json.dumps(rec) + '\n')
            fout.flush()
            if (n + 1) % 25 == 0:
                print(f'  [{n+1}/{len(sample)}] {time.time()-t0:.0f}s', flush=True)
            del clean_q
            torch.cuda.empty_cache()
    print(f'[done] {len(sample)} in {time.time()-t0:.0f}s -> {args.out}', flush=True)


if __name__ == '__main__':
    main()
