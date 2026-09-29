#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Same-axis entropy baseline (Qwen2.5-Omni-7B, PHD 906).

Preregistered spec (from task order 2026-08-18):
  * sample: phd base greedy full run (outputs/phd_base_model/
    phd_base_model_full_single_scored.jsonl), rows with token_ids
  * at the SAME decision token as probe_views_phd.py (find_decision_index),
    teacher-forced forward of the FULL prompt (v1, marker intact, media kept)
  * signal: full-vocabulary softmax entropy of the next-token distribution
    (NOT two-class entropy)
  * statistic: AUROC + percentile bootstrap CI, bootstrap n=1000, seed=42,
    labels reported both vs gt_yes (same label as margin_full AUC 0.739 in
    ablation/a3_dose_response.json, same-table comparable) and vs
    base_correct
Outputs:
  outputs/cf_probe/phd_entropy_probe.jsonl  (per-row entropy etc.)
  outputs/cf_probe/phd_entropy_probe_summary.json
"""
import os
import sys
import time
import argparse
import json

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'OmniConfess', 'methods'))
os.chdir(ROOT)

import torch
from shared import (
    load_dataset, load_media_for_item, load_model, build_inputs,
    prepare_for_thinker, build_user_content_with_media_tokens, make_messages,
    get_system_prompt, get_user_prompt,
)

YES_ID, NO_ID = 9454, 2753


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
def decision_entropy(model, clean, prefix_ids):
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
    p = torch.exp(lp)
    entropy = -(p * lp).sum().item()
    return entropy, lp[YES_ID].item(), lp[NO_ID].item()


def bootstrap_auc(x, y, n=1000, seed=42):
    import numpy as np
    from sklearn.metrics import roc_auc_score
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)
    ok = np.isfinite(x)
    x, y = x[ok], y[ok]
    rng = np.random.RandomState(seed)
    aucs = []
    m = len(x)
    for _ in range(n):
        idx = rng.randint(0, m, m)
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        aucs.append(roc_auc_score(yy, x[idx]))
    pt = roc_auc_score(y, x)
    return float(pt), (float(np.percentile(aucs, 2.5)),
                       float(np.percentile(aucs, 97.5)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scored', default='outputs/phd_base_model/phd_base_model_full_single_scored.jsonl')
    ap.add_argument('--data_path', default='./OmniHalluBench/phd/annotations/phd.json')
    ap.add_argument('--data_root', default='./OmniHalluBench/')
    ap.add_argument('--out', default='outputs/cf_probe/phd_entropy_probe.jsonl')
    ap.add_argument('--summary', default='outputs/cf_probe/phd_entropy_probe_summary.json')
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

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
    recs = []
    tmp = args.out + '.tmp'
    with open(tmp, 'w') as fout:
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
            u_q = get_user_prompt(r['question'], 'phd', 'v1')
            uc = build_user_content_with_media_tokens(pv, av, iv, u_q)
            clean = prepare_for_thinker(
                build_inputs(processor, model,
                             make_messages(system_prompt, uc),
                             pv, av, iv, 'phd'), model)
            ent, ly, ln = decision_entropy(model, clean, prefix)
            rec = dict(id=r['id'], gt=r['gt'], pred=r['pred'],
                       gt_yes=int(r['gt'] == 'yes'),
                       base_correct=int(r['pred'] == r['gt']),
                       dec_index=prefix_len, dec_token=dec_tok,
                       entropy=round(ent, 4),
                       lp_yes_full=round(ly, 4), lp_no_full=round(ln, 4),
                       margin_full=round(ly - ln, 4))
            fout.write(json.dumps(rec) + '\n')
            if (n + 1) % 20 == 0:
                fout.flush()
                os.replace(tmp, args.out)
                print(f'  [{n+1}/{len(sample)}] {time.time()-t0:.0f}s ent={ent:.3f}', flush=True)
            del clean
            torch.cuda.empty_cache()
    os.replace(tmp, args.out)

    recs = [json.loads(l) for l in open(args.out)]
    e = [r['entropy'] for r in recs]
    neg_e = [-v for v in e]
    m = [r['margin_full'] for r in recs]
    am = [abs(v) for v in m]
    y_gt = [r['gt_yes'] for r in recs]
    y_bc = [r['base_correct'] for r in recs]
    summary = dict(
        n=len(recs),
        spec='full-vocab softmax entropy @ base greedy decision token, '
             'teacher-forced full prompt (v1, media kept); bootstrap n=1000 seed=42',
        entropy_vs_gt_yes=dict(zip(('auc', 'ci'), bootstrap_auc(neg_e, y_gt))),
        entropy_vs_base_correct=dict(zip(('auc', 'ci'), bootstrap_auc(neg_e, y_bc))),
        margin_full_vs_gt_yes=dict(zip(('auc', 'ci'), bootstrap_auc(m, y_gt))),
        absmargin_vs_base_correct=dict(zip(('auc', 'ci'), bootstrap_auc(am, y_bc))),
        reference_margin_full_gt_yes=0.739,
        note='entropy direction: lower entropy -> model committed -> '
             'reported as -entropy so AUC>0.5 means low entropy associates '
             'with label=1; reference 0.739 (a3) uses label gt_yes, '
             'signal margin_full',
    )
    def rnd(b):
        b['auc'] = round(b['auc'], 3)
        b['ci'] = [round(b['ci'][0], 3), round(b['ci'][1], 3)]
        return b
    for k in ('entropy_vs_gt_yes', 'entropy_vs_base_correct',
              'margin_full_vs_gt_yes', 'absmargin_vs_base_correct'):
        if k in summary:
            summary[k] = rnd(summary[k])
    with open(args.summary, 'w') as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print('[summary]', json.dumps(summary, ensure_ascii=False), flush=True)
    print(f'[done] {len(recs)} in {time.time()-t0:.0f}s -> {args.out}', flush=True)


if __name__ == '__main__':
    main()
