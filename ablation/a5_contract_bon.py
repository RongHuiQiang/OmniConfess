#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A5 (ABLATION_PLAN.md): contract + BoN (NO gate) cell @ HaloQuest.

All 365 items answered from a v2-CONTRACT-prompt candidate pool
(1 greedy + 3 sampled T=0.7), assertion-level Delta scoring (contrast view =
a swapped image, identical to the shipped stacked run), g_content rank
selection. NO gate routing and NO v1-evade routing.

Reuses the 81 passthrough-branch pools already on disk
(outputs/haloquest_bon_stacked/haloquest_bon_stacked_full.jsonl) and
generates the remaining 284 pools with the SAME protocol (same seeds rule,
same partner sampling, same scoring) -- mechanism frozen.

Usage: CUDA_VISIBLE_DEVICES=4 python ablation/a5_contract_bon.py
"""
import os, sys, time, json, random, argparse

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'OmniConfess', 'methods'))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from shared import (  # noqa: E402
    load_dataset, load_media_for_item, resolve_media_path,
    load_model, get_user_prompt, parse_pred,
)
from bon_grounding import (  # noqa: E402
    claim_spans, view_logp, build_view, rollout_scores, rank_value, set_seed,
)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("MMHALU_MODEL_ROOT", "/ultrafast_share/tsq/MODELS")

STACKED = 'outputs/haloquest_bon_stacked/haloquest_bon_stacked_full.jsonl'
OUT_DIR = 'outputs/haloquest_a5'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--num_sampled', type=int, default=3)
    ap.add_argument('--temperature', type=float, default=0.7)
    ap.add_argument('--max_new_tokens', type=int, default=512)
    ap.add_argument('--rank_score', default='g_content')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(OUT_DIR, exist_ok=True)
    out_main = os.path.join(OUT_DIR, 'haloquest_a5_contract_bon.jsonl')

    stacked = [json.loads(l) for l in open(STACKED)]
    have = {r['id']: r for r in stacked if r.get('trajectories')}
    print(f'[reuse] {len(have)} v2-pools already on disk', flush=True)

    done = set()
    if os.path.exists(out_main):
        for line in open(out_main):
            try:
                done.add(json.loads(line)['id'])
            except Exception:
                pass
        print(f'[resume] {len(done)}', flush=True)

    data_path = os.path.join('OmniHalluBench', 'haloquest', 'haloquest.json')
    data_root = os.path.join('OmniHalluBench')
    data = load_dataset(data_path, data_root)
    by_id = {it['id']: it for it in data}
    pool_images = [it for it in data if it.get('image_path') and os.path.exists(
        resolve_media_path(it['image_path'], os.path.dirname(data_path), data_root))]

    processor, model = load_model()
    tokenizer = processor.tokenizer
    thinker = model.thinker
    system_prompt = None
    from shared import get_system_prompt
    system_prompt = get_system_prompt('haloquest', 'v2')

    rng = random.Random(args.seed + 7)
    rng_rand = random.Random(args.seed + 13)
    t0 = time.time()
    n_gen = 0

    for rec in stacked:                       # same iteration order as shipped
        i = rec['id']
        if i in done:
            continue
        item = by_id[i]
        if i in have:
            trajs = have[i]['trajectories']
            sel = max(range(len(trajs)),
                      key=lambda k: rank_value(trajs[k]['grounding_scores'], args.rank_score))
            pred = trajs[sel]['pred']
            entry = dict(id=i, question=rec['question'], gt=rec['gt'],
                         group_key=rec.get('group_key'), dataset='haloquest',
                         method='a5_contract_bon', pred=pred,
                         selected_rollout=int(sel),
                         random_sel_rollout=rng_rand.randrange(len(trajs)),
                         trajectories=trajs, source='reused_stacked_pool')
        else:
            cands, cand_ids, scores_list = [], [], []
            try:
                img = load_media_for_item(item, data_path, data_root)[2]
                user_prompt = get_user_prompt(item['question'], 'haloquest', 'v2')
                clean = build_view(processor, model, system_prompt, user_prompt,
                                   'haloquest', img)
                input_len = clean['input_ids'].shape[1]
                partner = None
                while partner is None:
                    cand_it = rng.choice(pool_images)
                    if cand_it['id'] != item['id']:
                        partner = cand_it
                pimg = load_media_for_item(partner, data_path, data_root)[2]
                ctrl = build_view(processor, model, system_prompt, user_prompt,
                                  'haloquest', pimg)
                for s, (do_sample, temp) in enumerate(
                        [(False, 1.0)] + [(True, args.temperature)] * args.num_sampled):
                    gk = dict(max_new_tokens=args.max_new_tokens,
                              do_sample=do_sample,
                              pad_token_id=tokenizer.eos_token_id,
                              stop_strings=['<end_of_reasoning>'],
                              tokenizer=tokenizer)
                    if do_sample:
                        gk.update(temperature=temp)
                    with torch.inference_mode():
                        out = thinker.generate(**clean, **gk)
                    gen_ids = out[0][input_len:]
                    gen_ids = gen_ids[gen_ids != tokenizer.pad_token_id]
                    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                    cands.append(text)
                    cand_ids.append(gen_ids.tolist())
                    torch.cuda.empty_cache()
                for gl in cand_ids:
                    gt_t = torch.tensor(gl, dtype=torch.long, device=model.device)
                    delta = (view_logp(thinker, clean, gt_t)
                             - view_logp(thinker, ctrl, gt_t))
                    toks = [tokenizer.decode([t], skip_special_tokens=True) for t in gl]
                    cls = [__import__('bon_grounding').token_class(t, toks[k - 1] if k else None)
                           for k, t in enumerate(toks)]
                    scores_list.append(rollout_scores(delta, claim_spans(toks, cls), cls))
                sel = max(range(len(cands)),
                          key=lambda k: rank_value(scores_list[k], args.rank_score))
                entry = dict(id=i, question=rec['question'], gt=rec['gt'],
                             group_key=rec.get('group_key'), dataset='haloquest',
                             method='a5_contract_bon',
                             pred=parse_pred(cands[sel], 'haloquest'),
                             selected_rollout=int(sel),
                             random_sel_rollout=rng_rand.randrange(len(cands)),
                             trajectories=[{'rollout': k, 'sampled': bool(k > 0),
                                            'token_ids': cand_ids[k],
                                            'pred': parse_pred(cands[k], 'haloquest'),
                                            'grounding_scores': scores_list[k]}
                                           for k in range(len(cands))],
                             partner=partner['id'], source='generated')
                n_gen += 1
            except Exception as e:
                print(f'  [ERROR] {i}: {e}', flush=True)
                torch.cuda.empty_cache()
                entry = dict(id=i, question=rec['question'], gt=rec['gt'],
                             method='a5_contract_bon', pred=rec.get('v2_pred'),
                             selected_rollout=None, random_sel_rollout=None,
                             trajectories=None, error=str(e)[:200],
                             source='fallback_v2_pred')
        with open(out_main, 'a') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        if n_gen and n_gen % 20 == 0:
            print(f'  [gen {n_gen}] {time.time()-t0:.0f}s', flush=True)

    print(f'[done] {time.time()-t0:.0f}s generated {n_gen} pools -> {out_main}', flush=True)


if __name__ == '__main__':
    main()
