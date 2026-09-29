# -*- coding: utf-8 -*-
"""bon_stacked_feat_combo.py — 回灌收尾：特征组合版 BoN 进三牌合体放行支线.

Coordinator brief (pre-registered): the stacked pipeline's pass-through branch
(v1 affirmative & gate-clean, 81/365 items) currently picks candidates by pure
assertion-Delta; swap the ranker for the feature combo calibrated in
analyze_bon_features.py (score = z(Delta) - 0.25*z(CE) + 2.0*z(VAS), weights and
z-params both fixed on the haloquest v1-pool dev split -- NOTHING is re-tuned
here). All other branches byte-identical. Judgement: full-365 Token-F1
> 40.06 (the current three-layer terminal value).

Requires one GPU pass to extract CE/VAS for the 81x4 pass-through candidates
(same v2-contract prompt + own image used at generation; Delta is already
stored in the trajectories).

Usage:
    CUDA_VISIBLE_DEVICES=2 python bon_stacked_feat_combo.py
"""
import os, sys, json, random, time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'OmniConfess', 'methods'))
sys.path.insert(0, os.path.join(HERE, 'scripts'))
sys.path.insert(0, HERE)
from shared import (  # noqa: E402
    load_dataset, load_media_for_item, get_system_prompt, get_user_prompt,
    build_user_content_with_media_tokens, make_messages,
)
from probe_bon_features import forward_with_feats, build_view, evidence_mask  # noqa: E402
from bon_grounding import token_class  # noqa: E402

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("MMHALU_MODEL_ROOT", "/ultrafast_share/tsq/MODELS")

STACKED = os.path.join(HERE, 'outputs', 'haloquest_bon_stacked',
                       'haloquest_bon_stacked_full.jsonl')
V1_FEAT = os.path.join(HERE, 'outputs', 'bon_probe', 'haloquest_bon_feat.json')
V1_POOL = os.path.join(HERE, 'outputs', 'haloquest_bon_grounding',
                       'haloquest_bon_grounding.jsonl')
BETA, GAMMA = 0.25, 2.0     # fixed by the phase-1 dev grid; NOT re-tuned here


def dev_zpars():
    """Re-derive the dev(100) z-params of the calibrated scorer (v1 pool)."""
    rows = json.load(open(V1_FEAT))['rows']
    pool = {r['id']: {t['rollout']: t for t in r['trajectories']}
            for r in map(json.loads, open(V1_POOL))}

    def feats(c):
        v = c['VAS']['14']['vas_content']
        if v is None:
            v = c['VAS']['14']['vas_all']
        gs = pool[c['_id']][c['rollout']].get('grounding_scores')
        g = gs['g_traj_mean'] if gs else 0.0
        return (g, c['CE']['CE_top5mean'], v)

    for r in rows:
        for c in r['candidates']:
            c['_id'] = r['id']
    rng = random.Random(42)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    dev = [rows[i] for i in idx[:100]]
    zp = []
    for fi in range(3):
        vals = [feats(c)[fi] for r in dev for c in r['candidates']]
        zp.append((float(np.mean(vals)), float(np.std(vals)) + 1e-9))
    return zp


def main():
    zp = dev_zpars()
    print(f'[z-params] delta {zp[0][0]:.3f}/{zp[0][1]:.3f} '
          f'CE {zp[1][0]:.3f}/{zp[1][1]:.3f} VAS {zp[2][0]:.4f}/{zp[2][1]:.4f}', flush=True)

    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
    from shared.paths import resolve_model
    model_path = resolve_model("Qwen/Qwen2.5-Omni-7B")
    processor = Qwen2_5OmniProcessor.from_pretrained(model_path, trust_remote_code=True)
    processor.tokenizer.padding_side = 'left'
    if not processor.tokenizer.pad_token:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map='auto',
        trust_remote_code=True, attn_implementation='eager').eval()
    model.thinker.config.vocab_size = 152064
    tokenizer = processor.tokenizer
    system_prompt = get_system_prompt('haloquest', 'v2')

    data = load_dataset(os.path.join(HERE, 'OmniHalluBench', 'haloquest', 'haloquest.json'),
                        os.path.join(HERE, 'OmniHalluBench'))
    by_id = {it['id']: it for it in data}

    recs = [json.loads(l) for l in open(STACKED)]
    out_path = STACKED.replace('_full.jsonl', '_feat_combo.jsonl')
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            try:
                done.add(json.loads(line)['id'])
            except Exception:
                pass
    outf = open(out_path, 'a')

    t0 = time.time()
    n = 0
    for rec in recs:
        if rec['id'] in done:
            continue
        item = by_id[rec['id']]
        if rec['branch'] != 'passthrough_bon':
            outf.write(json.dumps(rec, ensure_ascii=False) + '\n')
            outf.flush()
            continue
        # feature extraction for the 4 candidates under the generation prompt
        img = load_media_for_item(item, os.path.join(
            HERE, 'OmniHalluBench', 'haloquest', 'haloquest.json'),
            os.path.join(HERE, 'OmniHalluBench'))[2]
        user_prompt = get_user_prompt(item['question'], 'haloquest', 'v2')
        messages = make_messages(system_prompt,
                                 build_user_content_with_media_tokens(None, None, img, user_prompt))
        clean = build_view(processor, model, system_prompt, user_prompt, 'haloquest', img)
        ev_mask, _ = evidence_mask(tokenizer, processor, clean, 'haloquest', item, messages)
        best, bv = None, -1e18
        cands = []
        for tr in rec['trajectories']:
            gen_ids = torch.tensor(tr['token_ids'], dtype=torch.long, device=model.device)
            toks = [tokenizer.decode([t], skip_special_tokens=True) for t in tr['token_ids']]
            cls = [token_class(t, toks[k - 1] if k else None) for k, t in enumerate(toks)]
            content_pos = torch.tensor([c in ('content', 'num') for c in cls],
                                       device=model.device)
            f = forward_with_feats(thinker := model.thinker, clean, gen_ids, ev_mask, content_pos)
            gs = tr.get('grounding_scores') or {}
            g = gs.get('g_traj_mean', 0.0)
            ce = f['CE']['CE_top5mean']
            v = f['VAS'][14]['vas_content']   # in-memory dict: int layer keys
            if v is None:
                v = f['VAS'][14]['vas_all']
            z = lambda x, i: (x - zp[i][0]) / zp[i][1]
            score = z(g, 0) - BETA * z(ce, 1) + GAMMA * z(v, 2)
            cands.append({'rollout': tr['rollout'], 'score': score,
                          'g_traj': g, 'CE': ce, 'VAS': v})
            if score > bv:
                bv, best = score, tr
        rec = dict(rec)
        rec['method'] = 'bon_stacked_feat_combo'
        rec['pred'] = best['pred']
        rec['raw_output'] = best['pred']
        rec['rank_score'] = f'feat_combo_beta{BETA}_gamma{GAMMA}_devz'
        rec['selected_rollout_feat'] = best['rollout']
        rec['feat_combo_candidates'] = cands
        outf.write(json.dumps(rec, ensure_ascii=False) + '\n')
        outf.flush()
        n += 1
        if n % 20 == 0:
            print(f'  [{n}] {time.time()-t0:.0f}s', flush=True)
        torch.cuda.empty_cache()
    outf.close()

    # random-selection control arm (same rollouts, same recorded random picks)
    ctrl = out_path.replace('_feat_combo.jsonl', '_random_sel.jsonl')
    with open(out_path) as fin, open(ctrl, 'w') as fout:
        for line in fin:
            r = json.loads(line)
            if r.get('random_sel_rollout') is not None and r.get('trajectories'):
                r['pred'] = r['trajectories'][r['random_sel_rollout']]['pred']
            r['method'] = 'bon_stacked_random_sel'
            fout.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'[done] -> {out_path}', flush=True)


if __name__ == '__main__':
    main()
