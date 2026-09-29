# -*- coding: utf-8 -*-
"""bon_stacked_feat_combo_qwen3.py — Qwen3-Omni port of bon_stacked_feat_combo.py.

四层链路第 4 层：BoN VAS 特征组合 rerank，移植到 Qwen3，接在层3
(bon_stacked_haloquest_qwen3.py, passthrough_bon 分支) 之上。

沿用 Q2.5 版设计（不重调权重，只重定标 z-params）：
  score = z(Delta) - BETA*z(CE_top5mean) + GAMMA*z(VAS_content)
  BETA, GAMMA = 0.25, 2.0  — 由 analyze_bon_features.py 在 haloquest v1 pool 上
  网格搜索预注册，这里只是复用，不重新网格搜索。

z-params 重定标：复用已产出的 outputs/bon_probe_qwen3/feat_full.json
（probe_bon_features_qwen3.py 对 v1 pool 的产出，不重跑探针），在其 haloquest
行上重新抽 100 dev 做 z-params（同 Q2.5 版 dev_zpars() 逐字节复刻的抽样协议：
seed=42，shuffle 全量 id 后取前100）。

VAS/CE 特征来源分两段，因为 passthrough_bon 分支在 v2 契约 prompt 下生成的
候选是全新样本（v1 pool 里没有）：
  - z-params 定标：从 feat_full.json（v1 pool 特征）算 mean/std（同 Q2.5：
    dev_zpars 也是在 V1_FEAT 上算的，不是在待重排的候选上算）。
  - 待重排候选自身的 VAS/CE：必须现算（Q2.5 版同样要求“one GPU pass”，
    见 bon_stacked_feat_combo.py 顶部 docstring）——对层3产出里
    branch=='passthrough_bon' 的候选，用 v2 契约 prompt + 各自图片，做一次
    teacher-forced forward 取 CE/VAS（复用 probe_bon_features_qwen3.py 的
    forward_with_feats/evidence_mask/load_model_eager，避免重复实现 Qwen3
    的 attn-hook 适配）。非 passthrough 分支原样直通。

VAS primary layer 用 '24'（Qwen3 48层重定标主层，对应 Q2.5 的 '14'/28层）。

判据（负责人）：四层终值（Token-F1） > 阶段2 契约+门合体 A4 终值 40.93。

Usage:
    CUDA_VISIBLE_DEVICES=0 python bon_stacked_feat_combo_qwen3.py
"""
import os, sys, json, random, time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'OmniConfess', 'methods'))
sys.path.insert(0, HERE)
from shared import (  # noqa: E402
    load_dataset, load_media_for_item, get_system_prompt, get_user_prompt,
)
from shared.qwen3_shim import (  # noqa: E402
    build_user_content_with_media_tokens, make_messages,
)
from probe_bon_features_qwen3 import (  # noqa: E402
    forward_with_feats, build_view, evidence_mask, load_model_eager,
    register_vas_hooks, VAS_LAYERS, VAS_PRIMARY,
)
from bon_grounding import token_class  # noqa: E402

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("MMHALU_MODEL_ROOT", "/ultrafast_share/tsq/MODELS")

STACKED = os.path.join(HERE, 'outputs', 'haloquest_bon_stacked_qwen3',
                       'haloquest_bon_stacked_qwen3_full.jsonl')
V1_FEAT = os.path.join(HERE, 'outputs', 'bon_probe_qwen3', 'feat_full.json')
BETA, GAMMA = 0.25, 2.0     # fixed by the phase-1 (Q2.5) dev grid; NOT re-tuned here
VAS_KEY = str(VAS_PRIMARY)  # '24'


def dev_zpars():
    """Re-derive the dev(100) z-params of the calibrated scorer, on
    feat_full.json (v1-pool haloquest features) — same sampling protocol as
    the Q2.5 dev_zpars() (seed=42 shuffle, first 100)."""
    rows = json.load(open(V1_FEAT))['rows']

    def feats(c):
        v = c['VAS'][VAS_KEY]['vas_content']
        if v is None:
            v = c['VAS'][VAS_KEY]['vas_all']
        g = c.get('g_traj_mean')
        if g is None:
            g = 0.0
        return (g, c['CE']['CE_top5mean'], v)

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

    processor, model = load_model_eager()
    tokenizer = processor.tokenizer
    thinker = model  # no .thinker hop on Qwen3 (self-aliased in load_model_eager)
    handles = register_vas_hooks(thinker, VAS_LAYERS)
    print(f'[hooks] registered on layers {VAS_LAYERS} (primary {VAS_PRIMARY})', flush=True)
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
        print(f'[resume] skipping {len(done)}', flush=True)
    outf = open(out_path, 'a')

    t0 = time.time()
    n = 0
    for rec in recs:
        if rec['id'] in done:
            continue
        if rec['branch'] != 'passthrough_bon' or not rec.get('trajectories'):
            outf.write(json.dumps(rec, ensure_ascii=False) + '\n')
            outf.flush()
            continue
        item = by_id[rec['id']]
        try:
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
                f = forward_with_feats(thinker, clean, gen_ids, ev_mask, content_pos)
                gs = tr.get('grounding_scores') or {}
                g = gs.get('g_traj_mean', 0.0) or 0.0
                ce = f['CE']['CE_top5mean']
                v = f['VAS'][VAS_PRIMARY]['vas_content']
                if v is None:
                    v = f['VAS'][VAS_PRIMARY]['vas_all']
                z = lambda x, i: (x - zp[i][0]) / zp[i][1]
                score = z(g, 0) - BETA * z(ce, 1) + GAMMA * z(v, 2)
                cands.append({'rollout': tr['rollout'], 'score': score,
                              'g_traj': g, 'CE': ce, 'VAS': v})
                if score > bv:
                    bv, best = score, tr
            rec = dict(rec)
            rec['method'] = 'bon_stacked_feat_combo_qwen3'
            rec['pred'] = best['pred']
            rec['raw_output'] = best['pred']
            rec['rank_score'] = f'feat_combo_beta{BETA}_gamma{GAMMA}_devz'
            rec['selected_rollout_feat'] = best['rollout']
            rec['feat_combo_candidates'] = cands
        except Exception as e:
            print(f'  [ERROR] {rec["id"]}: {e}', flush=True)
            torch.cuda.empty_cache()
            rec = dict(rec)
            rec['method'] = 'bon_stacked_feat_combo_qwen3_fallback'
            rec['feat_combo_error'] = str(e)
        outf.write(json.dumps(rec, ensure_ascii=False) + '\n')
        outf.flush()
        n += 1
        if n % 20 == 0:
            print(f'  [{n}] {time.time()-t0:.0f}s', flush=True)
        torch.cuda.empty_cache()
    outf.close()
    for h in handles:
        h.remove()

    # random-selection control arm (same rollouts, same recorded random picks)
    ctrl = out_path.replace('_feat_combo.jsonl', '_random_sel.jsonl')
    with open(out_path) as fin, open(ctrl, 'w') as fout:
        for line in fin:
            r = json.loads(line)
            if r.get('random_sel_rollout') is not None and r.get('trajectories'):
                r['pred'] = r['trajectories'][r['random_sel_rollout']]['pred']
            r['method'] = 'bon_stacked_qwen3_random_sel'
            fout.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'[done] {time.time()-t0:.0f}s -> {out_path}', flush=True)


if __name__ == '__main__':
    main()
