#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""cmm_prompt_iter.py -- dev-set tooling for the cmm x CF-Confess prompt
evolution program (PROMPT_PROGRAM.md, v1->v5).

Subcommands
-----------
build-dev : stratified 1/4 dev split of OmniHalluBench/cmm/annotations/cmm.json.
    Stratification: (group_key x gt) round-robin, offset = stratum index,
    exactly the PROMPT_PROGRAM.md rule ("按 group_key（亚组）分层轮转抽样 1/4，
    判断类按 gt 标签再分层"). Items keep their ORIGINAL cmm.json index injected
    as `id`, so result rows pair 1:1 with the full-run rows (which are in
    cmm.json order: shard round-robin concat, verified by question match).
    Output: OmniHalluBench/cmm/dev_quarter/cmm_dev_quarter.json + manifest.json
    (fixed, reused verbatim for every prompt version).

eval : score one or more dev result jsonl files against the v1+CF reference
    (extracted from the full 721 run, no re-run needed: greedy deterministic).
    Reports F1/acc/yes-rate/TPR/FPR/Youden overall + per subgroup, paired
    fix/break vs reference, McNemar. Decision rule (pre-registered):
    keep version v if dev F1 >= ref F1 + 1.0 AND dev Youden >= ref Youden.
"""
import json
import os
import sys
import argparse
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'OmniHalluBench', 'cmm', 'cmm.json')
DEV_DIR = os.path.join(HERE, 'OmniHalluBench', 'cmm', 'dev_quarter')
DEV_JSON = os.path.join(DEV_DIR, 'cmm_dev_quarter.json')
MANIFEST = os.path.join(DEV_DIR, 'manifest.json')
REF_FULL = os.path.join(HERE, 'outputs', 'cmm_cf_confess',
                        'cf_lam1.0_full_single_scored.jsonl')
BASE_FULL = os.path.join(HERE, 'outputs', 'cmm_base_model',
                         'cmm_base_model_full_single_scored.jsonl')

SUBGROUPS = ['overrely_audio_ignore_visual', 'overrely_visual_ignore_audio']


def build_dev():
    data = json.load(open(DATA))
    assert len(data) == 721, len(data)
    strata = defaultdict(list)
    for i, it in enumerate(data):
        strata[(it['facets']['hallucination_type'], it['label'])].append(i)
    order = sorted(strata)          # deterministic stratum order
    pick = []
    for k, key in enumerate(order):
        idxs = strata[key]          # already in cmm.json order
        pick.extend(idxs[k % 4::4])  # round-robin, offset = stratum index
    pick = sorted(pick)
    out = []
    for i in pick:
        it = dict(data[i])
        it['id'] = str(i)           # original index -> stable result id
        out.append(it)
    os.makedirs(DEV_DIR, exist_ok=True)
    json.dump(out, open(DEV_JSON, 'w'), ensure_ascii=False, indent=1)
    manifest = {
        'source': 'OmniHalluBench/cmm/annotations/cmm.json (721, order = full-run order)',
        'rule': 'stratified (group_key x gt) round-robin 1/4, offset=stratum idx',
        'stratum_order': [list(k) for k in order],
        'stratum_sizes': {f'{k[0]}|{k[1]}': len(v) for k, v in strata.items()},
        'stratum_picked': {f'{k[0]}|{k[1]}': len([i for i in strata[k] if i in set(pick)])
                           for k in order},
        'n': len(pick),
        'ids': pick,
        'fixed': True,
        'note': 'frozen dev split, reused verbatim for v2..v5 and final full runs',
    }
    json.dump(manifest, open(MANIFEST, 'w'), indent=1)
    print(f'dev_quarter: n={len(pick)} -> {DEV_JSON}')
    print('strata:', {f'{k[0][:12]}|{k[1]}': len(v) for k, v in strata.items()})


# ----------------------------- scoring -------------------------------------

def m(rows):
    yt = [1 if r['gt'] == 'yes' else 0 for r in rows]
    yp = [1 if r['pred'] == 'yes' else 0 for r in rows]
    tp = sum(1 for g, p in zip(yt, yp) if g == 1 and p == 1)
    fn = sum(1 for g, p in zip(yt, yp) if g == 1 and p == 0)
    tn = sum(1 for g, p in zip(yt, yp) if g == 0 and p == 0)
    fp = sum(1 for g, p in zip(yt, yp) if g == 0 and p == 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return dict(n=len(yt), F1=round(f1 * 100, 2), acc=round((tp + tn) / len(yt) * 100, 2),
                yes=round(float(np.mean(yp)) * 100, 1), TP=tp, FP=fp, TN=tn, FN=fn,
                TPR=round(rec, 3), FPR=round(fp / max(fp + tn, 1), 3),
                Youden=round(rec - fp / max(fp + tn, 1), 3))


def mcnemar(a, b, yt):
    b01 = int(sum(1 for x, y, g in zip(a, b, yt) if x != y and x == g))
    b10 = int(sum(1 for x, y, g in zip(a, b, yt) if x != y and y == g))
    n = b01 + b10
    if n == 0:
        return b01, b10, 1.0
    from math import comb
    p = sum(comb(n, k) for k in range(min(b01, b10) + 1)) / 2 ** n * 2
    return b01, b10, min(1.0, p)


def score_pred(pred_str):
    """Same binarization as the repo scorer for yes/no judgement rows."""
    s = str(pred_str).lower().strip()
    if s.startswith(('yes', '\\boxed{yes}')):
        return 1
    if s.startswith(('no', '\\boxed{no}')):
        return 0
    return 2


def _full_run_map(full_rows):
    """full-run jsonl rows are in 8-shard CONCAT order (shards are contiguous
    slices of cmm.json), and the in-row `id` is the shard-local row number.
    Map each row to its original cmm.json index via the shard items' uid."""
    data = json.load(open(DATA))
    uid2idx = {d['uid']: i for i, d in enumerate(data)}
    concat = []
    for sh in range(8):
        concat.extend(json.load(open(os.path.join(
            HERE, 'OmniHalluBench', 'cmm', f'cmm_shard{sh}of8.json'))))
    assert len(concat) == len(full_rows) == 721
    out = {}
    for row, item in zip(full_rows, concat):
        assert row['question'] == item['question']
        out[str(uid2idx[item['uid']])] = row
    return out


def eval_arms(paths):
    ids = {str(i) for i in json.load(open(MANIFEST))['ids']}
    ref_full = [json.loads(l) for l in open(REF_FULL)]
    base_full = [json.loads(l) for l in open(BASE_FULL)]
    # key by str(original cmm.json index) on both sides
    ref_all = _full_run_map(ref_full)
    base_all = _full_run_map(base_full)
    ref_map = {i: ref_all[i] for i in ids}
    assert len(ref_map) == len(ids)
    base_map = {i: base_all[i] for i in ids}

    print('== reference v1+CF on dev (from full 721 run) ==')
    print('overall:', m(list(ref_map.values())))
    for g in SUBGROUPS:
        print(f'  {g}:', m([r for r in ref_map.values() if r['group_key'] == g]))
    ref_m, ref_y = m(list(ref_map.values()))['F1'], m(list(ref_map.values()))['Youden']

    for path in paths:
        rows = [json.loads(l) for l in open(path)]
        rows = {r['id']: r for r in rows if r['id'] in ids}
        common = sorted(set(rows) & set(ref_map), key=int)
        print(f'\n== {os.path.basename(path)} (n={len(common)}/{len(ids)}) ==')
        arm = m([rows[i] for i in common])
        print('overall:', arm)
        for g in SUBGROUPS:
            print(f'  {g}:', m([rows[i] for i in common if rows[i]['group_key'] == g]))
        a = [1 if rows[i]['pred'] == 'yes' else 0 for i in common]
        b = [1 if ref_map[i]['pred'] == 'yes' else 0 for i in common]
        yt = [1 if rows[i]['gt'] == 'yes' else 0 for i in common]
        fix = sum(1 for x, y, g in zip(a, b, yt) if x != y and x == g)
        brk = sum(1 for x, y, g in zip(a, b, yt) if x != y and x != g)
        b01, b10, p = mcnemar(a, b, yt)
        print(f'paired vs v1: fix {fix} / break {brk}  McNemar {b01}:{b10} p={p:.2e}')
        d_f1 = arm['F1'] - ref_m
        d_y = round(arm['Youden'] - ref_y, 3)
        keep = (d_f1 >= 1.0) and (d_y >= 0)
        print(f'GATE: dF1 {d_f1:+.2f} (need >= +1.00), dYouden {d_y:+.3f} (need >= 0) '
              f'-> {"KEEP" if keep else "ROLLBACK"}')
        # subgroup flip detail
        for g in SUBGROUPS:
            sel = [i for i in common if rows[i]['group_key'] == g]
            f2 = sum(1 for i in sel if rows[i]['pred'] != ref_map[i]['pred']
                     and rows[i]['pred'] == rows[i]['gt'])
            b2 = sum(1 for i in sel if rows[i]['pred'] != ref_map[i]['pred']
                     and rows[i]['pred'] != rows[i]['gt'])
            print(f'  {g.split("_")[1]}: fix {f2} / break {b2}')


def eval_full(shard_glob_prefix):
    """Score a full-721 prompt-version run given its shard file prefix, e.g.
    outputs/cmm_cf_confess/cf_lam1.0_v5_full  (files ..._shard{k}of4... jsonl,
    produced on OmniHalluBench/cmm/cmm_shard{k}of4.json). Rows are mapped back
    to original cmm.json indices via the shard files' uid; paired against the
    v1+CF full run (mapped via the of8 shards) and the base run."""
    import glob as _glob
    parts = sorted(_glob.glob(shard_glob_prefix + '_shard*.jsonl'),
                   key=lambda p: int(p.split('_shard')[1].split('.')[0]))
    assert len(parts) == 4, parts
    shard_files = [json.load(open(os.path.join(
        HERE, 'OmniHalluBench', 'cmm', f'cmm_shard{k}of4.json'))) for k in range(4)]
    data = json.load(open(DATA))
    uid2idx = {d['uid']: i for i, d in enumerate(data)}
    arm = {}
    for p, shard in zip(parts, shard_files):
        for row, item in zip([json.loads(l) for l in open(p)], shard):
            assert row['question'] == item['question']
            arm[str(uid2idx[item['uid']])] = row
    assert len(arm) == 721, len(arm)

    ref_all = _full_run_map([json.loads(l) for l in open(REF_FULL)])
    base_all = _full_run_map([json.loads(l) for l in open(BASE_FULL)])

    print(f'== {shard_glob_prefix} (n={len(arm)}) ==')
    print('arm  :', m(list(arm.values())))
    print('v1+CF:', m(list(ref_all.values())))
    print('base :', m(list(base_all.values())))
    for g in SUBGROUPS:
        print(f'  {g}')
        print('    arm  :', m([r for r in arm.values() if r['group_key'] == g]))
        print('    v1+CF:', m([r for r in ref_all.values() if r['group_key'] == g]))

    for label, other in [('v1+CF', ref_all), ('base', base_all)]:
        ks = sorted(arm, key=int)
        a = [1 if arm[i]['pred'] == 'yes' else 0 for i in ks]
        b = [1 if other[i]['pred'] == 'yes' else 0 for i in ks]
        yt = [1 if arm[i]['gt'] == 'yes' else 0 for i in ks]
        fix = sum(1 for x, y, g in zip(a, b, yt) if x != y and x == g)
        brk = sum(1 for x, y, g in zip(a, b, yt) if x != y and x != g)
        yn = sum(1 for x, y in zip(a, b) if y == 1 and x == 0)
        ny = sum(1 for x, y in zip(a, b) if y == 0 and x == 1)
        b01, b10, p = mcnemar(a, b, yt)
        print(f'paired vs {label}: N->Y {ny}, Y->N {yn}, fix {fix}, break {brk}, '
              f'McNemar {b01}:{b10} p={p:.2e}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['build-dev', 'eval', 'full'])
    ap.add_argument('paths', nargs='*')
    args = ap.parse_args()
    if args.cmd == 'build-dev':
        build_dev()
    elif args.cmd == 'full':
        eval_full(args.paths[0])
    else:
        eval_arms(args.paths)


if __name__ == '__main__':
    main()
