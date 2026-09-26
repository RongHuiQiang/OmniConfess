# -*- coding: utf-8 -*-
"""answerability_gate_qwen3.py — Qwen3-Omni port of answerability_gate.py.

Stage-1 (probe verdict) + Stage-2 (operator) of the ANSWERABILITY GATE, run
offline over the Qwen3 teacher-forced probe forwards (answerability_probe_qwen3.py).

Logic is byte-identical to answerability_gate.py (same feature set, same 5-fold
CV scorer selection on dev, same tau/template dev grid, same >0.70 test flag
precision criterion). Only three paths change for the Qwen3 arm:
  * default --probe  -> the Qwen3 probe json
  * full-set filler  -> the Qwen3 base_model haloquest trajectory (for the 3
                        image-missing items kept untouched, 365 vs 365)
  * default --outdir -> outputs/answerability_qwen3

The scorer machinery, templates, token_f1, stratified_split are imported from
answerability_gate.py (backbone-agnostic — they operate on the probe rows).

Usage:
    python answerability_gate_qwen3.py \
        --probe outputs/answerability_qwen3/haloquest_answerability_probe_qwen3_p3.json
"""
import os, sys, json, argparse, random

import numpy as np
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# reuse the full Qwen2.5 gate machinery (all backbone-agnostic post-probe logic)
from answerability_gate import (  # noqa: E402
    token_f1, TEMPLATES, BARECOUNT, FEATURES, feature_vector,
    SCORERS, auc, stratified_split, is_flagged, flag_stats, apply_gate,
)

Q3_BASE_SCORED = ('/ultrafast_share/tsq/qwen3omni_outputs/haloquest_qwen3omni/'
                  'haloquest_qwen3omni_full_single_scored.jsonl')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--probe', default=os.path.join(
        HERE, 'outputs', 'answerability_qwen3',
        'haloquest_answerability_probe_qwen3_p3.json'))
    ap.add_argument('--base_scored', default=Q3_BASE_SCORED,
                    help='Qwen3 base_model trajectory (full-set filler source)')
    ap.add_argument('--n_dev', type=int, default=100)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--outdir', default=os.path.join(HERE, 'outputs', 'answerability_qwen3'))
    args = ap.parse_args()

    d = json.load(open(args.probe))
    rows = [r for r in d['rows'] if r['ok']]
    for r in rows:
        r['base_f1f'] = token_f1(r['pred'], r['gt'])
        r['barecount'] = bool(BARECOUNT.match((r['pred'] or '').strip()))
    n = len(rows)
    n_ng = sum(1 for r in rows if r['needs_gate'])
    n_pos = sum(1 for r in rows if not r['neg_gt'])
    n_evade = sum(1 for r in rows if r['pred_evade'])
    print(f'[pool] n={n} neg_gt={sum(r["neg_gt"] for r in rows)} pos_gt={n_pos} '
          f'pred_evade={n_evade} needs_gate={n_ng} barecount={sum(r["barecount"] for r in rows)}')

    dev, test = stratified_split(rows, args.n_dev, args.seed)
    pool_dev = [r for r in dev if not r['pred_evade']]
    pool_test = [r for r in test if not r['pred_evade']]
    pool_all = [r for r in rows if not r['pred_evade']]

    # ---- scorer selection: 5-fold CV on dev, criterion = CV precision@~24% --
    def cv_precision(scorer_name, k=5):
        make = SCORERS[scorer_name]
        rng = random.Random(7)
        idx = list(range(len(pool_dev)))
        rng.shuffle(idx)
        folds = [idx[i::k] for i in range(k)]
        precs = []
        for i in range(k):
            va = [pool_dev[j] for j in folds[i]]
            tr = [pool_dev[j] for j in idx if j not in folds[i]]
            if not any(r['needs_gate'] for r in tr) or not any(
                    not r['needs_gate'] for r in tr):
                continue
            sc = make(tr)
            tr_sc = [sc(r) for r in tr if not r['barecount']]
            tau = float(np.quantile(tr_sc, 0.24))
            st = flag_stats(va, sc, tau)
            if st['n_flag'] >= 2:
                precs.append(st['precision'])
        return float(np.mean(precs)) if precs else 0.0

    cv_scores = {name: round(cv_precision(name), 4) for name in SCORERS}
    best_scorer = max(cv_scores, key=cv_scores.get)
    score = SCORERS[best_scorer](pool_dev)
    print(f'[scorer CV on dev] {cv_scores} -> {best_scorer}')

    # ---- stage-1 report ----------------------------------------------------
    pos = [score(r) for r in pool_test if not r['neg_gt']]   # answerable
    neg = [score(r) for r in pool_test if r['needs_gate']]
    a, p = auc(pos, neg)
    rngb = random.Random(0)
    aucs = []
    pool_pairs = [(score(r), r['needs_gate']) for r in pool_test]
    for _ in range(2000):
        b = [rngb.choice(pool_pairs) for _ in pool_pairs]
        a2, _ = auc([x[0] for x in b if not x[1]], [x[0] for x in b if x[1]])
        if a2 == a2:
            aucs.append(a2)
    report = {'n': n, 'n_dev': len(dev), 'n_test': len(test),
              'n_needs_gate': n_ng, 'n_pos_gt': n_pos, 'n_evade': n_evade,
              'scorer_cv_dev': cv_scores, 'scorer': best_scorer,
              'AUC_score_test': {'n_pos': len(pos), 'n_neg': len(neg),
                                 'AUC': round(a, 4), 'MWU_p': f'{p:.2e}',
                                 'boot_CI95': [round(float(np.quantile(aucs, .025)), 4),
                                               round(float(np.quantile(aucs, .975)), 4)]}}
    for f in FEATURES:
        j = FEATURES.index(f)
        posf = [feature_vector(r)[j] for r in pool_test if not r['neg_gt']]
        negf = [feature_vector(r)[j] for r in pool_test if r['needs_gate']]
        a, p = auc(posf, negf)
        report[f'AUC_{f}_test'] = round(a, 4) if a == a else None

    dev_sc = [score(r) for r in pool_dev if not r['barecount']]
    tau24 = float(np.quantile(dev_sc, 0.24))
    report['probe_gate_test'] = flag_stats(pool_test, score, tau24)
    report['probe_gate_all'] = flag_stats(pool_all, score, tau24)
    report['probe_gate_test']['pass_70'] = bool(
        report['probe_gate_test']['precision'] and
        report['probe_gate_test']['precision'] > 0.70)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    gate_pass = report['probe_gate_test']['pass_70']

    # ================= Stage 2: operator ====================================
    grid = {}
    for rate in (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40):
        tau = float(np.quantile(dev_sc, rate))
        for tname, tpl in TEMPLATES.items():
            edited = apply_gate(dev, score, tau, tpl)
            f1s = [token_f1(e['pred'], e['gt']) for e in edited]
            grid[(tname, round(rate, 2))] = (round(float(np.mean(f1s)), 4),
                                             sum(e['flagged'] for e in edited))
    best = max(grid, key=lambda k: grid[k][0])
    tname, rate = best
    tau = float(np.quantile(dev_sc, rate))
    tpl = TEMPLATES[tname]
    print('\n[dev grid] top-5 (template, flag_rate) -> (dev F1, n_flag):')
    for k in sorted(grid, key=lambda k: grid[k][0], reverse=True)[:5]:
        print('   ', k, grid[k])
    print(f'[dev] base {np.mean([r["base_f1f"] for r in dev]):.4f} -> '
          f'{grid[best][0]:.4f} ({tname}, rate={rate}, tau={tau:.3f})')

    # ---- full-set arms -----------------------------------------------------
    edited_main = apply_gate(rows, score, tau, tpl)
    probe_ids = {r['id'] for r in rows}
    recs_all = [json.loads(l) for l in open(args.base_scored)]
    filler = [{'id': r['id'], 'question': r['question'], 'gt': r['gt'],
               'group_key': r.get('group_key'), 'dataset': 'haloquest',
               'method': 'answerability_gate', 'base_pred': r['pred'],
               'flagged': False, 'pred': r['pred']}
              for r in recs_all if r['id'] not in probe_ids]
    edited_main.extend(filler)
    n_flag = sum(e['flagged'] for e in edited_main)
    eligible = [r['sample_i'] for r in rows if not r['pred_evade']]
    rng2 = random.Random(49)
    rand_ids = set(rng2.sample(eligible, min(n_flag, len(eligible))))
    edited_rand = apply_gate(rows, score, tau, tpl, flag_ids=rand_ids)
    edited_rand.extend(filler)

    base_f1 = {r['id']: r['base_f1f'] for r in rows}
    base_f1.update({e['id']: token_f1(e['base_pred'], e['gt']) for e in filler})
    ng_ids = {r['id'] for r in rows if r['needs_gate']}
    pos_ids = {r['id'] for r in rows if not r['neg_gt']}
    test_ids = {r['id'] for r in test}
    subsets = {
        'overall': lambda e: True,
        'needs_gate(unanswerable affirmed)': lambda e: e['id'] in ng_ids,
        'pos_gt(answerable)': lambda e: e['id'] in pos_ids,
        'test_only': lambda e: e['id'] in test_ids,
    }

    def paired_block(edited):
        out = {}
        for sname, pred_fn in subsets.items():
            ids = [i for i, e in enumerate(edited) if pred_fn(e)]
            a = np.array([token_f1(edited[i]['pred'], edited[i]['gt']) for i in ids])
            b = np.array([base_f1[edited[i]['id']] for i in ids])
            dd = a - b
            wp = wilcoxon(a, b).pvalue if np.any(dd != 0) else 1.0
            out[sname] = {'n': len(ids), 'arm': round(float(a.mean()) * 100, 2),
                          'base': round(float(b.mean()) * 100, 2),
                          'delta_pp': round(float(dd.mean()) * 100, 2),
                          'wilcoxon_p': round(float(wp), 5)}
        return out

    res = {'report_stage1': report, 'gate_pass': gate_pass,
           'backbone': 'Qwen3-Omni-30B-A3B-Instruct',
           'dev_choice': {'template': tname, 'flag_rate_on_dev_pool': rate,
                          'tau': tau,
                          'dev_grid': {f'{k[0]}|{k[1]}': v for k, v in grid.items()}},
           'n_flag_main': n_flag,
           'flag_rate_overall': round(n_flag / n, 4),
           'flag_precision_needs_gate': round(sum(
               1 for e in edited_main if e['flagged'] and e['id'] in ng_ids)
               / max(1, n_flag), 4),
           'main': paired_block(edited_main),
           'randctrl': paired_block(edited_rand)}
    a = np.array([token_f1(e['pred'], e['gt']) for e in edited_main])
    b = np.array([token_f1(e['pred'], e['gt']) for e in edited_rand])
    res['main_vs_rand_p'] = round(float(wilcoxon(a, b).pvalue), 5)

    os.makedirs(args.outdir, exist_ok=True)
    out_main = os.path.join(args.outdir, 'haloquest_answerability_gate_qwen3_full.jsonl')
    with open(out_main, 'w') as f:
        for e in edited_main:
            f.write(json.dumps(e, ensure_ascii=False) + '\n')
    out_rand = os.path.join(args.outdir, 'haloquest_answerability_gate_qwen3_randctrl.jsonl')
    with open(out_rand, 'w') as f:
        for e in edited_rand:
            f.write(json.dumps({**e, 'method': 'answerability_gate_randctrl'},
                               ensure_ascii=False) + '\n')
    res_path = os.path.join(args.outdir, 'answerability_gate_qwen3_result.json')
    json.dump(res, open(res_path, 'w'), indent=2, ensure_ascii=False)
    print('\n[stage2 full-set]')
    print(json.dumps({k: res[k] for k in ('gate_pass', 'n_flag_main',
                                          'flag_rate_overall',
                                          'flag_precision_needs_gate', 'main',
                                          'randctrl', 'main_vs_rand_p')},
                     indent=2, ensure_ascii=False))
    print(f'\n[done] -> {out_main}\n        -> {out_rand}\n        -> {res_path}')

    m = res['main']
    c1 = m['overall']['arm'] > m['overall']['base']
    c2 = m['needs_gate(unanswerable affirmed)']['arm'] >= 30.0
    c3 = (m['pos_gt(answerable)']['base'] - m['pos_gt(answerable)']['arm']) < 1.0
    print(f'\n[CRITERIA] overall>base: {c1} ({m["overall"]["arm"]} vs {m["overall"]["base"]}) | '
          f'needs_gate>=30: {c2} ({m["needs_gate(unanswerable affirmed)"]["arm"]}) | '
          f'pos_gt drop<1pp: {c3} '
          f'({m["pos_gt(answerable)"]["base"] - m["pos_gt(answerable)"]["arm"]:.2f})')


if __name__ == '__main__':
    main()
