# -*- coding: utf-8 -*-
"""analyze_bon_features.py — 第一阶段判别力检查（离线，零 GPU）.

Consumes outputs/bon_probe/{ragtruth,haloquest}_bon_feat.json (from
probe_bon_features.py) and answers the coordinator's two questions:

  1. VAS sanity (first 50 samples / all candidates):
       - within-pool AUC(VAS, good-vs-bad candidate) with good/bad labeled by
         per-candidate F1 (sample-median split, >= median = good)
       - hallucination subgroup: haloquest negative-gt (EVADERE) items with an
         affirmative candidate; haloquest affirmative-on-negative-gt candidates
         are hallucinated by construction -> VAS should be lower
       - layer sanity 9 vs 14 vs 23
  2. three-feature combo on dev(100)/test(rest):
       score = 1*g_content - beta*CE + gamma*VAS   (features z-scored on dev)
       beta in {0,.25,.5,1,2}, gamma in {0,+-0.25,+-0.5,+-1,+-2}; single-feature
       arms (CE only, VAS only) for reference. Selection = within-sample argmax.
       Gate: best-on-dev combo's TEST F1 > pure-delta (beta=gamma=0) TEST F1 + 1.0
"""
import json, os, sys, random
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'scripts'))
from tokenf1_fullset import token_f1  # noqa: E402

import re
EVADERE = re.compile(
    r"(not\s+visible|can\s*not|cannot|unable\s+to|do(es)?\s+not\s+(contain|show|"
    r"provide|appear|seem|determine)|not\s+(shown|seen|present|clearly)|"
    r"there\s+(is|are)\s+no|no\s+one|not\s+wearing|is\s+no\s|hard\s+to\s"
    r"(see|tell|determine)|insufficient|not\s+enough\s+detail)", re.I)


def auc(hits, misses):
    from scipy.stats import mannwhitneyu
    if len(hits) < 2 or len(misses) < 2:
        return float('nan')
    u, p = mannwhitneyu(hits, misses, alternative='greater')
    return float(u / (len(hits) * len(misses))), float(p)


def load(ds):
    d = json.load(open(os.path.join(HERE, 'outputs', 'bon_probe', f'{ds}_bon_feat.json')))
    rows = d['rows']
    # join the ALWAYS-DEFINED Delta (g_traj_mean) from the BoN pool file:
    # g_content is None for claim-less (evasion) rollouts, which would silently
    # exclude them from selection on haloquest (evasions are high-F1 there).
    pool = {r['id']: {t['rollout']: t for t in r['trajectories']}
            for r in map(json.loads, open(os.path.join(
                HERE, 'outputs', f'{ds}_bon_grounding', f'{ds}_bon_grounding.jsonl')))}
    for r in rows:
        for c in r['candidates']:
            gs = pool[r['id']][c['rollout']].get('grounding_scores')
            c['g_traj'] = gs['g_traj_mean'] if gs else 0.0
    return rows


def feats(c, vas_key='vas_content', layer='14'):
    v = c['VAS'][layer][vas_key]
    if v is None:
        v = c['VAS'][layer]['vas_all']
    return (c['g_traj'], c['CE']['CE_top5mean'], v)


def analyze(ds):
    rows = load(ds)
    print(f'\n===================== {ds} (n={len(rows)}) =====================')

    # ---- 1. VAS sanity -------------------------------------------------------
    san = rows[:50]
    for layer in ('9', '14', '23'):
        for vk in ('vas_content', 'vas_all'):
            good, bad = [], []
            for r in san:
                f1s = [c['f1'] for c in r['candidates']]
                med = float(np.median(f1s))
                for c in r['candidates']:
                    v = c['VAS'][layer][vk]
                    if v is None:
                        v = c['VAS'][layer]['vas_all']
                    if v is None:
                        continue
                    (good if c['f1'] >= med else bad).append(v)
            a = auc(good, bad)
            print(f'[sanity] VAS L{layer} {vk}: good {np.mean(good):.4f} vs bad '
                  f'{np.mean(bad):.4f} AUC={a[0]:.3f} p={a[1]:.1e}' if a == a else
                  f'[sanity] VAS L{layer} {vk}: degenerate')
    # CE / g_content sanity on the same labels for footing
    for name, fn in (('CE_top5mean', lambda c: -c['CE']['CE_top5mean']),
                     ('g_content', lambda c: c['g_content'])):
        good, bad = [], []
        for r in san:
            med = float(np.median([c['f1'] for c in r['candidates']]))
            for c in r['candidates']:
                v = fn(c)
                if v is None:
                    continue
                (good if c['f1'] >= med else bad).append(v)
        if len(good) < 2 or len(bad) < 2:
            print(f'[sanity] {name}: degenerate')
            continue
        a = auc(good, bad)
        print(f'[sanity] {name}: good {np.mean(good):.3f} vs bad {np.mean(bad):.3f} '
              f'AUC={a[0]:.3f} p={a[1]:.1e}')

    # hallucination subgroup (haloquest): affirmative candidates on negative-gt
    if ds == 'haloquest':
        hall_v, aff_v = [], []
        for r in rows:
            neg = bool(EVADERE.search(r['gt'] or ''))
            if not neg:
                continue
            for c in r['candidates']:
                v = c['VAS']['14']['vas_content']
                if v is None:
                    v = c['VAS']['14']['vas_all']
                if v is None:
                    continue
                if EVADERE.search(_pred_text(ds, r, c)):
                    hall_v.append(v)   # evasion on a negative-gt item (fine)
                else:
                    aff_v.append(v)    # affirmative on negative-gt = hallucinated
        if aff_v:
            a = auc(hall_v, aff_v)
            print(f'[sanity][halluc] VAS on neg-gt items: evasion {np.mean(hall_v):.4f}'
                  f' (n={len(hall_v)}) vs hallucinated-affirm {np.mean(aff_v):.4f}'
                  f' (n={len(aff_v)}), AUC(evasion>halluc)={a[0]:.3f} p={a[1]:.1e}')

    # ---- 2. dev/test combo ----------------------------------------------------
    rng = random.Random(42)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    dev_idx = set(idx[:100])
    dev = [rows[i] for i in range(len(rows)) if i in dev_idx]
    test = [rows[i] for i in range(len(rows)) if i not in dev_idx]

    def zfit(pool, fi):
        vals = [feats(c)[fi] for r in pool for c in r['candidates']]
        return float(np.mean(vals)), float(np.std(vals) + 1e-9)

    zpars = [zfit(dev, i) for i in range(3)]
    Z = lambda c, i: (feats(c)[i] - zpars[i][0]) / zpars[i][1]

    def sel_f1(pool, w):
        out = []
        for r in pool:
            best, bv = None, -1e18
            for c in r['candidates']:
                v = w[0] * Z(c, 0) + w[1] * Z(c, 1) + w[2] * Z(c, 2)
                if v > bv:
                    bv, best = v, c
            out.append(best['f1'] if best else 0.0)
        return 100 * float(np.mean(out))

    base_w = (1.0, 0.0, 0.0)
    f1_base_dev = sel_f1(dev, base_w)
    best = (None, -1)
    grid = []
    for beta in (0.0, 0.25, 0.5, 1.0, 2.0):
        for gamma in (0.0, 0.25, 0.5, 1.0, 2.0, -0.25, -0.5, -1.0, -2.0):
            w = (1.0, -beta, gamma)
            fd = sel_f1(dev, w)
            grid.append((fd, beta, gamma))
            if fd > best[1]:
                best = ((beta, gamma), fd)
    grid.sort(reverse=True)
    f1_base_test = sel_f1(test, base_w)
    bb, gg = best[0]
    f1_combo_test = sel_f1(test, (1.0, -bb, gg))
    # single-feature arms (dev + test)
    f1_ce_dev = sel_f1(dev, (0.0, -1.0, 0.0))
    f1_vas_dev = sel_f1(dev, (0.0, 0.0, 1.0))
    f1_ce_test = sel_f1(test, (0.0, -1.0, 0.0))
    f1_vas_test = sel_f1(test, (0.0, 0.0, 1.0))
    f1_rand = 100 * float(np.mean([np.mean([c['f1'] for c in r['candidates']])
                                   for r in test]))
    f1_oracle = 100 * float(np.mean([max(c['f1'] for c in r['candidates'])
                                     for r in test]))
    print(f'[combo] dev base (pure delta) {f1_base_dev:.2f} | best dev {best[1]:.2f} '
          f'at beta={bb} gamma={gg}')
    print(f'[combo] TEST pure-delta {f1_base_test:.2f} | combo {f1_combo_test:.2f} '
          f'(delta {f1_combo_test - f1_base_test:+.2f}) | CE-only {f1_ce_test:.2f} | '
          f'VAS-only {f1_vas_test:.2f}')
    print(f'[combo] TEST random {f1_rand:.2f} oracle {f1_oracle:.2f}')
    print(f'[single] dev: CE-only {f1_ce_dev:.2f} VAS-only {f1_vas_dev:.2f}')
    print(f'[combo] top-5 dev grid: ' + ', '.join(
        f'b={b} g={g} -> {v:.2f}' for v, b, g in grid[:5]))
    gate = f1_combo_test > f1_base_test + 1.0
    print(f'[gate] combo > pure-delta + 1.0 on test: {gate}')
    return {'dataset': ds, 'f1_base_test': round(f1_base_test, 2),
            'f1_combo_test': round(f1_combo_test, 2),
            'best_beta_gamma': [bb, gg], 'gate_pass': bool(gate),
            'f1_ce_only': round(f1_ce_test, 2), 'f1_vas_only': round(f1_vas_test, 2),
            'f1_random': round(f1_rand, 2), 'f1_oracle': round(f1_oracle, 2)}


_PRED_CACHE = {}


def _pred_text(ds, r, c):
    """Recover candidate pred text from the pool file (trajectories)."""
    key = (ds, r['id'])
    if key not in _PRED_CACHE:
        pool = os.path.join(HERE, 'outputs', f'{ds}_bon_grounding',
                            f'{ds}_bon_grounding.jsonl')
        _PRED_CACHE[key] = {t['rollout']: t['pred'] for t in
                            json.loads(next(l for l in open(pool)
                                            if json.loads(l)['id'] == r['id']))['trajectories']}
    return _PRED_CACHE[key][c['rollout']]


if __name__ == '__main__':
    out = [analyze(ds) for ds in ('ragtruth', 'haloquest')]
    json.dump(out, open(os.path.join(HERE, 'outputs', 'bon_probe',
                                     'feature_combo_result.json'), 'w'),
              ensure_ascii=False, indent=1)
    print('\n[done] -> outputs/bon_probe/feature_combo_result.json')
