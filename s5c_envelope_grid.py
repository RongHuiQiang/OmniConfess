# -*- coding: utf-8 -*-
"""s5c_envelope_grid.py — S5c 前置二（零 GPU）：envelope 离线网格，参数冻结候选.

PRE-REGISTERED criteria (written BEFORE running):
  - Pool: haloquest v1 BoN candidate pool (outputs/bon_probe/haloquest_bon_feat.json,
    365 items x ~4 candidates, per-candidate f1 / g_content / CE / VAS / H_per_tok).
  - Protocol byte-aligned with analyze_bon_features.py: dev = first 100 of
    Random(42) shuffle, z-params fit on dev, selection = argmax score, all F1
    are mean per-candidate token-F1 x100 on the TEST half (265 items).
  - ENV (suffix-max envelope, C1 口径 lit_mech_transfer_analysis.py):
    env = max over i in range(0, len(H)-4, 5) of mean(H[i:]).
  - Grid: score = z(Delta) - beta*z(CEslot) + gamma*z(VAS)
    CEslot in {CE, ENV, (CE+ENV)/2}; beta in {0.15,0.25,0.35}; gamma in {1.5,2,2.5}.
  - Current version (release): CEslot=CE, beta=0.25, gamma=2.0 (terminal 40.14).
  - Selection rule for "offline best": highest DEV F1 in grid (no peeking at test).
  - VERDICT: "worth_online" iff test F1 of offline best > test F1 of current
    version + 0.5 (paired per-item, Wilcoxon signed-rank p reported); else
    "not_worth_online". 负结果照登。
No GPU used.
"""
import json, os, random
import numpy as np
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
FEAT = os.path.join(HERE, 'outputs', 'bon_probe', 'haloquest_bon_feat.json')
OUT = os.path.join(HERE, 'outputs', 's5c_envelope_grid.json')

CURRENT = ('CE', 0.25, 2.0)
CE_SLOTS = ['CE', 'ENV', 'HALF']
BETAS = [0.15, 0.25, 0.35]
GAMMAS = [1.5, 2.0, 2.5]


def env_of(H):
    H = np.asarray(H, float)
    if len(H) < 5:
        return float(H.mean()) if len(H) else np.nan
    suf = [H[i:].mean() for i in range(0, len(H) - 4, 5)]
    return float(max(suf))


def feats(c):
    v = c['VAS']['14']['vas_content']
    if v is None:
        v = c['VAS']['14']['vas_all']
    H = c['CE'].get('H_per_tok')
    return (c['g_content'] if c.get('g_content') is not None else 0.0,
            c['CE']['CE_top5mean'],
            env_of(H) if H else np.nan,
            v)


def main():
    rows = json.load(open(FEAT))['rows']
    for r in rows:
        r['_c'] = []
        for c in r['candidates']:
            g, ce, env, v = feats(c)
            r['_c'].append({'f1': c['f1'], 'g': g, 'CE': ce, 'ENV': env, 'VAS': v,
                            'rollout': c['rollout']})
    rng = random.Random(42)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    dev = [rows[i] for i in idx[:100]]
    test = [rows[i] for i in idx[100:]]

    def zfit(pool, key):
        vals = [c[key] for r in pool for c in r['_c'] if not np.isnan(c[key])]
        return float(np.mean(vals)), float(np.std(vals)) + 1e-9
    zp = {k: zfit(dev, k) for k in ('g', 'CE', 'ENV', 'VAS')}

    def sel(pool, slot, beta, gamma):
        """Returns per-item f1 of argmax-score candidate."""
        out = []
        for r in pool:
            bv, bf = -1e18, 0.0
            for c in r['_c']:
                zc = lambda k: (c[k] - zp[k][0]) / zp[k][1]
                if np.isnan(c['CE']) or np.isnan(c['ENV']) or np.isnan(c['VAS']):
                    continue
                xce = {'CE': zc('CE'), 'ENV': zc('ENV'), 'HALF': 0.5 * (zc('CE') + zc('ENV'))}[slot]
                s = zc('g') - beta * xce + gamma * zc('VAS')
                if s > bv:
                    bv, bf = s, c['f1']
            out.append(bf)
        return np.array(out)

    cur_test = sel(test, *CURRENT)
    cur_dev = sel(dev, *CURRENT)
    cur_full = sel(rows, *CURRENT)
    grid = []
    best = (None, -1)
    for slot in CE_SLOTS:
        for beta in BETAS:
            for gamma in GAMMAS:
                fd = float(sel(dev, slot, beta, gamma).mean() * 100)
                ft = sel(test, slot, beta, gamma)
                ffull = float(sel(rows, slot, beta, gamma).mean() * 100)
                rec = {'slot': slot, 'beta': beta, 'gamma': gamma,
                       'dev_F1': round(fd, 2), 'test_F1': round(float(ft.mean() * 100), 2),
                       'full365_F1': round(ffull, 2)}
                if (slot, beta, gamma) == CURRENT:
                    rec['is_current'] = True
                grid.append(rec)
                if fd > best[1]:
                    best = ((slot, beta, gamma), fd, ft)
    (bs, bb, bg), bd, bt = best
    diff = bt - cur_test
    d = diff[diff != 0]
    if len(d) >= 10:
        w = wilcoxon(bt, cur_test, zero_method='wilcox')
        p = float(w.pvalue)
    else:
        p = float('nan')
    verdict = 'worth_online' if float(diff.mean() * 100) > 0.5 else 'not_worth_online'
    out = {
        'preregistered': {
            'pool': 'haloquest v1 BoN candidate pool (bon_probe/haloquest_bon_feat.json), dev100/test265 Random(42), z on dev',
            'env_def': 'suffix-max envelope: max over stride-5 suffix means of H_per_tok (C1 口径)',
            'grid': 'CEslot{CE,ENV,HALF} x beta{0.15,0.25,0.35} x gamma{1.5,2,2.5}',
            'current_release': 'CE slot, beta=0.25, gamma=2.0 (online terminal Token-F1 40.14)',
            'verdict_rule': 'offline-best(dev-selected) test F1 > current test F1 + 0.5 -> worth_online'},
        'current_version': {'slot': 'CE', 'beta': 0.25, 'gamma': 2.0,
                            'dev_F1': round(float(cur_dev.mean() * 100), 2),
                            'test_F1': round(float(cur_test.mean() * 100), 2),
                            'full365_F1': round(float(cur_full.mean() * 100), 2),
                            'online_terminal_365': 40.14},
        'offline_best_dev_selected': {'slot': bs, 'beta': bb, 'gamma': bg,
                                      'dev_F1': round(bd, 2),
                                      'test_F1': round(float(bt.mean() * 100), 2)},
        'offline_vs_current': {'test_delta_F1': round(float(diff.mean() * 100), 3),
                               'wilcoxon_p': None if np.isnan(p) else round(p, 4),
                               'n_test_items_changed': int((diff != 0).sum())},
        'verdict': verdict,
        'grid': sorted(grid, key=lambda r: -r['dev_F1'])}
    json.dump(out, open(OUT, 'w'), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in out.items() if k != 'grid'}, indent=1))
    print('top5 dev:', grid_sorted := sorted(grid, key=lambda r: -r['dev_F1'])[:5])


if __name__ == '__main__':
    main()
