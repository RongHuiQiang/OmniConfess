#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""lit_mech_transfer_analysis.py -- zero-GPU mechanism-transplant checks from the 9 lit papers.

Pre-registered criteria (written BEFORE running; see TRAJECTORY_ANALYSIS.md §15):
  A. Decision-Tokens two-stage gate vs CF flip-band (cmm/phd/pubmedqa probes):
     - dev half selects params, test half validates.
     - Rule CF: pred' = sign(margin_full + lam * d_cf), lam grid {0.5,1,1.5,2,2.5,3}.
     - Rule DT: s = |d_cf|; trigger iff s > tau_q (dev global quantile, q in {.5,.7,.8,.9})
       AND s > lam2 * s_local (s_local = mean of per-view |d_cf_i|; local spike analog).
       Triggered samples get the same lam* shift; untriggered keep base pred.
     - REPORT upgrade iff DT net gain (dF1 vs base, test half) > CF net gain + 1.0.
     - Jaccard(trigger sets) > 0.9  => "same rule, different skin".
     - avhbench token-level position Jaccard: CF rule = word-initial content tokens with
       max(|d_audio|,|d_video|) > eps (eps dev-calibrated); DT rule = content positions with
       s > tau_q AND s > lam2 * rolling-mean(W=16). Jaccard > 0.9 => same-rule verdict.
  B. Descriptive commitment-point sparsity (ragtruth+haloquest claim probes joined with
     base_model logp_list):
     - Gini over the pooled error-position mass (10 position deciles); earliness =
       mean normalized error position; surprisal(-logp) AUC for error-claim tokens
       vs correct-claim tokens (token level).
     - SUPPORT iff Gini > 0.6 AND AUC > 0.65; else "diffuse errors -> selection/voting".
  C1. InfoDensity entropy trajectory (ragtruth BoN features, per-candidate H_per_tok):
     - end-entropy (mean last 5), decline slope (1st-half mean - 2nd-half mean),
       suffix-max envelope (max over suffix means, window 5);
     - candidate-level Spearman with f1 and partial Spearman controlling g_content.
     - usable iff |rho| > 0.15 AND |partial| > 0.10.
  C2. SmartThinker length calibration (inference-time version):
     - within-sample Spearman(n_tokens, f1) over candidates; population length-f1
       quadratic; haloquest sample length vs correctness.
     - routing justified iff mean within-sample |rho| significant (perm p<0.05).
  C3. TTSP aggregation-loss decomposition (ragtruth BoN, 4 rollouts):
     - oracle = mean max f1; SC = answer-cluster majority (bag-of-words cosine,
       argmax cluster, best f1 inside it); BoN = argmax g_content f1.
     - minority-correct rate = P(best candidate outside majority cluster);
       rescue rate = P(g_content picks a top candidate when it is minority).
"""
import json, glob, sys, argparse
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
from scipy.stats import spearmanr


def gini(x):
    x = np.sort(np.asarray(x, float))
    if x.sum() <= 0:
        return float('nan')
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def half_split(ids, seed=0):
    rng = np.random.default_rng(seed)
    ids = np.array(sorted(set(ids)))
    perm = rng.permutation(len(ids))
    dev = set(ids[perm[: len(ids) // 2]])
    return dev, set(ids) - dev


def sim_pred(m, d, lam):
    return (m + lam * d > 0).astype(int)


# ---------------- A ----------------

def load_probe(path, kind):
    rows = [json.loads(l) for l in open(path)]
    for r in rows:
        if kind == 'cmm':
            r['d'] = r['d_cf_sum']
            r['s_local'] = (abs(r['d_cf_audio']) + abs(r['d_cf_video'])) / 2.0
            r['base'] = 1 if r['pred'] == 'yes' else 0
        elif kind == 'phd':  # validated: -delta_noctx, lam=2 reproduces online CF F1 84.60
            r['d'] = -r['delta_noctx']
            r['s_local'] = abs(r['delta_noimg'])
            r['base'] = 1 if r['pred'] == 'yes' else 0
        elif kind == 'pubmedqa':  # 3-class yes/no/maybe; gt is yes/no, eval on pos=yes
            d = [r['lp_yes_nopassage'] - r['lp_yes_full'],
                 r['lp_no_nopassage'] - r['lp_no_full'],
                 r['lp_maybe_nopassage'] - r['lp_maybe_full']]
            r['d_vec'] = d
            r['d'] = d[0] - d[1]  # yes-vs-no shift (report-only)
            r['s_local'] = float(np.mean(np.abs(d)))
            r['base'] = 1 if r['pred_tok'] == 1 else 0
    return rows


def pred_scores(r, lam):
    """Class scores under CF intervention with strength lam. Binary: returns
    array([no, yes]); pubmedqa: array([yes, no, maybe]) with per-class view shift."""
    if 'd_vec' in r:
        return np.array([r['lp_yes_full'], r['lp_no_full'], r['lp_maybe_full']]) + lam * np.array(r['d_vec'])
    return np.array([r['lp_no_full'], r['lp_yes_full']]) + lam * np.array([0.0, r['d']])


def argmax_pred(r, lam):
    s = pred_scores(r, lam)
    return int(s[1] > s[0]) if 'd_vec' not in r else int(np.argmax(s) == 0)


def eval_rule(rows, preds, name):
    gt = np.array([r['gt_yes'] for r in rows])
    base = np.array([r['base'] for r in rows])
    f1b = f1_score(gt, base, zero_division=0) * 100
    f1 = f1_score(gt, preds, zero_division=0) * 100
    trig = preds != base
    wrong = base != gt
    fixes = int((trig & (base != gt) & (preds == gt)).sum())
    breaks = int((trig & (base == gt) & (preds != gt)).sum())
    prec = fixes / trig.sum() if trig.sum() else float('nan')
    rec = fixes / wrong.sum() if wrong.sum() else float('nan')
    return dict(rule=name, F1=round(f1, 2), base_F1=round(f1b, 2),
                net_gain=round(f1 - f1b, 2), trigger_rate=round(float(trig.mean()) * 100, 1),
                fixes=fixes, breaks=breaks,
                precision=round(float(prec), 3) if trig.sum() else None,
                recall=round(float(rec), 3) if wrong.sum() else None)


def run_A():
    out = {}
    for kind, path in [('cmm', 'outputs/cf_probe/cmm_probe_full.jsonl'),
                       ('phd', 'outputs/cf_probe/phd_probe_full.jsonl'),
                       ('pubmedqa', 'outputs/cf_probe/pubmedqa_probe_full.jsonl')]:
        rows = load_probe(path, kind)
        dev, test = half_split([r['id'] for r in rows])
        rd = [r for r in rows if r['id'] in dev]
        rt = [r for r in rows if r['id'] in test]
        gt_d = np.array([r['gt_yes'] for r in rd])
        best, lam = -1, 1
        for l in [0.5, 1, 1.5, 2, 2.5, 3]:
            f = f1_score(gt_d, [argmax_pred(r, l) for r in rd], zero_division=0)
            if f > best:
                best, lam = f, l
        # validate against known online operator values
        known = {'cmm': None, 'phd': (2.0, 84.60), 'pubmedqa': None}[kind]
        s_d = np.array([abs(r['d_vec'][0] - r['d_vec'][1]) if 'd_vec' in r else abs(r['d'])
                        for r in rd])
        sl_d = np.array([r['s_local'] for r in rd])
        best_dt, bq, bl2 = -1, None, None
        for q in [0.5, 0.7, 0.8, 0.9]:
            tau = np.quantile(s_d, q)
            for l2 in [1.0, 1.5, 2.0]:
                p = np.array([argmax_pred(r, lam) for r in rd], float)
                for i, r in enumerate(rd):
                    if not (s_d[i] > tau and s_d[i] > l2 * sl_d[i]):
                        p[i] = r['base']
                f = f1_score(gt_d, p.astype(int), zero_division=0)
                if f > best_dt:
                    best_dt, bq, bl2 = f, q, l2
        tau = float(np.quantile(s_d, bq))
        res = {'n_dev': len(rd), 'n_test': len(rt), 'lam_star_dev': lam,
               'known_online_lam': known, 'DT_params': {'q': bq, 'lam2': bl2, 'tau': round(tau, 3)},
               'test': []}
        gt_t = np.array([r['gt_yes'] for r in rt])
        p_cf = np.array([argmax_pred(r, lam) for r in rt])
        s_t = np.array([abs(r['d_vec'][0] - r['d_vec'][1]) if 'd_vec' in r else abs(r['d'])
                        for r in rt])
        sl_t = np.array([r['s_local'] for r in rt])
        gate = (s_t > tau) & (s_t > bl2 * sl_t)
        p_dt = p_cf.copy()
        base_t = np.array([r['base'] for r in rt])
        p_dt[~gate] = base_t[~gate]
        res['test'].append(eval_rule(rt, p_cf, 'CF_flip_band'))
        res['test'].append(eval_rule(rt, p_dt, 'DT_two_stage_gate'))
        t_cf, t_dt = set(np.where(p_cf != base_t)[0]), set(np.where(p_dt != base_t)[0])
        j = len(t_cf & t_dt) / len(t_cf | t_dt) if (t_cf | t_dt) else 1.0
        res['trigger_jaccard'] = round(j, 3)
        cfng = res['test'][0]['net_gain']; dtng = res['test'][1]['net_gain']
        res['verdict'] = ('DT_upgradable' if dtng > cfng + 1.0 else
                          ('same_rule_skin' if j > 0.9 else 'DT_not_better'))
        out[kind] = res
    # avhbench token-level position comparison (CF word rule vs DT two-stage on series)
    rows = []
    for f in glob.glob('outputs/cf_probe/avh_probe_shard*of4.jsonl'):
        rows += [json.loads(l) for l in open(f)]
    dev, test = half_split([r['id'] for r in rows], seed=1)
    rd = [r for r in rows if r['id'] in dev]
    rt = [r for r in rows if r['id'] in test]

    def content_positions(r):
        return [i for i, c in enumerate(r['classes']) if c == 'content']

    def s_series(r):
        return np.array([max(abs(a), abs(b)) for a, b in zip(r['delta_audio'], r['delta_video'])])

    all_s = np.concatenate([s_series(r)[content_positions(r)] for r in rd if content_positions(r)])
    out['avhbench_position'] = []
    for q, eps in zip([0.5, 0.7, 0.8, 0.9],
                      np.quantile(all_s, [0.5, 0.7, 0.8, 0.9])):
        tau = np.quantile(all_s, q)
        for l2 in [1.0, 1.5, 2.0]:
            jac = []
            for r in rt:
                pos = content_positions(r)
                if not pos:
                    continue
                s_all = s_series(r)
                sp = s_all[pos]
                sel_cf = set(np.array(pos)[sp > eps])
                roll = np.convolve(s_all, np.ones(16) / 16, mode='same')
                sel_dt = set(np.array(pos)[(sp > tau) & (sp > l2 * roll[pos])])
                if sel_cf or sel_dt:
                    jac.append(len(sel_cf & sel_dt) / len(sel_cf | sel_dt))
            out['avhbench_position'].append(
                {'q': q, 'lam2': l2, 'eps': round(float(eps), 3),
                 'mean_jaccard': round(float(np.mean(jac)), 3), 'n': len(jac)})
    return out


# ---------------- B ----------------
def load_claim(ds):
    d = json.load(open(f'outputs/claim_probe/{ds}_claim_probe_full.json'))
    base = {}
    for f in glob.glob(f'outputs/{ds}_base_model/{ds}_shard*.jsonl'):
        for l in open(f):
            r = json.loads(l)
            base[r['id']] = r
    return d, base


def run_B():
    out = {}
    for ds in ['ragtruth', 'haloquest']:
        d, base = load_claim(ds)
        sm = {s['id']: s for s in d['samples']}
        # error position deciles (claims, in_primary, non-evade)
        us, us_err = [], []
        tok_surp, tok_err = [], []
        for c in d['claims']:
            if not c.get('in_primary') or c.get('is_evade'):
                continue
            s = sm[c['id']]
            u = ((c['lo'] + c['hi']) / 2) / max(s['n_tokens'], 1)
            us.append(u)
            if not c['hit']:
                us_err.append(u)
        for s in d['samples']:
            lp = base.get(s['id'], {}).get('logp_list')
            if lp is None or len(lp) != s['n_tokens']:
                continue
            errmask = np.zeros(s['n_tokens'], bool)
            anyclaim = np.zeros(s['n_tokens'], bool)
            for c in d['claims']:
                if c['id'] != s['id'] or not c.get('in_primary') or c.get('is_evade'):
                    continue
                anyclaim[c['lo']:c['hi']] = True
                if not c['hit']:
                    errmask[c['lo']:c['hi']] = True
            ok = anyclaim & ~errmask
            if errmask.sum() and ok.sum():
                tok_surp += list(-np.array(lp[:s['n_tokens']])[errmask])
                tok_err += [1] * int(errmask.sum())
                tok_surp += list(-np.array(lp[:s['n_tokens']])[ok])
                tok_err += [0] * int(ok.sum())
        # secondary operationalization: word-level (word-initial content tokens whose
        # word is absent from the gt vocabulary) -- robustness check for sparse-claim haloquest
        w_surp, w_err, w_u = [], [], []
        for s in d['samples']:
            lp = base.get(s['id'], {}).get('logp_list')
            if lp is None or len(lp) != s['n_tokens']:
                continue
            bgt = base.get(s['id'], {}).get('gt', '')
            gtw = set(w.lower().strip('.,!?;:"()') for w in str(bgt).split())
            lp = -np.array(lp[:s['n_tokens']])
            for i, (t, c) in enumerate(zip(s['tokens'], s['classes'])):
                if c != 'content' or not t.strip():
                    continue
                w = t.strip().lower().strip('.,!?;:"()')
                if not w:
                    continue
                w_surp.append(float(lp[i])); w_err.append(0 if w in gtw else 1)
                w_u.append(i / max(s['n_tokens'] - 1, 1))
        w_err = np.array(w_err); w_u = np.array(w_u)
        wauc = roc_auc_score(w_err, w_surp) if len(set(w_err)) > 1 else float('nan')
        wdec = np.histogram(w_u[w_err == 1], bins=10, range=(0, 1))[0].astype(float)
        dec = np.histogram(us_err, bins=10, range=(0, 1))[0].astype(float)
        g = gini(dec)
        auc = roc_auc_score(tok_err, tok_surp) if len(set(tok_err)) > 1 else float('nan')
        out[ds] = {'n_claims': len(us), 'n_err_claims': len(us_err),
                   'gini_err_pos_deciles': round(float(g), 3),
                   'err_pos_deciles': dec.astype(int).tolist(),
                   'mean_err_u': round(float(np.mean(us_err)), 3),
                   'frac_err_u_lt_25': round(float(np.mean(np.array(us_err) < 0.25)), 3),
                   'surprisal_AUC': round(float(auc), 4),
                   'n_tok_pairs': len(tok_err),
                   'verdict': 'support_CF' if (g > 0.6 and auc > 0.65) else 'diffuse_selection_voting',
                   'wordlevel': {'n': int(len(w_err)), 'err_rate': round(float(w_err.mean()), 3),
                                 'surprisal_AUC': round(float(wauc), 4),
                                 'gini_err_pos_deciles': round(float(gini(wdec)), 3),
                                 'mean_err_u': round(float(w_u[w_err == 1].mean()) if (w_err == 1).sum() else float('nan'), 3)}}
    return out


# ---------------- C1/C2/C3 ----------------
def partial_spearman(x, y, z):
    rx = spearmanr(x, z).correlation
    ry = spearmanr(y, z).correlation
    rxy = spearmanr(x, y).correlation
    den = np.sqrt((1 - rx ** 2) * (1 - ry ** 2))
    return float((rxy - rx * ry) / den) if den > 0 else float('nan')


def run_C():
    out = {}
    d = json.load(open('outputs/bon_probe/ragtruth_bon_feat.json'))
    rows = d['rows']
    f1s, lens, gs, endH, slope, env, means = [], [], [], [], [], [], []
    within = []
    for r in rows:
        cs = [c for c in r['candidates']]
        if len(cs) < 2:
            continue
        wf1 = [c['f1'] for c in cs]
        wl = [c['n_tokens'] for c in cs]
        if np.std(wl) > 0 and np.std(wf1) > 0:
            within.append(spearmanr(wl, wf1).correlation)
        for c in cs:
            if c.get('g_content') is None or c.get('CE') is None:
                continue
            H = np.array(c['CE']['H_per_tok'])
            if len(H) < 4:
                continue
            f1s.append(c['f1']); lens.append(c['n_tokens']); gs.append(c['g_content'])
            endH.append(H[-5:].mean())
            slope.append(H[:len(H) // 2].mean() - H[len(H) // 2:].mean())
            suf = [H[i:].mean() for i in range(0, len(H) - 4, 5)]
            env.append(max(suf))
            means.append(H.mean())
    f1s, gs = np.array(f1s), np.array(gs)
    # C1
    c1 = {}
    for nm, v in [('end_entropy', endH), ('decline_slope', slope),
                  ('suffix_max_env', env), ('mean_entropy', means)]:
        v = np.array(v)
        c1[nm] = {'spearman_f1': round(float(spearmanr(v, f1s).correlation), 3),
                  'partial_vs_g_content': round(partial_spearman(v, f1s, gs), 3)}
    c1['verdict'] = 'usable' if any(abs(c1[k]['spearman_f1']) > 0.15 and
                                    abs(c1[k]['partial_vs_g_content']) > 0.10
                                    for k in ['end_entropy', 'decline_slope', 'suffix_max_env']) \
        else 'not_usable'
    out['C1_infodensity'] = c1
    # C2
    lens = np.array(lens)
    quad = np.polyfit(np.log(lens), f1s, 2)
    pred = np.polyval(quad, np.log(lens))
    ssr = ((f1s - f1s.mean()) ** 2).sum()
    r2 = 1 - ((f1s - pred) ** 2).sum() / ssr if ssr else float('nan')
    within = np.array([w for w in within if np.isfinite(w)])
    rng = np.random.default_rng(0)
    null = np.array([np.mean(rng.permutation(within)) for _ in range(2000)])
    p = float((np.abs(null) >= abs(within.mean())).mean())
    out['C2_smartthinker'] = {
        'len_f1_logquad_R2': round(float(r2), 3),
        'within_sample_mean_spearman_len_f1': round(float(within.mean()), 3),
        'within_sample_perm_p': p, 'n_samples_within': len(within),
        'verdict': 'routing_justified' if p < 0.05 else 'no_length_signal'}
    # C3 TTSP decomposition
    oracle, sc, bon, minority, rescued = [], [], [], 0, 0
    for r in rows:
        cs = [c for c in r['candidates'] if c.get('g_content') is not None]
        f = np.array([c['f1'] for c in cs])
        if len(cs) < 2:
            continue
        g = np.array([c['g_content'] for c in cs])
        # bag-of-words clustering on candidate texts: use token-list proxy via n-grams of ids?
        # candidates lack text; use f1-similarity unavailable -> cluster by length? No:
        # use simple answer clustering proxy: cosine on token id sets not stored.
        # -> use CE/VAS/g feature-space clustering is NOT answer-level.
        # Answer-level proxy available: 'sampled' flag + identical n_tokens identical f1
        # indicates duplicates; instead use greedy clustering on (round(f1,4), n_tokens)
        # equality (exact-duplicate answers). Documented as proxy.
        keys = [(round(c['f1'], 4), c['n_tokens']) for c in cs]
        clusters = {}
        for i, k in enumerate(keys):
            clusters.setdefault(k, []).append(i)
        biggest = max(clusters.values(), key=len)
        oracle.append(f.max())
        sc.append(f[biggest].max())
        bon.append(f[np.argmax(g)])
        top = set(np.argsort(f)[-max(1, len(f) // 2):])
        if not (set(biggest) & top):
            minority += 1
            if np.argmax(g) in top:
                rescued += 1
    out['C3_ttsp'] = {
        'note': 'answer-cluster proxy = exact-duplicate (f1,n_tokens) keys; conservative',
        'oracle': round(float(np.mean(oracle)), 2),
        'answer_level_SC': round(float(np.mean(sc)), 2),
        'evidence_level_BoN': round(float(np.mean(bon)), 2),
        'minority_correct_rate': round(minority / len(oracle), 3),
        'minority_rescued_by_g': round(rescued / max(minority, 1), 3),
        'n': len(oracle)}
    # haloquest length-error (41.5% length loss context)
    h, hb = load_claim('haloquest')
    hl, hc = [], []
    for s in h['samples']:
        b = hb.get(s['id'])
        if b and b.get('logp_list'):
            hl.append(len(b['logp_list']))
            hc.append(1 if s['f1'] > 0 else 0)
    out['C2_haloquest'] = {'spearman_len_correct': round(float(spearmanr(hl, hc).correlation), 3),
                           'n': len(hl)}
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default='ABC')
    a = ap.parse_args()
    res = {'pre_registration': __doc__.split('"""')[0].strip()}
    if 'A' in a.only:
        res['A'] = run_A()
    if 'B' in a.only:
        res['B'] = run_B()
    if 'C' in a.only:
        res['C'] = run_C()
    json.dump(res, open('outputs/lit_mech_transfer.json', 'w'), indent=1)
    print(json.dumps(res, indent=1)[:6000])
