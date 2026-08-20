# -*- coding: utf-8 -*-
"""
zero_gpu_threshold_analysis.py -- zero-GPU threshold trajectory analysis (2026-08-17).

Three pre-registered analyses over already-on-disk probes / vote pools. No GPU.

  A1  cmm x Qwen3 residual calibration disease: view-free margin threshold.
      Operator: pred=yes AND margin_full < tau  -> flip yes->no. tau scan 0..15.
      Gate: probe net gain > +2.0 F1 (yes-F1) -> worth full 721 (which needs
      GPU teacher-forced lp_no; margins are NOT recoverable from logp_list,
      which stores only the chosen token's logprob).
  A2  phd x Nemotron FN-side one-sided CF operator.
      Operator: only rows with greedy=no AND Delta pointing to yes
      (m + lam*d_noctx > 0) flip to yes; FP side untouched.
      Gate: probe net gain >= +1.5 F1 -> "one-sided CF worth full validation".
  A3  avhbench Vote-SC family-balanced voting rules over the q3x pools
      (families: Qwen2.5 = {224,384,448,f16}, Qwen3 = {q3,q3r384,q3f16}).
      Rules: CF>=2 (>=2 families each >=1 vote), CF>=2&votes>=3 (families>=2
      AND total votes>=3), HYB (Qwen2.5>=2 AND Qwen3>=1); exploratory:
      HYB-sym, FAMINT (max within-family votes>=2). Pools 5/6/6R/7 arm.
      Gate: any rule > 22.06 (pre-registered adoption line = 21.56+0.5).
      Regression check: plain >=2 on 5arm must reproduce 21.56 exactly.

Usage: python3 zero_gpu_threshold_analysis.py   (writes .json results next to
       the outputs dirs and prints a full report)
"""
import json, re, sys, glob, collections, statistics
import numpy as np
from scipy import stats as st

OUT_JSON = '/home/tsq/rhq/omniconfess/OmniConfess/outputs/zero_gpu_threshold_analysis.json'

# ---------------------------------------------------------------------------
# shared metrics (no sklearn dependency: F1/Youden/AUC are 3 lines each)
# ---------------------------------------------------------------------------

def f1_yes(gt, pr):
    gt = np.asarray(gt, int); pr = np.asarray(pr, int)
    tp = int(((gt == 1) & (pr == 1)).sum()); fp = int(((gt == 0) & (pr == 1)).sum())
    fn = int(((gt == 1) & (pr == 0)).sum())
    if tp == 0:
        return 0.0
    p = tp / (tp + fp); r = tp / (tp + fn)
    return 2 * p * r / (p + r)


def youden(gt, pr):
    gt = np.asarray(gt, int); pr = np.asarray(pr, int)
    tpr = ((gt == 1) & (pr == 1)).sum() / max((gt == 1).sum(), 1)
    fpr = ((gt == 0) & (pr == 1)).sum() / max((gt == 0).sum(), 1)
    return tpr - fpr


def auc(label, score):
    """ROC AUC of score -> label (1 = positive), Mann-Whitney form."""
    label = np.asarray(label, int); score = np.asarray(score, float)
    pos, neg = score[label == 1], score[label == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    u, _ = st.mannwhitneyu(pos, neg, alternative='greater')
    return u / (len(pos) * len(neg))


def boot_ci(fn, n, B=2000, seed=42):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        s = rng.integers(0, n, n)
        vals.append(fn(s))
    v = np.array(vals)
    return float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))


# ---------------------------------------------------------------------------
# A1 -- cmm x Qwen3 margin threshold
# ---------------------------------------------------------------------------

def a1():
    rows = [json.loads(l) for l in open(
        '/home/tsq/rhq/omniconfess/OmniConfess/outputs/cf_probe/cmm_probe_qwen3.jsonl')]
    n = len(rows)
    gt = np.array([r['gt_yes'] for r in rows])
    base = np.array([1 if r['pred'] == 'yes' else 0 for r in rows])
    m = np.array([r['margin_full'] for r in rows])
    base_f1 = f1_yes(gt, base) * 100
    base_yd = youden(gt, base)

    # cell margin stats
    cells = {}
    for gy in (1, 0):
        for pb in (1, 0):
            sel = (gt == gy) & (base == pb)
            name = ('gt=%s base=%s' % ('yes' if gy else 'no', 'yes' if pb else 'no'))
            if sel.sum():
                cells[name] = dict(n=int(sel.sum()),
                                   mean_m=round(float(m[sel].mean()), 3),
                                   med_m=round(float(np.median(m[sel])), 3),
                                   q10=round(float(np.percentile(m[sel], 10)), 3),
                                   q90=round(float(np.percentile(m[sel], 90)), 3))
    # separation inside pred=yes: correct yes vs wrong yes (FP)
    sel_yes = base == 1
    fp = sel_yes & (gt == 0); tp = sel_yes & (gt == 1)
    auc_yes = auc(gt[sel_yes], m[sel_yes])

    # tau scan
    taus = np.round(np.arange(0.0, 15.01, 0.25), 2)
    grid = []
    for tau in taus:
        pr = base.copy()
        pr[(base == 1) & (m < tau)] = 0
        grid.append(dict(tau=float(tau), f1=round(f1_yes(gt, pr) * 100, 2),
                         dF1=round(f1_yes(gt, pr) * 100 - base_f1, 2),
                         youden=round(youden(gt, pr), 3),
                         dYd=round(youden(gt, pr) - base_yd, 3),
                         yes=round(float(pr.mean()) * 100, 1),
                         changed=int((pr != base).sum())))
    best_f1 = max(grid, key=lambda r: r['f1'])
    best_yd = max(grid, key=lambda r: r['youden'])
    plateau = [r for r in grid if r['dF1'] >= 2.0]

    # oracle ceiling on the yes side: flip exactly the FPs
    pr_or = base.copy(); pr_or[fp] = 0
    oracle_f1 = f1_yes(gt, pr_or) * 100
    # unfixable at any tau: FPs whose margin >= max FP margin ... all FPs with
    # m >= tau of the best cut; report max-FP margin (any tau above it fixes none)
    max_fp_m = float(m[fp].max()) if fp.any() else float('nan')
    n_fp_above_12 = int((m[fp] > 12).sum())

    # bootstrap of the BEST-tau gain (selection inside each replicate) and of
    # the fixed best tau
    tau_star = best_f1['tau']

    def gain_star(s):
        g, b, mm = gt[s], base[s], m[s]
        bf = f1_yes(g, b) * 100
        best = max(f1_yes(g, np.where((b == 1) & (mm < t), 0, b)) * 100 - bf
                   for t in taus)
        return best

    def gain_fixed(s):
        g, b, mm = gt[s], base[s], m[s]
        pr = np.where((b == 1) & (mm < tau_star), 0, b)
        return f1_yes(g, pr) * 100 - f1_yes(g, b) * 100

    ci_star = boot_ci(gain_star, n)
    ci_fixed = boot_ci(gain_fixed, n)

    # leave-one-out honest tau
    loo_pr = base.copy()
    for i in range(n):
        idx = np.array([j for j in range(n) if j != i])
        g, b, mm = gt[idx], base[idx], m[idx]
        bf = f1_yes(g, b)
        best_t, best_v = 0.0, -1
        for t in taus:
            v = f1_yes(g, np.where((b == 1) & (mm < t), 0, b))
            if v > best_v:
                best_v, best_t = v, t
        loo_pr[i] = 0 if (base[i] == 1 and m[i] < best_t) else base[i]
    loo_dF1 = f1_yes(gt, loo_pr) * 100 - base_f1

    # reweight probe cells to full-set cell shares (721 scored rows:
    # yy 274 / yn 27 / ny 160 / nn 260)
    full_cell = {(1, 1): 274, (1, 0): 27, (0, 1): 160, (0, 0): 260}
    N = 721
    w = np.ones(n)
    for i in range(n):
        w[i] = (full_cell[(gt[i], base[i])] / N) / (((gt == gt[i]) & (base == base[i])).sum() / n)

    def wf1(pr):
        tp = w[(gt == 1) & (pr == 1)].sum(); fp = w[(gt == 0) & (pr == 1)].sum()
        fn = w[(gt == 1) & (pr == 0)].sum()
        if tp == 0:
            return 0.0
        p, r = tp / (tp + fp), tp / (tp + fn)
        return 200 * p * r / (p + r)

    wbase = wf1(base)
    wgrid = []
    for tau in taus:
        pr = base.copy(); pr[(base == 1) & (m < tau)] = 0
        wgrid.append(dict(tau=float(tau), f1_full_proj=round(wf1(pr), 2),
                          dF1_full=round(wf1(pr) - wbase, 2)))
    wbest = max(wgrid, key=lambda r: r['f1_full_proj'])
    wbest_f1_tau = [r for r in wgrid if r['tau'] == tau_star][0]

    # flip composition at key taus: FP fixed vs TP broken (+ yes-rate)
    def comp(tau):
        prr = base.copy(); prr[(base == 1) & (m < tau)] = 0
        ch = prr != base
        return dict(tau=float(tau),
                    fp_fixed=int((ch & (gt == 0) & (base == 1)).sum()),
                    tp_broken=int((ch & (gt == 1) & (base == 1)).sum()),
                    yes=round(float(prr.mean()) * 100, 1))

    flip_comp = [comp(t) for t in (2.0, 4.0, 6.0, tau_star, 9.75, 11.25)]

    # binomial sign test on the discordant pairs at tau*
    prr = base.copy(); prr[(base == 1) & (m < tau_star)] = 0
    ch = prr != base
    n_fix = int((ch & (gt == 0)).sum()); n_break = int((ch & (gt == 1)).sum())
    sign_p = float(2 * st.binom.cdf(min(n_fix, n_break), n_fix + n_break, 0.5)) \
        if (n_fix + n_break) else 1.0

    # published full base F1 for reference: 74.56
    res = dict(
        n=n, base_f1_probe=round(base_f1, 2), base_yd_probe=round(base_yd, 3),
        base_yes_probe=round(float(base.mean()) * 100, 1),
        cells=cells,
        sep_pred_yes=dict(auc_margin_to_gt=round(auc_yes, 3),
                          mean_m_correct_yes=round(float(m[tp].mean()), 3),
                          mean_m_wrong_yes=round(float(m[fp].mean()), 3),
                          med_m_correct_yes=round(float(np.median(m[tp])), 3),
                          med_m_wrong_yes=round(float(np.median(m[fp])), 3),
                          max_m_wrong_yes=round(max_fp_m, 3),
                          n_wrong_yes_above12=n_fp_above_12),
        grid=grid, best_f1=best_f1, best_yd=best_yd,
        plateau_dF1_ge2=[r['tau'] for r in plateau],
        tau_star=tau_star,
        ci_best_gain=ci_star, ci_fixed_tau_gain=ci_fixed,
        loo_dF1=round(loo_dF1, 2),
        oracle_yes_side_f1=round(oracle_f1, 2),
        reweighted=dict(base_full_proj=round(wbase, 2),
                        best_full_proj=wbest,
                        at_tau_star=wbest_f1_tau),
        flip_composition=flip_comp,
        sign_test_tau_star=dict(fp_fixed=n_fix, tp_broken=n_break,
                                p_two_sided=round(sign_p, 4)),
    )
    return res


# ---------------------------------------------------------------------------
# A2 -- phd x Nemotron one-sided (FN-only) CF operator
# ---------------------------------------------------------------------------

def a2():
    rows = [json.loads(l) for l in open(
        '/home/tsq/rhq/omniconfess/OmniConfess/outputs/cf_probe/phd_probe_nemotron.jsonl')]
    n = len(rows)
    gt = np.array([r['gt_yes'] for r in rows])
    base = np.array([1 if r['pred'] == 'yes' else 0 for r in rows])
    m = np.array([r['margin_full'] for r in rows])
    d = np.array([r['d_noctx'] for r in rows])   # operator conv: masked - full (yes side)
    base_f1 = f1_yes(gt, base) * 100
    base_yd = youden(gt, base)

    fn = (base == 0) & (gt == 1)
    tn = (base == 0) & (gt == 0)

    lams = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0]
    grid = []
    for lam in lams:
        pr = base.copy()
        adj = m + lam * d
        flip = (base == 0) & (adj > 0)          # one-sided: greedy=no, Delta->yes
        pr[flip] = 1
        grid.append(dict(
            lam=lam, f1=round(f1_yes(gt, pr) * 100, 2),
            dF1=round(f1_yes(gt, pr) * 100 - base_f1, 2),
            youden=round(youden(gt, pr), 3), dYd=round(youden(gt, pr) - base_yd, 3),
            yes=round(float(pr.mean()) * 100, 1), changed=int(flip.sum()),
            good_flips=int((flip & fn).sum()), bad_flips=int((flip & tn).sum())))
    best = max(grid, key=lambda r: r['f1'])
    best_yd = max(grid, key=lambda r: r['youden'])

    # two-sided reference (the pre-existing grid, recomputed for comparison)
    two = []
    for lam in lams:
        pr = (m + lam * d > 0).astype(int)
        two.append(dict(lam=lam, f1=round(f1_yes(gt, pr) * 100, 2),
                        changed=int((pr != base).sum()),
                        fp_flips=int(((pr == 1) & (base == 0) & (gt == 0)).sum()),
                        fn_fixes=int(((pr == 1) & (base == 0) & (gt == 1)).sum()),
                        yes_broken=int(((pr == 0) & (base == 1) & (gt == 1)).sum()),
                        tp_kept=int(((pr == 0) & (base == 1) & (gt == 0)).sum())))

    # bootstrap CI of the best-lam one-sided gain (selection inside replicate)
    def gain_star(s):
        g, b, mm, dd = gt[s], base[s], m[s], d[s]
        bf = f1_yes(g, b) * 100
        best_v = max(f1_yes(g, np.where((b == 0) & ((mm + L * dd) > 0), 1, b)) * 100 - bf
                     for L in lams)
        return best_v

    ci_star = boot_ci(gain_star, n)

    # reweight to full 906 (yy 498 / yn 110 / ny 92 / nn 206)
    full_cell = {(1, 1): 498, (1, 0): 110, (0, 1): 92, (0, 0): 206}
    N = 906
    w = np.ones(n)
    for i in range(n):
        w[i] = (full_cell[(gt[i], base[i])] / N) / (((gt == gt[i]) & (base == base[i])).sum() / n)

    def wf1(pr):
        tp = w[(gt == 1) & (pr == 1)].sum(); fp = w[(gt == 0) & (pr == 1)].sum()
        fnw = w[(gt == 1) & (pr == 0)].sum()
        if tp == 0:
            return 0.0
        p, r = tp / (tp + fp), tp / (tp + fnw)
        return 200 * p * r / (p + r)

    wbase = wf1(base)
    wbest_lam, wbest_v = None, -1
    for lam in lams:
        pr = base.copy(); pr[(base == 0) & ((m + lam * d) > 0)] = 1
        v = wf1(pr)
        if v > wbest_v:
            wbest_v, wbest_lam = v, lam

    def wgain(s, L):
        g, b, mm, dd = gt[s], base[s], m[s], d[s]
        ww = w[s]
        def W(pr):
            tp = ww[(g == 1) & (pr == 1)].sum(); fp = ww[(g == 0) & (pr == 1)].sum()
            fnw = ww[(g == 1) & (pr == 0)].sum()
            if tp == 0:
                return 0.0
            p, r = tp / (tp + fp), tp / (tp + fnw)
            return 200 * p * r / (p + r)
        pr = b.copy(); pr[(b == 0) & ((mm + L * dd) > 0)] = 1
        return W(pr) - W(b)

    ci_wgain_lam3 = boot_ci(lambda s: wgain(s, 3.0), n)
    ci_wgain_best = boot_ci(lambda s: wgain(s, wbest_lam), n)

    # sign test at lam=3 (moderate operating point)
    lam3 = [g for g in grid if g['lam'] == 3.0][0]
    g3, b3 = lam3['good_flips'], lam3['bad_flips']
    sign_p3 = float(2 * st.binom.cdf(min(g3, b3), g3 + b3, 0.5)) if (g3 + b3) else 1.0

    # pull thickness diagnostics on FN side
    fn_d_pos = int((d[fn] > 0).sum())
    fn_cross = {L: int(((m + L * d > 0) & fn).sum()) for L in lams}
    res = dict(
        n=n, base_f1_probe=round(base_f1, 2), base_yd=round(base_yd, 3),
        grid=grid, best=best, best_yd=best_yd, two_sided_ref=two,
        ci_best_gain=ci_star,
        fn_diag=dict(n_fn=int(fn.sum()), fn_with_d_pos=fn_d_pos,
                     mean_d_fn=round(float(d[fn].mean()), 3),
                     med_d_fn=round(float(np.median(d[fn])), 3),
                     mean_m_fn=round(float(m[fn].mean()), 3),
                     med_absm_fn=round(float(np.median(np.abs(m[fn]))), 3),
                     fn_crossing_by_lam=fn_cross),
        tn_diag=dict(n_tn=int(tn.sum()), mean_d_tn=round(float(d[tn].mean()), 3),
                     med_d_tn=round(float(np.median(d[tn])), 3)),
        reweighted=dict(base_full_proj=round(wbase, 2),
                        best_lam=wbest_lam, best_full_proj=round(wbest_v, 2),
                        best_dF1_full=round(wbest_v - wbase, 2),
                        ci_lam3=ci_wgain_lam3, ci_best=ci_wgain_best),
        sign_test_lam3=dict(good=g3, bad=b3, p_two_sided=round(sign_p3, 4)),
    )
    return res


# ---------------------------------------------------------------------------
# A3 -- avhbench Vote-SC family-balanced rules
# ---------------------------------------------------------------------------

# token_f1 + voting tokenization inlined VERBATIM from answerability_gate.py /
# avhbench_vote_sc_rebuild.py (their env deps are unavailable here)
STOPWORDS = set([
    "a", "an", "the", "and", "or", "but", "if", "because", "as", "what",
    "which", "this", "that", "these", "those", "then", "just", "so", "than",
    "such", "both", "through", "about", "for", "is", "are", "was", "were",
    "of", "while", "during", "to", "in", "on", "at", "by", "with", "it", "there",
])

def token_f1(pred, gt):
    def tok(s):
        s = str(s).lower().translate(str.maketrans('', '', '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'))
        return [w for w in s.split() if w not in STOPWORDS and w.strip()]
    p_t, g_t = tok(pred), tok(gt)
    if not p_t or not g_t:
        return 0.0
    common = collections.Counter(p_t) & collections.Counter(g_t)
    hit = sum(common.values())
    p, r = hit / len(p_t), hit / len(g_t)
    return (2 * p * r) / (p + r) if (p + r) > 0 else 0.0

F = lambda a, b: token_f1(a, b) * 100

STOP = set('''a an the and or of to in on at as is are was were be been being it its this that these those with while by for from into near nearby
there here his her their our your my he she they we you i who which what when where how not no nor but so then than also very just only
one two single describe see hear seen heard video audio image picture clip scene shows show can cannot could would should may might'''.split())
PUNCT = "!\"#$%&'()*+,-./:;<=>?@[]^_`{|}~\\"

def normgt(s):
    return re.sub(r'\s+', ' ', str(s).strip().lower())

def clean(s):
    m = re.search(r'\\boxed\{(.*)\}', str(s), re.S)
    return (m.group(1) if m else str(s)).strip()

def cwords(s):
    s = clean(s).lower().translate(str.maketrans('', '', PUNCT))
    return [w for w in s.split() if w not in STOP and len(w) > 1]

FAMILY = {'224': 'Q2.5', '384': 'Q2.5', '448': 'Q2.5', 'f16': 'Q2.5',
          'q3': 'Q3', 'q3r384': 'Q3', 'q3f16': 'Q3'}

OUT = '/home/tsq/rhq/omniconfess/OmniConfess/outputs/avhbench_vote_sc'
PATHS = {
    '224':   '/home/tsq/rhq/omniconfess/OmniConfess/outputs/avhbench_base_model/avhbench_base_model_full_single_scored.jsonl',
    '384':   '/home/tsq/rhq/omniconfess/OmniConfess/outputs/avhbench_res384/avhbench_res384_full_single_scored.jsonl',
    '448':   '/home/tsq/rhq/omniconfess/OmniConfess/outputs/avhbench_res448/avhbench_res448_full_single_scored.jsonl',
    'f16':   '/home/tsq/rhq/omniconfess/OmniConfess/outputs/avhbench_frames16/avhbench_frames16_full_single_scored.jsonl',
    'q3':    '/ultrafast_share/tsq/qwen3omni_outputs/avhbench_qwen3omni/avhbench_qwen3omni_full_single_scored.jsonl',
    'q3r384': OUT + '/q3_res384/avhbench_q3_res384_full_single_scored.jsonl',
    'q3f16':  OUT + '/q3_f16/avhbench_q3_f16_full_single_scored.jsonl',
}

def load_arms():
    arms = {}
    for k, p in PATHS.items():
        recs = [json.loads(l) for l in open(p, encoding='utf-8')]
        d = {}
        for r in recs:
            d.setdefault(normgt(r['gt']), r)
        arms[k] = d
    keys = sorted(set.intersection(*[set(a) for a in arms.values()]))
    return arms, keys


def votes_for(arms, keys, armlist):
    """Per key: word -> {count, families:set, fam_count:Counter, arms:list}."""
    out = []
    for g in keys:
        cnt = collections.Counter()
        contrib = collections.defaultdict(list)
        for a in armlist:
            for w in set(cwords(clean(arms[a][g]['pred']))):
                cnt[w] += 1
                contrib[w].append(a)
        fc = {w: collections.Counter(FAMILY[a] for a in contrib[w]) for w in cnt}
        out.append((g, cnt, contrib, fc))
    return out


def keep(rule, w, cnt, fc, thr):
    """Return True if word w passes the rule at base threshold thr."""
    fams = {f for f, c in fc[w].items() if c > 0}
    v = cnt[w]
    q25, q3 = fc[w].get('Q2.5', 0), fc[w].get('Q3', 0)
    if rule == 'plain':            # shipped >=thr gate (no family condition)
        return v >= thr
    if rule == 'cfam':             # >=2 families each >=1 vote (thr implied by thr)
        return len(fams) >= 2 and v >= thr
    if rule == 'cfam3':            # >=2 families AND total votes >=3
        return len(fams) >= 2 and v >= 3
    if rule == 'hyb':              # Qwen2.5-internal >=2 AND cross-generation >=1
        return q25 >= 2 and q3 >= 1
    if rule == 'hybsym':           # symmetric hybrid (either direction)
        return (q25 >= 2 and q3 >= 1) or (q3 >= 2 and q25 >= 1)
    if rule == 'famint':           # some single family corroborates internally
        return max(fc[w].values()) >= 2 and v >= thr
    raise ValueError(rule)


RULES = [('plain', 2), ('cfam', 2), ('cfam3', 2), ('hyb', 2), ('hybsym', 2), ('famint', 2),
         ('plain', 3), ('cfam', 3), ('hyb', 3)]


def a3():
    arms, keys = load_arms()
    A5 = ['224', '384', '448', 'f16', 'q3']
    A7 = A5 + ['q3r384', 'q3f16']
    A6 = A5 + ['q3r384']
    A6R = ['224', '384', '448', 'q3', 'q3r384']
    pools = {'5arm': A5, '6arm': A6, '6armR': A6R, '7arm': A7}

    vt = {name: votes_for(arms, keys, al) for name, al in pools.items()}

    # regression: plain@2 on 5arm must equal 21.56
    def score(pool, rule, thr):
        f1s, base_f1s, keptw, hitw = [], [], 0, 0
        for g, cnt, contrib, fc in vt[pool]:
            voted = [w for w in cnt if keep(rule, w, cnt, fc, thr)]
            order = []
            for a in pools[pool]:
                for w in cwords(clean(arms[a][g]['pred'])):
                    if w in voted and w not in order:
                        order.append(w)
            pred = ' '.join(order)
            f1s.append(F(pred, arms['224'][g]['gt']))
            base_f1s.append(F(arms['224'][g]['pred'], arms['224'][g]['gt']))
            gtw = set(cwords(clean(arms['224'][g]['gt'])))
            keptw += len(voted)
            hitw += sum(1 for w in voted if w in gtw)
        w_ = st.wilcoxon(f1s, base_f1s)
        return dict(f1=round(statistics.mean(f1s), 2), n=len(f1s),
                    p=float(w_.pvalue),
                    win=int(sum(1 for a, b in zip(f1s, base_f1s) if a > b)),
                    lose=int(sum(1 for a, b in zip(f1s, base_f1s) if a < b)),
                    kept_words=keptw, kept_gt_hit=round(100 * hitw / max(keptw, 1), 1))

    results = {}
    for pool in pools:
        for rule, thr in RULES:
            if pool == '5arm' and rule in ('cfam3',):
                pass  # still computable (q3 counts as second family)
            results[f'{pool}|{rule}@{thr}'] = score(pool, rule, thr)

    reg = results['5arm|plain@2']['f1']

    # word-flow audit on the 7arm pool: reproduce the 89.8% family-private claim
    # and measure each rule's filter rate on the family-private rescued cohort
    audit = dict(rescued=0, rescued_fam_private=0, rescued_crossfam=0,
                 stable_core=0, private_gt_hit=[0, 0], cross_gt_hit=[0, 0],
                 core_gt_hit=[0, 0])
    filt = {f'{r}@{t}': [0, 0] for r, t in RULES}      # [filtered, total] on private
    core_kept = {f'{r}@{t}': [0, 0] for r, t in RULES}  # stable-core survival
    for g, cnt7, contrib7, fc7 in vt['7arm']:
        gtw = set(cwords(clean(arms['224'][g]['gt'])))
        cnt5 = collections.Counter()
        for a in A5:
            for w in set(cwords(clean(arms[a][g]['pred']))):
                cnt5[w] += 1
        for w, c7 in cnt7.items():
            b5 = cnt5[w] >= 2
            b7 = c7 >= 2
            fams = {f for f, c in fc7[w].items() if c > 0}
            if not b5 and b7:
                audit['rescued'] += 1
                if len(fams) == 1 and 'Q3' in fams:
                    audit['rescued_fam_private'] += 1
                    audit['private_gt_hit'][1] += 1
                    if w in gtw:
                        audit['private_gt_hit'][0] += 1
                    for rule, thr in RULES:
                        filt[f'{rule}@{thr}'][1] += 1
                        if not keep(rule, w, cnt7, fc7, thr):
                            filt[f'{rule}@{thr}'][0] += 1
                else:
                    audit['rescued_crossfam'] += 1
                    audit['cross_gt_hit'][1] += 1
                    if w in gtw:
                        audit['cross_gt_hit'][0] += 1
            elif b5 and b7:
                audit['stable_core'] += 1
                audit['core_gt_hit'][1] += 1
                if w in gtw:
                    audit['core_gt_hit'][0] += 1
                for rule, thr in RULES:
                    core_kept[f'{rule}@{thr}'][1] += 1
                    if keep(rule, w, cnt7, fc7, thr):
                        core_kept[f'{rule}@{thr}'][0] += 1

    audit['private_share_pct'] = round(100 * audit['rescued_fam_private'] /
                                       max(audit['rescued'], 1), 1)
    audit['private_gt_hit_pct'] = round(100 * audit['private_gt_hit'][0] /
                                        max(audit['private_gt_hit'][1], 1), 1)
    audit['cross_gt_hit_pct'] = round(100 * audit['cross_gt_hit'][0] /
                                      max(audit['cross_gt_hit'][1], 1), 1)
    audit['core_gt_hit_pct'] = round(100 * audit['core_gt_hit'][0] /
                                     max(audit['core_gt_hit'][1], 1), 1)
    filter_rates = {k: round(100 * v[0] / max(v[1], 1), 1) for k, v in filt.items()}
    core_survival = {k: round(100 * v[0] / max(v[1], 1), 1) for k, v in core_kept.items()}

    gate = [k for k, v in results.items() if v['f1'] > 22.06 and not k.startswith('5arm|plain')]
    return dict(n_keys=len(keys), regression_5arm_plain2=reg,
                combos=results, word_flow=audit,
                private_filter_rate_pct=filter_rates,
                stable_core_survival_pct=core_survival,
                alive_over_gate=gate)


def main():
    res = {}
    print('=== A1: cmm x Qwen3 margin threshold (view-free calibration) ===')
    res['a1'] = a1()
    r = res['a1']
    print(f"n={r['n']}  base probe F1={r['base_f1_probe']}  yes={r['base_yes_probe']}%  Youden={r['base_yd_probe']}")
    for k, v in r['cells'].items():
        print(f"  {k}: n={v['n']} mean_m={v['mean_m']:+.2f} med={v['med_m']:+.2f} [q10 {v['q10']}, q90 {v['q90']}]")
    s = r['sep_pred_yes']
    print(f"  within pred=yes: AUC(margin->gt)={s['auc_margin_to_gt']}  "
          f"correct-yes mean {s['mean_m_correct_yes']:+.2f} vs wrong-yes {s['mean_m_wrong_yes']:+.2f} "
          f"(med {s['med_m_correct_yes']:+.2f} / {s['med_m_wrong_yes']:+.2f}), "
          f"max wrong-yes margin {s['max_m_wrong_yes']:+.2f}, wrong-yes>12nats: {s['n_wrong_yes_above12']}")
    print(f"  best tau={r['best_f1']['tau']}  F1 {r['best_f1']['f1']} (dF1 {r['best_f1']['dF1']:+.2f}, "
          f"changed {r['best_f1']['changed']})  plateau dF1>=2: tau in {r['plateau_dF1_ge2']}")
    print(f"  best-Youden tau={r['best_yd']['tau']} Youden {r['best_yd']['youden']} (dYd {r['best_yd']['dYd']:+.3f})")
    print(f"  bootstrap CI best-tau gain: {r['ci_best_gain']}  fixed tau*={r['tau_star']}: {r['ci_fixed_tau_gain']}")
    print(f"  LOO honest dF1: {r['loo_dF1']:+.2f}   oracle yes-side F1: {r['oracle_yes_side_f1']}")
    print(f"  full-set projection (cell-reweighted): base {r['reweighted']['base_full_proj']} -> "
          f"best {r['reweighted']['best_full_proj']}  at tau*: {r['reweighted']['at_tau_star']}")
    print(f"  flip composition (tau: fp_fixed/tp_broken/yes%): {r['flip_composition']}")
    print(f"  sign test at tau*: {r['sign_test_tau_star']}")

    print('\n--- A1b: same operator on cmm x Nemotron probe (transfer check) ---')
    res['a1b'] = a1_transfer()
    r = res['a1b']
    print(f"  n={r['n']} base probe F1={r['base_f1_probe']}  AUC(margin|pred=yes)={r['auc_margin_within_predyes']}  "
          f"m(correct-yes)={r['mean_m_correct_yes']:+.2f} vs m(wrong-yes)={r['mean_m_wrong_yes']:+.2f}")
    print(f"  best tau={r['best_tau']} dF1={r['best_dF1_probe']:+.2f} "
          f"(fp_fixed={r['best_fp_fixed']}, tp_broken={r['best_tp_broken']})  "
          f"full proj: {r['reweighted']['base_full_proj']} -> {r['reweighted']['best_full_proj']} "
          f"({r['reweighted']['dF1_full']:+.2f}, CI {tuple(round(x,2) for x in r['reweighted']['ci_best_gain'])})")
    r = res['a1']

    print('\n=== A2: phd x Nemotron FN-side one-sided CF ===')
    res['a2'] = a2()
    r = res['a2']
    print(f"n={r['n']}  base probe F1={r['base_f1_probe']}  Youden={r['base_yd']}")
    print('  lam   F1     dF1    Youden  yes%   chg  good/bad')
    for g in r['grid']:
        print(f"  {g['lam']:<5} {g['f1']:6.2f} {g['dF1']:+6.2f}  {g['youden']:+.3f} {g['yes']:5.1f} "
              f"{g['changed']:4d}  {g['good_flips']}/{g['bad_flips']}")
    print(f"  best: lam={r['best']['lam']} dF1={r['best']['dF1']:+.2f}  CI(best-lam gain)={r['ci_best_gain']}")
    f = r['fn_diag']
    print(f"  FN pull: n={f['n_fn']}  d>0 on {f['fn_with_d_pos']}/{f['n_fn']}  mean d={f['mean_d_fn']:+.3f} "
          f"med |m|={f['med_absm_fn']:.2f}  crossings by lam: {f['fn_crossing_by_lam']}")
    print(f"  reweighted full proj: base {r['reweighted']['base_full_proj']} -> "
          f"{r['reweighted']['best_full_proj']} (lam={r['reweighted']['best_lam']}, "
          f"dF1 {r['reweighted']['best_dF1_full']:+.2f})")
    print(f"  sign test lam=3 (good/bad {r['sign_test_lam3']['good']}/{r['sign_test_lam3']['bad']}): "
          f"p={r['sign_test_lam3']['p_two_sided']}  reweighted gain CI lam=3: {r['reweighted']['ci_lam3']}")

    print('\n=== A3: avhbench family-balanced voting ===')
    res['a3'] = a3()
    r = res['a3']
    print(f"aligned keys n={r['n_keys']}  regression 5arm plain@2 = {r['regression_5arm_plain2']} (must be 21.56)")
    print('  combo                    F1     p       win/lose  kept  hit%')
    for k, v in r['combos'].items():
        print(f"  {k:<24} {v['f1']:6.2f} {v['p']:.1e}  {v['win']:>3}/{v['lose']:<3}   {v['kept_words']:5d} {v['kept_gt_hit']:5.1f}")
    a = r['word_flow']
    print(f"  word flow: rescued={a['rescued']} private={a['rescued_fam_private']} "
          f"({a['private_share_pct']}%) hit {a['private_gt_hit_pct']}% | cross {a['rescued_crossfam']} "
          f"hit {a['cross_gt_hit_pct']}% | core {a['stable_core']} hit {a['core_gt_hit_pct']}%")
    print('  private-word filter rate:', r['private_filter_rate_pct'])
    print('  stable-core survival    :', r['stable_core_survival_pct'])
    print('  ALIVE over 22.06 gate   :', r['alive_over_gate'])

    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f'\n[written] {OUT_JSON}')


def a1_transfer():
    """Same tau operator on the Nemotron cmm probe (cross-backbone transfer
    check: Nemotron cmm margin_full AUC is also >0.9, but its behavioural
    over-yes is only +3.4pp -- separation without headroom?)."""
    rows = [json.loads(l) for l in open(
        '/home/tsq/rhq/omniconfess/OmniConfess/outputs/cf_probe/cmm_probe_nemotron.jsonl')]
    n = len(rows)
    gt = np.array([r['gt_yes'] for r in rows])
    base = np.array([1 if r['pred'] == 'yes' else 0 for r in rows])
    m = np.array([r['margin_full'] for r in rows])
    base_f1 = f1_yes(gt, base) * 100
    sel = base == 1
    auc_yes = auc(gt[sel], m[sel])
    grid = []
    for tau in np.round(np.arange(0, 15.01, 0.25), 2):
        pr = base.copy(); pr[(base == 1) & (m < tau)] = 0
        grid.append(dict(tau=float(tau), dF1=round(f1_yes(gt, pr) * 100 - base_f1, 2),
                         fp_fixed=int(((pr != base) & (gt == 0)).sum()),
                         tp_broken=int(((pr != base) & (gt == 1)).sum())))
    best = max(grid, key=lambda r: r['dF1'])
    # full reweight (721: yy 239 / yn 62 / ny 86 / nn 334)
    full_cell = {(1, 1): 239, (1, 0): 62, (0, 1): 86, (0, 0): 334}
    N = 721
    w = np.ones(n)
    for i in range(n):
        w[i] = (full_cell[(gt[i], base[i])] / N) / (((gt == gt[i]) & (base == base[i])).sum() / n)

    def wf1(pr):
        tp = w[(gt == 1) & (pr == 1)].sum(); fp = w[(gt == 0) & (pr == 1)].sum()
        fnw = w[(gt == 1) & (pr == 0)].sum()
        if tp == 0:
            return 0.0
        p, r = tp / (tp + fp), tp / (tp + fnw)
        return 200 * p * r / (p + r)

    wbase = wf1(base)
    wbest_tau, wbest_v = 0.0, -1
    for tau in np.round(np.arange(0, 15.01, 0.25), 2):
        pr = base.copy(); pr[(base == 1) & (m < tau)] = 0
        v = wf1(pr)
        if v > wbest_v:
            wbest_v, wbest_tau = v, tau

    taus_all = np.round(np.arange(0, 15.01, 0.25), 2)

    def wgain_best(s):
        g, b, mm, ww = gt[s], base[s], m[s], w[s]
        def W(pr):
            tp = ww[(g == 1) & (pr == 1)].sum(); fp = ww[(g == 0) & (pr == 1)].sum()
            fnw = ww[(g == 1) & (pr == 0)].sum()
            if tp == 0:
                return 0.0
            p, r = tp / (tp + fp), tp / (tp + fnw)
            return 200 * p * r / (p + r)
        return max(W(np.where((b == 1) & (mm < t), 0, b)) for t in taus_all) - W(b)

    ci_wgain = boot_ci(wgain_best, n, B=2000)
    return dict(n=n, base_f1_probe=round(base_f1, 2),
                auc_margin_within_predyes=round(auc_yes, 3),
                mean_m_correct_yes=round(float(m[sel & (gt == 1)].mean()), 2),
                mean_m_wrong_yes=round(float(m[sel & (gt == 0)].mean()), 2),
                best_tau=best['tau'], best_dF1_probe=best['dF1'],
                best_fp_fixed=best['fp_fixed'], best_tp_broken=best['tp_broken'],
                reweighted=dict(base_full_proj=round(wbase, 2),
                                best_full_proj=round(wbest_v, 2),
                                dF1_full=round(wbest_v - wbase, 2),
                                best_tau=float(wbest_tau),
                                ci_best_gain=ci_wgain))


if __name__ == '__main__':
    main()
