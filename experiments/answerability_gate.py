# -*- coding: utf-8 -*-
"""answerability_gate.py — the ANSWERABILITY GATE operator (descriptive card 3).

Two stages, both offline over the teacher-forced probe forwards recorded by
answerability_probe.py (axiom A2 inherited: generation is never perturbed).

  Stage 1 — probe verdict:
      ANSWERABILITY SCORE (higher = more likely answerable) = candidate scorer
      selected by 5-fold CV ON DEV ONLY among:
        * logistic combos of the two-view grounding Deltas + priors
        * single feature d_attr (mean Delta over attribute tokens)
      plus the zero-cost hard rules from the haloquest autopsy:
        * answers already avoidance-phrased (EVADERE) are never eligible;
        * bare-count answers ("0" / "zero" / "none" / "no ...") are ALWAYS
          flagged — a counting hallucination on a false-premise question is
          image-grounded by construction (the count really is 0-looking under
          the true image), so Delta CANNOT see it; the rule can.
      Reported on the held-out test split: AUC(needs_gate vs affirmed
      answerable), precision/recall at flag rate ~= 24% (the unanswerable
      rate). GATE PASSES iff test precision > 0.70.

  Stage 2 — operator (run regardless; "建门" only changes the framing):
      eligible samples with score < tau (tau + echo template chosen on the
      dev grid) OR bare-count answers have their WHOLE answer replaced by the
      question-echo avoidance sentence. Affirmative content elsewhere is
      never touched. Echo style is mandatory — the generic template was
      -11~-14pp under an oracle flag, the echo style +10.5pp/item.
      Arms: main vs base vs random-flag control (same flag count, seed 49).

Judgement criteria (full set): Token-F1 > base; needs_gate subset F1 18 -> 30+;
answerable subset F1 drop < 1pp; random control arm always run.

Usage:
    python answerability_gate.py --probe outputs/answerability/haloquest_answerability_probe_p3.json
"""
import os, sys, json, argparse, random, re
from collections import Counter

import numpy as np
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from probe_claim_cf import EVADERE  # noqa: E402

STOPWORDS = set([
    "a", "an", "the", "and", "or", "but", "if", "because", "as", "what",
    "which", "this", "that", "these", "those", "then", "just", "so", "than",
    "such", "both", "through", "about", "for", "is", "are", "was", "were",
    "of", "while", "during", "to", "in", "on", "at", "by", "with", "it", "there",
])


# ---------------------------------------------------------------------------
# Token-F1 — verbatim convention of eval/single_eval.py (the reported metric)
# ---------------------------------------------------------------------------
def token_f1(pred, gt):
    def tok(s):
        s = str(s).lower().translate(str.maketrans('', '', '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'))
        return [w for w in s.split() if w not in STOPWORDS and w.strip()]
    p_t, g_t = tok(pred), tok(gt)
    if not p_t or not g_t:
        return 0.0
    common = Counter(p_t) & Counter(g_t)
    hit = sum(common.values())
    p, r = hit / len(p_t), hit / len(g_t)
    return (2 * p * r) / (p + r) if (p + r) > 0 else 0.0


# ---------------------------------------------------------------------------
# Echo avoidance templates. Autopsy mandate: the replacement must echo the
# question. E1 = the coordinator-specified canonical form.
# ---------------------------------------------------------------------------
_PLACEBO_TAIL = re.compile(r'\s+(?:in|on|at)\s+(?:this|the|a)\s+'
                           r'(?:image|photo|picture|painting|figure)\s*$', re.I)
_VERB_TAIL = re.compile(
    r'\s+(?:is|are|was|were|does|do|did|can|has|have|had|wearing|holding|'
    r'carrying|doing|looking|sitting|standing|lying|painted|catching|sleeping|'
    r'present|visible)\b.*$', re.I)


def extract_obj(q):
    """Best-effort object noun phrase from the question (rule-tunable).
    Returns None when no pattern fires — caller falls back to full echo."""
    q = q.strip().rstrip('?').strip()
    low = q.lower()
    obj = None
    if low.startswith('how many'):
        obj = q[len('how many'):].strip()
        obj = _VERB_TAIL.sub('', obj)
        obj = re.sub(r'^(?:are|is)\s+', '', obj, flags=re.I)
    else:
        m = re.match(r"what(?:'s| is| are)?\s+(?:the\s+)?"
                     r"(?:type|kind|breed|color|colour|brand|model|make|shade|"
                     r"name|number|part)\s+of\s+(.+)$", q, re.I)
        if m:
            obj = _VERB_TAIL.sub('', m.group(1))
        else:
            m = re.match(r"what\s+(?:color|colour)\s+(?:is|are)\s+(.+)$", q, re.I)
            if m:
                obj = _VERB_TAIL.sub('', m.group(1))
            else:
                m = re.match(r"what\s+is\s+(?:the\s+)?(.+)$", q, re.I)
                if m:
                    obj = _VERB_TAIL.sub('', m.group(1))
                else:
                    m = re.match(r"(?:is|are|do|does|did|can)\s+"
                                 r"(?:the|a|an|this|these|those)?\s*(.+)$", q, re.I)
                    if m:
                        obj = _VERB_TAIL.sub('', m.group(1))
                    else:
                        m = re.match(r"^(?:the|a|an)\s+(.+)$", q, re.I)
                        if m:
                            obj = _VERB_TAIL.sub('', m.group(1))
    if obj is None:
        return None
    obj = _PLACEBO_TAIL.sub('', obj).strip(' .,;:')
    obj = re.sub(r'^(?:the|a|an)\s+', '', obj, flags=re.I).strip(' .,;:')
    return obj or None


def _echo_T3(q):
    return q.strip().rstrip('?').strip() + ' is not visible in this image.'


def echo_E1(q):
    """'There is no [object] visible in this image.' (T3 echo fallback)."""
    obj = extract_obj(q)
    if obj is None:
        return _echo_T3(q)
    return f'There is no {obj} visible in this image.'


def echo_T4(q):
    obj = extract_obj(q)
    if obj is None:
        return _echo_T3(q)
    return f'The {obj} is not visible in the image.'


TEMPLATES = {'E1': echo_E1, 'T3': _echo_T3, 'T4': echo_T4}

# bare-count hallucination: Delta-blind by construction (see module docstring)
BARECOUNT = re.compile(r'^(?:0|zero|none|no)\b\.?', re.I)


# ---------------------------------------------------------------------------
# Answerability score: features + candidate scorers
# ---------------------------------------------------------------------------
FEATURES = ['d_content_ne', 'd_top2', 'd_attr_filled', 'lp_full_mean', 'howmany_digit']


def feature_vector(r):
    d_base = r['d_content'] if r['d_content'] is not None else 0.0
    d_ne = r['d_content_ne'] if r['d_content_ne'] is not None else d_base
    d_attr = r['d_attr'] if r['d_attr'] is not None else d_ne
    return [d_ne, r['d_top2'] if r['d_top2'] is not None else d_base,
            d_attr, r['lp_full_mean'] if r['lp_full_mean'] is not None else 0.0,
            1.0 if r['howmany_digit'] else 0.0]


def _lr_scorer(feats, C):
    def make(train_rows):
        from sklearn.linear_model import LogisticRegression
        X = np.asarray([[feature_vector(r)[FEATURES.index(f)] for f in feats]
                        for r in train_rows], dtype=float)
        y = np.array([1.0 if r['needs_gate'] else 0.0 for r in train_rows])
        mu, sd = X.mean(0), X.std(0) + 1e-8
        clf = LogisticRegression(max_iter=2000, C=C).fit((X - mu) / sd, y)
        def score(r):
            x = np.asarray([feature_vector(r)[FEATURES.index(f)] for f in feats])
            # label 1 = needs_gate -> negate so that higher = more ANSWERABLE
            return -float(clf.decision_function(((x - mu) / sd).reshape(1, -1))[0])
        return score
    return make


SCORERS = {
    'lr4': _lr_scorer(['d_content_ne', 'd_top2', 'd_attr_filled', 'howmany_digit'], 0.5),
    'lr3': _lr_scorer(['d_content_ne', 'd_top2', 'd_attr_filled'], 0.5),
    'd_attr': lambda train_rows: (lambda r: feature_vector(r)[2]),
    'z3': lambda train_rows: _z_mean(['d_content_ne', 'd_top2', 'd_attr_filled'], train_rows),
}


def _z_mean(feats, train_rows):
    X = np.asarray([[feature_vector(r)[FEATURES.index(f)] for f in feats]
                    for r in train_rows], dtype=float)
    mu, sd = X.mean(0), X.std(0) + 1e-8
    return lambda r: float(np.mean(
        (np.asarray([feature_vector(r)[FEATURES.index(f)] for f in feats]) - mu) / sd))


def auc(pos, neg):
    """AUC that pos scores above neg; pos = ANSWERABLE here."""
    from scipy.stats import mannwhitneyu
    if len(pos) < 2 or len(neg) < 2:
        return float('nan'), float('nan')
    u, p = mannwhitneyu(pos, neg, alternative='greater')
    return float(u / (len(pos) * len(neg))), float(p)


def stratified_split(rows, n_dev, seed=42):
    rng = random.Random(seed)
    strata = {}
    for r in rows:
        strata.setdefault((r['needs_gate'], r['neg_gt']), []).append(r)
    dev, test = [], []
    for k, g in sorted(strata.items()):
        rng.shuffle(g)
        nd = round(n_dev * len(g) / len(rows))
        dev.extend(g[:nd]); test.extend(g[nd:])
    return dev, test


# ---------------------------------------------------------------------------
# Flagging: hard rules + score threshold
# ---------------------------------------------------------------------------
def is_flagged(r, score, tau):
    if r['pred_evade']:
        return False                       # already avoidance-phrased
    if r['barecount']:
        return True                        # Delta-blind counting hallucination
    return score(r) < tau


def flag_stats(pool, score, tau):
    flags = [r for r in pool if is_flagged(r, score, tau)]
    tp = sum(1 for r in flags if r['needs_gate'])
    npos = sum(1 for r in pool if r['needs_gate'])
    return {'n_flag': len(flags),
            'flag_rate_pool': round(len(flags) / max(1, len(pool)), 4),
            'precision': round(tp / len(flags), 4) if flags else None,
            'recall': round(tp / npos, 4) if npos else None}


def apply_gate(rows, score, tau, template, flag_ids=None):
    out = []
    for r in rows:
        flagged = (r['sample_i'] in flag_ids) if flag_ids is not None \
            else is_flagged(r, score, tau)
        pred = template(r['question']) if flagged else r['pred']
        out.append({'id': r['id'], 'question': r['question'], 'gt': r['gt'],
                    'group_key': r.get('group_key'), 'dataset': 'haloquest',
                    'method': 'answerability_gate', 'base_pred': r['pred'],
                    'flagged': bool(flagged), 'pred': pred})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--probe', default=os.path.join(
        HERE, 'outputs', 'answerability', 'haloquest_answerability_probe_p3.json'))
    ap.add_argument('--n_dev', type=int, default=100)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--outdir', default=os.path.join(HERE, 'outputs', 'answerability'))
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
            # tau = flag-rate 24% quantile of the train scores (non-barecount)
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

    # tau at flag rate ~= 24% of the eligible pool, from DEV scores
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
    # tau + template chosen on dev ONLY (grid over flag-rate quantiles)
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
    # keep the full 365: items whose image was missing from the probe forwards
    # stay untouched (base pred, unflagged) so single_eval compares 365 vs 365
    probe_ids = {r['id'] for r in rows}
    recs_all = [json.loads(l) for l in open(os.path.join(
        HERE, 'outputs', 'haloquest_base_model',
        'haloquest_base_model_full_single_scored.jsonl'))]
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
        by_id = {e['id']: e for e in edited}
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
    out_main = os.path.join(args.outdir, 'haloquest_answerability_gate_full.jsonl')
    with open(out_main, 'w') as f:
        for e in edited_main:
            f.write(json.dumps(e, ensure_ascii=False) + '\n')
    out_rand = os.path.join(args.outdir, 'haloquest_answerability_gate_randctrl.jsonl')
    with open(out_rand, 'w') as f:
        for e in edited_rand:
            f.write(json.dumps({**e, 'method': 'answerability_gate_randctrl'},
                               ensure_ascii=False) + '\n')
    res_path = os.path.join(args.outdir, 'answerability_gate_result.json')
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
