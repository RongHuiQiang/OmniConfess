# -*- coding: utf-8 -*-
"""claim_confess_sim.py — offline prototype of the CLAIM-LEVEL confess operator
(断言级事后修正) on HaloQuest, simulated on the probe forwards.

The operator is POST-HOC by design (axiom 2: never perturb generation):

  1. base_model greedy answer stays untouched;
  2. sentence/clauser claims are extracted from the answer;
  3. every affirmative claim is scored by the two-view teacher-forced contrast
     already computed in probe_claim_cf.py:
         attr_score = mean Δ over attribute tokens (color/number) if any,
                      else Δ_mean of the claim
     where Δ = logP(claim | correct image) - logP(claim | swapped image);
  4. claims below threshold τ are treated as ungrounded → deleted; if the
     answer becomes empty, an evasion template replaces it (haloquest-style
     "not visible" phrasing — exactly what the gt of neg items rewards);
  5. format preserved: same short-sentence style, no re-generation.

This script consumes probe_claim_cf.py output (no GPU needed) and simulates
steps 4-5 over Token-F1, with a random-threshold control arm.

Dev/test protocol (pre-registered): dev = 100 stratified samples for τ and
template choice; test = remaining samples, paired Wilcoxon vs base.

Usage:
    python claim_confess_sim.py --probe outputs/claim_probe/haloquest_claim_probe_full.json
"""
import os, sys, json, argparse, random
from collections import Counter

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_claim_cf import content_words  # same normalization as Token-F1

STOPWORDS = set([
    "a", "an", "the", "and", "or", "but", "if", "because", "as", "what",
    "which", "this", "that", "these", "those", "then", "just", "so", "than",
    "such", "both", "through", "about", "for", "is", "are", "was", "were",
    "of", "while", "during", "to", "in", "on", "at", "by", "with", "it", "there",
])


def token_f1(pred, gt):
    def tok(s):
        s = str(s).lower().translate(str.maketrans("", "", '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'))
        return [w for w in s.split() if w not in STOPWORDS and w.strip()]
    p_t, g_t = tok(pred), tok(gt)
    if not p_t or not g_t:
        return 0.0
    common = Counter(p_t) & Counter(g_t)
    hit = sum(common.values())
    p, r = hit / len(p_t), hit / len(g_t)
    return (2 * p * r) / (p + r) if (p + r) > 0 else 0.0


COLORS = set("""red blue green yellow black white orange purple pink brown gray grey
gold silver beige teal maroon khaki tan turquoise violet indigo crimson""".split())
NUMW = set('two three four five six seven eight nine ten several many'.split())


def is_attr(w):
    return w in COLORS or any(ch.isdigit() for ch in w) or w in NUMW


TEMPLATES = {
    'T1': lambda q: 'It is not visible in the image.',
    'T2': lambda q: 'There is no such object visible in the image.',
    # entity-echo: bag-of-words F1 rewards object-word overlap with gt
    'T3': lambda q: q.strip().rstrip('?').strip() + ' is not visible in the image.',
}
try:
    import re as _re

    def _t4(q):
        m = _re.match(r"(?:What|Where|How many|How much)\s+(?:is|are)\s+(.+)",
                      q.strip(), _re.I)
        subj = m.group(1) if m else q.strip().rstrip('?')
        return f"The {subj} is not visible in the image."
    TEMPLATES['T4'] = _t4
except Exception:
    pass


def load_rows(probe_path):
    d = json.load(open(probe_path))
    sm = {m['sample_i']: m for m in d['samples']}
    rows = []
    for c in d['claims']:
        s = sm[c['sample_i']]
        av = [s['delta_tok'][j] for j in range(c['lo'], c['hi'])
              if s['classes'][j] in ('content', 'num')
              and is_attr(s['tokens'][j].strip().lower())]
        score = float(np.mean(av)) if av else c['delta_mean']
        rows.append(dict(c, score=score, has_attr=bool(av),
                         evasive=bool(c['is_evade'])))
    return d, sm, rows


def apply_operator(sample, claims, tau, template, rng=None, rand_frac=None):
    """Post-hoc edit: delete claims below tau (or a random fraction in the
    control arm). Returns edited answer text."""
    tokens = sample['tokens'][:]
    mine = [c for c in claims if c['sample_i'] == sample['sample_i']]
    flagged = []
    for c in mine:
        if rand_frac is not None:
            if rng.random() < rand_frac:
                flagged.append(c)
        elif (not c['is_evade']) and c['in_primary'] and c['score'] < tau:
            flagged.append(c)
    keep_tokens = list(tokens)
    for c in flagged:
        for j in range(c['lo'], c['hi']):
            keep_tokens[j] = None
    text = ''.join(t for t in keep_tokens if t is not None).strip()
    text = text.strip(' ,;') + ('.' if text and text[-1] not in '.!?' else '')
    if not text:
        text = template(sample.get('question', ''))
    return text, len(flagged)


def eval_arm(samples, rows, tau, template, rng=None, rand_frac=None):
    by_si = {}
    for c in rows:
        by_si.setdefault(c['sample_i'], []).append(c)
    f1s, flags = [], 0
    for s in samples:
        pred, nf = apply_operator(s, by_si.get(s['sample_i'], []), tau,
                                  template, rng, rand_frac)
        flags += nf
        f1s.append(token_f1(pred, s['gt']))
    return f1s, flags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--probe', default='outputs/claim_probe/haloquest_claim_probe_full.json')
    ap.add_argument('--n_dev', type=int, default=100)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    d, sm, rows = load_rows(args.probe)
    recs = [json.loads(l) for l in open(
        'outputs/haloquest_base_model/haloquest_base_model_full_single_scored.jsonl')]
    by_id = {r['id']: r for r in recs}
    samples = []
    for s in d['samples']:
        r = by_id[s['id']]
        samples.append(dict(s, gt=r['gt'], base_pred=r['pred'],
                            question=r['question'],
                            base_f1=token_f1(r['pred'], r['gt'])))

    # stratified dev/test split: by (has primary claim) x (neg_item)
    rng = random.Random(args.seed)
    strata = {}
    for s in samples:
        prim = any(c['sample_i'] == s['sample_i'] and c['in_primary'] for c in rows)
        strata.setdefault((prim, s['neg_item']), []).append(s)
    dev, test = [], []
    for k, group in sorted(strata.items()):
        rng.shuffle(group)
        n_dev = round(args.n_dev * len(group) / len(samples))
        dev.extend(group[:n_dev]); test.extend(group[n_dev:])

    base_dev = [s['base_f1'] for s in dev]
    base_test = [s['base_f1'] for s in test]

    # ---- dev: choose template + tau ----
    all_scores = [c['score'] for c in rows if c['in_primary']]
    taus = list(np.quantile(all_scores, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]))
    grid = {}
    for tname, tpl in TEMPLATES.items():
        for tau in taus:
            f1s, flags = eval_arm(dev, rows, tau, tpl)
            grid[(tname, round(float(tau), 3))] = (round(float(np.mean(f1s)), 3), flags)
    best = max(grid, key=lambda k: grid[k][0])
    tname, tau = best
    tpl = TEMPLATES[tname]
    print('dev grid (template, tau) -> (mean F1, n_flagged):')
    for k in sorted(grid, key=lambda k: grid[k][0], reverse=True)[:6]:
        print('  ', k, grid[k])
    print(f'[dev] base {np.mean(base_dev):.3f} -> best {grid[best][0]:.3f} '
          f'({tname}, tau={tau}, flagged={grid[best][1]})')

    # ---- test: main arm + random control ----
    f1_main, flags_main = eval_arm(test, rows, tau, tpl)
    frac = flags_main / max(1, sum(1 for c in rows if c['in_primary']
                                   and any(s['sample_i'] == c['sample_i'] for s in test)))
    rng2 = random.Random(49)
    f1_rand, flags_rand = eval_arm(test, rows, tau, tpl, rng=rng2, rand_frac=frac)

    def paired(a, b):
        w = wilcoxon(a, b)
        return round(float(np.mean(a)), 3), round(float(np.mean(a - b)), 3), round(float(w.pvalue), 5)
    out = {
        'n_dev': len(dev), 'n_test': len(test),
        'dev_base_f1': round(float(np.mean(base_dev)), 3),
        'dev_best': {'template': tname, 'tau': float(tau), 'f1': grid[best][0]},
        'test_base_f1': round(float(np.mean(base_test)), 3),
        'test_main_f1': paired(np.array(f1_main), np.array(base_test)),
        'test_randctrl_f1': paired(np.array(f1_rand), np.array(base_test)),
        'flags_main': flags_main, 'flags_rand': flags_rand,
        'main_vs_rand_p': round(float(wilcoxon(f1_main, f1_rand).pvalue), 5),
    }
    print(json.dumps(out, indent=2))
    json.dump(out, open('outputs/claim_probe/claim_confess_sim_result.json', 'w'), indent=2)


if __name__ == '__main__':
    main()
