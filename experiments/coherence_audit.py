# -*- coding: utf-8 -*-
"""Z9 coherence audit (zero-GPU): does CF / contract intervention degrade
output fluency / coherence proxies?

Pairs base outputs with intervention outputs (matched on question+gt, id),
computes per-pair text statistics and a paired sign test (two-sided, exact
binomial via scipy).

Metrics (all word-level, lowercase, simple \\w+ tokenisation):
  len_ratio        n_words(intervention) / n_words(base)
  ttr              distinct words / n_words          (per side, delta reported)
  rep_bigram       fraction of bigram tokens that are repeats of an earlier
                   bigram (degeneration indicator, higher = worse)
  rep_trigram      same for trigrams
  novel_bigram     fraction of output bigrams absent from the prompt
                   (higher = more novel content; not a degeneration metric)
  sentlen_var      variance of sentence lengths (low = monotone rhythm)
  n_words          output length

Pairings:
  judgement family (base_model vs cf_confess at lam*):
    phd      outputs/phd_base_model/phd_base_model_full.jsonl
             vs outputs/phd_cf_confess/cf_lam2.0_full.jsonl
    cmm      outputs/cmm_base_model/cmm_base_model_full.jsonl
             vs outputs/cmm_cf_confess/cf_lam1.0_v6_full.jsonl
    pubmedqa outputs/pubmedqa_base_model/pubmedqa_base_model_full.jsonl
             vs outputs/pubmedqa_cf_confess/cf_lam0.25_full.jsonl
  descriptive family (base_model vs contract/gated):
    ragtruth   ragtruth_base_model_full vs ragtruth_contract_v4_full
    haloquest  haloquest_base_model_full vs haloquest_contract_v2_full

Outputs: outputs/problem_structure_fig/coherence_audit.{md,json}

NOTE this is a text-statistics proxy only; human coherence rating is a
separate protocol (HUMAN_EVAL_PACKAGE -- recommend adding an L4 coherence
column to the annotation sheet and recording it in the SOP).
"""
import json, os, re, math
from collections import Counter
import numpy as np
from scipy.stats import binomtest

ROOT = '/home/tsq/rhq/omniconfess/OmniConfess/outputs'
OUT_DIR = os.path.join(ROOT, 'problem_structure_fig')

PAIRS = [
    # (label, family, base_path, cf_path)
    ('phd_cf_lam2.0', 'judgement',
     f'{ROOT}/phd_base_model/phd_base_model_full.jsonl',
     f'{ROOT}/phd_cf_confess/cf_lam2.0_full.jsonl'),
    ('cmm_cf_lam1.0', 'judgement',
     f'{ROOT}/cmm_base_model/cmm_base_model_full.jsonl',
     f'{ROOT}/cmm_cf_confess/cf_lam1.0_v6_full.jsonl'),
    ('pubmedqa_cf_lam0.25', 'judgement',
     f'{ROOT}/pubmedqa_base_model/pubmedqa_base_model_full.jsonl',
     f'{ROOT}/pubmedqa_cf_confess/cf_lam0.25_full.jsonl'),
    ('ragtruth_contract_v4', 'descriptive',
     f'{ROOT}/ragtruth_base_model/ragtruth_base_model_full.jsonl',
     f'{ROOT}/ragtruth_contract_v4/ragtruth_contract_v4_full.jsonl'),
    ('haloquest_contract_v2', 'descriptive',
     f'{ROOT}/haloquest_base_model/haloquest_base_model_full.jsonl',
     f'{ROOT}/haloquest_contract_v2/haloquest_contract_v2_full.jsonl'),
]

WORD = re.compile(r"[a-z0-9']+")


def words(t):
    return WORD.findall((t or '').lower())


def sentences(t):
    return [s for s in re.split(r'[.!?;\n]+', t or '') if s.strip()]


def repeat_ngram_rate(w, n):
    if len(w) < n:
        return 0.0
    grams = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    c = Counter(grams)
    rep = sum(v - 1 for v in c.values())
    return rep / len(grams)


def novel_bigram_rate(w, prompt_w):
    if len(w) < 2:
        return 0.0
    pset = set(prompt_w)
    grams = [tuple(w[i:i + 2]) for i in range(len(w) - 1)]
    nov = sum(1 for g in grams if g[0] not in pset or g[1] not in pset)
    return nov / len(grams)


def stats(text, prompt):
    w = words(text)
    sents = sentences(text)
    slens = [len(words(s)) for s in sents] or [len(w)]
    return dict(
        n_words=len(w),
        ttr=len(set(w)) / len(w) if w else 0.0,
        rep_bigram=repeat_ngram_rate(w, 2),
        rep_trigram=repeat_ngram_rate(w, 3),
        novel_bigram=novel_bigram_rate(w, words(prompt)),
        sentlen_var=float(np.var(slens)),
    )


def load(path):
    """Return list of (pair_key, row). Key = (question, gt); duplicates are
    queued in file order so multiplicity-aware pairing works (cmm reuses the
    same question text across many media samples and run ids are not aligned
    across runs -- media identity is not recorded, so cmm pairs with the same
    question+gt are matched greedily; caveat noted in the report)."""
    rows = [json.loads(l) for l in open(path)]
    queues = {}
    out = []
    for r in rows:
        k = (r['question'][:300], str(r['gt']))
        queues.setdefault(k, []).append(r)
    for r in rows:
        k = (r['question'][:300], str(r['gt']))
        out.append((k, queues[k].pop(0)))
    return out


def sign_test(d):
    """d = per-pair metric (intervention - base); two-sided sign test on nonzero."""
    d = np.asarray(d, float)
    nz = d[d != 0]
    n = len(nz)
    if n == 0:
        return dict(n_pairs=len(d), n_nonzero=0, median_delta=float(np.median(d)),
                    p_value=None)
    res = binomtest(int((nz > 0).sum()), n, 0.5)
    return dict(n_pairs=len(d), n_nonzero=n,
                median_delta=round(float(np.median(d)), 4),
                mean_delta=round(float(np.mean(d)), 4),
                pos=int((nz > 0).sum()), neg=int((nz < 0).sum()),
                p_value=round(res.pvalue, 5))


DELTA_METRICS = ['ttr', 'rep_bigram', 'rep_trigram', 'novel_bigram',
                 'sentlen_var', 'n_words']


def audit(label, family, bp, cp):
    bq, cq = {}, {}
    for k, r in load(bp):
        bq.setdefault(k, []).append(r)
    for k, r in load(cp):
        cq.setdefault(k, []).append(r)
    keys = sorted(set(bq) & set(cq))
    rows, ratios = [], []
    deltas = {m: [] for m in DELTA_METRICS}
    for k in keys:
        for b, c in zip(bq[k], cq[k]):
            sb = stats(b.get('raw_output') or b.get('pred', ''),
                       b.get('question', ''))
            sc = stats(c.get('raw_output') or c.get('pred', ''),
                       c.get('question', ''))
            rows.append((k, sb, sc))
            if sb['n_words'] > 0:
                ratios.append(sc['n_words'] / sb['n_words'])
            for m in DELTA_METRICS:
                deltas[m].append(sc[m] - sb[m])
    agg = {}
    for m in DELTA_METRICS:
        b_mean = float(np.mean([sb[m] for _, sb, _ in rows]))
        c_mean = float(np.mean([sc[m] for _, _, sc in rows]))
        agg[m] = dict(base_mean=round(b_mean, 4), cf_mean=round(c_mean, 4),
                      sign_test=sign_test(deltas[m]))
    return dict(
        label=label, family=family, n_pairs=len(rows),
        len_ratio_mean=round(float(np.mean(ratios)), 4) if ratios else None,
        len_ratio_median=round(float(np.median(ratios)), 4) if ratios else None,
        metrics=agg,
    )


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    results = [audit(*p) for p in PAIRS]
    with open(f'{OUT_DIR}/coherence_audit.json', 'w') as f:
        json.dump(results, f, indent=1, ensure_ascii=False)

    lines = ['# Z9 Coherence audit: does intervention degrade output '
             'coherence? (text-statistics proxy, zero-GPU)',
             '',
             'Paired by (question, gt) with multiplicity (file order): base '
             'output vs intervention '
             'output. Sign test = exact two-sided binomial on per-pair '
             'deltas (intervention - base).',
             '',
             '**Proxy only**: these are surface text statistics, not human '
             'coherence ratings. Human L4 coherence rating belongs to '
             'HUMAN_EVAL_PACKAGE (recommend adding an L4 column to the '
             'annotation sheet and recording it in the SOP).',
             '']
    hdr = ('| pair | family | n | len_ratio | TTR b->c (p) | rep-bigram b->c (p) '
           '| rep-trigram b->c (p) | novel-bigram b->c (p) | sentlen-var b->c (p) '
           '| words b->c (p) |')
    lines += [hdr, '|' + '---|' * 10]
    for r in results:
        m = r['metrics']
        cell = lambda k: (f"{m[k]['base_mean']:.3f}->{m[k]['cf_mean']:.3f} "
                          f"(p={m[k]['sign_test'].get('p_value')})")
        lines.append(
            f"| {r['label']} | {r['family']} | {r['n_pairs']} | "
            f"{r['len_ratio_mean']} | " + ' | '.join(
                cell(k) for k in ['ttr', 'rep_bigram', 'rep_trigram',
                                  'novel_bigram', 'sentlen_var', 'n_words'])
            + ' |')
    lines += ['', '## Interpretation guide',
              '- rep_bigram / rep_trigram UP with p<0.05 = degeneration '
              'signal (repetitive output) -> limitations material.',
              '- TTR DOWN + rep UP jointly would be a real coherence worry; '
              'Token-F1 stable while repetition rises must be disclosed.',
              '- novel_bigram is a content metric, not degeneration.',
              '- judgement-family outputs are very short (often 1-2 tokens); '
              'their TTR/repetition numbers are near-degenerate by '
              'construction -- weight the descriptive family more heavily.']
    open(f'{OUT_DIR}/coherence_audit.md', 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
