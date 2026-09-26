#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""gamma_deadzone_scan.py -- NEW FILE. CPU-only. Candidate-anchored + dead-zone
gamma variant, tau-swept, compared against literal-gamma / old-lambda / base,
reusing the same cached per-view scores as gamma_offline_recompute.py.

Candidate-anchored decision (see gamma_operator.candidate_anchored_decision):
  d*        = base full-view argmax (committed answer, taken from greedy_token
              for phd/pubmedqa, dec_token for cmm).
  E_{d*}^R  = sum_{m in R} [ F_{d*}(X) - F_{d*}(X^{-m}) ]  (relevant support for d*).
  flip only if E_{d*}^R < -tau; else keep d*. tau=0 -> literal sign rule.
  flip target: other class with largest E_c^R (if positive), else full-view runner-up.

Per-dataset E_c^R:
  phd       ablated channel = irrelevant context, so relevant support for class
            c = clean (image+question) margin toward c: F_c(X^{-context}).
            er[yes]=ly_cf-ln_cf, er[no]=-(ly_cf-ln_cf).  (tau=0 must reproduce 0.775)
  cmm       er[yes]=sum_{m in R}(margin_full-margin_{-m}); er[no]=-er[yes].
  pubmedqa  er[c]=lp_c_full-lp_c_cf (passage contribution to class c logprob).
"""
import os, sys, json, argparse, collections

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gamma_operator import (
    YES_ID, NO_ID, MAYBE_ID,
    relevant_modalities_cmm, candidate_anchored_decision,
)

TOK2LAB = {YES_ID: 'yes', NO_ID: 'no', MAYBE_ID: 'maybe'}
TAUS = [0.0, 0.05, 0.1, 0.25, 0.5, 1.0]


def _load(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def _cells(base_ok, new_ok):
    c = collections.Counter()
    for b, n in zip(base_ok, new_ok):
        if not b and n:      c['fixed'] += 1
        elif b and not n:    c['broke'] += 1
        elif b and n:        c['kept'] += 1
        else:                c['still_wrong'] += 1
    return c


def build_phd(path):
    rows = _load(path)
    samples = []
    for r in rows:
        cf = r['cf']
        if cf.get('dec_index') is None:
            continue
        gt = str(r['gt']).strip().lower()
        er = {YES_ID: cf['lp_yes_cf'] - cf['lp_no_cf']}
        er[NO_ID] = -er[YES_ID]
        full_lp = {YES_ID: cf['lp_yes_full'], NO_ID: cf['lp_no_full']}
        samples.append((gt, cf['greedy_token'], er, full_lp))
    return samples


def build_pubmedqa(path):
    rows = _load(path)
    samples = []
    for r in rows:
        cf = r['cf']
        if cf.get('dec_index') is None:
            continue
        gt = str(r['gt']).split()[0].rstrip('.').lower()
        er = {YES_ID: cf['lp_yes_full'] - cf['lp_yes_cf'],
              NO_ID: cf['lp_no_full'] - cf['lp_no_cf'],
              MAYBE_ID: cf['lp_maybe_full'] - cf['lp_maybe_cf']}
        full_lp = {YES_ID: cf['lp_yes_full'], NO_ID: cf['lp_no_full'],
                   MAYBE_ID: cf['lp_maybe_full']}
        samples.append((gt, cf['greedy_token'], er, full_lp))
    return samples


def build_cmm(path):
    rows = _load(path)
    samples = []
    src = collections.Counter()
    for r in rows:
        gt = str(r['gt']).strip().lower()
        R, s = relevant_modalities_cmm(r.get('question'), r.get('group'))
        src[s] += 1
        er_yes = 0.0
        if 'audio' in R:
            er_yes += r['margin_full'] - r['margin_noaudio']
        if 'video' in R:
            er_yes += r['margin_full'] - r['margin_novideo']
        er = {YES_ID: er_yes, NO_ID: -er_yes}
        full_lp = {YES_ID: r['lp_yes_full'], NO_ID: r['lp_no_full']}
        samples.append((gt, r['dec_token'], er, full_lp))
    return samples, dict(src)


def scan(name, samples, old_acc=None, note=''):
    gt = [s[0] for s in samples]
    base_tok = [s[1] for s in samples]
    n = len(gt)
    base_ok = [TOK2LAB.get(b) == g for b, g in zip(base_tok, gt)]
    base_acc = sum(base_ok) / n
    print(f'\n===== {name}  (n={n}) {note} =====')
    print(f'  base acc={base_acc:.4f}' + (f'   old-lambda acc={old_acc:.4f}' if old_acc is not None else ''))
    print(f'  {"tau":>5} | {"acc":>7} | {"d-base":>7} | fixed broke  kept  still  | vs-old')
    for tau in TAUS:
        new_tok = [candidate_anchored_decision(d, er, fl, tau)
                   for (_, d, er, fl) in samples]
        new_ok = [TOK2LAB.get(x) == g for x, g in zip(new_tok, gt)]
        acc = sum(new_ok) / n
        c = _cells(base_ok, new_ok)
        vsold = f'{acc-old_acc:+.4f}' if old_acc is not None else '  n/a '
        print(f'  {tau:>5} | {acc:>7.4f} | {acc-base_acc:>+7.4f} | '
              f'{c["fixed"]:>5} {c["broke"]:>5} {c["kept"]:>5} {c["still_wrong"]:>5}  | {vsold}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='.')
    ap.add_argument('--phd', default='outputs/phd_cf_confess/cf_lam1.0_full.jsonl')
    ap.add_argument('--pubmedqa', default='outputs/pubmedqa_cf_confess/cf_lam0.25_full.jsonl')
    ap.add_argument('--cmm', default='outputs/cf_probe/cmm_probe_full.jsonl')
    args = ap.parse_args()
    j = lambda p: os.path.join(args.root, p)
    # old-lambda / base reference accuracies from the prior offline recompute
    scan('PHD', build_phd(j(args.phd)), old_acc=0.7561,
         note='(self-check: tau=0 should reproduce literal 0.7748)')
    scan('PUBMEDQA', build_pubmedqa(j(args.pubmedqa)), old_acc=0.9017)
    cmm_s, src = build_cmm(j(args.cmm))
    scan('CMM', cmm_s, old_acc=0.8125, note=f'(R-source {src})')


if __name__ == '__main__':
    main()
