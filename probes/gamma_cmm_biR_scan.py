#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""gamma_cmm_biR_scan.py -- NEW FILE. CPU-only. cmm with R FORCED to
{audio,video}: E^R = kappa_audio + kappa_video (unweighted, lambda-free),
decision E^R>0 keep candidate else flip, plus dead-zone tau in {0,0.1,0.25}.
Reuses outputs/cf_probe/cmm_probe_full.jsonl. No GPU.
"""
import os, sys, json, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gamma_operator import YES_ID, NO_ID, candidate_anchored_decision

TOK2LAB = {YES_ID: 'yes', NO_ID: 'no'}
PATH = 'outputs/cf_probe/cmm_probe_full.jsonl'
BASE, OLDL, SINGLE = 0.6583, 0.8125, 0.7139


def cells(base_ok, new_ok):
    c = collections.Counter()
    for b, n in zip(base_ok, new_ok):
        if not b and n:   c['fixed'] += 1
        elif b and not n: c['broke'] += 1
        elif b and n:     c['kept'] += 1
        else:             c['still_wrong'] += 1
    return c


def main():
    rows = [json.loads(l) for l in open(PATH) if l.strip()]
    samples = []
    for r in rows:
        gt = str(r['gt']).strip().lower()
        er_yes = ((r['margin_full'] - r['margin_noaudio'])
                  + (r['margin_full'] - r['margin_novideo']))
        er = {YES_ID: er_yes, NO_ID: -er_yes}
        full_lp = {YES_ID: r['lp_yes_full'], NO_ID: r['lp_no_full']}
        samples.append((gt, r['dec_token'], er, full_lp))
    n = len(samples)
    base_ok = [TOK2LAB.get(d) == g for (g, d, _, _) in samples]
    print(f'===== CMM dual-modality R={{audio,video}}  (n={n}) =====')
    print(f'  refs: base={BASE:.4f}  old-lambda={OLDL:.4f}  single-mod-gamma(best tau=0)={SINGLE:.4f}')
    print(f'  {"tau":>5} | {"acc":>7} | {"d-base":>8} | {"vs-old":>8} | fixed broke  kept  still')
    for tau in (0.0, 0.1, 0.25):
        new_tok = [candidate_anchored_decision(d, er, fl, tau)
                   for (_, d, er, fl) in samples]
        new_ok = [TOK2LAB.get(x) == g for x, (g, _, _, _) in zip(new_tok, samples)]
        acc = sum(new_ok) / n
        c = cells(base_ok, new_ok)
        print(f'  {tau:>5} | {acc:>7.4f} | {acc-BASE:>+8.4f} | {acc-OLDL:>+8.4f} | '
              f'{c["fixed"]:>5} {c["broke"]:>5} {c["kept"]:>5} {c["still_wrong"]:>5}')


if __name__ == '__main__':
    main()
