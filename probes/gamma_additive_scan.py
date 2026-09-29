#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""gamma_additive_scan.py -- NEW FILE. CPU-only. Additive gamma variant:
  margin'_{d*} = F_{d*}(X) + E_{d*}^R
where E_{d*}^R = sum_{m in R}[F_{d*}(X) - F_{d*}(X^{-m})] (per-sample
single-modality R, same as the single-mod gamma scan). Dead-zone: flip only
when |E_{d*}^R| >= tau AND E_{d*}^R pushes decision; else keep d*.

Specifically: flip d* only when margin'_{d*} <= 0 AND E_{d*}^R < -tau.
(tau=0 => flip whenever margin'_{d*} <= 0, i.e. pure additive sign rule.)
Flip target: full-view runner-up.

Checks phd and pubmedqa at tau=0 for non-regression, then scans cmm over
tau in {0, 0.05, 0.1, 0.25, 0.5, 1.0}.
"""
import os, sys, json, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gamma_operator import YES_ID, NO_ID, MAYBE_ID, relevant_modalities_cmm

TOK2LAB = {YES_ID: 'yes', NO_ID: 'no', MAYBE_ID: 'maybe'}
TAUS = [0.0, 0.05, 0.1, 0.25, 0.5, 1.0]


def _load(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def cells(base_ok, new_ok):
    c = collections.Counter()
    for b, n in zip(base_ok, new_ok):
        if not b and n:   c['fixed'] += 1
        elif b and not n: c['broke'] += 1
        elif b and n:     c['kept'] += 1
        else:             c['still_wrong'] += 1
    return c


def additive_decision(d_star, margin_full, er_star, full_lp, tau):
    """margin'_{d*} = margin_full + er_star.
    Flip only if margin'_{d*} <= 0 AND er_star < -tau.
    Flip target: full-view runner-up."""
    margin_prime = margin_full + er_star
    if margin_prime > 0 or er_star >= -tau:
        return d_star
    # flip to runner-up
    others = {k: v for k, v in full_lp.items() if k != d_star}
    return max(others, key=others.get)


# ---------- phd ----------
def run_phd(path):
    rows = _load(path)
    gt, base_tok, new_tok = [], [], []
    for r in rows:
        cf = r['cf']
        if cf.get('dec_index') is None:
            continue
        gt.append(str(r['gt']).strip().lower())
        d_star = cf['greedy_token']
        base_tok.append(d_star)
        # margin_full for d*=YES: ly-ln; for d*=NO: ln-ly
        if d_star == YES_ID:
            margin_full = cf['lp_yes_full'] - cf['lp_no_full']
            # E^R_yes = F_yes(X) - F_yes(X^{-context}); F_yes = ly-ln
            er_star = margin_full - (cf['lp_yes_cf'] - cf['lp_no_cf'])
            full_lp = {YES_ID: cf['lp_yes_full'], NO_ID: cf['lp_no_full']}
        else:
            margin_full = cf['lp_no_full'] - cf['lp_yes_full']
            er_star = margin_full - (cf['lp_no_cf'] - cf['lp_yes_cf'])
            full_lp = {YES_ID: cf['lp_yes_full'], NO_ID: cf['lp_no_full']}
        tok = additive_decision(d_star, margin_full, er_star, full_lp, tau=0.0)
        new_tok.append(tok)
    n = len(gt)
    base_ok = [TOK2LAB.get(b) == g for b, g in zip(base_tok, gt)]
    new_ok  = [TOK2LAB.get(x) == g for x, g in zip(new_tok,  gt)]
    acc_base = sum(base_ok)/n; acc_new = sum(new_ok)/n
    print(f'PHD tau=0: base={acc_base:.4f}  additive-gamma={acc_new:.4f}  '
          f'(expect >=0.775, old-lambda=0.7561)')


# ---------- pubmedqa ----------
def run_pubmedqa(path):
    rows = _load(path)
    gt, base_tok, new_tok = [], [], []
    for r in rows:
        cf = r['cf']
        if cf.get('dec_index') is None:
            continue
        gt.append(str(r['gt']).split()[0].rstrip('.').lower())
        d_star = cf['greedy_token']
        base_tok.append(d_star)
        full_lp = {YES_ID: cf['lp_yes_full'], NO_ID: cf['lp_no_full'],
                   MAYBE_ID: cf['lp_maybe_full']}
        cf_lp   = {YES_ID: cf['lp_yes_cf'], NO_ID: cf['lp_no_cf'],
                   MAYBE_ID: cf['lp_maybe_cf']}
        # margin for d* = lp_d* - max_{c!=d*} lp_c (softmax gap)
        runner_up = max((c for c in full_lp if c != d_star), key=full_lp.get)
        margin_full = full_lp[d_star] - full_lp[runner_up]
        er_star = (full_lp[d_star] - cf_lp[d_star])   # passage contribution
        tok = additive_decision(d_star, margin_full, er_star, full_lp, tau=0.0)
        new_tok.append(tok)
    n = len(gt)
    base_ok = [TOK2LAB.get(b) == g for b, g in zip(base_tok, gt)]
    new_ok  = [TOK2LAB.get(x) == g for x, g in zip(new_tok,  gt)]
    acc_base = sum(base_ok)/n; acc_new = sum(new_ok)/n
    print(f'PUBMEDQA tau=0: base={acc_base:.4f}  additive-gamma={acc_new:.4f}  '
          f'(expect >=0.900, old-lambda=0.9017)')


# ---------- cmm (tau sweep) ----------
def run_cmm(path):
    rows = _load(path)
    # precompute per-sample (gt, d_star, margin_full, er_star, full_lp)
    samples = []
    for r in rows:
        gt = str(r['gt']).strip().lower()
        d_star = r['dec_token']
        R, _ = relevant_modalities_cmm(r.get('question'), r.get('group'))
        er_yes = 0.0
        if 'audio' in R:
            er_yes += r['margin_full'] - r['margin_noaudio']
        if 'video' in R:
            er_yes += r['margin_full'] - r['margin_novideo']
        # margin_full for d*
        if d_star == YES_ID:
            margin_full = r['margin_full']
            er_star = er_yes
        else:
            margin_full = -r['margin_full']
            er_star = -er_yes
        full_lp = {YES_ID: r['lp_yes_full'], NO_ID: r['lp_no_full']}
        samples.append((gt, d_star, margin_full, er_star, full_lp))

    n = len(samples)
    base_ok = [TOK2LAB.get(d) == g for (g, d, _, _, _) in samples]
    BASE, OLDL = 0.6583, 0.8125
    print(f'\nCMM additive-gamma tau sweep  (n={n})')
    print(f'  refs: base={BASE:.4f}  old-lambda={OLDL:.4f}  '
          f'single-mod-gamma(tau=0)=0.7139')
    print(f'  {"tau":>5} | {"acc":>7} | {"d-base":>8} | {"vs-old":>8} | '
          f'fixed broke  kept  still')
    for tau in TAUS:
        new_tok = [additive_decision(d, mf, er, fl, tau)
                   for (_, d, mf, er, fl) in samples]
        new_ok = [TOK2LAB.get(x) == g for x, (g, *_) in zip(new_tok, samples)]
        acc = sum(new_ok)/n
        c = cells(base_ok, new_ok)
        print(f'  {tau:>5} | {acc:>7.4f} | {acc-BASE:>+8.4f} | {acc-OLDL:>+8.4f} | '
              f'{c["fixed"]:>5} {c["broke"]:>5} {c["kept"]:>5} {c["still_wrong"]:>5}')


def main():
    root = '.'
    j = lambda p: os.path.join(root, p)
    run_phd(j('outputs/phd_cf_confess/cf_lam1.0_full.jsonl'))
    run_pubmedqa(j('outputs/pubmedqa_cf_confess/cf_lam0.25_full.jsonl'))
    run_cmm(j('outputs/cf_probe/cmm_probe_full.jsonl'))


if __name__ == '__main__':
    main()
