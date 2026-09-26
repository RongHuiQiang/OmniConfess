#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""gamma_offline_recompute.py -- NEW FILE. CPU-only offline evaluation of the
lambda-free gamma operator against the old lambda operator and the base model,
reusing per-sample per-view scores already cached by the v1 CF-Confess runs.
No GPU. No model load. Reads only existing outputs/*.jsonl caches.

Sources (all pre-existing):
  phd       outputs/phd_cf_confess/cf_lam1.0_full.jsonl
            cf.{lp_yes_full,lp_no_full,lp_yes_cf,lp_no_cf,greedy_token,forced_token}
  pubmedqa  outputs/pubmedqa_cf_confess/cf_lam0.25_full.jsonl
            cf.{lp_{yes,no,maybe}_{full,cf},greedy_token,forced_token}
  cmm       outputs/cf_probe/cmm_probe_full.jsonl
            {margin_full,margin_noaudio,margin_novideo,d_cf_audio,d_cf_video,
             gt,dec_token,group}  (probe carries the NO logprob under each CF
            view, which the cmm operator record does not).
"""
import os, sys, json, argparse, collections

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gamma_operator import (
    YES_ID, NO_ID, MAYBE_ID,
    gamma_margin_phd, gamma_margin_cmm, relevant_modalities_cmm,
    gamma_scores_pubmedqa,
)

TOK2LAB = {YES_ID: 'yes', NO_ID: 'no', MAYBE_ID: 'maybe'}


def _load(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def confusion(base_ok, new_ok):
    """4-cell of NEW vs a reference correctness list."""
    c = collections.Counter()
    for b, n in zip(base_ok, new_ok):
        if not b and n:
            c['fixed'] += 1          # 修对
        elif b and not n:
            c['broke'] += 1          # 破坏对
        elif b and n:
            c['kept_correct'] += 1   # 保持对
        else:
            c['still_wrong'] += 1    # 仍错
    return c


def report(name, gt, base_tok, old_tok, new_tok, extra=''):
    n = len(gt)
    base_ok = [TOK2LAB.get(b) == g for b, g in zip(base_tok, gt)]
    old_ok = [TOK2LAB.get(o) == g for o, g in zip(old_tok, gt)]
    new_ok = [TOK2LAB.get(x) == g for x, g in zip(new_tok, gt)]
    acc = lambda ok: sum(ok) / n
    print(f'\n===== {name}  (n={n}) {extra} =====')
    print(f'  base acc      : {acc(base_ok):.4f}  ({sum(base_ok)}/{n})')
    print(f'  old-lambda acc: {acc(old_ok):.4f}  ({sum(old_ok)}/{n})')
    print(f'  new-gamma acc : {acc(new_ok):.4f}  ({sum(new_ok)}/{n})')
    cvb = confusion(base_ok, new_ok)
    print(f'  gamma vs BASE : fixed(修对)={cvb["fixed"]}  broke(破坏对)={cvb["broke"]}  '
          f'kept(保持对)={cvb["kept_correct"]}  still_wrong(仍错)={cvb["still_wrong"]}'
          f'  net={cvb["fixed"]-cvb["broke"]:+d}')
    cvo = confusion(old_ok, new_ok)
    print(f'  gamma vs OLD  : fixed={cvo["fixed"]}  broke={cvo["broke"]}  '
          f'kept={cvo["kept_correct"]}  still_wrong={cvo["still_wrong"]}'
          f'  net={cvo["fixed"]-cvo["broke"]:+d}')
    print(f'  delta acc     : gamma-base={acc(new_ok)-acc(base_ok):+.4f}   '
          f'gamma-old={acc(new_ok)-acc(old_ok):+.4f}   '
          f'old-base={acc(old_ok)-acc(base_ok):+.4f}')
    return dict(n=n, base=acc(base_ok), old=acc(old_ok), new=acc(new_ok))


def run_phd(path):
    rows = _load(path)
    gt, base_tok, old_tok, new_tok = [], [], [], []
    for r in rows:
        cf = r['cf']
        if cf.get('dec_index') is None:
            continue
        gt.append(str(r['gt']).strip().lower())
        base_tok.append(cf['greedy_token'])
        old_tok.append(cf['forced_token'])           # shipped lambda decision
        _, f = gamma_margin_phd(cf['lp_yes_full'], cf['lp_no_full'],
                                cf['lp_yes_cf'], cf['lp_no_cf'])
        new_tok.append(f)
    return report('PHD', gt, base_tok, old_tok, new_tok,
                  extra=f'(old lambda={rows[0]["cf"].get("lambda_cf")})')


def run_pubmedqa(path):
    rows = _load(path)
    gt, base_tok, old_tok, new_tok = [], [], [], []
    for r in rows:
        cf = r['cf']
        if cf.get('dec_index') is None:
            continue
        gt.append(str(r['gt']).split()[0].rstrip('.').lower())
        base_tok.append(cf['greedy_token'])
        old_tok.append(cf['forced_token'])
        _, f = gamma_scores_pubmedqa(cf['lp_yes_full'], cf['lp_no_full'],
                                     cf['lp_maybe_full'], cf['lp_yes_cf'],
                                     cf['lp_no_cf'], cf['lp_maybe_cf'])
        new_tok.append(f)
    return report('PUBMEDQA', gt, base_tok, old_tok, new_tok,
                  extra=f'(old lambda={rows[0]["cf"].get("lambda_cf")})')


def run_cmm(path, lam):
    rows = _load(path)
    gt, base_tok, old_tok, new_tok = [], [], [], []
    src_tally = collections.Counter()
    for r in rows:
        g = str(r['gt']).strip().lower()
        gt.append(g)
        base_tok.append(r['dec_token'])
        # old-lambda decision reproduced from probe fields (equal weights):
        d_sum = r['d_cf_audio'] + r['d_cf_video']
        old_margin = r['margin_full'] + lam * d_sum
        old_tok.append(YES_ID if old_margin > 0 else NO_ID)
        R, src = relevant_modalities_cmm(r.get('question'), r.get('group'))
        src_tally[src] += 1
        _, f = gamma_margin_cmm(r['margin_full'], r['margin_noaudio'],
                                r['margin_novideo'], R)
        new_tok.append(f)
    res = report('CMM', gt, base_tok, old_tok, new_tok,
                 extra=f'(old lambda={lam}; R-source {dict(src_tally)})')
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='.')
    ap.add_argument('--phd', default='outputs/phd_cf_confess/cf_lam1.0_full.jsonl')
    ap.add_argument('--pubmedqa', default='outputs/pubmedqa_cf_confess/cf_lam0.25_full.jsonl')
    ap.add_argument('--cmm', default='outputs/cf_probe/cmm_probe_full.jsonl')
    ap.add_argument('--cmm_lambda', type=float, default=1.0)
    args = ap.parse_args()
    j = lambda p: os.path.join(args.root, p)
    run_phd(j(args.phd))
    run_pubmedqa(j(args.pubmedqa))
    run_cmm(j(args.cmm), args.cmm_lambda)


if __name__ == '__main__':
    main()
