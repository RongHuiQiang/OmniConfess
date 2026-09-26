#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""H4 formal subgroup statistic: confident-but-fragile -- zero-GPU, CPU only.

Formalizes the number cited in paper RQ4 (main.tex L277):
  "gaining +14.2 F1 in the confident-but-fragile subgroup of high confidence
   with large confessed dependence, against +3.9 for the strongest
   contrastive baseline, and leaving the confident-and-solid subgroup
   untouched"
which until now rested only on the ad-hoc audit /tmp/audit_b_h4.py (H4 panel
[5]) and a coordinator spot check (n=44 caliber, +18.2 vs +4.5, same
direction).  This script is the registered, reproducible source.

Registered caliber (fixed, no free parameters):

  Data.   PHD full set, n=906, three arms on disk:
            base    outputs/phd_base_model/phd_base_model_full_single_scored.jsonl
            CF      outputs/phd_cf_confess/cf_lam2.0_full_single_scored.jsonl
            CAD     outputs/phd_base_cad/phd_base_cad_full_single_scored.jsonl
          Pairing = (question, gt) multiset, positional occurrence order,
          i.e. ablation/util.py pair_by_question (the registrar-verified
          pairing caliber; identical to audit B's pair()).  All rows have
          cf fields (delta_cf, margin_full) so no row is dropped.

  Grouping. Double median split on the CF arm's confession fields, medians
          taken over the FULL 906 (not within any subset):
            conf  = |cf.margin_full|   (full-set margin, high = confident)
            frag  = |cf.delta_cf|      (confessed dependence, high = fragile)
            HIGHconf: conf >= median(conf);  HIGHfrag: frag >= median(frag).
          Four quadrants reported; the three cited in the paper are
            HIGHconf & HIGHfrag  = confident-but-fragile   (claim: CF moves)
            HIGHconf & LOWfrag   = confident-and-solid     (claim: untouched)
            LOWconf  & HIGHfrag  = unconfident-but-fragile (bonus reading)

  Metric. Per-sample accuracy (eval_metrics Ground_Truth/Prediction, no
          maybe-gt rows in PHD, maybe-pred counted as miss) -- the audit B
          caliber and the caliber of the +14.2/+3.9 citation.  Yes-F1 per
          subgroup is ALSO reported as a secondary caliber to expose the
          metric-label question (the paper sentence says "F1" while the
          cited number is the accuracy gain).

  Stats.  Gains are paired (same rows, arm vs base).  Paired exact McNemar
          vs base and CF vs CAD are reported with discordant counts.

Output: ablation/h4_subgroup.json + stdout table.  Deterministic, no GPU,
no new sampling, does not touch RESULTS.md (registration is a separate act).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from util import pair_by_question, mcnemar_exact  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.path.join(HERE, "h4_subgroup.json")

FILES = {
    'base': os.path.join(ROOT, 'outputs/phd_base_model/phd_base_model_full_single_scored.jsonl'),
    'cf': os.path.join(ROOT, 'outputs/phd_cf_confess/cf_lam2.0_full_single_scored.jsonl'),
    'cad': os.path.join(ROOT, 'outputs/phd_base_cad/phd_base_cad_full_single_scored.jsonl'),
}


def ok(r):
    """Per-sample correctness, single_eval caliber (maybe-pred = miss)."""
    em = r['eval_metrics']
    g, p = em['Ground_Truth'], em['Prediction']
    if g == 2:
        return None
    p = (1 - g) if p == 2 else p
    return int(p == g)


def yes_f1(rows_idx, preds, gts):
    """Yes-F1 (positive class = yes = 1) over the given row indices."""
    idx = list(rows_idx)
    tp = sum(1 for i in idx if gts[i] == 1 and preds[i] == 1)
    fp = sum(1 for i in idx if gts[i] == 0 and preds[i] == 1)
    fn = sum(1 for i in idx if gts[i] == 1 and preds[i] == 0)
    return 200.0 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0


def pred01(r):
    g, p = r['eval_metrics']['Ground_Truth'], r['eval_metrics']['Prediction']
    return (1 - g) if p == 2 else p


def med(xs):
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def subgroup_stats(tag, idx, arms, gts):
    """Accuracy + Yes-F1 per arm, paired gains, McNemar."""
    out = {'n': len(idx)}
    for name, (okv, pred) in arms.items():
        acc = 100.0 * sum(okv[i] for i in idx) / len(idx)
        out[name] = {'acc': round(acc, 2), 'f1': round(yes_f1(idx, pred, gts), 2)}
    for name in ('cf', 'cad'):
        out[name]['acc_gain'] = round(out[name]['acc'] - out['base']['acc'], 2)
        out[name]['f1_gain'] = round(out[name]['f1'] - out['base']['f1'], 2)
    out['cf_minus_cad'] = {
        'acc': round(out['cf']['acc'] - out['cad']['acc'], 2),
        'f1': round(out['cf']['f1'] - out['cad']['f1'], 2),
    }
    # paired exact McNemar (arm vs base; then cf vs cad), accuracy discordants
    for a, b in (('cf', 'base'), ('cad', 'base'), ('cf', 'cad')):
        n_ab = sum(1 for i in idx if arms[a][0][i] == 0 and arms[b][0][i] == 1)
        n_ba = sum(1 for i in idx if arms[a][0][i] == 1 and arms[b][0][i] == 0)
        # fix = a right where b wrong; destroy = a wrong where b right
        out.setdefault('mcnemar', {})['%s_vs_%s' % (a, b)] = {
            'fix': n_ba, 'destroy': n_ab, 'p': round(mcnemar_exact(n_ab, n_ba), 5)}
    return out


def main():
    base, cf, cad = pair_by_question(FILES['base'], FILES['cf'], FILES['cad'])
    n = len(base)
    assert n == 906, n

    arms = {
        'base': ([ok(r) for r in base], [pred01(r) for r in base]),
        'cf': ([ok(r) for r in cf], [pred01(r) for r in cf]),
        'cad': ([ok(r) for r in cad], [pred01(r) for r in cad]),
    }
    assert all(v is not None for m in arms.values() for v in m[0]), 'maybe-gt row found'
    gts = [r['eval_metrics']['Ground_Truth'] for r in base]

    # grouping fields from the CF arm, full-set medians
    conf = [abs(r['cf']['margin_full']) for r in cf]
    frag = [abs(r['cf']['delta_cf']) for r in cf]
    mc, mf = med(conf), med(frag)

    def quad(hi_conf, hi_frag):
        return [i for i in range(n)
                if ((conf[i] >= mc) if hi_conf else (conf[i] < mc))
                and ((frag[i] >= mf) if hi_frag else (frag[i] < mf))]

    res = {
        'script': 'ablation/h4_subgroup_confident_fragile.py',
        'caliber': ('PHD full n=906, arms base/cf_lam2.0/base_cad, pairing '
                    '= (question,gt) multiset positional (util.pair_by_question), '
                    'subgroup = double full-set median split on |margin_full| '
                    '(conf) and |delta_cf| (frag), HIGH = >= median'),
        'files': {k: os.path.relpath(v, ROOT) for k, v in FILES.items()},
        'n': n,
        'median_abs_margin_full': round(mc, 4),
        'median_abs_delta_cf': round(mf, 4),
        'validate': {
            'base_acc_full': round(100.0 * sum(arms['base'][0]) / n, 2),
            'cf_acc_full': round(100.0 * sum(arms['cf'][0]) / n, 2),
            'cad_acc_full': round(100.0 * sum(arms['cad'][0]) / n, 2),
            'base_yes_f1_full': round(yes_f1(range(n), arms['base'][1], gts), 2),
            'cf_yes_f1_full': round(yes_f1(range(n), arms['cf'][1], gts), 2),
            'cad_yes_f1_full': round(yes_f1(range(n), arms['cad'][1], gts), 2),
            'registered_expect': ('base F1 71.68 / CF F1 84.60 / CAD F1 79.40 '
                                  '(RESULTS.md rows: base_model, cf_lam2.0, base_cad)'),
        },
        'subgroups': {},
    }
    for tag, hc, hf, label in [
            ('high_conf_high_frag', True, True, 'confident-but-fragile'),
            ('high_conf_low_frag', True, False, 'confident-and-solid'),
            ('low_conf_high_frag', False, True, 'unconfident-but-fragile'),
            ('low_conf_low_frag', False, False, 'low-conf-low-frag')]:
        idx = quad(hc, hf)
        res['subgroups'][tag] = {'label': label}
        res['subgroups'][tag].update(subgroup_stats(tag, idx, arms, gts))

    with open(OUT, 'w') as f:
        json.dump(res, f, ensure_ascii=False, indent=1)

    v = res['validate']
    print('== H4 subgroup formal statistic (PHD, n=%d, zero GPU) ==' % n)
    print('medians: |margin_full|=%.4f  |delta_cf|=%.4f  (full-set)' % (mc, mf))
    print('validate full-set: acc base %.2f / CF %.2f / CAD %.2f ; '
          'Yes-F1 base %.2f / CF %.2f / CAD %.2f (registered 71.68/84.60/79.40)'
          % (v['base_acc_full'], v['cf_acc_full'], v['cad_acc_full'],
             v['base_yes_f1_full'], v['cf_yes_f1_full'], v['cad_yes_f1_full']))
    print()
    hdr = '%-24s %5s | %13s | %17s | %17s | %s' % (
        'subgroup', 'n', 'base acc/F1', 'CF acc(F1) gains', 'CAD acc(F1) gains', 'CF-CAD acc (McNemar p cf-vs-base)')
    print(hdr)
    print('-' * len(hdr))
    for tag in ('high_conf_high_frag', 'high_conf_low_frag',
                'low_conf_high_frag', 'low_conf_low_frag'):
        s = res['subgroups'][tag]
        print('%-24s %5d | %6.2f %6.2f | %6.2f (%+6.2f) [%+6.2f] | %6.2f (%+6.2f) [%+6.2f] | %+6.2f  (p=%.3g)' % (
            tag, s['n'],
            s['base']['acc'], s['base']['f1'],
            s['cf']['acc'], s['cf']['acc_gain'], s['cf']['f1_gain'],
            s['cad']['acc'], s['cad']['acc_gain'], s['cad']['f1_gain'],
            s['cf_minus_cad']['acc'], s['mcnemar']['cf_vs_base']['p']))
    print()
    print('paper citation caliber (accuracy gains, confident-but-fragile): '
          'CF %+0.2f vs CAD %+0.2f' % (
              res['subgroups']['high_conf_high_frag']['cf']['acc_gain'],
              res['subgroups']['high_conf_high_frag']['cad']['acc_gain']))
    print('same subgroup, Yes-F1 caliber: CF %+0.2f vs CAD %+0.2f' % (
        res['subgroups']['high_conf_high_frag']['cf']['f1_gain'],
        res['subgroups']['high_conf_high_frag']['cad']['f1_gain']))
    print('confident-and-solid (paper: untouched): CF acc %+0.2f, F1 %+0.2f' % (
        res['subgroups']['high_conf_low_frag']['cf']['acc_gain'],
        res['subgroups']['high_conf_low_frag']['cf']['f1_gain']))
    print('written:', OUT)


if __name__ == '__main__':
    main()
