# -*- coding: utf-8 -*-
"""Z10 entropy-threshold routing baseline (zero-GPU, offline).

Question: why route interventions with the counterfactual delta (CF delta)
instead of a free entropy signal?  This script builds the trivial baseline:
trigger on the generation-time entropy trajectory (approximated by the
per-token surprisal -logp of the recorded greedy trajectory, `logp_list`)
at the SAME intervention budget as the CF-delta trigger, and compares
fix / destroy / no-change outcomes.

Data (self-contained rows: each cf_confess run records both the offline-arm
quantities -- lp_yes/no(/maybe)_full, margin_full, delta -- and the base
greedy trajectory logp_list):
  phd      outputs/phd_cf_confess/cf_lam2.0_full.jsonl      (lam*=2.0)
  cmm      outputs/cmm_cf_confess/cf_lam1.0_v6_full.jsonl   (lam*=1.0)
  pubmedqa outputs/pubmedqa_cf_confess/cf_lam0.25_full.jsonl (lam*=0.25)

Offline decision rules (identical to lambda_sweep.py, line-checked):
  phd/cmm  binary: base = yes iff margin_full>0 ;
           adj = yes iff margin_full + lam*delta_eff > 0
  pubmedqa 3-way: argmax over {yes,no,maybe} of lp_*_full, with
           yes_adj = lp_yes_full + lam*d_eff ; pred 'maybe' counts as a
           miss for yes-correctness (identical to sim_pubmedqa).

Trigger = base_pred != adj_pred (this equals the online `intervened` flag).
Entropy scores per sample (from logp_list, surprisal proxy):
  peak   = max surprisal over trajectory
  mean   = mean surprisal
  tail_q = top-decile mean surprisal
Baseline trigger = top-B samples by score, B = CF trigger budget.
Both arms apply the SAME intervention (the CF-adjusted decision).

Caveat: logp_list gives surprisal of the chosen tokens, not the full-vocab
softmax entropy; the full-vocab variant exists only as a 3-sample smoke
(outputs/cf_probe/phd_entropy_smoke*).  Surprisal is the standard cheap
proxy; recorded here as a limitation.

Output: outputs/problem_structure_fig/entropy_routing_baseline.{md,json}
"""
import json, os
import numpy as np

ROOT = '/home/tsq/rhq/omniconfess/OmniConfess/outputs'
OUT = os.path.join(ROOT, 'problem_structure_fig')

RUNS = [
    ('phd', f'{ROOT}/phd_cf_confess/cf_lam2.0_full.jsonl', 2.0),
    ('cmm', f'{ROOT}/cmm_cf_confess/cf_lam1.0_v6_full.jsonl', 1.0),
    ('pubmedqa', f'{ROOT}/pubmedqa_cf_confess/cf_lam0.25_full.jsonl', 0.25),
]

YES = {'yes', 'y', 'true', '1'}


def gt01(g):
    return 1 if str(g).strip().lower() in YES else 0


def binary_arm(r):
    cf = r['cf']
    m, d, lam = cf['margin_full'], cf['delta_eff'], cf['lambda_cf']
    base = 1 if m > 0 else 0
    adj = 1 if m + lam * d > 0 else 0
    return base, adj


def pubmedqa_arm(r):
    cf = r['cf']
    l = {'yes': cf['lp_yes_full'], 'no': cf['lp_no_full'],
         'maybe': cf['lp_maybe_full']}
    base = max(l, key=l.get)
    l['yes'] += cf['lambda_cf'] * cf['d_eff']
    adj = max(l, key=l.get)
    return base, adj


def correct(ds, pred, gt):
    if ds == 'pubmedqa':
        return int(pred == 'yes' and gt01(gt) == 1) if gt01(gt) == 1 \
            else int(pred == 'no')
    return int(gt01(gt) == (1 if pred in (1, 'yes') else 0))


def outcome(ds, base, adj, gt):
    """fix / destroy / neutral for a sample the router selected."""
    cb, ca = correct(ds, base, gt), correct(ds, adj, gt)
    if cb == 0 and ca == 1:
        return 'fix'
    if cb == 1 and ca == 0:
        return 'destroy'
    return 'neutral'   # includes wrong->wrong and no decision change


def surprisal_scores(logp):
    s = np.asarray([-min(p, 0.0) for p in (logp or [])], float)
    if len(s) == 0:
        return dict(peak=0.0, mean=0.0, tail_q=0.0)
    k = max(1, int(np.ceil(0.1 * len(s))))
    return dict(peak=round(float(s.max()), 4),
                mean=round(float(s.mean()), 4),
                tail_q=round(float(np.sort(s)[-k:].mean()), 4))


def analyse(ds, path, lam):
    rows = [json.loads(l) for l in open(path)]
    recs = []
    for r in rows:
        base, adj = (binary_arm(r) if ds != 'pubmedqa' else pubmedqa_arm(r))
        trig = base != adj
        recs.append(dict(id=r['id'], gt=r['gt'], base=base, adj=adj,
                         trig=trig, ent=surprisal_scores(r.get('logp_list'))))
    n = len(recs)
    budget = sum(r['trig'] for r in recs)

    def arm_stats(sel):
        c = {'fix': 0, 'destroy': 0, 'neutral': 0}
        for r in sel:
            c[outcome(ds, r['base'], r['adj'], r['gt'])] += 1
        return c

    out = dict(ds=ds, path=path, lam=lam, n=n, budget=budget,
               trigger_rate=round(budget / n, 4),
               cf_delta=arm_stats([r for r in recs if r['trig']]))
    for key in ('peak', 'mean', 'tail_q'):
        order = sorted(recs, key=lambda r: -r['ent'][key])
        out[f'entropy_{key}'] = arm_stats(order[:budget])
        # agreement between selectors (which samples both arms pick)
        s_ent = {r['id'] for r in order[:budget]}
        s_cf = {r['id'] for r in recs if r['trig']}
        out[f'entropy_{key}']['overlap_with_cf'] = len(s_ent & s_cf)
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    res = [analyse(*a) for a in RUNS]
    json.dump(res, open(f'{OUT}/entropy_routing_baseline.json', 'w'),
              indent=1, ensure_ascii=False)

    L = ['# Z10 Entropy-threshold routing baseline vs CF-delta routing '
         '(zero-GPU, offline)', '',
         'Same-budget comparison: the CF-delta trigger (margin + lam*delta '
         'crossing the decision boundary, lam* per dataset) vs a trivial '
         'entropy trigger (top-B by trajectory surprisal, B = CF budget). '
         'Both arms apply the identical intervention (CF-adjusted decision).',
         '',
         'Entropy proxy: per-token surprisal (-logp) of the recorded greedy '
         'trajectory; NOT the full-vocab softmax entropy (that exists only '
         'as a 3-sample smoke in cf_probe/phd_entropy_smoke*). Limitation.',
         '']
    hdr = ('| dataset | n | budget | router | fix | destroy | neutral | '
           'net gain (fix-destroy) | actionable | overlap with CF |')
    L += [hdr, '|' + '---|' * 10]
    for r in res:
        for router in (['cf_delta'] + [f'entropy_{k}' for k in
                                       ('peak', 'mean', 'tail_q')]):
            a = r[router]
            ov = a.get('overlap_with_cf', '')
            act = a['fix'] + a['destroy']
            L.append(f"| {r['ds']} | {r['n']} | {r['budget']} ({r['trigger_rate']:.1%}) "
                     f"| {router} | {a['fix']} | {a['destroy']} | {a['neutral']} "
                     f"| {a['fix'] - a['destroy']} | {act}/{r['budget']} | {ov} |")
    L += ['', '## Reading',
          '- fix = wrong->right among selected; destroy = right->wrong among '
          'selected; neutral = no decision change or wrong->wrong.',
          '- A useful router maximises fix and minimises destroy at the same '
          'budget; entropy "wins" only if its fix/destroy ratio beats CF.',
          '- overlap tells how much of the entropy selection is even in the '
          'CF-actionable set.',
          '- headline number is NET GAIN at equal budget: the entropy router '
          'mostly burns the budget on samples the intervention cannot change '
          '(low actionable fraction), so its raw fix/destroy ratio is not '
          'comparable -- compare net gain.']
    open(f'{OUT}/entropy_routing_baseline.md', 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
