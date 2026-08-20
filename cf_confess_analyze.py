# -*- coding: utf-8 -*-
"""cf_confess_analyze.py — offline scoring for CF-Confess ragtruth runs.

Computes, for one or more result jsonl files (base or cf_confess):
  - mean Token-F1 (tokenf1_fullset footing)
  - refusal rate ("unable to answer based on given passages" in pred)
  - mean pred word count
  - for cf files: content-position rootless rate of the FINAL trajectory
    (|delta_chosen| < eps among recorded content positions), intervention /
    changed rates
  - per-id F1 delta vs the base run (paired comparison, same ids)

Usage:
    python cf_confess_analyze.py --base outputs/ragtruth_base_model/..._scored.jsonl \
        a.jsonl b.jsonl ...
"""
import json, sys, argparse, os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts'))
from tokenf1_fullset import token_f1  # noqa: E402

REFUSAL = 'unable to answer based on given passages'


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def summarize(recs, eps=0.1):
    f1s = [token_f1(r.get('pred', ''), r.get('gt', '')) for r in recs]
    refusals = [1 if REFUSAL in str(r.get('pred', '')).lower() else 0 for r in recs]
    out = {
        'n': len(recs),
        'token_f1': round(100 * sum(f1s) / max(len(f1s), 1), 2),
        'refusal_rate': round(sum(refusals) / max(len(recs), 1), 4),
        'mean_pred_words': round(sum(len(str(r.get('pred', '')).split()) for r in recs) / max(len(recs), 1), 1),
    }
    cf = [r for r in recs if r.get('cf_stats')]
    if cf:
        cpos = sum(r['cf_stats']['content_pos'] for r in cf)
        rootless = sum(r['cf_stats']['rootless_greedy'] for r in cf)
        interv = sum(r['cf_stats']['intervened'] for r in cf)
        changed = sum(r['cf_stats']['changed'] for r in cf)
        # rootless rate of the final trajectory: |delta of the chosen token| < eps
        n_ch = 0
        rootless_ch = 0
        for r in cf:
            for d in r['cf_stats'].get('deltas_chosen', []):
                n_ch += 1
                rootless_ch += int(abs(d) < eps)
        out.update({
            'content_pos': cpos,
            'rootless_rate_greedy': round(rootless / max(cpos, 1), 4),
            'rootless_rate_final': round(rootless_ch / max(n_ch, 1), 4),
            'intervene_rate': round(interv / max(cpos, 1), 4),
            'changed_rate': round(changed / max(cpos, 1), 4),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default='/home/tsq/rhq/omniconfess/OmniConfess/outputs/ragtruth_base_model/ragtruth_base_model_full_single_scored.jsonl')
    ap.add_argument('--eps', type=float, default=0.1)
    ap.add_argument('files', nargs='+')
    args = ap.parse_args()

    base = {r['id']: r for r in load(args.base)} if args.base else {}

    for path in args.files:
        recs = load(path)
        s = summarize(recs, args.eps)
        print(f'\n== {os.path.basename(path)}')
        print(json.dumps(s, indent=1))
        if base:
            common = [r for r in recs if r['id'] in base]
            if common:
                diffs = []
                for r in common:
                    b = base[r['id']]
                    diffs.append(token_f1(r['pred'], r['gt']) - token_f1(b['pred'], b['gt']))
                n = len(diffs)
                mean = 100 * sum(diffs) / n
                up = sum(1 for d in diffs if d > 0)
                down = sum(1 for d in diffs if d < 0)
                print(f'paired vs base: n={n} mean_deltaF1={mean:+.2f} '
                      f'(up {up} / down {down} / equal {n-up-down})')
                try:
                    from scipy.stats import wilcoxon
                    if up + down > 0:
                        print('wilcoxon p =', round(wilcoxon(diffs, zero_method='pratt').pvalue, 5))
                except Exception:
                    pass
                bs = summarize([base[r['id']] for r in common], args.eps)
                print(f'base on same ids: F1={bs["token_f1"]} refusal={bs["refusal_rate"]} '
                      f'pred_words={bs["mean_pred_words"]}')


if __name__ == '__main__':
    main()
