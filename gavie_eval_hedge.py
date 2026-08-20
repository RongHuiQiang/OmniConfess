# -*- coding: utf-8 -*-
"""gavie_eval_hedge.py — 检测层的评分口径转移实验 (coordinator-approved):

HYPOTHESIS: flag+hedge (weakening low-Δ_confidence claims with "appears to
be"-style verb substitution, the action that was Token-F1-neutral at −0.05pp)
is POSITIVE on GAVIE-Acc (fact accuracy) while staying Token-F1-neutral. If
so, the descriptive-class confess mechanism stands on the metric intrinsic
to hallucination control.

Arms (all post-hoc text edits over the ALREADY-COMPUTED claim scores; no
generation, no GPU):
    base        : the untouched base_model prediction
    hedge       : flagged claims (claim score < τ, primary non-evasive only)
                  rewritten with the hedge verb substitution
    rand-hedge  : control — the SAME NUMBER of claims flagged per sample,
                  chosen uniformly at random (seed 49), same rewrite

τ is pre-registered as the Token-F1 dev-grid winner for the hedge action:
    avhbench  : from avh_claim_probe_report_sim.json dev grid (already run)
    haloquest : computed here on a 50/50 dev split (Token-F1, same protocol)

PRE-REGISTERED PASS CRITERIA (on the paired test/subset, glm-4 judge,
temperature 0.1, one call per (sample, arm)):
    1. hedge GAVIE-Acc > base,  paired Wilcoxon p < 0.05
    2. hedge GAVIE-Acc > rand-hedge, paired Wilcoxon p < 0.05
    3. hedge Token-F1 not significantly worse than base (paired p > 0.05
       or mean drop < 1pp)
Secondary (detection-layer deliverable): flag vs low-GAVIE-Acc alignment —
    precision = P(base-Acc in bottom tercile | sample flagged),
    recall    = P(flagged | base-Acc in bottom tercile),
    Spearman(mean claim score, base GAVIE-Acc).

Usage:
    source /home/tsq/rhq/omniconfess/.env_zhipu
    python gavie_eval_hedge.py --field avhbench        # 130 samples x 3 arms
    python gavie_eval_hedge.py --field haloquest --n_subset 150
"""
import os, sys, json, argparse, random, re, time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from probe_claim_avhbench import (  # noqa: E402
    sim_edit, token_f1, stratified_sample,
)
from probe_claim_cf import content_words  # noqa: E402

ZHIPU_KEY = os.environ.get('ZHIPU_API_KEY')
assert ZHIPU_KEY, 'source /home/tsq/rhq/omniconfess/.env_zhipu first'

GAVIE_PROMPT = (
    "Reference: {gt}\nModel Prediction: {pred}\n"
    "Score the prediction (0.0-10.0) on:\n"
    "1. relevancy: Main event coverage.\n"
    "2. accuracy: Detail correctness.\n"
    "Use decimal points for partial correctness (e.g. 7.4). "
    "Output strictly in this format: relevancy: X.X, accuracy: Y.Y"
)


def gavie_call(client, gt, pred, retries=3):
    import concurrent.futures as cf
    prompt = GAVIE_PROMPT.format(gt=gt, pred=pred)
    for i in range(retries):
        try:
            with cf.ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(lambda: client.chat.completions.create(
                    model='glm-4',
                    messages=[{'role': 'user', 'content': prompt}],
                    temperature=0.1))
                res = fut.result(timeout=30)
            c = res.choices[0].message.content.lower()
            rm = re.search(r'relevancy:\s*(\d+(?:\.\d+)?)', c)
            am = re.search(r'accuracy:\s*(\d+(?:\.\d+)?)', c)
            r = float(rm.group(1)) if rm else -1.0
            a = float(am.group(1)) if am else -1.0
            if r >= 0 and a >= 0:
                return min(r, 10.0), min(a, 10.0)
        except Exception:
            pass
        time.sleep(2 * (i + 1))
    return -1.0, -1.0


# ---------------------------------------------------------------------------
# avhbench field: reuse the claim probe artifacts.
# ---------------------------------------------------------------------------
def load_avhbench():
    import glob as _glob
    shards = sorted(_glob.glob(os.path.join(
        HERE, 'outputs', 'claim_probe', 'avh_claim_shard*of*.jsonl')))
    per_sample = []
    for f in shards:
        for line in open(f):
            if line.strip():
                r = json.loads(line)
                r['sample_i'] = len(per_sample)
                per_sample.append(r)
    claims = [json.loads(l) for l in open(os.path.join(
        HERE, 'outputs/claim_probe', 'avh_claim_probe_report_claims.jsonl'))]
    trows = [json.loads(l) for l in open(os.path.join(
        HERE, 'outputs/claim_probe', 'avh_claim_probe_report_tokens.jsonl'))]
    rows_by_sample = {}
    for r in trows:
        rows_by_sample.setdefault(r['sample_i'], []).append(r)
    for c in claims:
        c['attr_rows'] = [r for r in rows_by_sample.get(c['sample_i'], [])
                          if c['lo'] <= r['j'] < c['hi'] and r['ac']]
    # filter to primary non-evasive flaggable claims (gate-2 protocol:
    # sim_edit flags ANY claim with score<tau; claims jsonl has no is_evade —
    # avhbench captions rarely evade; keep all with score not None)
    return per_sample, claims


def tau_avhbench():
    """Pre-registered τ set for the hedge action from the gate-2 dev grid
    (all entries F1-neutral within 0.5pp of the hedge best): low / mid /
    high-coverage. The HIGH arm (max coverage subject to dev-F1 within
    0.5pp of best) is the primary arm; lo/mid are sensitivity."""
    sim = json.load(open(os.path.join(
        HERE, 'outputs/claim_probe', 'avh_claim_probe_report_sim.json')))
    hedge_entries = sorted([(eval(k)[1], float(v)) for k, v in
                            sim['dev_grid_top'] if eval(k)[0] == 'hedge'])
    best = max(v for _, v in hedge_entries)
    elig = [t for t, v in hedge_entries if v >= best - 0.005]
    if len(elig) >= 3:
        return {'lo': elig[0], 'mid': elig[len(elig) // 2], 'hi': elig[-1]}
    return {'lo': 0.398, 'mid': 1.044, 'hi': 2.352}


# ---------------------------------------------------------------------------
# haloquest field: reuse haloquest_claim_probe_full.json (claim_confess_sim
# scoring: attr words = color/number, matched contrast = swapped image).
# ---------------------------------------------------------------------------
def load_haloquest():
    d = json.load(open(os.path.join(
        HERE, 'outputs/claim_probe', 'haloquest_claim_probe_full.json')))
    sm = {m['sample_i']: m for m in d['samples']}
    recs = [json.loads(l) for l in open(os.path.join(
        HERE, 'outputs/haloquest_base_model',
        'haloquest_base_model_full_single_scored.jsonl'))]
    by_id = {r['id']: r for r in recs}

    from claim_confess_sim import is_attr
    rows = []
    for c in d['claims']:
        s = sm[c['sample_i']]
        av = [s['delta_tok'][j] for j in range(c['lo'], c['hi'])
              if s['classes'][j] in ('content', 'num')
              and is_attr(s['tokens'][j].strip().lower())]
        score = float(np.mean(av)) if av else c['delta_mean']
        rows.append(dict(c, score=score, has_attr=bool(av)))
    # per-sample assembly
    per_sample = []
    claims = []
    for m in d['samples']:
        r = by_id[m['id']]
        si = len(per_sample)
        per_sample.append({
            'sample_i': si, 'id': m['id'], 'gt': r['gt'], 'pred': r['pred'],
            'question': r['question'],
            'f1': r['eval_metrics']['Token-F1'],
            'tokens': m['tokens'], 'classes': m['classes'],
            'pred_words': len(r['pred'].split()),
        })
        for c in rows:
            if c['sample_i'] == m['sample_i']:
                c2 = dict(c)
                c2['sample_i'] = si
                c2['flaggable'] = (not c2.get('is_evade', False)
                                   and c2.get('in_primary', True))
                claims.append(c2)
    return per_sample, claims


def tau_haloquest(per_sample, claims, seed=42):
    """Pre-registered: Token-F1 dev grid for the hedge action (50/50
    F1-tercile-stratified dev split, same protocol as gate 2); arms =
    lo / mid / hi among grid entries within 0.005 of the hedge best."""
    rng = random.Random(seed)
    order = sorted(range(len(per_sample)), key=lambda i: per_sample[i]['f1'])
    k = max(1, len(order) // 3)
    tiers = [order[:k], order[k:2 * k], order[2 * k:]]
    dev = []
    for t in tiers:
        rng.shuffle(t)
        dev.extend(t[:round(len(t) * 0.5)])
    dev_scores = [c['score'] for c in claims
                  if c['sample_i'] in dev and c.get('score') is not None]
    grid_taus = list(np.quantile(dev_scores, [0.1, 0.2, 0.3, 0.4, 0.5,
                                              0.6, 0.7]))
    by_si = {}
    for c in claims:
        by_si.setdefault(c['sample_i'], []).append(c)
    entries = []
    for tau in grid_taus:
        f1s = [token_f1(edit_hedge(per_sample[i], by_si.get(i, []),
                                   float(tau))[0],
                        per_sample[i]['gt']) for i in dev]
        entries.append((float(tau), float(np.mean(f1s))))
    best = max(v for _, v in entries)
    elig = [t for t, v in entries if v >= best - 0.005]
    if len(elig) >= 3:
        return {'lo': elig[0], 'mid': elig[len(elig) // 2], 'hi': elig[-1]}
    e = sorted(entries, key=lambda x: -x[1])[:3]
    e = sorted(x[0] for x in e)
    return {'lo': e[0], 'mid': e[len(e) // 2], 'hi': e[-1]}


# ---------------------------------------------------------------------------
# Hedge editing (avhbench and haloquest share the verb-substitution rule).
# ---------------------------------------------------------------------------
import re as _re


def hedge_span_text(text):
    """Weaken ONE claim, always applying (pre-registered rule):
      1. copula/verb substitution when the pattern exists ("is" ->
         "appears to be", ...);
      2. otherwise prepend "Seemingly," (avhbench captions are participial
         clauses without copulas — the substitution alone is a near no-op
         there, which would make the GAVIE experiment null by construction).
    """
    low = ' ' + text.strip() + ' '
    for a, b in ((' is ', ' appears to be '), (' are ', ' appear to be '),
                 (' has ', ' appears to have '), (' have ', ' appear to have '),
                 (' wears ', ' appears to wear ')):
        if a in low:
            return low.replace(a, b, 1).strip()
    t = text.strip()
    if t[:1].isupper():
        t = t[0].lower() + t[1:]
    return 'seemingly, ' + t


def apply_hedges(tokens, flagged):
    """Rewrite flagged claims in a token list, return final text."""
    out = tokens[:]
    for c in flagged:
        span = ''.join(t for t in out[c['lo']:c['hi']] if t is not None)
        out[c['lo']] = ' ' + hedge_span_text(span) + ' '
        for j in range(c['lo'] + 1, c['hi']):
            out[j] = None
    text = ''.join(t for t in out if t is not None).strip()
    text = _re.sub(r'\s+', ' ', text).strip(' ,;')
    if text and text[-1] not in '.!?':
        text += '.'
    return text if text else 'The content is unclear.'


def edit_hedge(sample, claims, tau):
    """Flag claims with score < tau and hedge them. Returns (text, flagged)."""
    mine = [c for c in claims if c['sample_i'] == sample['sample_i']]
    flagged = [c for c in mine
               if c.get('flaggable', True) and c.get('score') is not None
               and c['score'] < tau]
    return apply_hedges(sample['tokens'], flagged), flagged


def edit_rand_hedge(sample, claims, tau, rng):
    """Control: same NUMBER of claims flagged as the main arm would flag,
    chosen uniformly at random among flaggable claims, same hedge rewrite."""
    mine = [c for c in claims if c['sample_i'] == sample['sample_i']]
    n_flag = sum(1 for c in mine
                 if c.get('flaggable', True) and c.get('score') is not None
                 and c['score'] < tau)
    pool = [c for c in mine if c.get('flaggable', True)
            and c.get('score') is not None]
    picked = rng.sample(pool, min(n_flag, len(pool)))
    return apply_hedges(sample['tokens'], picked)


# ---------------------------------------------------------------------------
# Field runner.
# ---------------------------------------------------------------------------
def run_field(args):
    if args.field == 'avhbench':
        per_sample, claims = load_avhbench()
        taus = tau_avhbench()
    else:
        per_sample, claims = load_haloquest()
        taus = tau_haloquest(per_sample, claims)
    print(f'[{args.field}] {len(per_sample)} samples, {len(claims)} claims, '
          f'taus={taus}', flush=True)

    if args.n_subset and args.n_subset < len(per_sample):
        for s in per_sample:
            s['_f1'] = s['f1']
        per_sample = stratified_sample(per_sample, args.n_subset, args.seed)
        keep = {s['sample_i'] for s in per_sample}
        claims = [c for c in claims if c['sample_i'] in keep]
        print(f'[{args.field}] subset -> {len(per_sample)} samples / '
              f'{len(claims)} claims', flush=True)

    rng = random.Random(49)
    by_si = {}
    for c in claims:
        by_si.setdefault(c['sample_i'], []).append(c)

    # arms: base, hedge_{lo,mid,hi} (hi = primary), rand_{lo,mid,hi}
    arms = {}
    for s in per_sample:
        mine = by_si.get(s['sample_i'], [])
        s = dict(s)
        s['tokens'] = s['tokens'][:]
        arms.setdefault('base', []).append((s, s['pred']))
        for ti, (tag, tau) in enumerate(taus.items()):
            hedge_text, _ = edit_hedge(s, mine, tau)
            arms.setdefault(f'hedge_{tag}', []).append((s, hedge_text))
            arms.setdefault(f'rand_{tag}', []).append(
                (s, edit_rand_hedge(s, mine, tau,
                                    random.Random(49 + 101 * ti))))

    # ---- GAVIE scoring (threaded) ----
    from zhipuai import ZhipuAI
    client = ZhipuAI(api_key=ZHIPU_KEY)
    jobs = []
    for arm, lst in arms.items():
        for si, (s, text) in enumerate(lst):
            jobs.append((arm, si, s['gt'], text))
    print(f'[{args.field}] {len(jobs)} GAVIE calls ...', flush=True)
    results = {}
    t0 = time.time()
    done = [0]

    def work(j):
        arm, si, gt, pred = j
        r, a = gavie_call(client, gt, pred)
        return arm, si, r, a

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for arm, si, r, a in ex.map(work, jobs):
            results[(arm, si)] = (r, a)
            done[0] += 1
            if done[0] % 200 == 0:
                print(f'  {done[0]}/{len(jobs)} {time.time()-t0:.0f}s',
                      flush=True)

    # ---- assemble per-sample records ----
    rows = []
    for si, (s, _) in enumerate(arms['base']):
        mine = by_si.get(s['sample_i'], [])
        scores = [c['score'] for c in mine
                  if c.get('flaggable', True) and c.get('score') is not None]
        mean_score = float(np.mean(scores)) if scores else None
        row = {
            'field': args.field, 'sample_i': s['sample_i'], 'id': s['id'],
            'gt': s['gt'], 'base_pred': s['pred'],
            'n_claims': len(mine),
            'mean_claim_score': round(mean_score, 4)
            if mean_score is not None else None,
            'f1_base': token_f1(s['pred'], s['gt']),
            'rel_base': results.get(('base', si), (-1, -1))[0],
            'acc_base': results.get(('base', si), (-1, -1))[1],
        }
        for tag, tau in taus.items():
            ht = arms[f'hedge_{tag}'][si][1]
            rt = arms[f'rand_{tag}'][si][1]
            row[f'hedge_pred_{tag}'] = ht
            row[f'rand_pred_{tag}'] = rt
            row[f'f1_hedge_{tag}'] = token_f1(ht, s['gt'])
            row[f'f1_rand_{tag}'] = token_f1(rt, s['gt'])
            row[f'rel_hedge_{tag}'] = results.get((f'hedge_{tag}', si), (-1, -1))[0]
            row[f'acc_hedge_{tag}'] = results.get((f'hedge_{tag}', si), (-1, -1))[1]
            row[f'rel_rand_{tag}'] = results.get((f'rand_{tag}', si), (-1, -1))[0]
            row[f'acc_rand_{tag}'] = results.get((f'rand_{tag}', si), (-1, -1))[1]
            row[f'n_flagged_{tag}'] = sum(
                1 for c in mine if c.get('flaggable', True)
                and c.get('score') is not None and c['score'] < tau)
        rows.append(row)

    # ---- statistics ----
    from scipy.stats import wilcoxon, spearmanr

    def arr(key):
        return np.array([r[key] for r in rows], dtype=float)

    def paired(a, b):
        x, y = arr(a), arr(b)
        m = (x >= 0) & (y >= 0)
        d = x[m] - y[m]
        if len(d) < 5:
            return None
        try:
            p = wilcoxon(x[m], y[m]).pvalue
        except Exception:
            p = float('nan')
        return {'n': int(m.sum()), 'mean_diff': round(float(d.mean()), 4),
                'p': round(float(p), 5)}

    stats = {'field': args.field, 'n_samples': len(rows), 'taus': taus}
    stats['gavie'] = {
        'acc_base': round(float(arr('acc_base')[arr('acc_base') >= 0].mean()), 3),
        'rel_base': round(float(arr('rel_base')[arr('rel_base') >= 0].mean()), 3),
    }
    for tag in taus:
        stats['gavie'][tag] = {
            'acc_hedge': round(float(arr(f'acc_hedge_{tag}')[arr(f'acc_hedge_{tag}') >= 0].mean()), 3),
            'acc_rand': round(float(arr(f'acc_rand_{tag}')[arr(f'acc_rand_{tag}') >= 0].mean()), 3),
            'rel_hedge': round(float(arr(f'rel_hedge_{tag}')[arr(f'rel_hedge_{tag}') >= 0].mean()), 3),
            'hedge_vs_base': paired(f'acc_hedge_{tag}', 'acc_base'),
            'hedge_vs_rand': paired(f'acc_hedge_{tag}', f'acc_rand_{tag}'),
            'rand_vs_base': paired(f'acc_rand_{tag}', 'acc_base'),
            'f1_hedge_vs_base': paired(f'f1_hedge_{tag}', 'f1_base'),
            'flag_rate': round(float(np.mean(
                [r[f'n_flagged_{tag}'] / max(1, r['n_claims'])
                 for r in rows])), 4),
            'samples_with_flag': int(sum(1 for r in rows if r[f'n_flagged_{tag}'] > 0)),
        }

    # detection-layer alignment: flag (primary hi arm) vs low base-Acc
    acc = arr('acc_base')
    m = acc >= 0
    thr = float(np.quantile(acc[m], 1 / 3))
    low = (acc <= thr) & m
    ms = [r['mean_claim_score'] for r in rows]
    det = {'low_acc_threshold_bottom_tercile': round(thr, 3),
           'base_rate_low': round(float(np.mean(low[m])), 4)}
    for tag in ('hi', 'mid', 'lo'):
        flagged = arr(f'n_flagged_{tag}') > 0
        if (flagged & m).any():
            det[f'precision_p_low_given_flagged_{tag}'] = round(
                float(np.mean(low[flagged & m])), 4)
        if low.any():
            det[f'recall_p_flagged_given_low_{tag}'] = round(
                float(np.mean(flagged[low])), 4)
    have = [(a, s) for a, s, ok in zip(acc, ms, m) if ok and s is not None]
    if len(have) > 10:
        rho, rp = spearmanr([x[0] for x in have], [x[1] for x in have])
        det['spearman_mean_claim_score_vs_acc'] = [round(float(rho), 4),
                                                   round(float(rp), 5)]
    stats['detection_layer'] = det

    # pre-registered primary-arm rule: the largest-coverage τ whose offline
    # Token-F1 mean drop stays < 1pp (F1 neutrality is part of the operator
    # spec; the GAVIE hypothesis itself is judged only by the paired tests).
    primary = None
    for tag in ('hi', 'mid', 'lo'):            # coverage descends with τ
        f1d = stats['gavie'][tag]['f1_hedge_vs_base']
        if f1d and f1d['mean_diff'] > -0.01:
            primary = tag
            break
    primary = primary or 'mid'
    g = stats['gavie'][primary]
    c1 = g['hedge_vs_base'] and g['hedge_vs_base']['mean_diff'] > 0 \
        and g['hedge_vs_base']['p'] < 0.05
    c2 = g['hedge_vs_rand'] and g['hedge_vs_rand']['mean_diff'] > 0 \
        and g['hedge_vs_rand']['p'] < 0.05
    f1d = g['f1_hedge_vs_base']
    c3 = f1d and (f1d['p'] > 0.05 or f1d['mean_diff'] > -0.01)
    stats['verdict'] = {
        'primary_arm': primary,
        'c1_hedge_gt_base': bool(c1), 'c2_hedge_gt_rand': bool(c2),
        'c3_f1_neutral': bool(c3),
        'overall': 'PASS' if (c1 and c2 and c3) else 'FAIL',
        'note': 'other arms are sensitivity, not gating',
    }

    out = os.path.join(HERE, 'outputs', 'claim_probe',
                       f'gavie_hedge_{args.field}.json')
    with open(out.replace('.json', '_rows.jsonl'), 'w') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    json.dump(stats, open(out, 'w'), ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f'[done] -> {out}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--field', required=True, choices=['avhbench', 'haloquest'])
    ap.add_argument('--n_subset', type=int, default=0)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    run_field(args)


if __name__ == '__main__':
    main()
