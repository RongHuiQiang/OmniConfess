# -*- coding: utf-8 -*-
"""
avhbench_vote_sc_q3arms.py — offline re-voting over the cross-generation
(Qwen3-native input-perturbation) Vote-SC pool extension.

Pre-registered extension (2026-08-16, EXPERIMENT_DESIGN.md avhbench §6): add two
Qwen3-Omni-native *input* arms to the five shipped arms (224/384/448/f16 on
Qwen2.5 + q3 bare on Qwen3):
    q3r384  Qwen3-Omni, OMNI_FRAME_SIZE=384, 8 frames   (outputs/avhbench_vote_sc/q3_res384/)
    q3f16   Qwen3-Omni, OMNI_FRAME_SIZE=224, 16 frames  (outputs/avhbench_vote_sc/q3_f16/)
Both v1 prompt, greedy, eos-token-id fix; 645 samples.

Combos scored (adoption criterion: any combo Token-F1 > 21.56 + 0.5 = 22.06):
    7arm @>=2/@>=3   224/384/448/f16/q3/q3r384/q3f16          (main)
    6arm @>=2/@>=3   5arm + q3r384
    6armR @>=2/@>=3  q3r384 replaces the weakest arm f16
Context rows (not adoption candidates): 5arm @>=2 must reproduce 21.56 exactly
(regression of this file against the shipped artifact); per-arm means; oracles;
arm-pair content-word Jaccard; and the §5-style word-flow audit of what the new
arms do to the ≥2 gate (rescued isolated words / new corroborated words /
injected isolated words, each with gt hit-rate).

Voting logic is imported verbatim from avhbench_vote_sc_rebuild.py (the file
that reproduced the shipped 21.56 record-for-record); only the arm table and
the combo list are new.

Usage:  python avhbench_vote_sc_q3arms.py
"""
import json, sys, statistics, collections
from scipy import stats as st

sys.path.insert(0, '/home/tsq/rhq/omniconfess/OmniConfess')
import avhbench_vote_sc_rebuild as R          # verbatim voting logic (21.56-reproduced)

F = R.F                                        # token_f1 * 100

OUT = '/home/tsq/rhq/omniconfess/OmniConfess/outputs/avhbench_vote_sc'
R.PATHS = {
    '224':   '/home/tsq/rhq/omniconfess/OmniConfess/outputs/avhbench_base_model/avhbench_base_model_full_single_scored.jsonl',
    '384':   '/home/tsq/rhq/omniconfess/OmniConfess/outputs/avhbench_res384/avhbench_res384_full_single_scored.jsonl',
    '448':   '/home/tsq/rhq/omniconfess/OmniConfess/outputs/avhbench_res448/avhbench_res448_full_single_scored.jsonl',
    'f16':   '/home/tsq/rhq/omniconfess/OmniConfess/outputs/avhbench_frames16/avhbench_frames16_full_single_scored.jsonl',
    'q3':    '/ultrafast_share/tsq/qwen3omni_outputs/avhbench_qwen3omni/avhbench_qwen3omni_full_single_scored.jsonl',
    'q3r384': OUT + '/q3_res384/avhbench_q3_res384_full_single_scored.jsonl',
    'q3f16':  OUT + '/q3_f16/avhbench_q3_f16_full_single_scored.jsonl',
}

A5 = ['224', '384', '448', 'f16', 'q3']
NEW = ['q3r384', 'q3f16']
A7 = A5 + NEW
A6 = A5 + ['q3r384']
A6R = ['224', '384', '448', 'q3', 'q3r384']     # q3r384 replaces weakest arm f16


def jac(a, b):
    A, B = set(a), set(b)
    return len(A & B) / len(A | B) if (A | B) else 1.0


def diversity_audit(arms, keys):
    """§5-style audit: Jaccard matrix, per-arm/oracle means, word-flow of the
    ≥2 gate when the two Qwen3 input arms join the five shipped arms."""
    print('\n=== per-arm F1 over aligned keys ===')
    for a in A7:
        vals = [F(arms[a][g]['pred'], arms[a][g]['gt']) for g in keys]
        print(f'  {a:>6}: mean {statistics.mean(vals):.2f}')
    for name, pool in [('oracle-of-5', A5), ('oracle-of-7', A7)]:
        orc = [max(F(arms[a][g]['pred'], arms[a][g]['gt']) for a in pool) for g in keys]
        print(f'  {name} {statistics.mean(orc):.2f}')

    print('\n=== arm-pair content-word Jaccard (mean over samples) ===')
    cw = {a: {g: set(R.cwords(R.clean(arms[a][g]['pred']))) for g in keys} for a in A7}
    pairs = [('q3r384', a) for a in A5] + [('q3f16', a) for a in A5] + \
            [('q3r384', 'q3f16')] + [('q3', 'q3r384'), ('q3', 'q3f16')] + \
            [('224', '384'), ('224', 'f16'), ('384', 'f16')]
    for x, y in pairs:
        js = [jac(cw[x][g], cw[y][g]) for g in keys]
        print(f'  {x:>6} vs {y:<6}: {statistics.mean(js):.3f}')

    # word-flow: how do the ≥2 counts move when the 2 new arms join?
    print('\n=== word-flow: 5arm counts -> 7arm counts (per content word) ===')
    flow = collections.Counter()
    hit = collections.defaultdict(lambda: [0, 0])   # bucket -> [hits, total]
    for g in keys:
        gt_w = set(R.cwords(R.clean(arms['224'][g]['gt'])))
        c5 = collections.Counter()
        c7 = collections.Counter()
        old_seen = set()
        for a in A5:
            ws = R.cwords(R.clean(arms[a][g]['pred']))
            old_seen |= set(ws)
            for w in set(ws):
                c5[w] += 1
        for a in A7:
            for w in set(R.cwords(R.clean(arms[a][g]['pred']))):
                c7[w] += 1
        for w in set(c5) | set(c7):
            b5, b7 = c5[w] >= 2, c7[w] >= 2
            in_old = w in old_seen
            if not b5 and b7:
                bucket = 'rescued_old_isolated' if in_old else 'new_corroborated'
            elif b5 and not b7:
                bucket = 'lost'                    # ≥2 in 5-arm, <2 in 7-arm (impossible unless cnt changes? cnt only grows)
            elif b5 and b7:
                bucket = 'stable_core'
            else:
                bucket = 'isolated'
            flow[bucket] += 1
            hit[bucket][1] += 1
            if w in gt_w:
                hit[bucket][0] += 1
    for b in ['stable_core', 'rescued_old_isolated', 'new_corroborated', 'isolated']:
        h, t = hit[b]
        print(f'  {b:>22}: {flow[b]:>6} words, gt hit {h / max(t, 1) * 100:.1f}%')
    return cw


def main():
    arms, keys = R.load_arms()

    print('\n=== regression: original 5-arm @>=2 (must equal 21.56) ===')
    m5, _ = R.build(arms, keys, A5, 2, OUT + '/avhbench_vote_sc_q3x_5arm_thr2_regress.jsonl',
                    'vote_sc_5arm')

    print('\n=== adoption candidates (gate: > 22.06) ===')
    res = {}
    res['7arm@2'],  _ = R.build(arms, keys, A7, 2, OUT + '/avhbench_vote_sc_q3x_7arm_thr2.jsonl', 'vote_sc_q3x_7arm')
    res['7arm@3'],  _ = R.build(arms, keys, A7, 3, OUT + '/avhbench_vote_sc_q3x_7arm_thr3.jsonl', 'vote_sc_q3x_7arm')
    res['6arm@2'],  _ = R.build(arms, keys, A6, 2, OUT + '/avhbench_vote_sc_q3x_6arm_thr2.jsonl', 'vote_sc_q3x_6arm')
    res['6arm@3'],  _ = R.build(arms, keys, A6, 3, OUT + '/avhbench_vote_sc_q3x_6arm_thr3.jsonl', 'vote_sc_q3x_6arm')
    res['6armR@2'], _ = R.build(arms, keys, A6R, 2, OUT + '/avhbench_vote_sc_q3x_6armR_thr2.jsonl', 'vote_sc_q3x_6armR')
    res['6armR@3'], _ = R.build(arms, keys, A6R, 3, OUT + '/avhbench_vote_sc_q3x_6armR_thr3.jsonl', 'vote_sc_q3x_6armR')

    diversity_audit(arms, keys)

    # paired new-vs-5arm@2 on every adoption candidate
    old = {r['idx']: r['local_f1'] for r in map(
        json.loads, open(OUT + '/avhbench_vote_sc_q3x_5arm_thr2_regress.jsonl', encoding='utf-8'))}
    print('\n=== paired vs 5arm@2 (adoption candidates) ===')
    for name in ['7arm@2', '7arm@3', '6arm@2', '6arm@3', '6armR@2', '6armR@3']:
        f = {'7arm': '7arm', '6arm': '6arm', '6armR': '6armR'}[name.split('@')[0]]
        t = name.split('@')[1]
        p = OUT + f'/avhbench_vote_sc_q3x_{f}_thr{t}.jsonl'
        new = {r['idx']: r['local_f1'] for r in map(json.loads, open(p, encoding='utf-8'))}
        ks = sorted(new)
        w = st.wilcoxon([new[k] for k in ks], [old[k] for k in ks])
        d = statistics.mean([new[k] for k in ks]) - statistics.mean([old[k] for k in ks])
        print(f'  {name:>8}: delta {d:+.2f}, p={w.pvalue:.2e}, win/lose '
              f'{sum(1 for k in ks if new[k] > old[k])}/{sum(1 for k in ks if new[k] < old[k])}')

    print('\n=== summary (adoption gate: > 22.06; saturation: <= 21.56) ===')
    print(f'  5arm@2 (shipped, regression) {m5:.2f}   <- must equal 21.56')
    for k in ['7arm@2', '7arm@3', '6arm@2', '6arm@3', '6armR@2', '6armR@3']:
        print(f'  {k:>8}                      {res[k]:.2f}')
    best = max(res, key=res.get)
    verdict = 'ADOPT' if res[best] > 22.06 else ('GRAY' if res[best] > 21.56 else 'SATURATED')
    print(f'  best: {best} {res[best]:.2f} -> {verdict}')


if __name__ == '__main__':
    main()
