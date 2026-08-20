# -*- coding: utf-8 -*-
"""
avhbench_vote_sc_rebuild.py — offline re-voting over the extended Vote-SC pool.

Pre-registered extension (2026-08-16): add two *phrasing* arms (temperature=0.7
samples, avhbench_temp07_gen.py) to the five *input* arms (224/384/448/f16/q3).

Combos scored (adoption criterion: any combo Token-F1 > 21.56 + 0.5):
    7arm @>=2   224/384/448/f16/q3/t07a1/t07a2     (main)
    7arm @>=3
    6arm @>=2   f16 replaced by the two temp arms
    6arm @>=3
Context rows (not adoption candidates): 5arm @>=2 must reproduce 21.56 exactly;
5arm @>=3 is a within-old-system ablation; oracle-of-7 for headroom.

Skeleton backfill (pre-registered in EXPERIMENT_DESIGN.md avhbench section):
take the highest-norm_logp arm's sentence, drop its non-voted content words,
keep its function words (phrasing), append voted words missing from the
sentence in vote order, capitalise + final period. Token-F1 reported as-is.

Voting logic (STOP list, cwords, vote counting, first-seen order, gt-keyed arm
alignment) is carried over verbatim from the script that produced
avhbench_vote_sc_full.jsonl (21.56), so the 5arm@2 row is an exact regression
test of this file against the shipped artifact.

Usage:  python avhbench_vote_sc_rebuild.py
"""
import json, re, collections, statistics, os, sys, glob
from scipy import stats as st

sys.path.insert(0, '/home/tsq/rhq/omniconfess/OmniConfess')
import answerability_gate as G

token_f1 = G.token_f1
F = lambda a, b: token_f1(a, b) * 100

OUT = '/home/tsq/rhq/omniconfess/OmniConfess/outputs/avhbench_vote_sc'
PATHS = {
    '224':   '/home/tsq/rhq/omniconfess/OmniConfess/outputs/avhbench_base_model/avhbench_base_model_full_single_scored.jsonl',
    '384':   '/home/tsq/rhq/omniconfess/OmniConfess/outputs/avhbench_res384/avhbench_res384_full_single_scored.jsonl',
    '448':   '/home/tsq/rhq/omniconfess/OmniConfess/outputs/avhbench_res448/avhbench_res448_full_single_scored.jsonl',
    'f16':   '/home/tsq/rhq/omniconfess/OmniConfess/outputs/avhbench_frames16/avhbench_frames16_full_single_scored.jsonl',
    'q3':    '/ultrafast_share/tsq/qwen3omni_outputs/avhbench_qwen3omni/avhbench_qwen3omni_full_single_scored.jsonl',
    't07a1': OUT + '/temp07_arm1_full.jsonl',
    't07a2': OUT + '/temp07_arm2_full.jsonl',
}

# ── voting tokenization (verbatim from the 21.56 run) ────────────────────────
STOP = set('''a an the and or of to in on at as is are was were be been being it its this that these those with while by for from into near nearby
there here his her their our your my he she they we you i who which what when where how not no nor but so then than also very just only
one two single describe see hear seen heard video audio image picture clip scene shows show can cannot could would should may might'''.split())
PUNCT = "!\"#$%&'()*+,-./:;<=>?@[]^_`{|}~\\"


def normgt(s):
    return re.sub(r'\s+', ' ', str(s).strip().lower())


def clean(s):
    m = re.search(r'\\boxed\{(.*)\}', str(s), re.S)
    return (m.group(1) if m else str(s)).strip()


def cwords(s):
    s = clean(s).lower().translate(str.maketrans('', '', PUNCT))
    return [w for w in s.split() if w not in STOP and len(w) > 1]


def merge_temp07_arms():
    """Merge the 8 shard files of each temp arm into <arm>_full.jsonl."""
    for a in ('t07a1', 't07a2'):
        src = OUT + f'/temp07_arm{a[-1]}/avhbench_shard*.jsonl'
        dst = OUT + f'/temp07_arm{a[-1]}_full.jsonl'
        files = sorted(glob.glob(src))
        n = 0
        with open(dst, 'w', encoding='utf-8') as out:
            for p in files:
                for line in open(p, encoding='utf-8'):
                    if line.strip():
                        out.write(line)
                        n += 1
        print(f'[merge] {a}: {len(files)} shards -> {n} lines -> {dst}', flush=True)


def load_arms():
    arms = {}
    for k, p in PATHS.items():
        recs = [json.loads(l) for l in open(p, encoding='utf-8')]
        d = {}
        for r in recs:
            d.setdefault(normgt(r['gt']), r)      # first occurrence wins
        arms[k] = d
        print(f'[arm {k}] {len(recs)} lines -> {len(d)} gt-unique', flush=True)
    keys = sorted(set.intersection(*[set(a) for a in arms.values()]))
    print(f'[align] gt-key intersection: {len(keys)} samples', flush=True)
    return arms, keys


def backfill(skeleton_sent, voted):
    """Phrasing backfill: skeleton arm's function words + voted content words."""
    out, seen = [], set()
    for tok in skeleton_sent.split():
        w = tok.lower().translate(str.maketrans('', '', PUNCT))
        if not w:
            continue
        if w in STOP or len(w) <= 1:
            out.append(tok)                       # function word / phrasing
        elif w in voted:
            out.append(tok)
            seen.add(w)
        # else: non-voted content word of the skeleton arm -> dropped
    missing = [w for w in voted if w not in seen]
    if not any(t.lower().translate(str.maketrans('', '', PUNCT)) in voted
               for t in out):
        return ' '.join(voted) + '.'             # skeleton had no voted content
    s = ' '.join(out + missing)
    s = re.sub(r'\s+([,.!?;:])', r'\1', s)
    s = re.sub(r'\s+', ' ', s).strip()
    if s and s[0].islower():
        s = s[0].upper() + s[1:]
    if s and s[-1] not in '.!?':
        s += '.'
    return s


def build(arms, keys, armlist, thr, out_path, method, backfill_path=None):
    recs, f1s, bf_f1s = [], [], []
    for gi, g in enumerate(keys):
        base = arms['224'][g]
        arm_pred, arm_f1, arm_nlp = {}, {}, {}
        for a in armlist:
            r = arms[a][g]
            arm_pred[a] = clean(r['pred'])
            arm_f1[a] = round(F(r['pred'], r['gt']), 2)
            arm_nlp[a] = round(r.get('norm_logp') or 0.0, 4)
        cnt = collections.Counter()
        contrib = collections.defaultdict(list)
        for a in armlist:
            for w in set(cwords(arm_pred[a])):
                cnt[w] += 1
                contrib[w].append(a)
        voted = [w for w, c in cnt.items() if c >= thr]
        order = []
        for a in armlist:
            for w in cwords(arm_pred[a]):
                if w in voted and w not in order:
                    order.append(w)
        pred = ' '.join(order)
        f1 = F(pred, base['gt'])
        f1s.append(f1)
        skeleton = max(armlist, key=lambda a: arm_nlp[a])
        rec = dict(
            idx=gi, id=base['id'], dataset='avhbench', method=method,
            question=base['question'], gt=base['gt'], pred=pred,
            vote_threshold=thr, n_arms=len(armlist), arms=armlist,
            votes={w: cnt[w] for w in voted},
            word_contributors={w: contrib[w] for w in voted},
            dropped_words_below_thr={w: cnt[w] for w, c in cnt.items() if c < thr},
            skeleton_source_arm=skeleton,
            arm_preds=arm_pred, arm_f1=arm_f1, arm_norm_logp=arm_nlp,
            local_f1=round(f1, 2),
        )
        recs.append(rec)
        if backfill_path is not None:
            bf = backfill(arm_pred[skeleton], order)
            rec['pred_backfill'] = bf
            rec['backfill_source_arm'] = skeleton
            bf_f1 = F(bf, base['gt'])
            rec['local_f1_backfill'] = round(bf_f1, 2)
            bf_f1s.append(bf_f1)
    with open(out_path, 'w', encoding='utf-8') as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    base_f1s = [r['arm_f1']['224'] for r in recs]
    w = st.wilcoxon(f1s, base_f1s)
    print(f"{method:>28} thr>={thr} -> {out_path.split('/')[-1]}")
    print("   n=%d  F1=%.2f  (base %.2f, delta %+.2f, p=%.2e, win/lose %d/%d)"
          % (len(recs), statistics.mean(f1s), statistics.mean(base_f1s),
             statistics.mean(f1s) - statistics.mean(base_f1s), w.pvalue,
             sum(1 for a, b in zip(f1s, base_f1s) if a > b),
             sum(1 for a, b in zip(f1s, base_f1s) if a < b)), flush=True)
    if backfill_path is not None:
        with open(backfill_path, 'w', encoding='utf-8') as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        wb = st.wilcoxon(bf_f1s, f1s)
        print("   backfill F1=%.2f (delta %+.2f vs plain vote, p=%.2e, win/lose %d/%d)"
              % (statistics.mean(bf_f1s),
                 statistics.mean(bf_f1s) - statistics.mean(f1s), wb.pvalue,
                 sum(1 for a, b in zip(bf_f1s, f1s) if a > b),
                 sum(1 for a, b in zip(bf_f1s, f1s) if a < b)), flush=True)
        return statistics.mean(f1s), statistics.mean(bf_f1s)
    return statistics.mean(f1s), None


def main():
    merge_temp07_arms()
    arms, keys = load_arms()
    A5 = ['224', '384', '448', 'f16', 'q3']
    T = ['t07a1', 't07a2']
    A7 = A5 + T
    A6 = ['224', '384', '448', 'q3'] + T          # f16 replaced by temp arms

    print('\n=== regression: original 5-arm @>=2 (must equal 21.56) ===')
    m5, _ = build(arms, keys, A5, 2, OUT + '/avhbench_vote_sc_5arm_thr2_regress.jsonl', 'vote_sc_5arm')

    print('\n=== adoption candidates ===')
    m7b, b7b = build(arms, keys, A7, 2, OUT + '/avhbench_vote_sc_7arm_thr2.jsonl',
                     'vote_sc_7arm',
                     backfill_path=OUT + '/avhbench_vote_sc_7arm_thr2_backfill.jsonl')
    m7c, _ = build(arms, keys, A7, 3, OUT + '/avhbench_vote_sc_7arm_thr3.jsonl', 'vote_sc_7arm')
    m6b, _ = build(arms, keys, A6, 2, OUT + '/avhbench_vote_sc_6arm_nof16_thr2.jsonl', 'vote_sc_6arm_nof16')
    m6c, _ = build(arms, keys, A6, 3, OUT + '/avhbench_vote_sc_6arm_nof16_thr3.jsonl', 'vote_sc_6arm_nof16')

    print('\n=== context rows (not adoption candidates) ===')
    m5c, _ = build(arms, keys, A5, 3, OUT + '/avhbench_vote_sc_5arm_thr3.jsonl', 'vote_sc_5arm')
    m5b, _ = build(arms, keys, ['224', '384', '448', 'q3'], 2,
                   OUT + '/avhbench_vote_sc_4arm_nof16_thr2_regress.jsonl', 'vote_sc_4arm_noF16')

    # oracle-of-7 and per-arm means over the aligned keys
    print('\n=== per-arm F1 over aligned keys ===')
    for a in A7:
        vals = [F(arms[a][g]['pred'], arms[a][g]['gt']) for g in keys]
        print(f'  {a:>6}: mean {statistics.mean(vals):.2f}')
    orc = [max(F(arms[a][g]['pred'], arms[a][g]['gt']) for a in A7) for g in keys]
    orc5 = [max(F(arms[a][g]['pred'], arms[a][g]['gt']) for a in A5) for g in keys]
    print(f'  oracle-of-7 {statistics.mean(orc):.2f} | oracle-of-5 {statistics.mean(orc5):.2f}')

    print('\n=== summary (adoption gate: > 22.06; saturation: <= 21.56) ===')
    print(f'  5arm@2 (shipped)      {m5:.2f}   <- regression check vs 21.56')
    print(f'  4arm no-f16@2         {m5b:.2f}   <- regression check vs 21.48')
    print(f'  7arm@2                {m7b:.2f}')
    print(f'  7arm@3                {m7c:.2f}')
    print(f'  6arm no-f16@2         {m6b:.2f}')
    print(f'  6arm no-f16@3         {m6c:.2f}')
    print(f'  5arm@3                {m5c:.2f}')
    if b7b is not None:
        print(f'  7arm@2 + backfill     {b7b:.2f}')

    # paired new-vs-old on the adoption candidates
    for name, path in [('7arm@2', OUT + '/avhbench_vote_sc_7arm_thr2.jsonl'),
                       ('7arm@3', OUT + '/avhbench_vote_sc_7arm_thr3.jsonl'),
                       ('6arm@2', OUT + '/avhbench_vote_sc_6arm_nof16_thr2.jsonl')]:
        new = {r['idx']: r['local_f1'] for r in map(json.loads, open(path, encoding='utf-8'))}
        old = {r['idx']: r['local_f1'] for r in map(json.loads, open(
            OUT + '/avhbench_vote_sc_5arm_thr2_regress.jsonl', encoding='utf-8'))}
        ks = sorted(new)
        w = st.wilcoxon([new[k] for k in ks], [old[k] for k in ks])
        print(f'  paired {name} vs 5arm@2: delta '
              f'{statistics.mean([new[k] for k in ks]) - statistics.mean([old[k] for k in ks]):+.2f}, '
              f'p={w.pvalue:.2e}, win/lose '
              f'{sum(1 for k in ks if new[k] > old[k])}/{sum(1 for k in ks if new[k] < old[k])}')


if __name__ == '__main__':
    main()
