# -*- coding: utf-8 -*-
"""halo_dev_split.py — PROMPT_PROGRAM fixed 1/4 dev split for HaloQuest prompt
iteration (v3+).

Protocol (coordinator brief, frozen for all prompt versions):
  * universe = the 365-item full set used by the four-layer pipeline
    (outputs/haloquest_base_model/haloquest_base_model_full_single_scored.jsonl)
  * dev = 1/4 -> 91 items, stratified by gt avoidance/affirmative
    (probe_claim_cf.EVADERE on gt; 251 neg / 114 pos), then proportionally by
    group_key within each gt stratum (largest-remainder apportionment, seed 42)
  * the same split is reused verbatim for v3/v4/v5 and the final full-set check

Writes:
    OmniHalluBench/haloquest/dev_quarter/dev_ids.json          (list of ids)
    OmniHalluBench/haloquest/dev_quarter/haloquest_dev_quarter.json (dataset slice)
"""
import json, os, random
from collections import Counter

import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'OmniConfess', 'methods'))
sys.path.insert(0, HERE)
from probe_claim_cf import EVADERE  # noqa: E402
from shared import load_dataset  # noqa: E402

FULL = os.path.join(HERE, 'outputs', 'haloquest_base_model',
                    'haloquest_base_model_full_single_scored.jsonl')
DATA = os.path.join(HERE, 'OmniHalluBench', 'haloquest', 'haloquest.json')
OUTD = os.path.join(HERE, 'OmniHalluBench', 'haloquest', 'dev_quarter')


def main():
    rows = [json.loads(l) for l in open(FULL)]
    data = load_dataset(DATA, os.path.join(HERE, 'OmniHalluBench'))
    by_id = {it['id']: it for it in data}
    raw_by_id = {it['id']: it['_raw'] for it in data}
    assert len(rows) == 365 and all(r['id'] in by_id for r in rows)

    # strata: (gt_neg, group_key)
    strata = {}
    for r in rows:
        k = (bool(EVADERE.search(r['gt'] or '')), r.get('group_key') or 'NA')
        strata.setdefault(k, []).append(r['id'])

    n_dev = round(len(rows) / 4)          # 91
    rng = random.Random(42)
    # proportional allocation with largest remainder
    quot = {k: n_dev * len(v) / len(rows) for k, v in strata.items()}
    alloc = {k: int(q) for k, q in quot.items()}
    rem = n_dev - sum(alloc.values())
    for k in sorted(strata, key=lambda k: -(quot[k] - alloc[k]))[:rem]:
        alloc[k] += 1

    dev_ids = []
    for k in sorted(strata):
        g = sorted(strata[k])
        rng.shuffle(g)
        dev_ids.extend(g[:alloc[k]])
    assert len(dev_ids) == n_dev == len(set(dev_ids))

    os.makedirs(OUTD, exist_ok=True)
    json.dump(dev_ids, open(os.path.join(OUTD, 'dev_ids.json'), 'w'), indent=0)
    slice_ = [raw_by_id[i] for i in dev_ids]
    json.dump(slice_, open(os.path.join(OUTD, 'haloquest_dev_quarter.json'), 'w'),
              ensure_ascii=False, indent=1)

    neg = sum(1 for i in dev_ids if EVADERE.search(
        next(r['gt'] for r in rows if r['id'] == i) or ''))
    print(f'[dev] n={len(dev_ids)} gt_neg={neg} gt_pos={len(dev_ids)-neg}')
    print('[dev] group_key:', dict(sorted(Counter(
        by_id[i].get('group_key') or 'NA' for i in dev_ids).items())))
    full_gc = Counter(r.get('group_key') or 'NA' for r in rows)
    print('[full] group_key:', dict(sorted(full_gc.items())))
    # branch composition on dev (routing frozen from the published four-layer run)
    fc = [json.loads(l) for l in open(os.path.join(
        HERE, 'outputs', 'haloquest_bon_stacked',
        'haloquest_bon_stacked_feat_combo.jsonl'))]
    br = Counter(r['branch'] for r in fc if r['id'] in set(dev_ids))
    print('[dev] four-layer branches:', dict(br))


if __name__ == '__main__':
    main()
