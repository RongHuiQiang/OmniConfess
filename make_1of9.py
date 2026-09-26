#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_1of9.py -- deterministic stratified 1/9 subsample for the degenerate
method family (tot/gd/pd) on the NEMOTRON grid.

Decision (user, 2026-08-24): tot/gd/pd drop from 1/3 to 1/9. The 1/9 files are
drawn FROM the existing {ds}_1of3.json (i.e. 1of9 is a subset of 1of3, so all
1/9 rows keep their 1of3 provenance), stratified by (label, facets values) so
group compositions are preserved, with a FIXED seed.

Method: per stratum, largest-remainder allocation of the target count over
stratum sizes; within a stratum, random.Random(SEED + stratum-index-stable
salt derived from the sorted stratum key) sample. Deterministic across runs
and machines (pure stdlib).

Usage: python make_1of9.py [--root .] [--seed 20260824]
Outputs: OmniHalluBench/{ds}/{ds}_1of9.json + a manifest printed to stdout.
"""
import json
import os
import random
import argparse
from collections import Counter, OrderedDict

TARGETS = {  # 1/3 of the 1of3 file, rounded (user-fixed expectation rows)
    'pubmedqa': 52, 'cmm': 80, 'phd': 101,
    'ragtruth': 48, 'haloquest': 41, 'avhbench': 72,
}


def stratum_key(item):
    parts = [str(item.get('label'))]
    facets = item.get('facets') or {}
    for k in sorted(facets):
        parts.append(f'{k}={facets[k]}')
    return '|'.join(parts)


def stratified_subset(items, target, seed):
    strata = OrderedDict()
    for it in items:
        strata.setdefault(stratum_key(it), []).append(it)
    sizes = {k: len(v) for k, v in strata.items()}
    total = sum(sizes.values())
    # largest-remainder allocation
    exact = {k: target * n / total for k, n in sizes.items()}
    alloc = {k: int(exact[k]) for k in sizes}
    rema = target - sum(alloc.values())
    for k in sorted(sizes, key=lambda k: (-(exact[k] - alloc[k]), k))[:rema]:
        alloc[k] += 1
    rng_out = []
    for k in sorted(strata):
        take = min(alloc[k], len(strata[k]))
        if take <= 0:
            continue
        r = random.Random(f'{seed}|{k}')   # stratum-key-salted, stable
        rng_out.extend(r.sample(strata[k], take))
    # preserve original dataset order
    wanted = {id(x) for x in rng_out}
    return [it for it in items if id(it) in wanted], alloc, sizes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument('--seed', default='20260824')
    args = ap.parse_args()
    manifest = []
    for ds, target in TARGETS.items():
        src = os.path.join(args.root, 'OmniHalluBench', ds, f'{ds}_1of3.json')
        items = json.load(open(src))
        sub, alloc, sizes = stratified_subset(items, target, args.seed)
        out = os.path.join(args.root, 'OmniHalluBench', ds, f'{ds}_1of9.json')
        with open(out, 'w') as f:
            json.dump(sub, f, ensure_ascii=False, indent=1)
        comp = {k: f'{alloc[k]}/{sizes[k]}' for k in sizes}
        manifest.append((ds, len(items), len(sub), target, comp, out))
        print(f'[1of9] {ds}: {len(sub)}/{len(items)} (target {target}) '
              f'seed={args.seed} strata={comp}')
        assert len(sub) == target, f'{ds}: got {len(sub)} != target {target}'
    print('\nmanifest rows (ds, n_1of3, n_1of9):',
          [(m[0], m[1], m[2]) for m in manifest])


if __name__ == '__main__':
    main()
