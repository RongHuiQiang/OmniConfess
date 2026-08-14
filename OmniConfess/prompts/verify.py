# -*- coding: utf-8 -*-
"""verify.py -- assert that the registry and the live code have not drifted.

Every published number is attached to specific prompt bytes. Reword one line of
a prompt and the historical readings silently lose their provenance -- nothing
raises. This script turns that into a loud failure.

Usage:
    python prompts/verify.py          # verify everything, exit 1 on drift
    python prompts/verify.py --list   # also print the verdict table

CI should run the first form.
"""
import argparse
import hashlib
import json
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_GUARD = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_GUARD)

sys.path.insert(0, _HERE)
from build_registry import build   # noqa: E402


def _load(name):
    with open(os.path.join(_HERE, name), encoding='utf-8') as f:
        return json.load(f)


def verify_facts():
    """Re-extract from code and compare entry by entry against registry.json."""
    live = build()
    saved = _load('registry.json')
    drift, missing, added = [], [], []

    for ds in saved:
        for arm in ('baseline', 'ours'):
            for v, rec in saved[ds][arm].items():
                cur = live.get(ds, {}).get(arm, {}).get(v)
                if cur is None:
                    missing.append(f'{ds}/{arm}/{v}')
                elif cur.get('md5') != rec.get('md5'):
                    drift.append(f'{ds}/{arm}/{v}: registered {rec.get("md5")[:8]} '
                                 f'actual {cur.get("md5", "??")[:8]} '
                                 f'(len  {rec.get("len")} → {cur.get("len")})')
    for ds in live:
        for arm in ('baseline', 'ours'):
            for v in live[ds][arm]:
                if v not in saved.get(ds, {}).get(arm, {}):
                    added.append(f'{ds}/{arm}/{v}')
    return drift, missing, added


def verify_hardcoded():
    """ragtruth / halueval are hardcoded in the trunk with no version mechanism; check them separately.

    Method: concatenate the string literals of the matching get_system_prompt branch and recompute the md5.
    The trunk is deliberately not imported (it would pull in torch); this is a source-level regex extraction.
    """
    src_path = os.path.join(_GUARD, 'omniconfess_qwen.py')
    if not os.path.exists(src_path):
        return ['base_hook_best_0317.py  not found, skipping hardcoded check']
    with open(src_path, encoding='utf-8') as f:
        src = f.read()

    saved = _load('registry.json')
    problems = []
    targets = {
        'ragtruth': ('ragtruth', 'hardcoded',
                     'You are given passages and a question. Follow these steps:'),
        'halueval': ('halueval', 'hardcoded_fallback',
                     'You will be presented with a question.'),
    }
    for label, (ds, key, head) in targets.items():
        rec = saved.get(ds, {}).get('ours', {}).get(key)
        if not rec:
            continue
        i = src.find(head)
        if i < 0:
            problems.append(f'{label}: : opening sentence not found in the trunk: 「{head[:40]}...」'
                            f' -- the prompt may have been rewritten')
            continue
        start = src.rfind('"', 0, i)
        chunk = src[start:start + 4000]
        parts = re.findall(r'"((?:[^"\\]|\\.)*)"', chunk)
        text, taken = '', 0
        for p in parts:
            text += p.encode().decode('unicode_escape')
            taken += 1
            if len(text) >= rec['len']:
                break
        got = hashlib.md5(text[:rec['len']].encode()).hexdigest()
        if got != rec['md5']:
            problems.append(f'{label}: registered {rec["md5"][:8]} actual {got[:8]} '
                            f'(concatenated  {taken}  literals, len  {len(text)})')
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--list', action='store_true', help='also print the verdict table')
    args = ap.parse_args()

    drift, missing, added = verify_facts()
    hard = verify_hardcoded()

    ok = True
    if drift:
        ok = False
        print('❌ Byte drift (registry disagrees with code; historical readings lost provenance):')
        for d in drift:
            print('   ', d)
    if missing:
        ok = False
        print('❌ Registered but absent from code (version deleted?):')
        for m in missing:
            print('   ', m)
    if added:
        print('⚠️  Present in code but unregistered (new version -- rerun build_registry.py):')
        for a in added:
            print('   ', a)
    if hard:
        ok = False
        print('❌ Hardcoded prompt check:')
        for p in hard:
            print('   ', p)

    if ok and not added:
        n = sum(len(v[a]) for v in _load('registry.json').values()
                for a in ('baseline', 'ours'))
        print(f'✅ {n} prompt versions all match.')

    if args.list:
        vd = _load('registry_verdicts.json')
        print('\n=== Verdict table (Qwen)===')
        for ds, d in vd.items():
            if ds.startswith('_'):
                continue
            print(f'  {ds:10s} shipping={d["shipping"]:18s} '
                  f'{d["verdict"]}  [{d["grade"]}]')

    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
