# -*- coding: utf-8 -*-
"""build_registry.py -- generate the *fact layer* of prompts/registry.json.

Design rules (read before changing):

1. **The registry never copies prompt text.** It records only
   (dataset, arm, version) -> length + md5 + source (module:symbol). The real
   bytes always come from the original source. Copying text invites drift, and
   drift means published numbers lose their provenance.
2. **The md5 is an assertion, not a record.** verify.py recomputes and compares
   on every run; a mismatch raises rather than passing silently.
3. **The verdict layer lives separately in registry_verdicts.json** and is
   human-maintained. The fact layer can be regenerated from code; the verdict
   layer cannot.

Usage (from the repository root):
    python prompts/build_registry.py > prompts/registry.json
"""
import hashlib
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_GUARD = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_GUARD)

sys.path.insert(0, os.path.join(_GUARD, 'methods'))
sys.path.insert(0, _GUARD)

# Load the baseline prompt module by file path rather than as ``shared.prompts``:
# the package __init__ eagerly imports the data and model loaders, which would
# drag numpy and torch into a check that only ever touches strings.
import importlib.util                                      # noqa: E402
_spec = importlib.util.spec_from_file_location(
    'shared_prompts', os.path.join(_GUARD, 'methods', 'shared', 'prompts.py'))
_bp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bp)
base_sp = _bp.get_system_prompt

import judge_prompts as JP                                # noqa: E402
import halo_prompts as HP                                 # noqa: E402
import avh_prompts as AP                                  # noqa: E402

DATASETS = ['cmm', 'phd', 'pubmedqa', 'haloquest', 'avhbench', 'ragtruth', 'halueval']


def _h(s):
    return hashlib.md5(s.encode()).hexdigest()


def _rec(text, source):
    return {'len': len(text), 'md5': _h(text), 'source': source}


def build():
    reg = {ds: {'baseline': {}, 'ours': {}} for ds in DATASETS}

    for ds in DATASETS:
        for v in ('v0', 'v1'):
            try:
                p = base_sp(ds, v)
            except Exception:
                continue
            if p:
                reg[ds]['baseline'][v] = _rec(
                    p, f'OmniConfess/methods/shared/prompts.py::get_system_prompt({ds},{v})')

    for ds in ('cmm', 'phd', 'pubmedqa'):
        for v in JP.JUDGE_PROMPTS.get(ds, {}):
            p = JP.get_judge_prompt(ds, v)
            if isinstance(p, str):
                reg[ds]['ours'][v] = _rec(
                    p, f'OmniConfess/judge_prompts.py::get_judge_prompt({ds},{v})')

    for v in sorted(HP.HALO_PROMPTS):
        reg['haloquest']['ours'][v] = _rec(
            HP.get_halo_prompt(v)[0], f'OmniConfess/halo_prompts.py::get_halo_prompt({v})')

    for v in sorted(getattr(AP, 'AVH_PROMPTS', {})):
        try:
            r = AP.get_avh_prompt(v)
            s = r[0] if isinstance(r, tuple) else r
        except Exception as e:
            reg['avhbench']['ours'][v] = {'error': repr(e)}
            continue
        reg['avhbench']['ours'][v] = _rec(
            s, f'OmniConfess/avh_prompts.py::get_avh_prompt({v})')

    reg['ragtruth']['ours']['hardcoded'] = {
        'len': 346, 'md5': 'b32d0499bfb4bb8b75b4e185c502b618',
        'source': 'OmniConfess/omniconfess_qwen.py:1607-1615 (hardcoded, no version branch)',
        'note': 'A variant of baseline v1: the refusal line is removed and the short-answer anchor `Answer:` is replaced by'
                ' an instruction to answer directly without wrapping, plus two elaboration instructions. Both changes push toward longer output.',
    }
    reg['halueval']['ours']['hardcoded_fallback'] = {
        'len': 223, 'md5': 'c1aa547f13ec4c98b14aec7ff7e97d33',
        'source': 'OmniConfess/omniconfess_qwen.py:1601-1605 (hardcoded fallback)',
        'note': 'Unused when halueval goes through halueval_prompts.py; this is the fallback when that module is absent.'
                ' It pushes the opposite way from the ragtruth prompt -- it asks for a 1-5 word short answer.',
    }

    for ds in DATASETS:
        for arm in ('baseline', 'ours'):
            by_md5 = {}
            for v, rec in reg[ds][arm].items():
                if 'md5' in rec:
                    by_md5.setdefault(rec['md5'], []).append(v)
            for md5, vs in by_md5.items():
                if len(vs) > 1:
                    for v in vs:
                        reg[ds][arm][v]['duplicate_of'] = sorted(vs)

    for ds in DATASETS:
        b = {rec['md5']: v for v, rec in reg[ds]['baseline'].items() if 'md5' in rec}
        for v, rec in reg[ds]['ours'].items():
            if rec.get('md5') in b:
                rec['byte_equal_to_baseline'] = b[rec['md5']]

    return reg


if __name__ == '__main__':
    print(json.dumps(build(), ensure_ascii=False, indent=1))
