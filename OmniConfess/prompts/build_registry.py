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
base_up = _bp.get_user_prompt

import judge_prompts as JP                                # noqa: E402
import halo_prompts as HP                                 # noqa: E402
import avh_prompts as AP                                  # noqa: E402

DATASETS = ['cmm', 'phd', 'pubmedqa', 'haloquest', 'avhbench', 'ragtruth', 'halueval']

# Hand-written provenance for baseline versions that are *designed* rather than
# inherited. The fact layer above is mechanical (len + md5); these notes say
# what the version is for. Keyed (dataset, arm, version) -> note.
VERSION_NOTES = {
    ('ragtruth', 'baseline', 'v2'):
        'Output-contract repair (TRAJECTORY_ANALYSIS.md §2): R1 drop Answer: anchor + ban closing '
        'summary (62% of low-F1 = extraction truncation, raw rescoring +11.7 F1); R4 answer partially '
        'instead of refusing (25 wrong refusals, F1 9.9); user template drops "general knowledge" (R3).',
    ('ragtruth', 'baseline', 'v3'):
        'v2 + R2 completeness/numbered list/80-150 words (F1 rises 13.1→57.7 monotonically with length, '
        'r=0.586) + R5 "Based on the given passages," opener (236/435 gt share it, ≈9% of token mass).',
    ('avhbench', 'baseline', 'v4'):
        'First output-contract iteration for avhbench (mechanism-migration arm @Qwen3-Omni, 2026-08-16); '
        'all prior runs used the bare v1 analyst prompt (v2/v3 are legacy GUARD strings, never run). '
        'Qwen3 v1 profile (645 outputs, 19.03): 633/645 already single-sentence, zero extraction '
        'truncation; residual is word-choice precision (oracle best-prefix 24.26, blunt 20-content-word '
        'cap 19.04 = +0.0). v4 = mechanical contract: exactly one sentence, direct emission (no label/'
        'preamble/closing -- ragtruth v2 R1 analog), 12-25 word window, main subject + dominant sound, '
        'no background enumeration. User template byte-identical to v1 (avh user prompt = bare question).',
    ('cmm', 'baseline', 'v2'):
        'Direction-symmetric judgment rules (PROMPT_PROGRAM.md cmm round 1): v1 rule 3 '
        '("based ONLY on what is audible") is one-directional and wrong for the 329 SEE questions '
        '(audio-stream leak -> 40 residual FP, FPR 0.206, median margin_adj +0.48); on the 392 HEAR '
        'questions the visual stream props Yes (61 residual FN, 52 greedy-Yes flipped to No, d_cf_video '
        'median -1.4..-3.1). v2: SEE<-video / HEAR<-audio verb-matched rules, sight never creates sound '
        'and sound never creates sight, mirror commitment clause. User template byte-identical to v1.',
    ('cmm', 'baseline', 'v3'):
        'Evidence-identity calibration (round 2): v2 killed the cross-modal leak (SEE FP 8->0, '
        'HEAR TPR 0.537->0.756) but swapped the error mass -- 13 HEAR FP (ambient/related sound taken '
        'for the named one: fan blowing x3, motorcycle revving x2, sheep bleating x2, cat meowing x2; '
        '11/13 were No under v1) and 10 SEE FN (small/diffuse objects lost to "clearly visible": rain, '
        'bee, door; 6/10 were Yes under v1). v3: SEE leniency on identity (object itself, even '
        'small/brief/partial), HEAR strictness on identity (exact named action; related sound/background '
        'hum/same-source noise = No), "answer Yes firmly" dropped. User template still byte-identical to v1.',
    ('cmm', 'baseline', 'v4'):
        'Percept-provenance commitment on HEAR (round 3): v3 left HEAR at TPR 0.683 / FPR 0.179 '
        'vs SEE 0.912 / 0.102. Of the 17 dev HEAR items with model-lean-Yes forced to No, 9 saves '
        '(gt=no, delta from d_cf_video with d_cf_audio~0 -- expected-not-heard sounds like fan '
        'blowing) vs 8 breaks (gt=yes, genuine d_cf_audio -1.1..-2.6, margin_adj -0.18..-0.77 in '
        '6/8). v4 rule 2 adds the provenance split: faint/brief/distant but genuinely heard = Yes; '
        'expected-from-the-scene but not heard = No. All else byte-identical to v3.',
    ('cmm', 'baseline', 'v5'):
        'Symmetric commitment completion (round 4): v3 rule 1 (SEE) carries "Answer Yes as soon as the '
        'object is visible ... even if small" -> SEE TPR 0.912, while rule 2 (HEAR) carried only the '
        'necessary-condition bar "only if identifiable" -> HEAR TPR 0.683 with 8 dev breaks holding '
        'genuine audio evidence (d_cf_audio -1.1..-2.6). v5 gives rule 2 the mirror commitment '
        '(as soon as identifiable, even if faint/brief/partial). Identity gate and all else '
        'byte-identical to v3.',
    ('cmm', 'baseline', 'v6'):
        'Question-type conditional routing of the oracle hybrid (round 5, coordinator endgame; no dev '
        '-- dev was shown to favor recall-boosting wording that does not replicate). Offline hybrid '
        '(v3 rows on SEE + v1 rows on HEAR) = 77.66/0.626, paired vs v1 fix 22 / break 9, p=0.029, '
        'while uniform v3 75.72/0.586 and v5 74.96/0.566 both regressed. Single system prompt: SEE '
        'branch = v3 rule 1 verbatim (the piece that transferred: SEE subgroup 78.47->81.37, FPR '
        '0.206->0.108); HEAR branch = v1 rule 3 + rule 1 sentences verbatim; header/format/example '
        'byte-identical to v1. Pre-registered full-721 gate: F1 > 76.36 AND Youden >= 0.600 AND '
        'paired significance vs v1.',
    ('haloquest', 'baseline', 'v2'):
        'Output-contract repair (TRAJECTORY_ANALYSIS.md §7.3/§7.4 D1+D2): minimum complete-sentence '
        'answer (≥8 words, ≤5-word replies banned: 122/365 v1 answers were ≤5 words vs gt 18.6) + '
        'fixed avoidance template "There is no X visible in this image." for non-visible objects '
        '(direct answer on unanswerable items scored F1 18.0 vs 30.6 when avoided).',
    ('haloquest', 'baseline', 'v3'):
        'P3 lever (TRAJECTORY_ANALYSIS.md §8.2), PROMPT_PROGRAM round 1: sentence-pattern echo '
        '(rebuild the answer from the question\'s own object/location/medium words — pass-through cell '
        'misses gt question-tail nouns, not length) + counting complete sentences ("How many" must '
        'yield "There are N <question words>.", number as a word; 8/31 counting items were wrongly '
        'answered with the absence template) + anti-false-avoidance (absence template demoted from '
        'rule 2 to rule 5; 10/54 gt-affirmative pass-through items got F1≈0 absence answers). '
        'Supersedes the legacy unused "meticulous visual auditor" v3 user template.',
    ('haloquest', 'baseline', 'v4'):
        'Round 2 (precision/brevity): v3 failed its dev gate (dev -0.98pp, pass-through cell '
        '50.03->44.45); trajectory autopsy showed sampled rollouts padded with un-asked scenery '
        '(gt content-word mass ~6, every extra noun is precision loss) and useless "In the photo," '
        'openers. v4 rewrites the echo rule as subject+asked-attribute-only with an 8-12 word window '
        'and hard one-sentence cap; keeps v3 counting rule and anti-false-avoidance.',
    # pubmedqa v5/v6 notes restored 2026-08-16: the two entries had been dropped
    # from this table by an earlier edit (the saved registry.json still carried
    # them), so a rebuild silently stripped the provenance of the two full-set
    # arms living in outputs/pubmedqa_cf_confess_{v5full,v6full}/. Text below is
    # byte-identical to the saved registry.json entries.
    ('pubmedqa', 'baseline', 'v5'):
        'Retained v3 system (v4 rolled back) + user-template surgery only: \'based on the provided '
        'passage and your general knowledge\' -> \'based only on the provided passage\'. DEV 115: '
        'F1 97.00 / Idx 90.58 -- passes the pre-registered gate (Idx +7.94 / F1 -0.06), '
        'dev-selected. FULL 460 final validation: F1 94.58 / Idx 82.96 (r_yes 92.82 / r_no 75.00) '
        '= Idx +1.58 / F1 -0.68 vs CF-v1; dev-to-full shrinkage (dev no-class n=14, 4 iterations '
        'overfit). outputs/pubmedqa_cf_confess_v5full/.',
    ('pubmedqa', 'baseline', 'v6'):
        'ATTRIBUTION ARM (not dev-selected): v1 system + the v5 user template = the (system v1/v3) '
        'x (user v1/fixed) 2x2 completion. FULL 460: F1 95.27 / Idx 80.30 -- the user fix alone is '
        'a no-op (paired 1:1 discordants, p=1.0); v5\'s +1.58 Idx is an interaction effect (system '
        'rules alone -1.62). outputs/pubmedqa_cf_confess_v6full/.',
}

# Versions whose designed delta lives in the USER template while the system
# bytes alias an earlier version. For these the registry also pins the user
# template itself (len + md5 + literal), because that is where the version's
# real bytes are and the system-prompt md5 alone would say "no change".
# Restored 2026-08-16 together with the two notes above -- the same edit that
# stripped the notes had also dropped this sub-record, leaving v5/v6 pointing
# at aliased system bytes with nothing to distinguish them.
USER_TEMPLATE_VERSIONS = {
    ('pubmedqa', 'v5'):
        "drop 'and your general knowledge', 'based on' -> 'based only on'",
    ('pubmedqa', 'v6'):
        "drop 'and your general knowledge', 'based on' -> 'based only on'",
}


def _h(s):
    return hashlib.md5(s.encode()).hexdigest()


def _rec(text, source):
    return {'len': len(text), 'md5': _h(text), 'source': source}


def build():
    reg = {ds: {'baseline': {}, 'ours': {}} for ds in DATASETS}

    for ds in DATASETS:
        seen_md5 = set()
        for v in ('v0', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6'):
            try:
                p = base_sp(ds, v)
            except Exception:
                continue
            if not p:
                continue
            h = _h(p)
            designed = (ds, 'baseline', v) in VERSION_NOTES
            # Datasets without a real v2/v3 dispatch fall back to their v1
            # system prompt; registering that byte-identical text under a new
            # version tag would be a lie, so skip the duplicate -- UNLESS the
            # version is designed and documented: pubmedqa v5/v6 deliberately
            # reuse the v3/v1 system bytes because their delta lives in the
            # user template (see prompts.py get_user_prompt pubmedqa v5/v6 and
            # the full-set arms in outputs/pubmedqa_cf_confess_{v5,v6}full/).
            # A VERSION_NOTES entry is exactly the assertion "this version is
            # real", so a documented alias is registered and tagged
            # duplicate_of below, while accidental fallbacks stay skipped.
            if h in seen_md5 and not designed:
                continue
            seen_md5.add(h)
            rec = _rec(p, f'OmniConfess/methods/shared/prompts.py::get_system_prompt({ds},{v})')
            if designed:
                rec['note'] = VERSION_NOTES[(ds, 'baseline', v)]
            if (ds, v) in USER_TEMPLATE_VERSIONS:
                # Placeholder-substituted template: len/md5 computed on the
                # string with '{q}' / '{passage}' left literal, exactly as the
                # shipped run sees it after formatting. (The registry copy that
                # this restoration replaces carried len 116 / md5 f08d32aa..,
                # which matches no string derivable from the live code -- a
                # stale one-off. Registry rule 2 says the md5 is an assertion
                # recomputed from code, so the live values are recorded.)
                tpl = base_up('{q}', ds, v, passage='{passage}')
                rec['user_prompt'] = {
                    'len': len(tpl), 'md5': _h(tpl), 'template': tpl,
                    'diff_from_v1': USER_TEMPLATE_VERSIONS[(ds, v)],
                }
            reg[ds]['baseline'][v] = rec

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
