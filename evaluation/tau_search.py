#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""tau_search.py — τ grid search for OmniConfess Stage-3 thresholds.

Each dataset exposes exactly one continuous Stage-3 threshold.  For the three
judgement datasets the decision is a pure function of already-stored
log-probabilities, so the whole grid is replayed offline from each row's `cf`
dict — no model rerun, no GPU.  The three free-form datasets route over
regeneration / token-swap candidates whose alternative text is *not* persisted,
so their grids cannot be replayed offline; they are reported as rerun-required.

Thresholds
----------
    dataset     param            range            selected  metric
    phd         tau_p            [-4.0, 2.0]        -1.6     Yes-F1
    cmm         tau_m            [-4.0, 0.0]        -1.6     Yes-F1   (margin_conf=2.0)
    pubmedqa    gate_tau_maybe   [-1.0, 2.0]         0.5     Yes-F1
    ragtruth    sig_floor        [-2.0, 0.0]        -0.5     Token-F1 (rerun-required)
    haloquest   halo_overlap     [ 0.0, 0.5]         0.10    Token-F1 (rerun-required)
    avhbench    tau_delta        [ 0.0, 2.0]         0.5     Token-F1 (rerun-required)

Usage
-----
# Score every dataset from the built-in default full-run files
python evaluation/tau_search.py

# Score one dataset from a specific file
python evaluation/tau_search.py --dataset phd \
    --input phd_cf_confess/cf_lam1.0_full.jsonl

# Search a folder for confession outputs (all datasets)
python evaluation/tau_search.py --input_dir .

# Print the grid definition without scoring
python evaluation/tau_search.py --dry_run

Interpreter: plain `python3` (numpy + scikit-learn only). Do NOT source
scripts/env.sh — its tmpfs numpy is built for a different CPython.
"""
import argparse
import csv
import glob
import json
import os
import re
import string
from collections import Counter

from sklearn.metrics import f1_score, recall_score

# ---------------------------------------------------------------------------
# Canonical τ grid — one continuous threshold per dataset, 5 grid points.
# The selected value (used in all main-table results) is marked with *.
# ---------------------------------------------------------------------------
GRIDS = {
    "phd": {
        "param":    "tau_p",
        "values":   [-4.0, -2.4, -1.6, 0.0, 2.0],
        "selected": -1.6,
        "metric":   "Yes-F1",
        "kind":     "judgment",
    },
    "cmm": {
        "param":    "tau_m",
        "values":   [-4.0, -2.4, -1.6, -0.8, 0.0],
        "selected": -1.6,
        "metric":   "Yes-F1",
        "kind":     "judgment",
        "fixed":    {"margin_conf": 2.0},   # held at selected value during tau_m sweep
    },
    "pubmedqa": {
        "param":    "gate_tau_maybe",
        "values":   [-1.0, 0.0, 0.5, 1.0, 2.0],
        "selected": 0.5,
        "metric":   "Yes-F1",
        "kind":     "judgment",
    },
    "ragtruth": {
        "param":    "sig_floor",
        "values":   [-2.0, -1.0, -0.5, -0.2, 0.0],
        "selected": -0.5,
        "metric":   "Token-F1",
        "kind":     "freeform",
        "note": (
            "cf_stats stores per-word deltas but not the replacement tokens; "
            "only the single committed output is persisted. A different "
            "sig_floor changes which words are swapped and to what, so the "
            "grid requires re-running Stage-3 on a GPU."
        ),
    },
    "haloquest": {
        "param":    "halo_overlap",
        "values":   [0.0, 0.05, 0.10, 0.25, 0.50],
        "selected": 0.10,
        "metric":   "Token-F1",
        "kind":     "freeform",
        "note": (
            "Decision is bucket-routed directed regeneration; regen_candidate "
            "text exists only for rows that actually regenerated. A different "
            "overlap gate would regenerate other rows whose candidate text is "
            "not stored, so the grid requires a GPU rerun."
        ),
    },
    "avhbench": {
        "param":    "tau_delta",
        "values":   [0.0, 0.25, 0.5, 1.0, 2.0],
        "selected": 0.5,
        "metric":   "Token-F1",
        "kind":     "freeform",
        "note": (
            "Vote / shared outputs do not persist per-arm confession deltas, "
            "and directed-regeneration candidates are not stored. The grid "
            "requires re-running the cross-arm aggregation on a GPU."
        ),
    },
}

# Default full-run files (relative to repo root) confirmed to carry the
# replayable `cf` signals for the judgement datasets.
DEFAULT_INPUTS = {
    "phd":       "phd_cf_confess/cf_lam1.0_full.jsonl",       # 906 rows
    "cmm":       "cmm_cf_confess/cf_lam0.5_full.jsonl",       # 721 rows
    "pubmedqa":  "pubmedqa_cf_confess/cf_lam0.25_full.jsonl", # 468 rows
    "ragtruth":  "cf_confess_ragtruth_full/cf_confess_full.jsonl",
    "haloquest": "omniconfess_haloquest/haloquest_confess.jsonl",
    "avhbench":  "omniconfess_avhbench/avhbench_vote.jsonl",
}

# ---------------------------------------------------------------------------
# Shared constants / helpers
# ---------------------------------------------------------------------------
YES_ID, NO_ID, MAYBE_ID = 9454, 2753, 21390

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "because", "as", "what",
    "which", "this", "that", "these", "those", "then", "just", "so", "than",
    "such", "both", "through", "about", "for", "is", "are", "was", "were",
    "of", "while", "during", "to", "in", "on", "at", "by", "with", "it", "there"
}


def _tokens(s):
    s = str(s).lower().translate(str.maketrans('', '', string.punctuation))
    return [w for w in s.split() if w not in STOPWORDS and w.strip()]


def _token_f1(pred, gt):
    p_t, g_t = _tokens(pred), _tokens(gt)
    if not p_t or not g_t:
        return 0.0
    common = Counter(p_t) & Counter(g_t)
    hit = sum(common.values())
    pr = hit / len(p_t)
    rc = hit / len(g_t)
    return (2 * pr * rc) / (pr + rc) if (pr + rc) else 0.0


def _parse_yesno(text):
    text = str(text).lower().strip()
    m = re.search(r'\\boxed\{(.*?)\}', text)
    if m:
        ans = m.group(1).strip()
        if 'maybe' in ans: return MAYBE_ID
        return YES_ID if 'yes' in ans else NO_ID
    for tag, tok in (('final answer: maybe', MAYBE_ID),
                     ('final answer: yes', YES_ID),
                     ('final answer: no', NO_ID)):
        if tag in text: return tok
    first = text.split('.')[0].split(',')[0].split()[0] if text.split() else ''
    if first == 'maybe': return MAYBE_ID
    if first == 'yes':   return YES_ID
    if first == 'no':    return NO_ID
    tail = text[-30:]
    if 'maybe' in tail: return MAYBE_ID
    if 'yes'   in tail: return YES_ID
    if 'no'    in tail: return NO_ID
    return NO_ID


def _load_jsonl(path):
    rows = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                m = re.search(r'(\{.*\})', line)
                if m:
                    rows.append(json.loads(m.group(1)))
            except Exception:
                continue
    return rows


def _gt_yesno(row):
    gt_raw = str(row.get('gt', row.get('answer', row.get('label', ''))))
    gl = gt_raw.lower().strip()
    if gl in ('yes',) or gl.startswith(('yes.', 'yes,', 'yes ')): return YES_ID
    if gl in ('no',)  or gl.startswith(('no.', 'no,', 'no ')):   return NO_ID
    return _parse_yesno(gt_raw)


def _cf(row):
    """Return the stored Stage-3 signal dict (`cf`; legacy alias `record`)."""
    return row.get('cf') or row.get('record') or {}


# ---------------------------------------------------------------------------
# Stage-3 replayers — judgement datasets (mirror omniconfess/operators.py)
# Signals are read from row['cf']; a degenerate row (no decision token) falls
# back to parsing its committed `pred`, matching the runner's own behaviour.
# ---------------------------------------------------------------------------

def _phd_replay(row, tau_p, w_extrap=4.0):
    cf = _cf(row)
    ly_f, ln_f = cf.get('lp_yes_full'), cf.get('lp_no_full')
    ly_cf, ln_cf = cf.get('lp_yes_cf'), cf.get('lp_no_cf')
    if any(v is None for v in (ly_f, ln_f, ly_cf, ln_cf)):
        return _parse_yesno(row.get('pred', ''))
    margin_full = ly_f - ln_f
    margin_cf   = ly_cf - ln_cf
    margin_prime = margin_full + w_extrap * (margin_cf - margin_full)
    return YES_ID if margin_prime > tau_p else NO_ID


# CMM per-sample relevant modality set R (mirrors operators.relevant_modalities_cmm)
_CMM_AUDIO_RE = re.compile(
    r'\bhear\b|\baudio\b|\bsound\b|\bnoise\b|\bmusic\b|'
    r'\bsaid\b|\bsay\b|\bspeak\b|\bspoke\b|\btalk\b|'
    r'\bvoice\b|\bsinging\b|\bsong\b', re.I)
_CMM_VIDEO_RE = re.compile(
    r'\bsee\b|\bseen\b|\bvideo\b|\bwatch\b|\bvisual\b|'
    r'\blook\b|\bshow\b|\bshown\b|\bappear\b|\bscreen\b|'
    r'\bcolou?r\b|\bframe\b', re.I)
_CMM_GROUP_TO_R = {
    'overrely_audio_ignore_visual': {'video'},
    'overrely_visual_ignore_audio': {'audio'},
}


def _cmm_relevant_R(question, group_key):
    if question:
        a = bool(_CMM_AUDIO_RE.search(question))
        v = bool(_CMM_VIDEO_RE.search(question))
        if a and not v: return {'audio'}
        if v and not a: return {'video'}
        if a and v:     return {'audio', 'video'}
    if group_key in _CMM_GROUP_TO_R:
        return set(_CMM_GROUP_TO_R[group_key])
    return {'audio', 'video'}


def _cmm_replay(row, tau_m, margin_conf=2.0):
    cf = _cf(row)
    am = cf.get('greedy_token')
    if am is None:
        return _parse_yesno(row.get('pred', ''))
    if am != YES_ID:
        return am
    if cf.get('margin_full', float('-inf')) >= margin_conf:
        return YES_ID
    R = _cmm_relevant_R(row.get('question'), row.get('group_key'))
    da, dv = cf.get('d_cf_audio'), cf.get('d_cf_video')
    vals = ([da] if 'audio' in R and da is not None else []) + \
           ([dv] if 'video' in R and dv is not None else [])
    if not vals:
        vals = [x for x in (da, dv) if x is not None]
    if not vals:
        return NO_ID
    return YES_ID if min(vals) <= tau_m else NO_ID


def _pubmedqa_replay(row, tau_maybe):
    cf = _cf(row)
    am = cf.get('greedy_token')
    if am is None:
        return _parse_yesno(row.get('pred', ''))
    lm, lm_cf = cf.get('lp_maybe_full'), cf.get('lp_maybe_cf')
    ly, ln = cf.get('lp_yes_full'), cf.get('lp_no_full')
    if am == MAYBE_ID and None not in (lm, lm_cf, ly, ln) \
            and (lm - lm_cf) <= tau_maybe:
        return YES_ID if ly > ln else NO_ID
    return am


_REPLAYERS = {
    'phd':      _phd_replay,
    'cmm':      _cmm_replay,
    'pubmedqa': _pubmedqa_replay,
}

# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score_judgment(rows, replayer, tau_val, fixed_kwargs):
    y_true, y_pred = [], []
    for row in rows:
        gt = _gt_yesno(row)
        if gt == MAYBE_ID:
            continue
        pred = replayer(row, tau_val, **fixed_kwargs)
        if pred is None:
            continue
        pred_bin = (YES_ID if gt == NO_ID else NO_ID) if pred == MAYBE_ID else pred
        y_true.append(1 if gt == YES_ID else 0)
        y_pred.append(1 if pred_bin == YES_ID else 0)
    if not y_true:
        return None
    r_yes = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    r_no  = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    phd   = (2 * r_yes * r_no) / (r_yes + r_no) if (r_yes + r_no) else 0.0
    return {
        'n':         len(y_true),
        'Yes-F1':    round(f1_score(y_true, y_pred, zero_division=0) * 100, 2),
        'PhD-Index': round(phd * 100, 2),
    }


# ---------------------------------------------------------------------------
# Per-dataset search
# ---------------------------------------------------------------------------

def search_dataset(dataset, input_path):
    cfg      = GRIDS[dataset]
    kind     = cfg['kind']
    param    = cfg['param']
    metric   = cfg['metric']
    values   = cfg['values']
    selected = cfg['selected']
    fixed    = cfg.get('fixed', {})

    if kind == 'freeform':
        # τ selects over regeneration / swap candidates whose alternative text
        # is not persisted — the grid cannot be replayed offline.
        print(f'\n[{dataset}] param={param}  metric={metric}  '
              f'[rerun-required, no offline replay]')
        print(f'  {cfg["note"]}')
        return ('rerun', param, metric, selected, cfg['note'])

    rows = _load_jsonl(input_path)
    print(f'\n[{dataset}] {len(rows)} rows  param={param}  metric={metric}  '
          f'src={input_path}')
    replayer = _REPLAYERS[dataset]
    results = []
    for tau in values:
        m = _score_judgment(rows, replayer, tau, fixed)
        if m is None:
            continue
        results.append({'tau': tau, **m})
    return ('grid', param, metric, selected, results)


def print_table(all_results):
    print('\n' + '=' * 74)
    print(f'{"Dataset":<11}{"Param":<16}{"τ":>7}   {"Score":>8}  {"Metric":<9}{"n":>6}')
    print('-' * 74)
    for ds, payload in all_results.items():
        status = payload[0]
        if status == 'rerun':
            _, param, metric, selected, _note = payload
            print(f'{ds:<11}{param:<16}{"—":>7}   {"rerun":>8}  {metric:<9}{"—":>6}')
            continue
        _, param, metric, selected, rows = payload
        for i, row in enumerate(rows):
            sel = '*' if row['tau'] == selected else ' '
            score = row.get(metric, '—')
            print(f'{(ds if i==0 else ""):<11}{(param if i==0 else ""):<16}'
                  f'{row["tau"]:>7.2f}{sel}  {score:>8}  {metric:<9}'
                  f'{row.get("n",""):>6}')
    print('=' * 74)
    print('  * = selected value used in main-table results')
    print('  rerun = grid needs a GPU Stage-3 rerun (candidates not persisted)')


def save_csv(all_results, out_path):
    with open(out_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['dataset', 'param', 'tau', 'selected', 'metric',
                    'score', 'n', 'status'])
        for ds, payload in all_results.items():
            status = payload[0]
            if status == 'rerun':
                _, param, metric, selected, _note = payload
                w.writerow([ds, param, '', selected, metric, '', '', 'rerun'])
                continue
            _, param, metric, selected, rows = payload
            for row in rows:
                w.writerow([ds, param, row['tau'],
                            'yes' if row['tau'] == selected else 'no',
                            metric, row.get(metric, ''), row.get('n', ''), 'grid'])
    print(f'\n[ok] results -> {out_path}')


# ---------------------------------------------------------------------------
# Input resolution
# ---------------------------------------------------------------------------

_DS_KEYWORDS = {
    'phd':       ('phd',),
    'cmm':       ('cmm',),
    'pubmedqa':  ('pubmedqa',),
    'ragtruth':  ('ragtruth',),
    'haloquest': ('halo',),
    'avhbench':  ('avh',),
}


def _find_input(input_dir, dataset):
    # match dataset keyword anywhere in the path (dir names carry the dataset),
    # and prefer full-run files over shards / smoke / scored.
    hits = []
    for kw in _DS_KEYWORDS[dataset]:
        for f in glob.glob(os.path.join(input_dir, '**', '*.jsonl'),
                           recursive=True):
            low = f.lower()
            if kw in low and '_scored' not in low and 'smoke' not in low:
                hits.append(f)
    if not hits:
        return None
    hits.sort(key=lambda f: (0 if 'full' in f.lower() else 1, len(f)))
    return hits[0]


def _resolve_input(dataset, args, repo_root):
    if args.input and args.dataset == dataset:
        return args.input
    if args.input_dir:
        return _find_input(args.input_dir, dataset)
    default = DEFAULT_INPUTS.get(dataset)
    if default:
        cand = default if os.path.isabs(default) else os.path.join(repo_root, default)
        return cand if os.path.exists(cand) else None
    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', choices=list(GRIDS), default=None,
                    help='Single dataset to score (omit to score all).')
    ap.add_argument('--input', default=None,
                    help='JSONL confession output for --dataset.')
    ap.add_argument('--input_dir', default=None,
                    help='Folder to search for confession outputs.')
    ap.add_argument('--repo_root', default='.',
                    help='Repo root for resolving DEFAULT_INPUTS (default: .).')
    ap.add_argument('--out', default='tau_search_results.csv',
                    help='Output CSV path.')
    ap.add_argument('--dry_run', action='store_true',
                    help='Print the grid τ values without scoring.')
    args = ap.parse_args()

    if args.dry_run:
        print(f'\n{"Dataset":<11}{"Param":<16}{"τ values":<38}'
              f'{"Sel":>6}  {"Kind":<9}')
        print('-' * 80)
        for ds, cfg in GRIDS.items():
            vals = '  '.join(f'{v:.2f}' for v in cfg['values'])
            print(f'{ds:<11}{cfg["param"]:<16}{vals:<38}'
                  f'{cfg["selected"]:>6.2f}  {cfg["kind"]:<9}')
        return

    datasets = [args.dataset] if args.dataset else list(GRIDS.keys())
    all_results = {}
    for ds in datasets:
        path = _resolve_input(ds, args, args.repo_root)
        if GRIDS[ds]['kind'] == 'judgment' and (path is None or not os.path.exists(path)):
            print(f'[{ds}] no input file found (looked for '
                  f'{DEFAULT_INPUTS.get(ds)}), skipping.')
            continue
        all_results[ds] = search_dataset(ds, path)

    if all_results:
        print_table(all_results)
        save_csv(all_results, args.out)


if __name__ == '__main__':
    main()
