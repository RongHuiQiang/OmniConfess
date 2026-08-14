#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""tokenf1_fullset.py -- recompute Token-F1 over a full sample set.

Offline: no GAVIE API call and no GPU required. The Token-F1 column of the main
table can therefore be reproduced on a machine with no model weights and no API
key at all.

``_compute_token_f1`` and ``STOPWORDS`` are reused verbatim from
``OmniConfess/eval/single_eval.py`` so that the numbers stay on the same
footing as the ones reported in the paper.

Descriptive subsets may carry multiple references in ``gt`` separated by ';'.
Multi-reference scoring (max F1 over references) is available but **off by
default**, because single_eval treats the whole ``gt`` as one bag of words.

Usage:
    python scripts/tokenf1_fullset.py scripts/desc_fullset_manifest.json
"""
import json
import os
import string
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "GUARD", "eval"))

STOPWORDS = set([
    "a", "an", "the", "and", "or", "but", "if", "because", "as", "what",
    "which", "this", "that", "these", "those", "then", "just", "so", "than",
    "such", "both", "through", "about", "for", "is", "are", "was", "were",
    "of", "while", "during", "to", "in", "on", "at", "by", "with", "it", "there"
])


def _tokens(s):
    s = str(s).lower().translate(str.maketrans("", "", string.punctuation))
    return [w for w in s.split() if w not in STOPWORDS and w.strip()]


def token_f1(pred, gt):
    p_t, g_t = _tokens(pred), _tokens(gt)
    if not p_t or not g_t:
        return 0.0
    common = Counter(p_t) & Counter(g_t)
    hit = sum(common.values())
    p = hit / len(p_t)
    r = hit / len(g_t)
    return (2 * p * r) / (p + r) if (p + r) > 0 else 0.0


def token_f1_multiref(pred, gt, multiref=False):
    """Multi-reference scoring: max F1 across ';'-separated references.

        Off by default. ``_compute_token_f1`` in single_eval.py does not split on
        references -- it treats the entire ``gt`` as a single bag of words. The
        default here matches that behaviour so the numbers remain comparable to the
        ones already reported.
    """
    if multiref and isinstance(gt, str) and ";" in gt:
        return max(token_f1(pred, g.strip()) for g in gt.split(";") if g.strip())
    return token_f1(pred, gt)


def _get_value(item, keys):
    """Mirror the lookup order of ``_get_value`` in single_eval.py.
    """
    for k in keys:
        if k in item and item[k] is not None:
            return item[k]
    return ""


def score_file(path, multiref=False, drop_failed=True):
    """Reproduce the Token-F1 statistic of ``evaluate_descriptive_tasks``.

        drop_failed=True  matches single_eval: samples whose ``status`` is not
                          'success' are excluded from the mean. Note the asymmetry
                          -- baseline files usually carry no ``status`` field and so
                          are all counted as successes, whereas OmniConfess output
                          does carry it and failed samples are dropped.
        drop_failed=False counts every sample, scoring failures as F1=0. Use this
                          to quantify how much that asymmetry is worth.
    """
    scores, n_total, n_dropped = [], 0, 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            n_total += 1
            status = str(o.get("status", "success")).lower()
            gt = _get_value(o, ["gt", "ground_truth", "answer"])
            pred = _get_value(o, ["pred", "response", "output"])
            f1 = token_f1_multiref(pred, gt, multiref)
            if status != "success":
                n_dropped += 1
                if drop_failed:
                    continue
                f1 = 0.0
            scores.append(f1)
    if not scores:
        return n_total, 0, 0.0, n_dropped
    return n_total, len(scores), round(100 * sum(scores) / len(scores), 2), n_dropped


def main():
    manifest_path = sys.argv[1] if len(sys.argv) > 1 else "scripts/desc_fullset_manifest.json"
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    for ds, files in manifest.items():
        print(f"\n===== {ds} (Token-F1, full set, no GAVIE) =====")
        print(f"{'Method':<22}{'n_total':>9}{'n_scored':>10}{'F1(drop)':>10}"
              f"{'F1(all)':>9}{'failed':>8}")
        for method, path in files.items():
            if method.startswith("_"):   # manifest metadata, not an arm
                continue
            if not os.path.exists(path):
                print(f"{method:<22}{'MISSING':>9}")
                continue
            n, ns, f1_drop, nd = score_file(path, drop_failed=True)
            _, _, f1_all, _ = score_file(path, drop_failed=False)
            print(f"{method:<22}{n:>9}{ns:>10}{f1_drop:>10}{f1_all:>9}{nd:>8}")


if __name__ == "__main__":
    main()
