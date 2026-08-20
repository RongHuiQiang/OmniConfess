# -*- coding: utf-8 -*-
"""window_merge_gate.py — avhbench 聚合路线（coverage 而非 selection）判生死门.

Zero GPU. Consumes the 130-sample window-candidate pool already on disk
(outputs/avhbench_bon_window/avh_probe_shard*of4.jsonl).

Three merge simulations per sample (all scored with the standard Token-F1
footing, stopwords stripped):

  UNION  (pure upper bound) : content-word UNION of the 4 candidates
  (a) concat-dedup          : candidates joined in window order (full ->
                              front -> middle -> back), each word kept only
                              at its FIRST occurrence
  (b) incremental merge     : full-video greedy as the body; append only those
                              window-candidate clauses (sentence/comma spans,
                              >=2 content words) that contain >=1 modality
                              word (audio-source / color / count / visual)
                              AND >=1 content word absent from the body
  (b-cap)                   : (b) with a length cap at the gt median content-
                              word count (the online merger's precision guard)

Judgement (pre-registered): any of (a)/(b)/(b-cap) > 21.0 -> build the online
merger; all < 20.5 -> aggregation route dead, avhbench boundary chapter final.
"""
import os, sys, json, re, string
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from probe_cf_avhbench import AUDIO_WORDS, VISUAL_WORDS  # noqa: E402
from probe_claim_avhbench import token_f1, COLORS  # noqa: E402  (COLORS via probe_claim? fallback below)

try:
    from probe_claim_avhbench import COLORS  # noqa: F811
except Exception:
    COLORS = set()

STOPWORDS = set([
    "a", "an", "the", "and", "or", "but", "if", "because", "as", "what",
    "which", "this", "that", "these", "those", "then", "just", "so", "than",
    "such", "both", "through", "about", "for", "is", "are", "was", "were",
    "of", "while", "during", "to", "in", "on", "at", "by", "with", "it", "there",
])
NUMW = set('two three four five six seven eight nine ten several many couple '
           'dozen multiple numerous dozens'.split())


def content_words(text):
    s = str(text).lower().translate(str.maketrans("", "", string.punctuation))
    return [w for w in s.split() if w not in STOPWORDS and w.strip()]


def is_modality_word(w):
    return w in AUDIO_WORDS or w in VISUAL_WORDS or w in COLORS or w in NUMW \
        or any(c.isdigit() for c in w)


def clauses(text):
    """Sentence / comma-clause spans with >=2 content words (text-level
    approximation of the token-level claim_spans)."""
    out = []
    for seg in re.split(r'(?<=[.!?])\s+|,', str(text)):
        cw = content_words(seg)
        if len(cw) >= 2:
            out.append((seg.strip(), cw))
    return out


def load_rows():
    rows = []
    for k in range(4):
        p = os.path.join(HERE, 'outputs', 'avhbench_bon_window',
                         f'avh_probe_shard{k}of4.jsonl')
        rows += [json.loads(l) for l in open(p)]
    return rows


def main():
    rows = load_rows()
    print(f'[gate] {len(rows)} samples')

    arms = {k: [] for k in ('greedy', 'union', 'concat_dedup',
                            'incremental', 'incremental_cap')}
    gt_len = sorted(len(content_words(r['gt'])) for r in rows)
    cap = int(np.median(gt_len))
    print(f'[gate] gt median content-word count: {cap}')

    for r in rows:
        cands = r['candidates']                 # full, front, middle, back
        greedy = cands[0]['pred']
        arms['greedy'].append(token_f1(greedy, r['gt']))

        seq = [c['pred'] for c in cands]
        cw_all = [content_words(p) for p in seq]

        # UNION (upper bound): set union of content words
        union = []
        seen = set()
        for cw in cw_all:
            for w in cw:
                if w not in seen:
                    seen.add(w)
                    union.append(w)
        arms['union'].append(token_f1(' '.join(union), r['gt']))

        # (a) concat-dedup: join FULL TEXTS, but drop a candidate word that
        # already appeared in an earlier candidate (order preserved)
        merged, seen = [], set()
        for p in seq:
            words = str(p).split()
            kept = []
            for w in words:
                wl = w.lower().strip(string.punctuation)
                if wl and wl not in STOPWORDS:
                    if wl in seen:
                        continue
                    seen.add(wl)
                kept.append(w)
            merged.append(' '.join(kept))
        arms['concat_dedup'].append(token_f1(' '.join(merged), r['gt']))

        # (b) incremental: greedy body + modality-matched novel clauses
        body_words = set(content_words(greedy))
        body_cw = len(body_words)
        added = []
        for c in cands[1:]:
            for seg, cw in clauses(c['pred']):
                novel = [w for w in cw if w not in body_words]
                if not novel:
                    continue
                if not any(is_modality_word(w) for w in cw):
                    continue
                added.append((seg, len(cw)))
                body_words.update(cw)
        inc = greedy + (' ' + ' '.join(s for s, _ in added) if added else '')
        arms['incremental'].append(token_f1(inc, r['gt']))
        # capped: stop adding once total content words reach the gt median
        capped, total = [], body_cw
        for seg, n in added:
            if total >= cap:
                break
            capped.append(seg)
            total += n
        inc_cap = greedy + (' ' + ' '.join(capped) if capped else '')
        arms['incremental_cap'].append(token_f1(inc_cap, r['gt']))

    print(f"{'arm':<16}{'F1':>8}   (vs greedy)")
    g = 100 * float(np.mean(arms['greedy']))
    for k, v in arms.items():
        m = 100 * float(np.mean(v))
        print(f'{k:<16}{m:>8.2f}   {m - g:+.2f}')

    approx = {k: 100 * float(np.mean(arms[k])) for k in
              ('concat_dedup', 'incremental', 'incremental_cap')}
    build = any(v > 21.0 for v in approx.values())
    dead = all(v < 20.5 for v in approx.values())
    verdict = 'BUILD' if build else ('DEAD' if dead else 'INCONCLUSIVE')
    print(f'[verdict] {verdict} (any>21.0 build / all<20.5 dead)')
    json.dump({'n': len(rows), 'gt_median_cw': cap,
               'arms': {k: round(100 * float(np.mean(v)), 2) for k, v in arms.items()},
               'verdict': verdict},
              open(os.path.join(HERE, 'outputs', 'avhbench_bon_window',
                                'merge_gate.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
