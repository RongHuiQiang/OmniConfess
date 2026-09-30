# -*- coding: utf-8 -*-
"""three_stage_freeform.py -- the FREE-TEXT sibling of the three-stage spine.

The narrative is identical to three_stage.py (cmm / phd share it):

  Stage 1  Generate   the surrounding greedy generate() produces the base answer.
  Stage 2  Confess     bind the emitted answer to the evidence channel(s) by
                      teacher-forcing masked/counterfactual view(s) on the SAME
                      committed answer. Runs ONCE PER SAMPLE, UNCONDITIONALLY.
  Stage 3  Decide      map (base answer + confession signal) to a final answer.

The ONLY point where the three datasets diverge is Stage 3's OUTPUT TYPE:

    cmm / phd     : YES/NO classification -> Stage 3 forces a single TOKEN
                    (three_stage.ThreeStageConfessProcessor, a LogitsProcessor
                     that fires at the decision token).
    HaloQuest     : free-text generation  -> Stage 3 produces a SPAN OF TEXT
                    (this file: a per-sample driver, NOT a LogitsProcessor,
                     because free text has no single decision token to hook).

Because there is no decision token, the free-text spine is a plain per-sample
DRIVER: the base class runs Stage 2 (confess) then Stage 3 (decide) and
assembles the record; subclasses implement only the two dataset hooks.

Stage-2 confession for free text is delta_img_tok: for every answer token,
lp_full(token) - lp_noimg(token), i.e. how much the image raised that token's
log-prob. LOW delta = the token was driven by text priors, not the image =
the ungrounded / fabricated part of the answer. HIGH delta = grounded in the
image. This reuses the shared primitive three_stage.teacher_force_seq_lp
(the sequence-level analogue of the two-class teacher_force_lp), so the
free-text and binary spines share one masked-view forward path.

Hooks (subclass implements)
---------------------------
confess(base_pred, clean_full, clean_noimg, gen_ids, item) -> dict
    Stage 2. Returns the confession signal. The base-class DEFAULT already
    computes delta_img_tok (+ per-token text + lp_full) -- HaloQuest uses it
    verbatim; a subclass may override to add channels.

decide(base_pred, confess_signal, item) -> (final_pred, record)
    Stage 3. Maps the base answer + confession signal to the final free text
    and the record dict. This is the dataset's essence and the only required
    hook.
"""
import string

import numpy as np

from .spine import teacher_force_seq_lp


class FreeformThreeStageDriver:
    """Per-sample free-text three-stage driver (Stage-1 generate is external).

    Guarantees mirroring three_stage.ThreeStageConfessProcessor:
      * Stage 2 (confess) runs UNCONDITIONALLY for every sample.
      * Stage 3 (decide) owns the final text + record.
      * the confession signal is bound to the committed base answer regardless
        of whether Stage 3 ends up changing it.
    """

    def __init__(self, model, processor, tokenizer):
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer

    # ---- shared Stage-2 primitive ---------------------------------------
    def tokenize_pred(self, pred_text):
        """Tokenize a free-text answer into (token_texts, ids_tensor)."""
        enc = self.tokenizer(pred_text, add_special_tokens=False,
                             return_tensors='pt')
        ids = enc['input_ids'][0]
        toks = [self.tokenizer.decode([t]) for t in ids.tolist()]
        return toks, ids

    def delta_img_tok(self, clean_full, clean_noimg, gen_ids):
        """delta_img_tok[i] = lp_full[i] - lp_noimg[i] over the answer tokens.

        Positive = token grounded in the image; low/negative = token driven by
        text priors (ungrounded / possibly fabricated). Returns
        (delta_list, lp_full_list)."""
        lp_full = teacher_force_seq_lp(self.model, clean_full, gen_ids)
        lp_noimg = teacher_force_seq_lp(self.model, clean_noimg, gen_ids)
        delta = [f - n for f, n in zip(lp_full, lp_noimg)]
        return delta, lp_full

    # ---- Stage-2 hook (default = delta_img_tok confession) --------------
    def confess(self, base_pred, clean_full, clean_noimg, gen_ids, item):
        """Stage 2 (default). Always-on delta_img_tok confession."""
        toks = [self.tokenizer.decode([t]) for t in gen_ids.tolist()]
        delta, lp_full = self.delta_img_tok(clean_full, clean_noimg, gen_ids)
        return dict(tokens=toks, delta_img_tok=delta, lp_full=lp_full,
                    n_tok=len(toks))

    # ---- Stage-3 hook (subclass must implement) -------------------------
    def decide(self, base_pred, confess_signal, item):
        raise NotImplementedError

    # ---- orchestrator ---------------------------------------------------
    def run_sample(self, base_pred, clean_full, clean_noimg, gen_ids, item):
        """Stage 2 (confess, unconditional) -> Stage 3 (decide). Returns
        (final_pred, record). The record always carries the confession signal
        summary so the delta channel is bound to the answer even when Stage 3
        leaves the answer unchanged (the point of confession)."""
        confess_signal = self.confess(base_pred, clean_full, clean_noimg,
                                      gen_ids, item)
        final_pred, record = self.decide(base_pred, confess_signal, item)
        # bind the confession channel into every record (always-on)
        delta = confess_signal.get('delta_img_tok') or []
        record.setdefault('delta_img_tok', [round(float(x), 4) for x in delta])
        record.setdefault('delta_img_agg_mean',
                          round(float(np.mean(delta)), 4) if delta else 0.0)
        record.setdefault('delta_img_agg_min',
                          round(float(np.min(delta)), 4) if delta else 0.0)
        record.setdefault('channel_computed', 1)
        return final_pred, record


# --------------------------------------------------------------------------
# Shared free-text helpers (localization on the confession signal + accept
# guard). Kept here so any free-text subclass can reuse them.
# --------------------------------------------------------------------------

_PUNCT = str.maketrans('', '', string.punctuation)
_STOP = {
    'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'in', 'on', 'at', 'of', 'to', 'for', 'with', 'by', 'and', 'or', 'but',
    'this', 'that', 'these', 'those', 'it', 'its', 'as', 'so', 'there',
}


def lowest_delta_span(tokens, delta, max_words=3):
    """Locate the ungrounded span: the content word(s) with the LOWEST
    delta_img_tok (least driven by the image). Returns (span_text, span_delta,
    token_indices). Word-boundary aware (Qwen BPE: leading space = new word)."""
    if not tokens or not delta:
        return '', 0.0, []
    # group BPE tokens into words
    words = []            # list of (text, [tok_idx...], mean_delta)
    cur_idx, cur_txt = [], ''
    for i, t in enumerate(tokens):
        starts = (i == 0) or (t[:1] in (' ', '\n', '\t')) or \
                 (t[:1] and not t[0].isalnum())
        if starts and cur_idx:
            words.append(cur_txt)
            m = float(np.mean([delta[j] for j in cur_idx]))
            words[-1] = (cur_txt, cur_idx, m)
            cur_idx, cur_txt = [], ''
        cur_idx.append(i)
        cur_txt += t
    if cur_idx:
        m = float(np.mean([delta[j] for j in cur_idx]))
        words.append((cur_txt, cur_idx, m))
    # keep only content words (drop stopwords / pure punctuation)
    content = [(w, idx, m) for (w, idx, m) in words
               if w.strip().translate(_PUNCT).lower() not in _STOP
               and w.strip().translate(_PUNCT).strip()]
    if not content:
        return '', 0.0, []
    # anchor = single lowest-delta content word (most ungrounded)
    content_sorted = sorted(content, key=lambda x: x[2])
    anchor = content_sorted[0]
    anchor_delta = anchor[2]
    # extend to at most (max_words-1) other content words that are ALSO low
    # (within 0.5 nats of the anchor); keeps the span focused on the
    # ungrounded region rather than grabbing grounded neighbours.
    picked = [anchor]
    for c in content_sorted[1:]:
        if len(picked) >= max_words:
            break
        if c[2] <= anchor_delta + 0.5:
            picked.append(c)
    picked.sort(key=lambda x: x[1][0])   # restore reading order
    all_idx = sorted(j for (_, idx, _) in picked for j in idx)
    span_text = ''.join(tokens[j] for j in all_idx).strip()
    span_delta = float(np.mean([delta[j] for j in all_idx]))
    return span_text, round(span_delta, 4), all_idx


def word_overlap(a, b):
    sa = set(str(a).lower().translate(_PUNCT).split())
    sb = set(str(b).lower().translate(_PUNCT).split())
    if not sa:
        return 0.0
    return len(sa & sb) / len(sa)
