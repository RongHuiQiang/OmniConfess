# -*- coding: utf-8 -*-
"""three_stage.py -- the shared three-stage counterfactual-confession spine.

phd (cf_confess_phd.py) and cmm (cf_confess_cmm.py) used to carry byte-identical
copies of the decision-point detection, the Stage-2 masked-view teacher-force,
and the Stage-3 hard-force + record-assembly boilerplate, differing only in the
counterfactual channels they ablate and the Stage-3 decision math. That skeleton
lives here now as ONE LogitsProcessor base class; each dataset subclasses it and
implements only the two hooks that carry its per-dataset essence:

    confess(prefix_ids) -> dict         # Stage 2: the counterfactual forwards
    decide(sig)         -> (tok, dict)  # Stage 3: the decision + its record

The three stages (unchanged semantics):

  Stage 1  Generate   the surrounding greedy generate() loop (NOT in this class).
  Stage 2  Confess     at the decision token, teacher-force the counterfactual
                      view(s) on the same prefix. This runs ONCE PER SAMPLE, at
                      the single decision token, and is UNCONDITIONAL -- every
                      sample that reaches a decision token confesses, even when
                      the confession will not change the committed answer. That
                      is the point of confession: bind the emitted token to the
                      evidence channels regardless of whether the decision moves.
  Stage 3  Decide      the subclass maps the full-view signal + the confession
                      signal to a forced token; the base class hard-forces it
                      (scores[:] = -inf; scores[forced] = 0) and assembles the
                      record.

The decision-point test is identical to the historical inlined copies:
    len(gen) == 0  and greedy argmax in {YES, NO}     (bare immediate answer)
    or the generated tail ends with `\\boxed{` / `\\boxed{{`.
"""
import torch
from transformers import LogitsProcessor

# Qwen2.5-Omni decision-token ids (shared by phd and cmm).
YES_ID, NO_ID = 9454, 2753


class ThreeStageConfessProcessor(LogitsProcessor):
    """Base class: owns Stage-1 decision detection + Stage-3 force/record.

    Subclasses provide the per-dataset counterfactual views and math via two
    hooks. The base class guarantees:
      * intervention fires at most ONCE (self.used), at the decision token;
      * Stage 2 (confess) is called UNCONDITIONALLY at that token;
      * Stage 3 (decide) returns the forced token, which is then hard-forced;
      * the subclass-supplied record dict is stored verbatim on self.record.

    Hooks
    -----
    confess(prefix_ids) -> dict
        Stage 2. `prefix_ids` is the list of already-generated token ids (the
        greedy prefix up to but excluding the decision token). Returns a dict of
        counterfactual-channel readouts (subclass-defined keys). Called once,
        for every sample that reaches the decision token.

    decide(full_view, confess_signal) -> (forced_token_id, record_dict)
        Stage 3. `full_view` is a dict {am, ly_full, ln_full, margin_full}
        computed by the base class from the full-view logits; `confess_signal`
        is exactly what confess() returned. Returns the token id to force and
        the record dict to attach (the base class adds nothing to it -- the
        subclass owns key order so the JSONL stays byte-comparable).
    """

    def __init__(self, tokenizer, prompt_len):
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len
        self.used = False
        self.record = None

    # ---- Stage 2 / Stage 3 hooks (subclass must implement) ----------------
    def confess(self, prefix_ids):
        raise NotImplementedError

    def decide(self, full_view, confess_signal):
        raise NotImplementedError

    # ---- Stage 1 decision detection + Stage 3 force (shared) --------------
    def _at_decision(self, gen, am):
        if len(gen) == 0:
            return am in (YES_ID, NO_ID)            # bare immediate answer
        tail = self.tokenizer.decode(gen[-4:])
        return tail.endswith('\\boxed{') or tail.endswith('\\boxed{{')

    @torch.inference_mode()
    def __call__(self, input_ids, scores):
        if self.used:
            return scores
        gen = input_ids[0, self.prompt_len:].tolist()
        logits = scores[0].float()
        am = int(logits.argmax().item())

        if not self._at_decision(gen, am):
            return scores

        # Full-view decision-token signal.
        lp = torch.log_softmax(logits, dim=-1)
        ly_full, ln_full = lp[YES_ID].item(), lp[NO_ID].item()
        full_view = dict(am=am, ly_full=ly_full, ln_full=ln_full,
                         margin_full=ly_full - ln_full, dec_index=len(gen))

        # Stage 2 (Confess): UNCONDITIONAL for every decision-point sample.
        confess_signal = self.confess(gen)

        # Stage 3 (Decide): subclass maps signals -> forced token + record.
        forced, record = self.decide(full_view, confess_signal)
        scores[0][:] = float('-inf')
        scores[0][forced] = 0.0                     # hard-force the token

        self.used = True
        self.record = record
        return scores


@torch.inference_mode()
def teacher_force_lp(model, clean_view, prefix_ids, token_ids):
    """Teacher-force `clean_view` (a prepared thinker input dict) extended by
    `prefix_ids`, and return the last-position log-probs for each id in
    `token_ids` as a list, in order.

    This is the single copy of the masked/counterfactual-view forward that phd
    and cmm both used inline (same key set, same float()/log_softmax path)."""
    ids = torch.cat([clean_view['input_ids'][0],
                     torch.tensor(prefix_ids, dtype=torch.long,
                                  device=clean_view['input_ids'].device)],
                    dim=0).unsqueeze(0)
    kw = dict(input_ids=ids, attention_mask=torch.ones_like(ids))
    for k in ('pixel_values_images', 'pixel_values', 'image_grid_thw',
              'pixel_values_videos', 'video_grid_thw',
              'video_second_per_grid', 'input_features',
              'feature_attention_mask'):
        if k in clean_view:
            kw[k] = clean_view[k]
    logits = model.thinker(**kw).logits[0, -1].float()
    lp = torch.log_softmax(logits, dim=-1)
    return [lp[t].item() for t in token_ids]


@torch.inference_mode()
def teacher_force_seq_lp(model, clean_view, gen_ids):
    """Teacher-force the WHOLE answer sequence `gen_ids` appended to a prepared
    view, and return per-token log-probs (one float per generated token).

    This is the sequence-level analogue of `teacher_force_lp`: where that reads
    a single decision position (the YES/NO token), this reads every answer token
    under a given view. It is the shared Stage-2 primitive for the FREEFORM
    spine (three_stage_freeform.py): running it under the full-image view and
    the no-image view and subtracting gives delta_img_tok per answer token
    (lp_full - lp_noimg), the confession signal for free-text generation.

    The binary-classification spine is untouched -- this is purely additive and
    reuses the same media-key set / float()/log_softmax path as the two-class
    teacher force above."""
    dev = clean_view['input_ids'].device
    gen_ids = gen_ids.to(dev)
    input_len = clean_view['input_ids'].shape[1]
    ids = torch.cat([clean_view['input_ids'], gen_ids.unsqueeze(0)], dim=1)
    kw = dict(input_ids=ids, attention_mask=torch.ones_like(ids))
    for k in ('pixel_values_images', 'pixel_values', 'image_grid_thw',
              'pixel_values_videos', 'video_grid_thw',
              'video_second_per_grid', 'input_features',
              'feature_attention_mask'):
        if k in clean_view:
            kw[k] = clean_view[k]
    logits = model.thinker(**kw, use_cache=False).logits[0].float()
    lsm = torch.log_softmax(logits, dim=-1)
    pos = torch.arange(input_len - 1, ids.shape[1] - 1, device=lsm.device)
    return lsm[pos, gen_ids.to(lsm.device)].cpu().tolist()
