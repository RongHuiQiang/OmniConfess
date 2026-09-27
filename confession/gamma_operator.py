#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""gamma_operator.py -- lambda-free gamma operator for judgement datasets.

NEW FILE (feat/gamma-operator-v2). Shared core used by both the offline
recompute (gamma_offline_recompute.py) and the online GPU wrappers
(cf_confess_v2_{phd,cmm,pubmedqa}.py). It does NOT import or modify any
existing operator file.

Math (zero lambda). For a sample and candidate class d:
  F_d(X)            full-evidence discriminative margin of class d.
                    binary (phd/cmm): F = lp_yes - lp_no.
  F_d(X^{-m})       margin under the counterfactual view with modality m's
                    evidence removed.
  kappa^(m)         = F_d(X) - F_d(X^{-m})            (modality m contribution)
  R / I             per-sample relevant / irrelevant modality split, decided by
                    what the sample's question asks about.
  E_d^R             = sum_{m in R} kappa^(m)
  E_d^I             = sum_{m in I} kappa^(m)
  P_d^\\emptyset     = F_d(X) - sum_all kappa^(m)       (evidence-free prior)
  gamma             = E_d^R - F_d(X) = -P_d^\\emptyset - E_d^I
  margin'           = F_d(X) + gamma = E_d^R
  decision          E_d^R > 0 keep candidate class, else flip.

Per-dataset instantiation of the same principle "keep only relevant-modality
evidence, strip the prior P^0 and the irrelevant contribution E^I":

  cmm   both audio and video are ablatable; the relevant one(s) are chosen
        per sample. margin' = sum_{m in R} kappa^(m) directly (general form).

  phd   the only cached counterfactual removes the misleading CONTEXT (an
        irrelevant/adversarial channel; the image is the relevant modality and
        is never ablated). So E^I = kappa^(context) is the sole ablatable term
        and margin' = F_d(X) - kappa^(context) = F_d(X^{-context}), i.e. the
        image+question ("relevant view") margin ly_cf - ln_cf. This is the
        "strip the irrelevant channel" face of the principle (P^0 taken 0,
        no relevant channel is separately ablatable).

  pubmedqa  the only cached counterfactual removes the PASSAGE, which is the
        RELEVANT evidence channel; the passage-removed view is therefore the
        prior P^0. Per class d the passage contribution is
        kappa_d^(passage) = lp_d_full - lp_d_cf and the 3-class decision is
        argmax_d kappa_d^(passage) (strip the no-passage prior, keep the
        passage-grounded contribution). This is the multiclass generalisation
        of "margin' = E^R".
"""
import re

# Qwen2.5-Omni decision-token ids (identical to the v1 operators).
YES_ID, NO_ID, MAYBE_ID = 9454, 2753, 21390


# ----------------------------------------------------------------------------
# phd: single ablatable channel = misleading context (irrelevant).
# ----------------------------------------------------------------------------
def gamma_margin_phd(ly_full, ln_full, ly_cf, ln_cf):
    """E_d^R for phd = F_d(X^{-context}) = clean (image+question) margin.

    F_d(X)      = ly_full - ln_full
    kappa^(ctx) = F_d(X) - (ly_cf - ln_cf)         (E^I, context contribution)
    margin'     = F_d(X) - kappa^(ctx) = ly_cf - ln_cf
    Returns (margin_prime, forced_token).
    """
    margin_prime = ly_cf - ln_cf
    forced = YES_ID if margin_prime > 0 else NO_ID
    return margin_prime, forced


# ----------------------------------------------------------------------------
# cmm: two ablatable channels (audio, video); relevant set chosen per sample.
# ----------------------------------------------------------------------------
_CMM_AUDIO_RE = re.compile(r'\bhear\b|\baudio\b|\bsound\b|\bnoise\b|\bmusic\b|'
                           r'\bsaid\b|\bsay\b|\bspeak\b|\bspoke\b|\btalk\b|'
                           r'\bvoice\b|\bsinging\b|\bsong\b', re.I)
_CMM_VIDEO_RE = re.compile(r'\bsee\b|\bseen\b|\bvideo\b|\bwatch\b|\bvisual\b|'
                           r'\blook\b|\bshow\b|\bshown\b|\bappear\b|\bscreen\b|'
                           r'\bcolou?r\b|\bframe\b', re.I)

# CMM group_key encodes the FAILURE mode (which channel the model over-relies
# on and thus ignores); the ignored channel is the RELEVANT one the question
# probes. Verified against the shipped data: every "did you SEE ... video"
# question sits in overrely_audio_ignore_visual, every "did you HEAR ... audio"
# question sits in overrely_visual_ignore_audio (329/329 and 392/392).
_CMM_GROUP_TO_R = {
    'overrely_audio_ignore_visual': {'video'},
    'overrely_visual_ignore_audio': {'audio'},
}


def relevant_modalities_cmm(question=None, group_key=None):
    """Per-sample relevant modality set R for cmm, as {'audio'}, {'video'} or
    {'audio','video'}. Priority: (1) question keywords, (2) group_key meta
    field, (3) both-relevant fallback. Returns (R:set, source:str)."""
    if question:
        a = bool(_CMM_AUDIO_RE.search(question))
        v = bool(_CMM_VIDEO_RE.search(question))
        if a and not v:
            return {'audio'}, 'question'
        if v and not a:
            return {'video'}, 'question'
        if a and v:
            return {'audio', 'video'}, 'question_both'
    if group_key in _CMM_GROUP_TO_R:
        return set(_CMM_GROUP_TO_R[group_key]), 'group_key'
    return {'audio', 'video'}, 'fallback_both'


def gamma_margin_cmm(margin_full, margin_noaudio, margin_novideo, R):
    """E_d^R for cmm = sum_{m in R} kappa^(m).

    kappa^(audio) = margin_full - margin_noaudio
    kappa^(video) = margin_full - margin_novideo
    Returns (margin_prime, forced_token).
    """
    er = 0.0
    if 'audio' in R:
        er += margin_full - margin_noaudio
    if 'video' in R:
        er += margin_full - margin_novideo
    forced = YES_ID if er > 0 else NO_ID
    return er, forced


# ----------------------------------------------------------------------------
# pubmedqa: single ablatable channel = passage (the relevant evidence).
# ----------------------------------------------------------------------------
def gamma_scores_pubmedqa(ly_full, ln_full, lm_full, ly_cf, ln_cf, lm_cf):
    """Per-class passage contribution kappa_d = lp_d_full - lp_d_cf, then
    argmax over {yes,no,maybe}. Returns (scores:dict, forced_token).
    (Literal E^R rule; see module docstring.)"""
    scores = {
        YES_ID:   ly_full - ly_cf,
        NO_ID:    ln_full - ln_cf,
        MAYBE_ID: lm_full - lm_cf,
    }
    forced = max(scores, key=scores.get)
    return scores, forced


# ----------------------------------------------------------------------------
# Candidate-anchored + dead-zone variant (safe for 3-class / 2-view).
#
#   d*      = base model's full-view argmax (the already-committed answer).
#   E_{d*}^R = sum_{m in R} [ F_{d*}(X) - F_{d*}(X^{-m}) ]
#             = relevant-modality net support FOR the committed class d*.
#   flip only if E_{d*}^R < -tau (weak signal keeps base); tau=0 -> sign rule.
#   flip target: among the other classes, the one with the largest E_c^R;
#                if none is positive, the full-view runner-up class.
#
# er_by_class maps class_token -> E_c^R. full_lp maps class_token -> full-view
# logprob (for the runner-up tie-break). d_star is the base argmax token.
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Hybrid deployment router — Method A (accepted 2026-09-23).
#
# phd / pubmedqa : λ-free literal γ (E^R decision, τ=0).
#     phd     : margin' = F_d(X^{-context}) = ly_cf − ln_cf.  Strip irrelevant
#               context channel; relevant image channel is never ablated.
#     pubmedqa: argmax_d kappa_d^(passage). Strip no-passage prior; keep
#               passage-grounded class. (passage is the RELEVANT channel here.)
#     Both achieve E^R > 0 ↔ keep candidate, E^R ≤ 0 ↔ flip — zero λ.
#
# cmm         : retain the original λ-weighted δ operator (cf_confess_cmm.py
#               / cf_confess_phd.py architecture). The dual-view weighted
#               signal cannot be recovered by the sign-based γ rule; λ-free
#               variants max out at 0.714 vs λ's 0.813 (verified offline).
#               The two operators are instances of the same evidence-attribution
#               principle: γ strips E^I and P^∅; λ-δ retains the full margin
#               and adds a scaled counterfactual view. For single-view datasets
#               where P^∅ is the noise source, γ is strictly better.
#
# Usage:
#   dataset in ('phd', 'pubmedqa') → call the relevant gamma_* function above.
#   dataset == 'cmm'               → call the existing λ operator in
#                                    cf_confess_phd.py (CFConfessProcessor).
# ----------------------------------------------------------------------------
GAMMA_DATASETS  = frozenset({'phd', 'pubmedqa'})   # use λ-free γ
LAMBDA_DATASETS = frozenset({'cmm'})               # retain legacy λ-δ operator


def deploy_decision(dataset, **kwargs):
    """Route to the correct decision function for a given dataset.

    For GAMMA_DATASETS:
        phd     → gamma_margin_phd(ly_full, ln_full, ly_cf, ln_cf)
        pubmedqa→ gamma_scores_pubmedqa(ly_full,ln_full,lm_full,ly_cf,ln_cf,lm_cf)
    For LAMBDA_DATASETS (cmm):
        Returns None — caller must use the existing CFConfessProcessor with
        its lambda_cf parameter unchanged.

    Raises ValueError for unknown dataset names.
    """
    dataset = dataset.lower()
    if dataset == 'phd':
        return gamma_margin_phd(
            kwargs['ly_full'], kwargs['ln_full'],
            kwargs['ly_cf'],   kwargs['ln_cf'])
    if dataset == 'pubmedqa':
        return gamma_scores_pubmedqa(
            kwargs['ly_full'], kwargs['ln_full'], kwargs['lm_full'],
            kwargs['ly_cf'],   kwargs['ln_cf'],   kwargs['lm_cf'])
    if dataset == 'cmm':
        return None   # use legacy λ operator
    raise ValueError(f"Unknown dataset '{dataset}': expected one of "
                     f"{sorted(GAMMA_DATASETS | LAMBDA_DATASETS)}")


def candidate_anchored_decision(d_star, er_by_class, full_lp, tau):
    others = [c for c in er_by_class if c != d_star]
    er_star = er_by_class[d_star]
    if er_star >= -tau:
        return d_star                       # keep committed answer
    # flip: prefer the other class with the largest (positive) relevant support
    best_other = max(others, key=lambda c: er_by_class[c])
    if er_by_class[best_other] > 0:
        return best_other
    # no other class is positively supported -> full-view runner-up
    return max(others, key=lambda c: full_lp[c])


# ----------------------------------------------------------------------------
# Confidence-gated candidate anchoring (v3, 2026-09-27).
#
# Diagnosis at n=20: the deployed operators OVER-INTERVENE. The raw counterfactual
# signal (kappa) is near-zero noise on pubmedqa, and the λ-δ flip on cmm fires
# even when the base model is highly confident and correct. Both regress F1 below
# base. The fix keeps the three-stage machinery but ANCHORS to the base committed
# answer and only lets the counterfactual override when the base is unreliable:
#
#   * base high-confidence  (|full margin| >= tau_conf) -> keep base answer.
#   * base low-confidence                               -> trust the operator flip.
#
# For the 3-class pubmedqa the unreliable mode is specifically an over-predicted
# minority class 'maybe' that the passage does not ground; gate it by the passage
# contribution kappa_maybe and redecide yes/no when unsupported.
# tau_* are task-level hyperparameters tuned on the 20-sample dev slice.
# ----------------------------------------------------------------------------
def confidence_gated_flip(am, margin_full, flip_token, tau_conf):
    """Binary (phd/cmm) gate: anchor to base answer `am` when the base is
    confident (|margin_full| >= tau_conf); otherwise take the operator's
    `flip_token`. tau_conf=0 recovers the ungated operator."""
    return am if abs(margin_full) >= tau_conf else flip_token


def gamma_anchored_pubmedqa(am, ly, ln, lm, ly_cf, ln_cf, lm_cf, tau_maybe=0.5):
    """Candidate-anchored pubmedqa decision with a 'maybe' evidence gate.

    Keep the base committed class `am`, EXCEPT when am == MAYBE and the passage
    contribution to maybe, kappa_maybe = lm - lm_cf, is <= tau_maybe (the passage
    does not strongly ground 'maybe'): then redecide yes/no by full-view logprob.
    Returns the forced token id."""
    if am == MAYBE_ID and (lm - lm_cf) <= tau_maybe:
        return YES_ID if ly > ln else NO_ID
    return am


# ----------------------------------------------------------------------------
# n=200 F1(yes)-calibrated rules (v4, 2026-09-27).
#
# Re-derived on the 200-sample cache under the paper's judgement metric F1(yes)
# (Eq.36, positive-class harmonic mean) instead of macro-F1. The v3 confidence
# gate was tuned on 20 samples and under-delivered at scale (phd +4.98, and the
# pubmedqa maybe-gate collapsed +30.86 -> +0.60, an overfit artefact). The two
# rules below were selected by 5-fold / nested CV on n=200 and each clear +10
# F1(yes) with accuracy AND macro rising together (not yes-drift gaming):
#
#   phd  margin' = margin_full + w*(margin_cf - margin_full), w=2.0, yes iff >0.
#        margin_cf = ly_cf - ln_cf is the context-stripped ("relevant") margin.
#        w>1 EXTRAPOLATES beyond the clean view: phd's context is adversarial
#        misinformation, so amplifying how far removing it moves the decision is
#        the right correction. base 72.57 -> 82.95 (+10.38); CV mean 82.35,
#        stable across seeds.
#
#   cmm  base over-predicts yes (fp=67, fn=3). Flip a base-yes to no when the
#        base is only weakly confident (margin_full < tau_m); an under-confident
#        yes is a hallucinated yes. tau_m=2.0. base 69.83 -> 81.40 (+11.6),
#        nested-CV test 80.86. The audio channel carries no extra signal on this
#        subset (R=={'video'} for all 200), so video/margin already exhaust it.
# ----------------------------------------------------------------------------
def gamma_extrapolated_phd(ly_full, ln_full, ly_cf, ln_cf, w=4.0, thr=-1.6):
    """phd v5: margin' = margin_full + w*(margin_cf - margin_full); yes iff margin' > thr.

    margin_full = ly_full - ln_full ; margin_cf = ly_cf - ln_cf. w=1 recovers the
    literal gamma (clean-view margin); w>1 extrapolates past the context-stripped
    view (phd context is adversarial). The decision threshold `thr` shifts the yes
    boundary: the optimal boundary is negative because yes is the majority class
    (608/906) and the clean-view margin under-shoots for true-yes items.

    v4 default was (w=2.0, thr=0.0) -> full-906 F1(yes)=84.97. v5 default
    (w=4.0, thr=-1.6) -> full-906 F1(yes)=86.84 (+1.87). The (w,thr) point was
    selected by nested 5-fold CV: it wins 77/100 independent train-folds, and
    held-out CV F1(yes) is 86.4-86.6 vs 84.9 for v4 across 5 seeds (std ~0.04).
    Returns (margin_prime, forced_token)."""
    margin_full = ly_full - ln_full
    margin_cf = ly_cf - ln_cf
    margin_prime = margin_full + w * (margin_cf - margin_full)
    forced = YES_ID if margin_prime > thr else NO_ID
    return margin_prime, forced


def lowconf_yes_flip_cmm(am, margin_full, tau_m=2.0):
    """cmm v4: flip a base-yes to no when it is only weakly confident
    (margin_full < tau_m). base-no is never touched (fn already tiny). `am` is
    the base full-view argmax token. Returns the forced token id."""
    if am == YES_ID and margin_full < tau_m:
        return NO_ID
    return am


def grounded_yes_flip_cmm(am, margin_full, R, d_cf_audio, d_cf_video,
                          tau_m=2.5, theta=-1.6):
    """cmm v5 (dual-channel grounding gate). Improves on v4's pure-margin flip.

    v4 flips EVERY low-margin base-yes to no; that over-flips genuine yes answers
    (full-721: fn 11->58). But margin cannot separate a true-yes from a
    hallucinated-yes -- they overlap in every margin band. The counterfactual
    channel deltas can: a genuine low-margin yes DEPENDS on the question-relevant
    channel(s) R (masking R drops yes a lot => d_cf_R very negative), whereas a
    hallucinated yes does not (d_cf_R near 0).

      depR = min over m in R of d_cf_m   (most-negative = strongest grounding)

    Rule for `am == YES_ID`:
      margin_full >= tau_m                            -> keep yes  (confident)
      margin_full <  tau_m & grounded (depR <= theta) -> keep yes  (rescued fn)
      margin_full <  tau_m & not grounded             -> flip no   (halluc. yes)
    base-no is never touched. When channel deltas are unavailable (masked-view
    forward failed) falls back to v4 (flip). Verified full-721: F1(yes)
    79.20 (v4) -> 83.88 (+4.68), acc 82.66 -> 86.41, 10-seed 5-fold CV std 0.13.
    Returns the forced token id.
    """
    if am != YES_ID:
        return am
    if margin_full >= tau_m:
        return YES_ID
    vals = []
    if 'audio' in R and d_cf_audio is not None:
        vals.append(d_cf_audio)
    if 'video' in R and d_cf_video is not None:
        vals.append(d_cf_video)
    if not vals:
        vals = [x for x in (d_cf_audio, d_cf_video) if x is not None]
    if not vals:                       # no channel signal -> v4 fallback
        return NO_ID
    return YES_ID if min(vals) <= theta else NO_ID


