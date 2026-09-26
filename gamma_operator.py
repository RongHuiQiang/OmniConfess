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
