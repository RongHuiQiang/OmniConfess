# -*- coding: utf-8 -*-
"""operators.py -- Stage-3 decision operators (gamma / grounding rules).

These are the per-dataset instantiations of the confession decision:

    kappa_d^(m) = F_d(X) - F_d(X^{-m})      # channel-m contribution to answer d
    E_d^R       = sum_{m in R} kappa_d^(m)  # accumulate over relevant channels R
    decision    = argmax_d E_d^R

Each function below is one dataset's concrete realisation of that rule (PHD's
single text channel, CMM's dual audio/video channels, PubMedQA's passage
channel). They are pure functions of already-computed log-probs so the Stage-3
math is testable in isolation and byte-identical to the historical inlined
copies in the unified runner.
"""
import re

# Decision-token ids (Qwen2.5-Omni; shared by all datasets).
YES_ID, NO_ID, MAYBE_ID = 9454, 2753, 21390


# ---- PHD: single ablatable channel (text context) --------------------------

def gamma_extrapolated_phd(ly_full, ln_full, ly_cf, ln_cf, w=4.0, tau_p=-1.6):
    """PHD Stage-3 rule.

    margin' = margin_full + w*(margin_cf - margin_full)
    yes iff margin' > tau_p.

    w>1 extrapolates past the clean (context-stripped) view: PHD context is
    adversarial misinformation, so amplifying how far removing it moves the
    decision is the correct correction.
    Returns (margin_prime, forced_token).
    """
    margin_full = ly_full - ln_full
    margin_cf   = ly_cf  - ln_cf
    margin_prime = margin_full + w * (margin_cf - margin_full)
    return margin_prime, (YES_ID if margin_prime > tau_p else NO_ID)


# ---- CMM: two ablatable channels (audio, video) ----------------------------

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


def relevant_modalities_cmm(question=None, group_key=None):
    """Per-sample relevant modality set R for CMM: {'audio'}, {'video'}, or both."""
    if question:
        a = bool(_CMM_AUDIO_RE.search(question))
        v = bool(_CMM_VIDEO_RE.search(question))
        if a and not v: return {'audio'}, 'question'
        if v and not a: return {'video'}, 'question'
        if a and v:     return {'audio', 'video'}, 'question_both'
    if group_key in _CMM_GROUP_TO_R:
        return set(_CMM_GROUP_TO_R[group_key]), 'group_key'
    return {'audio', 'video'}, 'fallback_both'


def confidence_gated_flip(am, margin_full, flip_token, tau_conf):
    """Anchor to base when |margin_full| >= tau_conf; else take flip_token."""
    return am if abs(margin_full) >= tau_conf else flip_token


def grounded_yes_flip_cmm(am, margin_full, R, d_cf_audio, d_cf_video,
                           margin_conf=2.0, tau_m=-1.6):
    """CMM Stage-3 rule (dual-channel grounding gate).

    For am==YES_ID:
      margin_full >= margin_conf                          -> keep yes  (confident)
      margin_full <  margin_conf, grounded (depR<=tau_m)  -> keep yes
      margin_full <  margin_conf, not grounded            -> flip no
    base-no is never touched.
    """
    if am != YES_ID:
        return am
    if margin_full >= margin_conf:
        return YES_ID
    vals = []
    if 'audio' in R and d_cf_audio is not None: vals.append(d_cf_audio)
    if 'video' in R and d_cf_video is not None: vals.append(d_cf_video)
    if not vals:
        vals = [x for x in (d_cf_audio, d_cf_video) if x is not None]
    if not vals:
        return NO_ID
    return YES_ID if min(vals) <= tau_m else NO_ID


def gamma_margin_cmm(margin_full, margin_noaudio, margin_novideo, R):
    """CMM lambda-delta ablation: E_d^R = sum_{m in R} kappa^(m)."""
    er = 0.0
    if 'audio' in R: er += margin_full - margin_noaudio
    if 'video' in R: er += margin_full - margin_novideo
    return er, (YES_ID if er > 0 else NO_ID)


# ---- PubMedQA: passage is the relevant channel (3-class) -------------------

def gamma_scores_pubmedqa(ly_full, ln_full, lm_full, ly_cf, ln_cf, lm_cf):
    """Literal E^R rule: argmax_d kappa_d^(passage), kappa_d = lp_d_full - lp_d_cf."""
    scores = {YES_ID: ly_full - ly_cf, NO_ID: ln_full - ln_cf,
              MAYBE_ID: lm_full - lm_cf}
    return scores, max(scores, key=scores.get)


def gamma_anchored_pubmedqa(am, ly, ln, lm, ly_cf, ln_cf, lm_cf, tau_maybe=0.5):
    """PubMedQA Stage-3 rule (candidate-anchored + maybe gate).

    Keep base class am, EXCEPT when am==MAYBE and kappa_maybe = lm - lm_cf
    <= tau_maybe (passage does not ground 'maybe'): then redecide yes/no.
    """
    if am == MAYBE_ID and (lm - lm_cf) <= tau_maybe:
        return YES_ID if ly > ln else NO_ID
    return am
