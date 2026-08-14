# -*- coding: utf-8 -*-
# ==============================================================================
# OmniConfess -- Qwen2.5-Omni backbone
#
# A training-free decoding framework that audits and repairs cross-modal
# hallucinations at three granularities during generation:
#
#   Dist-Confess    token level      flags prior-driven tokens via the modality
#                                    dependency index (MDI); emits _dist_flag
#   Layer-Confess   layer level      measures cross-layer drift between anchor
#                                    layers and the final layer; emits
#                                    total_hallu_penalty
#   Path-Confess    trajectory level consumes the two signals above to veto and
#                                    re-rank candidate generation paths
#
# NOTE -- the three modules are NOT orthogonal; any ablation must state this:
#   Dist  produces _dist_flag, which both gates Layer repair and supplies Path
#         with its dist_ev evidence
#   Layer produces total_hallu_penalty, the sole input to Path's veto
#   Path  only consumes the above; it produces no new observable of its own
#   See the --disable_dist / --disable_layer / --disable_path help strings.
#
# Example:
#   python omniconfess_qwen.py \
#       --datasets cmm \
#       --data_path ./OmniHalluBench/cmm/cmm.json \
#       --data_root ./OmniHalluBench/cmm/ \
#       --output_dir ./outputs \
#       --file_name cmm_run
# ==============================================================================
import time
import numpy as np
import torch
import json
import os
import argparse
import cv2
import librosa
from PIL import Image
import re
import sys as _sys
import math
_sys.path.insert(0, os.path.dirname(__file__))
try:
    from halo_prompts import get_halo_prompt as _get_halo_prompt
    _HALO_PROMPTS_LOADED = True
except ImportError:
    _HALO_PROMPTS_LOADED = False
try:
    from halueval_prompts import get_halueval_prompt as _get_halueval_prompt
    _HALUEVAL_PROMPTS_LOADED = True
except ImportError:
    _HALUEVAL_PROMPTS_LOADED = False
try:
    from avh_prompts import get_avh_prompt as _get_avh_prompt
    _AVH_PROMPTS_LOADED = True
except ImportError:
    _AVH_PROMPTS_LOADED = False
try:
    from judge_prompts import get_judge_prompt as _get_judge_prompt
    from judge_prompts import get_stage1_scaffold as _get_stage1_scaffold
    from judge_prompts import cmm_stream_of as _cmm_stream_of
    from judge_prompts import cmm_target_of as _cmm_target_of
    from judge_prompts import cmm_stage1_heads as _cmm_stage1_heads
    from judge_prompts import JUDGE_PROMPTS as _JUDGE_PROMPTS_REGISTRY
    _JUDGE_PROMPT_VERSIONS = sorted({v for _d in _JUDGE_PROMPTS_REGISTRY.values() for v in _d})
    _JUDGE_PROMPTS_LOADED = True
except ImportError:
    _JUDGE_PROMPTS_LOADED = False
    _JUDGE_PROMPT_VERSIONS = ['v0']
try:
    import format_control as _FC
    _FORMAT_CONTROL_LOADED = True
except ImportError:
    _FORMAT_CONTROL_LOADED = False
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
try:
    from sentence_transformers import SentenceTransformer
    _SBERT_AVAILABLE = True
except ImportError:
    _SBERT_AVAILABLE = False
from transformers import (
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniProcessor,
)


INF = 10
TEMPERATURE = 0.6
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# ================= 1. Multimodal loading utilities =================

def load_image(image_path):
    """Load a single image."""
    if not image_path or not os.path.exists(image_path): return None
    try:
        return Image.open(image_path).convert('RGB')
    except:
        return None

def load_video_frames(video_path, num_frames=8):
    if not video_path or not os.path.exists(video_path): return None
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0: return None
    indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.resize(frame, (224, 224))
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.array(frames) if frames else None
    
#
AUDIO_SR = 24000


def load_audio_data(audio_path, sr=None):
    """Load audio. sr=None falls back to the global AUDIO_SR set by --audio_sr."""
    if not audio_path or not os.path.exists(audio_path):
        return None
    try:
        audio, _ = librosa.load(audio_path, sr=(AUDIO_SR if sr is None else sr))
        return audio if (audio is not None and len(audio) > 0) else None
    except:
        return None


#
#
USE_AUDIO_IN_VIDEO = False


def _aiv_kw(pixel_values=None, audio_values=None):
    """Pass-through of use_audio_in_video for processor(...) call sites.

    When disabled this returns an empty dict, so the call site stays byte-identical
    to the original behaviour instead of explicitly passing False.
    True is only returned when both video and audio are present: in the interleaved
    branch <|VIDEO|> advances next(audio_lengths), and with video but no audio that
    iterator is empty, which raises StopIteration.
    """
    ok = (USE_AUDIO_IN_VIDEO and pixel_values is not None and audio_values is not None)
    return {"use_audio_in_video": True} if ok else {}


def _aiv_kw_for(inputs):
    """Pass-through for generate(...) call sites, decided from whether the processor
    output actually contains both video and audio. Equivalent to _aiv_kw.

    The two call sites that go through get_rope_index must agree with the
    processor, otherwise TMRoPE counts multimodal segments using the wrong branch
    (the video_nums criterion is itself the use_audio_in_video branch).
    """
    if not USE_AUDIO_IN_VIDEO:
        return {}
    try:
        ok = (inputs.get('input_features', None) is not None
              and inputs.get('pixel_values_videos', None) is not None)
    except Exception:
        ok = False
    return {"use_audio_in_video": True} if ok else {}


def _aiv_audio_folded(example):
    """Whether this sample's audio has already been folded into the video blocks, in
    which case the chat template must not add a separate audio placeholder.

    Folding only happens when both video and audio are present. For audio-only
    samples the interleaved branch never triggers and <|AUDIO|> must still appear,
    otherwise the audio is dropped entirely. The criterion must match _aiv_kw.
    """
    return (USE_AUDIO_IN_VIDEO
            and example.get('pixel_values') is not None
            and example.get('audio_values') is not None)


# ================= Token-Guard plugin =================
class DualLayerAuditor:
    def __init__(self, model, processor, target_layers=[16, 22], store_hidden=False,
                 scope='last'):
        self.model = model
        self.processor = processor
        self.target_layers = target_layers
        self.store_hidden = store_hidden
        # -- scope: which forward pass the hook should read from --
        self.scope = scope
        self.active = True
        self.intermediate_logits = {}
        self.intermediate_hidden = {}
        self.blind_logits = {}
        self.blind_hidden = {}
        self.shadow_logits = {}
        self.shadow_hidden = {}
        self.handles = []

    def _make_hook(self, layer_idx):
        def _hook(module, input, output):
            hidden_states = output[0] if isinstance(output, tuple) else output
            with torch.no_grad():
                if self.scope == 'full' and not self.active:
                    L, H = self.blind_logits, self.blind_hidden
                elif self.scope == 'shadow' and self.active:
                    L, H = self.shadow_logits, self.shadow_hidden
                else:
                    L, H = self.intermediate_logits, self.intermediate_hidden
                if self.store_hidden:
                    H[layer_idx] = hidden_states[:, -1, :].clone()
                normed_hidden = self.model.thinker.model.norm(hidden_states)
                logits = self.model.thinker.lm_head(normed_hidden)
                L[layer_idx] = logits[:, -1, :]
        return _hook

    def register(self):
        layers = list(self.target_layers)
        if self.store_hidden:
            final_idx = len(self.model.thinker.model.layers) - 1
            if final_idx not in layers:
                layers.append(final_idx)
        for idx in layers:
            handle = self.model.thinker.model.layers[idx].register_forward_hook(self._make_hook(idx))
            self.handles.append(handle)

    def remove(self):
        for h in self.handles: h.remove()
        self.handles = []
        self.intermediate_logits = {}
        self.intermediate_hidden = {}
        self.blind_logits = {}
        self.blind_hidden = {}
        self.shadow_logits = {}
        self.shadow_hidden = {}

STOP_WORDS = {"i", "is", "the", "a", "an", "and", "are", "was", "were", "be", "have", "has", "it", "they", "in", "on", "at", "to", "for", "of", "with", "yes", "no", "sound", "video", "audio", "hear", "heard", "see", "saw", "contains", "contain"}

# Calibratable constants (previously hardcoded inline).
# Defaults reproduce the original behaviour exactly — every value here is the
# literal that used to appear in the code, so existing runs stay reproducible.
# Exposed via CLI so they can be swept (see --sweep_* arguments).
_DEFAULT_THRESHOLDS = {
    'res_22_fin': 1.35,        # Layer-Confess LATE detection
    'l22_conf_late': 0.90,     # Layer-Confess LATE confidence gate
    'res_16_22': 1.40,         # Layer-Confess HIJACK detection
    'l22_conf_hijack': 0.85,   # Layer-Confess HIJACK confidence gate
    'penalty_late': 0.5,       # Layer-Confess LATE base penalty
    'penalty_late_boost': 0.5, # Layer-Confess LATE high-confidence surcharge
    'penalty_hijack': 0.6,     # Layer-Confess HIJACK penalty
    'mdi_threshold': -0.3,     # Dist-Confess CD trigger
    'layer_repair_alpha': 0.3, # Layer-Confess hidden interpolation weight
    'veto_threshold': 0.45,    # Path-Confess veto cutoff
}


def thresholds_from_args(args):
    """Build a thresholds dict from CLI overrides, falling back to defaults."""
    th = dict(_DEFAULT_THRESHOLDS)
    for key in th:
        val = getattr(args, key, None)
        if val is not None:
            th[key] = val
    return th


_BOXED_RE = re.compile(r'^\s*\\?boxed\s*\{(.*?)\}?\s*$', re.S)


def unwrap_boxed(s):
    """Strip a `\\boxed{...}` wrapper, including the unclosed variety.

    This matters because Token-F1 removes punctuation without inserting a space,
    so `\\boxed{no}` collapses into the single token `boxedno`, which both adds a
    junk token and swallows the actual answer word.
    Note that the older `lstrip('boxed{')` never fired: the leading character is a
    backslash, which is not in that character set.
    """
    t = str(s).strip()
    m = _BOXED_RE.match(t)
    return m.group(1).strip() if m else t


def _final_verdict(text):
    """Return the last explicit yes/no in the text, or None if there is none.
    Deliberately does not fall back to No."""
    s = re.sub(r"\\boxed\{?", " ", str(text).lower())
    m = re.findall(r"\b(yes|no)\b", s)
    return m[-1] if m else None


def resolve_final_tiebreak(final_texts, second_texts, cur_idx):
    """Second-stage self-consistency vote, applied only when the two final candidates
    disagree on the verdict.

    The majority verdict over all second-stage completions decides which side to
    take, and selected_idx is moved to the final candidate carrying that verdict.
    An even split leaves the selection untouched.

    The rule is symmetric, so it presupposes neither answer class, and it costs
    nothing extra: the completions it votes over have already been generated and
    stored; they were simply not being used to decide the final answer.

    Kept as a module-level pure function so that offline analysis can import it and
    recompute the rule exactly as it is applied here.
    """
    finals = [_final_verdict(t) for t in final_texts]
    if len(finals) < 2 or any(v is None for v in finals) or len(set(finals)) < 2:
        return cur_idx
    votes = [v for v in (_final_verdict(t) for t in second_texts) if v]
    if not votes:
        return cur_idx
    y = votes.count("yes")
    n = votes.count("no")
    if y == n:
        return cur_idx
    target = "yes" if y > n else "no"
    for i, v in enumerate(finals):
        if v == target:
            return i
    return cur_idx


def calculate_hallu_penalty(report, tokenizer, thresholds=None):
    """Compute the hallucination penalty and report the triggering details.

    Uses purely cross-layer signals: res_22_fin, res_16_22 and l22_conf. It does
    not depend on MDI.
    thresholds: dict of calibratable values (None falls back to _DEFAULT_THRESHOLDS).
    """
    if thresholds is None:
        thresholds = _DEFAULT_THRESHOLDS

    token_id = report['token_id']
    token_str = tokenizer.decode([token_id]).lower().strip()

    clean_word = ''.join(e for e in token_str if e.isalpha())
    if len(clean_word) <= 1 or clean_word in STOP_WORDS:
        return 0.0

    penalty = 0.0

    if report['res_22_fin'] > thresholds['res_22_fin']:
        p_inc = thresholds['penalty_late']
        if report['l22_conf'] > thresholds['l22_conf_late']:
            p_inc += thresholds['penalty_late_boost']

        penalty += p_inc
        print(f"🚨 [AUDIT: LATE] Token: '{token_str}' | Res: {report['res_22_fin']:.2f} | L22_Conf: {report['l22_conf']:.2f} | Penalty: +{p_inc}", flush=True)

    if report['res_16_22'] > thresholds['res_16_22'] and report['l22_conf'] > thresholds['l22_conf_hijack']:
        p_inc = thresholds['penalty_hijack']
        penalty += p_inc
        print(f"🚨 [AUDIT: HIJACK] Token: '{token_str}' | Res: {report['res_16_22']:.2f} | Conf: {report['l22_conf']:.2f} | Penalty: +{p_inc}", flush=True)

    return penalty

def _grnd_score(st, field='all', agg='sqrt'):
    """Collapse the per-path sufficient statistics into a single scalar, with the
    convention that larger always means better supported by evidence.

    `field` selects the channel:
      all/content/neg  masking contrast g_t = log p_full - log p_blind. 'neg'
                       accumulates only the negative half, i.e. the log-domain
                       version of the older one-sided dist_ev convention, kept for
                       comparison.
      late22/late16    cross-layer injection d_t = log p_final - log p_22,16, which
                       needs no masking. A larger d means the token looks more like
                       a prior forced in by the final layer, so after negation
                       larger is still better.
      late22c          as late22, restricted to content words.
      lp22/lp16        bounded counterpart of the cross-layer injection,
                       e_t = p_final(y) - p_22,16(y), which lies in [-1, 1]. These
                       ask the same question as late22/late16, but the log ratio is
                       unbounded and diverges when a mid-stack layer assigns the
                       token near-zero probability, which saturates the clipping and
                       leaves extreme-value aggregation unable to rank candidates at
                       all. The probability difference is bounded, so min and worst3
                       are only well defined on this channel.

    `agg` selects the normalisation and aggregation:
      sum     the log Bayes factor itself; long paths are favoured
      mean    length-independent, but divides down the signal of long paths
      sqrt    best signal-to-noise ratio, since signal grows like n and noise like
              sqrt(n); this is the default
      min     the extreme value, on the view that hallucination is an extreme-value
              risk: one unsupported assertion ruins the whole answer, and a mean
              would dilute it with the surrounding tokens
      worst3  the mean of the three worst tokens, a robust version of min

    The unbounded log-ratio channels saturate under min and worst3; use lp22/lp16
    when an extreme-value aggregation is wanted.
    """
    if not st:
        return 0.0
    _n_all = st.get('g_n', 0)
    _n_late = st.get('d_n', 0)
    if field in ('all', 'neg') and _n_all <= 0:
        return 0.0
    if field == 'content' and st.get('g_n_c', 0) <= 0:
        return 0.0
    if field in ('late22', 'late16', 'late22c', 'lp22', 'lp16') and _n_late <= 0:
        return 0.0
    if field in ('all', 'content', 'neg'):
        s = {'all': st.get('g_sum', 0.0), 'content': st.get('g_sum_c', 0.0),
             'neg': st.get('g_neg_sum', 0.0)}[field]
        n = st.get('g_n_c', 0) if field == 'content' else st.get('g_n', 0)
        if agg == 'min':
            return float(st.get('g_min_c' if field == 'content' else 'g_min', 0.0))
        if agg == 'worst3':
            b = st.get('g_bot') or []
            return (sum(b) / len(b)) if b else 0.0
    elif field in ('late22', 'late16', 'late22c'):
        s = -{'late22': st.get('d22_sum', 0.0), 'late16': st.get('d16_sum', 0.0),
              'late22c': st.get('d22_sum_c', 0.0)}[field]
        n = st.get('d_n_c', 0) if field == 'late22c' else st.get('d_n', 0)
        if agg == 'min':
            return -float(st.get('d16_max' if field == 'late16' else 'd22_max', 0.0))
        if agg == 'worst3':
            # lp22/lp16（e_t = p_final − p_l ∈ [−1,1]）。
            return 0.0
    elif field in ('lp22', 'lp16'):
        s = -{'lp22': st.get('e22_sum', 0.0), 'lp16': st.get('e16_sum', 0.0)}[field]
        n = st.get('d_n', 0)
        if agg == 'min':
            return -float(st.get('e16_max' if field == 'lp16' else 'e22_max', 0.0))
        if agg == 'worst3':
            b = st.get('e22_top') or []
            return -(sum(b) / len(b)) if b else 0.0
    else:
        return 0.0
    if n <= 0:
        return 0.0
    if agg == 'sum':
        return float(s)
    if agg == 'mean':
        return float(s) / n
    return float(s) / (n ** 0.5)      # 'sqrt'


def layer_confess_repair(h_anchors, h_final, omega, alpha_l, lambda_l, eps=1e-8,
                         project='perp'):
    """Transported anchor response and orthogonal restoration for Layer-Confess.

    Transported anchor response. The transport operator is
        A_t^(l) = I + alpha_l * (h_final - h_l) h_l^T / (||h_l||^2 + eps)
    and since h_l^T h_l = ||h_l||^2 the matrix never has to be materialised:
        A_t^(l) h_l = (1 - alpha_l) * h_l + alpha_l * h_final          (d-vector)

    Cross-layer support representation:
        h_sup = sum_l omega_l * A_t^(l) h_l,        sum_l omega_l = 1

    Orthogonal restoration:
        h~ = h_final + lambda_L * P_perp(h_sup - h_final)
        P_perp(v) = v - h_final <h_final, v> / ||h_final||^2

    Because the update is orthogonal to h_final, ||h~||^2 = ||h_final||^2 +
    lambda_L^2 ||P_perp(v)||^2, i.e. ||h~|| = ||h_final|| + O(lambda_L^2), which is
    the norm-preservation property.

    On the choice of projection: P_perp removes the component of (h_sup - h_final)
    that lies along h_final, and that component is exactly the part which would
    attenuate the final layer's own prior-driven prediction; what remains is a
    lateral adjustment. Norm preservation is therefore bought by barely moving the
    representation at all, at the cost of not being able to suppress the prior
    direction.
        project='perp'  h~ = h_final + lambda_L * P_perp(h_sup - h_final)
        project='full'  h~ = h_final + lambda_L * (h_sup - h_final)
    'full' keeps the parallel component and can shorten the vector along h_final,
    which is what suppressing the prior requires, but it no longer preserves the
    norm, so the norm-preservation property does not hold for it.

    Args:
        h_anchors: list of anchor hidden states (each shape [d]), excluding final
        h_final:   final-layer hidden state, shape [d]
        omega:     weights over h_anchors, summing to 1
        alpha_l:   depth-alignment coefficient alpha_l in [0, 1]
        lambda_l:  repair strength lambda_L
        project:   'perp' (norm-preserving) or 'full' (keeps the parallel component)
    Returns:
        (h_repaired, diagnostics dict). h_repaired is cast back to h_final's
        original dtype. diag['_delta'] is the float32 update direction before
        multiplication by lambda_L, so the caller can sweep lambda with no extra
        forward pass; it should be popped before serialisation.

    Numerics: the model runs in bfloat16, whose spacing near 1.0 is 1/128
    (~7.8e-3). The norm deviation this repair induces is O(lambda_L^2), which for
    lambda_L=0.3 lands around 1e-3 - below bf16 resolution, so measuring it in bf16
    yields quantised garbage (norm ratios pinned to 1+-1/128) and an orthogonality
    residual inflated to ~5e-2. All arithmetic here is therefore done in float32;
    only the returned tensor is cast back.
    """
    orig_dtype = h_final.dtype
    h_final = h_final.float()
    h_anchors = [h.float() for h in h_anchors]

    h_sup = None
    for w, h_l in zip(omega, h_anchors):
        transported = (1.0 - alpha_l) * h_l + alpha_l * h_final
        h_sup = transported * w if h_sup is None else h_sup + transported * w

    delta = h_sup - h_final
    fn_sq = (h_final * h_final).sum()
    proj_coeff = (h_final * delta).sum() / (fn_sq + eps)
    delta_perp = delta - h_final * proj_coeff          # P_perp(h_sup - h_final)
    _upd = delta if project == 'full' else delta_perp
    h_repaired = h_final + lambda_l * _upd

    n_final = h_final.norm()
    n_rep = h_repaired.norm()
    diag = {
        'norm_final': float(n_final),
        'norm_repaired': float(n_rep),
        'norm_ratio': float(n_rep / (n_final + eps)),
        # residual orthogonality: should be ~0 up to fp error
        'orth_residual': float((h_final * delta_perp).sum() / (n_final * delta_perp.norm() + eps)),
        'delta_perp_norm': float(delta_perp.norm()),
        'delta_par_norm': float((h_final * proj_coeff).norm()),
        'project': project,
        '_delta': _upd,
    }
    return h_repaired.to(orig_dtype), diag


def layer_confess_barycenter(q_list, omega, tau_b=0.0, eps=1e-8):
    """Closed form of Eq.(layer-support) / Proposition 2, first claim.

        s_t(v) proportional to ( prod_l q_t^(l)(v)^omega_l )^(1/(1+tau_B))

    i.e. a depth-weighted geometric mean tempered by tau_B. Computed in log
    space for stability. q_list are anchor distributions over the vocabulary
    (or over a top-K restriction).
    """
    log_s = None
    for w, q in zip(omega, q_list):
        term = w * torch.log(q + eps)
        log_s = term if log_s is None else log_s + term
    log_s = log_s / (1.0 + tau_b)
    log_s = log_s - log_s.max()
    s = torch.exp(log_s)
    return s / (s.sum() + eps)


def safe_model_generate(model, processor, inputs, **gen_kwargs):
    """generate() wrapper that installs the feature interceptor and prints detailed
    debug information."""
    print(f"\n    [GEN_DEBUG] ---> entering safe_model_generate, preparing inputs...", flush=True)
    import traceback

    # Apply max_new_tokens_override if set via _args passthrough
    _args = gen_kwargs.pop('_args', None)
    if _args is not None:
        if getattr(_args, 'max_new_tokens_override', None) is not None:
            gen_kwargs['max_new_tokens'] = _args.max_new_tokens_override
        # Apply global repetition_penalty only if not already set per-call
        if getattr(_args, 'repetition_penalty', 1.0) != 1.0:
            if 'repetition_penalty' not in gen_kwargs:
                gen_kwargs['repetition_penalty'] = _args.repetition_penalty

    for k, v in inputs.items():
        if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
            inputs[k] = v.to(torch.bfloat16)

    class StopExecution(Exception): pass
    captured = {}
    
    def hook(module, args, kwargs):
        target = kwargs.get('inputs_embeds', args[0] if len(args) > 0 else None)
        if target is not None:
            captured['inputs_embeds'] = target.detach().clone()
            print(f"    [GEN_DEBUG] 🪝 hook captured inputs_embeds, shape: {captured['inputs_embeds'].shape}", flush=True)
            raise StopExecution("Captured")

    print(f"    [GEN_DEBUG] installing interceptor hook...", flush=True)
    h1 = model.thinker.register_forward_pre_hook(hook, with_kwargs=True)
    h2 = model.thinker.model.register_forward_pre_hook(hook, with_kwargs=True)
    
    try:
        print(f"    [GEN_DEBUG] running an empty generate to trigger feature mapping...", flush=True)
        with torch.inference_mode():
            model.generate(**inputs, max_new_tokens=1, use_cache=False, **_aiv_kw_for(inputs))
    except StopExecution:
        print(f"    [GEN_DEBUG] ✅ StopExecution fired as expected; features captured.", flush=True)
    except Exception as e:
        print(f"    [GEN_DEBUG] ⚠️ unexpected exception during the trigger phase: {repr(e)}", flush=True)
    finally:
        h1.remove()
        h2.remove()
        print(f"    [GEN_DEBUG] hook removed.", flush=True)
        
    embeds = captured.get('inputs_embeds')
    if embeds is None:
        print("    [GEN_DEBUG] ⚠️ capture failed; falling back to text-only embeddings...", flush=True)
        try:
            embeds = model.thinker.get_input_embeddings()(inputs['input_ids'])
            print(f"    [GEN_DEBUG] text-only embeddings obtained, shape: {embeds.shape}", flush=True)
        except Exception as e:
            print(f"    [GEN_DEBUG] ❌ fallback to text embeddings failed: {repr(e)}", flush=True)
            raise e
            
    _tok = gen_kwargs.get('tokenizer')
    _stops = gen_kwargs.get('stop_strings')
    clean_kwargs = {k: v for k, v in gen_kwargs.items() if k not in ['tokenizer', 'stop_strings']}
    if (_args is not None and getattr(_args, 'fix_stop_strings', False)
            and _tok is not None and _stops and _FORMAT_CONTROL_LOADED):
        _sc = _FC.build_stopping_criteria(_tok, _stops)
        if _sc is not None:
            _prev = clean_kwargs.get('stopping_criteria')
            if _prev is not None:
                for _c in _prev:
                    _sc.append(_c)
            clean_kwargs['stopping_criteria'] = _sc
    print(f"    [GEN_DEBUG] cleaned kwargs for thinker: {list(clean_kwargs.keys())}", flush=True)
    
    try:
        print(f"    [GEN_DEBUG] 🚀 calling model.thinker.generate...", flush=True)
        out = model.thinker.generate(
            inputs_embeds=embeds,
            attention_mask=inputs['attention_mask'],
            pad_token_id=processor.tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
            **clean_kwargs
        )
        print(f"    [GEN_DEBUG] ✅ thinker.generate succeeded.", flush=True)
    except Exception as e:
        print(f"    [GEN_DEBUG] ❌ thinker.generate crashed: {repr(e)}", flush=True)
        traceback.print_exc()
        raise e
    
    try:
        if hasattr(out, 'sequences'):
            print(f"    [GEN_DEBUG] stitching generated sequences; new sequence shape: {out.sequences.shape}", flush=True)
            full_seqs = []
            for seq in out.sequences:
                prompt_ids = inputs['input_ids'][0].to(seq.device)
                full_seq = torch.cat([prompt_ids, seq], dim=0)
                full_seqs.append(full_seq)
            out.sequences = torch.stack(full_seqs)
            print(f"    [GEN_DEBUG] stitching done; final sequence shape: {out.sequences.shape}", flush=True)
    except Exception as e:
        print(f"    [GEN_DEBUG] ❌ sequence stitching crashed: {repr(e)}", flush=True)
        traceback.print_exc()
        raise e
        
    print(f"    [GEN_DEBUG] <--- safe_model_generate finished, returning.\n", flush=True)
    return out

def softmax(x):
    e_x = np.exp(np.array(x) - np.max(x))
    return e_x / e_x.sum(axis=0)


def get_modality_indices(processor, input_ids, device):
    """Locate the positions of video and audio tokens in the input sequence."""
    v_bos = processor.tokenizer.convert_tokens_to_ids("<|vision_bos|>")
    v_eos = processor.tokenizer.convert_tokens_to_ids("<|vision_eos|>")
    a_bos = processor.tokenizer.convert_tokens_to_ids("<|audio_bos|>")
    a_eos = processor.tokenizer.convert_tokens_to_ids("<|audio_eos|>")

    #   <|vision_bos|><|audio_bos|> [VIDEO…][AUDIO…][VIDEO…][AUDIO…] … <|audio_eos|><|vision_eos|>
    if USE_AUDIO_IN_VIDEO:
        def get_pad_idx(tok):
            tid = processor.tokenizer.convert_tokens_to_ids(tok)
            if tid is None:
                return None
            hit = (input_ids == tid).nonzero(as_tuple=True)
            return hit[1].to(device) if hit[1].numel() > 0 else None
        return get_pad_idx("<|VIDEO|>"), get_pad_idx("<|AUDIO|>")

    def get_range(bos, eos):
        b_pos = (input_ids == bos).nonzero(as_tuple=True)
        e_pos = (input_ids == eos).nonzero(as_tuple=True)
        if b_pos[0].numel() > 0 and e_pos[0].numel() > 0:
            return torch.arange(b_pos[1][0].item(), e_pos[1][0].item() + 1, device=device)
        return None

    return get_range(v_bos, v_eos), get_range(a_bos, a_eos)


def get_passage_indices(input_ids, tokenizer, device):
    """Locate passage token indices in text-only (halueval/pubmedqa/ragtruth) inputs.

    The prompt structure (from _prepare_chat_template_for_first_stage) is:
        ... Passage: {passage_text}\nQuestion: {question_text}\n ...

    Strategy: find the token sequence for "Passage" (start marker), then find
    the *next* newline + "Question" pair (end marker).  BPE is context-sensitive,
    so we search for individual tokens rather than encoding full boundary strings.

    Returns a 1-D LongTensor of content indices (between markers), or None.
    """
    ids = input_ids[0] if input_ids.dim() == 2 else input_ids  # (seq_len,)
    ids_list = ids.tolist()
    seq_len = len(ids_list)

    # Key token IDs (stable across BPE contexts for Qwen tokenizer)
    # "Passage" -> [12187, 424]  ("Pass" + "age")
    # "Question" -> [14582]
    passage_marker = tokenizer.encode("Passage", add_special_tokens=False)   # [12187, 424]
    question_tok = tokenizer.encode("Question", add_special_tokens=False)[0]  # 14582

    def find_subseq(haystack, needle, start=0):
        n = len(needle)
        for i in range(start, len(haystack) - n + 1):
            if haystack[i:i+n] == needle:
                return i
        return -1

    # Find "Passage" marker
    p_start = find_subseq(ids_list, passage_marker)
    if p_start == -1:
        return None

    # Passage content starts after "Passage: " — skip marker + ":" + " "
    # Typical: [Pass, age, :, <space>] = marker_len + 2
    passage_begin = p_start + len(passage_marker) + 2  # skip ": "
    if passage_begin >= seq_len:
        return None

    # Find "Question" token after passage content.
    # NOTE: the \n before "Question" may be merged with the preceding token
    # by BPE (e.g. ".\n" = token 624), so we search for "Question" (14582)
    # directly and include the preceding merged-newline token in the mask.
    for i in range(passage_begin, seq_len):
        if ids_list[i] == question_tok:
            # passage_end = position of the token *before* "Question"
            # (that token likely contains the trailing \n of the passage)
            passage_end = i  # exclusive
            if passage_end > passage_begin:
                return torch.arange(passage_begin, passage_end, device=device)
            return None

    return None


def calculate_entropy(logits):
    """Entropy of the current logits, used to observe how uncertain the model is."""
    probs = torch.softmax(logits, dim=-1)
    return -torch.sum(probs * torch.log(probs + 1e-9), dim=-1).item()


def safe_thinker_step_audit(model, processor, inputs, past_key_values=None, position_ids=None, **gen_kwargs):
    """Minimal forward pass: a single decoding step only, without any of the
    generate() machinery."""
    valid_keys = ['input_ids', 'attention_mask', 'pixel_values_videos', 'video_grid_thw', 
                  'video_second_per_grid', 'feature_attention_mask', 'input_features']
    clean_inputs = {k: v for k, v in inputs.items() if k in valid_keys}
    
    for k, v in clean_inputs.items():
        if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
            clean_inputs[k] = v.to(torch.bfloat16)

    outputs = model.thinker(
        **clean_inputs,
        past_key_values=past_key_values,
        position_ids=position_ids,
        use_cache=True,
        return_dict=True
    )
    
    next_logits = outputs.logits[:, -1, :]
    updated_cache = outputs.past_key_values
    
    return next_logits, updated_cache

def parse_arguments():
    """Parse command line arguments specifically for Qwen2.5-Omni Phi-Decoding"""
    class _NoPercentHelp(argparse.HelpFormatter):
        """Disable argparse's %-expansion of help text.

        argparse unconditionally applies `% params` to help strings, so a literal
        percent sign raises ValueError. This file uses no %(default)s, so turning
        the expansion off is safe.
        """
        def _expand_help(self, action):
            return self._get_help_string(action)

    parser = argparse.ArgumentParser(description="Phi-Decoding Algorithm for Qwen-Omni",
                                     formatter_class=_NoPercentHelp)

    # -- The three real module switches --
    _abl = parser.add_argument_group('module ablation switches')
    _abl.add_argument('--disable_dist', action='store_true',
                      help='Ablation: disable Dist-Confess. _dist_flag is always False, so no contrastive '
                           'resampling and no dist_ev accumulation. Because --layer_repair_gated defaults to on, '
                           'this also stops the Layer repair operator from firing (use --layer_repair_gated off '
                           'to decouple them). Layer penalty statistics are unaffected.')
    _abl.add_argument('--disable_layer', action='store_true',
                      help='Ablation: disable Layer-Confess. No hidden-state repair, and total_hallu_penalty is '
                           'always 0, which simultaneously disables the Path veto and the head-mode penalty '
                           'term.')
    _abl.add_argument('--disable_path', action='store_true',
                      help='Ablation: disable Path-Confess. No path is vetoed and advantage falls back to plain '
                           'logprob. Dist resampling and Layer repair still run; they simply no longer influence '
                           'path selection.')

    # Model configuration
    parser.add_argument('--model_id', type=str, default='qwen2.5-omni',
                        help='Model identifier')
    parser.add_argument('--model_path', type=str, default='Qwen/Qwen2.5-Omni-7B',
                        help='Model path')
    parser.add_argument('--gpus', type=int, default=2,
                        help='Number of GPUs to use (tensor_parallel is handled by device_map in transformers)')

    # Data configuration
    parser.add_argument('--datasets', type=str, default='cmm',
                        help='Dataset type: cmm, gsm, math, etc.')
    parser.add_argument('--data_root', type=str, default='./OmniHalluBench/cmm/',
                        help='Root path for multimodal data (videos/audios)')
    parser.add_argument('--data_path', type=str,
                        default='./OmniHalluBench/cmm/cmm.json',
                        help='Path to input json data')
    parser.add_argument('--output_dir', type=str,
                        default='./phi_results/base_phi_hook_observe/',
                        help='Output directory for results')

    parser.add_argument('--step_beam_size', type=int, default=2,
                        help='Beam size for each step')
    parser.add_argument('--num_rollout', type=int, default=4,
                        help='Number of rollouts')
    parser.add_argument('--num_foresight', type=int, default=3,
                        help='Number of foresight steps')
    parser.add_argument('--final_tiebreak', type=str, default='r5', choices=['off', 'r5'],
                        help="How to break a tie when the two final candidates disagree on the verdict. 'r5' "
                             '(default) takes the majority verdict over all second-stage completions and switches '
                             "to the candidate that agrees with it; an even split is left untouched. 'off' keeps "
                             'the legacy selector.')
    parser.add_argument('--strategy', type=str, default='cluster',
                        help='Response selection strategy')
    parser.add_argument('--width_pruning_strategy', type=str, default='low_sigma',
                        help='Width pruning strategy')
    parser.add_argument('--depth_pruning_strategy', type=str, default='cluster',
                        help='Depth pruning strategy')
    parser.add_argument('--cluster_num', type=int, default=2,
                        help='Number of clusters for clustering strategy')
    parser.add_argument('--threshold', type=float, default=0.7,
                        help='Threshold for early stopping')
    parser.add_argument('--least_foresight_num', type=int, default=2,
                        help='Minimum number of foresight steps before early stop')
    parser.add_argument('--sigma_rate', type=float, default=1.0,
                        help='Sigma rate for width pruning')

    # Execution configuration
    parser.add_argument('--record_process', type=bool, default=True,
                        help='Whether to record the decoding process')
    parser.add_argument('--file_name', type=str, default='omni_phi_test',
                        help='Output file name')
    parser.add_argument('--time_path', type=str,
                        default='./outputs/time/',
                        help='Path to save timing information')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--disable_per_sample_seed', action='store_true',
                        help='Ablation: seed once at initialisation instead of once per sample (makes results '
                             'depend on sample order).')
    parser.add_argument('--tree_vote_tau', type=float, default=0.20,
                        help='Judgement tasks: predict Yes when the fraction of Yes verdicts among the '
                             'Confess-tree second-stage completions exceeds this threshold. The default sits below '
                             '0.5 because the tree is systematically biased toward No.')
    parser.add_argument('--tree_vote_tau_map', type=str, default='',
                        help='Per-dataset thresholds, e.g. "cmm=0.22,phd=0.18,pubmedqa=0.14". Overrides '
                             '--tree_vote_tau when the current --datasets matches, and falls back to it otherwise. '
                             'Needed because the Yes prior differs widely across the judgement datasets.')
    parser.add_argument('--disable_tree_vote', action='store_true',
                        help='Ablation: disable tree voting; the final verdict is decided solely by '
                             'select_response.')
    parser.add_argument('--final_pool_yes_if_any', action='store_true',
                        help='Pool the two final candidates: if either one predicts yes, take that one. Judgement '
                             'datasets only, and one-directional. Default off.')
    parser.add_argument('--max_samples', type=int, default=-1,
                        help='Max samples to process (-1 = all)')
    parser.add_argument('--halo_prompt_version', type=str, default='v0',
                        help='HaloQuest prompt version: v0 (current), v1 (explicit format), v2 (sentence-prefix), v3 (few-shot template)')
    parser.add_argument('--halueval_prompt_version', type=str, default='v0',
                        help='HaluEval prompt version: v0 (baseline), v1 (yes/no fix + few-shot)')
    parser.add_argument('--avh_prompt_version', type=str, default='v0',
                        help='AVHBench prompt version: v0 (baseline), v1 (25w cap + English-only)')
    # -- Judgement-task (cmm/phd/pubmedqa) few-shot prompt version --
    parser.add_argument('--judge_prompt_version', type=str, default='v0',
                        choices=_JUDGE_PROMPT_VERSIONS,
                        help="Prompt version for cmm/phd/pubmedqa. 'v0' is the baseline; 'fs6' adds six "
                             "hand-written few-shot exemplars; 'mc', 'mc6', 'mc2', 'mc3', 'mc26', 'mc27' and "
                             "'mc28' are cmm-only modality-conditioned variants (see judge_prompts.py). Variants "
                             'that are undefined for phd/pubmedqa fall back to v0.')
    # -- Format control: enforce the output format at decoding time --
    parser.add_argument('--format_control', type=str, default='off',
                        choices=['off', 'template', 'anticopy', 'both'],
                        help='Decoding-time format constraint for the judgement branch; see format_control.py.')
    parser.add_argument('--anti_copy_n', type=int, default=6,
                        help='n-gram order for anti-copy: forbid continuations that overlap the prompt for n '
                             'consecutive tokens.')
    parser.add_argument('--fix_stop_strings', action='store_true',
                        help='Restore the stop_strings argument that safe_model_generate drops, so stop strings '
                             'actually take effect. Default off to keep existing arms reproducible.')
    parser.add_argument('--verdict_lock', action='store_true',
                        help='Restrict the second stage to emit exactly one token, Yes or No, so verdict parsing '
                             'can never fail. Changes the all_logprobs convention; default off.')
    # -- Task profile: keyed on measurable input properties, not dataset names --
    parser.add_argument('--task_profile', type=str, default='off', choices=['off', 'auto'],
                        help="'auto' selects the CD and length strategy from the answer space and reference "
                             "length; 'off' reproduces the baseline behaviour.")
    parser.add_argument('--cd_alpha', type=float, default=0.5,
                        help='Contrastive amplification strength for Dist-Confess.')
    parser.add_argument('--cd_resample', type=str, default='sample',
                        choices=['sample', 'argmax'],
                        help="How to re-select a token after CD correction. 'sample' draws from the multinomial "
                             "(default, stochastic); 'argmax' takes the most probable token.")
    parser.add_argument('--ref_len_words', type=float, default=None,
                        help='Median reference-answer length in words for this task. When supplied and below 15, '
                             '--task_profile auto enables length shrinkage.')
    parser.add_argument('--cd_mode', type=str, default='hard', choices=['hard', 'soft'],
                        help='Dist-Confess CD correction mode: hard (zero out token) or soft (scale down)')
    parser.add_argument('--cd_soft_factor', type=float, default=0.1,
                        help='Soft CD scaling factor (only used when cd_mode=soft)')
    parser.add_argument('--cd_plausibility_beta', type=float, default=0.0,
                        help='Adaptive plausibility constraint for contrastive decoding. When beta > 0, tokens '
                             'with p_full(v) < beta * max_w p_full(w) are masked out before amplification, which '
                             'stops the contrastive term from promoting special or punctuation tokens. 0.0 '
                             'disables it; 0.1 is recommended.')
    parser.add_argument('--cd_contrast_lambda', type=float, default=None,
                        help='Weight of the blind term in the contrastive objective: refined = logits_f - lambda * '
                             'blind_logits. Standard contrastive decoding uses lambda = 1. None keeps the legacy '
                             'formulation unchanged. Should be used together with --cd_plausibility_beta, which '
                             'bounds the tail noise that a larger lambda amplifies.')
    parser.add_argument('--cd_forbid_degenerate', action='store_true',
                        help='Exclude degenerate tokens (special tokens, pure punctuation, pure whitespace) from '
                             'the contrastive candidate set. The plausibility gate only thresholds on p_full and '
                             'does not catch these. Default off.')
    parser.add_argument('--cd_diag', action='store_true',
                        help='Emit one [CD-DIAG] JSON line per Dist event listing the top-3 candidates under a '
                             'grid of lambda and beta settings. Diagnostic only; does not change decoding '
                             'behaviour.')
    parser.add_argument('--path_score', type=str, default='head',
                        choices=['head', 'graded', 'graded_sym'],
                        help="Layer-to-Path penalty channel. 'head' reproduces the original behaviour, where the "
                             "penalty contributes a single veto bit. 'graded' uses a within-tree standardised "
                             "graded penalty and never vetoes. 'graded_sym' standardises within siblings carrying "
                             'the same verdict, removing Yes/No class bias. Non-head modes require '
                             '--veto_threshold 1e9.')
    parser.add_argument('--path_kappa', type=float, default=0.5,
                        help='Graded penalty strength: how many logprob standard deviations one penalty standard '
                             'deviation is worth. Scale-free, so it transfers across datasets.')
    parser.add_argument('--dump_probe_features', type=str, default='',
                        help='Directory for dumping candidate-level probe features (empty = off). For each '
                             'first-stage candidate it exports the mean intermediate-layer hidden state over that '
                             "candidate's own tokens, together with texts, logprob and is_vetoed, as "
                             '{dir}/{file_name}/{idx:06d}.npz. Collection only; the probe is trained offline.')
    parser.add_argument('--probe_layers', type=str, default='14,20,24',
                        help='Comma-separated, 0-indexed block outputs to use as probe features. The features are '
                             'concatenated in the given order.')
    parser.add_argument('--auditor_scope', type=str, default='last',
                        choices=['last', 'full', 'shadow'],
                        help='Which forward pass the layer-wise audit reads from. Each decoding step runs two to '
                             'three forward passes (full modality, video-masked, audio-masked) and the forward '
                             "hook overwrites its dictionary on every one of them, so 'last' (default, "
                             "byte-identical to previously reported runs) in fact reads the blind pass. 'full' "
                             'lets only the full-modality pass write the main dictionary, which changes the '
                             "decoding trajectory. 'shadow' keeps decoding byte-identical to 'last' while "
                             'additionally recording the full-modality values in a shadow dictionary, enabling '
                             'paired comparison within a single run.')
    parser.add_argument('--auditor_diag', action='store_true',
                        help='Use together with --auditor_scope full: print, at every step, the difference between '
                             'the full-modality and blind layer representations (whether the top-1 changes, the KL '
                             'divergence, and the cosine between h_final vectors).')
    parser.add_argument('--dump_layer_audit', action='store_true',
                        help='Use together with --auditor_scope shadow: record the raw layer-audit triples '
                             '(res_16_22, res_22_fin, l22_conf) along with token_id, keeping one copy from the '
                             'blind pass and one from the full-modality pass, into candidates[].layer_audit of the '
                             'TRAJ file. Because calculate_hallu_penalty is a pure function of those triples, this '
                             'lets all four of its thresholds be re-scored offline from a single run.')
    parser.add_argument('--path_kappa_dist', type=float, default=0.0,
                        help='Strength of the Dist-to-Path channel (0 = off, reproducing the original behaviour). '
                             'When enabled, the per-token prior-trap depth is standardised within the tree and '
                             'used as a second penalty term alongside the Layer penalty.')
    parser.add_argument('--path_kappa_grnd', type=float, default=0.0,
                        help='Strength of the continuous evidence-support reward channel for Path (0 = off, '
                             'reproducing the original behaviour). Rather than keeping only the single bit "MDI '
                             'below threshold", this uses the per-token log-likelihood ratio g_t = log p_full - '
                             'log p_blind (symmetrically clipped at +/-6 nat), aggregates it into a path score, '
                             'standardises that score within the sibling candidates of the same step, and adds it '
                             'as a bonus. It only ever adds score and never vetoes. The blind branch masks the '
                             'sensory input for multimodal data and the passage for text-only data, so the '
                             'quantity is defined on every dataset. Requires --path_score graded.')
    parser.add_argument('--path_grnd_field', type=str, default='all',
                        choices=['all', 'content', 'neg', 'late22', 'late16',
                                 'late22c', 'lp22', 'lp16'],
                        help='Which channel feeds the evidence reward. Masking-based channels, which need the '
                             "blind forward pass: 'all' uses g_t over all tokens, 'content' restricts it to "
                             "content words, and 'neg' accumulates only min(g, 0). Cross-layer channels, which "
                             "need no masking: 'late22' and 'late16' use d_t = log p_final(y_t) - log "
                             'p_22,16(y_t), where a large d indicates a token with no mid-stack support that the '
                             "final layer injected; 'late22c' is late22 restricted to content words; 'lp22' and "
                             "'lp16' are the bounded probability-difference counterparts. The sufficient "
                             'statistics of every channel are written to the TRAJ file, so the choice can be '
                             'settled offline.')
    parser.add_argument('--path_grnd_agg', type=str, default='sqrt',
                        choices=['sum', 'mean', 'sqrt', 'min', 'worst3'],
                        help="How to aggregate the per-token signal into a path score. 'sum' is the log Bayes "
                             "factor itself and favours long paths; 'mean' is length-independent but shrinks the "
                             "signal of long paths; 'sqrt' divides by sqrt(n), giving the best signal-to-noise "
                             "ratio (default); 'min' takes the extreme value, on the view that a single "
                             "unsupported assertion ruins the whole answer; 'worst3' averages the three worst "
                             'tokens as a robust version of min. Sibling candidates at a step differ in length, so '
                             'this choice changes path selection directly.')
    parser.add_argument('--layer_repair_project', type=str, default='perp',
                        choices=['perp', 'full'],
                        help="Projection used by the Layer-Confess repair. 'perp' is the norm-preserving "
                             'orthogonal projection of Proposition 2, which however removes the component along '
                             "h_final that is exactly what attenuating the prior requires. 'full' keeps the "
                             'parallel component and can genuinely weaken the final-layer prior prediction, at the '
                             'cost of no longer preserving the norm, so Proposition 2 does not apply to it.')
    parser.add_argument('--layer_diag', action='store_true',
                        help='Emit a diagnostic on every Layer repair: pass the hidden state before and after '
                             'repair through lm_head, record whether the argmax flips, and sweep the flip '
                             'threshold over a range of lambda values. lm_head is already available, so this adds '
                             'no extra forward pass.')
    parser.add_argument('--layer_fix_mode', type=str, default='linear',
                        choices=['linear', 'linear_norm', 'slerp'],
                        help='Layer-Confess hidden state repair mode')
    parser.add_argument('--path_cluster_mode', type=str, default='tfidf',
                        choices=['tfidf', 'sbert'],
                        help='Path-Confess clustering mode: tfidf or sbert')
    parser.add_argument('--disable_passage_mdi', action='store_true', default=False,
                        help='Disable passage-MDI for text-only datasets (halueval/pubmedqa/ragtruth)')
    parser.add_argument('--mdi_mask_mode', type=str, default='all',
                        choices=['all', 'audio_only', 'video_only', 'sequential'],
                        help='MDI mask mode for AVHBench: all (mask both), audio_only, video_only, '
                             'sequential (mask each separately, pick more anomalous)')
    parser.add_argument('--ablation_suite', type=str, default=None,
                        help='Path to ablation config JSON; runs multiple configs sequentially '
                             'without reloading model')
    parser.add_argument('--max_new_tokens_override', type=int, default=None,
                        help='Override max_new_tokens for specific dataset tuning')
    parser.add_argument('--repetition_penalty', type=float, default=1.0,
                        help='Repetition penalty for generation (1.0 = no penalty)')
    parser.add_argument('--audio_sr', type=int, default=24000,
                        help='Audio sample rate passed to the processor. The default of 24000 is retained so that '
                             'previously produced cmm and avhbench results stay reproducible, but it is incorrect: '
                             'the model ships a WhisperFeatureExtractor configured for 16000 and sampling_rate is '
                             'never passed to the processor, so 24000 causes a 9.73 s clip to be encoded as if it '
                             'were 14.60 s. Pass 16000 for the correct behaviour.')
    parser.add_argument('--use_audio_in_video', action='store_true',
                        help='Interleave audio and video in time following the official Qwen TMRoPE layout '
                             '(roughly one chunk every 2 s) instead of the default layout of one contiguous video '
                             'block followed by one contiguous audio block. The total token count is identical; '
                             'what changes is whether the model can tell which frame a given sound occurs in. '
                             'Default off, so all previously produced runs are unchanged. Enabling it also changes '
                             'the chat template, every processor call, the TMRoPE path in generate, and how '
                             'get_modality_indices locates modality spans.')

    # ---- Calibratable constants (E6 sensitivity sweep) -------------------
    # Each defaults to None → falls back to _DEFAULT_THRESHOLDS, i.e. the
    # original hardcoded literal, so omitting them reproduces prior runs.
    thr = parser.add_argument_group('calibratable thresholds')
    thr.add_argument('--res_22_fin', type=float, default=None,
                     help='Layer-Confess LATE detection threshold (default 1.35)')
    thr.add_argument('--l22_conf_late', type=float, default=None,
                     help='Layer-Confess LATE confidence gate (default 0.90)')
    thr.add_argument('--res_16_22', type=float, default=None,
                     help='Layer-Confess HIJACK detection threshold (default 1.40)')
    thr.add_argument('--l22_conf_hijack', type=float, default=None,
                     help='Layer-Confess HIJACK confidence gate (default 0.85)')
    thr.add_argument('--penalty_late', type=float, default=None,
                     help='Layer-Confess LATE base penalty (default 0.5)')
    thr.add_argument('--penalty_late_boost', type=float, default=None,
                     help='Layer-Confess LATE high-confidence surcharge (default 0.5)')
    thr.add_argument('--penalty_hijack', type=float, default=None,
                     help='Layer-Confess HIJACK penalty (default 0.6)')
    thr.add_argument('--mdi_threshold', type=float, default=None,
                     help='Dist-Confess CD trigger on MDI (default -0.3)')
    thr.add_argument('--layer_repair_alpha', type=float, default=None,
                     help='Layer-Confess hidden interpolation weight (default 0.3)')
    thr.add_argument('--veto_threshold', type=float, default=None,
                     help='Path-Confess veto cutoff on cumulative penalty (default 0.45)')

    # ---- Layer-Confess hidden-state repair (P0) --------------------------
    parser.add_argument('--layer_repair', type=str, default='none',
                        choices=['none', 'paper', 'linear', 'linear_norm', 'slerp'],
                        help="Layer-Confess hidden-state repair operator. "
                             "'none' (default) reproduces all previously reported runs — "
                             "no hidden repair is applied. 'paper' implements Eq.(layer-repair): "
                             "orthogonal projection against the weighted cross-layer support "
                             "representation. 'linear_norm'/'slerp' are the legacy interpolation "
                             "variants (equivalent to the old --layer_fix_mode).")
    parser.add_argument('--lambda_layer', type=float, default=0.3,
                        help='Repair strength lambda_L in Eq.(layer-repair) (--layer_repair paper)')
    parser.add_argument('--layer_repair_gated', type=str, default='on',
                        choices=['on', 'off'],
                        help="Whether the hidden-state repair is conditionally invoked. "
                             "'on' (default, matches method_layer_confess.tex: the repair "
                             "'is conditionally invoked only on steps flagged by Dist-Confess') "
                             "applies it exclusively on Dist-Confess-flagged steps; 'off' applies "
                             "it at every decoding step. Only meaningful with --layer_repair != none.")
    parser.add_argument('--anchor_layers', type=str, default='16,22',
                        help='Comma-separated anchor layer indices for Layer-Confess (default 16,22)')
    parser.add_argument('--omega_weights', type=str, default=None,
                        help='Comma-separated barycenter weights omega_l over anchor layers; '
                             'must sum to 1 and match --anchor_layers (default: uniform)')
    parser.add_argument('--layer_lambda_grid', type=str, default=None,
                        help='Marginalise over the repair strength: a comma-separated grid such as "0,0.5,2,5,10". '
                             'When supplied, Layer stops reprojecting with a single --lambda_layer and instead '
                             'treats lambda_L as a nuisance parameter, averaging the resulting distributions. '
                             'lm_head is already available, so this adds no extra forward pass. Default off.')

    _a = parser.parse_args()

    # -- Argument consistency checks --
    if getattr(_a, 'path_score', 'head') != 'head':
        _vt = getattr(_a, 'veto_threshold', None)
        if _vt is None or _vt < 1e8:
            parser.error(
                f"--path_score {_a.path_score} also requires --veto_threshold 1e9 to disable the veto"
                f"(currently {_vt}). The veto overwrites the logprob of a vetoed path with -10000, "
                f"so the graded recomputation cannot recover the original value.")
    if getattr(_a, 'cd_contrast_lambda', None) is not None \
            and _a.cd_contrast_lambda > 0.5 and getattr(_a, 'cd_plausibility_beta', 0.0) <= 0.0:
        parser.error(
            f"--cd_contrast_lambda {_a.cd_contrast_lambda} > 0.5 requires --cd_plausibility_beta "
            f"(0.1 recommended). The contrastive term -lambda*blind_logits rewards the tokens least likely under the blind branch, "
            f"so without the plausibility cutoff it promotes special and punctuation tokens into the candidate pool.")
    if getattr(_a, 'path_kappa_dist', 0.0) > 0.0 \
            and getattr(_a, 'path_score', 'head') == 'head':
        parser.error(
            f"--path_kappa_dist {_a.path_kappa_dist} also requires --path_score graded or graded_sym. "
            f"In head mode the advantage is decided by the veto, leaving nowhere to consume Dist evidence.")
    if getattr(_a, 'path_kappa_grnd', 0.0) > 0.0 \
            and getattr(_a, 'path_score', 'head') == 'head':
        parser.error(
            f"--path_kappa_grnd {_a.path_kappa_grnd} also requires --path_score graded or graded_sym. "
            f"In head mode the advantage is decided by the veto, leaving nowhere for a bonus term.")
    return _a


def softmax(x):
    """
    Compute softmax values for the input array with numerical stability.
    """
    x = np.array(x)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

class PhiDecoder:
    """
    Main class for phi-decoding algorithm implementation.
    Combines clustering and sampling strategies for response selection.
    """
    def __init__(self, args):
        """
        Initialize the decoder
        Args:
            args: Command line arguments containing configuration
        """
        self.args = args
        self.halo_prompt_version = getattr(args, 'halo_prompt_version', 'v0')
        self.halueval_prompt_version = getattr(args, 'halueval_prompt_version', 'v0')
        self.avh_prompt_version = getattr(args, 'avh_prompt_version', 'v0')
        self.judge_prompt_version = getattr(args, 'judge_prompt_version', 'v0')
        self.model = None
        self.processor = None
        self.tokenizer = None
        self._sbert_model = None  # lazy-loaded for path_cluster_mode=sbert
        self.initialize_model()

    @property
    def thresholds(self):
        """Calibratable constants, rebuilt from self.args on each access.

        Read lazily rather than cached because run_ablation_suite() mutates
        self.args between configs.
        """
        return thresholds_from_args(self.args)

    @property
    def anchor_layers(self):
        """Parsed --anchor_layers, e.g. '16,22' -> [16, 22]."""
        raw = getattr(self.args, 'anchor_layers', '16,22') or '16,22'
        return [int(x) for x in str(raw).split(',') if str(x).strip()]

    @property
    def omega_weights(self):
        """Barycenter weights over anchor layers; uniform when unspecified."""
        raw = getattr(self.args, 'omega_weights', None)
        layers = self.anchor_layers
        if not raw:
            return [1.0 / len(layers)] * len(layers)
        w = [float(x) for x in str(raw).split(',') if str(x).strip()]
        if len(w) != len(layers):
            raise ValueError(
                f"--omega_weights has {len(w)} entries but --anchor_layers has {len(layers)}")
        total = sum(w)
        if total <= 0:
            raise ValueError("--omega_weights must sum to a positive value")
        return [x / total for x in w]

    @property
    def _layer_lambda_grid(self):
        """Parse and cache --layer_lambda_grid. Returns None when unset or single-valued,
        in which case the single-lambda path is used."""
        cached = getattr(self, '_llg_cache', 'unset')
        if cached != 'unset':
            return cached
        raw = getattr(self.args, 'layer_lambda_grid', None)
        grid = None
        if raw:
            g = [float(x) for x in str(raw).split(',') if str(x).strip()]
            if len(g) > 1:
                grid = g
                print(f"    ℹ️ [Layer-lambda] grid {g} ({len(g)} lm_head calls per repair point, "
                      f"no extra forward pass)", flush=True)
            elif len(g) == 1:
                print(f"    ⚠️ [Layer-lambda] grid has a single value {g}; falling back to the single-lambda path", flush=True)
        self._llg_cache = grid
        return grid

    def task_profile(self, system_prompt):
        """Derive which CD and length branch this task should use from measurable
        properties of the input.

        There are only two properties, both public parts of the task specification
        rather than test-set labels:

          answer_space : 'binary' if and only if the system prompt asks for a closed
                         binary verdict such as \\boxed{Yes} / \\boxed{No};
                         otherwise 'open'.
          ref_len      : --ref_len_words, the median reference-answer length.

        Binary answers need separate handling because in hard mode the CD correction
        zeroes the probability of the flagged token and resamples. When the answer
        space is just {Yes, No}, banning Yes is mathematically equivalent to forcing
        No, so triggering the correction flips the answer regardless of how strong
        the contrastive evidence actually is; in practice the bias simply changes
        direction while overall accuracy barely moves. Using soft mode with argmax
        lets the flagged token still win, so the answer only changes when the
        blind-branch contrast genuinely outweighs it.

        Returns a dict; with task_profile='off' every value is None and the caller
        falls back to the original behaviour.
        """
        if getattr(self.args, 'task_profile', 'off') != 'auto':
            return {'answer_space': None, 'cd_mode': None, 'cd_resample': None,
                    'len_budget': None}

        sp = system_prompt or ''
        binary = ('\\boxed{Yes}' in sp) or ('\\boxed{yes}' in sp)

        ref_len = getattr(self.args, 'ref_len_words', None)
        len_budget = None
        if ref_len is not None and ref_len < 15:
            len_budget = float(ref_len)

        return {
            'answer_space': 'binary' if binary else 'open',
            'cd_mode':     'soft'   if binary else None,
            'cd_resample': 'argmax' if binary else None,
            'len_budget':  len_budget,
        }

    def initialize_model(self):
        """Initialize the Qwen2.5-Omni model and multimodal processor"""
        model_path = self._get_model_path()
        
        self.processor = Qwen2_5OmniProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.tokenizer = self.processor.tokenizer

        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        ).eval()

        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)

    def _get_model_path(self):
        """Get the appropriate model path"""
        return self.args.model_path

    def _vectorize_texts(self, texts):
        """Vectorize texts using tfidf or sbert based on args.path_cluster_mode."""
        mode = getattr(self.args, 'path_cluster_mode', 'tfidf')
        if mode == 'sbert':
            if not _SBERT_AVAILABLE:
                print("[WARN] sentence-transformers not installed, falling back to tfidf", flush=True)
            else:
                if self._sbert_model is None:
                    self._sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
                return self._sbert_model.encode(texts)
        vectorizer = TfidfVectorizer()
        return vectorizer.fit_transform(texts)

    def get_system_prompt(self, dataset_type, question=None):
        """Structured system prompt for Qwen-Omni.

        Covers containment matching, the CMM binary verification task, the AVH
        single-sentence description task, and the stop-token and format constraints.

        `question` is only used by prompt versions that branch on the question
        itself. Omitting it, or passing it to a version that registers no branch,
        reproduces the original behaviour exactly. The modality is derived
        deterministically from the question and never looks at the ground truth; if
        it cannot be determined, that version's static text is used, since leaving
        the prompt unchanged is preferable to guessing the wrong modality.
        """
        omni_identity = "You are Qwen, a virtual human developed by Alibaba Group, capable of perceiving auditory and visual inputs."

        if "cmm" in dataset_type.lower():
            if _JUDGE_PROMPTS_LOADED:
                _jp = _get_judge_prompt('cmm', getattr(self, 'judge_prompt_version', 'v0'),
                                        question=question)
                if _jp:
                    return _jp
            return (
                "You are an expert audio-visual reporter. Your task is to provide a single-sentence narrative "
                "combining what you see and hear, followed by a final judgment.\n\n"
                "STRICT OUTPUT FORMAT:\n"
                "Description: [One flowing sentence describing the scene and the EXACT sounds heard, like: 'In a room, a man spoke by a window where birds were seen but not heard.']\n"
                "Judgment: \\boxed{Yes} or \\boxed{No}\n\n"
                "STRICT AUDIT RULES:\n"
                "1. If a visual object (like a bird) is present but its specific sound (chirping) is NOT in the audio, you MUST state that it was not heard in the Description.\n"
                "2. The Description must be a single, descriptive sentence.\n"
                "3. Judgment must be based ONLY on what is audible."
            )
        elif "halo" in dataset_type.lower():
            hpv = getattr(self, 'halo_prompt_version', 'v0')
            if _HALO_PROMPTS_LOADED and hpv in ('v4', 'v5', 'v6', 'v7', 'v8', 'v9'):
                sys_prompt, _, _ = _get_halo_prompt(hpv)
                return sys_prompt
            elif hpv == 'v1':
                return (
                    "You are an expert visual assistant. Answer the question based ONLY on what you see in the image.\n\n"
                    "OUTPUT RULES — read carefully:\n"
                    "1. Write EXACTLY ONE complete sentence (minimum 8 words).\n"
                    "2. NEVER use \\boxed{}, brackets, or single-word/number answers.\n"
                    "3. If the object/detail is visible: 'The [object] is [description].'\n"
                    "4. If the object/detail is NOT visible or absent: 'There is no [object] visible in this image.' or 'The [detail] is not visible in this image.'\n"
                    "5. If the image is ambiguous: 'It is not possible to determine [X] from this image.'\n"
                    "Be strictly factual. Do not hallucinate."
                )
            elif hpv == 'v2':
                return (
                    "You are an expert visual assistant. Answer the question based ONLY on the provided image.\n"
                    "CRITICAL: If the question asks about something NOT present, explicitly state it is absent.\n"
                    "Answer in a complete, natural sentence. Be strictly factual and do not hallucinate."
                )
            elif hpv == 'v3':
                return (
                    "You are an expert visual assistant. Answer the question based ONLY on what you see in the image.\n\n"
                    "STRICT FORMAT: One complete sentence. No \\boxed{}. No single words or numbers alone.\n\n"
                    "EXAMPLES of correct answers:\n"
                    "- Q: 'How many cats?' (none visible) → 'There are no cats visible in this image.'\n"
                    "- Q: 'How many cats?' (two visible) → 'There are two cats visible in this image.'\n"
                    "- Q: 'What color is the hat?' (hat present) → 'The hat in the image is red.'\n"
                    "- Q: 'What color is the hat?' (no hat) → 'There is no hat visible in this image.'\n"
                    "- Q: 'What is written on the sign?' (unreadable) → 'The text on the sign is not legible in this image.'\n"
                    "Always produce a sentence of this form. Never output just a number or a color word."
                )
            else:  # v0 — original
                return (
                    "You are an expert visual assistant. Your task is to answer the question based ONLY on the provided image.\n"
                    "CRITICAL: If the question asks about a detail or object that is NOT present in the image, you MUST explicitly state that it is not present or visible.\n"
                    "Answer in a complete, natural sentence (e.g. 'The girl is wearing red sneakers.' or 'There is no cat visible in the image.'). "
                    "Be strictly factual and do not hallucinate."
                )

        elif "avh" in dataset_type.lower():
            apv = getattr(self, 'avh_prompt_version', 'v0')
            if _AVH_PROMPTS_LOADED:
                sys_prompt, _, _ = _get_avh_prompt(apv)
                return sys_prompt
            # fallback: v0 hardcoded
            return (
                "You are an objective audiovisual forensic analyst. Your task is to decode the video and audio streams into a high-density, expert-level description in ONE SHORT sentence.\n"
                "CRITICAL ANTI-HALLUCINATION DIRECTIVES:\n"
                "1. MODALITY INDEPENDENCE: Analyze what you SEE and HEAR independently. DO NOT let the visual scene change your interpretation of the audio, and vice versa.\n"
                "2. NO SEMANTIC SMOOTHING: If visuals and audio seem contradictory (e.g., a forest visual but an engine sound), report BOTH exactly as they are. Do not invent details like 'chirping birds' to make it natural.\n"
                "3. AVOID ASSUMPTIONS: Rely on audio to confirm ambiguous visuals (e.g., a 'baaing' sound confirms a goat, do not assume it is a cat).\n\n"
                "FORMATTING INSTRUCTIONS:\n"
                "1. Compression: Pack all objective sensory details into a maximum of 50 words.\n"
                "2. Directness: Skip all introductory phrases. Start immediately with the main subject.\n"
                "3. Keywords: Prioritize specific nouns and adjectives.\n"
                "CORE RULE: Be extremely dense, strictly factual, and under 50 words. Output a plain sentence with NO formatting wrappers."
            )
        
        elif "phd" in dataset_type.lower():
            if _JUDGE_PROMPTS_LOADED:
                _jp = _get_judge_prompt('phd', getattr(self, 'judge_prompt_version', 'v0'))
                if _jp:
                    return _jp
            return (
                "You are a scientific visual analyst. Your task is to evaluate a specific hypothesis against raw visual data.\n\n"
                "EVALUATION RULES:\n"
                "1. Treat the provided text context as an unverified background story. It may contain errors.\n"
                "2. Treat the final question as a 'Hypothesis' that needs to be tested.\n"
                "3. Look for explicit visual confirmation or contradiction in the image. Do not rely on assumptions or the text's claims.\n\n"
                "STRICT OUTPUT FORMAT:\n"
                "Hypothesis to Test: [Restate the core question simply]\n"
                "Image Observation:\n- [Observation 1]\n- [Observation 2]\n"
                "Judgment: \\boxed{Yes} or \\boxed{No}"
            )

        elif "halueval" in dataset_type.lower():
            hepv = getattr(self, 'halueval_prompt_version', 'v0')
            if _HALUEVAL_PROMPTS_LOADED:
                sys_prompt, _, _ = _get_halueval_prompt(hepv)
                return sys_prompt
            # fallback: v0 hardcoded
            return (
                "You will be presented with a question.\n"
                "Answer the user's question strictly based on the given information.\n"
                "Do not make up information.\n"
                "Output ONLY the answer as a SHORT PHRASE (1-5 words). No full sentences, no explanation."
            )
        elif "ragtruth" in dataset_type.lower():
            return (
                "You are given passages and a question. Follow these steps:\n"
                "Answer the question using only the information from the given passages.\n"
                " - Include specific examples, numbers, or comparisons if mentioned.\n"
                " - Include all details that support the answer.\n"
                " - Do not add external information.\n"
                "Give your answer directly without any prefix or format wrapper."
            )
        elif "pubmedqa" in dataset_type.lower():
            if _JUDGE_PROMPTS_LOADED:
                _jp = _get_judge_prompt('pubmedqa', getattr(self, 'judge_prompt_version', 'v0'))
                if _jp:
                    return _jp
            return (
                "You will be given a PubMed-style passage and a Yes/No/Maybe question.\n"
                "Answer rules:\n"
                "1. Begin with exactly one of: Yes. / No. / Maybe.\n"
                "2. Then add ONE short sentence (≤25 words) summarizing the key conclusion from the passage.\n"
                "3. Preserve key medical terms from the passage; do not replace them with synonyms.\n"
                "4. Do NOT add recommendations, explanations, or new information.\n"
                "Example format: Yes. The study found that treatment X significantly reduced symptom Y in patients with Z."
            )
        elif "summedits" in dataset_type.lower():
            return (
                "You are given a document and a summary. Your task is to determine whether the summary is "
                "factually consistent with the document. Answer only 'Yes' if the summary is consistent, "
                "or 'No' if it contains any factual inconsistency."
            )

        return f"{omni_identity}\nSolve the problem step by step based on the video/audio."

    def cluster_responses(self, responses, advantages):
        """
        Cluster responses using TF-IDF and K-means.
        Identify the 'consensus' reasoning paths.
        """
        valid_indices = [i for i, r in enumerate(responses) if r.strip()]
        if len(valid_indices) < self.args.cluster_num:
            return None, {"state": "cannot cluster", "reason": "too few valid responses"}

        try:
            valid_responses = [responses[i] for i in valid_indices]

            X = self._vectorize_texts(valid_responses)

            k = min(self.args.cluster_num, len(valid_responses))
            kmeans = KMeans(n_clusters=k, n_init=10, random_state=self.args.seed)
            kmeans.fit(X)

            clusters = [[] for _ in range(k)]
            for idx, label in enumerate(kmeans.labels_):
                clusters[label].append(valid_indices[idx])

            return clusters, {
                "state": "success",
                "cluster_sizes": [len(c) for c in clusters]
            }

        except Exception as e:
            return None, {"state": "fail", "error": str(e)}

    # ==================== Degenerate token set (--cd_forbid_degenerate) ====================
    def _cd_degenerate_ids(self):
        """Return the ids of degenerate tokens: special tokens, pure punctuation or
        formatting symbols, and pure whitespace.

        The contrastive rewrite logits_f - lambda * blind_logits rewards the tokens
        the blind model finds least likely, and those are exactly these
        semantically empty symbols: mid-sentence, a blind model puts almost no mass
        on <|endoftext|> or a run of asterisks, so their contrastive score is pushed
        very high. The plausibility gate thresholds on p_full alone and does not
        catch them, since formatting symbols are not improbable under full context.

        Computed once and cached on the instance; a full vocabulary pass is cheap.
        """
        cached = getattr(self, '_cd_bad_ids', None)
        if cached is not None:
            return cached
        try:
            tok = self.processor.tokenizer
            bad = set(int(i) for i in (tok.all_special_ids or []))
            vocab = tok.get_vocab()
            for _s, _i in vocab.items():
                _d = tok.convert_tokens_to_string([_s])
                if _d == '' or _d.isspace():
                    bad.add(int(_i))
                    continue
                _core = _d.strip()
                if _core and not any(ch.isalnum() for ch in _core):
                    bad.add(int(_i))
            ids = torch.tensor(sorted(bad), dtype=torch.long, device=self.model.device)
            print(f"    ℹ️ [CD-Forbid] degenerate tokens {ids.numel()} / {len(vocab)}", flush=True)
        except Exception as e:
            print(f"    ⚠️ [CD-Forbid] construction failed; no exclusion applied: {e}", flush=True)
            ids = torch.empty(0, dtype=torch.long, device=self.model.device)
        self._cd_bad_ids = ids
        return ids

    # ==================== Confess tree voting ====================
    _TV_JUDGE_SETS = ("cmm", "phd", "pubmedqa")

    @staticmethod
    def _tv_verdict(text):
        """Return the last explicit yes/no in the text, or None. Does not fall back to No."""
        s = re.sub(r"\\boxed\{?", " ", str(text).lower())
        m = re.findall(r"\b(yes|no)\b", s)
        return m[-1] if m else None

    def _tv_tau(self):
        """Threshold to use for the current dataset. --tree_vote_tau_map overrides the
        global --tree_vote_tau when it matches."""
        _ds = str(getattr(self.args, 'datasets', '')).lower()
        for _kv in str(getattr(self.args, 'tree_vote_tau_map', '') or '').split(','):
            _kv = _kv.strip()
            if not _kv or '=' not in _kv:
                continue
            _k, _v = _kv.split('=', 1)
            if _k.strip().lower() == _ds:
                return float(_v)
        return float(getattr(self.args, 'tree_vote_tau', 0.20))

    def _tree_vote(self, selected_response):
        """Decide the final verdict on judgement tasks from the second-stage consensus of
        the whole Confess tree.

        The selected beam is only one path through the tree, and the tree as a whole
        carries a systematic bias toward No, so reading the verdict off that single
        beam inherits the bias. Reading the fraction of Yes verdicts across the tree
        and thresholding it instead removes much of that bias.

        The threshold is not 0.5 and is not tuned on the evaluation set: it is
        calibrated leave-one-dataset-out, and the useful range is a plateau rather
        than a sharp peak.

        When the verdict flips, the response text is not fabricated: the
        highest-logprob second-stage completion that already agrees with the
        consensus is taken from the tree as supporting evidence, and an explicit
        verdict line is appended.
        """
        info = {'applied': False}
        if getattr(self.args, 'disable_tree_vote', False):
            return info
        if str(getattr(self.args, 'datasets', '')).lower() not in self._TV_JUDGE_SETS:
            return info
        pool = getattr(self, '_tv_pool', None) or []
        votes = [(t, lp, self._tv_verdict(t)) for t, lp in pool]
        y = sum(1 for _, _, v in votes if v == 'yes')
        n = sum(1 for _, _, v in votes if v == 'no')
        if y + n == 0:
            return info
        frac = y / (y + n)
        tau = self._tv_tau()
        target = 'yes' if frac > tau else 'no'
        info.update({'yes': y, 'no': n, 'frac': round(frac, 4), 'tau': tau,
                     'target': target, 'beam_verdict': self._tv_verdict(selected_response)})
        if info['beam_verdict'] == target:
            return info
        matching = [(t, lp) for t, lp, v in votes if v == target]
        if not matching:
            info['fallback'] = 'no_matching_text'
            return info
        evidence = max(matching, key=lambda x: x[1])[0]
        info['applied'] = True
        info['response'] = (f"{str(evidence).strip()}\n\n"
                            f"Judgment: \\boxed{{{'Yes' if target == 'yes' else 'No'}}}")
        print(f"    [TREE-VOTE] beam verdict {info['beam_verdict']}, tree Yes {y}/{y+n}"
              f"={frac:.2f} > τ={tau} => switching to {target}", flush=True)
        return info

    def select_response(self, responses, logprobs, advantages):
        """Trust-weighted clustering.

        Combines the consensus size of a cluster with its mean internal advantage
        (confidence), so that a large but low-confidence cluster does not
        automatically win.
        """
        if self.args.strategy == "advantage":
            valid_indices = [idx for idx, r in enumerate(responses) if r.strip()]
            if len(valid_indices) == 0:
                weights = softmax([-adv / TEMPERATURE for adv in advantages])
                return np.random.choice(len(advantages), p=weights)
            weights = softmax([advantages[i] / TEMPERATURE for i in valid_indices])
            return np.random.choice(valid_indices, p=weights)

        if self.args.strategy == "cluster":
            valid_indices = [idx for idx, r in enumerate(responses) if r.strip()]

            if len(valid_indices) == 0:
                print('Warning: All final responses are empty. Sampling by negative advantage.')
                weights = softmax([-adv/TEMPERATURE for adv in advantages])
                return np.random.choice(len(advantages), p=weights)

            if len(valid_indices) < self.args.cluster_num:
                print('Warning: Too few valid responses for clustering. Sampling by advantage.')
                weights = softmax([advantages[i]/TEMPERATURE for i in valid_indices])
                return np.random.choice(valid_indices, p=weights)

            try:
                valid_responses = [responses[i] for i in valid_indices]
                valid_advantages = [advantages[i] for i in valid_indices]

                X = self._vectorize_texts(valid_responses)
                
                k = self.args.cluster_num
                kmeans = KMeans(n_clusters=k, n_init=10, random_state=self.args.seed)
                kmeans.fit(X)
                cluster_labels = kmeans.labels_

                cluster_list = [[] for _ in range(k)]
                for idx, label in enumerate(cluster_labels):
                    cluster_list[label].append(idx)
                
                cluster_list = [c for c in cluster_list if len(c) > 0]
                
                # ==========================================================
                # ==========================================================
                ALPHA = 0.5
                total_valid = len(valid_indices)
                
                cluster_size_ratios = [len(c) / total_valid for c in cluster_list]
                
                cluster_avg_advs = [np.mean([valid_advantages[idx] for idx in c]) for c in cluster_list]
                cluster_adv_scores = softmax([adv / TEMPERATURE for adv in cluster_avg_advs])
                
                cluster_scores = []
                for i in range(len(cluster_list)):
                    score = ALPHA * cluster_size_ratios[i] + (1.0 - ALPHA) * cluster_adv_scores[i]
                    cluster_scores.append(score)
                    
                best_cluster_idx = int(np.argmax(cluster_scores))
                target_cluster = cluster_list[best_cluster_idx]
                
                print(f"    [CLUSTER DEBUG] cluster sizes: {[len(c) for c in cluster_list]}, "
                      f"size scores: {[round(s,3) for s in cluster_size_ratios]}, "
                      f"quality scores: {[round(s,3) for s in cluster_adv_scores]} "
                      f"-> selected cluster {best_cluster_idx}", flush=True)

                cluster_adv_list = [valid_advantages[idx] for idx in target_cluster]
                weights = softmax([adv/TEMPERATURE for adv in cluster_adv_list])
                
                selected_inner_idx = np.random.choice(len(target_cluster), p=weights)
                selected_valid_idx = target_cluster[selected_inner_idx]
                
                return valid_indices[selected_valid_idx]

            except Exception as e:
                print(f'Clustering failed: {e}. Falling back to advantage-based sampling.', flush=True)
                weights = softmax([advantages[i]/TEMPERATURE for i in valid_indices])
                return np.random.choice(valid_indices, p=weights)

        else:
            raise ValueError(f"Unknown strategy: {self.args.strategy}")
        
    def process_example(self, example, system_prompt):
        """Pipeline controller for a single multimodal example."""
        import os

        token_stats = {"input": 0, "output": 0}
        rollout_stats = {"total": 0, "saved": 0}

        self._tv_pool = []

        traj_pool = [[] for _ in range(self.args.num_foresight)]
        step_pool = [[] for _ in range(self.args.num_foresight)]
        prob_pool = [[] for _ in range(self.args.num_foresight + 1)]
        adv_pool = [[] for _ in range(self.args.num_foresight + 1)]

        v_rel = example.get('video') or example.get('video_path') or ""
        a_rel = example.get('audio') or example.get('audio_path') or ""
        i_rel = example.get('image') or example.get('image_path') or ""
        
        # ==========================================================
        # ==========================================================
        json_dir = os.path.dirname(self.args.data_path)
        
        def resolve_media_path(rel_path):
            if not rel_path: return ""
            clean_rel = rel_path.lstrip('./')
            
            path1 = os.path.abspath(os.path.join(json_dir, clean_rel))
            if os.path.exists(path1): return path1
            
            path2 = os.path.abspath(os.path.join(self.args.data_root, clean_rel))
            if os.path.exists(path2): return path2
            
            path3 = os.path.abspath(os.path.join(self.args.data_root, "PhD", clean_rel))
            if os.path.exists(path3): return path3
            
            return path1

        v_path = resolve_media_path(v_rel)
        a_path = resolve_media_path(a_rel)
        i_path = resolve_media_path(i_rel)
        # ==========================================================

        pixel_values = None
        if v_rel and os.path.exists(v_path):
            pixel_values = load_video_frames(v_path, num_frames=8)
            print(f"✅ [MEDIA LOAD] video loaded: {v_path}", flush=True)
            
        audio_values = None
        if a_rel and os.path.exists(a_path):
            audio_values = load_audio_data(a_path)
            print(f"✅ [MEDIA LOAD] audio loaded: {a_path}", flush=True)

        image_values = None
        if i_rel:
            if not os.path.exists(i_path):
                if 'val2014' in i_path:
                    i_path = i_path.replace('val2014', 'train2014')
                elif 'train2014' in i_path:
                    i_path = i_path.replace('train2014', 'val2014')

            if os.path.exists(i_path):
                image_values = load_image(i_path)
                print(f"✅ [MEDIA LOAD] image loaded: {i_path}", flush=True)
            else:
                print(f"❌ [MEDIA ERROR] invalid image path: {i_path}", flush=True)

        example['pixel_values'] = pixel_values
        example['audio_values'] = audio_values
        example['image_values'] = image_values
        
        pixel_values = None
        if v_rel:
            if os.path.exists(v_path):
                pixel_values = load_video_frames(v_path, num_frames=8)
                print(f"✅ [MEDIA LOAD] video loaded: {v_path}", flush=True)
            else:
                print(f"❌ [MEDIA ERROR] invalid video path: {v_path}", flush=True)
                
        audio_values = None
        if a_rel:
            if os.path.exists(a_path):
                audio_values = load_audio_data(a_path)
                print(f"✅ [MEDIA LOAD] audio loaded: {a_path}", flush=True)
            else:
                print(f"❌ [MEDIA ERROR] invalid audio path: {a_path}", flush=True)

        example['pixel_values'] = pixel_values
        example['audio_values'] = audio_values

        if "avh" in self.args.datasets.lower():
            previous_steps = []
            for _b in range(self.args.step_beam_size):
                if _b % 2 == 0:
                    previous_steps.append("Audio observation:\n\n")
                else:
                    previous_steps.append("Visual observation:\n\n")
        else:
            previous_steps = ["Perceptual analysis:\n\n" for _ in range(self.args.step_beam_size)]
        previous_values = [0.0 for _ in range(self.args.step_beam_size)]

        if "reclor" in self.args.datasets or "logiqa" in self.args.datasets:
            q_text = f"Passage: {example['context']}\nQuestion: {example['question']}\n" + \
                     f"A. {example['answers'][0]}\nB. {example['answers'][1]}\n" + \
                     f"C. {example['answers'][2]}\nD. {example['answers'][3]}"
            gt = example.get('label')
        else:
            q_text = example.get('input') or example.get('question')
            gt = example.get('target') or example.get('answer')

        traj_info = {
            'question_idx': example.get('id', 0),
            'question': q_text,
            'ground_truth': gt,
            'media_info': {
                'has_video': pixel_values is not None, 
                'has_audio': audio_values is not None,
                'v_path': v_path if pixel_values is not None else "NOT_FOUND",
                'a_path': a_path if audio_values is not None else "NOT_FOUND"
            },
            'foresight_part': [],
            'final_part': {},
            # Path-Confess audit counters — accumulated across all foresight steps
            'path_confess_stats': {
                'num_paths_explored': 0,
                'num_paths_vetoed': 0,
                'max_penalty_seen': 0.0,
                'beam_shrink_occurred': False,
                'cluster_fallback_occurred': False,
            },
            'config': {
                'num_rollout': self.args.num_rollout,
                'num_foresight': self.args.num_foresight,
                'step_beam_size': self.args.step_beam_size,
                'strategy': self.args.strategy,
                'width_pruning_strategy': self.args.width_pruning_strategy,
                'depth_pruning_strategy': self.args.depth_pruning_strategy,
                'threshold': self.args.threshold,
                'sigma_rate': self.args.sigma_rate,
                'cluster_num': self.args.cluster_num
            }
        }

        import time as _time
        _t_foresight_start = _time.perf_counter()
        for step in range(self.args.num_foresight):
            step_results = self._process_step(
                example,
                system_prompt,
                previous_steps,
                previous_values,
                token_stats,
                rollout_stats,
                traj_info
            )

            if self._should_stop_early(step_results, step):
                break

            next_steps = step_results["next_steps"]
            next_values = step_results["next_values"]
            # BUG FIX: pad to step_beam_size to prevent IndexError at next _process_step call
            while len(next_steps) < self.args.step_beam_size:
                next_steps.append(next_steps[-1] if next_steps else "Perceptual analysis:\n\n")
                next_values.append(next_values[-1] if next_values else 0.0)
            previous_steps = next_steps
            previous_values = next_values

            traj_pool[step] = step_results["trajectories"]
            step_pool[step] = step_results["steps"]
            prob_pool[step] = step_results["logprobs"]
            adv_pool[step] = step_results["advantages"]
        _t_foresight_end = _time.perf_counter()

        _t_final_start = _time.perf_counter()
        final_result = self._generate_final_response(
            example,
            system_prompt,
            previous_steps,
            previous_values,
            token_stats,
            rollout_stats,
            traj_info
        )
        _t_final_end = _time.perf_counter()

        traj_info['token_num'] = token_stats["input"] + token_stats["output"]
        # Timing breakdown for profiling
        traj_info['timing'] = {
            'foresight_seconds': round(_t_foresight_end - _t_foresight_start, 4),
            'final_generation_seconds': round(_t_final_end - _t_final_start, 4),
        }

        return {
            "response": final_result["response"],
            "token_stats": token_stats,
            "rollout_stats": rollout_stats,
            "trajectories": {
                "steps": step_pool,
                "probs": prob_pool,
                "advantages": adv_pool,
                "final": final_result["trajectories"]
            },
            "traj_info": traj_info
        }
    
    def _process_step(self, example, system_prompt, previous_steps, previous_values, token_stats, rollout_stats, traj_info):
        """Single decoding step, handling the 4D attention mask and the 3D RoPE
        position-id update."""
        stop_foresight = False
        _thr = self.thresholds  # calibratable constants for this run
        _tp = self.task_profile(system_prompt)
        pixel_values = example.get('pixel_values')
        audio_values = example.get('audio_values')
        image_values = example.get('image_values')

        all_responses_first_stage = []
        all_logprobs_first_stage = []
        all_advantages_first_stage = []
        all_penalties_first_stage = []
        all_dist_ev_first_stage = []
        all_dist_cnt_first_stage = []
        all_token_nums_first_stage = []
        #       d_t = log p_final(y_t) − log p₂₂(y_t)
        #
        all_grnd_stats_first_stage = []
        _probe_feats_step = []
        _aud_rows_step = []
        
        # -- Candidate-level probe feature dump (--dump_probe_features) --
        _dump_dir = getattr(self.args, 'dump_probe_features', '') or ''
        _probe_on = bool(_dump_dir)
        _probe_ls = []
        if _probe_on:
            _probe_ls = [int(x) for x in str(
                getattr(self.args, 'probe_layers', '14,20,24')).split(',') if x.strip()]

        # Hidden states are only captured when a repair operator needs them.
        # --layer_repair is the current switch; --layer_fix_mode is the legacy one.
        _repair_mode = getattr(self.args, 'layer_repair', 'none')
        _legacy_mode = getattr(self.args, 'layer_fix_mode', 'linear')
        _need_hidden = (_repair_mode != 'none') or (_legacy_mode != 'linear')
        _aud_scope = getattr(self.args, 'auditor_scope', 'last')
        _aud_dump = bool(getattr(self.args, 'dump_layer_audit', False)) and _aud_scope == 'shadow'
        if getattr(self.args, 'auditor_diag', False) and _aud_scope == 'full':
            _need_hidden = True
        auditor = DualLayerAuditor(self.model, self.processor,
                                   target_layers=self.anchor_layers,
                                   store_hidden=_need_hidden,
                                   scope=_aud_scope)
        auditor.register()

        for beam_idx in range(self.args.step_beam_size):
            print(f"\n🔍 [DEBUG] ================= Beam {beam_idx+1} =================", flush=True)
            
            chat = self._prepare_chat_template_for_first_stage(example, system_prompt)
            p_str = self.processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            if isinstance(p_str, list): p_str = p_str[0]
            
            inputs_text = p_str.replace(self.tokenizer.eos_token, "").strip() + previous_steps[beam_idx]
            
            processor_kwargs = {
                "text": [inputs_text], 
                "images": [image_values] if image_values is not None else None, 
                "videos": [pixel_values] if pixel_values is not None else None,
                "audio": [audio_values] if audio_values is not None else None,
                "fps": [1.0] if pixel_values is not None else None, 
                "return_tensors": "pt"
            }
            if "halo" in self.args.datasets.lower():
                processor_kwargs["max_pixels"] = 224 * 224

            m_inputs = self.processor(**processor_kwargs,
                                      **_aiv_kw(pixel_values, audio_values)).to(self.model.device)

            token_stats["input"] += m_inputs.input_ids.numel()

            # =====================================================================
            # =====================================================================
            class StopExecution(Exception): pass
            
            captured = {}
            def universal_hook(module, args, kwargs):
                target = kwargs.get('inputs_embeds', None)
                if target is None and len(args) > 0:
                    if isinstance(args[0], torch.Tensor) and args[0].dim() == 3:
                        target = args[0]
                
                if target is not None:
                    captured['inputs_embeds'] = target.detach().clone()
                    captured['position_ids'] = kwargs.get('position_ids', None)
                    raise StopExecution("Captured")

            h1 = self.model.thinker.register_forward_pre_hook(universal_hook, with_kwargs=True)
            h2 = self.model.thinker.model.register_forward_pre_hook(universal_hook, with_kwargs=True)
            
            try:
                with torch.inference_mode():
                    self.model.generate(**m_inputs, max_new_tokens=1, use_cache=False,
                                        **_aiv_kw_for(m_inputs))
            except StopExecution:
                print(f"✅ [DEBUG] features and position encodings captured.", flush=True)
            except Exception as e:
                print(f"❌ [DEBUG] capture error: {e}", flush=True)
            finally:
                h1.remove()
                h2.remove()

            inputs_embeds_f = captured.get('inputs_embeds')
            if inputs_embeds_f is None:
                inputs_embeds_f = self.model.thinker.get_input_embeddings()(m_inputs.input_ids)
            
            attn_mask_f = m_inputs.attention_mask.clone()
            pos_ids_f = captured.get('position_ids')

            # =====================================================================
            # =====================================================================
            m_nv = attn_mask_f.clone()
            v_idx, a_idx = get_modality_indices(self.processor, m_inputs.input_ids, self.model.device)
            _mdi_mode = getattr(self.args, 'mdi_mask_mode', 'all')
            _is_sequential = (_mdi_mode == 'sequential' and v_idx is not None and a_idx is not None)
            if _mdi_mode == 'all':
                if v_idx is not None:
                    m_nv[:, v_idx] = 0
                if a_idx is not None:
                    m_nv[:, a_idx] = 0
            elif _mdi_mode == 'audio_only':
                if a_idx is not None:
                    m_nv[:, a_idx] = 0
            elif _mdi_mode == 'video_only':
                if v_idx is not None:
                    m_nv[:, v_idx] = 0
            elif _mdi_mode == 'sequential':
                # m_nv = no-video mask; m_na = no-audio mask (built below)
                if v_idx is not None:
                    m_nv[:, v_idx] = 0  # m_nv masks video
                # Build second mask for no-audio
                m_na = attn_mask_f.clone()
                if a_idx is not None:
                    m_na[:, a_idx] = 0  # m_na masks audio
                if not _is_sequential:
                    # Fallback: if only one modality present, use 'all' behavior
                    if a_idx is not None:
                        m_nv[:, a_idx] = 0
                    print(f"    [MDI-SEQ] Fallback to 'all' mode (only one modality present)", flush=True)
            is_text_only = (v_idx is None and a_idx is None)  # P1: skip CD-Confess for text-only

            if is_text_only and not getattr(self.args, 'disable_passage_mdi', False) \
                    and self.args.datasets.lower() in ('halueval', 'pubmedqa', 'ragtruth', 'summedits'):
                passage_idx = get_passage_indices(m_inputs.input_ids, self.tokenizer, self.model.device)
                if passage_idx is not None and len(passage_idx) > 0:
                    m_nv[:, passage_idx] = 0
                    is_text_only = False
                    print(f"    📖 [Passage-MDI] masking {len(passage_idx)} passage tokens (idx {passage_idx[0].item()}–{passage_idx[-1].item()})", flush=True)

            # ================================================================
            # ================================================================
            is_avhbench = "avh" in self.args.datasets.lower()

            import time as _time
            _t_rollout_start = _time.perf_counter()
            for rollout_idx in range(self.args.num_rollout):
                torch.cuda.empty_cache()
                with torch.inference_mode():
                    p_full, p_nv = None, None
                    p_na = None  # sequential mode: no-audio KV cache
                    generated_ids_list = []
                    acc_logprob, total_hallu_penalty = 0.0, 0.0
                    total_dist_evidence = 0.0
                    _gs = {'g_sum': 0.0, 'g_n': 0, 'g_sq': 0.0, 'g_min': 0.0,
                           'g_neg_sum': 0.0, 'g_sum_c': 0.0, 'g_n_c': 0, 'g_min_c': 0.0,
                           'd22_sum': 0.0, 'd22_max': 0.0, 'd16_sum': 0.0,
                           'd16_max': 0.0, 'd_n': 0,
                           'd22_sum_c': 0.0, 'd_n_c': 0,
                           'g_bot': [],
                           'e22_sum': 0.0, 'e16_sum': 0.0, 'e22_max': -2.0,
                           'e16_max': -2.0, 'e22_top': []}
                    _probe_sum = None
                    _probe_steps = 0
                    _aud_rows = []
                    n_dist_events = 0

                    m_f_step = attn_mask_f.clone()
                    m_nv_step = m_nv.clone()
                    m_na_step = m_na.clone() if _is_sequential else None
                    pos_ids_step = pos_ids_f
                    curr_ids = None

                    for i in range(48):
                        if i == 0:
                            out_f = self.model.thinker(inputs_embeds=inputs_embeds_f, attention_mask=m_f_step, position_ids=pos_ids_step, use_cache=True)
                            auditor.active = False
                            out_nv = self.model.thinker(inputs_embeds=inputs_embeds_f, attention_mask=m_nv_step, position_ids=pos_ids_step, use_cache=True)
                            if _is_sequential:
                                out_na = self.model.thinker(inputs_embeds=inputs_embeds_f, attention_mask=m_na_step, position_ids=pos_ids_step, use_cache=True)
                            auditor.active = True
                        else:
                            out_f = self.model.thinker(input_ids=curr_ids, attention_mask=m_f_step, position_ids=pos_ids_step, past_key_values=p_full, use_cache=True, output_hidden_states=_probe_on)
                            auditor.active = False
                            out_nv = self.model.thinker(input_ids=curr_ids, attention_mask=m_nv_step, position_ids=pos_ids_step, past_key_values=p_nv, use_cache=True)
                            if _is_sequential:
                                out_na = self.model.thinker(input_ids=curr_ids, attention_mask=m_na_step, position_ids=pos_ids_step, past_key_values=p_na, use_cache=True)
                            auditor.active = True

                        logits_f = out_f.logits[:, -1, :]
                        # -- Probe feature accumulation (--dump_probe_features) --
                        if _probe_on and getattr(out_f, 'hidden_states', None) is not None:
                            _hs = out_f.hidden_states
                            _v = torch.cat([_hs[l + 1][0, -1, :].float()
                                            for l in _probe_ls], dim=0)
                            _probe_sum = _v if _probe_sum is None else _probe_sum + _v
                            _probe_steps += 1
                        p_full = out_f.past_key_values
                        p_nv = out_nv.past_key_values
                        if _is_sequential:
                            p_na = out_na.past_key_values
                        
                        _anchors = self.anchor_layers
                        _l_early, _l_mid = _anchors[0], _anchors[-1]
                        l16_logits = auditor.intermediate_logits.get(_l_early, logits_f).clone()
                        l22_logits = auditor.intermediate_logits.get(_l_mid, logits_f).clone()

                        # -- Contamination magnitude diagnostic (--auditor_scope full --auditor_diag) --
                        if getattr(self.args, 'auditor_diag', False) and auditor.scope == 'full':
                            for _nm, _li in (('early', _l_early), ('mid', _l_mid)):
                                _a = auditor.intermediate_logits.get(_li)
                                _b = auditor.blind_logits.get(_li)
                                if _a is None or _b is None:
                                    continue
                                _pa = torch.softmax(_a.float(), dim=-1)
                                _pb = torch.softmax(_b.float(), dim=-1)
                                _kl = float((_pa * (torch.log(_pa + 1e-12)
                                                    - torch.log(_pb + 1e-12))).sum())
                                _flip = int(_a.argmax(-1).item() != _b.argmax(-1).item())
                                print(f"    [AUD-DIAG] L{_li}({_nm}) top1_flip={_flip} "
                                      f"KL(full||blind)={_kl:.4f} "
                                      f"p_full_max={float(_pa.max()):.3f} "
                                      f"p_blind_max={float(_pb.max()):.3f}", flush=True)
                            _hf = auditor.intermediate_hidden.get(
                                len(self.model.thinker.model.layers) - 1)
                            _hb = auditor.blind_hidden.get(
                                len(self.model.thinker.model.layers) - 1)
                            if _hf is not None and _hb is not None:
                                _cos = float(torch.nn.functional.cosine_similarity(
                                    _hf.float().flatten(), _hb.float().flatten(), dim=0))
                                print(f"    [AUD-DIAG] h_final cos(full,blind)={_cos:.4f} "
                                      f"|full|={float(_hf.float().norm()):.2f} "
                                      f"|blind|={float(_hb.float().norm()):.2f}", flush=True)

                        # =====================================================================
                        # =====================================================================
                        temp_probs = torch.softmax(logits_f / TEMPERATURE, dim=-1)
                        token_id = torch.multinomial(temp_probs, num_samples=1).item()

                        prob_f_val = temp_probs[0, token_id].item()
                        prob_nv_val = torch.softmax(out_nv.logits[:, -1, :] / TEMPERATURE, dim=-1)[0, token_id].item()
                        mdi_av = (prob_f_val - prob_nv_val) / (prob_f_val + 1e-9)
                        # Sequential mode: compute both MDIs, pick the more anomalous one
                        if _is_sequential:
                            prob_na_val = torch.softmax(out_na.logits[:, -1, :] / TEMPERATURE, dim=-1)[0, token_id].item()
                            mdi_video = (prob_f_val - prob_nv_val) / (prob_f_val + 1e-9)  # video dependency
                            mdi_audio = (prob_f_val - prob_na_val) / (prob_f_val + 1e-9)  # audio dependency
                            # Pick the more anomalous (more negative) signal for CD correction
                            if mdi_video < mdi_audio:
                                mdi_av = mdi_video  # video hallucination is worse
                                _seq_dominant = 'video'
                            else:
                                mdi_av = mdi_audio  # audio hallucination is worse
                                prob_nv_val = prob_na_val  # use no-audio as the blind branch
                                _seq_dominant = 'audio'

                        token_str = self.tokenizer.decode([token_id]).lower().strip()
                        clean_word = ''.join(e for e in token_str if e.isalpha())

                        # -- Continuous audit signals: two orthogonal channels, both free of extra forward passes --
                        _pf_s = max(float(prob_f_val), 1e-12)
                        _pb_s = max(float(prob_nv_val), 1e-12)
                        _g_t = max(-6.0, min(6.0, math.log(_pf_s / _pb_s)))
                        _gs['g_sum'] += _g_t
                        _gs['g_sq'] += _g_t * _g_t
                        _gs['g_n'] += 1
                        _gs['g_min'] = min(_gs['g_min'], _g_t)
                        _gs['g_neg_sum'] += min(_g_t, 0.0)
                        _gb = _gs['g_bot']
                        _gb.append(_g_t)
                        if len(_gb) > 3:
                            _gb.sort()
                            del _gb[3:]
                        _is_content = (len(clean_word) > 1 and clean_word not in STOP_WORDS)
                        if _is_content:
                            _gs['g_sum_c'] += _g_t
                            _gs['g_n_c'] += 1
                            _gs['g_min_c'] = min(_gs['g_min_c'], _g_t)
                        try:
                            _p22_t = float(torch.softmax(l22_logits / TEMPERATURE, dim=-1)[0, token_id])
                            _p16_t = float(torch.softmax(l16_logits / TEMPERATURE, dim=-1)[0, token_id])
                            _d22 = max(-20.0, min(20.0, math.log(_pf_s / max(_p22_t, 1e-12))))
                            _d16 = max(-20.0, min(20.0, math.log(_pf_s / max(_p16_t, 1e-12))))
                            _gs['d22_sum'] += _d22
                            _gs['d16_sum'] += _d16
                            _gs['d22_max'] = max(_gs['d22_max'], _d22)
                            _gs['d16_max'] = max(_gs['d16_max'], _d16)
                            _e22 = _pf_s - _p22_t
                            _e16 = _pf_s - _p16_t
                            _gs['e22_sum'] += _e22
                            _gs['e16_sum'] += _e16
                            _gs['e22_max'] = max(_gs['e22_max'], _e22)
                            _gs['e16_max'] = max(_gs['e16_max'], _e16)
                            _t22 = _gs['e22_top']
                            _t22.append(_e22)
                            if len(_t22) > 3:
                                _t22.sort(reverse=True)
                                del _t22[3:]
                            _gs['d_n'] += 1
                            if _is_content:
                                _gs['d22_sum_c'] += _d22
                                _gs['d_n_c'] += 1
                        except Exception:
                            pass
                        # ─────────────────────────────────────────────────────────────

                        # P1: skip for text-only; P2: skip EOS and \boxed tokens
                        _dist_flag = (mdi_av < _thr['mdi_threshold'] and len(clean_word) > 3
                                      and clean_word not in STOP_WORDS
                                      and not is_text_only
                                      and token_id != self.tokenizer.eos_token_id
                                      and 'boxed' not in clean_word
                                      and not getattr(self.args, 'disable_dist', False))

                        # --- Layer-Confess hidden state repair ---
                        # 'paper'  : Eq.(layer-repair) — omega-weighted transported support
                        #            representation + orthogonal projection (Proposition 2).
                        # legacy   : --layer_fix_mode linear_norm / slerp, a fixed-coefficient
                        #            interpolation between one anchor and the final layer.
                        # 'none'   : no hidden repair (reproduces all previously reported runs).
                        #
                        # Gating (--layer_repair_gated): 'on' restricts the repair to steps the
                        # Dist-Confess probe has just flagged, which is what
                        # method_layer_confess.tex describes; 'off' applies it at every step.
                        # The probe itself always reads the *unrepaired* logits, so the flag is
                        # well defined before the repair runs.
                        _repair = getattr(self.args, 'layer_repair', 'none')
                        _lfm = getattr(self.args, 'layer_fix_mode', 'linear')
                        if _repair == 'none' and _lfm != 'linear':
                            _repair = _lfm  # honour the legacy switch
                        _gated = getattr(self.args, 'layer_repair_gated', 'on') == 'on'
                        _do_repair = (_dist_flag or not _gated)
                        _layer_acted = False
                        _upd_grid = None
                        if getattr(self.args, 'disable_layer', False):
                            _repair = 'none'
                        if _repair != 'none' and _do_repair and auditor.store_hidden and not is_text_only:
                            _final_idx = len(self.model.thinker.model.layers) - 1
                            h_final = auditor.intermediate_hidden.get(_final_idx)
                            _alpha_lr = _thr['layer_repair_alpha']
                            h_repaired = None

                            if _repair == 'paper':
                                _h_anchors = [auditor.intermediate_hidden.get(l) for l in _anchors]
                                if h_final is not None and all(h is not None for h in _h_anchors):
                                    h_repaired, _diag = layer_confess_repair(
                                        _h_anchors, h_final,
                                        omega=self.omega_weights,
                                        alpha_l=_alpha_lr,
                                        lambda_l=getattr(self.args, 'lambda_layer', 0.3),
                                        project=getattr(self.args, 'layer_repair_project', 'perp'),
                                    )
                                    _upd = _diag.pop('_delta', None)
                                    _upd_grid = _upd
                                    # -- Decisive diagnostic: does the repair actually change the output token --
                                    if getattr(self.args, 'layer_diag', False) and _upd is not None:
                                        with torch.no_grad():
                                            _hf32 = h_final.float()
                                            def _am(_h):
                                                _n = self.model.thinker.model.norm(
                                                    _h.to(h_final.dtype).unsqueeze(0))
                                                return int(torch.argmax(
                                                    self.model.thinker.lm_head(_n).squeeze(0)[0]).item())
                                            _base_id = _am(_hf32)
                                            _sw = {}
                                            for _lm in (0.3, 1.0, 2.0, 5.0, 10.0):
                                                _sw[f'{_lm:g}'] = int(_am(_hf32 + _lm * _upd) != _base_id)
                                            _diag['argmax_flip_by_lambda'] = _sw
                                            _diag['base_token'] = self.tokenizer.decode([_base_id])
                                            _diag['flip_token_at_cur_lambda'] = self.tokenizer.decode(
                                                [_am(_hf32 + getattr(self.args, 'lambda_layer', 0.3) * _upd)])
                                    # Diagnostics for E9 / Proposition 2 verification
                                    _diag['gated'] = _gated
                                    traj_info.setdefault('layer_repair_diag', []).append(_diag)
                            else:
                                h_early = auditor.intermediate_hidden.get(_l_mid)
                                if h_early is not None and h_final is not None:
                                    if _repair == 'linear_norm':
                                        h_repaired = (1 - _alpha_lr) * h_final + _alpha_lr * h_early
                                        _fn = h_final.norm()
                                        _rn = h_repaired.norm()
                                        if _rn > 1e-8:
                                            h_repaired = h_repaired * (_fn / _rn)
                                    elif _repair == 'linear':
                                        h_repaired = (1 - _alpha_lr) * h_final + _alpha_lr * h_early
                                    elif _repair == 'slerp':
                                        h_a = h_final / (h_final.norm() + 1e-8)
                                        h_b = h_early / (h_early.norm() + 1e-8)
                                        _cos = (h_a * h_b).sum(dim=-1, keepdim=True).clamp(-1, 1)
                                        _theta = torch.acos(_cos)
                                        if _theta.item() < 1e-6:
                                            h_repaired = h_final
                                        else:
                                            _sin = torch.sin(_theta)
                                            h_repaired = (torch.sin((1 - _alpha_lr) * _theta) / _sin) * h_final \
                                                       + (torch.sin(_alpha_lr * _theta) / _sin) * h_early
                                    else:
                                        h_repaired = h_final  # fallback

                            # Reproject to logits: p^L = sigma(W_LM LN(h~))
                            if h_repaired is not None:
                                if not bool(torch.isfinite(h_repaired).all()):
                                    h_repaired = h_final
                                else:
                                    # -- Lambda marginalisation (--layer_lambda_grid, off by default) --
                                    #
                                    #     p(v) = Σ_k w_k · softmax(W_LM LN(h_f + λ_k δ))
                                    #
                                    _grid = self._layer_lambda_grid
                                    if _grid and _upd_grid is not None and _repair == 'paper':
                                        with torch.no_grad():
                                            _hf32 = h_final.float()
                                            _acc = None
                                            _w = 1.0 / len(_grid)
                                            for _lm in _grid:
                                                _n = self.model.thinker.model.norm(
                                                    (_hf32 + _lm * _upd_grid).to(h_final.dtype).unsqueeze(0))
                                                _p = torch.softmax(
                                                    self.model.thinker.lm_head(_n).squeeze(0).float(), dim=-1)
                                                _acc = _p * _w if _acc is None else _acc + _p * _w
                                            logits_f = torch.log(_acc + 1e-12).to(logits_f.dtype)
                                        _diag['lambda_grid'] = list(_grid)
                                        _layer_acted = True
                                    else:
                                        if bool(torch.equal(h_repaired, h_final)):
                                            pass
                                        else:
                                            with torch.no_grad():
                                                _normed = self.model.thinker.model.norm(h_repaired.unsqueeze(0))
                                                logits_f = self.model.thinker.lm_head(_normed).squeeze(0)
                                            _layer_acted = True

                        if _dist_flag:
                            # -- Dist-to-Path channel (accumulation side) --
                            total_dist_evidence += min(-float(mdi_av), 3.0)
                            n_dist_events += 1
                            # ────────────────────────────────────────────────────
                            _seq_info = f", dominant={_seq_dominant}" if _is_sequential else ""
                            print(f"    💡 [Self-Confess] prior trap detected: '{token_str}' (MDI: {mdi_av:.2f}{_seq_info}); entering contrastive exploration mode", flush=True)

                            alpha = getattr(self.args, 'cd_alpha', 0.5)
                            # Sequential mode: use the dominant modality's blind branch
                            if _is_sequential and _seq_dominant == 'audio':
                                blind_logits = out_na.logits[:, -1, :]
                            else:
                                blind_logits = out_nv.logits[:, -1, :]
                            refined_logits = logits_f + alpha * (logits_f - blind_logits)

                            _lam = getattr(self.args, 'cd_contrast_lambda', None)
                            if _lam is not None:
                                refined_logits = logits_f - _lam * blind_logits

                            # -- Hard exclusion of degenerate tokens (--cd_forbid_degenerate, off by default) --
                            if getattr(self.args, 'cd_forbid_degenerate', False):
                                _bad = self._cd_degenerate_ids()
                                if _bad is not None and _bad.numel():
                                    refined_logits = refined_logits.index_fill(
                                        -1, _bad, float('-inf'))

                            _beta = getattr(self.args, 'cd_plausibility_beta', 0.0)
                            _refined_premask = refined_logits
                            if _beta > 0.0:
                                _p_full = torch.softmax(logits_f, dim=-1)
                                _cut = _beta * _p_full.max()
                                refined_logits = refined_logits.masked_fill(_p_full < _cut, float('-inf'))

                            probs_f = torch.softmax(refined_logits / TEMPERATURE, dim=-1)

                            # -- Lambda-by-beta grid diagnostic (--cd_diag) --
                            if getattr(self.args, 'cd_diag', False):
                                _pf = torch.softmax(logits_f, dim=-1)
                                _cut10 = 0.1 * _pf.max()
                                _grid = {}
                                for _l in (0.3333, 0.5, 1.0, 1.5):
                                    for _b in (0.0, 0.1):
                                        _r = logits_f - _l * blind_logits
                                        if _b > 0.0:
                                            _r = _r.masked_fill(_pf < _b * _pf.max(), float('-inf'))
                                        _r = _r.clone()
                                        _r[0, token_id] = float('-inf')
                                        _top = torch.topk(_r[0], 3)
                                        _grid[f'l{_l:g}_b{_b:g}'] = [
                                            self.tokenizer.decode([int(i)]) for i in _top.indices
                                        ]
                                _dg = {
                                    'orig': token_str,
                                    'mdi': round(float(mdi_av), 4),
                                    'head_pick_argmax': self.tokenizer.decode(
                                        [int(torch.argmax(probs_f, dim=-1).item())]),
                                    'n_plausible_b0.1': int((_pf >= _cut10).sum().item()),
                                    'grid': _grid,
                                }
                                print(f"    📐 [CD-DIAG] {json.dumps(_dg, ensure_ascii=False)}", flush=True)

                            _eff_cd_mode = _tp.get('cd_mode') or getattr(self.args, 'cd_mode', 'hard')
                            if _eff_cd_mode == 'soft':
                                _sf = getattr(self.args, 'cd_soft_factor', 0.1)
                                probs_f[0, token_id] *= _sf
                                print(f"    🔧 [CD-Mode=soft] token '{token_str}' scaled ×{_sf}", flush=True)
                            else:
                                probs_f[0, token_id] = 0.0
                            # -- Empty-distribution guard --
                            def _degenerate(_d):
                                return (not bool(torch.isfinite(_d).all())) or float(_d.max()) <= 1e-12
                            _den = probs_f.sum(dim=-1, keepdim=True)
                            if _degenerate(_den):
                                probs_f = torch.softmax(_refined_premask / TEMPERATURE, dim=-1)
                                probs_f[0, token_id] = 0.0
                                _den = probs_f.sum(dim=-1, keepdim=True)
                            _cd_abort = _degenerate(_den)

                            if _cd_abort:
                                print(f"    ⚠️ [CD-Guard] '{token_str}' distribution empty after masking; skipping the rewrite", flush=True)
                                next_id = torch.tensor([[token_id]], device=self.model.device)
                                pf_val = prob_f_val
                                pnv_val = prob_nv_val
                            else:
                                probs_f = probs_f / _den
                                _eff_resample = _tp.get('cd_resample') or getattr(self.args, 'cd_resample', 'sample')
                                if _eff_resample == 'argmax':
                                    next_id = torch.argmax(probs_f, dim=-1, keepdim=True)
                                else:
                                    next_id = torch.multinomial(probs_f, num_samples=1)
                                token_id = next_id.item()

                                pf_val = torch.softmax(logits_f, dim=-1)[0, token_id].item()
                                pnv_val = torch.softmax(out_nv.logits[:, -1, :] / TEMPERATURE, dim=-1)[0, token_id].item()  # P0: fix temperature mismatch
                            mdi_av = (pf_val - pnv_val) / (pf_val + 1e-9)
                            print(f"    🔄 [CD-Correction] found a more sensory-grounded token: '{self.tokenizer.decode([token_id]).strip()}'", flush=True)
                            
                        else:
                            if _layer_acted and not _gated:
                                # -- Layer's independent action surface (--layer_repair_gated off) --
                                probs_f = torch.softmax(logits_f / TEMPERATURE, dim=-1)
                                _eff_resample = _tp.get('cd_resample') or getattr(self.args, 'cd_resample', 'sample')
                                if _eff_resample == 'argmax':
                                    next_id = torch.argmax(probs_f, dim=-1, keepdim=True)
                                else:
                                    next_id = torch.multinomial(probs_f, num_samples=1)
                                token_id = next_id.item()
                                pf_val = float(probs_f[0, token_id].item())
                                pnv_val = torch.softmax(out_nv.logits[:, -1, :] / TEMPERATURE, dim=-1)[0, token_id].item()
                                mdi_av = (pf_val - pnv_val) / (pf_val + 1e-9)
                            else:
                                next_id = torch.tensor([[token_id]], device=self.model.device)
                                token_id = next_id.item()
                                pf_val = prob_f_val
                                pnv_val = prob_nv_val
                        _rep_head = {
                            "token_id": token_id,
                            "res_16_22": torch.norm(torch.softmax(l16_logits, dim=-1) - torch.softmax(l22_logits, dim=-1), p=2).item(),
                            "res_22_fin": torch.norm(torch.softmax(l22_logits, dim=-1) - torch.softmax(logits_f, dim=-1), p=2).item(),
                            "l22_conf": torch.max(torch.softmax(l22_logits, dim=-1)).item(),
                        }
                        penalty = (0.0 if getattr(self.args, 'disable_layer', False)
                                   else calculate_hallu_penalty(_rep_head, self.tokenizer,
                                                                thresholds=_thr))
                        total_hallu_penalty += penalty

                        # -- Shadow bookkeeping (--auditor_scope shadow --dump_layer_audit) --
                        if _aud_dump:
                            _sl16 = auditor.shadow_logits.get(_l_early)
                            _sl22 = auditor.shadow_logits.get(_l_mid)
                            _row = {
                                "tid": int(token_id),
                                "h": [round(_rep_head["res_16_22"], 5),
                                      round(_rep_head["res_22_fin"], 5),
                                      round(_rep_head["l22_conf"], 5)],
                            }
                            if _sl16 is not None and _sl22 is not None:
                                _p16 = torch.softmax(_sl16, dim=-1)
                                _p22 = torch.softmax(_sl22, dim=-1)
                                _pfin = torch.softmax(logits_f, dim=-1)
                                _row["f"] = [
                                    round(torch.norm(_p16 - _p22, p=2).item(), 5),
                                    round(torch.norm(_p22 - _pfin, p=2).item(), 5),
                                    round(torch.max(_p22).item(), 5),
                                ]
                            _aud_rows.append(_row)

                        if token_id == self.tokenizer.eos_token_id: break
                        generated_ids_list.append(token_id)
                        acc_logprob += np.log(max(pf_val, 1e-9))

                        curr_ids = next_id
                        
                        new_bit = torch.ones((1, 1), device=self.model.device, dtype=m_f_step.dtype)
                        m_f_step = torch.cat([m_f_step, new_bit], dim=-1)
                        m_nv_step = torch.cat([m_nv_step, new_bit], dim=-1)
                        if _is_sequential and m_na_step is not None:
                            m_na_step = torch.cat([m_na_step, new_bit], dim=-1)
                        
                        if pos_ids_step is not None:
                            if i == 0:
                                pos_ids_step = pos_ids_step[..., -1:] + 1
                            else:
                                pos_ids_step = pos_ids_step + 1

                    response = self.tokenizer.decode(generated_ids_list, skip_special_tokens=True).strip()
                    logprob = acc_logprob / (len(generated_ids_list) + 1e-9)
                    
                    _pcs = traj_info['path_confess_stats']
                    _pcs['num_paths_explored'] += 1
                    _pcs['max_penalty_seen'] = max(_pcs['max_penalty_seen'], float(total_hallu_penalty))
                    if getattr(self.args, 'disable_path', False):
                        advantage = logprob
                    elif total_hallu_penalty >= _thr['veto_threshold']:
                        advantage = -10000.0
                        logprob = -10000.0
                        _pcs['num_paths_vetoed'] += 1
                        print(f"    🚨 [LETHAL PENALTY] veto triggered. Hallucination score: {total_hallu_penalty:.2f}. Path discarded.", flush=True)
                    else:
                        advantage = logprob - (total_hallu_penalty * 0.8)

                    all_responses_first_stage.append(response)
                    all_logprobs_first_stage.append(logprob)
                    all_advantages_first_stage.append(advantage)
                    all_penalties_first_stage.append(float(total_hallu_penalty))
                    all_dist_ev_first_stage.append(
                        total_dist_evidence / max(len(generated_ids_list), 1))
                    all_dist_cnt_first_stage.append(n_dist_events)
                    all_token_nums_first_stage.append(len(generated_ids_list))
                    all_grnd_stats_first_stage.append(
                        {k: (list(v) if isinstance(v, list) else v)
                         for k, v in _gs.items()})
                    if _aud_dump:
                        _aud_rows_step.append(_aud_rows)
                    if _probe_on:
                        _probe_feats_step.append(
                            (_probe_sum / max(_probe_steps, 1)).cpu().numpy().astype('float16')
                            if _probe_sum is not None else None)
                    token_stats["output"] += len(generated_ids_list)
                    rollout_stats["total"] += 1

        # ══════════════════════════════════════════════════════════════════════
        # ══════════════════════════════════════════════════════════════════════
        #     if total_hallu_penalty >= 0.45:  advantage = -10000
        #     else:                            advantage = logprob - 0.8*penalty
        #
        #
        #     advantage_i = logprob_i − κ · σ_logprob · (penalty_i − μ_penalty) / σ_penalty
        #
        _ps_mode = getattr(self.args, 'path_score', 'head')
        if getattr(self.args, 'disable_path', False):
            _ps_mode = 'head'
        if _ps_mode != 'head' and len(all_advantages_first_stage) > 1:
            _kappa = getattr(self.args, 'path_kappa', 0.5)
            _kappa_d = getattr(self.args, 'path_kappa_dist', 0.0)
            _kappa_g = getattr(self.args, 'path_kappa_grnd', 0.0)
            _grnd_field = getattr(self.args, 'path_grnd_field', 'all')
            _grnd_agg = getattr(self.args, 'path_grnd_agg', 'sqrt')
            _lps = list(all_logprobs_first_stage)
            _pens = list(all_penalties_first_stage)
            _dev = list(all_dist_ev_first_stage)
            _grn = [_grnd_score(st, _grnd_field, _grnd_agg)
                    for st in all_grnd_stats_first_stage]
            _valid = [i for i, l in enumerate(_lps) if l > -9000.0]
            if len(_valid) > 1:
                _mu_l = sum(_lps[i] for i in _valid) / len(_valid)
                _sd_l = (sum((_lps[i] - _mu_l) ** 2 for i in _valid) / len(_valid)) ** 0.5

                def _groups():
                    if _ps_mode == 'graded_sym':
                        _g = {}
                        for i in _valid:
                            _m = re.findall(r'\b(yes|no)\b', all_responses_first_stage[i].lower())
                            _g.setdefault(_m[-1] if _m else '_none', []).append(i)
                        return list(_g.values())
                    return [_valid]

                for _grp in _groups():
                    if len(_grp) < 2:
                        continue
                    _mu_p = sum(_pens[i] for i in _grp) / len(_grp)
                    _sd_p = (sum((_pens[i] - _mu_p) ** 2 for i in _grp) / len(_grp)) ** 0.5
                    # -- Dist-to-Path channel (consumption side) --
                    _devs = [_dev[i] for i in _grp]
                    _mu_d = sum(_devs) / len(_devs)
                    _sd_d = (sum((d - _mu_d) ** 2 for d in _devs) / len(_devs)) ** 0.5
                    _use_d = _kappa_d > 0.0 and _sd_d >= 1e-9
                    # -- Continuous evidence-support channel --
                    _grns = [_grn[i] for i in _grp]
                    _mu_g = sum(_grns) / len(_grns)
                    _sd_g = (sum((g - _mu_g) ** 2 for g in _grns) / len(_grns)) ** 0.5
                    _use_g = _kappa_g > 0.0 and _sd_g >= 1e-9
                    if _sd_p < 1e-9 and not _use_d and not _use_g:
                        continue
                    for i in _grp:
                        _adj = 0.0
                        if _sd_p >= 1e-9:
                            _adj += _kappa * (_pens[i] - _mu_p) / _sd_p
                        if _use_d:
                            _adj += _kappa_d * (_dev[i] - _mu_d) / _sd_d
                        if _use_g:
                            _adj -= _kappa_g * (_grn[i] - _mu_g) / _sd_g
                        all_advantages_first_stage[i] = _lps[i] - _sd_l * _adj
                traj_info.setdefault('path_graded_diag', []).append({
                    'mode': _ps_mode, 'kappa': _kappa, 'kappa_dist': _kappa_d,
                    'kappa_grnd': _kappa_g, 'grnd_field': _grnd_field,
                    'mean_grnd': round(sum(_grn[i] for i in _valid) / len(_valid), 5),
                    'sd_grnd': round((sum((_grn[i] - sum(_grn[j] for j in _valid) / len(_valid)) ** 2
                                          for i in _valid) / len(_valid)) ** 0.5, 5),
                    'n_paths': len(_valid),
                    'sd_logprob': round(_sd_l, 4),
                    'sd_penalty': round(
                        (sum((p - sum(_pens[i] for i in _valid) / len(_valid)) ** 2
                             for p in (_pens[i] for i in _valid)) / len(_valid)) ** 0.5, 4),
                    'mean_dist_ev': round(sum(_dev[i] for i in _valid) / len(_valid), 4),
                    'n_paths_with_dist': sum(1 for i in _valid if all_dist_cnt_first_stage[i] > 0),
                    'n_nonzero_penalty': sum(1 for i in _valid if _pens[i] > 0),
                })

        auditor.remove()
        _t_rollout_end = _time.perf_counter()
        _t_selection_start = _time.perf_counter()

        if self.args.width_pruning_strategy != "none" and self.args.width_pruning_strategy != "":
            keep_foresight_list = []
            
            safe_indices = [i for i, logp in enumerate(all_logprobs_first_stage) if logp > -9000.0]
            
            if self.args.width_pruning_strategy == "low_sigma" and len(safe_indices) > 0:
                safe_logprobs = [all_logprobs_first_stage[i] for i in safe_indices]
                mean = np.mean(safe_logprobs)
                std = np.std(safe_logprobs)
                for idx in safe_indices:
                    if all_logprobs_first_stage[idx] > mean - self.args.sigma_rate * std:
                        keep_foresight_list.append(idx)
            else:
                keep_foresight_list = safe_indices.copy()

            if len(keep_foresight_list) < self.args.step_beam_size:
                print(f"⚠️ only {len(keep_foresight_list)} candidates left after pruning; topping up with safe fallbacks...", flush=True)
                
                safe_available = [
                    i for i in range(len(all_logprobs_first_stage)) 
                    if i not in keep_foresight_list and all_logprobs_first_stage[i] > -9000.0
                ]
                
                if safe_available:
                    num_to_add = min(self.args.step_beam_size - len(keep_foresight_list), len(safe_available))
                    
                    safe_available.sort(key=lambda x: all_logprobs_first_stage[x], reverse=True)
                    additional_indices = safe_available[:num_to_add]
                    
                    keep_foresight_list.extend(additional_indices)
                    print(f"    ➕ topped up with {num_to_add} safe paths, best first.", flush=True)
                else:
                    print("    ⚠️ no spare safe paths available; the beam size will shrink for this step.", flush=True)
                    traj_info['path_confess_stats']['beam_shrink_occurred'] = True

            keep_foresight_list.sort()
            rollout_stats["saved"] += (len(all_logprobs_first_stage) - len(keep_foresight_list))

            all_responses = [all_responses_first_stage[i] for i in keep_foresight_list]
            filtered_logprobs = [all_logprobs_first_stage[i] for i in keep_foresight_list]
            filtered_advantages = [all_advantages_first_stage[i] for i in keep_foresight_list]

        completed_responses = []
        completed_logprobs = []
        completed_advantages = []

        print(f"🔄 running deep completion for the {len(keep_foresight_list)} selected candidate paths...", flush=True)

        for idx in range(len(keep_foresight_list)):
            response = all_responses[idx]
            beam_idx = keep_foresight_list[idx] // self.args.num_rollout
            
            chat = self._prepare_chat_template(example, system_prompt)
            p_str = self.processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            if isinstance(p_str, list): p_str = p_str[0]
            
            clean_resp = response.replace("<|video|>", "").replace("<|audio|>", "").replace("<|vision_start|>", "").replace("<|vision_end|>", "")
            inputs_text = p_str.replace(self.tokenizer.eos_token, "").strip() + previous_steps[beam_idx] + clean_resp
            
            processor_kwargs_rollout = {
                "text": [inputs_text],
                "images": [image_values] if image_values is not None else None,
                "videos": [pixel_values] if pixel_values is not None else None,
                "audio": [audio_values] if audio_values is not None else None,
                "fps": [1.0] if pixel_values is not None else None,
                "return_tensors": "pt",
            }
            if "halo" in self.args.datasets.lower():
                processor_kwargs_rollout["max_pixels"] = 2_200_000
            m_inputs = self.processor(**processor_kwargs_rollout,
                                      **_aiv_kw(pixel_values, audio_values)).to(self.model.device)

            token_stats["input"] += m_inputs.input_ids.numel()
            input_len = m_inputs.input_ids.shape[1]

            with torch.inference_mode():
                out = safe_model_generate(
                    self.model, self.processor, m_inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    stop_strings=["<end_of_reasoning>", "Human:", "Assistant:", "Does that help?"],
                    repetition_penalty=1.1,
                    tokenizer=self.tokenizer,
                    _args=self.args
                )
                
                gen_ids = out.sequences[0][input_len:]
                new_gen_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
                
                for stop_word in ["Human:", "Assistant:", "Question:", "Does that help?"]:
                    new_gen_text = new_gen_text.split(stop_word)[0]
                
                full_response = response + new_gen_text.strip()
                
                if gen_ids.numel() > 0 and hasattr(out, 'scores'):
                    step_logprobs = []
                    for step_idx, step_logits in enumerate(out.scores):
                        token_id = gen_ids[step_idx]
                        log_probs = torch.nn.functional.log_softmax(step_logits[0].float(), dim=-1)
                        step_logprobs.append(log_probs[token_id].item())
                    logprob = sum(step_logprobs) / len(step_logprobs)
                else:
                    logprob = all_logprobs_first_stage[keep_foresight_list[idx]]
                
                advantage = logprob 

                completed_responses.append(full_response)
                completed_logprobs.append(logprob)
                completed_advantages.append(advantage)
                
                token_stats["output"] += len(gen_ids)
                rollout_stats["total"] += 1
            
            torch.cuda.empty_cache()
            import gc; gc.collect()

        try:
            self._tv_pool.extend(
                (t, float(l)) for t, l in zip(completed_responses, completed_logprobs))
        except AttributeError:
            self._tv_pool = [(t, float(l))
                             for t, l in zip(completed_responses, completed_logprobs)]
        try:
            k_num = self.args.cluster_num
            # Guard: need at least k_num non-empty responses for clustering
            non_empty_mask = [bool(r.strip()) for r in completed_responses]
            if sum(non_empty_mask) < k_num:
                raise ValueError(f"Too few non-empty responses ({sum(non_empty_mask)}) for k={k_num}")
            X = self._vectorize_texts(completed_responses)
            _n_rows = X.shape[0] if hasattr(X, 'shape') else len(X)
            if _n_rows < k_num:
                raise ValueError(f"Vectorized matrix rows ({_n_rows}) < k={k_num}")
            kmeans = KMeans(n_clusters=k_num, n_init=10, random_state=self.args.seed)
            kmeans.fit(X)
            cluster_labels = kmeans.labels_

            cluster_list = [[] for _ in range(k_num)]
            for idx, label in enumerate(cluster_labels):
                cluster_list[label].append(idx)
            cluster_list = [sorted(cluster) for cluster in cluster_list]

            cluster_len_ratio = [len(cluster)/len(completed_responses) for cluster in cluster_list]
            per_sample_cluster_ratio = [cluster_len_ratio[cluster_labels[i]] for i in range(len(completed_responses))]
            cluster_weights = softmax(per_sample_cluster_ratio)
            
            adv_weights = softmax([adv/TEMPERATURE for adv in completed_advantages])

            weights = [(cluster_weights[ii] + adv_weights[ii]) / 2 for ii in range(len(completed_responses))]
            weights = np.array(weights) / sum(weights) 

            selected = np.random.choice(len(weights), size=self.args.step_beam_size, p=weights, replace=False).tolist()

            sizes = np.bincount(cluster_labels)
            largest_ratio = max(sizes) / len(completed_responses)
            if largest_ratio >= self.args.threshold:
                stop_foresight = True

            if _probe_on and _probe_feats_step:
                _keep = [k for k, f in enumerate(_probe_feats_step) if f is not None]
                traj_info.setdefault('_probe_feats', []).append(
                    np.stack([_probe_feats_step[k] for k in _keep]) if _keep else None)
                traj_info.setdefault('_probe_meta', []).append({
                    'layers': list(_probe_ls),
                    'rows': _keep,
                    'texts': [all_responses_first_stage[k] for k in _keep],
                    'logprob': [float(all_logprobs_first_stage[k]) for k in _keep],
                    'penalty': [float(all_penalties_first_stage[k]) for k in _keep],
                    'dist_ev': [float(all_dist_ev_first_stage[k]) for k in _keep],
                    'is_vetoed': [bool(all_advantages_first_stage[k] <= -9000.0) for k in _keep],
                })

            step_info = {
                'first_stage': {
                    'candidates': [
                        {
                            'text': text,
                            'logprob': round(float(all_logprobs_first_stage[i]), 4),
                            'advantage': round(float(all_advantages_first_stage[i]), 4),
                            'penalty': round(float(all_penalties_first_stage[i]), 4),
                            'dist_ev': round(float(all_dist_ev_first_stage[i]), 5),
                            'dist_n': int(all_dist_cnt_first_stage[i]),
                            'grnd': {k: (round(v, 5) if isinstance(v, float) else v)
                                     for k, v in all_grnd_stats_first_stage[i].items()},
                            'ntok': int(all_token_nums_first_stage[i]),
                            'is_vetoed': bool(all_advantages_first_stage[i] <= -9000.0),
                            **({'layer_audit': _aud_rows_step[i]}
                               if _aud_dump and i < len(_aud_rows_step) else {})
                        } for i, text in enumerate(all_responses_first_stage)
                    ]
                },
                'width_pruning_survivors': keep_foresight_list,
                'second_stage': {
                    'completions': completed_responses,
                    'logprobs': [round(float(x), 4) for x in completed_logprobs]
                },
                'clustering': {
                    'labels': cluster_labels.tolist(),
                    'sizes': sizes.tolist(),
                    'consensus_ratio': round(float(largest_ratio), 4)
                },
                'final': {
                    'selected_steps': [
                        previous_steps[keep_foresight_list[idx]//self.args.num_rollout] + 
                        all_responses_first_stage[keep_foresight_list[idx]] + "\n" 
                        for idx in selected
                    ],
                    'selected_values': [round(float(completed_logprobs[idx]), 4) for idx in selected]
                }
            }
            traj_info['foresight_part'].append(step_info)
            _t_selection_end = _time.perf_counter()
            traj_info.setdefault('_step_timing', []).append({
                'rollout_seconds': round(_t_rollout_end - _t_rollout_start, 4),
                'selection_seconds': round(_t_selection_end - _t_selection_start, 4),
            })

            return {
                "next_steps": step_info['final']['selected_steps'],
                "next_values": step_info['final']['selected_values'],
                "trajectories": completed_responses,
                "steps": [keep_foresight_list[idx] for idx in selected],
                "logprobs": completed_logprobs,
                "advantages": completed_advantages,
                "stop_foresight": stop_foresight
            }

        except Exception as e:
            print(f'⚠️ clustering failed: {e}. Falling back to advantage-only sampling.', flush=True)
            traj_info['path_confess_stats']['cluster_fallback_occurred'] = True
            # BUG FIX: completed_advantages / keep_foresight_list may be empty when all beams
            # produced empty responses; fall back entirely to first-stage results in that case.
            if completed_advantages and keep_foresight_list:
                adv_source = completed_advantages
                resp_source = completed_responses
                logp_source = completed_logprobs
                idx_source  = keep_foresight_list
            else:
                print('⚠️ the completed list is empty; using first-stage results instead.', flush=True)
                adv_source  = all_advantages_first_stage
                resp_source = all_responses_first_stage
                logp_source = all_logprobs_first_stage
                idx_source  = list(range(len(all_responses_first_stage)))
            if not adv_source:
                print('⚠️ the advantage list is empty; using uniform weights.', flush=True)
                adv_source  = [0.0]
                resp_source = [""]
                logp_source = [0.0]
                idx_source  = [0]
            weights = softmax([adv/TEMPERATURE for adv in adv_source])
            n_avail = len(weights)
            n_select = min(self.args.step_beam_size, n_avail)
            selected = np.random.choice(n_avail, size=n_select, p=weights, replace=(n_avail < self.args.step_beam_size)).tolist()

            fallback_steps = [
                previous_steps[idx_source[idx]//self.args.num_rollout] +
                all_responses_first_stage[idx_source[idx]] + "\n"
                if idx_source[idx] < len(all_responses_first_stage)
                else previous_steps[0]
                for idx in selected
            ]

            _t_selection_end = _time.perf_counter()
            traj_info.setdefault('_step_timing', []).append({
                'rollout_seconds': round(_t_rollout_end - _t_rollout_start, 4),
                'selection_seconds': round(_t_selection_end - _t_selection_start, 4),
            })

            return {
                "next_steps": fallback_steps,
                "next_values": [logp_source[idx] for idx in selected],
                "trajectories": all_responses_first_stage,
                "steps": [idx_source[idx] for idx in selected],
                "logprobs": all_logprobs_first_stage,
                "advantages": all_advantages_first_stage,
                "stop_foresight": stop_foresight
            }

    def _should_stop_early(self, step_results, current_step):
        """Whether the early-stopping condition is met. On multimodal inputs this saves
        substantial compute, since video token sequences are long."""
        if current_step < self.args.least_foresight_num:
            return False

        just_stop = True
        trajectories = step_results.get("trajectories", [])
        if len(trajectories) > 1:
            first_response = trajectories[0].strip()
            for response in trajectories[1:]:
                if response.strip() != first_response:
                    just_stop = False
                    break
        else:
            just_stop = False

        if just_stop and len(trajectories) > 0:
            print(f"🛑 [early stop] depth {current_step}: all completions are identical.")
            return True

        if self.args.depth_pruning_strategy == "cluster":
            if step_results.get("stop_foresight", False):
                print(f"🛑 [early stop] depth {current_step}: the largest cluster ratio exceeds the threshold {self.args.threshold} (consensus reached).")
                return True

        return False

    def _s2_tail(self, scaf, question_text):
        """Prefix that precedes the second-stage verdict. Returns the original literal
        when `scaf` is None.

        The overriding versions place the instruction about which modality to
        consult immediately before `\\boxed{`. That is the closest point in the
        whole chain to the verdict token, and the second stage generates only a
        handful of tokens, so it constrains the verdict more strongly than the same
        rule placed in the system prompt.

        The modality is derived deterministically from the question and never looks
        at the ground truth; if it cannot be determined the original literal is used.
        """
        _default = (f"\n\nQuestion: {question_text}\n"
                    "Based STRICTLY on your analysis above, answer the Question. "
                    "Judgment: \\boxed{")
        tmpl = (scaf or {}).get("s2_template")
        if not tmpl:
            return _default
        stream = _cmm_stream_of(question_text) if _JUDGE_PROMPTS_LOADED else None
        if not stream:
            return _default
        tgt = _cmm_target_of(question_text) if _JUDGE_PROMPTS_LOADED else None
        deciding = "Visual" if stream == "video" else "Auditory"
        if not tgt:
            from judge_prompts import _MC_S2_NOTGT
            _ntg = (scaf or {}).get("s2_template_notgt") or _MC_S2_NOTGT
            return _ntg.format(question=question_text, stream=stream,
                               deciding=deciding)
        return tmpl.format(question=question_text, stream=stream,
                           target=tgt, deciding=deciding)

    # ══════════════════════════════════════════════════════════════════════
    # ══════════════════════════════════════════════════════════════════════
    def _format_segments(self, stage, s1_prefix=None):
        """Select the skeleton for the current (stage, dataset) pair, or None when there
        is no match, in which case only anti-copy remains active.

        The skeleton must line up exactly with the prefix: the first-stage prefix
        has already emitted the header of one checklist, so the first segment is a
        slot rather than a literal.

        When `s1_prefix` is given, the checklist order is read back out of the
        prefix rather than hardcoded, because the prefix has several possible
        sources and hardcoding the order would eventually misalign a whole segment.
        An assertion then checks the alignment; the symptom of a misalignment is two
        checklist headers in the output, which is very hard to spot by eye in a log.

        The order matters because the checklist written last sits closest to the
        verdict token, so the modality being asked about should come last.

        phd uses a skeleton with no auditory checklist: it is an image-only dataset,
        and forcing a checklist for an input it does not have amounts to instructing
        the model to invent one.
        """
        if stage == "stage2":
            return _FC.cmm_stage2_segments(self.tokenizer)
        ds = (self.args.datasets or "").lower()
        segs = None
        if ds == "cmm":
            first = _FC.first_from_prefix(s1_prefix) if s1_prefix else None
            segs = _FC.cmm_stage1_segments(self.tokenizer,
                                           first=first or "visual")
        elif ds == "phd":
            segs = _FC.phd_stage1_segments(self.tokenizer)
        if segs is not None:
            if s1_prefix:
                _FC.assert_prefix_matches(segs, s1_prefix)
            return segs
        if not getattr(self, '_fc_warned', False):
            print(f"⚠️ [format control] dataset {ds!r} has no matching skeleton; the template is disabled"
                  f" (anti-copy still applies if enabled)", flush=True)
            self._fc_warned = True
        return None

    def _format_procs(self, stage, anti_copy_text=None, s1_prefix=None):
        """Assemble the decoding-time format constraints, returning
        (LogitsProcessorList | None, template processor | None).

        With --format_control off and --verdict_lock unset this returns
        (None, None), so the call site passes no extra kwargs at all and behaviour
        is byte-identical to earlier runs.

        `anti_copy_text` is fed only the instruction text (system prompt plus the
        scaffold suffix) and deliberately not the notes held in the tree: the
        observed echoes all come from the instructions, whereas the checklist is
        supposed to restate facts from the notes, and banning those too would push
        the model to reword evidence it has just committed to purely to avoid an
        n-gram match.

        `s1_prefix` is the raw first-stage prefix, handed to _format_segments so it
        can read the checklist order and assert alignment. It is optional.
        """
        if not _FORMAT_CONTROL_LOADED:
            return None, None
        mode = getattr(self.args, 'format_control', 'off')
        lock = bool(getattr(self.args, 'verdict_lock', False))
        segs = None
        if stage == "stage2":
            segs = self._format_segments(stage) if lock else None
        elif mode in ('template', 'both'):
            segs = self._format_segments(stage, s1_prefix=s1_prefix)
        pid = None
        if stage != "stage2" and mode in ('anticopy', 'both') and anti_copy_text:
            pid = self.tokenizer.encode(anti_copy_text, add_special_tokens=False)
        if segs is None and pid is None:
            return None, None
        return _FC.make_processors(self.tokenizer, segs=segs, prompt_ids=pid,
                                   anti_copy_n=getattr(self.args, 'anti_copy_n', 6))

    def _fc_kwargs(self, lp, tf):
        """Merge the processors into generate()'s kwargs. Both None yields an empty dict,
        so the call site is unaffected."""
        kw = {}
        if lp is not None:
            kw['logits_processor'] = lp
        if tf is not None:
            kw['stopping_criteria'] = _FC.build_stopping_criteria(
                self.tokenizer, None, template_proc=tf)
        return kw

    def _generate_final_response(self, example, system_prompt, previous_steps, previous_values, token_stats, rollout_stats, traj_info):
        """Final stage: produce the answer. Handles AVH (single sentence), HaloQuest
        (short answer), the text-only datasets (Answer format) and CMM/PhD
        (two-stage verdict)."""
        pixel_values = example.get('pixel_values')
        audio_values = example.get('audio_values')
        image_values = example.get('image_values')

        traj_info['internal_reasoning'] = previous_steps 

        all_responses = []
        all_logprobs = []
        all_advantages = []

        print(f"🏁 running the final aligned generation ({self.args.step_beam_size} paths)...", flush=True)

        for beam_idx in range(self.args.step_beam_size):
            chat = self._prepare_chat_template(example, system_prompt)
            p_str = self.processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            if isinstance(p_str, list): p_str = p_str[0]
            
            if "avh" in self.args.datasets.lower():
                def _is_sane(text):
                    words = text.lower().split()
                    if len(words) < 3: return False
                    from collections import Counter as _C
                    if any(v > 4 for v in _C(words).values()): return False
                    return True
                labels = ["Audio" if b % 2 == 0 else "Visual" for b in range(len(previous_steps))]
                note_parts = []
                for b in range(len(previous_steps)):
                    note = previous_steps[b].strip()[:150]
                    if _is_sane(previous_steps[b]):
                        note_parts.append(f"[{labels[b]}]: {note}")
                if not note_parts:
                    note_parts = [previous_steps[0].strip()[:150]]
                reasoning_history = " | ".join(note_parts)
            else:
                reasoning_history = previous_steps[beam_idx]

            # ==========================================
            # ==========================================
            if "avh" in self.args.datasets.lower():
                apv = getattr(self.args, 'avh_prompt_version', 'v0')
                if _AVH_PROMPTS_LOADED:
                    _, avh_final_tmpl, avh_max_tok = _get_avh_prompt(apv)
                    avh_final_instruction = avh_final_tmpl.format(notes=reasoning_history.strip())
                else:
                    avh_max_tok = 60
                    avh_final_instruction = (
                        f"\n\n[Your observations]:\n{reasoning_history.strip()}\n\n"
                        "Based on your observations, write one sentence describing what is happening in this audio-visual clip.\n"
                        "Answer: "
                    )

                inputs_text_avh = p_str.replace(self.tokenizer.eos_token, "").strip() + avh_final_instruction

                print(f"    [DIAG-AVH] beam={beam_idx} reasoning_history (full):", flush=True)
                print(f"    [DIAG-AVH] notes={repr(reasoning_history)}", flush=True)
                print(f"    [DIAG-AVH] avh_final_instruction={repr(avh_final_instruction)}", flush=True)
                print(f"    [DIAG-AVH] inputs_text_avh (FULL)={repr(inputs_text_avh)}", flush=True)

                m_inputs_avh = self.processor(
                    text=[inputs_text_avh],
                    images=[image_values] if image_values is not None else None,
                    videos=[pixel_values] if pixel_values is not None else None,
                    audio=[audio_values] if audio_values is not None else None,
                    fps=[1.0] if pixel_values is not None else None, return_tensors="pt",
                    **_aiv_kw(pixel_values, audio_values)
                ).to(self.model.device)

                token_stats["input"] += m_inputs_avh.input_ids.numel()

                with torch.inference_mode():
                    out_avh = safe_model_generate(
                        self.model, self.processor, m_inputs_avh,
                        max_new_tokens=avh_max_tok,
                        do_sample=False,
                        stop_strings=["\n", "<|im_end|>"],
                        repetition_penalty=1.05,
                        tokenizer=self.tokenizer,
                        _args=self.args
                    )

                    gen_ids = out_avh.sequences[0][m_inputs_avh.input_ids.shape[1]:]
                    raw_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                    # [DIAG] dump raw generated tokens
                    _diag_tok_list = gen_ids.tolist()
                    _diag_tok_strs = [self.tokenizer.decode([t]) for t in _diag_tok_list]
                    print(f"    [DIAG-AVH] gen_ids ({len(_diag_tok_list)} tokens): {list(zip(_diag_tok_list, _diag_tok_strs))}", flush=True)
                    print(f"    [DIAG-AVH] raw_text={repr(raw_text)}", flush=True)

                    # [DIAG-AVH] logit top-5 for first 5 generated tokens
                    if hasattr(out_avh, 'scores') and len(out_avh.scores) > 0:
                        _n_diag = min(5, len(out_avh.scores))
                        for _si in range(_n_diag):
                            _logits = out_avh.scores[_si][0].float()
                            _probs = torch.nn.functional.softmax(_logits, dim=-1)
                            _top5_vals, _top5_ids = torch.topk(_probs, 5)
                            _top5_toks = [self.tokenizer.decode([tid]) for tid in _top5_ids.tolist()]
                            _top5_info = [(tid, tok, f"{p:.4f}") for tid, tok, p in zip(_top5_ids.tolist(), _top5_toks, _top5_vals.tolist())]
                            _actual = gen_ids[_si].item() if _si < len(gen_ids) else -1
                            print(f"    [DIAG-AVH] step[{_si}] actual={_actual}({self.tokenizer.decode([_actual]) if _actual>=0 else '?'}) top5={_top5_info}", flush=True)

                    gen_caption = raw_text.split('<|im_end|>')[0].split('\n')[0].strip()
                    gen_caption = unwrap_boxed(gen_caption)

                    if gen_ids.numel() > 0 and hasattr(out_avh, 'scores'):
                        step_logprobs = []
                        for step_idx, step_logits in enumerate(out_avh.scores):
                            token_id = gen_ids[step_idx]
                            log_probs = torch.nn.functional.log_softmax(step_logits[0].float(), dim=-1)
                            step_logprobs.append(log_probs[token_id].item())
                        logprob = sum(step_logprobs) / len(step_logprobs)
                    else:
                        logprob = 0.0
                
                final_output = gen_caption
                token_stats["output"] += len(gen_ids)

            # ==========================================
            # ==========================================
            elif "halo" in self.args.datasets.lower():
                # HaloQuest: full-sentence output to match sentence-level GT for Token-F1
                hpv = getattr(self.args, 'halo_prompt_version', 'v0')
                if _HALO_PROMPTS_LOADED and hpv in ('v4', 'v5', 'v6', 'v7', 'v8', 'v9'):
                    _, final_tmpl, halo_max_tok = _get_halo_prompt(hpv)
                    halo_final_instruction = final_tmpl.format(notes=reasoning_history.strip())
                elif hpv == 'v1':
                    halo_max_tok = 100
                    halo_final_instruction = (
                        f"\n\n[Visual Observation Notes]:\n{reasoning_history.strip()}\n\n"
                        "FINAL ANSWER RULES:\n"
                        "• Write ONE complete sentence (8+ words). NO \\boxed{}, no single words, no bare numbers.\n"
                        "• Object present → 'The [object] is [property].'\n"
                        "• Object absent  → 'There is no [object] visible in this image.'\n"
                        "• Unclear detail → 'The [detail] is not visible in this image.'\n"
                        "Complete sentence answer: "
                    )
                elif hpv == 'v2':
                    halo_max_tok = 100
                    halo_final_instruction = (
                        f"\n\n[Visual Observation Notes]:\n{reasoning_history.strip()}\n\n"
                        "Based on your visual observations above, complete this sentence to answer the question:\n"
                        "In this image, "
                    )
                elif hpv == 'v3':
                    halo_max_tok = 100
                    halo_final_instruction = (
                        f"\n\n[Visual Observation Notes]:\n{reasoning_history.strip()}\n\n"
                        "Using your observations, write ONE factual sentence to answer the question.\n"
                        "Do NOT use \\boxed{}. Do NOT write a single word or number.\n"
                        "If the object is absent, say so explicitly: 'There is no X in this image.'\n"
                        "If the object is present, describe it: 'The X is Y.'\n"
                        "One sentence answer: "
                    )
                else:  # v0
                    halo_max_tok = 100
                    halo_final_instruction = (
                        f"\n\n[Internal Observation Notes]:\n{reasoning_history.strip()}\n\n"
                        "CRITICAL INSTRUCTION:\n"
                        "Based on your notes, answer the question in ONE complete, natural sentence.\n"
                        "If the asked object/detail is NOT present in the image, explicitly state that.\n"
                        "Answer: "
                    )

                inputs_text_halo = p_str.replace(self.tokenizer.eos_token, "").strip() + halo_final_instruction

                m_inputs_halo = self.processor(
                    text=[inputs_text_halo],
                    images=[image_values] if image_values is not None else None,
                    videos=[pixel_values] if pixel_values is not None else None,
                    audio=[audio_values] if audio_values is not None else None,
                    fps=[1.0] if pixel_values is not None else None,
                    max_pixels=224 * 224,
                    return_tensors="pt",
                    **_aiv_kw(pixel_values, audio_values)
                ).to(self.model.device)

                token_stats["input"] += m_inputs_halo.input_ids.numel()

                with torch.inference_mode():
                    out_halo = safe_model_generate(
                        self.model, self.processor, m_inputs_halo,
                        max_new_tokens=halo_max_tok,
                        do_sample=False,
                        stop_strings=["\n\n", "<|im_end|>"],
                        repetition_penalty=1.05,
                        tokenizer=self.tokenizer,
                        _args=self.args
                    )

                    gen_ids = out_halo.sequences[0][m_inputs_halo.input_ids.shape[1]:]
                    raw_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                    # Take first sentence; strip trailing newlines/special tokens
                    gen_caption = raw_text.split('<|im_end|>')[0].split('\n\n')[0].strip()
                    # If model still outputs \boxed{}, unwrap it gracefully
                    gen_caption = unwrap_boxed(gen_caption)

                    if gen_ids.numel() > 0 and hasattr(out_halo, 'scores'):
                        step_logprobs = []
                        for step_idx, step_logits in enumerate(out_halo.scores):
                            token_id = gen_ids[step_idx]
                            log_probs = torch.nn.functional.log_softmax(step_logits[0].float(), dim=-1)
                            step_logprobs.append(log_probs[token_id].item())
                        logprob = sum(step_logprobs) / len(step_logprobs)
                    else:
                        logprob = 0.0

                final_output = gen_caption  # plain sentence, no \boxed{} wrapper
                token_stats["output"] += len(gen_ids)

            # ==========================================
            # ==========================================
            elif self.args.datasets.lower() in ["ragtruth", "halueval", "pubmedqa"]:
                # P3-a: HaluEval uses short-phrase constraint to match GT entity format
                if self.args.datasets.lower() == "halueval":
                    hepv = getattr(self.args, 'halueval_prompt_version', 'v0')
                    if _HALUEVAL_PROMPTS_LOADED:
                        _, halu_final_tmpl, halu_max_tok = _get_halueval_prompt(hepv)
                        text_final_instruction = halu_final_tmpl.format(notes=reasoning_history.strip())
                    else:
                        halu_max_tok = 100
                        text_final_instruction = (
                            f"\n\n[Internal Reasoning]:\n{reasoning_history.strip()}\n\n"
                            "CRITICAL INSTRUCTION:\n"
                            "Based on your reasoning above, output ONLY the answer as a short phrase (1-5 words). No explanation, no full sentence.\n"
                            "Answer:"
                        )
                elif self.args.datasets.lower() == "ragtruth":
                    halu_max_tok = 300
                    final_crit = "Based on your reasoning above, provide your final answer directly (no prefix, no brackets).\n"
                    text_final_instruction = (
                        f"\n\n[Internal Reasoning]:\n{reasoning_history.strip()}\n\n"
                        "CRITICAL INSTRUCTION:\n"
                        f"{final_crit}"
                        "Answer:"
                    )
                else:
                    halu_max_tok = 100
                    final_crit = "Based on your reasoning above, provide your final answer directly (no prefix, no brackets).\n"
                    text_final_instruction = (
                        f"\n\n[Internal Reasoning]:\n{reasoning_history.strip()}\n\n"
                        "CRITICAL INSTRUCTION:\n"
                        f"{final_crit}"
                        "Answer:"
                    )
                
                inputs_text_pure = p_str.replace(self.tokenizer.eos_token, "").strip() + text_final_instruction
                
                m_inputs_pure = self.processor(
                    text=[inputs_text_pure], 
                    return_tensors="pt"
                ).to(self.model.device)

                token_stats["input"] += m_inputs_pure.input_ids.numel()
                
                with torch.inference_mode():
                    out_pure = safe_model_generate(
                        self.model, self.processor, m_inputs_pure,
                        max_new_tokens=halu_max_tok,
                        do_sample=False,
                        stop_strings=["\n\n", "Human:"],
                        tokenizer=self.tokenizer,
                        _args=self.args,
                    )
                    
                    gen_ids = out_pure.sequences[0][m_inputs_pure.input_ids.shape[1]:]
                    raw_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                    
                    gen_answer = raw_text.split('\n\n')[0].strip()
                    gen_answer = unwrap_boxed(gen_answer)
                    if gen_answer.startswith("[") and gen_answer.endswith("]"):
                        gen_answer = gen_answer[1:-1]
                    # P3-b: HaluEval — take first line only, strip trailing period
                    if self.args.datasets.lower() == "halueval":
                        gen_answer = gen_answer.split('\n')[0].strip()
                        if gen_answer.endswith('.'):
                            gen_answer = gen_answer[:-1].strip()
                        # P3-c: yes/no truncation — if answer starts with Yes/No, keep only that word
                        _first = gen_answer.split()[0].rstrip('.,!?') if gen_answer else ''
                        if _first.lower() in ('yes', 'no'):
                            gen_answer = _first.capitalize()
                        
                    if gen_ids.numel() > 0 and hasattr(out_pure, 'scores'):
                        step_logprobs = []
                        for step_idx, step_logits in enumerate(out_pure.scores):
                            token_id = gen_ids[step_idx]
                            log_probs = torch.nn.functional.log_softmax(step_logits[0].float(), dim=-1)
                            step_logprobs.append(log_probs[token_id].item())
                        logprob = sum(step_logprobs) / len(step_logprobs)
                    else:
                        logprob = 0.0
                
                # P2: output raw answer without Answer:[...] wrapper (hurts Token-F1)
                final_output = gen_answer
                token_stats["output"] += len(gen_ids)

            # ==========================================
            # ==========================================
            else:
                if "\\boxed{" in reasoning_history:
                    base_input = p_str.replace(self.tokenizer.eos_token, "").strip() + f"\n\nBased on your detailed internal reasoning: '{reasoning_history.strip()}'\nNow, "
                else:
                    base_input = p_str.replace(self.tokenizer.eos_token, "").strip() + "\n" + reasoning_history

                question_text = example.get('question') or example.get('input')

                _scaf = None
                if _JUDGE_PROMPTS_LOADED:
                    _scaf = _get_stage1_scaffold(
                        self.args.datasets, getattr(self, 'judge_prompt_version', 'v0'))
                _s1_suffix = (_scaf or {}).get("s1_suffix") or (
                    "\nCRITICAL RULE: Do NOT repeat the context. You MUST use bullet points "
                    "to list the objective visual facts.\nVisual Checklist:\n- ")
                _s1_hdr = (_scaf or {}).get("header") or "Visual Checklist:\n- "
                if (_scaf or {}).get("s1_dynamic") and _JUDGE_PROMPTS_LOADED:
                    _dyn_sfx, _dyn_hdr = _cmm_stage1_heads(
                        question_text, getattr(self, 'judge_prompt_version', 'v0'))
                    if _dyn_sfx and _dyn_hdr:
                        _s1_suffix, _s1_hdr = _dyn_sfx, _dyn_hdr
                inputs_text_s1 = base_input + _s1_suffix
                
                m_inputs_s1 = self.processor(
                    text=[inputs_text_s1],
                    images=[image_values] if image_values is not None else None,
                    videos=[pixel_values] if pixel_values is not None else None,
                    audio=[audio_values] if audio_values is not None else None,
                    fps=[1.0] if pixel_values is not None else None, return_tensors="pt",
                    **_aiv_kw(pixel_values, audio_values)
                ).to(self.model.device)

                token_stats["input"] += m_inputs_s1.input_ids.numel()
                
                with torch.inference_mode():
                    _fc_s1, _fc_tf1 = self._format_procs(
                        "stage1", anti_copy_text=(system_prompt or "") + "\n" + _s1_suffix,
                        s1_prefix=_s1_suffix)
                    out_s1 = safe_model_generate(
                        self.model, self.processor, m_inputs_s1,
                        max_new_tokens=200,
                        do_sample=False,
                        stop_strings=([] if _fc_tf1 is not None
                                      else ["Judgment:", "\n\n", "Human:"]),
                        repetition_penalty=1.1,
                        tokenizer=self.tokenizer,
                        _args=self.args,
                        **self._fc_kwargs(_fc_s1, _fc_tf1)
                    )
                    
                    s1_gen_ids = out_s1.sequences[0][m_inputs_s1.input_ids.shape[1]:]
                    gen_desc = self.tokenizer.decode(s1_gen_ids, skip_special_tokens=True).strip()
                    clean_desc = gen_desc.split("Judgment:")[0].split("Final")[0].replace("\\boxed{Yes}", "").replace("\\boxed{No}", "").strip()

                # ==========================================================
                # ==========================================================
                inputs_text_s2 = inputs_text_s1 + " " + clean_desc + self._s2_tail(_scaf, question_text)
                
                m_inputs_s2 = self.processor(
                    text=[inputs_text_s2],
                    images=[image_values] if image_values is not None else None,
                    videos=[pixel_values] if pixel_values is not None else None,
                    audio=[audio_values] if audio_values is not None else None,
                    fps=[1.0] if pixel_values is not None else None, return_tensors="pt",
                    **_aiv_kw(pixel_values, audio_values)
                ).to(self.model.device)

                with torch.inference_mode():
                    _fc_s2, _fc_tf2 = self._format_procs("stage2")
                    out_s2 = safe_model_generate(
                        self.model, self.processor, m_inputs_s2,
                        max_new_tokens=20,
                        do_sample=False,
                        stop_strings=["}", "Final"],
                        tokenizer=self.tokenizer,
                        _args=self.args,
                        **self._fc_kwargs(_fc_s2, _fc_tf2)
                    )

                    s2_gen_ids = out_s2.sequences[0][m_inputs_s2.input_ids.shape[1]:]
                    raw_judg = self.tokenizer.decode(s2_gen_ids, skip_special_tokens=True).strip()

                    import re
                    match = re.search(r'(Yes|No)', raw_judg, re.IGNORECASE)
                    clean_judg = match.group(1).capitalize() if match else "No"
                    
                    if s2_gen_ids.numel() > 0 and hasattr(out_s2, 'scores'):
                        step_logprobs = []
                        for step_idx, step_logits in enumerate(out_s2.scores):
                            token_id = s2_gen_ids[step_idx]
                            log_probs = torch.nn.functional.log_softmax(step_logits[0].float(), dim=-1)
                            step_logprobs.append(log_probs[token_id].item())
                        logprob = sum(step_logprobs) / len(step_logprobs)
                    else:
                        logprob = 0.0

                _hdr = _s1_hdr
                final_output = f"{_hdr}{clean_desc}\nJudgment: \\boxed{{{clean_judg}}}"
                token_stats["output"] += (len(s1_gen_ids) + len(s2_gen_ids))

            all_responses.append(final_output)
            all_logprobs.append(logprob)
            
            if self.args.datasets.lower() in ["avh", "halo", "ragtruth", "halueval", "pubmedqa"]:
                all_advantages.append(logprob)
            else:
                all_advantages.append(previous_values[beam_idx]) 
            
            rollout_stats["total"] += 1
            torch.cuda.empty_cache()

        selected_idx = self.select_response(all_responses, all_logprobs, all_advantages)

        _tb_mode = getattr(self.args, 'final_tiebreak', 'off')
        if _tb_mode == 'r5':
            _sec = []
            for _fs in (traj_info.get('foresight_part') or []):
                _sec.extend(((_fs.get('second_stage') or {}).get('completions') or []))
            _new_idx = resolve_final_tiebreak(all_responses, _sec, selected_idx)
            if _new_idx != selected_idx:
                traj_info['r5_override'] = {'from': int(selected_idx), 'to': int(_new_idx),
                                            'n_votes': len(_sec)}
                selected_idx = _new_idx

        final_response = all_responses[selected_idx]
        tv_info = self._tree_vote(final_response)
        if tv_info.get('applied'):
            final_response = tv_info['response']

        # -- Final-pool 'yes if any' rule (--final_pool_yes_if_any, off by default) --
        pool_info = {'applied': False}
        if (getattr(self.args, 'final_pool_yes_if_any', False)
                and str(getattr(self.args, 'datasets', '')).lower() in self._TV_JUDGE_SETS):
            _cur_v = self._tv_verdict(final_response)
            if _cur_v != 'yes':
                for _i, _txt in enumerate(all_responses):
                    if self._tv_verdict(_txt) == 'yes':
                        pool_info = {'applied': True, 'from': _cur_v,
                                     'to': 'yes', 'src_idx': int(_i)}
                        final_response = _txt
                        break

        traj_info['final_part'] = {
            'response': final_response,
            'raw_completion_text': all_responses,
            'logprobs': [round(float(x), 4) for x in all_logprobs],
            'selected_idx': int(selected_idx),
            'tree_vote': tv_info,
            'pool_yes_if_any': pool_info,
        }

        return {
            "response": final_response,
            "trajectories": all_responses,
            "rollout_stats": rollout_stats,
            "token_stats": token_stats,
            "traj_info": traj_info
        }

    def _prepare_chat_template(self, example, system_prompt):
        """Structured chat template matching the Qwen2.5-Omni processor's requirements."""
        question = example.get('input') or example.get('question')
        user_content_list = []
        dataset_name = self.args.datasets.lower()

        if example.get('image_values') is not None:
            user_content_list.append({"type": "image"})
        if example.get('pixel_values') is not None:
            user_content_list.append({"type": "video"})
        if example.get('audio_values') is not None and not _aiv_audio_folded(example):
            user_content_list.append({"type": "audio"})

        if "cmm" in dataset_name:
            user_content_list.append({
                "type": "text", 
                "text": (
                    f"Task: Verify the existence of '{question}' through independent modal cross-checking.\n"
                    "Step 1: Check the video stream for visual evidence.\n"
                    "Step 2: Check the audio stream for auditory evidence.\n"
                    "Step 3: Compare both sources. Only confirm 'Yes' if there is direct, non-conflicting evidence. "
                    "If one modality suggests it but the other contradicts, prioritize empirical observation over intuition.\n"
                    "Please solve the problem step by step based on the provided media content."
                )
            })
        elif "avh" in dataset_name:
            user_content_list.append({
                "type": "text", 
                "text": f"Based STRICTLY on the raw visual and auditory evidence provided (do not guess or assume), answer this Question: {question}"
            })
        elif "phd" in dataset_name:
            user_content_list.append({
                "type": "text", 
                "text": f"Question: {question}\nPlease solve the problem step by step based on the provided media content."
            })
        elif "halo" in dataset_name:
            user_content_list.append({
                "type": "text", 
                "text": (
                    f"Question: {question}\n"
                    "Act as a meticulous visual auditor. Before answering, consider the possibility that the object is absent. "
                    "Only confirm its presence if you can identify unique, non-generic visual features that distinguish it from the background. "
                    "Prioritize precision over recall; avoid any common-sense completions.\n"
                    "Please solve the problem step by step based on the provided media content."
                )
            })
        elif dataset_name in ["ragtruth", "halueval", "pubmedqa", "summedits"]:
            passage = example.get('passage', '') or example.get('context', '')
            user_content_list.append({
                "type": "text",
                "text": (
                    f"Passage: {passage}\n"
                    f"Question: {question}\n"
                    "Please reason step by step based on the provided passage to arrive at the correct answer."
                )
            })
        else:
            user_content_list.append({
                "type": "text",
                "text": f"Question: {question}"
            })

        return [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": user_content_list},
            {"role": "assistant", "content": [{"type": "text", "text": ""}]}
        ]
        

    def _prepare_chat_template_for_first_stage(self, example, system_prompt):
        """Structured template used for first-stage sampling, keeping multimodal
        placeholders and dataset gating separate."""
        question = example.get('input') or example.get('question')
        user_content_list = []
        dataset_name = self.args.datasets.lower()

        if example.get('image_values') is not None:
            user_content_list.append({"type": "image"})
        if example.get('pixel_values') is not None:
            user_content_list.append({"type": "video"})
        if example.get('audio_values') is not None and not _aiv_audio_folded(example):
            user_content_list.append({"type": "audio"})

        if "cmm" in dataset_name:
            user_content_list.append({
                "type": "text", 
                "text": (
                    f"Task: Verify the existence of '{question}' through independent modal cross-checking.\n"
                    "Step 1: Check the video stream for visual evidence.\n"
                    "Step 2: Check the audio stream for auditory evidence.\n"
                    "Step 3: Compare both sources. Only confirm 'Yes' if there is direct, non-conflicting evidence. "
                    "If one modality suggests it but the other contradicts, prioritize empirical observation over intuition.\n"
                    "Reason step by step. If previous steps are provided, continue logically."
                )
            })
        elif "phd" in dataset_name:
            user_content_list.append({
                "type": "text", 
                "text": (
                    f"Background Text (Unverified): {question}\n"
                    "Please test the final question against the image objectively.\n"
                    "Provide your analysis using bullet points for clarity.\n"
                    "Hypothesis to Test:"
                )
            })
        elif "avh" in dataset_name:
            _step_txt = None
            if _AVH_PROMPTS_LOADED:
                try:
                    from avh_prompts import get_avh_step as _get_avh_step
                    _step_txt = _get_avh_step(
                        getattr(self.args, 'avh_prompt_version', 'v0'), question)
                except Exception as _e:
                    print(f"[WARN] get_avh_step unavailable; falling back to the hardcoded value: {_e}", flush=True)
            if _step_txt is None:
                _step_txt = (f"Based STRICTLY on the raw visual and auditory evidence "
                             f"provided (do not guess or assume), analyze this Question: "
                             f"{question}\nProvide the NEXT reasoning step or description.")
            user_content_list.append({"type": "text", "text": _step_txt})
        elif "halo" in dataset_name:
            user_content_list.append({
                "type": "text", 
                "text": (
                    f"Question: {question}\n"
                    "Act as a meticulous visual auditor. Before answering, consider the possibility that the object is absent. "
                    "Only confirm its presence if you can identify unique, non-generic visual features that distinguish it from the background. "
                    "Prioritize precision over recall; avoid any common-sense completions. "
                    "Please solve the problem step by step based on the provided media content."
                )
            })
        elif dataset_name in ["ragtruth", "halueval", "pubmedqa", "summedits"]:
            passage = example.get('passage', '') or example.get('context', '')
            user_content_list.append({
                "type": "text",
                "text": (
                    f"Passage: {passage}\n"
                    f"Question: {question}\n"
                    "Please reason step by step based on the provided passage to arrive at the correct answer."
                )
            })
        else:
            user_content_list.append({
                "type": "text", 
                "text": f"Question: {question}\nPlease continue the reasoning."
            })

        return [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": user_content_list},
            {"role": "assistant", "content": [{"type": "text", "text": ""}]}
        ]
        

def run_ablation_suite(args, decoder):
    """Run multiple ablation configs sequentially, model loaded once."""
    import subprocess, copy

    suite_path = args.ablation_suite
    with open(suite_path, 'r') as f:
        configs = json.load(f)
    print(f"\n{'='*60}")
    print(f"[ABLATION SUITE] Loaded {len(configs)} configs from {suite_path}")
    print(f"{'='*60}\n", flush=True)

    # Load dataset once
    with open(args.data_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    end_idx = args.max_samples if args.max_samples > 0 else len(test_data)
    test_data = test_data[:end_idx]
    total_samples = len(test_data)
    print(f"Dataset: {args.data_path} ({total_samples} samples)\n", flush=True)

    # Save original args for restoration
    original_args = copy.deepcopy(vars(args))

    suite_start = time.time()
    results_summary = []

    for cfg_idx, cfg in enumerate(configs):
        name = cfg["name"]
        cfg_args = cfg.get("args", {})
        output_dir = cfg["output_dir"]
        file_name = cfg["file_name"]

        print(f"\n{'='*60}")
        print(f"[ABLATION {cfg_idx+1}/{len(configs)}] {name}")
        print(f"  args override: {cfg_args}")
        print(f"  output: {output_dir}/{file_name}.json")
        print(f"  started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}\n", flush=True)

        # Restore original args then apply overrides
        for k, v in original_args.items():
            setattr(args, k, v)
        for k, v in cfg_args.items():
            setattr(args, k, v)
        args.output_dir = output_dir
        args.file_name = file_name

        os.makedirs(output_dir, exist_ok=True)
        time_dir = os.path.join(os.path.dirname(output_dir), "time")
        os.makedirs(time_dir, exist_ok=True)

        output_path = os.path.join(output_dir, f"{file_name}.json")
        error_log_path = os.path.join(output_dir, f"ERROR-{file_name}.log")

        # Checkpoint: skip if already complete
        processed_count = 0
        if os.path.exists(output_path):
            with open(output_path, 'r', encoding='utf-8') as f:
                processed_count = sum(1 for line in f if line.strip())
            if processed_count >= total_samples:
                print(f"[SKIP] {name}: already complete ({processed_count}/{total_samples})")
                results_summary.append((name, "skipped", 0))
                continue
            else:
                print(f"[RESUME] {name}: resuming from {processed_count}/{total_samples}")

        cfg_start = time.time()
        success_count, fail_count = 0, 0

        try:
            with open(output_path, "a", encoding="utf-8") as res_f:
                for i, example in enumerate(test_data):
                    if i < processed_count:
                        continue

                    result = None
                    import gc; gc.collect()
                    torch.cuda.empty_cache()
                    try:
                        sub_cat = example.get("sub_category", "unknown")
                        print(f"\n[{name}] Sample {i+1}/{total_samples} | {sub_cat}", flush=True)
                        system_prompt = decoder.get_system_prompt(
                            args.datasets, question=example.get("question"))
                        result = decoder.process_example(example, system_prompt)
                        success_count += 1

                        _pcs = result.get("traj_info", {}).get("path_confess_stats", {})
                        _n_vetoed = _pcs.get("num_paths_vetoed", 0)
                        _fb = _n_vetoed > 0
                        _fb_src = "none"
                        if _fb:
                            _fb_src = "random_vetoed_path" if (_pcs.get("beam_shrink_occurred") or _pcs.get("cluster_fallback_occurred")) else "best_surviving_path"

                        output_result = {
                            "id": i,
                            "category": example.get("category", ""),
                            "sub_category": sub_cat,
                            "question": example.get("question") or example.get("input"),
                            "ground_truth": example.get("answer") or example.get("target"),
                            "response": result["response"],
                            "status": "success",
                            "path_info": {
                                "fallback_triggered": _fb,
                                "fallback_source": _fb_src,
                                "veto_total_penalty": round(_pcs.get("max_penalty_seen", 0.0), 4),
                                "veto_threshold_used": decoder.thresholds["veto_threshold"],
                                "num_paths_explored": _pcs.get("num_paths_explored", 0),
                                "num_paths_vetoed": _n_vetoed,
                            },
                            "layer_repair": summarize_layer_repair(
                                result.get("traj_info", {})),
                        }
                        res_f.write(json.dumps(output_result, ensure_ascii=False) + "\n")
                        res_f.flush()
                        os.fsync(res_f.fileno())

                    except torch.cuda.OutOfMemoryError:
                        import gc
                        print(f"[{name}] OOM at sample #{i}", flush=True)
                        torch.cuda.empty_cache(); gc.collect()
                        fail_count += 1
                        fail_res = {"id": i, "sub_category": example.get("sub_category", "unknown"), "status": "OOM", "error": "CUDA OOM"}
                        res_f.write(json.dumps(fail_res, ensure_ascii=False) + "\n")
                        res_f.flush()
                        with open(error_log_path, "a") as ef:
                            ef.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | ID {i} | OOM\n")

                    except Exception as e:
                        import traceback
                        print(f"[{name}] Error at sample #{i}: {repr(e)}", flush=True)
                        traceback.print_exc()
                        fail_count += 1
                        fail_res = {"id": i, "sub_category": example.get("sub_category", "unknown"), "status": "failed", "error": repr(e)}
                        res_f.write(json.dumps(fail_res, ensure_ascii=False) + "\n")
                        res_f.flush()
                        with open(error_log_path, "a") as ef:
                            ef.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | ID {i} | {repr(e)}\n")

                    finally:
                        import gc
                        if result:
                            if "traj_info" in result: result["traj_info"].clear()
                            result.clear(); del result
                        gc.collect(); torch.cuda.empty_cache()

        except Exception as e:
            import traceback
            print(f"\n[ABLATION FATAL] {name} crashed: {repr(e)}", flush=True)
            traceback.print_exc()
            results_summary.append((name, f"CRASHED: {repr(e)}", time.time() - cfg_start))
            continue

        cfg_elapsed = time.time() - cfg_start
        print(f"\n[{name}] Done: {success_count} success, {fail_count} fail, {cfg_elapsed:.0f}s", flush=True)

        # Auto-eval
        eval_log_path = os.path.join(os.path.dirname(output_dir), "eval.log")
        if os.path.exists(output_path):
            print(f"[{name}] Running eval...", flush=True)
            try:
                eval_result = subprocess.run(
                    ["python", "GUARD/eval/single_eval.py", "--target_files", output_path],
                    capture_output=True, text=True, timeout=600
                )
                with open(eval_log_path, "w") as ef:
                    ef.write(eval_result.stdout)
                    if eval_result.stderr:
                        ef.write("\n--- stderr ---\n")
                        ef.write(eval_result.stderr)
                print(f"[{name}] Eval done (exit={eval_result.returncode})", flush=True)
            except Exception as eval_e:
                print(f"[{name}] Eval failed: {repr(eval_e)}", flush=True)

        results_summary.append((name, f"ok ({success_count}/{total_samples})", cfg_elapsed))

    # Final summary
    suite_elapsed = time.time() - suite_start
    print(f"\n{'='*60}")
    print(f"[ABLATION SUITE] Complete — {len(configs)} configs, {suite_elapsed:.0f}s total")
    print(f"{'='*60}")
    for name, status, elapsed in results_summary:
        print(f"  {name:20s} | {status:30s} | {elapsed:.0f}s")
    print(f"{'='*60}\n", flush=True)


def summarize_layer_repair(traj_info):
    """Compact summary of Layer-Confess hidden-repair diagnostics.

    The raw list holds one entry per repaired token per explored path, which is
    far too large to persist. E9 (Proposition 2 verification) only needs the
    aggregate: how often repair fired, and the range of the norm ratio and the
    orthogonality residual. Returns None when no repair occurred.
    """
    diags = traj_info.get('layer_repair_diag') or []
    if not diags:
        return None
    ratios = [d['norm_ratio'] for d in diags]
    orth = [abs(d['orth_residual']) for d in diags]
    dperp = [d['delta_perp_norm'] for d in diags]
    n = len(ratios)
    return {
        'n_repairs': n,
        'norm_ratio_mean': round(sum(ratios) / n, 6),
        'norm_ratio_min': round(min(ratios), 6),
        'norm_ratio_max': round(max(ratios), 6),
        'max_abs_orth_residual': float(f"{max(orth):.3e}"),
        'delta_perp_norm_mean': round(sum(dperp) / n, 4),
    }


def _stable_sample_seed(base_seed, example):
    """Derive the seed from the sample content rather than from its position.

    The original implementation seeded numpy once at initialisation, after which
    the global stream was consumed sample by sample, so the draws a given sample
    received depended on how many draws every preceding sample had consumed. That
    made sharded runs and full runs incomparable.
    """
    import hashlib
    key = json.dumps(
        {k: example.get(k) for k in
         ("question", "answer", "sub_category", "image", "audio", "video", "id")},
        sort_keys=True, ensure_ascii=False, default=str,
    )
    h = int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:8], 16)
    return (int(base_seed) + h) % (2 ** 31 - 1)


def main():
    """Entry point. Supports resuming from a checkpoint, isolating per-sample
    exceptions, and keeping the several output paths aligned."""
    global AUDIO_SR
    global USE_AUDIO_IN_VIDEO
    args = parse_arguments()
    AUDIO_SR = int(args.audio_sr)
    print(f"🔊 audio sample rate AUDIO_SR = {AUDIO_SR} "
          + ("(matches the model's WhisperFeatureExtractor rate of 16000)" if AUDIO_SR == 16000
             else "(legacy default; the model expects 16000, see --audio_sr)"),
          flush=True)
    USE_AUDIO_IN_VIDEO = bool(getattr(args, 'use_audio_in_video', False))
    print(f"🎬 audio/video layout USE_AUDIO_IN_VIDEO = {USE_AUDIO_IN_VIDEO} "
          + ("(time-interleaved, roughly one chunk every 2s; modality indices located by token id)" if USE_AUDIO_IN_VIDEO
             else "(default: one contiguous video block followed by one contiguous audio block)"),
          flush=True)
    decoder = PhiDecoder(args)

    if args.ablation_suite:
        run_ablation_suite(args, decoder)
        return

    with open(args.data_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
        
    start_idx = 0
    end_idx = args.max_samples if args.max_samples > 0 else len(test_data)
    test_data = test_data[start_idx : end_idx]
    print(f"📊 data slice: {start_idx} ~ {end_idx}, {len(test_data)} samples")
        
    total_samples = len(test_data)
    print(f"📊 total samples to process: {total_samples}")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.time_path, exist_ok=True)
    
    output_path = os.path.join(args.output_dir, f"{args.file_name}.json")
    traj_path_jsonl = os.path.join(args.time_path, f"TRAJ_INFO-{args.file_name}.jsonl")
    error_log_path = os.path.join(args.output_dir, f"ERROR-{args.file_name}.log")

    processed_count = 0
    print(f"🔍 probing progress file: {output_path}")
    
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            processed_count = sum(1 for line in f if line.strip())
        
        if processed_count >= total_samples:
            print(f"✅ all samples already processed ({processed_count}/{total_samples})。")
            return
        else:
            print(f"🔄 resuming: existing progress {processed_count}/{total_samples}。")
            print(f"⏩ resuming from index #{processed_count} (item {processed_count + 1}) ...", flush=True)
    else:
        print(f"🆕 no existing output found; starting from the first sample.", flush=True)

    start_time = time.time()
    total_stats = {
        "total_rollouts": 0, "saved_rollouts": 0,
        "input_tokens": 0, "output_tokens": 0,
        "success_count": 0, "fail_count": 0
    }

    
    with open(output_path, "a", encoding="utf-8") as res_f:
        for i, example in enumerate(test_data):
            if i < processed_count:
                continue

            result = None
            if not getattr(args, 'disable_per_sample_seed', False):
                _ss = _stable_sample_seed(args.seed, example)
                np.random.seed(_ss)
                torch.manual_seed(_ss)
            # P3-c: Pre-sample cache clear to prevent OOM accumulation (esp. PubMedQA)
            import gc; gc.collect()
            torch.cuda.empty_cache()
            try:
                sub_cat = example.get("sub_category", "unknown")
                print(f"\n🚀 [Progress {i+1}/{total_samples}] processing sample #{i} | subcategory: {sub_cat}", flush=True)
                system_prompt = decoder.get_system_prompt(
                    args.datasets, question=example.get("question"))
                
                result = decoder.process_example(example, system_prompt)

                total_stats["total_rollouts"] += result["rollout_stats"]["total"]
                total_stats["saved_rollouts"] += result["rollout_stats"]["saved"]
                total_stats["input_tokens"] += result["token_stats"]["input"]
                total_stats["output_tokens"] += result["token_stats"]["output"]
                total_stats["success_count"] += 1

                _pcs = result.get("traj_info", {}).get("path_confess_stats", {})
                _n_vetoed = _pcs.get("num_paths_vetoed", 0)
                _fallback_triggered = _n_vetoed > 0
                if not _fallback_triggered:
                    _fallback_source = "none"
                elif _pcs.get("beam_shrink_occurred") or _pcs.get("cluster_fallback_occurred"):
                    _fallback_source = "random_vetoed_path"
                else:
                    _fallback_source = "best_surviving_path"

                output_result = {
                    "id": i,
                    "category": example.get("category", ""),
                    "sub_category": sub_cat,
                    "question": example.get("question") or example.get("input"),
                    "ground_truth": example.get("answer") or example.get("target"),
                    "response": result["response"],
                    "status": "success",
                    "path_info": {
                        "fallback_triggered": _fallback_triggered,
                        "fallback_source": _fallback_source,
                        "veto_total_penalty": round(_pcs.get("max_penalty_seen", 0.0), 4),
                        "veto_threshold_used": decoder.thresholds["veto_threshold"],
                        "num_paths_explored": _pcs.get("num_paths_explored", 0),
                        "num_paths_vetoed": _n_vetoed,
                    },
                    "layer_repair": summarize_layer_repair(
                        result.get("traj_info", {})),
                }
                res_f.write(json.dumps(output_result, ensure_ascii=False) + "\n")
                res_f.flush()
                os.fsync(res_f.fileno())

                if getattr(args, 'dump_probe_features', '') and "traj_info" in result:
                    _pf = result["traj_info"].pop('_probe_feats', None)
                    _pm = result["traj_info"].pop('_probe_meta', None)
                    if _pf:
                        _pd = os.path.join(args.dump_probe_features, args.file_name)
                        os.makedirs(_pd, exist_ok=True)
                        np.savez_compressed(
                            os.path.join(_pd, f"{i:06d}.npz"),
                            meta=json.dumps({
                                'question_idx': i,
                                'ground_truth': result["traj_info"].get('ground_truth'),
                                'steps': _pm}, ensure_ascii=False),
                            **{f"step{k}": v for k, v in enumerate(_pf) if v is not None})

                if args.record_process and "traj_info" in result:
                    result["traj_info"]["question_idx"] = i
                    with open(traj_path_jsonl, "a", encoding="utf-8") as f_traj:
                        f_traj.write(json.dumps(result["traj_info"], ensure_ascii=False) + "\n")

                print(f"✅ index #{i} done. {total_stats['success_count']} new results so far.")

            except torch.cuda.OutOfMemoryError as oom_e:
                import gc
                print(f"\n💥 [OOM] index #{i} ran out of CUDA memory; cleaning up and skipping.", flush=True)
                torch.cuda.empty_cache()
                gc.collect()
                total_stats["fail_count"] += 1
                with open(error_log_path, "a", encoding="utf-8") as f_err:
                    f_err.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | ID {i} | OOM | {repr(oom_e)}\n")
                fail_res = {"id": i, "sub_category": example.get("sub_category", "unknown"), "status": "OOM", "error": "CUDA OOM"}
                res_f.write(json.dumps(fail_res, ensure_ascii=False) + "\n")
                res_f.flush()

            except Exception as e:
                import traceback
                print(f"\n💥 [traceback] index #{i} raised:", flush=True)
                traceback.print_exc()

                total_stats["fail_count"] += 1
                print(f"❌ index #{i} crashed: {repr(e)}", flush=True)
                with open(error_log_path, "a", encoding="utf-8") as f_err:
                    f_err.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | ID {i} | {repr(e)}\n")
                    f_err.write(traceback.format_exc() + "\n")

                fail_res = {"id": i, "sub_category": example.get("sub_category", "unknown"), "status": "failed", "error": repr(e)}
                res_f.write(json.dumps(fail_res, ensure_ascii=False) + "\n")
                res_f.flush()

            finally:
                import gc
                if result:
                    if "traj_info" in result: result["traj_info"].clear()
                    result.clear()
                    del result
                gc.collect()
                torch.cuda.empty_cache()
                print(f"🧹 index #{i} cleaned up.")

    time_span = time.time() - start_time
    time_info_path = os.path.join(args.time_path, f"{args.file_name}.txt")
    with open(time_info_path, "w", encoding="utf-8") as f_summary:
        f_summary.write(f'time:  {time_span}\n')
        f_summary.write(f'total_samples: {total_samples}\n')
        f_summary.write(f'success: {total_stats["success_count"]}\n')
        f_summary.write(f'all_input_tokens: {total_stats["input_tokens"]}\n')
        f_summary.write(f'all_output_tokens: {total_stats["output_tokens"]}\n')

    print(f"\n🏁 pipeline finished. succeeded: {total_stats['success_count']} | failed: {total_stats['fail_count']}")
    print(f"summary written to: {time_info_path}")

if __name__ == "__main__":
    main()