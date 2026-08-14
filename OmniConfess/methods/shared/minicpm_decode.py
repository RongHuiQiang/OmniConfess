# -*- coding: utf-8 -*-
"""
minicpm_decode.py -- per-token decoding infrastructure for MiniCPM-o-2.6.

Why this exists
---------------
MiniCPM-o's model.chat() has two limitations that make contrastive decoding
methods impossible to build on top of it:

1. chat(sampling=False) internally runs beam search (num_beams=3,
   repetition_penalty=1.2), not greedy decoding;
2. chat() only forwards keys that already exist in the generation config, so
   arguments such as logits_processor are silently dropped.

Methods like DoLa, VCD, CAD and AVCD need the logits at every step in order to
intervene, and therefore have to bypass chat(). This module reproduces chat()'s
preprocessing line for line, so the inputs are identical to those base_model
receives, and then exposes a pluggable step-by-step generation loop.
"""
import math
import os
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# --------------------------- Model loading ----------------------------

def patch_smart_apply():
    """Work around an incompatibility between transformers 4.52+ and MiniCPM-o's custom modules."""
    def patched_smart_apply(self, fn):
        for module in self.children():
            if hasattr(module, "_init_weights"):
                if hasattr(module, "_initialize_weights"):
                    module.smart_apply(module._initialize_weights)
                else:
                    module.smart_apply(fn)
            else:
                module.smart_apply(fn)
        fn(self)
        return self
    torch.nn.Module.smart_apply = patched_smart_apply


# ------------------- Input preprocessing (reproducing chat()) -------------------

def build_minicpm_inputs(model, processor, msgs, omni_input=False,
                         max_slice_nums=1, use_image_id=False,
                         chunk_input=True, max_inp_length=32768):
    """
    Reproduce MiniCPMO.chat()'s messages -> prompt -> processor pipeline.

    Following it line for line is deliberate: only then are the inputs seen by
    DoLa/VCD and by base_model guaranteed identical, so that any difference comes
    from the decoding process alone.

    Args:
        msgs: a message list of the same shape model.chat(msgs=...) expects, whose
              content may contain PIL.Image, np.ndarray (audio) or str
    Returns:
        inputs: the processor output, moved to model.device with image_sizes removed
    """
    copy_msgs = deepcopy(msgs)
    images, audios, audio_parts = [], [], []
    use_tts_template = False

    for i, msg in enumerate(copy_msgs):
        content = msg["content"]
        if isinstance(content, str):
            content = [content]
        cur_msgs = []
        for c in content:
            if isinstance(c, Image.Image):
                images.append(c)
                cur_msgs.append("(<image>./</image>)")
            elif isinstance(c, np.ndarray):
                audios.append(c)
                audio_parts.append(i)
                cur_msgs.append("(<audio>./</audio>)")
                use_tts_template = True
            elif isinstance(c, str):
                cur_msgs.append(c)
        msg["content"] = "".join(cur_msgs) if omni_input else "\n".join(cur_msgs)

    prompt = processor.tokenizer.apply_chat_template(
        copy_msgs,
        tokenize=False,
        add_generation_prompt=True,
        chat_template=model.default_tts_chat_template if use_tts_template else None,
    )

    inputs = processor(
        [prompt], [images], [audios], [audio_parts],
        max_slice_nums=max_slice_nums,
        use_image_id=use_image_id,
        chunk_input=chunk_input,
        return_tensors="pt",
        max_length=max_inp_length,
    ).to(model.device)
    inputs.pop("image_sizes", None)
    return inputs


_EMBED_KEYS = ("input_ids", "pixel_values", "tgt_sizes", "image_bound",
               "audio_features", "audio_feature_lens", "audio_bounds", "spk_bounds")


def get_multimodal_embeds(model, inputs):
    """
    Obtain the fused embedding through MiniCPM's own get_vllm_embedding and
    get_omni_embedding, exactly as MiniCPMO.generate() does internally.
    """
    model_inputs = {k: inputs[k] for k in _EMBED_KEYS if k in inputs}
    with torch.inference_mode():
        embeds, _ = model.get_vllm_embedding(model_inputs)
        if getattr(model.config, "init_audio", False) and hasattr(model, "apm"):
            model_inputs["inputs_embeds"] = embeds
            embeds = model.get_omni_embedding(
                model_inputs,
                input_embeddings=embeds,
                chunk_length=model.config.audio_chunk_length,
            )
    return embeds


def get_terminators(model, tokenizer):
    return [tokenizer.convert_tokens_to_ids(t) for t in model.terminators]


# ------------------- Per-layer logits hook (used by DoLa) -------------------

class PrematureLayerCapture:
    """
    Attach a forward hook to the given layer and take that layer's hidden state
    through norm + lm_head to obtain logits. Structurally the same as the
    DualLayerAuditor used on the OmniConfess side, keeping only what DoLa needs.
    """

    def __init__(self, model, layers):
        self.model = model
        self.layers = list(layers)
        self.logits = {}
        self.handles = []

    def _make_hook(self, idx):
        def _hook(module, _input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            with torch.no_grad():
                normed = self.model.llm.model.norm(hidden)
                self.logits[idx] = self.model.llm.lm_head(normed)[:, -1, :]
        return _hook

    def register(self):
        for idx in self.layers:
            h = self.model.llm.model.layers[idx].register_forward_hook(self._make_hook(idx))
            self.handles.append(h)
        return self

    def clear(self):
        self.logits = {}

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []
        self.logits = {}


# --------------------------- VCD visual perturbation ----------------------------

def add_diffusion_noise(image_tensor, noise_step):
    """VCD's diffusion noising, matching the official implementation."""
    num_steps = 1000
    betas = torch.linspace(-6, 6, num_steps)
    betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5
    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, dim=0)
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

    t = int(noise_step)
    noise = torch.randn_like(image_tensor)
    a = alphas_bar_sqrt[t].to(image_tensor.device, image_tensor.dtype)
    b = one_minus_alphas_bar_sqrt[t].to(image_tensor.device, image_tensor.dtype)
    return a * image_tensor + b * noise


def noise_visual_inputs(inputs, noise_mode="gaussian", noise_std=0.1, noise_step=500):
    """
    Add noise to pixel_values in the inputs and return a new dict; returns None
    when there is no visual input.

    MiniCPM's pixel_values is a list of lists of tensors, since each image may be
    sliced, so this recurses.

    noise_mode:
        'gaussian'  -- matches the Qwen-side base_vcd.py (std = 0.1 Gaussian noise
                       plus a clamp). This is the default and keeps the two
                       backbones comparable.
        'diffusion' -- the diffusion noising used by the original VCD paper and by
                       the official AVCD implementation.
        'none'      -- build no contrast branch; the caller degenerates to plain
                       greedy. This exists for a same-decoder control: base_model
                       goes through model.chat() with beam 3 and repetition penalty
                       1.2, whereas the contrastive decoding here is plain greedy,
                       so the two cannot simply be subtracted. Running 'none' once
                       separates the decoder difference from the effect of
                       contrastive decoding itself.
    """
    if noise_mode == "none":
        return None
    if "pixel_values" not in inputs:
        return None
    pv = inputs["pixel_values"]

    def _noise(obj):
        if isinstance(obj, torch.Tensor):
            if noise_mode == "diffusion":
                return add_diffusion_noise(obj, noise_step)
            noise = torch.randn_like(obj) * noise_std
            return torch.clamp(obj + noise, -10.0, 10.0)
        if isinstance(obj, (list, tuple)):
            return [_noise(x) for x in obj]
        return obj

    if not any(isinstance(x, torch.Tensor) for x in _flatten(pv)):
        return None

    out = dict(inputs)
    out["pixel_values"] = _noise(pv)
    return out


def _flatten(obj):
    if isinstance(obj, (list, tuple)):
        for x in obj:
            yield from _flatten(x)
    else:
        yield obj


# --------------------------- Step-by-step generation loop ----------------------------

def _grow_mask(mask):
    """Append a column of ones to the attention_mask; a newly generated token is always visible."""
    return torch.cat(
        [mask, torch.ones((mask.shape[0], 1), device=mask.device, dtype=mask.dtype)],
        dim=-1)


def _apply_repetition_penalty(logits, generated_ids, penalty):
    if penalty == 1.0 or not generated_ids:
        return logits
    ids = torch.tensor(sorted(set(generated_ids)), device=logits.device, dtype=torch.long)
    vals = logits[0, ids]
    logits[0, ids] = torch.where(vals > 0, vals / penalty, vals * penalty)
    return logits


@torch.inference_mode()
def greedy_generate(model, tokenizer, embeds, attention_mask=None,
                    max_new_tokens=512, step_fn=None, aux_branches=None,
                    repetition_penalty=1.0, terminators=None):
    """
    Greedy step-by-step generation that lets the caller rewrite the logits at each
    step.

    Args:
        embeds: fused embedding of the main branch prompt, [1, L, d]
        step_fn: optional callable(mature_logits, aux_logits, step) -> logits.
                 aux_logits holds this step's logits for each aux branch. DoLa uses
                 it for the mature/premature contrast; VCD and CAD use the aux
                 branches.
        aux_branches: prompt embeddings of the extra branches, such as VCD's noised
                      visual branch. Each keeps its own KV cache and is fed the
                      same generated tokens.
    Returns:
        (text, generated_ids)
    """
    llm = model.llm
    embed_layer = llm.get_input_embeddings()

    past = None
    cur = embeds
    mask = attention_mask
    generated = []

    aux_state = []
    for aux_spec in (aux_branches or []):
        if isinstance(aux_spec, dict):
            aux_state.append({"past": None, "cur": aux_spec["embeds"],
                              "mask": aux_spec.get("attention_mask")})
        else:
            aux_state.append({"past": None, "cur": aux_spec, "mask": None})

    for step in range(max_new_tokens):
        out = llm(inputs_embeds=cur, attention_mask=mask,
                  past_key_values=past, use_cache=True, return_dict=True)
        past = out.past_key_values
        logits = out.logits[:, -1, :].float()

        aux_logits = []
        for st in aux_state:
            a_out = llm(inputs_embeds=st["cur"], attention_mask=st["mask"],
                        past_key_values=st["past"], use_cache=True, return_dict=True)
            st["past"] = a_out.past_key_values
            aux_logits.append(a_out.logits[:, -1, :].float())

        if step_fn is not None:
            logits = step_fn(logits, aux_logits, step)

        logits = _apply_repetition_penalty(logits, generated, repetition_penalty)

        next_id = int(logits.argmax(dim=-1).item())
        if terminators and next_id in terminators:
            break
        generated.append(next_id)

        next_tok = torch.tensor([[next_id]], device=embeds.device, dtype=torch.long)
        cur = embed_layer(next_tok)
        for st in aux_state:
            st["cur"] = cur
            if st["mask"] is not None:
                st["mask"] = _grow_mask(st["mask"])
        if mask is not None:
            mask = _grow_mask(mask)

    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return text, generated


# --------------------------- Method-level step_fn ----------------------------


def make_dola_step_fn(capture, premature_layer, alpha=0.1, apc_beta=0.0):
    """
    DoLa: log_softmax(mature) - alpha * log_softmax(premature), matching the
    formula used on the Qwen side. An apc_beta above zero enables the adaptive
    plausibility constraint.
    """
    def step_fn(mature, _aux, _step):
        premature = capture.logits.get(premature_layer)
        capture.clear()
        if premature is None:
            return mature
        out = F.log_softmax(mature, dim=-1) - alpha * F.log_softmax(premature.float(), dim=-1)
        if apc_beta > 0:
            cutoff = math.log(apc_beta) + mature.max(dim=-1, keepdim=True).values
            out = out.masked_fill(mature < cutoff, -float("inf"))
        return out
    return step_fn


def make_contrast_step_fn(alpha=0.1, apc_beta=0.1, log_domain=False):
    """
    VCD / CAD: (1+alpha)*logit_full - alpha*logit_perturbed, plus APC.

    VCD's perturbed branch is the noised visual input and CAD's is the input with
    the evidence context removed; the formula is the same and only the
    construction of aux_branches differs.

    log_domain:
        False (default, preserving the historical behaviour) contrasts the raw
              logits directly;
        True  applies log_softmax to each branch first, matching the Qwen-side
              base_vcd.py formula exactly. Set it to True when comparing VCD or CAD
              across backbones, since otherwise the differing logit scales make the
              same alpha mean different strengths.
    """
    def step_fn(mature, aux, _step):
        if not aux:
            return mature
        if log_domain:
            m = F.log_softmax(mature, dim=-1)
            a = F.log_softmax(aux[0], dim=-1)
        else:
            m, a = mature, aux[0]
        contrast = (1 + alpha) * m - alpha * a
        if apc_beta > 0:
            cutoff = math.log(apc_beta) + mature.max(dim=-1, keepdim=True).values
            contrast = contrast.masked_fill(mature < cutoff, -float("inf"))
        return contrast
    return step_fn


# ═════════════════════════════ AVCD ═════════════════════════════
#
#
#

def get_modality_indices(input_ids, device, image_bound=None, audio_bounds=None):
    """
    Derive visual and audio token indices from the MiniCPM processor's image_bound
    and audio_bounds, which are [[start, end], ...] with end exclusive.
    Reimplemented here so that Baselines does not have to depend on the whole
    OmniConfess module.

    Returns:
        (v_idx, a_idx) -- 1-D long index tensors, or None when the modality is absent.
    """
    def _collect(bounds_obj):
        if bounds_obj is None or len(bounds_obj) == 0:
            return None
        bounds = bounds_obj[0] if isinstance(bounds_obj, list) else bounds_obj
        if not isinstance(bounds, torch.Tensor) or bounds.numel() == 0:
            return None
        spans = [torch.arange(int(row[0]), int(row[1]), device=device) for row in bounds]
        return torch.cat(spans) if spans else None

    return _collect(image_bound), _collect(audio_bounds)


def _prefill_past(llm, embeds, attention_mask):
    """
    Prefill purely to obtain the KV cache, discarding the logits.

    A default forward computes a row of logits for every position, which at
    L ~ 1200 means roughly 700 MB for [1, L, 151936]. Here logits_to_keep=1 makes
    the model compute only the last row where possible, and the output object is
    allowed to go out of scope immediately so its peak memory does not stack with
    the following forward passes.
    """
    for kw in ({"logits_to_keep": 1}, {"num_logits_to_keep": 1}, {}):
        try:
            return llm(inputs_embeds=embeds, attention_mask=attention_mask,
                       use_cache=True, return_dict=True, **kw).past_key_values
        except TypeError:
            continue
    raise RuntimeError("prefill failed for every logits_to_keep variant")


def _last_query_attention(llm, embeds, attention_mask, skip_last_layer=True):
    """
    Two-stage prefill: feed the first L-1 tokens, then feed the last one on its own
    and take the per-layer attention.

    This yields attentions shaped [B, H, 1, L] rather than [B, H, L, L], which is
    the only memory-feasible option on sequences of a thousand tokens or more such
    as avhbench. A single-token forward falls back to eager attention
    automatically, at negligible cost.

    Returns:
        (att, logits, past) -- att is the [L] last-attention row, summed over layers
        and averaged over heads; logits is the main branch's step-0 logits; past is
        the KV cache after the full prefill.
    """
    head, tail = embeds[:, :-1], embeds[:, -1:]
    mask_head = attention_mask[:, :-1] if attention_mask is not None else None

    past_head = _prefill_past(llm, head, mask_head)
    out2 = llm(inputs_embeds=tail, attention_mask=attention_mask,
               past_key_values=past_head, use_cache=True,
               output_attentions=True, return_dict=True)

    attns = out2.attentions
    if not attns:
        return None, out2.logits[:, -1, :].float(), out2.past_key_values

    layers = attns[:-1] if (skip_last_layer and len(attns) > 1) else attns
    att = None
    for a in layers:
        row = a[:, :, -1, :].float().mean(dim=1).squeeze(0)   # [L]
        att = row if att is None else att + row
    return att, out2.logits[:, -1, :].float(), out2.past_key_values


def rank_modality_dominance(att, v_idx, a_idx, seq_len):
    """
    Rank the three modalities by mean attention mass per token and return
    [(name, score), ...] in descending order.

    This follows the official implementation, which divides each span's total
    attention by the span length, i.e. a length-normalised mean, so that a long
    modality does not win merely by having more tokens.
    """
    device = att.device
    taken = torch.zeros(seq_len, dtype=torch.bool, device=device)
    if v_idx is not None:
        taken[v_idx] = True
    if a_idx is not None:
        taken[a_idx] = True
    l_idx = torch.nonzero(~taken, as_tuple=True)[0]

    def _score(idx):
        if idx is None or idx.numel() == 0:
            return float("-inf")
        return att[idx].mean().item()

    scored = [("language", _score(l_idx)),
              ("video", _score(v_idx)),
              ("audio", _score(a_idx))]
    return sorted(scored, key=lambda x: x[1], reverse=True), l_idx


_AVCD_MASK_PLAN = {
    "language": (("video", "audio"), ("audio",), ("video",)),
    "video":    (("language", "audio"), ("audio",), ("language",)),
    "audio":    (("language", "video"), ("video",), ("language",)),
}


def build_avcd_masks(base_mask, att, threshold, idx_map, plan):
    """
    Build the attention masks for the three auxiliary branches according to the
    AVCD rule: within the named modality's span, mask out the tokens whose
    attention is above the threshold, i.e. remove that modality's most influential
    part and keep the periphery.
    """
    masks = []
    for group in plan:
        m = base_mask.clone()
        for name in group:
            idx = idx_map.get(name)
            if idx is None or idx.numel() == 0:
                continue
            hi = idx[att[idx] > threshold]
            if hi.numel() > 0:
                m[0, hi] = 0
        m[0, 0] = 1
        if int(m.sum().item()) == 0:
            m = base_mask.clone()
        masks.append(m)
    return masks


@torch.inference_mode()
def avcd_generate(model, tokenizer, embeds, attention_mask,
                  v_idx=None, a_idx=None, alpha=2.5, apc_beta=0.2,
                  apc_fill=-1e-4, entropy_threshold=0.6,
                  max_new_tokens=512, repetition_penalty=1.0, terminators=None):
    """
    AVCD per-token decoding.

    Each step first computes the entropy of the main branch's logits. Below
    entropy_threshold it decodes greedily, matching the official pass-through
    branch; otherwise it runs the three masked branches and computes

        (2+2a)*orig - 2a*o1 + o2 + o3

    then applies the adaptive plausibility constraint with
    cutoff = log(beta) + max(orig).

    Returns:
        (text, generated_ids, stats)
    """
    llm = model.llm
    embed_layer = llm.get_input_embeddings()
    device = embeds.device
    seq_len = embeds.shape[1]

    if attention_mask is None:
        attention_mask = torch.ones((1, seq_len), device=device, dtype=torch.long)

    # -- prefill, also extracting the last-query attention for dominance and threshold --
    att, logits, past = _last_query_attention(llm, embeds, attention_mask)

    stats = {"gated_steps": 0, "total_steps": 0, "dominant": None,
             "threshold": None, "avcd_active": False, "apc_escape": 0}

    idx_map, masks, aux_state = {}, [], []
    if att is not None and (v_idx is not None or a_idx is not None):
        ranked, l_idx = rank_modality_dominance(att, v_idx, a_idx, seq_len)
        idx_map = {"language": l_idx, "video": v_idx, "audio": a_idx}
        threshold = att.median().item()
        dominant = ranked[0][0]
        plan = _AVCD_MASK_PLAN[dominant]
        masks = build_avcd_masks(attention_mask, att, threshold, idx_map, plan)
        aux_state = [{"past": None, "mask": m, "pending": []} for m in masks]
        stats.update({"dominant": dominant, "threshold": threshold,
                      "avcd_active": True,
                      "ranked": [(n, round(s, 6)) for n, s in ranked]})

    generated = []

    for step in range(max_new_tokens):
        stats["total_steps"] += 1

        if aux_state:
            logp = F.log_softmax(logits, dim=-1)
            entropy = -(logp.exp() * logp).sum(dim=-1).item()
        else:
            entropy = 0.0

        if aux_state and entropy >= entropy_threshold:
            stats["gated_steps"] += 1
            cutoff = math.log(apc_beta) + logits.max(dim=-1, keepdim=True).values
            aux_logits = []
            for st in aux_state:
                aux_logits.append(_avcd_branch_logits(
                    llm, embed_layer, st, embeds, device))
            o1, o2, o3 = aux_logits
            contrast = (2 + 2 * alpha) * logits - 2 * alpha * o1 + o2 + o3
            next_logits = contrast.masked_fill(logits < cutoff, apc_fill)
        else:
            next_logits = logits

        next_logits = _apply_repetition_penalty(next_logits.clone(), generated,
                                                repetition_penalty)
        next_id = int(next_logits.argmax(dim=-1).item())
        if aux_state and entropy >= entropy_threshold and logits[0, next_id] < cutoff[0, 0]:
            stats["apc_escape"] += 1
        if terminators and next_id in terminators:
            break
        generated.append(next_id)

        next_tok = torch.tensor([[next_id]], device=device, dtype=torch.long)
        cur = embed_layer(next_tok)
        for st in aux_state:
            st["pending"].append(cur)
        attention_mask = _grow_mask(attention_mask)

        out = llm(inputs_embeds=cur, attention_mask=attention_mask,
                  past_key_values=past, use_cache=True, return_dict=True)
        past = out.past_key_values
        logits = out.logits[:, -1, :].float()

    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return text, generated, stats


def _forward_keep_last(llm, embeds, attention_mask, past=None):
    """
    Forward a span of tokens, keeping only the last row of logits and the new KV
    cache.

    Numerically identical to calling without logits_to_keep, since lm_head is a
    per-position linear map and the last row is computed the same way; the only
    difference is that no [L, 151936] output is allocated for positions that are
    never used. With AVCD's three auxiliary branches this saves roughly three times
    700 MB of peak memory.
    """
    for kw in ({"logits_to_keep": 1}, {"num_logits_to_keep": 1}, {}):
        try:
            out = llm(inputs_embeds=embeds, attention_mask=attention_mask,
                      past_key_values=past, use_cache=True, return_dict=True, **kw)
            return out.logits[:, -1, :].float(), out.past_key_values
        except TypeError:
            continue
    raise RuntimeError("forward failed for every logits_to_keep variant")


def _avcd_branch_logits(llm, embed_layer, st, embeds, device):
    """
    Get one masked auxiliary branch's logits for the current step, first feeding it
    any tokens accumulated since its last use.

    The branch is only really prefilled on the first call, so if a sample never
    trips the entropy gate those three prefills are skipped entirely.
    """
    if st["past"] is None:
        pend = st["pending"]
        inp = torch.cat([embeds] + pend, dim=1) if pend else embeds
        mask = st["mask"]
        if pend:
            mask = torch.cat(
                [mask, torch.ones((1, len(pend)), device=device, dtype=mask.dtype)],
                dim=-1)
        st["mask"], st["pending"] = mask, []
        st["last"], st["past"] = _forward_keep_last(llm, inp, mask)
        return st["last"]

    pend = st["pending"]
    st["pending"] = []
    if not pend:
        return st["last"]
    inp = torch.cat(pend, dim=1)
    st["mask"] = torch.cat(
        [st["mask"], torch.ones((1, len(pend)), device=device, dtype=st["mask"].dtype)],
        dim=-1)
    st["last"], st["past"] = _forward_keep_last(llm, inp, st["mask"], past=st["past"])
    return st["last"]


# ═════════════════════════════ OPERA ═════════════════════════════
#
# Huang et al., CVPR 2024, "OPERA: Alleviating Hallucination in Multi-Modal Large
# Language Models via Over-Trust Penalty and Retrospection-Allocation"。
#
#
#
#
#
#   num_beams=5, threshold=15, num_attn_candidates=5, penalty_weights=1。
#

def _cache_reorder(past, beam_idx):
    past.reorder_cache(beam_idx)
    return past


@torch.inference_mode()
def opera_beam_generate(model, tokenizer, embeds, attention_mask=None,
                        num_beams=5, scale_factor=50.0, threshold=15,
                        num_attn_candidates=5, penalty_weights=1.0,
                        max_new_tokens=512, terminators=None,
                        length_penalty=1.0, max_rollbacks=8):
    """
    OPERA: beam search with the over-trust penalty and retrospection-allocation.

    Returns:
        (text, generated_ids, stats)
    """
    llm = model.llm
    embed_layer = llm.get_input_embeddings()
    device = embeds.device
    prompt_len = embeds.shape[1]
    term_set = set(terminators or [])

    if attention_mask is None:
        attention_mask = torch.ones((1, prompt_len), device=device, dtype=torch.long)

    stats = {"rollbacks": 0, "penalised_steps": 0, "total_steps": 0}

    need_attn = scale_factor != 0

    # -- prefill once (batch=1), then replicate the KV cache num_beams times --
    logits, past = _forward_keep_last(llm, embeds, attention_mask)
    past.batch_repeat_interleave(num_beams)
    mask = attention_mask.repeat(num_beams, 1)

    beam_scores = torch.full((num_beams,), -1e9, device=device, dtype=torch.float)
    beam_scores[0] = 0.0
    beam_tokens = [[] for _ in range(num_beams)]
    attn_rows = []
    col_hits = {}
    banned = {}
    finished = []                                     # (score, tokens)

    logits = logits.float().expand(num_beams, -1)

    step = 0
    while step < max_new_tokens:
        stats["total_steps"] += 1
        cur_len = prompt_len + len(beam_tokens[0])
        logprobs = F.log_softmax(logits, dim=-1)

        for tok in banned.get(cur_len, ()):
            logprobs[:, tok] = -float("inf")

        # ── Over-Trust Penalty ────────────────────────────────────────────
        penalty = torch.zeros(num_beams, device=device)
        if len(attn_rows) >= num_attn_candidates:
            k = num_attn_candidates
            win_start = cur_len - k
            rows = attn_rows[-k:]
            local = torch.zeros(num_beams, k, k, device=device)
            for i, r in enumerate(rows):
                seg = r[:, win_start:win_start + k]
                local[:, i, :seg.shape[1]] = seg

            strict = torch.tril(torch.ones(k, k, device=device), diagonal=-1).bool()
            counts = strict.sum(dim=0).clamp(min=1).float()
            logv = torch.where(strict, local.clamp_min(1e-12).log(),
                               torch.zeros_like(local))
            gmean = (logv.sum(dim=1) / counts).exp()                 # [B, k]
            gmean[:, -1] = 0.0
            sigma, argcol = gmean.max(dim=-1)
            penalty = penalty_weights * scale_factor * sigma
            stats["penalised_steps"] += 1
            if os.environ.get("OPERA_DEBUG"):
                import sys as _s
                print(f"    [dbg] step={step} penalty={[round(x,3) for x in penalty.tolist()]} "
                      f"gmean0={[round(x,4) for x in gmean[0].tolist()]}", file=_s.stderr)

            # -- retrospection-allocation: track the maximal column of the best beam --
            best_b = int(beam_scores.argmax().item())
            hit_pos = win_start + int(argcol[best_b].item())
            col_hits[hit_pos] = col_hits.get(hit_pos, 0) + 1

            if (col_hits[hit_pos] > threshold
                    and stats["rollbacks"] < max_rollbacks
                    and hit_pos >= prompt_len):
                g = hit_pos - prompt_len
                seq = beam_tokens[best_b]
                if g + 1 < len(seq):
                    banned.setdefault(hit_pos + 1, set()).add(seq[g + 1])
                    kept = seq[:g + 1]
                    past.crop(hit_pos)
                    past.batch_select_indices(
                        torch.full((num_beams,), best_b, device=device,
                                   dtype=torch.long))
                    mask = torch.ones((num_beams, hit_pos), device=device,
                                      dtype=attention_mask.dtype)
                    beam_tokens = [list(kept) for _ in range(num_beams)]
                    beam_scores = torch.full((num_beams,), -1e9, device=device,
                                             dtype=torch.float)
                    beam_scores[0] = 0.0
                    attn_rows = []
                    col_hits = {}
                    stats["rollbacks"] += 1
                    cur = embed_layer(torch.tensor(
                        [[kept[-1]]] * num_beams, device=device, dtype=torch.long))
                    mask = _grow_mask(mask)
                    out = llm(inputs_embeds=cur, attention_mask=mask,
                              past_key_values=past, use_cache=True,
                              output_attentions=need_attn, return_dict=True)
                    past = out.past_key_values
                    logits = out.logits[:, -1, :].float()
                    if need_attn:
                        attn_rows.append(
                            out.attentions[-1][:, :, -1, :].float().mean(1))
                    continue

        # -- beam selection --
        cand = beam_scores[:, None] + logprobs - penalty[:, None]
        flat = cand.view(-1)
        top_scores, top_idx = flat.topk(num_beams * 2, dim=-1)
        vocab = logprobs.shape[-1]

        next_beams, next_tokens, next_scores = [], [], []
        for s, fi in zip(top_scores.tolist(), top_idx.tolist()):
            b, tok = fi // vocab, fi % vocab
            if tok in term_set:
                seq = beam_tokens[b]
                if seq:
                    norm = s / (max(len(seq), 1) ** length_penalty)
                    finished.append((norm, list(seq)))
                continue
            next_beams.append(b)
            next_tokens.append(tok)
            next_scores.append(s)
            if len(next_beams) == num_beams:
                break

        if not next_beams or len(finished) >= num_beams:
            break

        while len(next_beams) < num_beams:
            next_beams.append(next_beams[-1])
            next_tokens.append(next_tokens[-1])
            next_scores.append(-1e9)

        beam_idx = torch.tensor(next_beams, device=device, dtype=torch.long)
        beam_scores = torch.tensor(next_scores, device=device, dtype=torch.float)
        beam_tokens = [beam_tokens[b] + [t] for b, t in zip(next_beams, next_tokens)]

        past = _cache_reorder(past, beam_idx)
        mask = _grow_mask(mask[beam_idx])
        attn_rows = [r[beam_idx] for r in attn_rows][-(num_attn_candidates - 1):] \
            if num_attn_candidates > 1 else []

        cur = embed_layer(torch.tensor([[t] for t in next_tokens],
                                       device=device, dtype=torch.long))
        out = llm(inputs_embeds=cur, attention_mask=mask,
                  past_key_values=past, use_cache=True,
                  output_attentions=need_attn, return_dict=True)
        past = out.past_key_values
        logits = out.logits[:, -1, :].float()
        if need_attn:
            attn_rows.append(out.attentions[-1][:, :, -1, :].float().mean(1))
        step += 1

    if finished:
        best = max(finished, key=lambda x: x[0])[1]
    else:
        b = int(beam_scores.argmax().item())
        best = beam_tokens[b]

    text = tokenizer.decode(best, skip_special_tokens=True).strip()
    return text, best, stats
