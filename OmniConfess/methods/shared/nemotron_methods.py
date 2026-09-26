# -*- coding: utf-8 -*-
"""
nemotron_methods.py -- shared plumbing for the Nemotron baseline method runners
(cad / dola / vcd / avcd / opera / tot / gd / pd). Extends shared/nemotron_shim.py
with: processor-input building (copied from base_sc_nemotron_hf.py), LLM-level
prefill on fused inputs_embeds, single-step decode on the hybrid cache, an outer
generate() helper with per-token logp, and hybrid-cache surgery for OPERA.

Structure notes (probed on this checkpoint, transformers 5.6.2 trust_remote_code):
  * model.language_model is a standard CausalLM interface: accepts
    inputs_embeds/attention_mask/past_key_values/use_cache; position_ids are
    1-D and auto-derived from cache_position (NO mRoPE -> no position hooking
    needed, unlike the Qwen arms).
  * NemotronHHybridDynamicCache has key_cache/value_cache lists (attention
    layers, [B,H,S,D]) plus conv_states/ssm_states dicts keyed by layer idx
    (mamba layers, batch-first). Mamba states update IN PLACE on every forward:
    any probe forward must snapshot+restore them (OPERA).
  * Attention impl is eager at load (load_nemotron), and eager attention always
    returns attn_weights as output[1] -> OPERA/AVCD read them with a plain
    forward hook on the attention module, no forward patching needed.
  * Only E blocks (attention) exist among MEMEM* hybrid blocks; Mamba blocks
    carry no attention -> AVCD masking is applied on the attention subset only.
"""
import os
import torch
import torch.nn.functional as F

from shared.data_loader import load_image
from shared.nemotron_shim import fuse_inputs_embeds
from base_model_nemotron_hf_pv import (
    load_video_frames_native, load_audio_wave, set_seed,
)

NEMOTRON_EOS_ID = 11  # config.json eos_token_id (both wrapper and llm_config)


def build_processor_inputs(processor, tokenizer, item, args, system_prompt,
                           device, passage_override=None, drop_media=False):
    """Build the multimodal processor inputs for one item (media + prompt).
    Same as base_sc_nemotron_hf.build_processor_inputs plus two knobs used by
    CAD's no-evidence branch: passage_override replaces the passage string and
    drop_media removes media tokens/inputs entirely.
    Returns (inputs_dict_on_device, input_len, media_flags) or raises."""
    from shared import resolve_media_path, get_user_prompt

    json_dir = os.path.dirname(args.data_path)
    videos = images = audio = None
    has_video = has_audio = has_image = False
    if not drop_media:
        if item.get('video_path'):
            vp = resolve_media_path(item['video_path'], json_dir, args.data_root)
            videos = load_video_frames_native(vp)
            has_video = videos is not None
        if item.get('audio_path'):
            ap = resolve_media_path(item['audio_path'], json_dir, args.data_root)
            audio = load_audio_wave(ap)
            has_audio = audio is not None
        if item.get('image_path'):
            ip = resolve_media_path(item['image_path'], json_dir, args.data_root)
            images = load_image(ip)
            has_image = images is not None

    passage = item.get('passage') if passage_override is None else passage_override
    user_prompt = get_user_prompt(
        item['question'], args.datasets, args.prompt_version, passage=passage)
    media_prefix = ''
    if has_video:
        media_prefix += '<video>\n'
    if has_audio:
        media_prefix += '<so_embedding>\n'
    if has_image:
        media_prefix += '<image>\n'
    user_text = (media_prefix + user_prompt) if media_prefix else user_prompt
    messages = [{'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_text}]
    try:
        text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
            enable_thinking=getattr(args, 'enable_thinking', False))
    except TypeError:
        text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False)

    proc_kwargs = dict(text=[text], return_tensors='pt')
    if has_image:
        proc_kwargs['images'] = images
    if has_video:
        proc_kwargs['videos'] = videos
    if has_audio:
        proc_kwargs['audio'] = audio
    inputs = processor(**proc_kwargs)
    inputs = {k: (v.to(device) if torch.is_tensor(v) else v)
              for k, v in inputs.items()}
    for _meta_key in ('num_patches', 'num_tokens', 'imgs_sizes'):
        inputs.pop(_meta_key, None)
    input_len = int(inputs['input_ids'].shape[1])
    return inputs, input_len, (has_video, has_audio, has_image)


def get_text_inputs(processor, tokenizer, item, args, system_prompt, device):
    """Processor inputs for the text-only (media dropped) branch."""
    return build_processor_inputs(processor, tokenizer, item, args,
                                  system_prompt, device, drop_media=True)


def load_item_media(item, args):
    """Load the item's media objects. Returns (videos, images, audio)."""
    from shared import resolve_media_path
    json_dir = os.path.dirname(args.data_path)
    videos = images = audio = None
    if item.get('video_path'):
        vp = resolve_media_path(item['video_path'], json_dir, args.data_root)
        videos = load_video_frames_native(vp)
    if item.get('audio_path'):
        ap = resolve_media_path(item['audio_path'], json_dir, args.data_root)
        audio = load_audio_wave(ap)
    if item.get('image_path'):
        ip = resolve_media_path(item['image_path'], json_dir, args.data_root)
        images = load_image(ip)
    return videos, images, audio


def inputs_from_text(processor, tokenizer, system_prompt, user_text, media,
                     device, enable_thinking=False):
    """Processor inputs for an ARBITRARY user text (ToT/GD/PD step/vote/judge
    prompts) with the item's media attached. media = (videos, images, audio)
    as returned by load_item_media. Returns (inputs, input_len)."""
    videos, images, audio = media
    has_video = videos is not None
    has_audio = audio is not None
    has_image = images is not None
    prefix = ''
    if has_video:
        prefix += '<video>\n'
    if has_audio:
        prefix += '<so_embedding>\n'
    if has_image:
        prefix += '<image>\n'
    user_full = (prefix + user_text) if prefix else user_text
    messages = [{'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_full}]
    try:
        text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
            enable_thinking=enable_thinking)
    except TypeError:
        text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False)
    kw = dict(text=[text], return_tensors='pt')
    if has_image:
        kw['images'] = images
    if has_video:
        kw['videos'] = videos
    if has_audio:
        kw['audio'] = audio
    inputs = processor(**kw)
    inputs = {k: (v.to(device) if torch.is_tensor(v) else v)
              for k, v in inputs.items()}
    for _mk in ('num_patches', 'num_tokens', 'imgs_sizes'):
        inputs.pop(_mk, None)
    return inputs, int(inputs['input_ids'].shape[1])


# ---- LLM-level prefill / step decode (media fused via inputs_embeds) ----

def llm_prefill(model, inputs):
    """Prefill one branch on model.language_model with fused media embeddings.
    Returns (logits_last[B,V], cache, attention_mask)."""
    llm = model.language_model
    embeds, attn_mask, input_ids = fuse_inputs_embeds(model, inputs)
    with torch.inference_mode():
        out = llm(inputs_embeds=embeds, attention_mask=attn_mask,
                  use_cache=True)
    return out.logits[:, -1, :], out.past_key_values, attn_mask


def llm_prefill_embeds(model, inputs, embeds=None, attn_mask=None):
    """llm_prefill over PRE-FUSED inputs_embeds. AVCD prefills the SAME item
    up to 4 times (original + masked branches); fusing media once instead of
    per-branch cut the C-RADIO/Parakeet encodes that OOM'd 4-branch AVCD on a
    single 80 GB card (2026-08-30). Returns (logits_last, cache, attn_mask)."""
    llm = model.language_model
    if embeds is None:
        embeds, attn_mask, _ = fuse_inputs_embeds(model, inputs)
    with torch.inference_mode():
        out = llm(inputs_embeds=embeds, attention_mask=attn_mask,
                  use_cache=True)
    return out.logits[:, -1, :], out.past_key_values, attn_mask


def llm_step(model, token_ids, attn_mask, cache):
    """One decode step on model.language_model. Grows the attention mask.
    Returns (logits_last, cache, attn_mask)."""
    llm = model.language_model
    device = next(llm.parameters()).device
    attn_mask = torch.cat(
        [attn_mask, torch.ones(attn_mask.shape[0], 1, device=device,
                               dtype=attn_mask.dtype)], dim=-1)
    with torch.inference_mode():
        out = llm(input_ids=token_ids, attention_mask=attn_mask,
                  past_key_values=cache, use_cache=True)
    return out.logits[:, -1, :], out.past_key_values, attn_mask


def contrastive_generate(model, tokenizer, branch_inputs, combine_fn,
                         max_new_tokens=256, stop_repeat=5, eos_id=None):
    """Step-by-step greedy decoding over N prefill branches.
      branch_inputs : list of processor-input dicts (branch 0 = original)
      combine_fn    : list_of_last_logits -> selection logits
    Returns (response_str, generated_ids). Branches with different prompt
    lengths each keep their own cache/mask (as in base_cad_qwen3.py)."""
    device = next(model.parameters()).device
    if eos_id is None:
        eos_id = tokenizer.eos_token_id
        if eos_id is None:
            eos_id = NEMOTRON_EOS_ID

    logits_list, caches, masks = [], [], []
    for bi, inp in enumerate(branch_inputs):
        lg, cache, mask = llm_prefill(model, inp)
        logits_list.append(lg)
        caches.append(cache)
        masks.append(mask)

    generated = []
    with torch.inference_mode():
        comb = combine_fn(logits_list)
        cur = comb.argmax(dim=-1, keepdim=True)
        tok = cur.item()
        generated.append(tok)
        for _ in range(1, max_new_tokens):
            if tok == eos_id:
                break
            if (len(generated) >= stop_repeat
                    and len(set(generated[-stop_repeat:])) == 1):
                break
            for i in range(len(caches)):
                logits_list[i], caches[i], masks[i] = llm_step(
                    model, cur, masks[i], caches[i])
            comb = combine_fn(logits_list)
            cur = comb.argmax(dim=-1, keepdim=True)
            tok = cur.item()
            generated.append(tok)
    del caches
    return (tokenizer.decode(generated, skip_special_tokens=True).strip(),
            generated)


# ---- outer generate() helper (media handled by the omni wrapper itself) ----

def outer_generate(model, tokenizer, inputs, max_new_tokens=256,
                   do_sample=True, temperature=0.7, top_p=0.9,
                   want_logp=False):
    """Generate via the omni wrapper (same path as base_sc_nemotron_hf.py).
    Returns (text, gen_ids, avg_logp or None); avg_logp is the mean per-token
    logp of the produced tokens under the sampling distribution (phi-Decoding
    path-score flavour, used by gd/pd; computed only when want_logp)."""
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=tokenizer.pad_token_id or 0,
    )
    if do_sample:
        gen_kwargs.update(temperature=temperature, top_p=top_p)
    if want_logp:
        gen_kwargs.update(output_scores=True, return_dict_in_generate=True)
    with torch.inference_mode():
        out = model.generate(**inputs, **gen_kwargs)
    input_len = int(inputs['input_ids'].shape[1])
    seq = out.sequences[0] if hasattr(out, 'sequences') else out[0]
    gen_ids = seq[input_len:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    avg_lp = None
    if want_logp and hasattr(out, 'scores') and out.scores:
        lps = [torch.log_softmax(sc[0].float(), dim=-1)[int(tid)].item()
               for tid, sc in zip(gen_ids.tolist(), out.scores)]
        avg_lp = float(sum(lps) / (len(lps) + 1e-8))
    return text, list(gen_ids.tolist()), avg_lp


# ---- hybrid-cache surgery (OPERA) ----

def cache_iter_attn(cache):
    """Yield (layer_idx, key, value) for ATTENTION layers only.

    NemotronHHybridDynamicCache pre-fills key_cache/value_cache for ALL layers;
    mamba slots hold the constructor's [batch, 0] 2-D dummies, whose shape[-2]
    is the BATCH size. Yielding those made cache_len() return num_beams, so
    OPERA's per-probe cache_crop(past_kv, base_len) cropped the attention kv
    to num_beams tokens -> next probe's captured row was ~6 wide -> row[:, vis_idx]
    indexed out of bounds -> CUDA device-side assert on every multimodal
    (visual-token) item (text-only items have an empty vis_idx and slipped
    through). 2026-08-30 empty-run diagnosis; iterate 4-D (attention) slots only.
    """
    for i, k in enumerate(cache.key_cache):
        if k.dim() == 4:
            yield i, k, cache.value_cache[i]


def cache_len(cache):
    for _, k, _ in cache_iter_attn(cache):
        return int(k.shape[-2])
    return 0


def cache_snapshot_mamba(cache):
    """Clone conv/ssm states (they update in place on EVERY forward)."""
    return ({i: t.clone() for i, t in cache.conv_states.items()},
            {i: t.clone() for i, t in cache.ssm_states.items()})


def cache_restore_mamba(cache, snap):
    conv, ssm = snap
    for i, t in conv.items():
        cache.conv_states[i] = t
    for i, t in ssm.items():
        cache.ssm_states[i] = t


def cache_expand(cache, n):
    for i, k, v in list(cache_iter_attn(cache)):
        cache.key_cache[i] = k.repeat_interleave(n, dim=0)
        cache.value_cache[i] = v.repeat_interleave(n, dim=0)
    for i, t in list(cache.conv_states.items()):
        if t.dim() == 3:  # non-empty mamba conv state
            cache.conv_states[i] = t.repeat_interleave(n, dim=0)
    for i, t in list(cache.ssm_states.items()):
        if t.dim() >= 3:  # non-empty mamba ssm state
            cache.ssm_states[i] = t.repeat_interleave(n, dim=0)
    return cache


def cache_select(cache, idx):
    for i, k, v in list(cache_iter_attn(cache)):
        cache.key_cache[i] = k.index_select(0, idx)
        cache.value_cache[i] = v.index_select(0, idx)
    for i, t in list(cache.conv_states.items()):
        if t.dim() == 3:
            cache.conv_states[i] = t.index_select(0, idx)
        elif t.dim() == 2 and t.shape[1] != 0:
            cache.conv_states[i] = t.index_select(0, idx)
    for i, t in list(cache.ssm_states.items()):
        if t.dim() == 3:
            cache.ssm_states[i] = t.index_select(0, idx)
    return cache


def cache_crop(cache, max_len):
    """Revert probe steps: crop attention kv. Mamba states MUST be restored
    separately from a cache_snapshot_mamba() snapshot."""
    for i, k, v in list(cache_iter_attn(cache)):
        if k.shape[-2] > max_len:
            cache.key_cache[i] = k[:, :, :max_len, :].contiguous()
            cache.value_cache[i] = v[:, :, :max_len, :].contiguous()
    return cache


def cache_clone(model, cache):
    """Deep copy of the whole hybrid cache."""
    import sys
    llm = model.language_model
    mod = sys.modules[type(llm).__module__]
    new = mod.NemotronHHybridDynamicCache(
        llm.config, cache.key_cache[0].shape[0] if cache.key_cache else 1,
        dtype=llm.dtype, device=llm.device)
    new.has_previous_state = cache.has_previous_state
    new.key_cache = [k.clone() for k in cache.key_cache]
    new.value_cache = [v.clone() for v in cache.value_cache]
    new.conv_states = {i: t.clone() for i, t in cache.conv_states.items()}
    new.ssm_states = {i: t.clone() for i, t in cache.ssm_states.items()}
    return new


# ---- layer structure helpers ----

def block_types(model):
    """List of 'mamba'/'attention'/'moe' per block, from the model config."""
    llm = model.language_model
    return list(llm.config.layers_block_type)


def attention_layer_indices(model):
    return [i for i, t in enumerate(block_types(model)) if t == 'attention']


def last_attention_block(model):
    idx = attention_layer_indices(model)[-1]
    return model.language_model.backbone.layers[idx]
