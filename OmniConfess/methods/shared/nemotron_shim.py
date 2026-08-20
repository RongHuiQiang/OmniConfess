# -*- coding: utf-8 -*-
"""
nemotron_shim.py -- shared loading + generation plumbing for the Nemotron
baseline method runners (base_sc / base_cad / base_dola).

WHY A SHIM (mirror of shared/qwen3_shim.py, but a DIFFERENT problem):
  The Qwen shim exists to re-expose a standard-HF layout under `model.thinker.*`
  via a self-alias, so the Qwen2.5 baselines run unedited. Nemotron does NOT
  share the Qwen baselines' call sites at all -- its LLM lives under
      model.language_model.backbone           (Mamba-style block list)
      model.language_model.backbone.norm_f    (final RMSNorm)
      model.language_model.lm_head            (tied vocab projection)
  and the omni wrapper is loaded with AutoModel+trust_remote_code and needs the
  _repair_buffers() fix on every load (summary_idxs + 2 non-persistent vision
  conditioner buffers) or image/video items device-assert.

  Rather than copy the ~60-line load+repair block into three runners, it lives
  here once. The three Nemotron runners import load_nemotron() + the media/
  prompt helpers and add only their method-specific decode loop.

LLM STRUCTURE (probed via the omni wrapper on this checkpoint):
  model.language_model                      -> NemotronHForCausalLM
  model.language_model.backbone             -> NemotronHModel (52 hybrid blocks)
  model.language_model.backbone.layers[i]   -> Mamba (M) or Attn+MLP (E) block
  model.language_model.backbone.norm_f      -> final RMSNorm
  model.language_model.lm_head              -> Linear(hidden, vocab)

  A single forward `model.language_model(input_ids=..., use_cache=True,
  output_hidden_states=True)` returns a CausalLMOutput-like object with
  .logits, .past_key_values, .hidden_states -- the standard CausalLM interface,
  so CAD (two forwards, logit contrast) and greedy step-decoding work exactly
  as on a plain HF CausalLM. DoLa's early-exit assumption is architecture-
  sensitive here (Mamba blocks are not clean transformer layers); see
  base_dola_nemotron_hf.py for the caveat.

  IMPORTANT: the OUTER omni wrapper's forward expects multimodal kwargs and
  fuses vision/audio embeddings; for the LLM-level contrastive loops we go
  through the outer `model(**inputs)` for the *prefill* (so media tokens are
  fused) and then step-decode on `model.language_model` with past_key_values
  (text-only continuation -- no media on subsequent steps).
"""
import os
import json as _json
import torch

REPO = os.environ.get('NEMO_REPO', '/home/tsq/rhq/omniconfess/OmniConfess')
DEFAULT_MODEL = '/ultrafast_share/tsq/MODELS/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16'


def _repair_buffers(model, model_path, device):
    """Fix transformers 5.x remote-code load artifacts on this checkpoint.
    (Verbatim behaviour from base_model_nemotron_hf_pv.py -- summary_idxs +
    non-persistent vision-conditioner float buffers.)"""
    from safetensors.torch import load_file as _load_file

    cfg = _json.load(open(os.path.join(model_path, 'config.json')))
    teachers = cfg['vision_config']['args']['teachers']
    idxs = [i for i, t in enumerate(teachers) if t.get('use_summary', True)]
    radio = model.vision_model.radio_model
    fixed = torch.tensor(idxs, dtype=torch.long, device=device)
    bad = (radio.summary_idxs is None or
           radio.summary_idxs.device != fixed.device or
           not torch.equal(radio.summary_idxs.long(), fixed))
    if bad:
        print(f'[repair] summary_idxs garbage -> {idxs}', flush=True)
        radio.summary_idxs = fixed

    idx = _json.load(open(os.path.join(model_path, 'model.safetensors.index.json')))['weight_map']
    want = {
        'vision_model.radio_model.input_conditioner.norm_mean':
            ('vision_model', 'radio_model', 'input_conditioner', 'norm_mean'),
        'vision_model.radio_model.input_conditioner.norm_std':
            ('vision_model', 'radio_model', 'input_conditioner', 'norm_std'),
        'sound_encoder.encoder.feature_extractor.featurizer.fb':
            ('sound_encoder', 'encoder', 'feature_extractor', 'featurizer', 'fb'),
        'sound_encoder.encoder.feature_extractor.featurizer.window':
            ('sound_encoder', 'encoder', 'feature_extractor', 'featurizer', 'window'),
    }
    fixes = 0
    for key, attr_path in want.items():
        shard = idx.get(key)
        if shard is None:
            continue
        try:
            t = _load_file(os.path.join(model_path, shard))[key]
            node = model
            for a in attr_path[:-1]:
                node = getattr(node, a)
            cur = getattr(node, attr_path[-1], None)
            if cur is None or cur.shape != t.shape or not torch.allclose(cur.float().cpu(), t.float(), atol=1e-3):
                setattr(node, attr_path[-1], t.to(device=device, dtype=cur.dtype if cur is not None else t.dtype))
                fixes += 1
        except Exception as e:
            print(f'[repair] skip {key}: {type(e).__name__}: {e}', flush=True)
    print(f'[repair] done ({fixes} constant buffers restored, summary_idxs ok)', flush=True)


def load_nemotron(model_path=None, device=None):
    """Load the Nemotron omni wrapper (AutoModel+trust_remote_code, bf16, eager),
    apply _repair_buffers, disable the EVS square-grid video pruning, and return
    (processor, model, tokenizer, device).

    device defaults to $NEMO_DEVICE or 'cuda:0' (claim_run sets
    CUDA_VISIBLE_DEVICES so cuda:0 is the claimed physical card)."""
    from transformers import AutoModel, AutoProcessor
    model_path = model_path or os.environ.get('NEMOTRON_MODEL_PATH', DEFAULT_MODEL)
    device = device or os.environ.get('NEMO_DEVICE', 'cuda:0')

    import time
    t0 = time.time()
    print(f'[model] loading {model_path} (bf16, eager)...', flush=True)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, attn_implementation='eager')
    model.to(device).eval()
    _repair_buffers(model, model_path, device)
    if getattr(model, 'video_pruning_rate', 0):
        print(f'[repair] video_pruning_rate {model.video_pruning_rate} -> 0.0 '
              f'(EVS square-grid bug, keep all tokens)', flush=True)
        model.video_pruning_rate = 0.0
    print(f'[model] loaded in {time.time()-t0:.0f}s', flush=True)
    return processor, model, processor.tokenizer, device


# ---- LLM sub-module accessors (single source of truth for the 3 runners) ----

def get_llm(model):
    """The inner NemotronHForCausalLM (has .backbone, .lm_head)."""
    return model.language_model


def get_backbone(model):
    """The block list container NemotronHModel (has .layers, .norm_f)."""
    return model.language_model.backbone


def get_final_norm(model):
    """Final RMSNorm before lm_head. Name probed across candidates."""
    bb = get_backbone(model)
    for cand in ('norm_f', 'norm', 'final_layernorm', 'ln_f'):
        if hasattr(bb, cand):
            return getattr(bb, cand)
    raise AttributeError('no final norm found on backbone; children=' +
                         str([n for n, _ in bb.named_children()]))


def get_lm_head(model):
    return model.language_model.lm_head


def get_layers(model):
    bb = get_backbone(model)
    for cand in ('layers', 'blocks', 'h'):
        if hasattr(bb, cand):
            return getattr(bb, cand)
    raise AttributeError('no layer list on backbone')


# ---- Multimodal fusion (replicates the omni wrapper's generate() fusion) ----
#
# The outer wrapper only exposes fusion inline inside forward()/generate(); to
# run our OWN contrastive step-decode loop on model.language_model we need the
# fused inputs_embeds up front. This mirrors modeling.py generate() lines
# ~421-568 EXACTLY (image/video/sound scatter), minus the EVS video pruning
# branch (video_pruning_rate is forced to 0.0 at load, so it never triggers and
# the reshape-to-(B,N,C) path is taken). Kept in one place so cad/dola agree.

def fuse_inputs_embeds(model, inputs):
    """Build the fused text+media inputs_embeds for a processor `inputs` dict.

    Returns (inputs_embeds[B,N,C], attention_mask, input_ids). For text-only
    items this is just the token embedding table lookup. Media items scatter
    ViT / Parakeet embeddings into the placeholder token positions, byte-for-
    byte as the wrapper's generate() does.
    """
    import numpy as np
    input_ids = inputs['input_ids']
    attention_mask = inputs.get('attention_mask')
    device = next(model.parameters()).device

    pixel_values = inputs.get('pixel_values')
    pixel_values_videos = inputs.get('pixel_values_videos')
    sound_clips = inputs.get('sound_clips')

    has_images = pixel_values is not None
    has_videos = pixel_values_videos is not None
    has_sound = sound_clips is not None and getattr(model, 'sound_encoder', None) is not None

    if not (has_images or has_videos or has_sound):
        return model.language_model.get_input_embeddings()(input_ids), attention_mask, input_ids

    image_vit_embeds = video_vit_embeds = sound_embeds = None
    if has_images:
        pixel_values = pixel_values.to(dtype=model.vision_model.config.torch_dtype)
        image_vit_embeds = model.extract_feature(pixel_values)
    if has_videos:
        pixel_values_videos = pixel_values_videos.to(dtype=model.vision_model.config.torch_dtype)
        video_vit_embeds = model.extract_video_feature(pixel_values_videos)
    sound_attention_mask = None
    if has_sound:
        is_raw = False
        if isinstance(sound_clips, (list, tuple)):
            is_raw, waveforms = True, sound_clips
        elif isinstance(sound_clips, np.ndarray):
            is_raw = True
            waveforms = [sound_clips.squeeze()] if sound_clips.ndim > 1 else [sound_clips]
        elif isinstance(sound_clips, torch.Tensor):
            if sound_clips.dim() == 1:
                is_raw, waveforms = True, [sound_clips.cpu().numpy()]
            elif sound_clips.dim() == 2:
                is_raw, waveforms = True, [c.cpu().numpy() for c in sound_clips]
        if is_raw:
            audio_inputs = model.sound_feature_extractor(
                waveforms, sampling_rate=model.sound_feature_extractor.sampling_rate,
                return_tensors='pt')
            sound_input_features = audio_inputs.input_features
            sound_attention_mask = audio_inputs.get('attention_mask', None)
        else:
            sound_input_features = sound_clips
        tgt_dev = model.sound_encoder.encoder.subsampling.linear.weight.device
        tgt_dtype = model.language_model.config.torch_dtype
        sound_input_features = sound_input_features.to(dtype=tgt_dtype, device=tgt_dev)
        if sound_attention_mask is not None:
            sound_attention_mask = sound_attention_mask.to(device=tgt_dev)
        sound_embeds = model.extract_sound_feature(sound_input_features, sound_attention_mask)

    inputs_embeds = model.language_model.get_input_embeddings()(input_ids)
    B, N, C = inputs_embeds.shape
    inputs_embeds = inputs_embeds.reshape(B * N, C)
    input_ids_copy = input_ids.reshape(B * N)

    if image_vit_embeds is not None:
        m = (input_ids_copy == model.img_context_token_id)
        inputs_embeds[m] = image_vit_embeds.reshape(-1, C).to(inputs_embeds.device, inputs_embeds.dtype)
    if video_vit_embeds is not None:
        m = (input_ids_copy == model.img_context_token_id)
        inputs_embeds[m] = video_vit_embeds.reshape(-1, C).to(inputs_embeds.device, inputs_embeds.dtype)
    if sound_embeds is not None and model.sound_context_token_id is not None:
        m = (input_ids_copy == model.sound_context_token_id)
        if sound_embeds.dim() == 3 and sound_embeds.shape[0] > 1 and sound_attention_mask is not None:
            natural = sound_attention_mask.sum(-1) + 1
            outl = model.sound_encoder.encoder._get_subsampling_output_length(natural)
            flat = torch.cat([sound_embeds[i, :int(n)] for i, n in enumerate(outl.tolist())], dim=0)
        else:
            flat = sound_embeds.reshape(-1, C)
        inputs_embeds[m] = flat.to(inputs_embeds.device, inputs_embeds.dtype)

    inputs_embeds = inputs_embeds.reshape(B, N, C)
    return inputs_embeds, attention_mask, input_ids


def new_cache(model, batch_size=1):
    """Fresh NemotronHHybridDynamicCache for one decode branch (Mamba conv/ssm
    state + attention kv all live here; each contrastive branch needs its own).
    The cache class lives in the same remote module as NemotronHForCausalLM."""
    import sys
    llm = model.language_model
    mod = sys.modules[type(llm).__module__]
    return mod.NemotronHHybridDynamicCache(
        llm.config, batch_size, dtype=llm.dtype, device=llm.device)
