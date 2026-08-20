# -*- coding: utf-8 -*-
"""
Model initialization and inference utilities for Qwen3-Omni-30B-A3B-Instruct.

Third true-omnimodal backbone (image+video+audio+text), alongside Qwen2.5-Omni-7B.

transformers 4.57.6 supports Qwen3-Omni natively under model_type
``qwen3_omni_moe`` -- the class names are Qwen3OmniMoe* (NOT Qwen3Omni*):
    Qwen3OmniMoeForConditionalGeneration          (full: thinker + talker)
    Qwen3OmniMoeThinkerForConditionalGeneration   (text-only thinker, ~5GB lighter)
    Qwen3OmniMoeProcessor

Protocol mirrors shared/model_utils.py (Qwen2.5-Omni) so the runner stays a
near-copy of methods/base_model.py. Deltas vs Qwen2.5-Omni:

  1. Media placeholder tokens are the Qwen2.5-VL family spellings, not the
     Qwen2.5-Omni ones:
        image  <|vision_start|><|image_pad|><|vision_end|>
        video  <|vision_start|><|video_pad|><|vision_end|>
        audio  <|audio_start|><|audio_pad|><|audio_end|>
  2. Thinker image tensor key is ``pixel_values`` (Qwen2.5-Omni used
     ``pixel_values_images``).
  3. Videos carry an extra ``video_second_per_grid`` input.
  4. fps must still be a SCALAR (transformers >=4.53 requirement, same lesson
     as the Qwen2.5 pipeline); the processor computes
     video_second_per_grid = temporal_patch_size / fps.

We load the thinker-only class: text output is all the protocol needs, and the
talker/code2wav heads (~4GB bf16) are dead weight on a card that must hold
~57GB of thinker weights anyway. Single process per card.
"""
from .paths import resolve_model
import os
import numpy as np
import torch
from transformers import (
    Qwen3OmniMoeThinkerForConditionalGeneration,
    Qwen3OmniMoeProcessor,
)

MODEL_PATH = resolve_model("Qwen/Qwen3-Omni-30B-A3B-Instruct")

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def load_model(model_path=MODEL_PATH):
    """
    Load Qwen3-Omni thinker + processor (text-only generation).
    Returns (processor, model); model is the thinker itself, so the runner
    calls model.generate(...) -- no .thinker hop.
    """
    processor = Qwen3OmniMoeProcessor.from_pretrained(
        model_path, trust_remote_code=True
    )
    tokenizer = processor.tokenizer
    tokenizer.padding_side = 'left'
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    # Device placement. Thinker weights are ~60.5 GiB bf16, which does NOT fit
    # a single 80GB card that carries co-tenant allocations (~22 GiB foreign
    # blocks sit on every card; verified -- no reclaimable process in this
    # container). Two modes:
    #   Q3O_DEVICE_MAP=auto  -> accelerate splits across ALL visible GPUs
    #                           (launcher exposes 2 cards per shard process)
    #   default "cuda:0"     -> single card (needs a ~62 GiB-free card)
    device_map = os.environ.get("Q3O_DEVICE_MAP", "cuda:0")
    model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True,
    ).eval()

    return processor, model


AUDIO_INPUT_SR = 24000   # what shared/data_loader.load_audio_data returns
AUDIO_MODEL_SR = 16000   # Qwen3-Omni Whisper feature extractor rate
                       # (preprocessor_config.json / Qwen3OmniMoeProcessorKwargs._defaults)


def _resample_audio(audio_values, target_sr):
    """24 kHz (data_loader) -> 16 kHz (model FE) resample, no-op if equal."""
    if target_sr == AUDIO_INPUT_SR:
        return audio_values
    import librosa
    return librosa.resample(
        np.asarray(audio_values, dtype=np.float32),
        orig_sr=AUDIO_INPUT_SR, target_sr=target_sr)


def build_inputs(processor, model, messages, pixel_values, audio_values, image_values, dataset_name=""):
    """
    Build tokenized model inputs from messages + media (same contract as
    shared/model_utils.build_inputs). Media placeholders are pre-inlined into
    the user text (see build_user_content_with_media_tokens), so this only
    needs to hand the raw media arrays to the processor for expansion.

    NOTE on audio rate: shared/data_loader loads at 24 kHz; the Qwen3-Omni
    Whisper feature extractor declares 16 kHz, so we resample here (librosa).
    The Qwen2.5 arm has historically fed 24 kHz directly into its own 16 kHz
    FE (time-warped mel); this arm feeds the model its native rate instead --
    each backbone is driven correctly rather than identically distorted.
    """
    raw_p = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    p_str = raw_p[0] if isinstance(raw_p, list) else raw_p

    processor_kwargs = {
        "text": [p_str],
        "return_tensors": "pt",
    }

    if image_values is not None:
        processor_kwargs["images"] = [image_values]
    if pixel_values is not None:
        processor_kwargs["videos"] = [pixel_values]
        # fps MUST be a scalar (list raises TypeError, transformers >=4.53).
        processor_kwargs["fps"] = 1.0
    if audio_values is not None:
        target_sr = getattr(processor.feature_extractor, "sampling_rate", AUDIO_MODEL_SR)
        processor_kwargs["audio"] = [_resample_audio(audio_values, target_sr)]

    # haloquest uses constrained resolution (same knob as the Qwen2.5 arm)
    if "halo" in dataset_name.lower():
        processor_kwargs["max_pixels"] = 224 * 224

    # model.device is the first device of the accelerate dispatch map when
    # pipeline-split across cards; hooks move activations between stages.
    first_device = getattr(model, "device", None)
    if first_device is None:
        first_device = next(model.parameters()).device
    inputs = processor(**processor_kwargs).to(first_device)

    for k, v in inputs.items():
        if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
            inputs[k] = v.to(torch.bfloat16)

    return inputs


_THINKER_VALID_KEYS = {
    'input_ids', 'attention_mask',
    'pixel_values_videos', 'video_grid_thw', 'video_second_per_grid',
    'pixel_values', 'image_grid_thw',
    'feature_attention_mask', 'input_features',
}


def prepare_for_thinker(inputs, model=None):
    """
    Filter to keys accepted by the thinker forward (drops processor-only keys)
    and cast float tensors to bfloat16. Keeps the defensive audio-mask length
    fix from the Qwen2.5 arm: feature_attention_mask temporal dim must equal
    input_features.shape[2] ([Batch, Hidden, Time] -> mask len = Time).
    """
    clean = {k: v for k, v in inputs.items()
             if k in _THINKER_VALID_KEYS}

    if 'input_features' in clean and 'feature_attention_mask' in clean:
        feat = clean['input_features']
        mask = clean['feature_attention_mask']
        if feat.dim() == 3:
            target_len = feat.shape[2]
            if mask.shape[1] != target_len:
                if mask.shape[1] < target_len:
                    pad = torch.ones(
                        (mask.shape[0], target_len - mask.shape[1]),
                        device=mask.device, dtype=mask.dtype)
                    clean['feature_attention_mask'] = torch.cat([mask, pad], dim=1)
                else:
                    clean['feature_attention_mask'] = mask[:, :target_len]

    for k, v in clean.items():
        if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
            clean[k] = v.to(torch.bfloat16)
    return clean


def build_user_content_with_media_tokens(pixel_values, audio_values, image_values, prompt_text):
    """
    Prepend Qwen3-Omni media placeholder tokens to user content string.
    NOTE: token spellings differ from Qwen2.5-Omni (see module docstring).
    """
    prefix = ""
    if image_values is not None:
        prefix += "<|vision_start|><|image_pad|><|vision_end|>\n"
    if pixel_values is not None:
        prefix += "<|vision_start|><|video_pad|><|vision_end|>\n"
    if audio_values is not None:
        prefix += "<|audio_start|><|audio_pad|><|audio_end|>\n"
    return prefix + prompt_text


def make_messages(system_prompt, user_content_str):
    """Standard 2-turn chat messages list."""
    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user",   "content": [{"type": "text", "text": user_content_str}]},
    ]
