# -*- coding: utf-8 -*-
"""
Model initialization and inference utilities for Qwen2.5-Omni-7B.
Verified patterns from GUARD/base_hook_best_0314.py lines 471-499, 947-963.
"""
import os
import torch
import numpy as np
from transformers import (
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniProcessor,
)

MODEL_PATH = "Qwen/Qwen2.5-Omni-7B"

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def load_model(model_path=MODEL_PATH):
    """
    Load Qwen2.5-Omni processor + model.
    Disables talker (speech head) — text-only output.
    Verified from GUARD lines 476-491.
    """
    processor = Qwen2_5OmniProcessor.from_pretrained(
        model_path, trust_remote_code=True
    )
    tokenizer = processor.tokenizer
    tokenizer.padding_side = 'left'
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    ).eval()

    # Fix: vocab_size in thinker config is sometimes wrong in the checkpoint
    model.thinker.config.vocab_size = 152064

    return processor, model


def build_inputs(processor, model, messages, pixel_values, audio_values, image_values, dataset_name=""):
    """
    Build tokenized model inputs from messages + media tensors.

    Media token injection pattern (verified from methods_haloquest/base_model_low.py lines 242-250):
      - image  -> <|vision_bos|><|IMAGE|><|vision_eos|>
      - video  -> <|vision_bos|><|VIDEO|><|vision_eos|>
      - audio  -> <|audio_bos|><|AUDIO|><|audio_eos|>
    These are prepended to the user content string BEFORE apply_chat_template.

    processor_kwargs pattern (verified from GUARD lines 952-963):
      fps=[1.0] when video present; max_pixels=224*224 for haloquest; None fields omitted.
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
        processor_kwargs["fps"] = [1.0]
    if audio_values is not None:
        processor_kwargs["audio"] = [audio_values]

    # haloquest uses constrained resolution (GUARD line 960-961)
    if "halo" in dataset_name.lower():
        processor_kwargs["max_pixels"] = 224 * 224

    inputs = processor(**processor_kwargs).to(model.device)

    # Cast float tensors to bfloat16 for memory efficiency
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
            inputs[k] = v.to(torch.bfloat16)

    return inputs


def safe_generate(model, inputs, max_new_tokens=256, do_sample=False,
                  temperature=1.0, top_p=1.0, num_return_sequences=1):
    """
    Wrapper around model.thinker.generate() — text-only branch.
    Returns list of decoded strings (one per return sequence).
    """
    input_ids = inputs["input_ids"]
    input_len = input_ids.shape[1]

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        num_return_sequences=num_return_sequences,
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    with torch.inference_mode():
        output_ids = model.thinker.generate(**inputs, **gen_kwargs)

    # Decode only newly generated tokens
    tokenizer = model.thinker.get_input_embeddings().__class__  # dummy — use processor
    return output_ids, input_len


_PROCESSOR_ONLY_KEYS = {'return_tensors', 'padding', 'truncation', 'max_length'}

_THINKER_VALID_KEYS = {
    'input_ids', 'attention_mask',
    'pixel_values_videos', 'video_grid_thw', 'video_second_per_grid',
    'pixel_values_images', 'pixel_values', 'image_grid_thw',
    'feature_attention_mask', 'input_features',
}


def prepare_for_thinker(inputs, model):
    """
    Remove processor-only keys, filter to keys accepted by model.thinker.generate(),
    fix audio feature_attention_mask shape mismatch, and cast float tensors to bfloat16.
    Verified approach from methods_text/base_sc.py and methods_text/base_pd.py.
    Audio mask fix from methods_text/base_pd.py safe_thinker_generate().
    """
    clean = {k: v for k, v in inputs.items()
             if k in _THINKER_VALID_KEYS}

    # Fix: feature_attention_mask temporal dim must match input_features.shape[2]
    # (Qwen2.5-Omni audio encoder: [Batch, Hidden, Time] -> mask must be len=Time)
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


def decode_outputs(processor, output_ids, input_len):
    """
    Decode newly generated tokens (strip prompt prefix).
    Returns list of strings.
    """
    new_tokens = output_ids[:, input_len:]
    texts = processor.batch_decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return texts


def build_user_content_with_media_tokens(pixel_values, audio_values, image_values, prompt_text):
    """
    Prepend Qwen-Omni media placeholder tokens to user content string.
    Token pattern verified from multiple baseline files.
    """
    prefix = ""
    if image_values is not None:
        prefix += "<|vision_bos|><|IMAGE|><|vision_eos|>\n"
    if pixel_values is not None:
        prefix += "<|vision_bos|><|VIDEO|><|vision_eos|>\n"
    if audio_values is not None:
        prefix += "<|audio_bos|><|AUDIO|><|audio_eos|>\n"
    return prefix + prompt_text


def make_messages(system_prompt, user_content_str):
    """Standard 2-turn chat messages list."""
    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user",   "content": [{"type": "text", "text": user_content_str}]},
    ]
