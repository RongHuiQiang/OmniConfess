# -*- coding: utf-8 -*-
"""
qwen3_shim.py — drop-in compatibility layer so the Qwen2.5-Omni baseline
runners (base_cad/sc/dola/vcd/tot/gd/pd.py) run on Qwen3-Omni-30B-A3B-Instruct
with only their `from shared import ...` line redirected here.

WHY A SHIM (adaptation-surface A, see QWEN3_BASELINE_PLAN.md §0):
  The Qwen2.5 baselines are hard-coupled to the OUTER wrapper
  Qwen2_5OmniForConditionalGeneration, whose text backbone lives under
  `model.thinker.*`:
      model.thinker.generate(...)          model.thinker(**clean)
      model.thinker.model.layers[i]        model.thinker.model.norm
      model.thinker.lm_head                model.thinker.compute_transition_scores
  Qwen3's shared/qwen3_omni_utils.load_model() instead returns the THINKER
  ITSELF (Qwen3OmniMoeThinkerForConditionalGeneration), with the same subtree
  one hop shallower:
      model.generate/model(...)  model.model.layers/norm  model.lm_head

  Rather than hand-edit every call site (error-prone across 7 files), we load
  the Qwen3 thinker and attach a SELF-ALIAS `model.thinker = model`. Then every
  `model.thinker.X` the baselines use resolves to `model.X` automatically:
      model.thinker.model  -> model.model          ✓
      model.thinker.lm_head-> model.lm_head         ✓
      model.thinker(...)   -> model(...)            ✓  (thinker __call__)
      model.thinker.generate -> model.generate      ✓
  This is a pure attribute alias; it changes no behaviour and touches no repo
  baseline source. Qwen2.5 arm is untouched.

EOS FIX (Qwen3 known pit, RESULTS.md line 65): Qwen3's generation_config.json
  ships without eos_token_id. HOWEVER the *tokenizer* already reports
  eos_token_id=151645 (<|im_end|>) and pad_token_id=151643 (<|endoftext|>)
  (probed 2026-08-17). The manual decode loops in base_cad/base_dola break on
  `token_id == tokenizer.eos_token_id`, which is therefore already correct.
  For runners that call .generate() we still pass eos/pad explicitly via the
  helper below so both stop ids are honoured.

USAGE: a base_<m>_qwen3.py copy replaces
    from shared import (load_dataset, ..., load_model, build_inputs,
                        prepare_for_thinker, ...)
with
    from shared import (load_dataset, load_media_for_item,
        build_user_content_with_media_tokens, make_messages,
        get_system_prompt, get_user_prompt, parse_pred, make_result, save_result)
    from shared.qwen3_shim import load_model, build_inputs, prepare_for_thinker
"""
import os
import torch

from .qwen3_omni_utils import (
    load_model as _q3_load_model,
    build_inputs as _q3_build_inputs,
    prepare_for_thinker as _q3_prepare_for_thinker,
    build_user_content_with_media_tokens,  # re-export: Qwen3 media-token spelling
    make_messages,                          # re-export (identical, but keep single source)
)


def load_model(model_path=None):
    """Load the Qwen3-Omni thinker and attach a self-alias `.thinker` so the
    Qwen2.5 baselines' `model.thinker.*` access paths resolve to `model.*`.
    Also normalises tokenizer eos/pad so manual decode-loop `== eos_id` checks
    stop at <|im_end|> (Qwen3 eos fix)."""
    if model_path is None:
        processor, model = _q3_load_model()
    else:
        processor, model = _q3_load_model(model_path)

    # self-alias: model.thinker.<attr> == model.<attr>
    try:
        model.thinker = model
    except Exception:
        # nn.Module may block plain attribute set if 'thinker' collides; use
        # object.__setattr__ to bypass the Module __setattr__ machinery.
        object.__setattr__(model, "thinker", model)

    tk = processor.tokenizer
    im_end = tk.convert_tokens_to_ids("<|im_end|>")
    endoftext = tk.convert_tokens_to_ids("<|endoftext|>")
    # tokenizer already reports eos=<|im_end|>; assert & pin defensively so the
    # single-value `token_id == eos_id` breaks in cad/dola are correct.
    if im_end is not None:
        tk.eos_token_id = im_end
    if endoftext is not None:
        tk.pad_token_id = endoftext

    # EOS FIX for .generate() callers (SC/ToT/GD/PD): Qwen3's generation_config
    # ships WITHOUT eos_token_id, so model.generate() never early-stops and runs
    # to max_new_tokens (RESULTS.md line 65). The Qwen2.5 baselines don't pass
    # eos_token_id to generate() explicitly -- they rely on generation_config.
    # Pin both stop ids onto the model's generation_config so every unmodified
    # .generate() call inherits them. (cad/dola manual loops use tokenizer.eos_id
    # above; this covers the generate()-based methods without editing call sites.)
    stops = [i for i in {im_end, endoftext} if i is not None]
    try:
        if getattr(model, "generation_config", None) is not None and stops:
            model.generation_config.eos_token_id = stops
            if endoftext is not None:
                model.generation_config.pad_token_id = endoftext
    except Exception:
        pass

    return processor, model


def build_inputs(processor, model, messages, pixel_values, audio_values,
                 image_values, dataset_name=""):
    """Delegate to the Qwen3 build_inputs (handles pixel_values key,
    video_second_per_grid, scalar fps, 24k->16k audio resample)."""
    return _q3_build_inputs(processor, model, messages, pixel_values,
                            audio_values, image_values, dataset_name)


def prepare_for_thinker(inputs, model=None):
    """Qwen3 prepare_for_thinker takes an optional model arg; the Qwen2.5
    baselines call it as prepare_for_thinker(inputs, model) — swallow the
    extra positional arg for signature compatibility."""
    return _q3_prepare_for_thinker(inputs, model)


def qwen3_stop_ids(tokenizer):
    """Both Qwen3 stop ids for explicit .generate() eos_token_id lists."""
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    endoftext = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    return [i for i in {im_end, endoftext} if i is not None]
