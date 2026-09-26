# -*- coding: utf-8 -*-
"""
nemotron_fuse.py — shared media-fusion helper for the manual-step decode
baselines (DoLa / CAD / VCD) on NVIDIA Nemotron-3-Nano-Omni-30B.

The Nemotron omni wrapper (NemotronH_Nano_Omni_Reasoning_V3) fuses image /
video / sound embeddings into `inputs_embeds` INSIDE its generate() (modeling.py
~L420-560) and only then calls self.language_model.generate(inputs_embeds=...).
Methods that need per-step contrastive logits (DoLa/CAD/VCD) cannot use the
wrapper's black-box generate — they must call model.language_model directly and
therefore must replicate that fusion.

`fuse_media()` is a faithful, line-checked transcription of that block for the
video_pruning_rate == 0 (EVS OFF) case — which is exactly how the proven base
runner loads the model (base_model_nemotron_hf.py sets video_pruning_rate = 0.0,
"keep all tokens" no-approximation setting). The EVS retention branch is
therefore intentionally omitted; an assert guards against it being enabled.

Returns (inputs_embeds [1,N,C], input_ids [1,N], attention_mask [1,N] or None).
All media tensors are the processor outputs (pixel_values / pixel_values_videos
/ sound_clips), i.e. the same kwargs the wrapper generate accepts.
"""
import numpy as np
import torch


def fuse_media(model, input_ids, attention_mask=None,
               pixel_values=None, pixel_values_videos=None, sound_clips=None):
    assert getattr(model, 'video_pruning_rate', 0) in (0, 0.0), \
        "fuse_media only supports EVS OFF (video_pruning_rate==0); base runner sets it to 0.0"
    assert model.img_context_token_id is not None

    has_images = pixel_values is not None
    has_videos = pixel_values_videos is not None
    has_sound = sound_clips is not None and getattr(model, 'sound_encoder', None) is not None

    if not (has_images or has_videos or has_sound):
        inputs_embeds = model.language_model.get_input_embeddings()(input_ids)
        return inputs_embeds, input_ids, attention_mask

    image_vit_embeds = video_vit_embeds = sound_embeds = None
    sound_attention_mask = None

    if has_images:
        pixel_values = pixel_values.to(dtype=model.vision_model.config.torch_dtype)
        image_vit_embeds = model.extract_feature(pixel_values)

    if has_videos:
        pixel_values_videos = pixel_values_videos.to(dtype=model.vision_model.config.torch_dtype)
        video_vit_embeds = model.extract_video_feature(pixel_values_videos)

    if has_sound:
        is_raw_waveform = False
        if isinstance(sound_clips, (list, tuple)):
            is_raw_waveform = True
            waveforms = sound_clips
        elif isinstance(sound_clips, np.ndarray):
            is_raw_waveform = True
            waveforms = [sound_clips.squeeze()] if sound_clips.ndim > 1 else [sound_clips]
        elif isinstance(sound_clips, torch.Tensor):
            if sound_clips.dim() == 1:
                is_raw_waveform = True
                waveforms = [sound_clips.cpu().numpy()]
            elif sound_clips.dim() == 2:
                is_raw_waveform = True
                waveforms = [clip.cpu().numpy() for clip in sound_clips]
            else:
                is_raw_waveform = False
        else:
            is_raw_waveform = False

        if is_raw_waveform:
            audio_inputs = model.sound_feature_extractor(
                waveforms,
                sampling_rate=model.sound_feature_extractor.sampling_rate,
                return_tensors="pt",
            )
            sound_input_features = audio_inputs.input_features
            sound_attention_mask = audio_inputs.get("attention_mask", None)
        else:
            sound_input_features = sound_clips
            sound_attention_mask = None

        target_device = model.sound_encoder.encoder.subsampling.linear.weight.device
        target_dtype = model.language_model.config.torch_dtype
        sound_input_features = sound_input_features.to(dtype=target_dtype, device=target_device)
        if sound_attention_mask is not None:
            sound_attention_mask = sound_attention_mask.to(device=target_device)
        sound_embeds = model.extract_sound_feature(sound_input_features, sound_attention_mask)

    inputs_embeds = model.language_model.get_input_embeddings()(input_ids)
    B, N, C = inputs_embeds.shape
    inputs_embeds = inputs_embeds.reshape(B * N, C)
    input_ids_copy = input_ids.reshape(B * N)

    if image_vit_embeds is not None:
        image_mask = (input_ids_copy == model.img_context_token_id)
        assert image_mask.sum() != 0, "No image tokens found in input_ids"
        inputs_embeds[image_mask] = image_vit_embeds.reshape(-1, C).to(inputs_embeds.device, inputs_embeds.dtype)

    if video_vit_embeds is not None:
        if B > 1:
            raise NotImplementedError("Video is not supported for batch size > 1")
        video_mask = (input_ids_copy == model.img_context_token_id)
        assert video_mask.sum() != 0, "No video tokens found in input_ids"
        inputs_embeds[video_mask] = video_vit_embeds.reshape(-1, C).to(inputs_embeds.device, inputs_embeds.dtype)

    if sound_embeds is not None and model.sound_context_token_id is not None:
        sound_mask = (input_ids_copy == model.sound_context_token_id)
        assert sound_mask.sum() != 0, "No sound tokens found in input_ids"
        if sound_embeds.dim() == 3 and sound_embeds.shape[0] > 1 and sound_attention_mask is not None:
            natural_input_lengths = sound_attention_mask.sum(-1) + 1
            output_lengths = model.sound_encoder.encoder._get_subsampling_output_length(natural_input_lengths)
            flat = torch.cat(
                [sound_embeds[i, : int(n)] for i, n in enumerate(output_lengths.tolist())],
                dim=0,
            )
        else:
            flat = sound_embeds.reshape(-1, C)
        assert sound_mask.sum().item() == flat.shape[0], (
            f"sound token count ({sound_mask.sum().item()}) != encoder output count ({flat.shape[0]})"
        )
        inputs_embeds[sound_mask] = flat.to(inputs_embeds.device, inputs_embeds.dtype)

    inputs_embeds = inputs_embeds.reshape(B, N, C)
    return inputs_embeds, input_ids, attention_mask
