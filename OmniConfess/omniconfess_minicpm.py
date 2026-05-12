# -*- coding: utf-8 -*-
# OmniConfess adapted for MiniCPM-o-2.6
# Based on omniconfess.py (Qwen2.5-Omni version)
# Key changes: model.thinker -> model.llm, processor API, embedding pipeline, layer indices
# -*- coding: utf-8 -*-
import time
from typing import List, Dict, Any, Optional, Tuple
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
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
try:
    from sentence_transformers import SentenceTransformer
    _SBERT_AVAILABLE = True
except ImportError:
    _SBERT_AVAILABLE = False
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig


INF = 10
TEMPERATURE = 0.6
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"


def load_image(image_path):
    """ """
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
    
def load_audio_data(audio_path, sr=24000):
    """"""
    if not audio_path or not os.path.exists(audio_path):
        return None
    try:
        audio, _ = librosa.load(audio_path, sr=sr)
        return audio if (audio is not None and len(audio) > 0) else None
    except:
        return None


class DualLayerAuditor:
    def __init__(self, model, processor, target_layers=[14, 20], store_hidden=False):
        self.model = model
        self.processor = processor
        self.target_layers = target_layers
        self.store_hidden = store_hidden
        self.intermediate_logits = {}
        self.intermediate_hidden = {}
        self.handles = []

    def _make_hook(self, layer_idx):
        def _hook(module, input, output):
            hidden_states = output[0] if isinstance(output, tuple) else output
            with torch.no_grad():
                if self.store_hidden:
                    self.intermediate_hidden[layer_idx] = hidden_states[:, -1, :].clone()
                normed_hidden = self.model.llm.model.norm(hidden_states)
                logits = self.model.llm.lm_head(normed_hidden)
                self.intermediate_logits[layer_idx] = logits[:, -1, :]
        return _hook

    def register(self):
        layers = list(self.target_layers)
        if self.store_hidden:
            final_idx = len(self.model.llm.model.layers) - 1
            if final_idx not in layers:
                layers.append(final_idx)
        for idx in layers:
            handle = self.model.llm.model.layers[idx].register_forward_hook(self._make_hook(idx))
            self.handles.append(handle)

    def remove(self):
        for h in self.handles: h.remove()
        self.handles = []
        self.intermediate_logits = {}
        self.intermediate_hidden = {}

STOP_WORDS = {"i", "is", "the", "a", "an", "and", "are", "was", "were", "be", "have", "has", "it", "they", "in", "on", "at", "to", "for", "of", "with", "yes", "no", "sound", "video", "audio", "hear", "heard", "see", "saw", "contains", "contain"}

def calculate_hallu_penalty(report, tokenizer, thresholds=None):
    """
    thresholds: dict with calibratable threshold values (or None for defaults).
    """
    if thresholds is None:
        thresholds = _DEFAULT_THRESHOLDS

    token_id = report['token_id']
    token_str = tokenizer.decode([token_id]).lower().strip()

    clean_word = ''.join(e for e in token_str if e.isalpha())
    if len(clean_word) <= 1 or clean_word in STOP_WORDS:
        return 0.0

    penalty = 0.0
    triggered = []

    if report['res_20_fin'] > thresholds['res_20_fin']:
        p_inc = 0.5
        if report['l20_conf'] > thresholds['l20_conf_late']:
            p_inc += 0.5
            triggered.append('LATE+CONF')
        else:
            triggered.append('LATE')

        penalty += p_inc

    if report['res_14_20'] > thresholds['res_14_20'] and report['l20_conf'] > thresholds['l20_conf_hijack']:
        p_inc = 0.6
        penalty += p_inc
        triggered.append('HIJACK')

    report['_triggered_modules'] = triggered
    return penalty

def _build_minicpm_data_dict(inputs):
    """Build MiniCPM-compatible data dict from processor output for get_vllm_embedding / get_omni_embedding."""
    device = inputs['input_ids'].device
    data = {
        'input_ids': inputs['input_ids'],
        'pixel_values': inputs.get('pixel_values', [[]]),
        'tgt_sizes': inputs.get('tgt_sizes', [[]]),
        'image_bound': inputs.get('image_bound', [torch.zeros(0, 2, dtype=torch.long, device=device)]),
        'audio_features': inputs.get('audio_features', []),
        'audio_feature_lens': inputs.get('audio_feature_lens', []),
        'audio_bounds': inputs.get('audio_bounds', [torch.zeros(0, 2, dtype=torch.long, device=device)]),
    }
    # Ensure list wrapping for batch dim
    if not isinstance(data['pixel_values'], list):
        data['pixel_values'] = [data['pixel_values']]
    if not isinstance(data['image_bound'], list):
        data['image_bound'] = [data['image_bound']]
    if not isinstance(data['audio_bounds'], list):
        data['audio_bounds'] = [data['audio_bounds']]
    return data


def _get_minicpm_embeds(model, inputs):
    """Get multimodal embeddings from MiniCPM model using its native pipeline."""
    data = _build_minicpm_data_dict(inputs)
    with torch.inference_mode():
        vllm_embedding, _ = model.get_vllm_embedding(data)
        if hasattr(model, 'apm') and model.config.init_audio:
            vllm_embedding = model.get_omni_embedding(
                data, input_embeddings=vllm_embedding,
                chunk_length=model.config.audio_chunk_length
            )
    return vllm_embedding


def safe_model_generate(model, processor, inputs, **gen_kwargs):
    """
    """
    import traceback

    # Apply max_new_tokens_override if set via _args passthrough
    _args = gen_kwargs.pop('_args', None)
    if _args is not None:
        if getattr(_args, 'max_new_tokens_override', None) is not None:
            gen_kwargs['max_new_tokens'] = _args.max_new_tokens_override
        if getattr(_args, 'repetition_penalty', 1.0) != 1.0:
            if 'repetition_penalty' not in gen_kwargs:
                gen_kwargs['repetition_penalty'] = _args.repetition_penalty

    # Get multimodal embeddings via MiniCPM pipeline
    try:
        embeds = _get_minicpm_embeds(model, inputs)
    except Exception as e:
        embeds = model.llm.get_input_embeddings()(inputs['input_ids'])

    # Clean incompatible kwargs
    clean_kwargs = {k: v for k, v in gen_kwargs.items() if k not in ['tokenizer', 'stop_strings']}

    try:
        out = model.llm.generate(
            inputs_embeds=embeds,
            attention_mask=inputs.get('attention_mask'),
            pad_token_id=model.processor.tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
            **clean_kwargs
        )
    except Exception as e:
        traceback.print_exc()
        raise e

    # Stitch prompt + generated sequences
    try:
        if hasattr(out, 'sequences'):
            full_seqs = []
            for seq in out.sequences:
                prompt_ids = inputs['input_ids'][0].to(seq.device)
                full_seq = torch.cat([prompt_ids, seq], dim=0)
                full_seqs.append(full_seq)
            out.sequences = torch.stack(full_seqs)
    except Exception as e:
        traceback.print_exc()
        raise e

    return out

def softmax(x):
    e_x = np.exp(np.array(x) - np.max(x))
    return e_x / e_x.sum(axis=0)


def get_modality_indices(processor, input_ids, device, image_bound=None, audio_bounds=None):
    """ MiniCPM processor  image_bound/audio_bounds 。

    """
    v_idx = None
    a_idx = None

    if image_bound is not None and len(image_bound) > 0:
        # image_bound: list of [batch_idx] of tensors [[start, end], ...]
        bounds = image_bound[0] if isinstance(image_bound, list) else image_bound
        if isinstance(bounds, torch.Tensor) and bounds.numel() > 0:
            all_indices = []
            for row in bounds:
                start, end = row[0].item(), row[1].item()
                all_indices.append(torch.arange(start, end, device=device))
            if all_indices:
                v_idx = torch.cat(all_indices)

    if audio_bounds is not None and len(audio_bounds) > 0:
        bounds = audio_bounds[0] if isinstance(audio_bounds, list) else audio_bounds
        if isinstance(bounds, torch.Tensor) and bounds.numel() > 0:
            all_indices = []
            for row in bounds:
                start, end = row[0].item(), row[1].item()
                all_indices.append(torch.arange(start, end, device=device))
            if all_indices:
                a_idx = torch.cat(all_indices)

    return v_idx, a_idx


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

    # Key token IDs (BPE tokenization — may differ for MiniCPM tokenizer)
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
    """ Logits """
    probs = torch.softmax(logits, dim=-1)
    return -torch.sum(probs * torch.log(probs + 1e-9), dim=-1).item()


def safe_thinker_step_audit(model, processor, inputs, past_key_values=None, position_ids=None, **gen_kwargs):
    """
    """
    valid_keys = ['input_ids', 'attention_mask', 'pixel_values', 'tgt_sizes',
                  'image_bound', 'audio_features', 'audio_feature_lens', 'audio_bounds']
    clean_inputs = {k: v for k, v in inputs.items() if k in valid_keys}
    
    for k, v in clean_inputs.items():
        if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
            clean_inputs[k] = v.to(torch.bfloat16)

    outputs = model.llm(
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
    """Parse command line arguments specifically for MiniCPM-o-2.6 OmniConfess"""
    parser = argparse.ArgumentParser(description="OmniConfess Algorithm for MiniCPM-o-2.6")

    # Model configuration
    # MiniCPM-o-2.6
    parser.add_argument('--model_id', type=str, default='minicpm-o-2.6',
                        help='Model identifier')
    parser.add_argument('--model_path', type=str, default='./models/MiniCPM-o-2.6',
                        help='Model path')
    parser.add_argument('--gpus', type=int, default=2,
                        help='Number of GPUs to use (tensor_parallel is handled by device_map in transformers)')

    # Data configuration
    parser.add_argument('--datasets', type=str, default='cmm',
                        help='Dataset type: cmm, gsm, math, etc.')
    parser.add_argument('--data_root', type=str, default='./OmniHalluBench/cmm/',
                        help='Root path for multimodal data (videos/audios)')
    parser.add_argument('--data_path', type=str,
                        default='./OmniHalluBench/cmm/all_data_final_reorg.json',
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
                        default='./results/time/',
                        help='Path to save timing information')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--max_samples', type=int, default=-1,
                        help='Max samples to process (-1 = all)')
    parser.add_argument('--halo_prompt_version', type=str, default='v0',
                        help='HaloQuest prompt version: v0 (current), v1 (explicit format), v2 (sentence-prefix), v3 (few-shot template)')
    parser.add_argument('--halueval_prompt_version', type=str, default='v0',
                        help='HaluEval prompt version: v0 (baseline), v1 (yes/no fix + few-shot)')
    parser.add_argument('--avh_prompt_version', type=str, default='v0',
                        help='AVHBench prompt version: v0 (baseline), v1 (25w cap + English-only)')
    parser.add_argument('--cd_mode', type=str, default='hard', choices=['hard', 'soft'],
                        help='Dist-Confess CD correction mode: hard (zero out token) or soft (scale down)')
    parser.add_argument('--cd_soft_factor', type=float, default=0.1,
                        help='Soft CD scaling factor (only used when cd_mode=soft)')
    parser.add_argument('--layer_fix_mode', type=str, default='linear',
                        choices=['linear', 'linear_norm', 'slerp'],
                        help='Layer-Confess hidden state repair mode')
    parser.add_argument('--path_cluster_mode', type=str, default='tfidf',
                        choices=['tfidf', 'sbert'],
                        help='Path-Confess clustering mode: tfidf or sbert')
    parser.add_argument('--disable_passage_mdi', action='store_true', default=False,
                        help='Disable passage-MDI for text-only datasets (halueval/pubmedqa/ragtruth)')
    parser.add_argument('--ablation_suite', type=str, default=None,
                        help='Path to ablation config JSON; runs multiple configs sequentially '
                             'without reloading model')
    parser.add_argument('--max_new_tokens_override', type=int, default=None,
                        help='Override max_new_tokens for specific dataset tuning')
    parser.add_argument('--repetition_penalty', type=float, default=1.0,
                        help='Repetition penalty for generation (1.0 = no penalty)')

    # === Calibration & Tuning Flags ===
    parser.add_argument('--dump_internals', action='store_true', default=False,
                        help='Dump per-token MDI/layer-drift internals during foresight. '
                             'Output: <output_dir>/internals/<dataset>/<sample_id>.jsonl')
    parser.add_argument('--bypass_path', action='store_true', default=False,
                        help='Bypass Path-Confess veto: skip veto judgment, pick lowest-penalty '
                             'path among all 4 rollouts. Dist/Layer-Confess still active.')
    parser.add_argument('--threshold_overrides', type=str, default=None,
                        help='JSON string or path to JSON file overriding calibration thresholds. '
                             'Keys: res_20_fin, l20_conf_late, res_14_20, l20_conf_hijack, '
                             'mdi_threshold, cd_alpha, layer_repair_alpha, veto_threshold')
    parser.add_argument('--use_base_prompt', action='store_true', default=False,
                        help='Replace OmniConfess prompts with base v0 prompts everywhere (system, user, '
                             'final generation). OmniConfess pipeline flow is unchanged. For Exp-X2 factorial.')

    return parser.parse_args()


# === Base v0 Prompt Dictionaries (for --use_base_prompt, Exp-X2 factorial) ===
_BASE_V0_SYSTEM = {
    "cmm": (
        "You are an audio-visual reporter. Your task is to provide a final judgment based on what you see and hear.\n\n"
        "Judgment: \\boxed{Yes} or \\boxed{No}"
    ),
    "phd": (
        "You are a visual analyst. Your task is to evaluate a specific hypothesis against visual data.\n\n"
        "Judgment: \\boxed{Yes} or \\boxed{No}"
    ),
    "halo": (
        "You are a visual assistant. Your task is to answer the question based on the provided image.\n"
        "Output EXACTLY ONE single sentence.\n"
        "At the end, output: Answer:[your answer here]."
    ),
    "avh": (
        "You are an AI assistant. Describe the video and audio content.\n"
        "Output EXACTLY ONE single sentence.\n"
        "At the end, output: Answer:[your answer here]."
    ),
    "ragtruth": (
        "You will be presented with a question.\n"
        "Answer the user's question based on the given information.\n"
        "At the end, output: Answer:[your answer here]."
    ),
    "halueval": (
        "You will be presented with a question.\n"
        "Answer the user's question based on the given information.\n"
        "At the end, output: Answer:[your answer here]."
    ),
    "pubmedqa": (
        "You will be given a PubMed-style passage and a Yes/No/Maybe question.\n"
        "Answer rules:\n"
        "1. Begin with exactly one of: \"Yes.\" / \"No.\" / \"Maybe.\"\n"
        "2. Summarize the main conclusion from the passage in EXACTLY ONE single sentence.\n"
        "At the end, output: Answer:[your answer here]."
    ),
}

def _base_v0_system_for(dataset_name):
    """Lookup base v0 system prompt by dataset name (substring match)."""
    ds = dataset_name.lower()
    for key in ["cmm", "avh", "phd", "halo", "ragtruth", "halueval", "pubmedqa"]:
        if key in ds:
            return _BASE_V0_SYSTEM[key]
    return "You are a helpful AI assistant. Please provide answers to user questions based on the provided media content.\nAt the end, output: Answer:[your answer here]."

def _base_v0_user_text(dataset_name, question, passage=None):
    """Return base v0 user text (no media prefix) for a given dataset."""
    ds = dataset_name.lower()
    if "cmm" in ds:
        return f"Task: Analyze the video and audio to answer the question.\nQuestion: {question}\nFinal Answer:"
    if "avh" in ds:
        return question
    if "phd" in ds or "halo" in ds:
        return f"Question: {question}\nAnswer:"
    if "ragtruth" in ds or "pubmedqa" in ds:
        passage_str = passage or ''
        return (
            f"Read the following text and answer the question.\n\n"
            f"Text: {passage_str}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )
    if "halueval" in ds:
        passage_str = passage or ''
        return (
            f"Read the following text and answer the question.\n\n"
            f"Text: {passage_str}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )
    # fallback
    passage_str = passage or ''
    return f"Question: {question}\nAnswer:"


# === Threshold Override System ===
# Default values (copied from Qwen, the "original hardcoded" constants)
_DEFAULT_THRESHOLDS = {
    'res_20_fin': 1.35,        # Layer-Confess LATE detection
    'l20_conf_late': 0.90,     # Layer-Confess LATE confidence gate
    'res_14_20': 1.40,         # Layer-Confess HIJACK detection
    'l20_conf_hijack': 0.85,   # Layer-Confess HIJACK confidence gate
    'mdi_threshold': -0.3,     # Dist-Confess CD trigger
    'cd_alpha': 0.5,           # Contrastive decoding strength
    'layer_repair_alpha': 0.3, # Layer-Confess logit interpolation weight
    'veto_threshold': 0.45,    # Path-Confess veto cutoff
}

def load_threshold_overrides(args):
    """Parse --threshold_overrides into a dict, merged with defaults."""
    thresholds = dict(_DEFAULT_THRESHOLDS)
    raw = getattr(args, 'threshold_overrides', None)
    if not raw:
        return thresholds
    # Try as file path first, then as JSON string
    if os.path.isfile(raw):
        with open(raw, 'r') as f:
            overrides = json.load(f)
    else:
        overrides = json.loads(raw)
    # Validate keys
    unknown = set(overrides.keys()) - set(_DEFAULT_THRESHOLDS.keys())
    if unknown:
        pass
    for k, v in overrides.items():
        if k in thresholds:
            thresholds[k] = float(v)
    return thresholds


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
        self.model = None
        self.processor = None
        self.tokenizer = None
        self._sbert_model = None  # lazy-loaded for path_cluster_mode=sbert
        self.thresholds = load_threshold_overrides(args)
        self.initialize_model()

    def initialize_model(self):
        """Initialize the MiniCPM-o-2.6 model and multimodal processor"""
        model_path = self._get_model_path()

        # Fix transformers 4.52+ incompatibility with MiniCPM-o custom modules
        import types as _types
        def _patched_smart_apply(self_mod, fn):
            for module in self_mod.children():
                if hasattr(module, "_init_weights"):
                    if hasattr(module, "_initialize_weights"):
                        module.smart_apply(module._initialize_weights)
                    else:
                        module.smart_apply(fn)
                else:
                    module.smart_apply(fn)
            fn(self_mod)
            return self_mod
        torch.nn.Module.smart_apply = _patched_smart_apply

        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            attn_implementation='sdpa',
            torch_dtype=torch.bfloat16,
            init_vision=True,
            init_audio=True,
            init_tts=False,
        ).eval().cuda()

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        # MiniCPM stores its own processor internally
        self.processor = self.model.processor

        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token


        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)

    def _get_model_path(self):
        """Get the appropriate model path"""
        return self.args.model_path

    def _call_minicpm_processor(self, text, image_values=None, audio_values=None):
        """Wrapper for MiniCPM processor call.

        Args:
            text: str or list of str
            image_values: PIL.Image or None
            audio_values: numpy array or None
        Returns:
            MiniCPMOBatchFeature with input_ids, attention_mask, pixel_values,
            tgt_sizes, image_bound, audio_features, audio_feature_lens, audio_bounds, etc.
        """
        if isinstance(text, str):
            text = [text]
        images_arg = [image_values] if image_values is not None else None
        audios_arg = [audio_values] if audio_values is not None else None
        result = self.processor(
            text=text,
            images=images_arg,
            audios=audios_arg,
            max_slice_nums=1,  # limit image slices to save VRAM
            use_image_id=False,
            return_tensors="pt",
        )
        # Move tensors to model device (handle nested lists of tensors, e.g. pixel_values)
        device = self.model.llm.model.embed_tokens.weight.device
        def _to_device(obj):
            if isinstance(obj, torch.Tensor):
                return obj.to(device)
            elif isinstance(obj, list):
                return [_to_device(item) for item in obj]
            return obj
        for k, v in result.items():
            result[k] = _to_device(v)
        return result

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

    def get_system_prompt(self, dataset_type):
        """
        """
        # --use_base_prompt: return base v0 system prompt instead of OmniConfess prompt
        if getattr(self.args, 'use_base_prompt', False):
            return _base_v0_system_for(dataset_type)

        omni_identity = "You are an expert multimodal assistant capable of perceiving auditory and visual inputs."
        
        if "cmm" in dataset_type.lower():
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

        return f"{omni_identity}\nSolve the problem step by step based on the provided content."

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

    def select_response(self, responses, logprobs, advantages):
        """
        """
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
                
                print(f"    [CLUSTER DEBUG] sizes: {[len(c) for c in cluster_list]}, "
                      f"size_scores: {[round(s,3) for s in cluster_size_ratios]}, "
                      f"quality_scores: {[round(s,3) for s in cluster_adv_scores]} "
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
        """
        """
        import os

        token_stats = {"input": 0, "output": 0}
        rollout_stats = {"total": 0, "saved": 0}

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
            
        audio_values = None
        if a_rel and os.path.exists(a_path):
            audio_values = load_audio_data(a_path)

        image_values = None
        if i_rel:
            if not os.path.exists(i_path):
                if 'val2014' in i_path:
                    i_path = i_path.replace('val2014', 'train2014')
                elif 'train2014' in i_path:
                    i_path = i_path.replace('train2014', 'val2014')

            if os.path.exists(i_path):
                image_values = load_image(i_path)
            else:
                pass

        example['pixel_values'] = pixel_values
        example['audio_values'] = audio_values
        example['image_values'] = image_values
        
        pixel_values = None
        if v_rel:
            if os.path.exists(v_path):
                pixel_values = load_video_frames(v_path, num_frames=8)
            else:
                pass
                
        audio_values = None
        if a_rel:
            if os.path.exists(a_path):
                audio_values = load_audio_data(a_path)
            else:
                pass

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

        final_result = self._generate_final_response(
            example,
            system_prompt,
            previous_steps,
            previous_values,
            token_stats,
            rollout_stats,
            traj_info
        )

        traj_info['token_num'] = token_stats["input"] + token_stats["output"]

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
        """
        """
        stop_foresight = False
        pixel_values = example.get('pixel_values')
        audio_values = example.get('audio_values')
        image_values = example.get('image_values')

        all_responses_first_stage = []
        all_logprobs_first_stage = []
        all_advantages_first_stage = []
        all_token_nums_first_stage = []
        
        _need_hidden = getattr(self.args, 'layer_fix_mode', 'linear') != 'linear'
        auditor = DualLayerAuditor(self.model, self.processor, store_hidden=_need_hidden)
        auditor.register()
        _all_dump_records = []  # for --dump_internals

        for beam_idx in range(self.args.step_beam_size):
            
            chat = self._prepare_chat_template_for_first_stage(example, system_prompt)
            p_str = self.processor.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            if isinstance(p_str, list): p_str = p_str[0]
            
            inputs_text = p_str.replace(self.tokenizer.eos_token, "").strip() + previous_steps[beam_idx]
            
            m_inputs = self._call_minicpm_processor(inputs_text, image_values=image_values, audio_values=audio_values)

            token_stats["input"] += m_inputs.input_ids.numel()

            # =====================================================================
            # =====================================================================
            inputs_embeds_f = _get_minicpm_embeds(self.model, m_inputs)

            attn_mask_f = m_inputs['attention_mask'].clone() if isinstance(m_inputs, dict) else m_inputs.attention_mask.clone()
            pos_ids_f = m_inputs.get('position_ids', None) if isinstance(m_inputs, dict) else getattr(m_inputs, 'position_ids', None)

            # =====================================================================
            # =====================================================================
            m_nv = attn_mask_f.clone()
            _img_bound = m_inputs.get('image_bound', None) if isinstance(m_inputs, dict) else getattr(m_inputs, 'image_bound', None)
            _aud_bounds = m_inputs.get('audio_bounds', None) if isinstance(m_inputs, dict) else getattr(m_inputs, 'audio_bounds', None)
            v_idx, a_idx = get_modality_indices(self.processor, m_inputs['input_ids'] if isinstance(m_inputs, dict) else m_inputs.input_ids, self.model.llm.model.embed_tokens.weight.device, image_bound=_img_bound, audio_bounds=_aud_bounds)
            if v_idx is not None:
                m_nv[:, v_idx] = 0
            if a_idx is not None:
                m_nv[:, a_idx] = 0
            is_text_only = (v_idx is None and a_idx is None)  # P1: skip CD-Confess for text-only

            if is_text_only and not getattr(self.args, 'disable_passage_mdi', False) \
                    and self.args.datasets.lower() in ('halueval', 'pubmedqa', 'ragtruth', 'summedits'):
                passage_idx = get_passage_indices(m_inputs.input_ids, self.tokenizer, self.model.device)
                if passage_idx is not None and len(passage_idx) > 0:
                    m_nv[:, passage_idx] = 0
                    is_text_only = False

            # ================================================================
            # ================================================================
            is_avhbench = "avh" in self.args.datasets.lower()

            for rollout_idx in range(self.args.num_rollout):
                torch.cuda.empty_cache()
                _dump_records = []  # per-token internals for --dump_internals
                with torch.inference_mode():
                    p_full, p_nv = None, None
                    generated_ids_list = []
                    acc_logprob, total_hallu_penalty = 0.0, 0.0

                    m_f_step = attn_mask_f.clone()
                    m_nv_step = m_nv.clone()
                    pos_ids_step = pos_ids_f
                    curr_ids = None

                    for i in range(48): 
                        if i == 0:
                            out_f = self.model.llm(inputs_embeds=inputs_embeds_f, attention_mask=m_f_step, position_ids=pos_ids_step, use_cache=True)
                            out_nv = self.model.llm(inputs_embeds=inputs_embeds_f, attention_mask=m_nv_step, position_ids=pos_ids_step, use_cache=True)
                        else:
                            out_f = self.model.llm(input_ids=curr_ids, attention_mask=m_f_step, position_ids=pos_ids_step, past_key_values=p_full, use_cache=True)
                            out_nv = self.model.llm(input_ids=curr_ids, attention_mask=m_nv_step, position_ids=pos_ids_step, past_key_values=p_nv, use_cache=True)

                        logits_f = out_f.logits[:, -1, :]
                        p_full = out_f.past_key_values
                        p_nv = out_nv.past_key_values
                        
                        l16_logits = auditor.intermediate_logits.get(14, logits_f).clone()
                        l22_logits = auditor.intermediate_logits.get(20, logits_f).clone()

                        # --- Layer-Confess hidden state repair (ablation A3) ---
                        _lfm = getattr(self.args, 'layer_fix_mode', 'linear')
                        if _lfm != 'linear' and auditor.store_hidden and not is_text_only:
                            _final_idx = len(self.model.llm.model.layers) - 1
                            h_early = auditor.intermediate_hidden.get(20)
                            h_final = auditor.intermediate_hidden.get(_final_idx)
                            if h_early is not None and h_final is not None:
                                _alpha_lr = self.thresholds['layer_repair_alpha']
                                if _lfm == 'linear_norm':
                                    h_repaired = (1 - _alpha_lr) * h_final + _alpha_lr * h_early
                                    _fn = h_final.norm()
                                    _rn = h_repaired.norm()
                                    if _rn > 1e-8:
                                        h_repaired = h_repaired * (_fn / _rn)
                                elif _lfm == 'slerp':
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
                                # Reproject to logits
                                with torch.no_grad():
                                    _normed = self.model.llm.model.norm(h_repaired.unsqueeze(0))
                                    logits_f = self.model.llm.lm_head(_normed).squeeze(0)

                        # =====================================================================
                        # =====================================================================
                        temp_probs = torch.softmax(logits_f / TEMPERATURE, dim=-1)
                        token_id = torch.multinomial(temp_probs, num_samples=1).item()
                        
                        prob_f_val = temp_probs[0, token_id].item()
                        prob_nv_val = torch.softmax(out_nv.logits[:, -1, :] / TEMPERATURE, dim=-1)[0, token_id].item()
                        mdi_av = (prob_f_val - prob_nv_val) / (prob_f_val + 1e-9)
                        
                        token_str = self.tokenizer.decode([token_id]).lower().strip()
                        clean_word = ''.join(e for e in token_str if e.isalpha())

                        # P1: skip for text-only; P2: skip EOS and \boxed tokens
                        _mdi_thresh = self.thresholds['mdi_threshold']
                        if (mdi_av < _mdi_thresh and len(clean_word) > 3 and clean_word not in STOP_WORDS
                                and not is_text_only
                                and token_id != self.tokenizer.eos_token_id
                                and 'boxed' not in clean_word):

                            alpha = self.thresholds['cd_alpha']
                            blind_logits = out_nv.logits[:, -1, :]
                            refined_logits = logits_f + alpha * (logits_f - blind_logits)
                            
                            probs_f = torch.softmax(refined_logits / TEMPERATURE, dim=-1)
                            
                            if getattr(self.args, 'cd_mode', 'hard') == 'soft':
                                _sf = getattr(self.args, 'cd_soft_factor', 0.1)
                                probs_f[0, token_id] *= _sf
                            else:
                                probs_f[0, token_id] = 0.0
                            probs_f = probs_f / probs_f.sum(dim=-1, keepdim=True)
                            
                            next_id = torch.multinomial(probs_f, num_samples=1)
                            token_id = next_id.item()
                            
                            pf_val = torch.softmax(logits_f, dim=-1)[0, token_id].item()
                            pnv_val = torch.softmax(out_nv.logits[:, -1, :] / TEMPERATURE, dim=-1)[0, token_id].item()  # P0: fix temperature mismatch
                            mdi_av = (pf_val - pnv_val) / (pf_val + 1e-9)
                            
                        else:
                            next_id = torch.tensor([[token_id]], device=self.model.device)
                            token_id = next_id.item()
                            pf_val = prob_f_val
                            pnv_val = prob_nv_val
                        _report = {
                            "token_id": token_id,
                            "res_14_20": torch.norm(torch.softmax(l16_logits, dim=-1) - torch.softmax(l22_logits, dim=-1), p=2).item(),
                            "res_20_fin": torch.norm(torch.softmax(l22_logits, dim=-1) - torch.softmax(logits_f, dim=-1), p=2).item(),
                            "l20_conf": torch.max(torch.softmax(l22_logits, dim=-1)).item(),
                        }
                        penalty = calculate_hallu_penalty(_report, self.tokenizer, self.thresholds)
                        total_hallu_penalty += penalty

                        # --dump_internals: record per-token metrics
                        if getattr(self.args, 'dump_internals', False):
                            _dump_records.append({
                                'token_idx': len(generated_ids_list),
                                'token_text': self.tokenizer.decode([token_id]),
                                'mdi_av': round(mdi_av, 6),
                                'res_14_20': round(_report['res_14_20'], 6),
                                'res_20_fin': round(_report['res_20_fin'], 6),
                                'l20_conf': round(_report['l20_conf'], 6),
                                'hallu_penalty': round(penalty, 6),
                                'triggered_modules': _report.get('_triggered_modules', []),
                            })

                        if token_id == self.tokenizer.eos_token_id: break
                        generated_ids_list.append(token_id)
                        acc_logprob += np.log(max(pf_val, 1e-9))

                        curr_ids = next_id
                        
                        new_bit = torch.ones((1, 1), device=self.model.device, dtype=m_f_step.dtype)
                        m_f_step = torch.cat([m_f_step, new_bit], dim=-1)
                        m_nv_step = torch.cat([m_nv_step, new_bit], dim=-1)
                        
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
                    _veto_thresh = self.thresholds['veto_threshold']
                    if getattr(self.args, 'bypass_path', False):
                        # --bypass_path: NO veto, use penalty as soft disadvantage
                        advantage = logprob - (total_hallu_penalty * 0.8)
                    elif total_hallu_penalty >= _veto_thresh:
                        advantage = -10000.0
                        logprob = -10000.0
                        _pcs['num_paths_vetoed'] += 1
                    else:
                        advantage = logprob - (total_hallu_penalty * 0.8)

                    all_responses_first_stage.append(response)
                    all_logprobs_first_stage.append(logprob)
                    all_advantages_first_stage.append(advantage)
                    all_token_nums_first_stage.append(len(generated_ids_list))
                    token_stats["output"] += len(generated_ids_list)
                    rollout_stats["total"] += 1

                    # --dump_internals: accumulate per-rollout records
                    if getattr(self.args, 'dump_internals', False) and _dump_records:
                        _all_dump_records.append({
                            'beam_idx': beam_idx,
                            'rollout_idx': rollout_idx,
                            'total_hallu_penalty': round(total_hallu_penalty, 6),
                            'vetoed': bool(advantage <= -9000.0),
                            'tokens': _dump_records,
                        })

        auditor.remove()

        # --dump_internals: attach records to traj_info for file writing in main loop
        if getattr(self.args, 'dump_internals', False) and _all_dump_records:
            if '_dump_internals' not in traj_info:
                traj_info['_dump_internals'] = []
            traj_info['_dump_internals'].extend(_all_dump_records)

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
                
                safe_available = [
                    i for i in range(len(all_logprobs_first_stage)) 
                    if i not in keep_foresight_list and all_logprobs_first_stage[i] > -9000.0
                ]
                
                if safe_available:
                    num_to_add = min(self.args.step_beam_size - len(keep_foresight_list), len(safe_available))
                    
                    safe_available.sort(key=lambda x: all_logprobs_first_stage[x], reverse=True)
                    additional_indices = safe_available[:num_to_add]
                    
                    keep_foresight_list.extend(additional_indices)
                else:
                    print("    ️ Beam Size 。", flush=True)
                    traj_info['path_confess_stats']['beam_shrink_occurred'] = True

            keep_foresight_list.sort()
            rollout_stats["saved"] += (len(all_logprobs_first_stage) - len(keep_foresight_list))

            all_responses = [all_responses_first_stage[i] for i in keep_foresight_list]
            filtered_logprobs = [all_logprobs_first_stage[i] for i in keep_foresight_list]
            filtered_advantages = [all_advantages_first_stage[i] for i in keep_foresight_list]

        completed_responses = []
        completed_logprobs = []
        completed_advantages = []


        for idx in range(len(keep_foresight_list)):
            response = all_responses[idx]
            beam_idx = keep_foresight_list[idx] // self.args.num_rollout
            
            chat = self._prepare_chat_template(example, system_prompt)
            p_str = self.processor.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            if isinstance(p_str, list): p_str = p_str[0]
            
            clean_resp = response.replace("<|video|>", "").replace("<|audio|>", "").replace("<|vision_start|>", "").replace("<|vision_end|>", "")
            inputs_text = p_str.replace(self.tokenizer.eos_token, "").strip() + previous_steps[beam_idx] + clean_resp

            m_inputs = self._call_minicpm_processor(inputs_text, image_values=image_values, audio_values=audio_values)

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
            k_num = self.args.cluster_num
            # Need at least k_num non-empty responses for clustering
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

            step_info = {
                'first_stage': {
                    'candidates': [
                        {
                            'text': text,
                            'logprob': round(float(all_logprobs_first_stage[i]), 4),
                            'advantage': round(float(all_advantages_first_stage[i]), 4),
                            'is_vetoed': bool(all_advantages_first_stage[i] <= -9000.0)
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
            print(f'️ : {e}。。', flush=True)
            traj_info['path_confess_stats']['cluster_fallback_occurred'] = True
            # BUG FIX: completed_advantages / keep_foresight_list may be empty when all beams
            # produced empty responses; fall back entirely to first-stage results in that case.
            if completed_advantages and keep_foresight_list:
                adv_source = completed_advantages
                resp_source = completed_responses
                logp_source = completed_logprobs
                idx_source  = keep_foresight_list
            else:
                print('️ completedfirst-stage。', flush=True)
                adv_source  = all_advantages_first_stage
                resp_source = all_responses_first_stage
                logp_source = all_logprobs_first_stage
                idx_source  = list(range(len(all_responses_first_stage)))
            if not adv_source:
                print('️ 。', flush=True)
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
        """
        """
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
            print(f" []  {current_step}: 。")
            return True

        if self.args.depth_pruning_strategy == "cluster":
            if step_results.get("stop_foresight", False):
                print(f" []  {current_step}:  {self.args.threshold} ()。")
                return True

        return False

    def _generate_final_response(self, example, system_prompt, previous_steps, previous_values, token_stats, rollout_stats, traj_info):
        """
        """
        pixel_values = example.get('pixel_values')
        audio_values = example.get('audio_values')
        image_values = example.get('image_values')

        traj_info['internal_reasoning'] = previous_steps 

        all_responses = []
        all_logprobs = []
        all_advantages = []


        for beam_idx in range(self.args.step_beam_size):
            chat = self._prepare_chat_template(example, system_prompt)
            p_str = self.processor.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
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
                # --use_base_prompt: simple base-style final instruction with reasoning notes
                if getattr(self.args, 'use_base_prompt', False):
                    avh_max_tok = 60
                    avh_final_instruction = (
                        f"\n\n[Observations]:\n{reasoning_history.strip()}\n\n"
                        "Answer:"
                    )
                elif _AVH_PROMPTS_LOADED:
                    apv = getattr(self.args, 'avh_prompt_version', 'v0')
                    _, avh_final_tmpl, avh_max_tok = _get_avh_prompt(apv)
                    avh_final_instruction = avh_final_tmpl.format(notes=reasoning_history.strip())
                else:
                    apv = getattr(self.args, 'avh_prompt_version', 'v0')
                    avh_max_tok = 60
                    avh_final_instruction = (
                        f"\n\n[Your observations]:\n{reasoning_history.strip()}\n\n"
                        "Based on your observations, write one sentence describing what is happening in this audio-visual clip.\n"
                        "Answer: "
                    )

                inputs_text_avh = p_str.replace(self.tokenizer.eos_token, "").strip() + avh_final_instruction


                m_inputs_avh = self._call_minicpm_processor(inputs_text_avh, image_values=image_values, audio_values=audio_values)

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

                    gen_caption = raw_text.split('<|im_end|>')[0].split('\n')[0].strip()
                    if gen_caption.startswith('\\boxed{') or gen_caption.startswith('boxed{'):
                        gen_caption = gen_caption.lstrip('\\').lstrip('boxed{').rstrip('}').strip()

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
                # --use_base_prompt: simple base-style final instruction with reasoning notes
                if getattr(self.args, 'use_base_prompt', False):
                    halo_max_tok = 100
                    halo_final_instruction = (
                        f"\n\n[Observations]:\n{reasoning_history.strip()}\n\n"
                        "Answer:"
                    )
                elif _HALO_PROMPTS_LOADED and hpv in ('v4', 'v5', 'v6', 'v7', 'v8', 'v9'):
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

                m_inputs_halo = self._call_minicpm_processor(inputs_text_halo, image_values=image_values, audio_values=audio_values)

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
                    if gen_caption.startswith('\\boxed{') or gen_caption.startswith('boxed{'):
                        gen_caption = gen_caption.lstrip('\\').lstrip('boxed{').rstrip('}').strip()

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
                # --use_base_prompt: simple base-style final instruction with reasoning notes
                if getattr(self.args, 'use_base_prompt', False):
                    halu_max_tok = 100 if self.args.datasets.lower() != "ragtruth" else 300
                    text_final_instruction = (
                        f"\n\n[Reasoning]:\n{reasoning_history.strip()}\n\n"
                        "Answer:"
                    )
                # P3-a: HaluEval uses short-phrase constraint to match GT entity format
                elif self.args.datasets.lower() == "halueval":
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
                
                m_inputs_pure = self._call_minicpm_processor(inputs_text_pure)

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
                # --use_base_prompt: single-stage generation with reasoning history + base-style prompt
                # NOTE: This bypasses the two-stage Visual Checklist + forced re-judgment.
                # The model receives reasoning notes from foresight but uses simpler instructions,
                # which may cause format issues (expected — measuring prompt effect).
                if getattr(self.args, 'use_base_prompt', False):
                    question_text = example.get('question') or example.get('input')
                    base_final_instruction = (
                        f"\n\n[Reasoning]:\n{reasoning_history.strip()}\n\n"
                        f"Question: {question_text}\n"
                        "Final Answer:"
                    )
                    inputs_text_d = p_str.replace(self.tokenizer.eos_token, "").strip() + base_final_instruction

                    m_inputs_d = self._call_minicpm_processor(inputs_text_d, image_values=image_values, audio_values=audio_values)
                    token_stats["input"] += m_inputs_d.input_ids.numel()

                    with torch.inference_mode():
                        out_d = safe_model_generate(
                            self.model, self.processor, m_inputs_d,
                            max_new_tokens=200,
                            do_sample=False,
                            stop_strings=["\n\n", "<|im_end|>", "Human:"],
                            repetition_penalty=1.1,
                            tokenizer=self.tokenizer,
                            _args=self.args
                        )

                        gen_ids_d = out_d.sequences[0][m_inputs_d.input_ids.shape[1]:]
                        raw_text_d = self.tokenizer.decode(gen_ids_d, skip_special_tokens=True).strip()
                        # Extract Yes/No judgment if present, otherwise keep raw text
                        import re
                        match_d = re.search(r'\\?boxed\{(Yes|No)\}|(Yes|No)', raw_text_d, re.IGNORECASE)
                        if match_d:
                            clean_judg = (match_d.group(1) or match_d.group(2)).capitalize()
                            final_output = f"\\boxed{{{clean_judg}}}"
                        else:
                            final_output = raw_text_d.split('\n\n')[0].strip()

                        if gen_ids_d.numel() > 0 and hasattr(out_d, 'scores'):
                            step_logprobs = []
                            for step_idx, step_logits in enumerate(out_d.scores):
                                token_id = gen_ids_d[step_idx]
                                log_probs = torch.nn.functional.log_softmax(step_logits[0].float(), dim=-1)
                                step_logprobs.append(log_probs[token_id].item())
                            logprob = sum(step_logprobs) / len(step_logprobs)
                        else:
                            logprob = 0.0

                    token_stats["output"] += len(gen_ids_d)

                else:
                    # Original two-stage Visual Checklist logic
                    if "\\boxed{" in reasoning_history:
                        base_input = p_str.replace(self.tokenizer.eos_token, "").strip() + f"\n\nBased on your detailed internal reasoning: '{reasoning_history.strip()}'\nNow, "
                    else:
                        base_input = p_str.replace(self.tokenizer.eos_token, "").strip() + "\n" + reasoning_history

                    question_text = example.get('question') or example.get('input')

                    inputs_text_s1 = base_input + "\nCRITICAL RULE: Do NOT repeat the context. You MUST use bullet points to list the objective visual facts.\nVisual Checklist:\n- "

                    m_inputs_s1 = self._call_minicpm_processor(inputs_text_s1, image_values=image_values, audio_values=audio_values)

                    token_stats["input"] += m_inputs_s1.input_ids.numel()

                    with torch.inference_mode():
                        out_s1 = safe_model_generate(
                            self.model, self.processor, m_inputs_s1,
                            max_new_tokens=200,
                            do_sample=False,
                            stop_strings=["Judgment:", "\n\n", "Human:"],
                            repetition_penalty=1.1,
                            tokenizer=self.tokenizer,
                            _args=self.args
                        )

                        s1_gen_ids = out_s1.sequences[0][m_inputs_s1.input_ids.shape[1]:]
                        gen_desc = self.tokenizer.decode(s1_gen_ids, skip_special_tokens=True).strip()
                        clean_desc = gen_desc.split("Judgment:")[0].split("Final")[0].replace("\\boxed{Yes}", "").replace("\\boxed{No}", "").strip()

                    # ==========================================================
                    # ==========================================================
                    inputs_text_s2 = inputs_text_s1 + " " + clean_desc + f"\n\nQuestion: {question_text}\nBased STRICTLY on your analysis above, answer the Question. Judgment: \\boxed{{"

                    m_inputs_s2 = self._call_minicpm_processor(inputs_text_s2, image_values=image_values, audio_values=audio_values)

                    with torch.inference_mode():
                        out_s2 = safe_model_generate(
                            self.model, self.processor, m_inputs_s2,
                            max_new_tokens=20,
                            do_sample=False,
                            stop_strings=["}", "Final"],
                            tokenizer=self.tokenizer,
                            _args=self.args
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

                    final_output = f"Visual Checklist:\n- {clean_desc}\nJudgment: \\boxed{{{clean_judg}}}"
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

        traj_info['final_part'] = {
            'response': all_responses[selected_idx],
            'raw_completion_text': all_responses,
            'logprobs': [round(float(x), 4) for x in all_logprobs],
            'selected_idx': int(selected_idx)
        }

        return {
            "response": all_responses[selected_idx],
            "trajectories": all_responses,
            "rollout_stats": rollout_stats,
            "token_stats": token_stats,
            "traj_info": traj_info
        }

    def _build_user_text(self, example, system_prompt, stage='completion'):
        """Build user text content string for MiniCPM chat template (shared between stages)."""
        question = example.get('input') or example.get('question')
        dataset_name = self.args.datasets.lower()
        prefix_parts = []

        # MiniCPM media tags
        if example.get('image_values') is not None:
            prefix_parts.append("(<image>./</image>)")
        if example.get('audio_values') is not None:
            prefix_parts.append("(<audio>./</audio>)")

        media_prefix = "\n".join(prefix_parts)
        if media_prefix:
            media_prefix += "\n"

        # --use_base_prompt: return base v0 user text (both stages)
        if getattr(self.args, 'use_base_prompt', False):
            passage = example.get('passage', '') or example.get('context', '')
            base_text = _base_v0_user_text(dataset_name, question, passage=passage)
            return media_prefix + base_text

        if "cmm" in dataset_name:
            text = (
                f"Task: Verify the existence of '{question}' through independent modal cross-checking.\n"
                "Step 1: Check the video stream for visual evidence.\n"
                "Step 2: Check the audio stream for auditory evidence.\n"
                "Step 3: Compare both sources. Only confirm 'Yes' if there is direct, non-conflicting evidence. "
                "If one modality suggests it but the other contradicts, prioritize empirical observation over intuition.\n"
                + ("Reason step by step. If previous steps are provided, continue logically." if stage == 'first' else
                   "Please solve the problem step by step based on the provided media content.")
            )
        elif "avh" in dataset_name:
            if stage == 'first':
                text = f"Based STRICTLY on the raw visual and auditory evidence provided (do not guess or assume), analyze this Question: {question}\nProvide the NEXT reasoning step or description."
            else:
                text = f"Based STRICTLY on the raw visual and auditory evidence provided (do not guess or assume), answer this Question: {question}"
        elif "phd" in dataset_name:
            if stage == 'first':
                text = (
                    f"Background Text (Unverified): {question}\n"
                    "Please test the final question against the image objectively.\n"
                    "Provide your analysis using bullet points for clarity.\n"
                    "Hypothesis to Test:"
                )
            else:
                text = f"Question: {question}\nPlease solve the problem step by step based on the provided media content."
        elif "halo" in dataset_name:
            text = (
                f"Question: {question}\n"
                "Act as a meticulous visual auditor. Before answering, consider the possibility that the object is absent. "
                "Only confirm its presence if you can identify unique, non-generic visual features that distinguish it from the background. "
                "Prioritize precision over recall; avoid any common-sense completions.\n"
                "Please solve the problem step by step based on the provided media content."
            )
        elif dataset_name in ["ragtruth", "halueval", "pubmedqa", "summedits"]:
            passage = example.get('passage', '') or example.get('context', '')
            text = (
                f"Passage: {passage}\n"
                f"Question: {question}\n"
                "Please reason step by step based on the provided passage to arrive at the correct answer."
            )
        else:
            if stage == 'first':
                text = f"Question: {question}\nPlease continue the reasoning."
            else:
                text = f"Question: {question}"

        return media_prefix + text

    def _prepare_chat_template(self, example, system_prompt):
        """MiniCPM  Chat  (Completion Stage)"""
        user_text = self._build_user_text(example, system_prompt, stage='completion')
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]
        

    def _prepare_chat_template_for_first_stage(self, example, system_prompt):
        """MiniCPM  Chat  (First Stage / Foresight)"""
        user_text = self._build_user_text(example, system_prompt, stage='first')
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]
        


def run_ablation_suite(args, decoder):
    """Run multiple ablation configs sequentially, model loaded once."""
    import subprocess, copy

    suite_path = args.ablation_suite
    with open(suite_path, 'r') as f:
        configs = json.load(f)
    print(f"\n{'='*60}")
    print(f"[ABLATION SUITE] Loaded {len(configs)} configs from {suite_path}")

    # Load dataset once
    with open(args.data_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    end_idx = args.max_samples if args.max_samples > 0 else len(test_data)
    test_data = test_data[:end_idx]
    total_samples = len(test_data)

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
                        system_prompt = decoder.get_system_prompt(args.datasets)
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
                                "veto_threshold_used": round(decoder.thresholds['veto_threshold'], 4),
                                "num_paths_explored": _pcs.get("num_paths_explored", 0),
                                "num_paths_vetoed": _n_vetoed,
                            }
                        }
                        # --dump_internals in ablation mode
                        if getattr(args, 'dump_internals', False) and '_dump_internals' in result.get("traj_info", {}):
                            _dump_dir = os.path.join(args.output_dir, "internals", args.datasets.lower())
                            os.makedirs(_dump_dir, exist_ok=True)
                            _sample_id = example.get('id', i)
                            _dump_path = os.path.join(_dump_dir, f"{_sample_id}.jsonl")
                            with open(_dump_path, "w", encoding="utf-8") as f_dump:
                                for _rec in result["traj_info"]['_dump_internals']:
                                    f_dump.write(json.dumps(_rec, ensure_ascii=False) + "\n")
                            del result["traj_info"]['_dump_internals']
                        res_f.write(json.dumps(output_result, ensure_ascii=False) + "\n")
                        res_f.flush()
                        os.fsync(res_f.fileno())

                    except torch.cuda.OutOfMemoryError:
                        import gc
                        torch.cuda.empty_cache(); gc.collect()
                        fail_count += 1
                        fail_res = {"id": i, "sub_category": example.get("sub_category", "unknown"), "status": "OOM", "error": "CUDA OOM"}
                        res_f.write(json.dumps(fail_res, ensure_ascii=False) + "\n")
                        res_f.flush()
                        with open(error_log_path, "a") as ef:
                            ef.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | ID {i} | OOM\n")

                    except Exception as e:
                        import traceback
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
            traceback.print_exc()
            results_summary.append((name, f"CRASHED: {repr(e)}", time.time() - cfg_start))
            continue

        cfg_elapsed = time.time() - cfg_start

        # Auto-eval
        eval_log_path = os.path.join(os.path.dirname(output_dir), "eval.log")
        if os.path.exists(output_path):
            try:
                eval_result = subprocess.run(
                    ["python", "OmniConfess/eval/single_eval.py", "--target_files", output_path],
                    capture_output=True, text=True, timeout=600
                )
                with open(eval_log_path, "w") as ef:
                    ef.write(eval_result.stdout)
                    if eval_result.stderr:
                        ef.write("\n--- stderr ---\n")
                        ef.write(eval_result.stderr)
            except Exception as eval_e:
                pass

        results_summary.append((name, f"ok ({success_count}/{total_samples})", cfg_elapsed))

    # Final summary
    suite_elapsed = time.time() - suite_start
    print(f"\n{'='*60}")
    print(f"[ABLATION SUITE] Complete — {len(configs)} configs, {suite_elapsed:.0f}s total")
    print(f"{'='*60}")
    for name, status, elapsed in results_summary:
        print(f"  {name:20s} | {status:30s} | {elapsed:.0f}s")


def main():
    """Main entry point for OmniConfess decoding."""
    args = parse_arguments()
    decoder = PhiDecoder(args)

    # Log active tuning flags
    if getattr(args, 'bypass_path', False):
        pass
    if getattr(args, 'dump_internals', False):
        pass
    if getattr(args, 'use_base_prompt', False):
        pass

    if args.ablation_suite:
        run_ablation_suite(args, decoder)
        return

    with open(args.data_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
        
    start_idx = 0
    end_idx = args.max_samples if args.max_samples > 0 else len(test_data)
    test_data = test_data[start_idx : end_idx]
    print(f" : {start_idx} ~ {end_idx} {len(test_data)} ")
        
    total_samples = len(test_data)
    print(f" : {total_samples}")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.time_path, exist_ok=True)
    
    output_path = os.path.join(args.output_dir, f"{args.file_name}.json")
    traj_path_jsonl = os.path.join(args.time_path, f"TRAJ_INFO-{args.file_name}.jsonl")
    error_log_path = os.path.join(args.output_dir, f"ERROR-{args.file_name}.log")

    processed_count = 0
    print(f"Output: {output_path}")
    
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            processed_count = sum(1 for line in f if line.strip())
        
        if processed_count >= total_samples:
            print(f"Already complete ({processed_count}/{total_samples}).")
            return
        else:
            print(f"Resuming from {processed_count}/{total_samples}.")
    else:
        pass

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
            # P3-c: Pre-sample cache clear to prevent OOM accumulation (esp. PubMedQA)
            import gc; gc.collect()
            torch.cuda.empty_cache()
            try:
                sub_cat = example.get("sub_category", "unknown")
                system_prompt = decoder.get_system_prompt(args.datasets)
                
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
                        "veto_threshold_used": round(decoder.thresholds['veto_threshold'], 4),
                        "num_paths_explored": _pcs.get("num_paths_explored", 0),
                        "num_paths_vetoed": _n_vetoed,
                        "bypass_path": getattr(args, 'bypass_path', False),
                        "use_base_prompt": getattr(args, 'use_base_prompt', False),
                    }
                }
                res_f.write(json.dumps(output_result, ensure_ascii=False) + "\n")
                res_f.flush()
                os.fsync(res_f.fileno())

                if args.record_process and "traj_info" in result:
                    result["traj_info"]["question_idx"] = i
                    # --dump_internals: write per-sample internals to separate file
                    if getattr(args, 'dump_internals', False) and '_dump_internals' in result["traj_info"]:
                        _dump_dir = os.path.join(args.output_dir, "internals", args.datasets.lower())
                        os.makedirs(_dump_dir, exist_ok=True)
                        _sample_id = example.get('id', i)
                        _dump_path = os.path.join(_dump_dir, f"{_sample_id}.jsonl")
                        with open(_dump_path, "w", encoding="utf-8") as f_dump:
                            for _rec in result["traj_info"]['_dump_internals']:
                                f_dump.write(json.dumps(_rec, ensure_ascii=False) + "\n")
                        del result["traj_info"]['_dump_internals']  # don't bloat traj file
                    with open(traj_path_jsonl, "a", encoding="utf-8") as f_traj:
                        f_traj.write(json.dumps(result["traj_info"], ensure_ascii=False) + "\n")


            except torch.cuda.OutOfMemoryError as oom_e:
                import gc
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
                traceback.print_exc()

                total_stats["fail_count"] += 1
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

    time_span = time.time() - start_time
    time_info_path = os.path.join(args.time_path, f"{args.file_name}.txt")
    with open(time_info_path, "w", encoding="utf-8") as f_summary:
        f_summary.write(f'time:  {time_span}\n')
        f_summary.write(f'total_samples: {total_samples}\n')
        f_summary.write(f'success: {total_stats["success_count"]}\n')
        f_summary.write(f'all_input_tokens: {total_stats["input_tokens"]}\n')
        f_summary.write(f'all_output_tokens: {total_stats["output_tokens"]}\n')

    print(f"\n 。: {total_stats['success_count']} | : {total_stats['fail_count']}")
    print(f": {time_info_path}")

if __name__ == "__main__":
    main()