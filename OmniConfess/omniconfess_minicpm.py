# -*- coding: utf-8 -*-
# ==============================================================================
# OmniConfess -- MiniCPM-o-2.6 backbone
#
# Port of the Qwen2.5-Omni implementation (see omniconfess_qwen.py) to
# MiniCPM-o-2.6. The audit/repair logic is unchanged; the differences are in
# the model plumbing:
#   model.thinker -> model.llm, a different processor API, a different
#   embedding pipeline, and backbone-specific anchor layer indices
#   (MiniCPM-o-2.6 has 28 decoder layers).
#
# Example:
#   python omniconfess_minicpm.py \
#       --datasets cmm \
#       --data_path ./OmniHalluBench/cmm/cmm.json \
#       --data_root ./OmniHalluBench/cmm/ \
#       --output_dir ./outputs \
#       --file_name cmm_run
# ==============================================================================
# -*- coding: utf-8 -*-
import time
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import torch
import json
import os
import argparse
import cv2        # 导入 OpenCV 用于视频处理
import librosa    # 导入 librosa 用于音频处理
from PIL import Image  #导入 PIL 用于处理纯图像
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

# 注意：移除了 vLLM，因为 Qwen-Omni 的双分支架构更适合使用 transformers 原生调用进行受控生成

# 常量配置
INF = 10
TEMPERATURE = 0.6
# 设定显存管理策略，防止多模态长序列导致 OOM
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# ================= 1. 多模态加载工具函数 =================

_BOXED_RE = re.compile(r'^\s*\\?boxed\s*\{(.*?)\}?\s*$', re.S)


def unwrap_boxed(s):
    """剥掉 `\\boxed{...}` 包装（含无闭合括号的那种）。

    eval 的 Token-F1 删标点时不补空格，所以 `\\boxed{no}` 会退化成单词 `boxedno`
    —— 多一个垃圾词且真答案词被吃掉，该条直接 0 分。haloquest 全量实测
    13.18 → 16.82（+3.65，218 条变好 0 条变差）。旧写法 `lstrip('boxed{')`
    从来没工作过：首字符 `\\` 不在字符集里。见 FINDINGS_0804.md §9.1。
    """
    t = str(s).strip()
    m = _BOXED_RE.match(t)
    return m.group(1).strip() if m else t

def load_image(image_path):
    """🌟 新增：加载单张图像数据"""
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
        # 🌟 在这里直接做物理 Resize，设为 224x224
        frame = cv2.resize(frame, (224, 224))
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.array(frames) if frames else None
    
def load_audio_data(audio_path, sr=24000):
    """加载音频数据"""
    if not audio_path or not os.path.exists(audio_path):
        return None
    try:
        audio, _ = librosa.load(audio_path, sr=sr)
        return audio if (audio is not None and len(audio) > 0) else None
    except:
        return None


# ================= Token-Guard 防御插件 =================
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
                # MiniCPM 的 LLM 分支特定层归一化与输出
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
    计算幻觉惩罚分，并实时输出触发详情。
    纯层间信号：res_20_fin, res_14_20, l20_conf（不依赖 MDI）。
    thresholds: dict with calibratable threshold values (or None for defaults).
    """
    if thresholds is None:
        thresholds = _DEFAULT_THRESHOLDS

    token_id = report['token_id']
    token_str = tokenizer.decode([token_id]).lower().strip()

    # 1. 基础过滤
    clean_word = ''.join(e for e in token_str if e.isalpha())
    if len(clean_word) <= 1 or clean_word in STOP_WORDS:
        return 0.0

    penalty = 0.0
    triggered = []

    # 2. 追踪晚期冲突 (L22 vs Final) —— 模型在最后时刻"突然改口"
    if report['res_20_fin'] > thresholds['res_20_fin']:
        p_inc = 0.5
        # 加重条件：L22 置信度极高却被 final 推翻 → 强行覆盖
        if report['l20_conf'] > thresholds['l20_conf_late']:
            p_inc += 0.5
            triggered.append('LATE+CONF')
        else:
            triggered.append('LATE')

        penalty += p_inc
        print(f"🚨 [AUDIT: LATE] Token: '{token_str}' | Res: {report['res_20_fin']:.2f} | L22_Conf: {report['l20_conf']:.2f} | Penalty: +{p_inc}", flush=True)

    # 3. 追踪早期劫持 (L16 vs L22) —— 语言模型套话覆盖了感官/上下文
    if report['res_14_20'] > thresholds['res_14_20'] and report['l20_conf'] > thresholds['l20_conf_hijack']:
        p_inc = 0.6
        penalty += p_inc
        triggered.append('HIJACK')
        print(f"🚨 [AUDIT: HIJACK] Token: '{token_str}' | Res: {report['res_14_20']:.2f} | Conf: {report['l20_conf']:.2f} | Penalty: +0.6", flush=True)

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
    MiniCPM-o-2.6 版：使用 get_vllm_embedding + get_omni_embedding 获取多模态 embeddings，
    然后调用 model.llm.generate 生成。
    """
    print(f"\n    [GEN_DEBUG] ---> 进入 safe_model_generate (MiniCPM)...", flush=True)
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
        print(f"    [GEN_DEBUG] Embeddings shape: {embeds.shape}", flush=True)
    except Exception as e:
        print(f"    [GEN_DEBUG] Embedding extraction failed, fallback to text-only: {repr(e)}", flush=True)
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
        print(f"    [GEN_DEBUG] llm.generate succeeded.", flush=True)
    except Exception as e:
        print(f"    [GEN_DEBUG] llm.generate crashed: {repr(e)}", flush=True)
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
        print(f"    [GEN_DEBUG] Sequence stitching failed: {repr(e)}", flush=True)
        traceback.print_exc()
        raise e

    print(f"    [GEN_DEBUG] <--- safe_model_generate done.\n", flush=True)
    return out

def softmax(x):
    e_x = np.exp(np.array(x) - np.max(x)) # 增加数值稳定性
    return e_x / e_x.sum(axis=0)


def get_modality_indices(processor, input_ids, device, image_bound=None, audio_bounds=None):
    """从 MiniCPM processor 输出的 image_bound/audio_bounds 获取模态索引。

    MiniCPM 的 processor 直接返回 image_bound 和 audio_bounds 张量,
    格式为 [[start, end], ...], 表示每个模态区间的 token 范围。
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
    """计算当前 Logits 的信息熵，用于观察模型纠结程度"""
    probs = torch.softmax(logits, dim=-1)
    return -torch.sum(probs * torch.log(probs + 1e-9), dim=-1).item()


def safe_thinker_step_audit(model, processor, inputs, past_key_values=None, position_ids=None, **gen_kwargs):
    """
    极简 Forward 版：只负责单步前向计算，不涉及复杂的 generate 逻辑。
    """
    valid_keys = ['input_ids', 'attention_mask', 'pixel_values', 'tgt_sizes',
                  'image_bound', 'audio_features', 'audio_feature_lens', 'audio_bounds']
    clean_inputs = {k: v for k, v in inputs.items() if k in valid_keys}
    
    for k, v in clean_inputs.items():
        if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
            clean_inputs[k] = v.to(torch.bfloat16)

    # 🌟 调用 forward 而不是 generate
    outputs = model.llm(
        **clean_inputs,
        past_key_values=past_key_values,
        position_ids=position_ids,
        use_cache=True,
        return_dict=True
    )
    
    # 拿到最后一个 Token 的 Logits
    next_logits = outputs.logits[:, -1, :]
    updated_cache = outputs.past_key_values
    
    return next_logits, updated_cache

def parse_arguments():
    """Parse command line arguments specifically for MiniCPM-o-2.6 GUARD"""
    parser = argparse.ArgumentParser(description="GUARD Algorithm for MiniCPM-o-2.6")

    # Model configuration
    # MiniCPM-o-2.6
    parser.add_argument('--model_id', type=str, default='minicpm-o-2.6',
                        help='Model identifier')
    parser.add_argument('--model_path', type=str, default='openbmb/MiniCPM-o-2_6',
                        help='Model path')
    parser.add_argument('--gpus', type=int, default=2,
                        help='Number of GPUs to use (tensor_parallel is handled by device_map in transformers)')

    # Data configuration
    # 增加多模态数据集支持 (cmm, mmhalu 等)
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

    # Algorithm parameters (保持 Phi-Decoding 核心参数不变)
    parser.add_argument('--step_beam_size', type=int, default=2, # 多模态显存占用高，建议调小 beam_size
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
    parser.add_argument('--cluster_num', type=int, default=2, # 通常设为 2 以符合 Phi-Decoding 论文
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
                        help='Replace GUARD prompts with base v0 prompts everywhere (system, user, '
                             'final generation). GUARD pipeline flow is unchanged. For Exp-X2 factorial.')

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
        print(f"[WARN] Unknown threshold keys ignored: {unknown}", flush=True)
    for k, v in overrides.items():
        if k in thresholds:
            thresholds[k] = float(v)
    print(f"[THRESHOLD] Using overrides: {overrides}", flush=True)
    print(f"[THRESHOLD] Effective thresholds: {thresholds}", flush=True)
    return thresholds


def softmax(x):
    """
    Compute softmax values for the input array with numerical stability.
    """
    x = np.array(x)
    # 减去最大值防止 exp(x) 溢出 (Numerical Stability)
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
        self.processor = None  # 增加处理器
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

        # 1. 加载 MiniCPM-o-2.6 模型
        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            attn_implementation='sdpa',
            torch_dtype=torch.bfloat16,
            init_vision=True,
            init_audio=True,
            init_tts=False,
        ).eval().cuda()

        # 2. 加载 Tokenizer 和 Processor
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        # MiniCPM stores its own processor internally
        self.processor = self.model.processor

        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"[MiniCPM] Loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)
        print(f"[MiniCPM] LLM layers: {len(self.model.llm.model.layers)}", flush=True)

        # 设置随机种子
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
        结构化系统提示词 (MiniCPM-o-2.6 版)。
        支持包含匹配、针对 CMM 二元验证任务、针对 AVH 单句描述、强化终止符和格式约束。
        """
        # --use_base_prompt: return base v0 system prompt instead of GUARD prompt
        if getattr(self.args, 'use_base_prompt', False):
            return _base_v0_system_for(dataset_type)

        omni_identity = "You are an expert multimodal assistant capable of perceiving auditory and visual inputs."
        
        # --- 1. 针对 CMM (Yes/No) 数据集的深度优化 ---
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

        # --- 2. 🌟 新增：针对 AVH 数据集的极致单句描述优化 ---
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

        # --- 5. 纯文本抽取/问答任务 ---
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

        # 默认回退
        return f"{omni_identity}\nSolve the problem step by step based on the provided content."

    def cluster_responses(self, responses, advantages):
        """
        Cluster responses using TF-IDF and K-means.
        Identify the 'consensus' reasoning paths.
        """
        # 过滤空响应
        valid_indices = [i for i, r in enumerate(responses) if r.strip()]
        if len(valid_indices) < self.args.cluster_num:
            return None, {"state": "cannot cluster", "reason": "too few valid responses"}

        try:
            valid_responses = [responses[i] for i in valid_indices]

            # 文本向量化
            X = self._vectorize_texts(valid_responses)

            # 执行聚类
            # 这里的 n_clusters 使用参数配置，通常为 2
            k = min(self.args.cluster_num, len(valid_responses))
            kmeans = KMeans(n_clusters=k, n_init=10, random_state=self.args.seed)
            kmeans.fit(X)

            # 按簇对响应索引进行分组
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
        🌟 优化点 2：动态聚类权重 (Trust-Weighted Clustering)
        结合簇的"共识大小"和"内部平均优势(置信度)"，打破多数派幻觉。
        """
        if self.args.strategy == "cluster":
            # 过滤有效索引
            valid_indices = [idx for idx, r in enumerate(responses) if r.strip()]

            # 情况 A: 全部响应为空
            if len(valid_indices) == 0:
                print('Warning: All final responses are empty. Sampling by negative advantage.')
                weights = softmax([-adv/TEMPERATURE for adv in advantages])
                return np.random.choice(len(advantages), p=weights)

            # 情况 B: 有效响应太少，无法进行有意义的聚类
            if len(valid_indices) < self.args.cluster_num:
                print('Warning: Too few valid responses for clustering. Sampling by advantage.')
                weights = softmax([advantages[i]/TEMPERATURE for i in valid_indices])
                return np.random.choice(valid_indices, p=weights)

            try:
                # 准备聚类数据
                valid_responses = [responses[i] for i in valid_indices]
                valid_advantages = [advantages[i] for i in valid_indices]

                # 向量化与 KMeans
                X = self._vectorize_texts(valid_responses)
                
                k = self.args.cluster_num
                kmeans = KMeans(n_clusters=k, n_init=10, random_state=self.args.seed)
                kmeans.fit(X)
                cluster_labels = kmeans.labels_

                # 分组
                cluster_list = [[] for _ in range(k)]
                for idx, label in enumerate(cluster_labels):
                    cluster_list[label].append(idx)
                
                # 过滤掉偶然产生的空簇
                cluster_list = [c for c in cluster_list if len(c) > 0]
                
                # ==========================================================
                # 🌟 核心修改：Cluster_Score 动态权重评估
                # ==========================================================
                ALPHA = 0.5  # 权重因子：0.5 表示规模和质量各占 50%
                total_valid = len(valid_indices)
                
                # 1. 计算 Size_Ratio
                cluster_size_ratios = [len(c) / total_valid for c in cluster_list]
                
                # 2. 计算 Avg_Advantage 并归一化 (Softmax 映射到 0~1 区间)
                cluster_avg_advs = [np.mean([valid_advantages[idx] for idx in c]) for c in cluster_list]
                cluster_adv_scores = softmax([adv / TEMPERATURE for adv in cluster_avg_advs])
                
                # 3. 融合得分
                cluster_scores = []
                for i in range(len(cluster_list)):
                    score = ALPHA * cluster_size_ratios[i] + (1.0 - ALPHA) * cluster_adv_scores[i]
                    cluster_scores.append(score)
                    
                # 4. 选出综合得分最高的簇
                best_cluster_idx = int(np.argmax(cluster_scores))
                target_cluster = cluster_list[best_cluster_idx]
                
                print(f"    [CLUSTER DEBUG] 簇分布: {[len(c) for c in cluster_list]}, "
                      f"规模分: {[round(s,3) for s in cluster_size_ratios]}, "
                      f"质量分: {[round(s,3) for s in cluster_adv_scores]} "
                      f"-> 选中簇 {best_cluster_idx}", flush=True)

                # 在选中的优质簇内，根据 Advantage 进行加权采样
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
        针对多模态数据优化的 Phi-Decoding 流水线控制器。
        修复：
        1. 路径拼接逻辑：解决 has_video/audio 为 false 的物理路径问题。
        2. 实时日志：在 nohup 中明确打印媒体加载状态。
        3. 轨迹挂载：确保审计数据 (panoramic_audit) 完整进入 traj_info。
        """
        import os

        # 1. 初始化跟踪变量与统计
        token_stats = {"input": 0, "output": 0}
        rollout_stats = {"total": 0, "saved": 0}

        # 初始化核心轨迹池
        traj_pool = [[] for _ in range(self.args.num_foresight)]
        step_pool = [[] for _ in range(self.args.num_foresight)]
        prob_pool = [[] for _ in range(self.args.num_foresight + 1)]
        adv_pool = [[] for _ in range(self.args.num_foresight + 1)]

        # 🌟 2. 增强型多模态数据加载 (新增纯图像支持)
        # 🌟 2. 增强型多模态数据加载 (新增纯图像支持)
        v_rel = example.get('video') or example.get('video_path') or ""
        a_rel = example.get('audio') or example.get('audio_path') or ""
        i_rel = example.get('image') or example.get('image_path') or "" # 👈 新增图像提取
        
        # ==========================================================
        # 🎯 核心修复：智能路径解析器 (Smart Path Resolver)
        # ==========================================================
        json_dir = os.path.dirname(self.args.data_path) # 直接获取 JSON 所在的真实目录 (如 .../dataset/PhD)
        
        def resolve_media_path(rel_path):
            if not rel_path: return ""
            clean_rel = rel_path.lstrip('./')
            
            # 策略 1: 优先基于 JSON 文件所在的同级目录拼接 (完美解决 PhD 数据集丢路径问题)
            path1 = os.path.abspath(os.path.join(json_dir, clean_rel))
            if os.path.exists(path1): return path1
            
            # 策略 2: 基于 Bash 传进来的全局 data_root 拼接 (兼容 CMM 等旧逻辑)
            path2 = os.path.abspath(os.path.join(self.args.data_root, clean_rel))
            if os.path.exists(path2): return path2
            
            # 策略 3: 终极容错，强制塞入 PhD 查找
            path3 = os.path.abspath(os.path.join(self.args.data_root, "PhD", clean_rel))
            if os.path.exists(path3): return path3
            
            return path1 # 都找不到的话，返回策略 1 的路径，交由后续的 train/val 替换逻辑处理

        # 重新应用解析器获取绝对路径
        v_path = resolve_media_path(v_rel)
        a_path = resolve_media_path(a_rel)
        i_path = resolve_media_path(i_rel)
        # ==========================================================

        # 下面继续保留原本的加载逻辑
        pixel_values = None
        if v_rel and os.path.exists(v_path):
            pixel_values = load_video_frames(v_path, num_frames=8)
            print(f"✅ [MEDIA LOAD] 视频加载成功: {v_path}", flush=True)
            
        audio_values = None
        if a_rel and os.path.exists(a_path):
            audio_values = load_audio_data(a_path)
            print(f"✅ [MEDIA LOAD] 音频加载成功: {a_path}", flush=True)

        # 👈 新增：加载纯图像
        # 👈 新增：加载纯图像
        image_values = None
        if i_rel:
            # 🌟 核心修复：如果找不到图片，自动在 train2014 和 val2014 之间切换查找！
            if not os.path.exists(i_path):
                if 'val2014' in i_path:
                    i_path = i_path.replace('val2014', 'train2014')
                elif 'train2014' in i_path:
                    i_path = i_path.replace('train2014', 'val2014')

            if os.path.exists(i_path):
                image_values = load_image(i_path)
                print(f"✅ [MEDIA LOAD] 图像加载成功: {i_path}", flush=True)
            else:
                print(f"❌ [MEDIA ERROR] 图像路径无效: {i_path}", flush=True)

        example['pixel_values'] = pixel_values
        example['audio_values'] = audio_values
        example['image_values'] = image_values # 👈 注入图像张量
        
        pixel_values = None
        if v_rel:
            if os.path.exists(v_path):
                pixel_values = load_video_frames(v_path, num_frames=8)
                print(f"✅ [MEDIA LOAD] 视频加载成功: {v_path}", flush=True)
            else:
                print(f"❌ [MEDIA ERROR] 视频路径无效: {v_path}", flush=True)
                
        audio_values = None
        if a_rel:
            if os.path.exists(a_path):
                audio_values = load_audio_data(a_path)
                print(f"✅ [MEDIA LOAD] 音频加载成功: {a_path}", flush=True)
            else:
                print(f"❌ [MEDIA ERROR] 音频路径无效: {a_path}", flush=True)

        # 注入张量供后续 _process_step 使用
        example['pixel_values'] = pixel_values
        example['audio_values'] = audio_values

        # 3. 初始化 Beam 状态
        # 改进 C：avhbench 专用音视频分步 foresight 初始化
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

        # 4. 初始化轨迹信息 (增强 media_info 字段)
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

        # 5. Multi-step Foresight 迭代 (思考过程)
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

        # 6. 生成最终响应 (下结论阶段)
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

        # 7. 最终数据打包
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
        🌟 终极修复版：解决 4D Attention Mask 污染与 3D RoPE 位置编码更新问题
        """
        stop_foresight = False
        pixel_values = example.get('pixel_values')
        audio_values = example.get('audio_values')
        image_values = example.get('image_values')  # 👈 直接加在这里！

        all_responses_first_stage = []
        all_logprobs_first_stage = []
        all_advantages_first_stage = []
        all_token_nums_first_stage = []
        
        _need_hidden = getattr(self.args, 'layer_fix_mode', 'linear') != 'linear'
        auditor = DualLayerAuditor(self.model, self.processor, store_hidden=_need_hidden)
        auditor.register()
        _all_dump_records = []  # for --dump_internals

        for beam_idx in range(self.args.step_beam_size):
            print(f"\n🔍 [DEBUG] ================= Beam {beam_idx+1} =================", flush=True)
            
            chat = self._prepare_chat_template_for_first_stage(example, system_prompt)
            p_str = self.processor.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            if isinstance(p_str, list): p_str = p_str[0]
            
            inputs_text = p_str.replace(self.tokenizer.eos_token, "").strip() + previous_steps[beam_idx]
            
            m_inputs = self._call_minicpm_processor(inputs_text, image_values=image_values, audio_values=audio_values)

            token_stats["input"] += m_inputs.input_ids.numel()

            # =====================================================================
            # 🚀 MiniCPM 特征提取：直接使用 get_vllm_embedding + get_omni_embedding
            # =====================================================================
            inputs_embeds_f = _get_minicpm_embeds(self.model, m_inputs)
            print(f"✅ [DEBUG] MiniCPM embeddings obtained, shape: {inputs_embeds_f.shape}", flush=True)

            attn_mask_f = m_inputs['attention_mask'].clone() if isinstance(m_inputs, dict) else m_inputs.attention_mask.clone()
            pos_ids_f = m_inputs.get('position_ids', None) if isinstance(m_inputs, dict) else getattr(m_inputs, 'position_ids', None)

            # =====================================================================
            # 🚀 审计线屏蔽 — 使用 MiniCPM 的 image_bound / audio_bounds
            # =====================================================================
            m_nv = attn_mask_f.clone()
            _img_bound = m_inputs.get('image_bound', None) if isinstance(m_inputs, dict) else getattr(m_inputs, 'image_bound', None)
            _aud_bounds = m_inputs.get('audio_bounds', None) if isinstance(m_inputs, dict) else getattr(m_inputs, 'audio_bounds', None)
            v_idx, a_idx = get_modality_indices(self.processor, m_inputs['input_ids'] if isinstance(m_inputs, dict) else m_inputs.input_ids, self.model.llm.model.embed_tokens.weight.device, image_bound=_img_bound, audio_bounds=_aud_bounds)
            if v_idx is not None:
                m_nv[:, v_idx] = 0
            if a_idx is not None:
                m_nv[:, a_idx] = 0  # 🌟 修复：必须把音频也屏蔽掉，才能审计音频幻觉！
            is_text_only = (v_idx is None and a_idx is None)  # P1: skip CD-Confess for text-only

            # P7: Passage-MDI — 纯文本数据集用 passage mask 替代 modality mask
            if is_text_only and not getattr(self.args, 'disable_passage_mdi', False) \
                    and self.args.datasets.lower() in ('halueval', 'pubmedqa', 'ragtruth', 'summedits'):
                passage_idx = get_passage_indices(m_inputs.input_ids, self.tokenizer, self.model.device)
                if passage_idx is not None and len(passage_idx) > 0:
                    m_nv[:, passage_idx] = 0
                    is_text_only = False  # 让 CD-Confess 和 Layer-Confess 正常运行
                    print(f"    📖 [Passage-MDI] 屏蔽 {len(passage_idx)} 个 passage token (idx {passage_idx[0].item()}–{passage_idx[-1].item()})", flush=True)

            # ================================================================
            # avhbench 标记（用于最终生成分支判断）
            # ================================================================
            is_avhbench = "avh" in self.args.datasets.lower()

            # 开始 Rollout
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

                        # 1. 获取带有感官 (Full) 和剥夺感官 (No-Vision/Audio) 的概率
                        # 1. 获取两组概率分布
                        # =====================================================================
                        # 🚀 终极提升版：CD-Confess (对比解码 + 动态自省) - 修复变量冲突版
                        # =====================================================================
                        # 1. 预采样一个词来探测"自省"信号 (先不修改原始 logits，只做探测)
                        temp_probs = torch.softmax(logits_f / TEMPERATURE, dim=-1)
                        token_id = torch.multinomial(temp_probs, num_samples=1).item()
                        
                        # 🌟 修复点：改用 prob_f_val 和 prob_nv_val，绝不污染 p_full 和 p_nv (KV-Cache)
                        prob_f_val = temp_probs[0, token_id].item()
                        prob_nv_val = torch.softmax(out_nv.logits[:, -1, :] / TEMPERATURE, dim=-1)[0, token_id].item()
                        mdi_av = (prob_f_val - prob_nv_val) / (prob_f_val + 1e-9)
                        
                        token_str = self.tokenizer.decode([token_id]).lower().strip()
                        clean_word = ''.join(e for e in token_str if e.isalpha())

                        # 🌟 触发条件：MDI 极低（幻觉嫌疑）且不是短词
                        # P1: skip for text-only; P2: skip EOS and \boxed tokens
                        _mdi_thresh = self.thresholds['mdi_threshold']
                        if (mdi_av < _mdi_thresh and len(clean_word) > 3 and clean_word not in STOP_WORDS
                                and not is_text_only
                                and token_id != self.tokenizer.eos_token_id
                                and 'boxed' not in clean_word):
                            print(f"    💡 [Self-Confess] 发现先验陷阱: '{token_str}' (MDI: {mdi_av:.2f}, thresh: {_mdi_thresh})，启动【对比探索模式】", flush=True)

                            # 【黑科技公式】：强行放大那些"只有看了图/听了音才产生的特征"，压制"瞎编特征"
                            alpha = self.thresholds['cd_alpha']
                            blind_logits = out_nv.logits[:, -1, :]
                            refined_logits = logits_f + alpha * (logits_f - blind_logits)
                            
                            # 在重塑后的概率分布中重新采样
                            probs_f = torch.softmax(refined_logits / TEMPERATURE, dim=-1)
                            
                            # 为了安全，把刚才那个探测出来的幻觉词彻底屏蔽掉
                            if getattr(self.args, 'cd_mode', 'hard') == 'soft':
                                _sf = getattr(self.args, 'cd_soft_factor', 0.1)
                                probs_f[0, token_id] *= _sf
                                print(f"    🔧 [CD-Mode=soft] token '{token_str}' scaled ×{_sf}", flush=True)
                            else:
                                probs_f[0, token_id] = 0.0
                            probs_f = probs_f / probs_f.sum(dim=-1, keepdim=True)
                            
                            # 进行二次采样 (真正的决定)
                            next_id = torch.multinomial(probs_f, num_samples=1)
                            token_id = next_id.item()
                            
                            # 🌟 [必须更新] 更新概率值，供后续的 Penalty 审计和 Logprob 累加使用
                            pf_val = torch.softmax(logits_f, dim=-1)[0, token_id].item()
                            pnv_val = torch.softmax(out_nv.logits[:, -1, :] / TEMPERATURE, dim=-1)[0, token_id].item()  # P0: fix temperature mismatch
                            mdi_av = (pf_val - pnv_val) / (pf_val + 1e-9)
                            print(f"    🔄 [CD-Correction] 探索出更具感官依据的词: '{self.tokenizer.decode([token_id]).strip()}'", flush=True)
                            
                        else:
                            # 正常情况：直接沿用第一次探测的采样结果
                            next_id = torch.tensor([[token_id]], device=self.model.device)
                            token_id = next_id.item()
                            pf_val = prob_f_val    # 直接使用第一轮获取的概率
                            pnv_val = prob_nv_val  # 同步一下盲态概率
                        # 4. 继续原来的逻辑
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
                        
                        # 更新 Attention Mask (正常添加 2D 维度的 1)
                        new_bit = torch.ones((1, 1), device=self.model.device, dtype=m_f_step.dtype)
                        m_f_step = torch.cat([m_f_step, new_bit], dim=-1)
                        m_nv_step = torch.cat([m_nv_step, new_bit], dim=-1)
                        
                        # 🌟 核心修复 2：动态推进 3D 位置编码 (兼容 mRoPE)
                        if pos_ids_step is not None:
                            if i == 0:
                                # 截取最后一个位置并加 1 (使用 ... 支持 1D 和 3D 的自适应切片)
                                pos_ids_step = pos_ids_step[..., -1:] + 1
                            else:
                                pos_ids_step = pos_ids_step + 1

                    response = self.tokenizer.decode(generated_ids_list, skip_special_tokens=True).strip()
                    logprob = acc_logprob / (len(generated_ids_list) + 1e-9)
                    
                    # 🌟 优化点 1：一票否决制 (Lethal Penalty)
                    _pcs = traj_info['path_confess_stats']
                    _pcs['num_paths_explored'] += 1
                    _pcs['max_penalty_seen'] = max(_pcs['max_penalty_seen'], float(total_hallu_penalty))
                    _veto_thresh = self.thresholds['veto_threshold']
                    if getattr(self.args, 'bypass_path', False):
                        # --bypass_path: NO veto, use penalty as soft disadvantage
                        advantage = logprob - (total_hallu_penalty * 0.8)
                        print(f"    🔓 [BYPASS_PATH] penalty={total_hallu_penalty:.2f}, advantage={advantage:.4f} (no veto)", flush=True)
                    elif total_hallu_penalty >= _veto_thresh:
                        # 赋值极小值，而非 -inf，防止后续 np.mean() 崩溃返回 NaN
                        advantage = -10000.0
                        logprob = -10000.0
                        _pcs['num_paths_vetoed'] += 1
                        print(f"    🚨 [LETHAL PENALTY] 触发一票否决! 幻觉分数: {total_hallu_penalty:.2f} (thresh: {_veto_thresh})。路径被强行抹杀。", flush=True)
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

        # --- Stage 2: 宽度剪枝 (Width Pruning) ---
        if self.args.width_pruning_strategy != "none" and self.args.width_pruning_strategy != "":
            keep_foresight_list = []
            
            # 🌟 在剪枝前，物理隔离被"一票否决"的毒瘤路径，防止它们拉低全局均值
            safe_indices = [i for i, logp in enumerate(all_logprobs_first_stage) if logp > -9000.0]
            
            if self.args.width_pruning_strategy == "low_sigma" and len(safe_indices) > 0:
                safe_logprobs = [all_logprobs_first_stage[i] for i in safe_indices]
                mean = np.mean(safe_logprobs)
                std = np.std(safe_logprobs)
                for idx in safe_indices:
                    # 只在安全候选者中进行 Sigma 剪枝
                    if all_logprobs_first_stage[idx] > mean - self.args.sigma_rate * std:
                        keep_foresight_list.append(idx)
            else:
                # 如果全部被否决（极端幻觉），或者不使用 low_sigma，保留未被彻底判死刑的
                keep_foresight_list = safe_indices.copy()

            if len(keep_foresight_list) < self.args.step_beam_size:
                print(f"⚠️ 剪枝后仅剩 {len(keep_foresight_list)} 条，准备补充安全备选...", flush=True)
                
                # 1. 找出还没被选入，且【绝对安全】（没有被一票否决）的路径索引
                safe_available = [
                    i for i in range(len(all_logprobs_first_stage)) 
                    if i not in keep_foresight_list and all_logprobs_first_stage[i] > -9000.0
                ]
                
                # 2. 如果还有安全的备选可以加
                if safe_available:
                    num_to_add = min(self.args.step_beam_size - len(keep_foresight_list), len(safe_available))
                    
                    # 🌟 核心修复：不掷骰子了，直接按 Logprob 降序排列，取最好的前几个
                    safe_available.sort(key=lambda x: all_logprobs_first_stage[x], reverse=True)
                    additional_indices = safe_available[:num_to_add]
                    
                    keep_foresight_list.extend(additional_indices)
                    print(f"    ➕ 已按优劣顺序补充 {num_to_add} 条安全路径。", flush=True)
                else:
                    print("    ⚠️ 没有多余的安全路径可供补充，Beam Size 将临时缩小。", flush=True)
                    traj_info['path_confess_stats']['beam_shrink_occurred'] = True

            keep_foresight_list.sort()
            rollout_stats["saved"] += (len(all_logprobs_first_stage) - len(keep_foresight_list))

            all_responses = [all_responses_first_stage[i] for i in keep_foresight_list]
            filtered_logprobs = [all_logprobs_first_stage[i] for i in keep_foresight_list]
            filtered_advantages = [all_advantages_first_stage[i] for i in keep_foresight_list]

        # --- Stage 3: 路径补全 (Completion Stage) ---
        # --- Stage 3: 路径补全 (Completion Stage) ---
        completed_responses = []
        completed_logprobs = []
        completed_advantages = []

        print(f"🔄 正在对筛选出的 {len(keep_foresight_list)} 个候选路径进行深度补全...", flush=True)

        for idx in range(len(keep_foresight_list)):
            response = all_responses[idx]
            beam_idx = keep_foresight_list[idx] // self.args.num_rollout
            
            chat = self._prepare_chat_template(example, system_prompt)
            p_str = self.processor.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            if isinstance(p_str, list): p_str = p_str[0]
            
            # 过滤掉模型在第一阶段可能误生成的特征词
            clean_resp = response.replace("<|video|>", "").replace("<|audio|>", "").replace("<|vision_start|>", "").replace("<|vision_end|>", "")
            inputs_text = p_str.replace(self.tokenizer.eos_token, "").strip() + previous_steps[beam_idx] + clean_resp

            m_inputs = self._call_minicpm_processor(inputs_text, image_values=image_values, audio_values=audio_values)

            token_stats["input"] += m_inputs.input_ids.numel()
            input_len = m_inputs.input_ids.shape[1]

            with torch.inference_mode():
                # 🌟 注意：这里必须是你之前修改好的 safe_model_generate
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

        # --- Stage 4: 聚类选择与早停检查 (Clustering & Final Selection) ---
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

            # --- 6. 结构化全景轨迹信息 (Trajectory Recording) ---
            step_info = {
                'first_stage': {
                    # 完整保留所有候选文本和对应的优势得分，方便对比被惩罚前后的差异
                    'candidates': [
                        {
                            'text': text,
                            # 🌟 增加 float() 强转，防止 NumPy 浮点数导致 JSON 崩溃
                            'logprob': round(float(all_logprobs_first_stage[i]), 4),
                            'advantage': round(float(all_advantages_first_stage[i]), 4),
                            # 🌟 增加 bool() 强转，解决 NumPy 布尔值无法序列化的 Bug
                            'is_vetoed': bool(all_advantages_first_stage[i] <= -9000.0)
                        } for i, text in enumerate(all_responses_first_stage)
                    ]
                },
                'width_pruning_survivors': keep_foresight_list,
                'second_stage': {
                    'completions': completed_responses,
                    # 🌟 同样加上 float() 强转
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
            print(f'⚠️ 聚类阶段异常: {e}。退回到纯优势采样。', flush=True)
            traj_info['path_confess_stats']['cluster_fallback_occurred'] = True
            # BUG FIX: completed_advantages / keep_foresight_list may be empty when all beams
            # produced empty responses; fall back entirely to first-stage results in that case.
            if completed_advantages and keep_foresight_list:
                adv_source = completed_advantages
                resp_source = completed_responses
                logp_source = completed_logprobs
                idx_source  = keep_foresight_list
            else:
                print('⚠️ completed列表全空，改用first-stage结果。', flush=True)
                adv_source  = all_advantages_first_stage
                resp_source = all_responses_first_stage
                logp_source = all_logprobs_first_stage
                idx_source  = list(range(len(all_responses_first_stage)))
            if not adv_source:
                print('⚠️ 优势列表全空，使用均匀权重。', flush=True)
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
        检查是否满足早停条件。
        在多模态推理中，这能显著节省由于长视频 Token 带来的计算开销。
        """
        # 1. 强制性约束：必须达到最小前瞻步数
        if current_step < self.args.least_foresight_num:
            return False

        # 2. 检查完全一致性 (Exact Match Stop)
        # 如果所有候选路径在补全后完全一模一样，说明搜索已收敛到唯一解
        just_stop = True
        trajectories = step_results.get("trajectories", [])
        if len(trajectories) > 1:
            first_response = trajectories[0].strip()
            for response in trajectories[1:]:
                if response.strip() != first_response:
                    just_stop = False
                    break
        else:
            just_stop = False # 只有一个候选时不触发此逻辑

        if just_stop and len(trajectories) > 0:
            print(f"🛑 [样本早停] 深度 {current_step}: 所有补全路径完全一致。")
            return True

        # 3. 检查聚类共识 (Cluster Consensus Stop)
        # 如果某一个类别的路径占比超过了 threshold (例如 70%)，则认为达成了多数共识
        if self.args.depth_pruning_strategy == "cluster":
            if step_results.get("stop_foresight", False):
                print(f"🛑 [样本早停] 深度 {current_step}: 最大簇占比已超过阈值 {self.args.threshold} (共识达成)。")
                return True

        return False

    def _generate_final_response(self, example, system_prompt, previous_steps, previous_values, token_stats, rollout_stats, traj_info):
        """
        最终阶段：生成最终结果。自动兼容 AVH (单句)、HaloQuest (简答)、文本数据集 (Answer格式) 和 CMM/PhD (两阶段判定)。
        """
        pixel_values = example.get('pixel_values')
        audio_values = example.get('audio_values')
        image_values = example.get('image_values')

        # 缓存中间推理
        traj_info['internal_reasoning'] = previous_steps 

        all_responses = []
        all_logprobs = []
        all_advantages = []

        print(f"🏁 正在执行最终对齐生成 ({self.args.step_beam_size} 条路径)...", flush=True)

        for beam_idx in range(self.args.step_beam_size):
            # --- 初始化上下文 ---
            chat = self._prepare_chat_template(example, system_prompt)
            p_str = self.processor.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            if isinstance(p_str, list): p_str = p_str[0]
            
            if "avh" in self.args.datasets.lower():
                # 改进 C：合并所有 beam 的推理记录，强制 AV 融合
                # sanity check：过滤掉退化/重复的 beam 内容，最多保留 150 字符/beam
                def _is_sane(text):
                    words = text.lower().split()
                    if len(words) < 3: return False
                    # 检测重复：任意单词出现超过 4 次视为退化
                    from collections import Counter as _C
                    if any(v > 4 for v in _C(words).values()): return False
                    return True
                labels = ["Audio" if b % 2 == 0 else "Visual" for b in range(len(previous_steps))]
                note_parts = []
                for b in range(len(previous_steps)):
                    note = previous_steps[b].strip()[:150]  # 最多 150 字符
                    if _is_sane(previous_steps[b]):
                        note_parts.append(f"[{labels[b]}]: {note}")
                if not note_parts:
                    note_parts = [previous_steps[0].strip()[:150]]
                reasoning_history = " | ".join(note_parts)
            else:
                reasoning_history = previous_steps[beam_idx]

            # ==========================================
            # 🚀 分支 A：AVH 专用逻辑 (视听结合单句生成)
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

                # [DIAG-AVH] 完整上下文诊断 print
                print(f"    [DIAG-AVH] beam={beam_idx} reasoning_history (full):", flush=True)
                print(f"    [DIAG-AVH] notes={repr(reasoning_history)}", flush=True)
                print(f"    [DIAG-AVH] avh_final_instruction={repr(avh_final_instruction)}", flush=True)
                print(f"    [DIAG-AVH] inputs_text_avh (FULL)={repr(inputs_text_avh)}", flush=True)

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
            # 🚀 分支 B：HaloQuest 专用逻辑 (纯视觉简答)
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
            # 🚀 分支 C：纯文本描述/QA任务 (HaluEval, RAGTruth, PubMedQA)
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
                
                # 纯文本任务不传 images/audios，降低显存和时间消耗
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
                    
                    # 提取最终内容 (防止模型多余输出)
                    gen_answer = raw_text.split('\n\n')[0].strip()
                    if gen_answer.startswith("[") and gen_answer.endswith("]"):
                        gen_answer = gen_answer[1:-1] # 去掉模型可能画蛇添足加的中括号
                    # 分支 C（haloquest / halueval）原先漏了 boxed 剥离；
                    # 必须放在下面 halueval 的 yes/no 截断之前。见 FINDINGS §9.1。
                    gen_answer = unwrap_boxed(gen_answer)
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
            # 🚀 分支 D：CMM / PhD 等二元分类任务 (两阶段判定)
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

                    # 1. 提取问题 (修复 NameError 崩溃)
                    question_text = example.get('question') or example.get('input')

                    # 2. 生成客观事实清单
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
                    # 🌟 核心修改：删除 match_history 捷径，强制要求模型进行二次审判！
                    # 强迫它对比自己刚写的 Checklist 和真实的问题，解决脑口不一！
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

                    # 🌟 最终组装
                    final_output = f"Visual Checklist:\n- {clean_desc}\nJudgment: \\boxed{{{clean_judg}}}"
                    token_stats["output"] += (len(s1_gen_ids) + len(s2_gen_ids))

            # --- 结果汇总 ---
            all_responses.append(final_output)
            all_logprobs.append(logprob)
            
            # 🌟 修复 2：动态评分机制！
            if self.args.datasets.lower() in ["avh", "halo", "ragtruth", "halueval", "pubmedqa"]:
                # 造句/抽取任务，以最终生成的质量(logprob)为最高优先级！
                all_advantages.append(logprob)
            else:
                # CMM / PhD 是打假任务，以前瞻防幻觉阶段(Foresight)的黄金置信度为最高优先级！
                all_advantages.append(previous_values[beam_idx]) 
            
            rollout_stats["total"] += 1
            torch.cuda.empty_cache()

        # 决策逻辑：选择最佳路径
        selected_idx = self.select_response(all_responses, all_logprobs, all_advantages)

        # 记录全景轨迹
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
        """MiniCPM 格式的 Chat 模板 (Completion Stage)"""
        user_text = self._build_user_text(example, system_prompt, stage='completion')
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]
        

    def _prepare_chat_template_for_first_stage(self, example, system_prompt):
        """MiniCPM 格式的 Chat 模板 (First Stage / Foresight)"""
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


def main():
    """TokenGuard 终极版：支持断点续传、异常隔离与多重路径对齐"""
    args = parse_arguments()
    decoder = PhiDecoder(args)

    # Log active tuning flags
    if getattr(args, 'bypass_path', False):
        print(f"[FLAG] --bypass_path ACTIVE: Path-Confess veto disabled", flush=True)
    if getattr(args, 'dump_internals', False):
        print(f"[FLAG] --dump_internals ACTIVE: per-token metrics will be saved to {args.output_dir}/internals/{args.datasets}/", flush=True)
    if getattr(args, 'use_base_prompt', False):
        print(f"[FLAG] --use_base_prompt ACTIVE: Using base v0 prompts instead of GUARD prompts (Exp-X2)", flush=True)

    if args.ablation_suite:
        run_ablation_suite(args, decoder)
        return

    # 1. 加载原始数据
    with open(args.data_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
        
    start_idx = 0
    end_idx = args.max_samples if args.max_samples > 0 else len(test_data)
    test_data = test_data[start_idx : end_idx]
    print(f"📊 数据截取: {start_idx} ~ {end_idx}，共 {len(test_data)} 条")
        
    total_samples = len(test_data)
    print(f"📊 当前待处理样本总数: {total_samples}")

    # 2. 路径对齐：确保输出到你 find 命令查到的那个目录 (/audit_logs/)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.time_path, exist_ok=True)
    
    output_path = os.path.join(args.output_dir, f"{args.file_name}.json")
    traj_path_jsonl = os.path.join(args.time_path, f"TRAJ_INFO-{args.file_name}.jsonl")
    error_log_path = os.path.join(args.output_dir, f"ERROR-{args.file_name}.log")

    # 🌟 3. 精准断点探测 (通过物理行数确定起点)
    processed_count = 0
    print(f"🔍 正在探测进度文件: {output_path}")
    
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            # 统计非空行数，确定已经完成了多少个样本
            processed_count = sum(1 for line in f if line.strip())
        
        if processed_count >= total_samples:
            print(f"✅ 任务状态：审计已全部完成 ({processed_count}/{total_samples})。")
            return
        else:
            print(f"🔄 断点同步：检测到已有进度 {processed_count}/{total_samples}。")
            print(f"⏩ 正在准备从索引 #{processed_count} (第 {processed_count + 1} 个) 开始续传...", flush=True)
    else:
        print(f"🆕 未发现历史文件，将从第一个样本开始全新的审计流程。", flush=True)

    # 4. 统计指标初始化
    start_time = time.time()
    total_stats = {
        "total_rollouts": 0, "saved_rollouts": 0,
        "input_tokens": 0, "output_tokens": 0,
        "success_count": 0, "fail_count": 0
    }

    # 🌟 5. 核心追加模式循环：软跳过已处理样本
    
    with open(output_path, "a", encoding="utf-8") as res_f:
        for i, example in enumerate(test_data):
            # 索引过滤：跳过已存在于 output_path 中的前 N 行
            if i < processed_count:
                continue

            result = None
            # P3-c: Pre-sample cache clear to prevent OOM accumulation (esp. PubMedQA)
            import gc; gc.collect()
            torch.cuda.empty_cache()
            try:
                sub_cat = example.get("sub_category", "unknown")
                print(f"\n🚀 [Progress {i+1}/{total_samples}] 正在处理审计任务 #{i} | 子类: {sub_cat}", flush=True)
                system_prompt = decoder.get_system_prompt(args.datasets)
                
                # 执行推理
                result = decoder.process_example(example, system_prompt)

                # 累加统计数据
                total_stats["total_rollouts"] += result["rollout_stats"]["total"]
                total_stats["saved_rollouts"] += result["rollout_stats"]["saved"]
                total_stats["input_tokens"] += result["token_stats"]["input"]
                total_stats["output_tokens"] += result["token_stats"]["output"]
                total_stats["success_count"] += 1

                # 实时保存结果 (每行一条有效 JSON)
                # 实时保存结果 (每行一条有效 JSON)
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
                # 物理同步，防止 nohup 缓冲区导致的崩溃丢数
                os.fsync(res_f.fileno())

                # 实时追加审计轨迹
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
                        print(f"    📊 [DUMP] Wrote {len(result['traj_info']['_dump_internals'])} rollout records to {_dump_path}", flush=True)
                        del result["traj_info"]['_dump_internals']  # don't bloat traj file
                    with open(traj_path_jsonl, "a", encoding="utf-8") as f_traj:
                        f_traj.write(json.dumps(result["traj_info"], ensure_ascii=False) + "\n")

                print(f"✅ 索引 #{i} 审计成功。已产出 {total_stats['success_count']} 个新结果。")

            except torch.cuda.OutOfMemoryError as oom_e:
                import gc
                print(f"\n💥 [OOM] 索引 #{i} CUDA 显存不足，立即清理并跳过。", flush=True)
                torch.cuda.empty_cache()
                gc.collect()
                total_stats["fail_count"] += 1
                with open(error_log_path, "a", encoding="utf-8") as f_err:
                    f_err.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | ID {i} | OOM | {repr(oom_e)}\n")
                # 写入 OOM 占位符，保证文件行数与数据集索引 1:1 对应（断点续传依赖行数）
                fail_res = {"id": i, "sub_category": example.get("sub_category", "unknown"), "status": "OOM", "error": "CUDA OOM"}
                res_f.write(json.dumps(fail_res, ensure_ascii=False) + "\n")
                res_f.flush()

            except Exception as e:
                import traceback
                print(f"\n💥 [致命崩溃追踪] 索引 #{i} 发生如下异常：", flush=True)
                traceback.print_exc()  # 🌟 强制打印完整的红色报错堆栈！

                total_stats["fail_count"] += 1
                print(f"❌ 索引 #{i} 发生崩溃: {repr(e)}", flush=True) # 使用 repr 确保空异常也会显示类型
                with open(error_log_path, "a", encoding="utf-8") as f_err:
                    f_err.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | ID {i} | {repr(e)}\n")
                    f_err.write(traceback.format_exc() + "\n")

                # 写入失败占位符，保证文件行数与数据集索引 1:1 对应
                fail_res = {"id": i, "sub_category": example.get("sub_category", "unknown"), "status": "failed", "error": repr(e)}
                res_f.write(json.dumps(fail_res, ensure_ascii=False) + "\n")
                res_f.flush()

            finally:
                # --- 彻底释放显存，防止显存碎片堆积 ---
                import gc
                if result:
                    if "traj_info" in result: result["traj_info"].clear()
                    result.clear()
                    del result
                gc.collect()
                torch.cuda.empty_cache()
                print(f"🧹 索引 #{i} 资源清理完毕。")

    # 6. 任务结束：保存汇总报告
    time_span = time.time() - start_time
    time_info_path = os.path.join(args.time_path, f"{args.file_name}.txt")
    with open(time_info_path, "w", encoding="utf-8") as f_summary:
        f_summary.write(f'time:  {time_span}\n')
        f_summary.write(f'total_samples: {total_samples}\n')
        f_summary.write(f'success: {total_stats["success_count"]}\n')
        f_summary.write(f'all_input_tokens: {total_stats["input_tokens"]}\n')
        f_summary.write(f'all_output_tokens: {total_stats["output_tokens"]}\n')

    print(f"\n🏁 审计任务流水线结束。成功: {total_stats['success_count']} | 失败: {total_stats['fail_count']}")
    print(f"统计汇总已更新: {time_info_path}")

if __name__ == "__main__":
    main()