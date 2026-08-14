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
import cv2        # 导入 OpenCV 用于视频处理
import librosa    # 导入 librosa 用于音频处理
from PIL import Image  #导入 PIL 用于处理纯图像
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
    # 白名单从注册表**推导**，不再手抄一份 —— 手抄那份漏过 mc3，
    # 四个分片在 argparse 阶段就被拒（0/20 空跑三轮）。加新版本只需改 judge_prompts.py。
    _JUDGE_PROMPT_VERSIONS = sorted({v for _d in _JUDGE_PROMPTS_REGISTRY.values() for v in _d})
    _JUDGE_PROMPTS_LOADED = True
except ImportError:
    _JUDGE_PROMPTS_LOADED = False
    _JUDGE_PROMPT_VERSIONS = ['v0']
try:
    # 格式控制（解码期约束）。只在 --format_control != off 时被调用，
    # 导入失败也不影响任何既有臂。理由与实测数见 format_control.py 文件头。
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
    Qwen2_5OmniForConditionalGeneration, # 替换为 Omni 专用类
    Qwen2_5OmniProcessor,                 # 增加多模态处理器
)

# 注意：移除了 vLLM，因为 Qwen-Omni 的双分支架构更适合使用 transformers 原生调用进行受控生成

# 常量配置
INF = 10
TEMPERATURE = 0.6
# 设定显存管理策略，防止多模态长序列导致 OOM
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# ================= 1. 多模态加载工具函数 =================

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
    
# 喂给处理器的音频采样率。**默认 24000 是历史值，保留它是为了让旧 run 逐字节可复现。**
#
# ⚠️ 24000 是个 bug：模型的 preprocessor_config.json 写的是 WhisperFeatureExtractor +
#    sampling_rate=16000，而我们 7 处 processor(...) 调用都没传 sampling_rate=。
#    实测（CPU，真 Qwen2_5OmniProcessor，9.733s 的 wav）：
#        16 kHz →  974 特征帧 / 243 个 <|AUDIO|> token
#        24 kHz → 1461 特征帧 / 365 个 token        （比值 1.500 / 1.502）
#    ⇒ 处理器**不重采样、不报警**，一段 9.73s 的音频被当成 14.60s 编码，
#      等于放慢 1.5 倍、音高下移约 7 个半音（谱心 2839→1888 Hz = 0.66×）。
#    改成 16000 才是对的；用 --audio_sr 16000 打开。别直接改默认值 ——
#    已落盘的 cmm / avhbench 结果全是 24000 跑出来的，改默认会让新旧 run 无声地不可比。
AUDIO_SR = 24000


def load_audio_data(audio_path, sr=None):
    """加载音频数据。sr=None ⇒ 用全局 AUDIO_SR（由 --audio_sr 设定）。"""
    if not audio_path or not os.path.exists(audio_path):
        return None
    try:
        audio, _ = librosa.load(audio_path, sr=(AUDIO_SR if sr is None else sr))
        return audio if (audio is not None and len(audio) > 0) else None
    except:
        return None


# 视频与音频在序列里怎么排。**默认 False = 历史行为**（两个整块：先一整段 <|VIDEO|>，
# 再一整段 <|AUDIO|>），用 --use_audio_in_video 换成 Qwen 官方的 TMRoPE 时间交错排布
# （每 ~2 s 一块，视频块与音频块交替，整体只包一层 vision_bos/audio_bos … eos）。
#
# 实测（CPU，真处理器，10 s 片段）：两种排布**token 总数完全一样**（599，AUDIO 250 /
# VIDEO 320），差别只在顺序 —— 2 个整块 vs 10 个交错块。⇒ 这不是省钱的旋钮，
# 是"模型能不能知道某个声音发生在哪一帧"的旋钮。
#
# ⚠️ 打开它牵动四处，缺一处就是错的（都在下面用 USE_AUDIO_IN_VIDEO 门控）：
#   1. 聊天模板不能再单独放 {"type": "audio"} —— 交错分支会**再吃一次**音频迭代器，
#      文本里同时有 <|VIDEO|> 和 <|AUDIO|> 时直接 StopIteration 崩掉（已实测）。
#   2. 7 处 processor(...) 都要传 use_audio_in_video=True，否则 token 排布不一致。
#   3. 走 get_rope_index 的那两处 generate 也要传，否则 TMRoPE 位置编码按错误分支算。
#   4. get_modality_indices 不能再用 bos/eos 找区间 —— 交错后 vision_bos…vision_eos
#      把音频一起框进去、audio_bos…audio_eos 把视频一起框进去，两个区间几乎完全重叠，
#      MDI 屏蔽（Dist/Path 的地基）就废了。交错时改用 <|VIDEO|>/<|AUDIO|> 本身的
#      token id 逐个挑，两种排布下都对。
USE_AUDIO_IN_VIDEO = False


def _aiv_kw(pixel_values=None, audio_values=None):
    """processor(...) 调用点的 use_audio_in_video 透传。

    关着就返回**空 dict** ⇒ 调用点与从前逐字节相同（而不是显式传 False）。
    ⚠️ 必须**视频和音频都在**才传 True：交错分支里 <|VIDEO|> 会去 `next(audio_lengths)`，
       有视频没音频时那个迭代器是空的 ⇒ StopIteration 直接崩。
    """
    ok = (USE_AUDIO_IN_VIDEO and pixel_values is not None and audio_values is not None)
    return {"use_audio_in_video": True} if ok else {}


def _aiv_kw_for(inputs):
    """generate(...) 调用点的透传：直接看处理器产出里有没有视频+音频，判据与 _aiv_kw 等价。

    走 get_rope_index 的那两处必须和处理器保持一致，否则 TMRoPE 会按错误分支数
    多模态段落（video_nums 的判据本身就是 `use_audio_in_video` 的分支，见
    modeling_qwen2_5_omni.py:359-363）。
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
    """这条样本的音频是不是已经被折进视频块里了（⇒ 聊天模板不该再单独放一个音频占位符）。

    只有**同时有视频和音频**才折。只有音频没视频时（纯音频样本）交错分支根本不会触发，
    <|AUDIO|> 还得照常出现，否则音频就整段丢了。判据与 _aiv_kw 必须一致。
    """
    return (USE_AUDIO_IN_VIDEO
            and example.get('pixel_values') is not None
            and example.get('audio_values') is not None)


# ================= Token-Guard 防御插件 =================
class DualLayerAuditor:
    def __init__(self, model, processor, target_layers=[16, 22], store_hidden=False,
                 scope='last'):
        self.model = model
        self.processor = processor
        self.target_layers = target_layers
        self.store_hidden = store_hidden
        # ── scope：钩子该在**哪一次**前向上取值 ────────────────────────────
        # 每个解码步要跑 2~3 次前向（全模态 out_f / 无视频 out_nv / 无音频 out_na），
        # 而 forward hook 每次都会触发、字典后写覆盖前写，读取又全在三次之后 ⇒
        #   'last'（HEAD 行为）：主字典里留下的是**最后那次（盲态）**前向的层内表示。
        #            Path 罚分的 l16/l22 logits、Layer-Confess 的 h_final/h_anchors
        #            拿到的都是「屏蔽掉视觉/音频之后」的表示 —— 正好是这两个模块
        #            要检测的那种状态本身，等于用病人的病历当体温计。
        #   'full'：只有全模态那次写主字典，盲态那次写 blind_*（留给诊断/对比）。
        #   'shadow'：**解码行为与 'last' 逐字节相同**（主字典仍由盲态那次最后写入），
        #            但同时把全模态那次的层内表示记进 shadow_*。
        #            这样一趟 run 就能同时拿到「HEAD 实际用的量」和「修好之后的量」，
        #            在**同一批候选、同一条轨迹**上做配对比较 —— 不用跑两臂 A/B。
        self.scope = scope
        self.active = True          # 由调用点在每次前向前置位；scope='last' 时无作用
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
                    # 全模态那次只进影子字典；主字典留给盲态那次覆盖 ⇒ 解码行为不变
                    L, H = self.shadow_logits, self.shadow_hidden
                else:
                    L, H = self.intermediate_logits, self.intermediate_hidden
                if self.store_hidden:
                    H[layer_idx] = hidden_states[:, -1, :].clone()
                # Qwen-Omni 的 Thinker 分支特定层归一化与输出
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
    """剥掉 `\\boxed{...}` 包装（含无闭合括号的那种）。

    为什么必须剥：eval 的 Token-F1 先删标点**且不补空格**，所以 `\\boxed{no}`
    会变成单个 token `boxedno` —— 不但多一个垃圾词，还把真正的答案词吃掉，
    该条 F1 直接归零。haloquest 全量实测：655 条里 465 条带包装，
    剥掉后 Token-F1 9.35 → 11.94（+2.59，受影响的 218 条均幅 +7.78）。
    注意旧写法 `lstrip('boxed{')` 一次也没触发过：首字符是 `\\` 不在字符集里。
    """
    t = str(s).strip()
    m = _BOXED_RE.match(t)
    return m.group(1).strip() if m else t


def _final_verdict(text):
    """取文本里最后一个显式 yes/no；取不到返回 None（**不**强行判 No）。
    口径与 GUARD/eval/single_eval.py 及 compare_tiebreak_rules.py 完全一致。"""
    s = re.sub(r"\\boxed\{?", " ", str(text).lower())
    m = re.findall(r"\b(yes|no)\b", s)
    return m[-1] if m else None


def resolve_final_tiebreak(final_texts, second_texts, cur_idx):
    """R5「二阶段自洽投票」：只在两条**最终**候选判决分歧时介入。

    介入方式：用所有 second_stage completions 的多数判决决定倒向哪一边，
    再把 selected_idx 改到判决与之相同的那条最终候选；多数票平票则不动。

    为什么是这条规则（离线全量配对实测，见 GUARD/compare_tiebreak_rules.py）：
        规则                 净改对   p        对称?
        R1 分歧时取肯定        +43   0.0003    ❌ 只奖 Yes 类，会被指为对着指标调
        R5 二阶段自洽投票      +40   0.0007    ✅ 对称，不预设任何一侧
    R5 拿到 R1 九成以上的收益，而且**零参数、零额外前向**：投票用的文本本来
    就已经生成并落盘了，以前只是没被用来决定最终答案。

    刻意做成模块级纯函数：compare_tiebreak_rules.py 直接 import 它来复算 R5，
    若那张表的 R5 行不变，就证明落地代码与离线验证过的规则是同一条。
    """
    finals = [_final_verdict(t) for t in final_texts]
    if len(finals) < 2 or any(v is None for v in finals) or len(set(finals)) < 2:
        return cur_idx                      # 不分歧（或取不到判决）⇒ 一律沿用现行选择器
    votes = [v for v in (_final_verdict(t) for t in second_texts) if v]
    if not votes:
        return cur_idx
    y = votes.count("yes")
    n = votes.count("no")
    if y == n:
        return cur_idx                      # 平票 = 不介入（保持对称）
    target = "yes" if y > n else "no"
    for i, v in enumerate(finals):
        if v == target:
            return i
    return cur_idx


def calculate_hallu_penalty(report, tokenizer, thresholds=None):
    """
    计算幻觉惩罚分，并实时输出触发详情。
    纯层间信号：res_22_fin, res_16_22, l22_conf（不依赖 MDI）。
    thresholds: dict of calibratable values (None → _DEFAULT_THRESHOLDS).
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

    # 2. 追踪晚期冲突 (L22 vs Final) —— 模型在最后时刻"突然改口"
    if report['res_22_fin'] > thresholds['res_22_fin']:
        p_inc = thresholds['penalty_late']
        # 加重条件：L22 置信度极高却被 final 推翻 → 强行覆盖
        if report['l22_conf'] > thresholds['l22_conf_late']:
            p_inc += thresholds['penalty_late_boost']

        penalty += p_inc
        print(f"🚨 [AUDIT: LATE] Token: '{token_str}' | Res: {report['res_22_fin']:.2f} | L22_Conf: {report['l22_conf']:.2f} | Penalty: +{p_inc}", flush=True)

    # 3. 追踪早期劫持 (L16 vs L22) —— 语言模型套话覆盖了感官/上下文
    if report['res_16_22'] > thresholds['res_16_22'] and report['l22_conf'] > thresholds['l22_conf_hijack']:
        p_inc = thresholds['penalty_hijack']
        penalty += p_inc
        print(f"🚨 [AUDIT: HIJACK] Token: '{token_str}' | Res: {report['res_16_22']:.2f} | Conf: {report['l22_conf']:.2f} | Penalty: +{p_inc}", flush=True)

    return penalty

def _grnd_score(st, field='all', agg='sqrt'):
    """把逐路径充分统计量折成一个标量，约定 **越大 = 越有据可依**。

    field 选通道：
      all/content/neg  掩码对比 g_t = log p_full − log p_blind（neg = 只累加负半边，
                       即旧 dist_ev 单边口径的对数版，做对照用）
      late22/late16    层间注入 d_t = log p_final − log p_{22,16}，**不需要掩码**。
                       d 越大 = 越像末层硬加上来的先验 ⇒ 取负号后仍是「越大越好」。
      late22c          同 late22，仅内容词。
      lp22/lp16        **有界版**层间注入 e_t = p_final(y) − p_{22,16}(y) ∈ [−1,1]。
                       和 late22/late16 问同一件事，但对数比是无界的（中层给该 token
                       ~0 概率时发散），实测 21/24 个候选顶穿 ±20 夹断，极值聚合
                       根本排不出序。概率差有界，所以 min/worst3 在这条通道上才有定义。
                       累计聚合（sum/sqrt）两种口径都可用，对数比那版理论上更对
                       （它才是对数贝叶斯因子），所以两条都留着，让数据选。

    agg 选归一化/聚合：
      sum   Σ         路径的对数贝叶斯因子本身；长路径天然占优
      mean  Σ/n       长度无关，但把长路径的信号除小了
      sqrt  Σ/√n      信号 ~ n·μ、噪声 ~ σ√n ⇒ 信噪比最高（推荐默认）
      min   极值       幻觉是极值风险：一个无依据断言就毁掉整条回答，
                      均值会被其余正常词稀释掉。
      worst3 最差3个均值  min 只看一个 token，太靠运气；worst-3 是同一个
                      「极值风险」假设的稳健版。
      ⚠️ late22/late16/late22c × {min,worst3} 已实测饱和（无界对数比），
         要极值请用 lp22/lp16。
    """
    if not st:
        return 0.0
    # n<=0 的检查必须在**极值分支之前**：钩子没抓到中间层时 d_n=0，而 e22_max
    # 的哨兵初值是 −2.0（在 [−1,1] 之外），取负号后会变成 +2.0 —— 一个凭空的
    # 高分。原来这个检查在函数末尾，min/worst3 提前 return 就绕过去了。
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
            # 故意不支持：worst-k 是极值聚合，而 late* 用的对数比
            # log(p_final/p_l) 是**无界的**（中层常给该 token ~0 概率，
            # 实测 21/24 候选顶穿 ±20 夹断）。极值聚合在无界量上没定义，
            # 换任何夹断值都会饱和成常数。要 worst-k 请用有界通道
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
    """Eq.(layer-anchor-response) + Eq.(layer-repair) of method_layer_confess.tex.

    Transported anchor response (Eq. layer-anchor-response). The transport
    operator is
        A_t^(l) = I + alpha_l * (h_final - h_l) h_l^T / (||h_l||^2 + eps)
    and since h_l^T h_l = ||h_l||^2 the matrix never has to be materialised:
        A_t^(l) h_l = (1 - alpha_l) * h_l + alpha_l * h_final          (d-vector)

    Cross-layer support representation (text after Eq. layer-repair):
        h_sup = sum_l omega_l * A_t^(l) h_l,        sum_l omega_l = 1

    Orthogonal restoration (Eq. layer-repair):
        h~ = h_final + lambda_L * P_perp(h_sup - h_final)
        P_perp(v) = v - h_final <h_final, v> / ||h_final||^2

    Because the update is orthogonal to h_final, ||h~||^2 = ||h_final||^2 +
    lambda_L^2 ||P_perp(v)||^2, i.e. ||h~|| = ||h_final|| + O(lambda_L^2) —
    the norm-preservation claim of Proposition 2.

    ── project='full' 的动机（2026-08-05 实测后加）──────────────────────────
    P_perp 剥掉的是 (h_sup - h_final) 中**沿 h_final 方向**的分量，而那一分量正是
    "削弱最终层自己那份先验驱动的预测"的部分；留下的只有侧向微调。实测 gate_on/
    gate_off 两个 run：delta_perp_norm ~75 而 ||h_final|| ~500，lambda_L=0.3 ⇒
    实际位移 22.5，夹角 2.6 度，norm_ratio 1.001。也就是说 Proposition 2 的范数
    保持性是靠"几乎不动"换来的，代价是修复无法压制先验方向。
        project='perp'  h~ = h_final + lambda_L * P_perp(h_sup - h_final)   （原式）
        project='full'  h~ = h_final + lambda_L * (h_sup - h_final)         （含平行分量）
    'full' 不再保范数（会沿 h_final 缩短，这正是压制先验所需），Proposition 2 的
    结论对它不成立 —— 用它就得改论文的那条命题，不能两头都要。

    Args:
        h_anchors: list of anchor hidden states (each shape [d]), excluding final
        h_final:   final-layer hidden state, shape [d]
        omega:     weights over h_anchors, summing to 1
        alpha_l:   depth-alignment coefficient alpha_l in [0, 1]
        lambda_l:  repair strength lambda_L
        project:   'perp' (Prop. 2 原式) 或 'full' (保留平行分量)
    Returns:
        (h_repaired, diagnostics dict) — h_repaired is cast back to h_final's
        original dtype; diagnostics carry the quantities E9 needs to verify
        Proposition 2 numerically. diag['_delta'] 是 float32 的更新方向（未乘
        lambda_L），供调用方做零额外前向的 lambda 扫描，用后应 pop 掉再落盘。

    Numerics: the model runs in bfloat16, whose spacing near 1.0 is 1/128
    (~7.8e-3). The norm deviation this repair induces is O(lambda_L^2), which for
    lambda_L=0.3 lands around 1e-3 — below bf16 resolution, so measuring it in
    bf16 yields quantised garbage (norm ratios pinned to 1±1/128) and an
    orthogonality residual inflated to ~5e-2. All arithmetic here is therefore
    done in float32; only the returned tensor is cast back.
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
        # 平行分量的大小：project='perp' 丢掉的正是这一份
        'delta_par_norm': float((h_final * proj_coeff).norm()),
        'project': project,
        '_delta': _upd,          # 调用方做 lambda 扫描用；落盘前 pop
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
    """
    终极封装版：带详细 Debug 打印的特征拦截器 generate。
    """
    print(f"\n    [GEN_DEBUG] ---> 进入 safe_model_generate，准备处理输入...", flush=True)
    import traceback # 导入 traceback 用于打印完整报错栈

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
            print(f"    [GEN_DEBUG] 🪝 Hook 捕获到 inputs_embeds! 形状: {captured['inputs_embeds'].shape}", flush=True)
            raise StopExecution("Captured")

    print(f"    [GEN_DEBUG] 正在挂载拦截钩子...", flush=True)
    h1 = model.thinker.register_forward_pre_hook(hook, with_kwargs=True)
    h2 = model.thinker.model.register_forward_pre_hook(hook, with_kwargs=True)
    
    try:
        print(f"    [GEN_DEBUG] 执行空 generate 诱发特征映射...", flush=True)
        with torch.inference_mode():
            model.generate(**inputs, max_new_tokens=1, use_cache=False, **_aiv_kw_for(inputs))
    except StopExecution:
        print(f"    [GEN_DEBUG] ✅ StopExecution 正常触发，特征已安全截取。", flush=True)
    except Exception as e:
        print(f"    [GEN_DEBUG] ⚠️ generate 诱发阶段发生意外异常: {repr(e)}", flush=True)
    finally:
        h1.remove()
        h2.remove()
        print(f"    [GEN_DEBUG] 钩子已卸载。", flush=True)
        
    embeds = captured.get('inputs_embeds')
    if embeds is None:
        print("    [GEN_DEBUG] ⚠️ 拦截失败！回退到纯文本 Embeddings...", flush=True)
        try:
            embeds = model.thinker.get_input_embeddings()(inputs['input_ids'])
            print(f"    [GEN_DEBUG] 纯文本 Embeddings 获取成功，形状: {embeds.shape}", flush=True)
        except Exception as e:
            print(f"    [GEN_DEBUG] ❌ 回退获取文本 Embeddings 失败: {repr(e)}", flush=True)
            raise e
            
    # 清理不兼容参数
    # ⚠️ 历史行为：`stop_strings` 连同 `tokenizer` 一起被丢掉 ⇒ 全项目 6 处停止串
    # 全是空操作（cmm 的 stage-1 声明了 stop_strings=["Judgment:","\n\n","Human:"]，
    # 实测 48.9% 的描述里含 "\n\n"；stage-2 声明了 ["}","Final"] 也从不生效）。
    # 只在 --fix_stop_strings 打开时用 StopStringCriteria 补回来，默认保持原样，
    # 这样既有臂逐字节可复现。
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
    print(f"    [GEN_DEBUG] 准备传入 thinker 的清理后参数: {list(clean_kwargs.keys())}", flush=True)
    
    try:
        print(f"    [GEN_DEBUG] 🚀 正在调用 model.thinker.generate 执行深层推理...", flush=True)
        out = model.thinker.generate(
            inputs_embeds=embeds,
            attention_mask=inputs['attention_mask'],
            pad_token_id=processor.tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
            **clean_kwargs
        )
        print(f"    [GEN_DEBUG] ✅ thinker.generate 执行成功！", flush=True)
    except Exception as e:
        print(f"    [GEN_DEBUG] ❌ thinker.generate 发生致命崩溃: {repr(e)}", flush=True)
        traceback.print_exc() # 强制打印红色堆栈，防止外层吞噬报错
        raise e
    
    try:
        # 🌟 缝合序列逻辑
        if hasattr(out, 'sequences'):
            print(f"    [GEN_DEBUG] 正在缝合生成的 Sequences... 新生成序列形状: {out.sequences.shape}", flush=True)
            full_seqs = []
            for seq in out.sequences:
                # 强行对齐 Device，防止 Tensor 不在一个 GPU 上导致的崩溃
                prompt_ids = inputs['input_ids'][0].to(seq.device)
                full_seq = torch.cat([prompt_ids, seq], dim=0)
                full_seqs.append(full_seq)
            out.sequences = torch.stack(full_seqs)
            print(f"    [GEN_DEBUG] 缝合完成，最终 Sequence 形状: {out.sequences.shape}", flush=True)
    except Exception as e:
        print(f"    [GEN_DEBUG] ❌ 序列缝合阶段崩溃: {repr(e)}", flush=True)
        traceback.print_exc()
        raise e
        
    print(f"    [GEN_DEBUG] <--- safe_model_generate 顺利结束，返回结果。\n", flush=True)
    return out

def softmax(x):
    e_x = np.exp(np.array(x) - np.max(x)) # 增加数值稳定性
    return e_x / e_x.sum(axis=0)


def get_modality_indices(processor, input_ids, device):
    """从输入序列中探测视频和音频 Token 的位置索引"""
    v_bos = processor.tokenizer.convert_tokens_to_ids("<|vision_bos|>")
    v_eos = processor.tokenizer.convert_tokens_to_ids("<|vision_eos|>")
    a_bos = processor.tokenizer.convert_tokens_to_ids("<|audio_bos|>")
    a_eos = processor.tokenizer.convert_tokens_to_ids("<|audio_eos|>")

    # 交错排布下 bos/eos 区间法会失效：序列长这样
    #   <|vision_bos|><|audio_bos|> [VIDEO…][AUDIO…][VIDEO…][AUDIO…] … <|audio_eos|><|vision_eos|>
    # ⇒ vision_bos…vision_eos 框住了全部音频、audio_bos…audio_eos 框住了全部视频，
    #   两个"区间"几乎完全重叠，屏蔽任一路等于把两路一起屏蔽。
    # 改用占位符本身的 token id 逐个挑（处理器最后会把 <|xxx_placeholder|> 还原成
    # <|VIDEO|> / <|AUDIO|> 重复若干次，见 processing_qwen2_5_omni.py:274-276），
    # 两种排布下都精确。**只在 --use_audio_in_video 打开时走这条**：它比旧路径少了
    # bos/eos 那两个 token，默认开着会让历史 run 的掩码差两位。
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
    """计算当前 Logits 的信息熵，用于观察模型纠结程度"""
    probs = torch.softmax(logits, dim=-1)
    return -torch.sum(probs * torch.log(probs + 1e-9), dim=-1).item()


def safe_thinker_step_audit(model, processor, inputs, past_key_values=None, position_ids=None, **gen_kwargs):
    """
    极简 Forward 版：只负责单步前向计算，不涉及复杂的 generate 逻辑。
    """
    valid_keys = ['input_ids', 'attention_mask', 'pixel_values_videos', 'video_grid_thw', 
                  'video_second_per_grid', 'feature_attention_mask', 'input_features']
    clean_inputs = {k: v for k, v in inputs.items() if k in valid_keys}
    
    for k, v in clean_inputs.items():
        if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
            clean_inputs[k] = v.to(torch.bfloat16)

    # 🌟 调用 forward 而不是 generate
    outputs = model.thinker(
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
    """Parse command line arguments specifically for Qwen2.5-Omni Phi-Decoding"""
    class _NoPercentHelp(argparse.HelpFormatter):
        """关掉 argparse 对 help 文本的 %-展开。

        🐞 主干上 `--help` 一直是崩的：help 里写满了实测百分数（"cmm 17.2% / phd 8.0%"），
        argparse 无条件对 help 做 `% params`，`% /` 直接抛 ValueError。
        本文件一个 %(default)s 都没用，关掉展开是安全的。
        """
        def _expand_help(self, action):
            return self._get_help_string(action)

    parser = argparse.ArgumentParser(description="Phi-Decoding Algorithm for Qwen-Omni",
                                     formatter_class=_NoPercentHelp)

    # ── 三个模块的真开关（tab:ablation 的前提）──────────────────────────────────
    # 之前全仓唯一的 disable 是 --disable_passage_mdi，所以论文那张 3×6 的模块消融表
    # 在实现上没有任何东西可以关。这里补上，并把「关掉之后到底还剩什么」写死在 help 里，
    # 因为三者在实现上**不是**正交的：
    #   Dist  产 _dist_flag（哪个 token 可疑）→ 同时是 Layer 修复的门控条件、
    #         也是 Path 的 dist_ev 证据来源；
    #   Layer 产 total_hallu_penalty（层间冲突罚分）→ 是 Path 一票否决的唯一输入；
    #   Path  只消费上面两个信号做路径打分/否决，自己不产生新的观测量。
    # 因此 \Dist 单独敲除必然连带 Layer 的门控与 Path 的一支证据 —— 这是实现事实，
    # 消融表里必须照实标注，不能声称是三个独立加项。
    _abl = parser.add_argument_group('模块消融开关（tab:ablation）')
    _abl.add_argument('--disable_dist', action='store_true',
                      help='关 Dist-Confess：_dist_flag 恒 False ⇒ 不做对比重采样、'
                           '不累计 dist_ev；且因 layer_repair_gated 默认 on，'
                           'Layer 的修复算子也不再被触发（用 --layer_repair_gated off 可解耦）。'
                           'Layer 的罚分统计不受影响（它本来就每 token 无条件跑）。')
    _abl.add_argument('--disable_layer', action='store_true',
                      help='关 Layer-Confess：不做隐状态修复，且 total_hallu_penalty 恒 0 ⇒ '
                           'Path 的一票否决与 head 模式罚分项同时失效（graded 模式下 Path 仍能'
                           '靠 dist_ev 打分，这是两者唯一能分开的口径）。')
    _abl.add_argument('--disable_path', action='store_true',
                      help='关 Path-Confess：不否决任何路径，advantage 退回纯 logprob '
                           '（不减罚分、不做树内标准化）。Dist 的重采样与 Layer 的修复照常执行，'
                           '只是不再影响选路。')

    # Model configuration
    # 默认指向 Qwen2.5-Omni 7B
    parser.add_argument('--model_id', type=str, default='qwen2.5-omni',
                        help='Model identifier')
    parser.add_argument('--model_path', type=str, default='Qwen/Qwen2.5-Omni-7B',
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
    parser.add_argument('--final_tiebreak', type=str, default='r5', choices=['off', 'r5'],
                        help='两条**最终**候选判决分歧时（cmm 17.2% / phd 8.0% / pubmedqa 7.6% '
                             '的样本上会发生）怎么裁决。'
                             '"r5" = 默认，用所有 second_stage completions 的多数判决裁决，'
                             '平票不动；"off" = 沿用旧选择器（2026-08-05 之前的历史结果都是'
                             '这一档，复现旧数字时用）。'
                             '⛔ **2026-08-13 订正：在出货配置下 R5 对六个数据集的分数贡献全是 0.00。**'
                             '出处 GUARD/analysis_0813/treevote_in_ablation.py（判断类）与 '
                             'r5_on_descriptive.py（描述类），读数在 '
                             'GUARD/results/robustness/ 下，零 GPU 可复算。'
                             '判断类（cmm/phd/pubmedqa，= _TV_JUDGE_SETS）：树共识默认开着，'
                             '_tree_vote() 的 target 只由整棵树的 Yes 占比决定、与选中哪条 beam 无关，'
                             '而最终判决恒等于 target ⇒ **R5 改不动判决，这是恒等式不是噪声**。'
                             '实测 3364 条：R5 真改了下标 241 次（7.16%），其中树共识没接管的 '
                             '**0 次（0.00%）**；18 个消融臂 2×2 里 ΔR5|树共识开 = +0.00，17/18 精确为零。'
                             '描述类（avhbench/haloquest/ragtruth）：树共识提前返回、盖不住 R5，'
                             '但 R5 自己的第一道门要两条最终候选**都取得到** yes/no 判决且互相分歧，'
                             '而描述类输出里通常没有这个词 ⇒ 5011 条上取到判决只 11.4%'
                             '（cmm 同代码 100%）、真改下标 **4 次 = 0.08%**，'
                             '那 4 条 Token-F1 合计 +19.41 ⇒ 摊平 +0.0039，两位小数印出来就是 0.00。'
                             '⚠️ 下面这个 +26 是**一个不出货配置**的读数，保留作出处，别再当增益引用：'
                             '离线全量配对实测，**按每数据集真实评分口径**（cmm/phd 用 '
                             'single_eval._parse_yesno_answer，pubmedqa 用 change_pubmed 的'
                             '取首个 yes/no，phd 只算论文的 566 条子集），n=1848：'
                             '改对 60 / 改错 34 = 净 +26 条（1.41pp 准确率），精确二项 p=0.0095 '
                             '—— 但那 1848 条正好就是三个判断集，且**两臂都关着树共识**'
                             '（FINDINGS_0804.md:367 那张表的 cmm 基线 67.13 = 我方 R0），'
                             '出货配置下它被树共识整个盖掉。'
                             '注意：早先用统一"取最后一个 yes/no"口径算出的 +40、p=0.0007 是'
                             '虚高 54% 的错数，不要再引用。'
                             'R5 零参数、零额外前向 —— 投票用的文本本来就生成并落盘了，'
                             '以前只是没参与最终决策；它在生成全部结束后才改选，不消耗 RNG，'
                             '所以离线复放的数字就是真机数字。'
                             '但它**仍然要付**「遍历全部二阶段补全再逐条正则」的成本 ⇒ 纯开销。'
                             'cmm 上 Yes 类 F1 −0.17（准确率 +1.27），是三集里唯一不涨的。')
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
    parser.add_argument('--disable_per_sample_seed', action='store_true',
                        help='消融：退回只在初始化时播种一次（HEAD 的行为，结果依赖样本顺序）')
    parser.add_argument('--tree_vote_tau', type=float, default=0.20,
                        help='判断类任务：Confess 树二阶段补全中判 Yes 的比例超过该阈值即判 Yes。'
                             '0.5 之所以不是默认值，是因为树本身被系统性推向 No（实测 GT=Yes 的 '
                             'cmm 样本树内 Yes 也只占 50.3%，GT=No 上 17.1%），阈值按留一数据集标定。')
    parser.add_argument('--tree_vote_tau_map', type=str, default='',
                        help='分数据集阈值，形如 "cmm=0.22,phd=0.18,pubmedqa=0.14"。'
                             '命中当前 --datasets 时覆盖 --tree_vote_tau，未命中则回落到它。'
                             '为什么需要它：三个判断集的 Yes 先验差得很远（pubmedqa 约 85% 是 Yes，'
                             'cmm 约 50%），同一个全局 τ 在 pubmedqa 上会把 PhD-Index 从 75.52 '
                             '压到 63.80 —— 是先验不匹配，不是规则本身错。'
                             '⚠️ 2026-08-06 已被更好的规则取代：真正的规律不是「每集一个 τ」，'
                             '而是 τ 正比于这次 run 自己的树 Yes 率 f̄（cmm 0.338 / phd 0.424 / '
                             'pubmedqa 0.802，τ*/f̄ 三集都是 0.6 上下）。用无量纲常数 '
                             'τ = 0.60·f̄ 时三集 × 三指标 9/9 为正、每格 p<0.02。'
                             'f̄ 要全集求均值，解码器里只能用滑动均值 ⇒ 会依赖样本顺序、'
                             '分片与全量对不上，所以自标定做成**后处理**：'
                             'GUARD/apply_tree_vote.py（分片安全、零 GPU、零标签）；'
                             '本旗标只保留给需要在解码时钉死绝对 τ 的场景。'
                             '切半复核见 GUARD/calib_tree_tau.py。')
    parser.add_argument('--disable_tree_vote', action='store_true',
                        help='消融：关闭树投票，最终判决完全由 select_response 决定（HEAD 行为）')
    parser.add_argument('--final_pool_yes_if_any', action='store_true',
                        help='终选池化：两条终选里只要有一条判 yes 就取那一条（只在判断类数据集、'
                             '只往上改）。动机是 select_response 与 argmax(logprob) 只约一半重合，'
                             '判决相当程度上在随机抽。⚠️ 实测 n=80 上 +3.01 Yes类F1，但 n=200 上'
                             '塌到 +0.81（胜负 3:4）—— **这是零效应规则**，别引用旧的 n=80 数字。'
                             '与后处理脚本 GUARD/apply_pool_yes_if_any.py 判决等价。默认 off。')
    parser.add_argument('--max_samples', type=int, default=-1,
                        help='Max samples to process (-1 = all)')
    parser.add_argument('--halo_prompt_version', type=str, default='v0',
                        help='HaloQuest prompt version: v0 (current), v1 (explicit format), v2 (sentence-prefix), v3 (few-shot template)')
    parser.add_argument('--halueval_prompt_version', type=str, default='v0',
                        help='HaluEval prompt version: v0 (baseline), v1 (yes/no fix + few-shot)')
    parser.add_argument('--avh_prompt_version', type=str, default='v0',
                        help='AVHBench prompt version: v0 (baseline), v1 (25w cap + English-only)')
    # ── 判断类（cmm/phd/pubmedqa）fewshot：v0 与旧硬编码逐字节相同，行为不变 ────
    # fs6 = 6 条**手写虚构**示例（cmm/phd 各 3 Yes+3 No，pubmedqa 4 Yes+2 No）。
    # 不取自任何数据集条目（cmm 795 / phd 1385 就是被评测集合本身，无留出池），
    # 也不把真值 Yes 比例写进提示词。详见 GUARD/judge_prompts.py 文件头两条纪律。
    parser.add_argument('--judge_prompt_version', type=str, default='v0',
                        choices=_JUDGE_PROMPT_VERSIONS,
                        help='cmm/phd/pubmedqa prompt version: v0 (baseline, 逐字节等于旧硬编码), '
                             'fs6 (6 hand-written synthetic few-shot exemplars), '
                             'mc (cmm only: 模态条件化规则 + 格式契约对齐, 无示例), '
                             'mc6 (mc + 6 条示例), '
                             'mc2 (cmm only: mc 的第二版, **只删不加** —— 依据是 mc 那 80 条链里 '
                             '61 条非树投票链的实测, 详见 judge_prompts.py 的 _CMM_MC2 注释)。'
                             'mc3 (mc2 但只写被问那一路一张清单, 已判否), '
                             'mc26 (mc2 + 6 条示例, 已判否), '
                             'mc27 (mc2 但**系统提示词按 see/hear 分岔**, 只删不加), '
                             'mc28 (mc27 + 决定路清单第一条必须是被问项)。'
                             'mc/mc6/mc2 在 phd/pubmedqa 上无定义 ⇒ 回落 v0')
    # ── 格式控制：把输出格式从「祈祷」改成「解码期约束」 ────────────────────────
    # 起因（cmm 785 条 3 月产物，2026-08-10 复算）：94.9% 的生成以列表序号开头、
    # 8.5% 逐字抄系统提示词的方括号占位符、51.2% 压根没写出 Auditory Checklist
    # （音频题上 49.5%），真正以 `Description:` 开头的 **0 条**。
    # 缺段不是预算问题：撞满 200 词元的只有 3.2%，缺段那批反而更短。
    # 提示词措辞治不了这个 —— **结构 token 是谁写的，才决定格式**。
    #   template  骨架强制：`Auditory Checklist:` 这类结构由我们逐 token 发射，
    #             模型只填槽 ⇒ 格式合规由构造保证；判决槽锁死 Yes/No
    #   anticopy  反抄袭：禁掉会续出「提示词里出现过的 6-gram」的词元（通用闸门）
    #   both      两个都开（出货候选）
    # off 时**一行行为都不变**，既有臂逐字节可复现。
    parser.add_argument('--format_control', type=str, default='off',
                        choices=['off', 'template', 'anticopy', 'both'],
                        help='判断类分支的解码期格式约束，见 GUARD/format_control.py')
    parser.add_argument('--anti_copy_n', type=int, default=6,
                        help='反抄袭的 n-gram 阶数：禁掉会与提示词连续重合 n 个词元的续写')
    parser.add_argument('--fix_stop_strings', action='store_true',
                        help='把 safe_model_generate:562 丢掉的 stop_strings 补回来'
                             '（全项目 6 处停止串现在都是空操作，cmm 48.9%% 的描述里'
                             '含本该停住的 \\n\\n）。默认 off 以保既有臂可复现。')
    # 二阶段判决锁**单独一个旗标**，不并进 --format_control：它约束的是判决词元
    # 本身，风险面与一阶段骨架完全不同。锁上之后 `out_s2.scores` 是**被处理过**的
    # 分数（HF 的 output_scores 返回处理后 logits）⇒ 6 选 1 的 log_softmax 比自由
    # 解码高得多，`all_logprobs` 的口径会变。
    # 对 cmm/phd 这只影响**日志**：分支 D 的 all_advantages 取的是前瞻价值
    # （base_hook:3681 `previous_values[beam_idx]`），select_response 排的是
    # advantage 不是 logprob。但既然口径变了就必须能单独开关、单独记账。
    parser.add_argument('--verdict_lock', action='store_true',
                        help='二阶段只许生成 Yes/No 且正好 1 个词元 ⇒ 判决解析'
                             '不可能失败（会让 :3653 那个单边兜底 else "No" 不可达）。'
                             '代价：改变 all_logprobs 的口径，默认 off。')
    # ── Task profile: 规则挂在"输入的可测属性"上，不挂数据集名字 ────────────────
    # 'off'  : 完全复现 HEAD 行为（默认，用于逐字节回归验证）
    # 'auto' : 从 system prompt / 参考答案长度统计推出属性，再决定用哪条分支。
    #          属性都是任务规格的公开信息，不是测试集标签，因此不构成 per-dataset 调参。
    parser.add_argument('--task_profile', type=str, default='off', choices=['off', 'auto'],
                        help="'auto' 时按答案空间/参考长度自动选择 CD 与长度策略；'off' 复现 HEAD")
    parser.add_argument('--cd_alpha', type=float, default=0.5,
                        help='CD-Confess 对比放大强度（原为硬编码 0.5）')
    parser.add_argument('--cd_resample', type=str, default='sample',
                        choices=['sample', 'argmax'],
                        help="CD 修正后重选方式。'sample'=multinomial（HEAD 行为，有随机性）；"
                             "'argmax'=取最大概率。二元答案任务下 sample 会把翻转变成掷骰子")
    parser.add_argument('--ref_len_words', type=float, default=None,
                        help='该任务参考答案的中位词数（数据集公开统计量）。'
                             '给了且 <15 时 task_profile=auto 会启用长度收缩')
    parser.add_argument('--cd_mode', type=str, default='hard', choices=['hard', 'soft'],
                        help='Dist-Confess CD correction mode: hard (zero out token) or soft (scale down)')
    parser.add_argument('--cd_soft_factor', type=float, default=0.1,
                        help='Soft CD scaling factor (only used when cd_mode=soft)')
    parser.add_argument('--cd_plausibility_beta', type=float, default=0.0,
                        help='对比解码的 adaptive plausibility constraint（Li et al. 2022 / DoLa / VCD 标配）。'
                             'β>0 时，先把 p_full(v) < β·max_w p_full(w) 的词元屏蔽掉，再做对比放大。'
                             '不加这一步时 −α·blind_logits 项会奖励盲态概率最低的词元，'
                             '实测把 <|endoftext|>/<|im_end|>/标点顶进候选（14709 次替换里 8.0%）。'
                             '0.0 = 关闭（复现 HEAD 行为）；推荐 0.1。')
    parser.add_argument('--cd_contrast_lambda', type=float, default=None,
                        help='对比目标里盲态项的权重：refined = logits_f − λ·blind_logits。'
                             'HEAD 的 `logits_f + α(logits_f − blind)` 排序上等价于 λ=α/(1+α)=1/3，'
                             '即**先验权重是感官对比的 3 倍**，且等效温度被压到 0.6/1.5=0.4（比探测采样更尖）。'
                             '标准 CD 用 λ=1（纯感官对比）。给了这个值就改用 λ 形式，'
                             '同时把等效温度还原为 TEMPERATURE。None = 保持 HEAD 公式不变。'
                             '必须与 --cd_plausibility_beta 一起用：λ 调大会放大尾部噪声，β 是它的安全栏。')
    parser.add_argument('--cd_forbid_degenerate', action='store_true',
                        help='把**退化词元**从对比候选集里彻底排掉：特殊符（<|endoftext|> 等）、'
                             '纯标点/格式符、纯空白。β 门控只按 p_full 截，挡不住它们 —— '
                             'Q 电池 a3（λ=0.33 β=0.1，cmm n=80）实测 2827 次改写里，'
                             '改成纯标点 194 次、空白 44 次、特殊符 53 次（endoftext 40 次），'
                             '实词净流失 498 次（536 次实词→非实词，反向只有 38 次）。'
                             '默认 off = 与既有结果逐字节一致。')
    parser.add_argument('--cd_diag', action='store_true',
                        help='每个 Dist 事件打一行 [CD-DIAG] JSON：在 λ∈{1/3,0.5,1,1.5} × β∈{0,0.1} '
                             '八种设置下各自的 top-3 候选词，外加 HEAD 现行公式实际选的词。'
                             'logits 都是现成的，网格零额外前向 ⇒ **一次 run 就能离线定 λ 和 β**，'
                             '不必每组参数各跑一遍。只诊断，不改变解码行为。')
    parser.add_argument('--path_score', type=str, default='head',
                        choices=['head', 'graded', 'graded_sym'],
                        help="Layer→Path 的罚分通道。'head'=复现原行为（罚分只贡献 1 bit："
                             "≥0.45 就 advantage=-10000，否则 -0.8*penalty 恒为 0，实测死代码率 100%%）；"
                             "'graded'=树内标准化的分级罚分，不否决；"
                             "'graded_sym'=在同答案类的兄弟路径内标准化，消除 Yes/No 类偏置。"
                             "非 head 模式必须同时传 --veto_threshold 1e9（会强制校验）。")
    parser.add_argument('--path_kappa', type=float, default=0.5,
                        help='分级罚分强度：一个罚分标准差折合多少个 logprob 标准差。'
                             '尺度无关，跨数据集可迁移。参照：HEAD 的 0.8 系数等效于 '
                             '一次 LATE 事件 1.00σ、两次 2.01σ（logprob σ 实测 0.398）。')
    parser.add_argument('--dump_probe_features', type=str, default='',
                        help='候选级探针特征的落盘目录（默认空 = 关闭，与 HEAD 逐字节一致）。'
                             '为每条一阶段候选导出它**自己 token 的中间层隐状态均值**，'
                             '外加 texts/logprob/is_vetoed，写成 {dir}/{file_name}/{idx:06d}.npz。'
                             '动机：本仓查过的七个量（logprob / Layer 罚分 / Dist 陷阱深度 / '
                             '前瞻置信度 / 簇规模 / 多数票 / 保多样性）对「这条候选是否正确」'
                             '的 AUC 全部落在 0.50~0.55，而文献报采样类检测器上限 0.541、'
                             '中间层隐状态 + 有监督线性探针 0.904~1.000 —— 说明问题在方法族，'
                             '不在参数。本旗标只负责**采数据**，探针在离线训（CPU 几秒）。')
    parser.add_argument('--probe_layers', type=str, default='14,20,24',
                        help='取哪几个 block 的输出做探针特征（逗号分隔，0-indexed）。'
                             '默认 14,20,24 覆盖 28 层模型的中段峰值带（文献报 Qwen 系为 19~25/28）。'
                             '特征按此顺序拼接成一个长向量。')
    parser.add_argument('--auditor_scope', type=str, default='last',
                        choices=['last', 'full', 'shadow'],
                        help='层内审计（Path 罚分的 l16/l22 logits、Layer-Confess 的 h_final/'
                             'h_anchors）从**哪一次前向**取值。'
                             '每个解码步跑 2~3 次前向（全模态 / 无视频 / 无音频），'
                             'forward hook 每次都触发、后写覆盖前写，而所有读取都在三次之后 ⇒ '
                             '"last"（默认 = HEAD 行为，逐字节可复现历史结果）实际拿到的是'
                             '**盲态**那次前向的层内表示；'
                             '"full" 只让全模态那次写主字典（修掉污染，会改变解码轨迹）；'
                             '"shadow" 解码行为与 last 逐字节相同，但**同时**把全模态那版'
                             '记进影子字典 —— 配 --dump_layer_audit 就能在同一批候选上做'
                             '配对比较，不必跑两臂 A/B（省一半卡时且没有轨迹分叉的混淆）。'
                             '这不是调参，是修一个会让两个模块读错输入的实现 bug。')
    parser.add_argument('--auditor_diag', action='store_true',
                        help='配 --auditor_scope full 用：逐步打印「全模态 vs 盲态」的层内表示'
                             '差异（top-1 是否改变、KL、h_final 余弦），用来量化污染幅度。'
                             '零额外前向 —— 两版本来都算出来了，以前只是被覆盖掉。')
    parser.add_argument('--dump_layer_audit', action='store_true',
                        help='配 --auditor_scope shadow 用：把每个 token 的层内审计**原始三元组**'
                             '（res_16_22 / res_22_fin / l22_conf）连同 token_id，'
                             '**盲态版（HEAD 实际用的）和全模态版（修好的）各记一份**，'
                             '写进 TRAJ 的 candidates[].layer_audit。'
                             '这样 calculate_hallu_penalty 的四个阈值'
                             '（res_22_fin / l22_conf_late / res_16_22 / l22_conf_hijack）'
                             '就能**全部离线重标**：那个函数是三元组的纯函数，'
                             '重算罚分→重算 veto/advantage→重算选路，一次 run 扫完整个阈值网格。'
                             '以前每换一个阈值要烧一次卡，这是本仓验证成本最大的一项。')
    parser.add_argument('--path_kappa_dist', type=float, default=0.0,
                        help='Dist→Path 通道强度（默认 0 = 关闭，复现 HEAD）。'
                             'HEAD 里 Dist-Confess 的信号只改当前 token 就消失，'
                             '从不进入路径打分 —— 三条模块间链路里这一条原本不存在。'
                             '打开后：同一棵树内标准化「每 token 先验陷阱深度」'
                             '(Σ min(−MDI,3) / 输出长度)，与 Layer 罚分并列作第二惩罚项。'
                             '推荐先用 0.5 与 --path_kappa 对等。')
    parser.add_argument('--path_kappa_grnd', type=float, default=0.0,
                        help='连续证据支持率 → Path 的**奖励**通道（默认 0 = 关闭，复现 HEAD）。'
                             '要修的病灶：mdi_av 每个 token 都算了（盲态前向每步都跑，零额外'
                             '开销），却只保留「是否 < mdi_threshold」这 1 bit，且只累加负半边 ⇒ '
                             '「证据支持」这一半证据被整个丢掉，实测 795 条 cmm 里 560 条的审计'
                             '信号在换探针前后一个比特都没动。'
                             '打开后改用逐 token 对数似然比 g_t = log p_full − log p_blind'
                             '（对称截断 ±6 nat），路径分 G_k = mean_t g_t，在**同一步的兄弟'
                             '候选内** z 标准化后作为加分项：adv += κ_g·z(G_k)。'
                             '三点针对性：(1) 带符号 ⇒ 有依据的路径得正分，不再与「什么都没'
                             '触发」同分；(2) 步内 z ⇒ 永远产生排序，且尺度自由，不需要为每个'
                             '数据集调阈值（全局常数对付尺度差 3 倍的分布是已记录的病灶）；'
                             '(3) 只加分不否决 —— 删路径已被实测证明净负，头寸在选路不在删路。'
                             'blind 分支：多模态=屏蔽感官，纯文本=屏蔽 passage（Passage-MDI），'
                             '所以这个量在所有数据集上都有定义。需配 --path_score graded。')
    parser.add_argument('--path_grnd_field', type=str, default='all',
                        choices=['all', 'content', 'neg', 'late22', 'late16',
                                 'late22c', 'lp22', 'lp16'],
                        help="κ_g 走哪条通道。掩码对比（需要盲态前向）："
                             "'all'=全部 token 的 g_t=log p_full−log p_blind；"
                             "'content'=只算内容词（与 calculate_hallu_penalty 同一过滤口径）；"
                             "'neg'=只累加 min(g,0)，即旧 dist_ev 单边口径的对数版（对照用）。"
                             "层间注入（**不需要掩码**）：'late22'/'late16'="
                             "d_t=log p_final(y_t)−log p_{22,16}(y_t)，"
                             "d 大 = 该词中层无支持、末层硬加上来 = 语言先验注入；"
                             "'late22c'=同 late22 但只算内容词。"
                             "层间通道要修的是 res_22_fin=‖p₂₂−p_final‖₂ 这个**分布级**距离"
                             "不看选了哪个词 —— 选中词被末层抬起 vs 被压下，方向相反却同分。"
                             "所有通道的充分统计量都写进 TRAJ ⇒ 选哪条可以离线扫完再定。")
    parser.add_argument('--path_grnd_agg', type=str, default='sqrt',
                        choices=['sum', 'mean', 'sqrt', 'min', 'worst3'],
                        help="G_k 的归一化/聚合方式。'sum'=Σ（路径的对数贝叶斯因子本身，"
                             "长路径天然占优）；'mean'=Σ/n（长度无关，但把长路径的信号"
                             "除小了）；'sqrt'=Σ/√n（信号 ~ n·μ、噪声 ~ σ√n ⇒ 信噪比最高，"
                             "默认）；'min'=极值（幻觉是极值风险：一个无依据断言就毁掉整条"
                             "回答，均值会被其余正常词稀释）；'worst3'=最差 3 个的均值"
                             "（min 的稳健版；层间通道的 d_max 实测 100% 顶到夹断，"
                             "单点极值排不出序，必须用 worst-3）。同一步的兄弟候选长度并不相同，"
                             "所以这个选择会直接改选路结果。")
    parser.add_argument('--layer_repair_project', type=str, default='perp',
                        choices=['perp', 'full'],
                        help="Layer-Confess 修复量的投影方式。'perp'=原式 P_perp（Prop. 2 保范数，"
                             "但剥掉了沿 h_final 的分量 —— 那正是压制先验方向所需的一份，"
                             "实测位移仅 2.6 度、norm_ratio 1.001）；"
                             "'full'=保留平行分量，可真正削弱最终层的先验预测，代价是不再保范数、"
                             "Proposition 2 对它不成立。")
    parser.add_argument('--layer_diag', action='store_true',
                        help='每次 Layer 修复打一条诊断：把修复前后的 h 各过一遍 lm_head，'
                             '记录 argmax 是否翻转，并在 λ∈{0.3,1,2,5,10} 上扫描翻转阈值。'
                             'lm_head 是现成的、只是几个 d 维向量的矩阵乘 ⇒ **零额外前向**。'
                             '这是判定"修复到底咬不咬得动输出"的唯一硬指标 —— '
                             'norm_ratio 1.001 既不能证明它无效也不能证明它有效。')
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
                        help='喂给处理器的音频采样率。**默认 24000 是历史值（有 bug）**，'
                             '保留它是为了让已落盘的 cmm/avhbench 结果可复现。'
                             '模型的 WhisperFeatureExtractor 配的是 16000，且我们从不传 '
                             'sampling_rate= 给 processor ⇒ 24000 会让 9.73s 的音频被当成 '
                             '14.60s 编码（放慢 1.5×、音高降约 7 个半音）。传 16000 修正。')
    parser.add_argument('--use_audio_in_video', action='store_true',
                        help='把音频与视频按 Qwen 官方 TMRoPE 方式**时间交错**排布（每 ~2s 一块），'
                             '而不是默认的"先一整段视频、再一整段音频"。token 总数完全一样（实测 '
                             '599 vs 599），差别只在模型能不能知道某个声音发生在哪一帧。'
                             '**默认 off ⇒ 所有历史 run 逐字节不变。** 打开会同时改：聊天模板'
                             '（不再单独放音频占位符）、7 处 processor、2 处 generate 的 TMRoPE、'
                             '以及 get_modality_indices 的取法（交错后 bos/eos 区间会互相包含）。')

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
                        help='λ 边缘化：逗号分隔的修复强度网格（如 "0,0.5,2,5,10"）。给定时，'
                             'Layer 不再用单个 --lambda_layer 重投影，而是把 λ_L 当讨厌参数积掉：'
                             'p(v)=Σ_k (1/K)·softmax(W_LM LN(h_f+λ_k δ))。动机是 Q3 电池证明'
                             '单挑任何一个 λ 都是掷硬币（173 次改判胜率 47.4%），而序列级 λ 投票'
                             '离线有 +4.22 Yes类F1（CI [+0.10,+8.69]）—— 起作用的是跨强度共识，'
                             '不是某个强度。lm_head 是现成的 ⇒ 零额外前向。默认 off。')

    _a = parser.parse_args()

    # ── 参数一致性校验 ──────────────────────────────────────────────────────
    # graded / graded_sym 是在整棵树生成完之后重算 advantage 的，需要每条路径的
    # **真实 logprob**。而 veto 一旦触发就把 logprob 也写成 -10000（见 veto 判据处），
    # 原值不可恢复 ⇒ 两者同时开会让分级模式只看到部分路径，静默地退化。
    if getattr(_a, 'path_score', 'head') != 'head':
        _vt = getattr(_a, 'veto_threshold', None)
        if _vt is None or _vt < 1e8:
            parser.error(
                f"--path_score {_a.path_score} 需要同时传 --veto_threshold 1e9 关掉一票否决"
                f"（当前 {_vt}）。理由：veto 会把被否决路径的 logprob 覆写成 -10000，"
                f"分级重算拿不到原值。")
    # λ 抬上去而不设 plausibility 约束，会奖励盲态概率趋零的尾部词元
    # （实测 <|endoftext|> 164 次、<|im_end|> 128 次被顶成替换词）。
    if getattr(_a, 'cd_contrast_lambda', None) is not None \
            and _a.cd_contrast_lambda > 0.5 and getattr(_a, 'cd_plausibility_beta', 0.0) <= 0.0:
        parser.error(
            f"--cd_contrast_lambda {_a.cd_contrast_lambda} > 0.5 必须配 --cd_plausibility_beta "
            f"（建议 0.1）。理由：对比项 −λ·blind_logits 对盲态概率最低的词元奖励最大，"
            f"没有 plausibility 截断时会把特殊符/标点顶进候选池。")
    # Dist→Path 只在分级打分里消费，head 模式下这个系数没有任何入口。
    if getattr(_a, 'path_kappa_dist', 0.0) > 0.0 \
            and getattr(_a, 'path_score', 'head') == 'head':
        parser.error(
            f"--path_kappa_dist {_a.path_kappa_dist} 需要同时传 --path_score graded 或 graded_sym。"
            f"head 模式的 advantage 由一票否决决定，没有消费 Dist 证据的位置。")
    # 证据支持率同理：只有分级打分有消费它的位置。
    if getattr(_a, 'path_kappa_grnd', 0.0) > 0.0 \
            and getattr(_a, 'path_score', 'head') == 'head':
        parser.error(
            f"--path_kappa_grnd {_a.path_kappa_grnd} 需要同时传 --path_score graded 或 graded_sym。"
            f"head 模式的 advantage 由一票否决决定，没有加分项的位置。")
    return _a


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
        self.judge_prompt_version = getattr(args, 'judge_prompt_version', 'v0')
        self.model = None
        self.processor = None  # 增加处理器
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
        """--layer_lambda_grid 解析并缓存；未给或只有一个值时返回 None（走单 λ 老路）。"""
        cached = getattr(self, '_llg_cache', 'unset')
        if cached != 'unset':
            return cached
        raw = getattr(self.args, 'layer_lambda_grid', None)
        grid = None
        if raw:
            g = [float(x) for x in str(raw).split(',') if str(x).strip()]
            if len(g) > 1:
                grid = g
                print(f"    ℹ️ [Layer-λ边缘化] 网格 {g}（{len(g)} 次 lm_head / 修复点，"
                      f"零额外前向）", flush=True)
            elif len(g) == 1:
                print(f"    ⚠️ [Layer-λ边缘化] 网格只有一个值 {g}，退回单 λ 路径", flush=True)
        self._llg_cache = grid
        return grid

    def task_profile(self, system_prompt):
        """从可测的输入属性推导本任务该用哪条 CD / 长度分支。

        属性只有两个，都是任务规格的公开信息（不是测试集标签）：

          answer_space : 'binary' 当且仅当 system prompt 要求 \\boxed{Yes}/\\boxed{No}
                         这类闭合的二元判决；否则 'open'。
          ref_len      : --ref_len_words，该任务参考答案的中位词数。

        为什么二元答案要单独处理：CD 修正在 hard 模式下把被标记的 token 概率直接
        置零再重采样。答案空间只有 {Yes, No} 时，"封禁 Yes" 数学上等价于"强制输出
        No" —— 触发即翻转，与对比证据的强弱无关。实测后果是 cmm 上 recall(yes)
        从 92.71 掉到 60.51、recall(no) 从 46.85 涨到 80.50：偏置只是换了个方向，
        净准确率几乎没动（69.81 → 70.57）。soft + argmax 让被标记的 token 仍然可以
        胜出，只有当盲分支对比确实压过它时才改口。

        返回 dict；task_profile='off' 时返回全 None，调用方退化为 HEAD 行为。
        """
        if getattr(self.args, 'task_profile', 'off') != 'auto':
            return {'answer_space': None, 'cd_mode': None, 'cd_resample': None,
                    'len_budget': None}

        sp = system_prompt or ''
        binary = ('\\boxed{Yes}' in sp) or ('\\boxed{yes}' in sp)

        ref_len = getattr(self.args, 'ref_len_words', None)
        # 短参考答案（<15 词）才启用长度收缩；ragtruth 那种 103 词的长文不启用，
        # 否则会把方向搞反（我们在长文任务上本来就说得偏少）。
        len_budget = None
        if ref_len is not None and ref_len < 15:
            len_budget = float(ref_len)

        return {
            'answer_space': 'binary' if binary else 'open',
            # 二元答案：软抑制 + 确定性重选，避免"触发即翻转"
            'cd_mode':     'soft'   if binary else None,
            'cd_resample': 'argmax' if binary else None,
            'len_budget':  len_budget,
        }

    def initialize_model(self):
        """Initialize the Qwen2.5-Omni model and multimodal processor"""
        model_path = self._get_model_path()
        
        # 1. 加载多模态 Processor (包含 Tokenizer 和图像/音频预处理器)
        self.processor = Qwen2_5OmniProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.tokenizer = self.processor.tokenizer

        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 2. 加载模型 (使用 Transformers 架构以支持 Omni 的双分支 Thinker 逻辑)
        # 注意：不再使用 vLLM，改为原生 Transformers 以获得更稳健的多模态支持
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto", # 自动分发到多张显卡
            trust_remote_code=True
        ).eval()

        # 设置随机种子
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
        """
        为 Qwen-Omni 定制的结构化系统提示词。
        优化点：支持包含匹配、针对 CMM 二元验证任务、针对 AVH 单句描述、强化终止符和格式约束。

        `question` 只有**按题面分岔**的提示词版本才用得上（目前只有 cmm/mc27）。
        不传 ⇒ 与本方法原来的行为逐字节相同；传了但版本没登记分岔 ⇒ 也逐字节相同。
        模态由 `judge_prompts.cmm_stream_of` 确定性推出，**不看真值**，认不出就回落
        该版本的静态文本 —— 宁可不改，也不能猜错模态。
        """
        omni_identity = "You are Qwen, a virtual human developed by Alibaba Group, capable of perceiving auditory and visual inputs."

        # --- 1. 针对 CMM (Yes/No) 数据集的深度优化 ---
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

        # 默认回退
        return f"{omni_identity}\nSolve the problem step by step based on the video/audio."

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

    # ==================== 退化词元集（--cd_forbid_degenerate）====================
    def _cd_degenerate_ids(self):
        """返回**退化词元**的 id 张量：特殊符、纯标点/格式符、纯空白。

        为什么要有这个：对比式改写 `logits_f − λ·blind_logits` 对盲态概率最低的
        词元奖励最大，而盲态最低的恰恰是这些没有语义的符号 —— 盲模型在句中
        几乎不会想输出 `<|endoftext|>` 或一串星号，于是它们的对比得分被顶到很高。
        β 门控只按全上下文概率 p_full 截，截不住（格式符在 p_full 下并不低）。

        只算一次，缓存在实例上；tokenizer 全表遍历约 15 万条，耗时可忽略。
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
                # 纯标点/格式符：没有任何字母或数字
                if _core and not any(ch.isalnum() for ch in _core):
                    bad.add(int(_i))
            ids = torch.tensor(sorted(bad), dtype=torch.long, device=self.model.device)
            print(f"    ℹ️ [CD-Forbid] 退化词元 {ids.numel()} / {len(vocab)}", flush=True)
        except Exception as e:
            print(f"    ⚠️ [CD-Forbid] 构建失败，改写不做排除：{e}", flush=True)
            ids = torch.empty(0, dtype=torch.long, device=self.model.device)
        self._cd_bad_ids = ids
        return ids

    # ==================== Confess 树投票 ====================
    _TV_JUDGE_SETS = ("cmm", "phd", "pubmedqa")

    @staticmethod
    def _tv_verdict(text):
        """取文本中最后一个显式 yes/no；取不到返回 None（不强行判 No）。"""
        s = re.sub(r"\\boxed\{?", " ", str(text).lower())
        m = re.findall(r"\b(yes|no)\b", s)
        return m[-1] if m else None

    def _tv_tau(self):
        """当前数据集该用哪个 τ。--tree_vote_tau_map 命中则覆盖全局 --tree_vote_tau。"""
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
        """判断类任务上用整棵 Confess 树的二阶段共识决定最终判决。

        动机：被选中的那条 beam 只是树上的一条路径，而树整体带有系统性的 No 偏置
        （实测 cmm 上 GT=Yes 的样本树内 Yes 也只占 50.3%，GT=No 上 17.1%）。
        直接看 beam 的判决把这个偏置原样继承下来；改看树内 Yes 比例并按标定阈值
        判决，AUC 由 0.703 升到 0.803（phd 0.760→0.836，pubmedqa 0.774→0.894）。

        阈值 τ 不是 0.5，也不是在测试集上调的：按留一数据集标定（在另两个数据集上
        选，本数据集只评），τ* 落在 0.18~0.22，且 [0.10, 0.25] 是一个平台而非尖峰。

        翻转时的响应文本不凭空生成：从树内已生成的、判决与共识一致的二阶段补全里
        取 logprob 最高的一条作为证据，再补上显式判决行。实测三个数据集上需要翻转的
        329 条**全部**都能找到匹配文本，无一退化。
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
        print(f"    [TREE-VOTE] beam 判 {info['beam_verdict']}，树内 Yes {y}/{y+n}"
              f"={frac:.2f} > τ={tau} ⇒ 改判 {target}", flush=True)
        return info

    def select_response(self, responses, logprobs, advantages):
        """
        🌟 优化点 2：动态聚类权重 (Trust-Weighted Clustering)
        结合簇的"共识大小"和"内部平均优势(置信度)"，打破多数派幻觉。
        """
        if self.args.strategy == "advantage":
            # 模块消融 \Path 专用：不做路径聚类，直接按 advantage 加权采样。
            # 等价于 cluster 分支里"有效响应太少"时的既有回退路径，行为已在线上验证过，
            # 只是把它提成显式可选项。默认仍是 cluster，不影响任何已发布的 run。
            valid_indices = [idx for idx, r in enumerate(responses) if r.strip()]
            if len(valid_indices) == 0:
                weights = softmax([-adv / TEMPERATURE for adv in advantages])
                return np.random.choice(len(advantages), p=weights)
            weights = softmax([advantages[i] / TEMPERATURE for i in valid_indices])
            return np.random.choice(valid_indices, p=weights)

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

        # 树投票用：累计本样本所有前瞻步的二阶段补全 (文本, logprob)
        self._tv_pool = []

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

        # 6. 生成最终响应 (下结论阶段)
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
        _thr = self.thresholds  # calibratable constants for this run
        _tp = self.task_profile(system_prompt)  # 输入属性 → CD/长度分支（'off' 时全 None）
        pixel_values = example.get('pixel_values')
        audio_values = example.get('audio_values')
        image_values = example.get('image_values')  # 👈 直接加在这里！

        all_responses_first_stage = []
        all_logprobs_first_stage = []
        all_advantages_first_stage = []
        all_penalties_first_stage = []   # Layer→Path 分级通道：逐路径罚分（HEAD 只用了它是否非零这 1 bit）
        all_dist_ev_first_stage = []     # Dist→Path 分级通道：逐路径先验陷阱深度（每 token 均值）
        all_dist_cnt_first_stage = []    # 同上，触发次数（诊断用）
        all_token_nums_first_stage = []
        # ── 连续证据支持率 + 层间注入量（--path_kappa_grnd 的输入；默认只记录不消费）──
        # 病灶一（掩码对比）：mdi_av 每个 token 都算了（盲态前向每步都跑，零额外开销），
        #   却只留「是否 < −0.3」1 bit 且只累加负半边 ⇒ 「证据支持」这一半从未入账。
        # 病灶二（层间，不需要掩码）：res_22_fin = ‖p₂₂ − p_final‖₂ 是**分布级**距离，
        #   不看选了哪个词。两条路径可以 L2 距离相同，而一条是选中词被末层抬起、另一条
        #   是被压下 —— 方向相反、得分一样。换成逐词
        #       d_t = log p_final(y_t) − log p₂₂(y_t)
        #   才是「末层给这个词加了多少」＝语言先验注入量。lm_head 是现成的 ⇒ 零额外前向。
        #
        # 存**原始和 + 计数 + 极值**，不存均值。理由：
        #   Σg 是路径的对数贝叶斯因子；信号 ~ n·μ、噪声 ~ σ√n ⇒ **Σg/√n 信噪比最高**，
        #   Σg/n（均值）反而把长路径的信号除小了，Σg 不除则让长路径天然占优。
        #   同一步的兄弟候选长度并不相同 ⇒ 归一化方式直接改选路结果。存原始量后
        #   三种归一化全部离线可扫，不用为每种烧一次卡。
        #   同理存 min：幻觉是**极值风险**（一个无依据断言就毁掉整条回答），
        #   均值是错的聚合子。
        all_grnd_stats_first_stage = []   # 每候选一个 dict（键见下面的累加处）
        _probe_feats_step = []           # 与 all_responses_first_stage 同序的探针特征
        _aud_rows_step = []              # 同序的层内审计三元组（--dump_layer_audit）
        
        # ── 候选级探针特征 dump（--dump_probe_features）──────────────────
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
        # 影子记账只在 scope='shadow' 下有意义（'last' 时影子字典是空的）
        _aud_dump = bool(getattr(self.args, 'dump_layer_audit', False)) and _aud_scope == 'shadow'
        if getattr(self.args, 'auditor_diag', False) and _aud_scope == 'full':
            _need_hidden = True     # 诊断要比 h_final 的全模态/盲态两版，得存隐状态
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
            
            # 🌟 动态组装 kwargs，仅对 haloquest 限制最大像素，拯救显存！
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
            # 🚀 强化版特征拦截器
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
                    # 🌟 关键修改：只拿 position_ids，坚决放弃内部被污染的 attention_mask！
                    captured['position_ids'] = kwargs.get('position_ids', None)
                    raise StopExecution("Captured")

            h1 = self.model.thinker.register_forward_pre_hook(universal_hook, with_kwargs=True)
            h2 = self.model.thinker.model.register_forward_pre_hook(universal_hook, with_kwargs=True)
            
            try:
                with torch.inference_mode():
                    self.model.generate(**m_inputs, max_new_tokens=1, use_cache=False,
                                        **_aiv_kw_for(m_inputs))
            except StopExecution:
                print(f"✅ [DEBUG] 特征与位置编码捕获成功！", flush=True)
            except Exception as e:
                print(f"❌ [DEBUG] 拦截出错: {e}", flush=True)
            finally:
                h1.remove()
                h2.remove()

            inputs_embeds_f = captured.get('inputs_embeds')
            if inputs_embeds_f is None:
                inputs_embeds_f = self.model.thinker.get_input_embeddings()(m_inputs.input_ids)
            
            # 🌟 核心修复 1：强制使用外层最纯净的 2D 掩码
            attn_mask_f = m_inputs.attention_mask.clone()
            pos_ids_f = captured.get('position_ids')

            # =====================================================================
            # 🚀 审计线屏蔽 (支持 --mdi_mask_mode: all/audio_only/video_only/sequential)
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
            import time as _time
            _t_rollout_start = _time.perf_counter()
            for rollout_idx in range(self.args.num_rollout):
                torch.cuda.empty_cache()
                with torch.inference_mode():
                    p_full, p_nv = None, None
                    p_na = None  # sequential mode: no-audio KV cache
                    generated_ids_list = []
                    acc_logprob, total_hallu_penalty = 0.0, 0.0
                    # Dist→Path 通道：逐路径累积"先验陷阱"证据。
                    # HEAD 里 Dist 的信号只作用在当前 token（改一个词就丢了），
                    # 从不上升到路径层 ⇒ 三条模块间链路里这一条完全不存在。
                    total_dist_evidence = 0.0
                    # 逐路径充分统计量（见 all_grnd_stats_first_stage 的说明）
                    #   g  = log p_full − log p_blind          掩码对比，需要盲态前向
                    #   d22/d16 = log p_final − log p_{22,16}  层间注入，**不需要掩码**
                    _gs = {'g_sum': 0.0, 'g_n': 0, 'g_sq': 0.0, 'g_min': 0.0,
                           'g_neg_sum': 0.0, 'g_sum_c': 0.0, 'g_n_c': 0, 'g_min_c': 0.0,
                           'd22_sum': 0.0, 'd22_max': 0.0, 'd16_sum': 0.0,
                           'd16_max': 0.0, 'd_n': 0,
                           'd22_sum_c': 0.0, 'd_n_c': 0,
                           'g_bot': [],
                           'e22_sum': 0.0, 'e16_sum': 0.0, 'e22_max': -2.0,
                           'e16_max': -2.0, 'e22_top': []}
                    _probe_sum = None       # 候选级探针特征：out_f 隐状态沿本候选的解码步累加
                    _probe_steps = 0
                    _aud_rows = []          # 层内审计原始三元组（盲态版 h / 全模态版 f）
                    n_dist_events = 0

                    m_f_step = attn_mask_f.clone()
                    m_nv_step = m_nv.clone()
                    m_na_step = m_na.clone() if _is_sequential else None
                    pos_ids_step = pos_ids_f
                    curr_ids = None

                    for i in range(48):
                        if i == 0:
                            # 预填这一次**故意不要** hidden_states：它会返回全序列 ×29 层
                            # （多模态 prompt 几千 token ⇒ 一次 ~0.8GB 瞬时峰值），
                            # 而探针只需要「本候选自己生成的 token」，第 0 步那个 token
                            # 的隐状态少算一步对均值影响可忽略。
                            out_f = self.model.thinker(inputs_embeds=inputs_embeds_f, attention_mask=m_f_step, position_ids=pos_ids_step, use_cache=True)
                            auditor.active = False      # 下面两次是盲态前向，别覆盖主字典
                            out_nv = self.model.thinker(inputs_embeds=inputs_embeds_f, attention_mask=m_nv_step, position_ids=pos_ids_step, use_cache=True)
                            if _is_sequential:
                                out_na = self.model.thinker(inputs_embeds=inputs_embeds_f, attention_mask=m_na_step, position_ids=pos_ids_step, use_cache=True)
                            auditor.active = True
                        else:
                            out_f = self.model.thinker(input_ids=curr_ids, attention_mask=m_f_step, position_ids=pos_ids_step, past_key_values=p_full, use_cache=True, output_hidden_states=_probe_on)
                            auditor.active = False      # 同上
                            out_nv = self.model.thinker(input_ids=curr_ids, attention_mask=m_nv_step, position_ids=pos_ids_step, past_key_values=p_nv, use_cache=True)
                            if _is_sequential:
                                out_na = self.model.thinker(input_ids=curr_ids, attention_mask=m_na_step, position_ids=pos_ids_step, past_key_values=p_na, use_cache=True)
                            auditor.active = True

                        logits_f = out_f.logits[:, -1, :]
                        # ── 探针特征累加（--dump_probe_features）─────────────
                        # 必须从 out_f（全模态那次前向）取。本步共 2~3 次前向，
                        # auditor 的钩子在每一次都触发，intermediate_hidden 里
                        # 最终留下的是**最后一次（盲态）**前向的隐状态 ——
                        # 拿它当特征会把"屏蔽了视觉/音频之后的表示"当成模型的判断依据。
                        # hidden_states[0] 是 embedding 输出，block l 的输出在 [l+1]。
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

                        # ── 污染幅度诊断（--auditor_scope full --auditor_diag）──────────
                        # 打印「审计本该看的（全模态）」和「HEAD 实际看到的（盲态）」
                        # 差多少。零额外前向：两版都已算出，以前只是被覆盖掉。
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

                        # ── 连续审计量：两条互相正交的通道，都是零额外前向 ──────────
                        # (a) 掩码对比 g_t = log p_full − log p_blind
                        #     与 mdi_av 用的是同一对概率，区别只在**怎么记**：
                        #       mdi_av = (p_f−p_b)/p_f → 正向饱和在 1、负向无界，必须截 3.0
                        #       g_t    = log(p_f/p_b)  → 对称、可加，Σ 就是序列级对数似然比
                        #     截断也改成**对称** ±6 nat：旧口径只截负半边，等于系统性地
                        #     把「证据支持」这一半压扁。
                        # (b) 层间注入 d_t = log p_final(y_t) − log p_{22,16}(y_t)
                        #     不需要掩码。d 大 = 这个词中层没支持、末层硬加上来 = 语言先验。
                        #     与 (a) 问的是不同问题：(a)「证据支不支持」、(b)「是不是凭空长的」。
                        _pf_s = max(float(prob_f_val), 1e-12)
                        _pb_s = max(float(prob_nv_val), 1e-12)
                        _g_t = max(-6.0, min(6.0, math.log(_pf_s / _pb_s)))
                        _gs['g_sum'] += _g_t
                        _gs['g_sq'] += _g_t * _g_t
                        _gs['g_n'] += 1
                        _gs['g_min'] = min(_gs['g_min'], _g_t)
                        _gs['g_neg_sum'] += min(_g_t, 0.0)
                        # 最差 3 个：单个 token 的 min 太靠运气，worst-k 才是幻觉的
                        # 「极值风险」的稳健写法（一句没根据的话就毁掉整个回答）
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
                            # 夹断放宽到 ±20（原 ±6 是照抄 g 的，抄错了：层间注入
                            # 的量级本来就比掩码对比大一个数量级）。但放宽也没救 ——
                            # 复验实测 21/24 个候选的 d22_max 仍顶在 20.0：
                            # log(p_final/p22) 在中层给该 token ~0 概率时是**无界的**，
                            # 任何夹断值都会被顶穿。所以对数比只能用于 sum/sqrt 这类
                            # 累计聚合，**极值聚合必须换一个有界量**（见下面 e22）。
                            _d22 = max(-20.0, min(20.0, math.log(_pf_s / max(_p22_t, 1e-12))))
                            _d16 = max(-20.0, min(20.0, math.log(_pf_s / max(_p16_t, 1e-12))))
                            _gs['d22_sum'] += _d22
                            _gs['d16_sum'] += _d16
                            _gs['d22_max'] = max(_gs['d22_max'], _d22)
                            _gs['d16_max'] = max(_gs['d16_max'], _d16)
                            # 有界版层间注入：e_t = p_final(y) − p_22(y) ∈ [−1,1]。
                            # 不会发散，所以 max / worst-3 是有定义、能排序的。
                            # 语义同向：e 大 = 末层把这个词硬顶上来、中层没支持。
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
                            pass    # 钩子没抓到中间层时静默跳过，不影响主路径
                        # ─────────────────────────────────────────────────────────────

                        # 🌟 触发条件：MDI 极低（幻觉嫌疑）且不是短词
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
                                    _upd_grid = _upd          # 给 λ 边缘化用（见下方重投影处）
                                    # ── 决定性诊断：修复到底改变输出词元了吗 ──────────────
                                    # 范数只动 0.1%（实测 norm_ratio 1.001）不代表 argmax 会变，
                                    # 反过来也一样 —— 唯一能定性的是把修复前后的 h 各过一遍
                                    # lm_head 看 argmax 翻没翻。lm_head 是现成的，
                                    # 一次 λ 扫描只是几个 d 维向量的矩阵乘 ⇒ **零额外前向**，
                                    # 一次 run 就能回答"要多大 λ、要不要平行分量才咬得动"。
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
                                        # 原来这里落到 `h_repaired = h_final` 的 fallback，
                                        # 即 `--layer_repair linear` 是个**静默空操作**
                                        # （而 --layer_fix_mode 的默认值恰恰就是 'linear'）。
                                        # 补成名副其实的线性插值：朝证据锚点走 alpha_l，不保范数。
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
                                # slerp 在 θ→π 时 sin(θ)→0，除法会出 inf/nan；上面只挡了
                                # θ→0 那一头。非有限的 h 会一路带到 softmax，最终在
                                # multinomial 上炸成 device-side assert，所以在这里就拦掉。
                                if not bool(torch.isfinite(h_repaired).all()):
                                    h_repaired = h_final
                                else:
                                    # ── λ 边缘化（--layer_lambda_grid，默认 off）───────────
                                    # Q3 电池的定论：λ 从 0.5 开到 10，token 翻转率 11%→75%
                                    # 都在咬，但 173 次判决改动合并胜率 47.4%（CI 上界 54.8%）
                                    # —— **单挑任何一个 λ 都是掷硬币**。原因是 δ 的方向由几何
                                    # 决定，不含正确性估计，所以"该走多远"这个问题没有正确答案。
                                    #
                                    # 既然 λ_L 是个没法定值的讨厌参数，就别赌它，把它积掉：
                                    #     p(v) = Σ_k w_k · softmax(W_LM LN(h_f + λ_k δ))
                                    # 混合分布保留了各强度都认可的词元、压掉只有单一强度才捧
                                    # 起来的词元 —— 这正是序列级 λ 投票（离线实测 +4.22 Yes类F1，
                                    # bootstrap 95% CI [+0.10, +8.69]）起作用的机制，但代价从
                                    # 5× 推理降到 len(grid) 次 lm_head（几个 d 维矩阵乘）。
                                    #
                                    # ⚠️ 混合是在**概率**层做的，logits_f 拿到的是 log p，
                                    # 不再与原 logits 同尺度（少了 logsumexp 那个常数）。
                                    # 下游只用它做 softmax / argmax / 排序，全是平移不变的，
                                    # 所以安全；但**别**再拿它和 blind_logits 做差之外的绝对比较。
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
                                        # λ_L=0（或 h 被上面的非有限护栏退回 h_final）时，
                                        # h_repaired 与 h_final 逐比特相同，本该是空操作。
                                        # 但这里若仍重算一遍 lm_head，得到的 logits 与模型
                                        # 自己前向产出的 logits_f 在数值上不完全一致（kernel/
                                        # dtype 路径不同），采样一步之差就会让整条轨迹偏离 ——
                                        # 于是「λ=0 = Layer 关掉」这个参照臂根本不成立。
                                        # 空操作就真的什么都不做。（#37）
                                        if bool(torch.equal(h_repaired, h_final)):
                                            pass
                                        else:
                                            with torch.no_grad():
                                                _normed = self.model.thinker.model.norm(h_repaired.unsqueeze(0))
                                                logits_f = self.model.thinker.lm_head(_normed).squeeze(0)
                                            _layer_acted = True

                        if _dist_flag:
                            # ── Dist→Path 通道（累积端）──────────────────────────
                            # mdi_av = (p_full − p_blind)/p_full，越负 = 这个词越是
                            # "不看图/不听音反而更想说" ⇒ 先验陷阱越深。
                            # 截到 3.0：p_blind/p_full 可以爆到几十倍，不截会让单个
                            # 极端 token 支配整条路径的证据值。
                            total_dist_evidence += min(-float(mdi_av), 3.0)
                            n_dist_events += 1
                            # ────────────────────────────────────────────────────
                            _seq_info = f", dominant={_seq_dominant}" if _is_sequential else ""
                            print(f"    💡 [Self-Confess] 发现先验陷阱: '{token_str}' (MDI: {mdi_av:.2f}{_seq_info})，启动【对比探索模式】", flush=True)

                            # 【黑科技公式】：强行放大那些"只有看了图/听了音才产生的特征"，压制"瞎编特征"
                            alpha = getattr(self.args, 'cd_alpha', 0.5)  # 引导强度
                            # Sequential mode: use the dominant modality's blind branch
                            if _is_sequential and _seq_dominant == 'audio':
                                blind_logits = out_na.logits[:, -1, :]
                            else:
                                blind_logits = out_nv.logits[:, -1, :]
                            refined_logits = logits_f + alpha * (logits_f - blind_logits)

                            # λ 形式：refined = logits_f − λ·blind_logits（标准 CD 目标）。
                            # HEAD 的公式排序上等价于 λ=α/(1+α)=1/3，即感官对比只有先验的 1/3 权重，
                            # 且 logits_f 前的 1+α 系数把等效温度压到 0.6/1.5=0.4 —— 比第一次探测采样
                            # （温度 0.6）更尖锐地扑向全上下文模型自己的最优词。这里把两者都还原成显式旋钮。
                            _lam = getattr(self.args, 'cd_contrast_lambda', None)
                            if _lam is not None:
                                refined_logits = logits_f - _lam * blind_logits

                            # ── 退化词元硬排除（--cd_forbid_degenerate，默认 off）──────
                            # −λ·blind_logits 对**盲态概率最低**的词元奖励最大，而盲态最低的
                            # 恰恰是特殊符、纯标点和空白 —— 盲模型在句中不会想输出这些。
                            # β 门控按 p_full 截，截不掉它们（全上下文模型对格式符也有概率）。
                            # 实测（Q 电池 a3，2827 次改写）：实词净流失 498 次，
                            # 改成纯标点 194 / 空白 44 / 特殊符 53（endoftext 40）。
                            if getattr(self.args, 'cd_forbid_degenerate', False):
                                _bad = self._cd_degenerate_ids()
                                if _bad is not None and _bad.numel():
                                    refined_logits = refined_logits.index_fill(
                                        -1, _bad, float('-inf'))

                            # Adaptive plausibility constraint（β>0 时启用）。
                            # −λ·blind_logits 这一项对**盲态概率最低**的词元奖励最大，
                            # 而盲态最低的恰恰是尾部词元和特殊符（盲模型在句中不会想输出 EOS），
                            # 所以不设约束时对比放大会把它们顶进候选池。CD/DoLa/VCD 三家
                            # 都先按 p_full 截一刀再做对比，这里补上。λ 调大时这是必需的安全栏。
                            _beta = getattr(self.args, 'cd_plausibility_beta', 0.0)
                            _refined_premask = refined_logits
                            if _beta > 0.0:
                                _p_full = torch.softmax(logits_f, dim=-1)
                                _cut = _beta * _p_full.max()
                                refined_logits = refined_logits.masked_fill(_p_full < _cut, float('-inf'))

                            # 在重塑后的概率分布中重新采样
                            probs_f = torch.softmax(refined_logits / TEMPERATURE, dim=-1)

                            # ── λ×β 网格诊断（--cd_diag）────────────────────────────────
                            # logits_f 和 blind_logits 此刻都在手上，所以**整个网格是零额外前向**：
                            # 一次 run 就能离线选出 λ、β，不用每组参数各跑一遍。
                            # 每个 Dist 事件打一行 JSON，事后用脚本聚合。
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
                                        _r[0, token_id] = float('-inf')   # 与主路径一致：排除被标记词
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

                            # 为了安全，把刚才那个探测出来的幻觉词彻底屏蔽掉
                            # task_profile=auto 且答案空间二元时降级为 soft：硬封禁在
                            # {Yes, No} 上等价于强制翻转（见 self.task_profile 的说明）。
                            _eff_cd_mode = _tp.get('cd_mode') or getattr(self.args, 'cd_mode', 'hard')
                            if _eff_cd_mode == 'soft':
                                _sf = getattr(self.args, 'cd_soft_factor', 0.1)
                                probs_f[0, token_id] *= _sf
                                print(f"    🔧 [CD-Mode=soft] token '{token_str}' scaled ×{_sf}", flush=True)
                            else:
                                probs_f[0, token_id] = 0.0
                            # ── 空分布护栏（2026-08-08）───────────────────────────
                            # β 门控先把可信集截出来，被标记词再清零。如果可信集**只剩
                            # 被标记词自己**，清零后整个张量全 0，除以 0 得到 nan，
                            # multinomial 就在 GPU 上炸成 device-side assert
                            # （Q2 的 b3 = Layer 论文原口径 gated on 第 1 条就是这么死的：
                            #  层修复把分布变尖，β=0.1 之后可信集塌成单点）。
                            # 退路分两级：先摘掉 β 门控重来，还不行就整个放弃改写 ——
                            # 保留第一次探测出的词，而不是从一个可能已经含 inf/nan 的
                            # 分布里硬抽（那样第三级退路自己也会被穿透）。
                            def _degenerate(_d):
                                return (not bool(torch.isfinite(_d).all())) or float(_d.max()) <= 1e-12
                            _den = probs_f.sum(dim=-1, keepdim=True)
                            if _degenerate(_den):
                                probs_f = torch.softmax(_refined_premask / TEMPERATURE, dim=-1)
                                probs_f[0, token_id] = 0.0
                                _den = probs_f.sum(dim=-1, keepdim=True)
                            _cd_abort = _degenerate(_den)

                            if _cd_abort:
                                print(f"    ⚠️ [CD-Guard] '{token_str}' 屏蔽后分布为空，放弃改写", flush=True)
                                next_id = torch.tensor([[token_id]], device=self.model.device)
                                pf_val = prob_f_val
                                pnv_val = prob_nv_val
                            else:
                                probs_f = probs_f / _den
                                # 进行二次采样 (真正的决定)
                                _eff_resample = _tp.get('cd_resample') or getattr(self.args, 'cd_resample', 'sample')
                                if _eff_resample == 'argmax':
                                    next_id = torch.argmax(probs_f, dim=-1, keepdim=True)
                                else:
                                    next_id = torch.multinomial(probs_f, num_samples=1)
                                token_id = next_id.item()

                                # 🌟 [必须更新] 更新概率值，供后续的 Penalty 审计和 Logprob 累加使用
                                pf_val = torch.softmax(logits_f, dim=-1)[0, token_id].item()
                                pnv_val = torch.softmax(out_nv.logits[:, -1, :] / TEMPERATURE, dim=-1)[0, token_id].item()  # P0: fix temperature mismatch
                            mdi_av = (pf_val - pnv_val) / (pf_val + 1e-9)
                            print(f"    🔄 [CD-Correction] 探索出更具感官依据的词: '{self.tokenizer.decode([token_id]).strip()}'", flush=True)
                            
                        else:
                            if _layer_acted and not _gated:
                                # ── Layer 的独立动作面（--layer_repair_gated off）─────────
                                # 2026-08-07 实测发现的真 bug：修复后的 logits_f 在 `_dist_flag`
                                # 分支之外**没有任何消费者** —— probs_f 只在上面那个 if 里赋值。
                                # 于是 `--layer_repair paper --layer_repair_gated off
                                # --disable_dist` 跑出 18777 个修复点，输出却与基线逐字节
                                # 20/20 相同：`gated off` 只解除了「算不算」的门，没解除
                                # 「生不生效」的门。Layer 因此没有独立动作面，三模块消融表里
                                # Layer 那一行不可能独立于 Dist。
                                # 这里补上：Dist 没标记但修复确实发生的步，从修复后的分布重采样。
                                # 只在非默认配置下可达（layer_repair 默认 none、gated 默认 on），
                                # 所以既有 run 的行为逐字节不变。
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
                                # 正常情况：直接沿用第一次探测的采样结果
                                next_id = torch.tensor([[token_id]], device=self.model.device)
                                token_id = next_id.item()
                                pf_val = prob_f_val    # 直接使用第一轮获取的概率
                                pnv_val = prob_nv_val  # 同步一下盲态概率
                        # 4. 继续原来的逻辑
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

                        # ── 影子记账（--auditor_scope shadow --dump_layer_audit）────────
                        # 同一个 token、同一条轨迹，把「修好之后」的三元组也算一份。
                        # 全模态的层内 logits 本来就已经算出来了（钩子在 out_f 上也触发过），
                        # 以前被盲态那次覆盖掉 ⇒ 这里零额外前向。
                        # 记原始三元组而不是罚分：calculate_hallu_penalty 是它的纯函数，
                        # 四个阈值全部可以离线重标，不用为每个阈值再烧一次卡。
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
                        
                        # 更新 Attention Mask (正常添加 2D 维度的 1)
                        new_bit = torch.ones((1, 1), device=self.model.device, dtype=m_f_step.dtype)
                        m_f_step = torch.cat([m_f_step, new_bit], dim=-1)
                        m_nv_step = torch.cat([m_nv_step, new_bit], dim=-1)
                        if _is_sequential and m_na_step is not None:
                            m_na_step = torch.cat([m_na_step, new_bit], dim=-1)
                        
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
                    # 阈值设为 1.0（意味着触发了严重层间冲突 LATE 或多次 HIJACK）
                    _pcs = traj_info['path_confess_stats']
                    _pcs['num_paths_explored'] += 1
                    _pcs['max_penalty_seen'] = max(_pcs['max_penalty_seen'], float(total_hallu_penalty))
                    if getattr(self.args, 'disable_path', False):
                        advantage = logprob          # 纯 logprob，不否决也不减罚分
                    elif total_hallu_penalty >= _thr['veto_threshold']:
                        # 赋值极小值，而非 -inf，防止后续 np.mean() 崩溃返回 NaN
                        advantage = -10000.0
                        logprob = -10000.0
                        _pcs['num_paths_vetoed'] += 1
                        print(f"    🚨 [LETHAL PENALTY] 触发一票否决! 幻觉分数: {total_hallu_penalty:.2f}。路径被强行抹杀。", flush=True)
                    else:
                        advantage = logprob - (total_hallu_penalty * 0.8)

                    all_responses_first_stage.append(response)
                    all_logprobs_first_stage.append(logprob)
                    all_advantages_first_stage.append(advantage)
                    all_penalties_first_stage.append(float(total_hallu_penalty))
                    # Dist→Path：用「每 token 的先验陷阱深度」而非总量，避免长路径天然吃亏
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
                        # 该候选**自己 token** 的隐状态均值（trajectory mean-pool）。
                        # 不用最后一个 token：那会把不同长度候选的位置效应混进特征。
                        _probe_feats_step.append(
                            (_probe_sum / max(_probe_steps, 1)).cpu().numpy().astype('float16')
                            if _probe_sum is not None else None)
                    token_stats["output"] += len(generated_ids_list)
                    rollout_stats["total"] += 1

        # ══════════════════════════════════════════════════════════════════════
        # Layer→Path 分级通道（--path_score，默认 head = 逐字节复现原行为）
        # ══════════════════════════════════════════════════════════════════════
        # HEAD 的判据：
        #     if total_hallu_penalty >= 0.45:  advantage = -10000
        #     else:                            advantage = logprob - 0.8*penalty
        # 而 calculate_hallu_penalty 的最小非零输出是 0.5（penalty_late）/ 0.6（hijack），
        # **恒 > 阈值 0.45** ⇒ else 支只在 penalty==0 时执行 ⇒ `-0.8*penalty` 是死代码。
        # 实测证实：3311 条未否决候选中 advantage==logprob 的比例 100.00%。
        # 所以 Path-Confess 从 Layer-Confess 只取到 1 bit（"罚分是否非零"），
        # 分级信息全部丢弃 —— 这正是论文声称的模块间信号传导断掉的地方。
        #
        # 另外两个已量化的病灶：
        #  · 尺度：candidate logprob 的 σ=0.398，而 0.8×0.5=0.40 ⇒ 一次 LATE 事件
        #    就把 advantage 推低 1.00σ，2 次推低 2.01σ。系数比被调制量的自然幅度大 ~4×，
        #    所以即便抬 τ 让 else 支活过来（V1 臂），它也只是从硬否决变成软否决。
        #  · 尺度跨数据集差 3 倍（ragtruth 罚分均值 1.867 vs 其他 0.56~0.69），
        #    全局常数阈值必然一边全否决（ragtruth 97%）一边永不触发（halueval 0%）。
        #
        # 修法：在**同一棵树内**把两个量各自标准化再相加，κ 的含义变成
        # "一个罚分标准差值多少个 logprob 标准差"，天然尺度无关、跨数据集可迁移。
        #     advantage_i = logprob_i − κ · σ_logprob · (penalty_i − μ_penalty) / σ_penalty
        # σ_penalty==0（整棵树罚分一样）时不做调整 —— 此时罚分对这棵树没有区分度，
        # 这也顺带让 halueval 那种从不触发的数据集自动退化为纯 logprob 排序。
        #
        # graded_sym 变体：把标准化限制在**同答案类**的兄弟路径内。
        # 依据 [[veto 是答案类偏置]]：Yes 路径被否决率高 1.7×（phd z=-5.09），
        # 且按 GT 分层四格全同向 ⇒ 罚分带的是答案类信息，不是正确性信息。
        # 类内标准化后 penalty 只能在同类里排序，永远不能改变 Yes/No 配比。
        _ps_mode = getattr(self.args, 'path_score', 'head')
        if getattr(self.args, 'disable_path', False):
            _ps_mode = 'head'    # 关 Path 时连树内标准化也不做
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
            # 被否决的路径在 head 模式下 logprob 已被写成 -10000；分级模式不否决，
            # 但那些路径的原始 logprob 已丢失，因此分级模式必须配 --veto_threshold 1e9
            # （下面会断言），保证这里拿到的都是真 logprob。
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
                    # ── Dist→Path 通道（消费端）──────────────────────────────
                    # 论文声称三个模块的信号彼此传导，但代码里 Dist 的输出只改当前
                    # token 就消失了，从不进入路径打分 ⇒ Dist→Path 这条链路原本不存在。
                    # 这里补上：同一棵树内标准化「每 token 先验陷阱深度」，
                    # 与 Layer 罚分并列作为第二个惩罚项。
                    # 语义：一条路径要么很少踩先验陷阱，要么踩得浅，才配拿高分。
                    _devs = [_dev[i] for i in _grp]
                    _mu_d = sum(_devs) / len(_devs)
                    _sd_d = (sum((d - _mu_d) ** 2 for d in _devs) / len(_devs)) ** 0.5
                    _use_d = _kappa_d > 0.0 and _sd_d >= 1e-9
                    # ── 连续证据支持率通道 ───────────────────────────────────
                    # 这一项与上面两项**方向相反**：penalty / dist_ev 是罚（越大越扣分），
                    # G_k 是奖（证据越支持这条路径越加分）。旧口径里只有罚没有奖，
                    # 所以「句句有依据」和「什么都没触发」得分相同 —— 这是感官依赖
                    # 在选路上失效的直接原因。
                    _grns = [_grn[i] for i in _grp]
                    _mu_g = sum(_grns) / len(_grns)
                    _sd_g = (sum((g - _mu_g) ** 2 for g in _grns) / len(_grns)) ** 0.5
                    _use_g = _kappa_g > 0.0 and _sd_g >= 1e-9
                    if _sd_p < 1e-9 and not _use_d and not _use_g:
                        continue   # 这棵树/这一类里两个信号都无区分度，退化为纯 logprob 排序
                    for i in _grp:
                        _adj = 0.0
                        if _sd_p >= 1e-9:
                            _adj += _kappa * (_pens[i] - _mu_p) / _sd_p
                        if _use_d:
                            _adj += _kappa_d * (_dev[i] - _mu_d) / _sd_d
                        if _use_g:
                            # 减号：_adj 整体是被扣掉的，所以「证据支持高于兄弟路径」要减小 _adj
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
            p_str = self.processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            if isinstance(p_str, list): p_str = p_str[0]
            
            # 🌟 修复 1：物理过滤掉模型在第一阶段可能误生成的视频/音频特征词，防止 Token 数量不匹配
            clean_resp = response.replace("<|video|>", "").replace("<|audio|>", "").replace("<|vision_start|>", "").replace("<|vision_end|>", "")
            inputs_text = p_str.replace(self.tokenizer.eos_token, "").strip() + previous_steps[beam_idx] + clean_resp
            
            # 🌟 修复 2：动态决定 fps，防止无视频时传 fps 导致崩溃
            # 🐞 与本文件 1414 / 2235 行保持一致：该调用点原先漏了 max_pixels，
            # 导致 rollout 中间步按原图分辨率进 ViT。ViT 全注意力显存 =
            # (4*px/784)^2 × 16头 × 2字节，原图 36.15M px 时 = 125.5 GiB，必 OOM；
            # 而 OOM 发生在 safe_model_generate 内部，前向钩子拦不到 inputs_embeds
            # → 回退成把图像占位符当纯文本 embed → 该 rollout 步**图像失明**。
            # 修法与 GUARD/_v8_fix1186/ 那份快照一致（那份已按 v8 验过），这里补进主干：
            # 上限取 2.2M px，因为 v8 中 458 条全程正常样本的最大像素是 2,117,500，
            # 故对它们逐字节无影响，只约束余下 143 条。
            # 注：dataset/haloquest/images/ 现已降采样（中位 1.05M，仅 ~3% 超 2.2M），
            # 所以这条修复今天的影响面很小，但不补上换回原图就会复发。
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
        # 树投票取证：把本前瞻步的二阶段补全并入样本级池子（与落盘 TRAJ 的
        # foresight_part[*].second_stage 内容一致，故离线重放与在线一致）。
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
                # 逐 foresight step 存一组。标签（这条候选判对了吗）不在这里算 ——
                # 离线用 texts + ground_truth 现算，免得口径写死在采集端。
                _keep = [k for k, f in enumerate(_probe_feats_step) if f is not None]
                traj_info.setdefault('_probe_feats', []).append(
                    np.stack([_probe_feats_step[k] for k in _keep]) if _keep else None)
                traj_info.setdefault('_probe_meta', []).append({
                    'layers': list(_probe_ls),
                    'rows': _keep,                       # 行 → 候选下标
                    'texts': [all_responses_first_stage[k] for k in _keep],
                    'logprob': [float(all_logprobs_first_stage[k]) for k in _keep],
                    'penalty': [float(all_penalties_first_stage[k]) for k in _keep],
                    'dist_ev': [float(all_dist_ev_first_stage[k]) for k in _keep],
                    'is_vetoed': [bool(all_advantages_first_stage[k] <= -9000.0) for k in _keep],
                })

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
                            # 逐路径罚分 / 先验陷阱深度：HEAD 只存了整棵树的
                            # max_penalty_seen 聚合值，导致 κ 这类打分系数**离线无法标定**，
                            # 每试一个值都要烧一次 GPU。存下这两列后，
                            # (logprob, penalty, dist_ev, 正确性) 四元组足以在离线复算
                            # 任意 (κ, κ_dist) 组合会选中哪条路径 —— 一次 run 扫完整个网格。
                            'penalty': round(float(all_penalties_first_stage[i]), 4),
                            'dist_ev': round(float(all_dist_ev_first_stage[i]), 5),
                            'dist_n': int(all_dist_cnt_first_stage[i]),
                            # 连续审计量的**充分统计量**（原始和 + 计数 + 极值，不是均值）。
                            # 有了它，(κ_g, field, agg) 整个网格 —— 三种归一化
                            # (Σ / Σ/n / Σ/√n) × 均值/极值 × 掩码对比/层间注入 ——
                            # 全部离线可复算，一次 run 扫完，不用为每个组合烧一次卡。
                            'grnd': {k: (round(v, 5) if isinstance(v, float) else v)
                                     for k, v in all_grnd_stats_first_stage[i].items()},
                            'ntok': int(all_token_nums_first_stage[i]),
                            # 🌟 增加 bool() 强转，解决 NumPy 布尔值无法序列化的 Bug
                            'is_vetoed': bool(all_advantages_first_stage[i] <= -9000.0),
                            # 层内审计原始三元组（--dump_layer_audit）：
                            # 每 token 一行 {tid, h:[res_16_22,res_22_fin,l22_conf] 盲态版,
                            #                     f:[…] 全模态版}。
                            # 有了它，四个阈值全部能离线重标 —— 不用为每个阈值烧一次卡。
                            **({'layer_audit': _aud_rows_step[i]}
                               if _aud_dump and i < len(_aud_rows_step) else {})
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

    def _s2_tail(self, scaf, question_text):
        """二阶段判决前缀。`scaf` 为 None 时返回主干原字面量（逐字节不变）。

        覆盖版（cmm 的 mc/mc6）会把"该看哪一路"贴在 `\\boxed{` 的正前方。
        这是全链路里离判决词元最近的一句话，而二阶段只生成 ≤20 个词元
        ⇒ 比同一条规则写在系统提示词里更有约束力。
        模态由题面**确定性**推出（`judge_prompts.cmm_stream_of`，只认 cmm 的两个
        模板、全库 795 条零例外），**不看真值**；认不出就回落原字面量 ——
        宁可不改，也不能猜错模态。
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
        # 被问项点名贴在判决前面。抽不出来（题面不合模板）就换掉 {target} 那半句，
        # 而不是整段回落 —— "该读哪张清单"这条判据与被问项能不能抽出无关。
        tgt = _cmm_target_of(question_text) if _JUDGE_PROMPTS_LOADED else None
        deciding = "Visual" if stream == "video" else "Auditory"
        if not tgt:
            from judge_prompts import _MC_S2_NOTGT
            # 抽不出被问项的回落模板也要跟着臂走：mc2 的判据措辞和 mc 不一样
            # （"说它在不在" vs "有没有提到"），写死 _MC_S2_NOTGT 会让 mc2 在这条
            # 分支上悄悄退回 mc 的口径。没登记就还用 _MC_S2_NOTGT ⇒ mc/mc6 不变。
            _ntg = (scaf or {}).get("s2_template_notgt") or _MC_S2_NOTGT
            return _ntg.format(question=question_text, stream=stream,
                               deciding=deciding)
        return tmpl.format(question=question_text, stream=stream,
                           target=tgt, deciding=deciding)

    # ══════════════════════════════════════════════════════════════════════
    # 格式控制（--format_control / --verdict_lock，默认全 off）
    # ══════════════════════════════════════════════════════════════════════
    def _format_segments(self, stage, s1_prefix=None):
        """按「阶段 × 数据集」选骨架。选不到就返回 None（只剩反抄袭那一件生效）。

        ⚠️ 骨架**必须与前缀严丝合缝**：一阶段的前缀已经把某张清单的表头写掉了
        （:3656 的默认字面量与 judge_prompts 的覆盖版都以 `"Visual Checklist:\\n- "`
        收尾），所以第一段是槽而不是字面量。

        `s1_prefix` 给出时，清单的**先后顺序由前缀读出来**（`first_from_prefix`）
        而不是写死 —— 因为前缀有三个来源（本文件的硬编码、`_STAGE_SCAFFOLD` 的
        覆盖、以后的模态条件化版本），写死就会在某一天悄悄错位一整段。
        读完还会 `assert_prefix_matches` 断言一次：错位的症状是输出里出现两个
        "Visual Checklist:"，在日志里肉眼几乎发现不了。

        为什么顺序要紧：**后写的那张清单离判决词元最近**。cmm 的陷阱按题面分流
        （see 题怕 overrely_audio、hear 题怕 overrely_visual，全库 795 条零例外），
        所以被问的那一路应当排在最后。见 `format_control.cmm_stage1_segments`。

        phd 用**不带听觉清单**的骨架：phd 是纯图像数据集，强制一段它没有输入的
        音频清单等于命令模型编 —— 那正是本项目要治的病。
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
            print(f"⚠️ [格式控制] 数据集 {ds!r} 没有对应骨架，template 不启用"
                  f"（反抄袭若已打开仍生效）", flush=True)
            self._fc_warned = True
        return None

    def _format_procs(self, stage, anti_copy_text=None, s1_prefix=None):
        """组装解码期的格式约束，返回 `(LogitsProcessorList|None, 模板处理器|None)`。

        `--format_control off` 且 `--verdict_lock` 未开 ⇒ 返回 (None, None)，
        调用点因此一个 kwarg 都不多传 ⇒ 与历史 run **逐字节相同**。

        `anti_copy_text` 只喂**指令原文**（系统提示词 + 脚手架后缀），
        **不喂树里的笔记**：实测的 8.5% 回声全部来自指令（方括号占位符、
        `CRITICAL RULE`），而清单本来就该复述笔记里的事实，把笔记也禁了会逼模型
        为了绕开 6-gram 而改写自己刚认定的证据。

        `s1_prefix` 是一阶段前缀原文，交给 `_format_segments` 读出清单顺序并断言
        对齐 —— 不给也能跑（回落到 `first="visual"`，即主干现状）。
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
        """把处理器塞进 generate 的 kwargs。两个都是 None ⇒ 空字典（调用点无感）。"""
        kw = {}
        if lp is not None:
            kw['logits_processor'] = lp
        if tf is not None:
            kw['stopping_criteria'] = _FC.build_stopping_criteria(
                self.tokenizer, None, template_proc=tf)
        return kw

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
            p_str = self.processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
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

                # [DIAG-AVH] 完整上下文诊断 print
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
            # 🚀 分支 B：HaloQuest 专用逻辑 (纯视觉简答)
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
            # 🚀 分支 C：纯文本描述/QA任务 (HaluEval, RAGTruth, PubMedQA)
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
                
                # 🌟 纯文本任务坚决不传 images/videos 参数，极大降低显存和时间消耗
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
                    
                    # 提取最终内容 (防止模型多余输出)
                    gen_answer = raw_text.split('\n\n')[0].strip()
                    # haloquest / halueval 也会自发套 \boxed{} —— 这一支原先漏了剥离，
                    # 而 `\boxed{no}` 在 Token-F1 里会退化成单词 `boxedno`（该条直接 0 分）。
                    # 必须放在下面的 yes/no 截断之前，否则首词是 `\boxed{Yes}` 匹配不上。
                    gen_answer = unwrap_boxed(gen_answer)
                    if gen_answer.startswith("[") and gen_answer.endswith("]"):
                        gen_answer = gen_answer[1:-1] # 去掉模型可能画蛇添足加的中括号
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
                if "\\boxed{" in reasoning_history:
                    base_input = p_str.replace(self.tokenizer.eos_token, "").strip() + f"\n\nBased on your detailed internal reasoning: '{reasoning_history.strip()}'\nNow, "
                else:
                    base_input = p_str.replace(self.tokenizer.eos_token, "").strip() + "\n" + reasoning_history

                # 1. 提取问题 (修复 NameError 崩溃)
                question_text = example.get('question') or example.get('input')

                # 2. 生成客观事实清单
                # 脚手架可被 judge_prompts.get_stage1_scaffold 覆盖（目前只有 cmm 的
                # mc/mc6）。覆盖不命中时 _scaf 为 None ⇒ 下面三处全部走原字面量，
                # 与历史 run **逐字节相同**（v0/fs6、phd、pubmedqa 都不受影响）。
                # 覆盖的理由见 judge_prompts.py 的 `_STAGE_SCAFFOLD` 注释：原脚手架
                # 与系统提示词的格式契约互相矛盾（一个要 "Description: 一句话"、
                # 一个按进 "Visual Checklist" 要点列表），实测 99.4% 两个都不遵守、
                # 8.5% 把提示词原文抄进答案；且它只提 "objective **visual** facts"，
                # 对 cmm 的 397 条 hear 题正好是 overrely_visual_ignore_audio 陷阱
                # （落盘成功的 388 条里 49.5% 连一行听觉清单都没有）。
                _scaf = None
                if _JUDGE_PROMPTS_LOADED:
                    _scaf = _get_stage1_scaffold(
                        self.args.datasets, getattr(self, 'judge_prompt_version', 'v0'))
                _s1_suffix = (_scaf or {}).get("s1_suffix") or (
                    "\nCRITICAL RULE: Do NOT repeat the context. You MUST use bullet points "
                    "to list the objective visual facts.\nVisual Checklist:\n- ")
                _s1_hdr = (_scaf or {}).get("header") or "Visual Checklist:\n- "
                # 前缀按题面翻转（只有 cmm 的 mc/mc6 开着 s1_dynamic）：**被问那一路
                # 的清单排第二**，紧贴判决。原来两种题面收到同一份前缀，hear 题恰好
                # 对、see 题恰好反 —— 3 月 785 条上「视觉清单里写了声音」命中 60.0%、
                # 带病准确率低 13.5pp，方向正是这个。题面认不出 ⇒ 保持上面的静态
                # 字面量，宁可不改也不猜模态。
                if (_scaf or {}).get("s1_dynamic") and _JUDGE_PROMPTS_LOADED:
                    # 第二个参数选前缀模板：mc/mc6 用 _MC_S1、mc2 用 _MC2_S1（去掉
                    # 'Judgment' 三个字）。不传就是 "mc" ⇒ 老臂逐字节不变。
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
                    # 格式控制（默认 off ⇒ _fc_s1 为 None，与历史 run 逐字节相同）
                    # 反抄袭只喂**指令原文**，不喂笔记 —— 理由见 _format_procs 的文档。
                    _fc_s1, _fc_tf1 = self._format_procs(
                        "stage1", anti_copy_text=(system_prompt or "") + "\n" + _s1_suffix,
                        s1_prefix=_s1_suffix)
                    out_s1 = safe_model_generate(
                        self.model, self.processor, m_inputs_s1,
                        max_new_tokens=200,
                        do_sample=False,
                        # 骨架强制开着时不能再用 "\n\n" 停：模板的字面量里就有换行，
                        # 会被这条停止串在第一段清单之后掐断。模板自带结束判据。
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
                # 🌟 核心修改：删除 match_history 捷径，强制要求模型进行二次审判！
                # 强迫它对比自己刚写的 Checklist 和真实的问题，解决脑口不一！
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
                    # 判决锁（--verdict_lock，默认 off ⇒ 下面两项都是 None）
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
                    # ⚠️ 这个兜底是**单边**的：解析不到就一律判 No。开 --verdict_lock
                    # 之后判决词元被锁死在 Yes/No 六个变体里 ⇒ 这一支不可达。
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
                # header 必须与本条实际用的一阶段前缀配对（前缀按题面翻转，见 :3684）；
                # 拿错就会凭空多/少一个清单表头，链子看起来像模型漏写。
                _hdr = _s1_hdr
                final_output = f"{_hdr}{clean_desc}\nJudgment: \\boxed{{{clean_judg}}}"
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

        # R5 打平规则：两条最终候选判决分歧时，交由二阶段文本的多数判决裁决。
        # ⚠️ **命令行跑的时候默认已 on**（argparse `--final_tiebreak` default='r5'，
        # 见 :779，与 :3614 的注释一致）；要复现 R5 落地前的逐字节历史结果，
        # 必须显式 `--final_tiebreak off`。原注释"默认 off"是陈旧的，已订正。
        # 下面 getattr 的兜底值**故意仍是 'off'**：`profile_guard.py:169` 手搓的
        # Namespace 里没有这个键，那是成本表（任务 #49）的计时脚本，兜底改成
        # 'r5' 会悄悄给它加上 R5 的开销、改掉已发布的成本数字。
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

        # Confess 树投票：判断类任务上，最终判决由整棵树的二阶段补全共识决定，
        # 而不是只看被选中的那一条 beam。详见 FINDINGS_0804.md §9。
        # 顺序说明：R5 只改**选哪条 beam**，树投票改**判决本身**，两者作用在不同层，
        # 因此先跑 R5 再跑树投票。R5 默认已 on（--final_tiebreak r5），
        # 树投票默认 on 但只在判断类数据集触发（见 _tree_vote）；两个都关回退到历史行为。
        final_response = all_responses[selected_idx]
        tv_info = self._tree_vote(final_response)
        if tv_info.get('applied'):
            final_response = tv_info['response']

        # ── 终选池化「有 yes 就判 yes」（--final_pool_yes_if_any，默认 off）────────
        # 二阶段留下的两条终选里，`select_response` 挑哪一条与 argmax(logprob)
        # 只有约一半重合 ⇒ 相当大比例上是在两条里随机抽。既然是抽，就换成有方向的
        # 规则：**只要有一条判 yes 就取那一条**。方向选 yes 是因为判断类上模型的
        # 系统性偏置是漏报 yes（cmm 树内 Yes 率 0.33 vs 真值 0.50）。
        # 只在判断类数据集上生效，且**只往上改**（没有 yes 时原样不动）。
        # ⚠️ 实测口径：n=80 上 +3.01，但 n=200 上塌到 +0.81（胜负 3:4）——
        # 这是个零效应规则，别拿旧的 n=80 数字当收益。
        # 与后处理脚本 GUARD/apply_pool_yes_if_any.py 判决等价（同一个 _tv_verdict）。
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

        # 记录全景轨迹
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
        """
        适配 Qwen2.5-Omni 处理器要求的结构化 Chat 模板 (已修复重复注入 Bug 与实现严格门控)
        """
        question = example.get('input') or example.get('question')
        user_content_list = []
        dataset_name = self.args.datasets.lower()

        # 1. 物理注入多模态标记 (与 First Stage 保持一致)
        if example.get('image_values') is not None:
            user_content_list.append({"type": "image"})
        if example.get('pixel_values') is not None:
            user_content_list.append({"type": "video"})
        if example.get('audio_values') is not None and not _aiv_audio_folded(example):
            user_content_list.append({"type": "audio"})

        # 2. 数据集门控 (Dataset Gating)
        if "cmm" in dataset_name:
            # 🌟 CMM 专属：保持基础设定一致，末尾引导其完成整段生成
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
            # 🌟 AVH 专属：回答问题
            user_content_list.append({
                "type": "text", 
                "text": f"Based STRICTLY on the raw visual and auditory evidence provided (do not guess or assume), answer this Question: {question}"
            })
        elif "phd" in dataset_name:
            # 🌟 PhD 专属
            user_content_list.append({
                "type": "text", 
                "text": f"Question: {question}\nPlease solve the problem step by step based on the provided media content."
            })
        elif "halo" in dataset_name:
            # 🌟 HaloQuest 专属 (Completion Stage)：保持与第一阶段完全一致的严苛防幻觉指令
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
            # 兜底
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
        """
        第一阶段采样使用的结构化模板 (已重构：多模态占位符与数据集门控彻底分离)
        """
        question = example.get('input') or example.get('question')
        user_content_list = []
        dataset_name = self.args.datasets.lower()

        # 1. 物理注入多模态标记 (解耦：只要数据有对应媒体，就自动加占位符，不依赖数据集名称)
        if example.get('image_values') is not None:
            user_content_list.append({"type": "image"})
        if example.get('pixel_values') is not None:
            user_content_list.append({"type": "video"})
        if example.get('audio_values') is not None and not _aiv_audio_folded(example):
            user_content_list.append({"type": "audio"})

        # 2. 数据集门控 (Dataset Gating)：严格按任务下发专属 Prompt
        if "cmm" in dataset_name:
            # 🌟 CMM 专属：三步独立交叉验证法
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
            # 🌟 AVH 专属：不强迫做判断，只做客观描述
            # 2026-08-04：这一句才是**管笔记内容**的指令。原先它硬编码在这里、不受
            # --avh_prompt_version 控制，导致第六波把别名规则写进 system 却到不了笔记
            # （实测模型写笔记时 system 的 18词/禁用词/括号三条全不遵守）。现改由
            # avh_prompts.get_avh_step() 下发；没有 `step` 字段的臂拿 _STEP_DEFAULT，
            # 与下面 fallback 逐字节相同，所以 v0~v22 的行为与历史读数不受影响。
            _step_txt = None
            if _AVH_PROMPTS_LOADED:
                try:
                    from avh_prompts import get_avh_step as _get_avh_step
                    _step_txt = _get_avh_step(
                        getattr(self.args, 'avh_prompt_version', 'v0'), question)
                except Exception as _e:
                    print(f"[WARN] get_avh_step 取不到，退回硬编码: {_e}", flush=True)
            if _step_txt is None:
                _step_txt = (f"Based STRICTLY on the raw visual and auditory evidence "
                             f"provided (do not guess or assume), analyze this Question: "
                             f"{question}\nProvide the NEXT reasoning step or description.")
            user_content_list.append({"type": "text", "text": _step_txt})
        elif "halo" in dataset_name:
            # 🌟 HaloQuest 专属 (Foresight Stage)：强调审计员身份，极高精度要求，防常识补全
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
            # 🌟 纯文本任务专属：提取 Passage 并引导思维链
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
            # 兜底逻辑
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
    """由样本内容而非样本位置导出种子。

    原实现只在初始化时 np.random.seed 一次（:777），此后 np.random 被各样本
    顺序消耗，于是一个样本收到的随机抽取取决于它前面所有样本消耗掉多少次抽取。
    后果是分片/子集运行与全量运行不可比：实测同代码同数据两次运行，
    cmm 上最终判决分歧 38.2%、响应文本逐字节相同率 0%。
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
    """TokenGuard 终极版：支持断点续传、异常隔离与多重路径对齐"""
    global AUDIO_SR
    global USE_AUDIO_IN_VIDEO
    args = parse_arguments()
    AUDIO_SR = int(args.audio_sr)
    print(f"🔊 音频采样率 AUDIO_SR = {AUDIO_SR} "
          + ("(✓ 与模型 WhisperFeatureExtractor 的 16000 一致)" if AUDIO_SR == 16000
             else "(❌ 历史默认，模型要的是 16000；见 load_audio_data 上方注释)"),
          flush=True)
    USE_AUDIO_IN_VIDEO = bool(getattr(args, 'use_audio_in_video', False))
    print(f"🎬 音视频排布 USE_AUDIO_IN_VIDEO = {USE_AUDIO_IN_VIDEO} "
          + ("(时间交错，每 ~2s 一块；modality 索引改走 token-id 取法)" if USE_AUDIO_IN_VIDEO
             else "(历史默认：视频一整块 + 音频一整块)"),
          flush=True)
    decoder = PhiDecoder(args)

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
            # 每样本独立播种，使随机抽取只依赖样本本身、不依赖它在本次运行中的位置。
            if not getattr(args, 'disable_per_sample_seed', False):
                _ss = _stable_sample_seed(args.seed, example)
                np.random.seed(_ss)
                torch.manual_seed(_ss)
            # P3-c: Pre-sample cache clear to prevent OOM accumulation (esp. PubMedQA)
            import gc; gc.collect()
            torch.cuda.empty_cache()
            try:
                sub_cat = example.get("sub_category", "unknown")
                print(f"\n🚀 [Progress {i+1}/{total_samples}] 正在处理审计任务 #{i} | 子类: {sub_cat}", flush=True)
                system_prompt = decoder.get_system_prompt(
                    args.datasets, question=example.get("question"))
                
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
                        "veto_threshold_used": decoder.thresholds["veto_threshold"],
                        "num_paths_explored": _pcs.get("num_paths_explored", 0),
                        "num_paths_vetoed": _n_vetoed,
                    },
                    "layer_repair": summarize_layer_repair(
                        result.get("traj_info", {})),
                }
                res_f.write(json.dumps(output_result, ensure_ascii=False) + "\n")
                res_f.flush()
                # 物理同步，防止 nohup 缓冲区导致的崩溃丢数
                os.fsync(res_f.fileno())

                # 探针特征落盘（numpy 数组，绝不进 JSON —— 必须先 pop 再 dumps）
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

                # 实时追加审计轨迹
                if args.record_process and "traj_info" in result:
                    result["traj_info"]["question_idx"] = i
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