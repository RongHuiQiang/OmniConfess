# -*- coding: utf-8 -*-
"""
freeform_repair.py — token 级修订风险 (rho_i) + span 重生成算子 (free-form / 描述类)

适用数据集: ragtruth (文本 grounding), haloquest (视觉 QA), avhbench (audio-visual)。
本算子替换旧的 contract / admission-gate / best-of-N / Vote-SC 机制,与它们无任何耦合。

============================ 数学定义 (严格照任务书) ============================
对候选回答里每个 token i:
  三分量 free-form gamma:
    gamma_i^free = ( E_i^R,
                     [ [P_i^0]_+ - [E_i^R]_+ ]_+,
                     sum_{m not in R} [ kappa_i^(m) ]_+ )
  其中
    kappa_i^(m) = token i 在 "完整证据" vs "移除模态 m" 下的分数差 (logP 差, nats)
    E_i^R       = sum_{m in R} kappa_i^(m)              相关模态证据支持之和
    P_i^0       = token i 的 "证据无关先验分量"          (见下面实现说明)
    R           = 该样本的相关模态集; I = (全部模态) - R  为无关模态集

  修订风险:
    rho_i = [ -gamma_{i,1}^free ]_+ + gamma_{i,2}^free + gamma_{i,3}^free
          = [ -E_i^R ]_+                                # 相关证据反对该 token
          + [ [P_i^0]_+ - [E_i^R]_+ ]_+                 # 先验主导超过相关证据支持
          + sum_{m in I} [ kappa_i^(m) ]_+              # 无关模态在硬撑该 token

  阈值: rho_i > tau 的 token 标记为待修订。
  Expand: 把标记 token 扩到词边界 (不切碎词)。
  Merge : 合并相邻/重叠的待修订区间成连续 span, 得到修订区间集合 B_rev。
  Regenerate: 对每个 span 用 Rev_theta (同 backbone 条件重生成) 在完整上下文下重写, Replace 回原文。

============================ P_i^0 的实现说明 (缓存映射) ============================
缓存里真实存在的是 "移除相关证据后" 的 token logprob (lp_no<R>):
  ragtruth  -> lp_nopsg   (移除源文档 passage)                直接存
  haloquest -> lp_noimg = lp_full - delta_img_tok             由 lp_full 与图像 delta 重构
  avhbench  -> lp_noR  = lp_full - (delta_audio + delta_video) 双相关模态, 一阶可加近似
"证据无关先验分量" P_i^0 取该 evidence-removed logprob 相对随机基线 (uniform, -log|V|)
的对数似然增益, 使其与 E_i^R 同为 nats 尺度、可在第二项里与 [E_i^R]_+ 相减:
  P_i^0 = lp_noR + log|V|     (prior_baseline='uniform', 默认; |V|=152064, 见 model_utils)
可配置 prior_baseline='none' 则 P_i^0 = lp_noR (原始 logprob, 第二项通常不激活)。
这是把抽象 P_i^0 落到"缓存里真有的量"的显式选择, 非金标, 已如实标注。

============================ 离线 vs GPU 边界 ============================
离线 (本文件可直接跑, 无需 GPU): rho_i 打分 + Expand/Merge span 选择, 依赖旧运行缓存的
per-token per-view 分数 (见 CACHE_PATHS)。可做阈值扫描、被标记 token 比例、span 数/长度统计。
GPU (必须, 本任务不自跑全量): Rev_theta 条件重生成。见 regenerate_spans() 与文末运行命令。

注意: 这三个数据集无 GT span 标注, 故 Corrected-Span-Ratio / Grounded-Preservation /
Over-edit-Rate 等需金标 span 的指标无法计算, 本文件只报可从真实输出测到的量。
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))

# 词表大小: 与 methods/shared/model_utils.py 第42行 thinker.config.vocab_size 一致
VOCAB_SIZE = 152064

# ---- 缓存 (旧运行产出的 per-token per-view 分数; 离线打分数据源) ----
CACHE_PATHS = {
    # ragtruth: 全 token 逐 view, 100 样本; 字段 tokens/classes/lp_full/lp_nopsg/delta
    "ragtruth": os.path.join(REPO, "outputs/cf_probe/probe_report_100_v2.json"),
    # haloquest: 全 token, 295 样本; delta_img_tok(视觉)/delta_text_tok(文本)/lp_full/lp_noq
    "haloquest": os.path.join(REPO, "outputs/cf_probe/haloquest_textcf_tokens.jsonl"),
    # avhbench: 全 token; lp_full/lp_noaudio/lp_novideo/delta_audio/delta_video (+text join)
    "avhbench": os.path.join(REPO, "outputs/cf_probe/avh_probe_bigN_shard0of1.jsonl"),
    "avhbench_text": os.path.join(REPO, "outputs/cf_probe/avh_probe_text_empty.jsonl"),
}

# unified `modality` 字段 (OmniHalluBench/*/*.json) -> (相关模态集 R, 全部候选模态集)
# 依据: R 判定子调查确认三数据集 modality 字段权威且逐样本一致
#   ragtruth  modality == "text"          -> R={text},           全={text}          I=空
#   haloquest modality == "visual"        -> R={visual},         全={visual,text}   I={text}
#   avhbench  modality == "visual+audio"  -> R={audio,visual},   全={audio,visual,text} I={text}
MODALITY_R_MAP = {
    "text": ({"text"}, {"text"}),
    "visual": ({"visual"}, {"visual", "text"}),
    "visual+audio": ({"audio", "visual"}, {"audio", "visual", "text"}),
    "audio+visual": ({"audio", "visual"}, {"audio", "visual", "text"}),
}


@dataclass
class RepairConfig:
    tau: float = 0.5                    # rho 阈值, > tau 标记待修订
    prior_baseline: str = "none"        # 默认 'none' -> P=lp_noR (离线验证唯一有区分度);
    #                                     'uniform' -> P=lp_noR+log|V| (g2 饱和, 不推荐)
    vocab_size: int = VOCAB_SIZE
    merge_gap: int = 0                  # Merge: 允许合并的最大 token 间隙 (0=严格相邻)
    content_only: bool = False          # 只在 classes=='content' 的 token 上标记 (func 词不修订)
    max_new_tokens: int = 96            # Rev_theta 每 span 重生成上限
    dataset_name: str = ""              # 供生成路径分派
    min_span_len: int = 1               # Round 2: 最小 span 长度 (token 数); 短于此的噪声 span 丢弃
    max_spans: int = 0                  # Round 2: 每样本最多保留 span 数 (0=不限); 按 span 内 rho 均值降序取 top-N
    select_mode: str = "rho"            # Round 3: span 选择信号
    #   'rho'           — 聚合修订风险 rho_i > tau (默认)
    #   'under_reliance'— 相关通道净反对 (-g1) > tau，即 E_R < -tau (依赖不足)
    #   'over_reliance' — 无关通道净驱动 g3 > tau (依赖过度)
    #   'either'        — under_reliance OR over_reliance 的并集
    diag_mode: str = "none"             # Round 5: 重生成 prompt 里的诊断注入模式
    #   'none' — 不注入任何诊断 (Round 2/3 的做法; 当前最优)
    #   'hard' — 注入完整机制诊断 (g1/g3 数值/通道名; Round 4 已测试, 反效果)
    #   'soft' — 注入操作性指令 (不暴露机制语言，只告知"重新观察证据/不要依赖常识")
    # Round 6: 证据约束局部修正
    evidence_mode: bool = False          # True → 两步证据查询路径 (Phase B)
    evidence_max_new_tokens: int = 64    # 证据查询 generate 的 token 上限 (短; verdict+一句)
    max_repl_ratio: float = 2.5          # 接受判据: len(repl) ≤ ratio*len(old)+slack
    max_repl_slack: int = 20             # 接受判据长度检查的绝对字符容量
    # Round 7: coverage-gated repair
    coverage_cap: float = 0.0            # 跳过修复的覆盖率上界 (0=不限; >0 则覆盖率超 cap 时直接跳过)

    @property
    def prior_offset(self) -> float:
        if self.prior_baseline == "uniform":
            return math.log(self.vocab_size)
        if self.prior_baseline == "none":
            return 0.0
        raise ValueError(f"unknown prior_baseline: {self.prior_baseline}")


def relevant_modalities(modality: str) -> Tuple[set, set, set]:
    """由 unified `modality` 字段定 (R, all, I=all-R)。问题偏向什么模态,什么就相关。"""
    key = (modality or "").strip().lower()
    if key not in MODALITY_R_MAP:
        raise KeyError(
            f"未知 modality={modality!r}; 已知 {list(MODALITY_R_MAP)}。"
            "avhbench 全样本为 'visual+audio' (AV Captioning, 问题固定含 see+hear),"
            "无逐样本 A2V/V2A 拆分, 故 R={audio,visual} 联合。"
        )
    R, allm = MODALITY_R_MAP[key]
    return set(R), set(allm), set(allm) - set(R)

def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def compute_gamma_rho(
    kappa: Dict[str, np.ndarray],
    lp_noR: np.ndarray,
    R: set,
    I: set,
    cfg: RepairConfig,
) -> Dict[str, np.ndarray]:
    """
    纯后验函数: 由 per-token per-view 分数算三分量 gamma 与修订风险 rho。

    参数
      kappa : {modality -> per-token kappa_i^(m) 数组}, 需覆盖 R 与 I 中出现且缓存里有的模态
      lp_noR: per-token 的 "移除所有相关证据" logprob (nats); 用于 P_i^0
      R, I  : 相关 / 无关模态集
      cfg   : RepairConfig (prior_baseline, vocab_size)

    返回 dict: g1(E_R), g2, g3, rho, P (P_i^0), 均为 per-token np.array
    """
    # 长度对齐检查
    n = len(lp_noR)
    for m, v in kappa.items():
        if len(v) != n:
            raise ValueError(f"模态 {m} kappa 长度 {len(v)} != lp_noR 长度 {n}")

    # E_i^R = sum_{m in R} kappa_i^(m)  (只累加缓存里真有的相关模态)
    E_R = np.zeros(n, dtype=np.float64)
    for m in R:
        if m in kappa:
            E_R = E_R + np.asarray(kappa[m], dtype=np.float64)

    # P_i^0 = lp_noR + offset  (offset = log|V| for 'uniform', 0 for 'none')
    P = np.asarray(lp_noR, dtype=np.float64) + cfg.prior_offset

    # 三分量 gamma
    g1 = E_R
    g2 = _relu(_relu(P) - _relu(E_R))
    g3 = np.zeros(n, dtype=np.float64)
    for m in I:
        if m in kappa:
            g3 = g3 + _relu(np.asarray(kappa[m], dtype=np.float64))

    # rho = [-g1]_+ + g2 + g3
    rho = _relu(-g1) + g2 + g3
    return {"g1": g1, "g2": g2, "g3": g3, "P": P, "rho": rho}

def _word_starts(tokens: Sequence[str]) -> np.ndarray:
    """
    标记每个 token 是否为一个词的起始。
    Qwen BPE 约定: 词首 token 带前导空格 (缓存里如 ' musicians'), 子词续片无前导空格
    (如 'To',' po','ach' 中 'ach' 续 ' po')。据此定词边界, Expand 时不切碎词。
    """
    starts = np.zeros(len(tokens), dtype=bool)
    for i, t in enumerate(tokens):
        if i == 0:
            starts[i] = True
        elif t[:1] in (" ", "\n", "\t") or (t[:1] and not t[0].isalnum()):
            # 前导空白 => 新词; 前导非字母数字 (标点等) 也视作独立单元
            starts[i] = True
    return starts


def expand_to_word_boundaries(tokens: Sequence[str], marked: np.ndarray) -> np.ndarray:
    """把 marked 的 token 扩到完整词: 若某 token 被标记, 其所属词的所有子词 token 全标记。"""
    n = len(tokens)
    ws = _word_starts(tokens)
    # 每个 token 所属词的 [起, 止) 区间
    word_bounds: List[Tuple[int, int]] = []
    i = 0
    while i < n:
        j = i + 1
        while j < n and not ws[j]:
            j += 1
        word_bounds.append((i, j))
        i = j
    out = np.zeros(n, dtype=bool)
    for (s, e) in word_bounds:
        if marked[s:e].any():
            out[s:e] = True
    return out


def merge_spans(marked: np.ndarray, gap: int = 0) -> List[Tuple[int, int]]:
    """
    合并相邻/重叠的标记 token 成连续 span, 返回 [(start, end_exclusive), ...]。
    gap: 两段之间未标记 token 数 <= gap 时并入同一 span (0 = 仅严格相邻)。
    """
    idx = np.flatnonzero(marked)
    if idx.size == 0:
        return []
    spans: List[Tuple[int, int]] = []
    s = int(idx[0])
    prev = int(idx[0])
    for k in idx[1:]:
        k = int(k)
        if k - prev - 1 <= gap:
            prev = k
        else:
            spans.append((s, prev + 1))
            s = k
            prev = k
    spans.append((s, prev + 1))
    return spans


def select_spans(
    tokens: Sequence[str],
    rho: np.ndarray,
    cfg: RepairConfig,
    classes: Optional[Sequence[str]] = None,
    g1: Optional[np.ndarray] = None,
    g3: Optional[np.ndarray] = None,
) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """信号选择 -> (可选 content_only 过滤) -> Expand 词边界 -> Merge ->
    (Round 2) min_span_len 过滤 -> max_spans top-N 截断。返回 (B_rev, marked)。

    select_mode 含义:
      'rho'            — 聚合修订风险 rho_i > tau (默认, Round 1/2)
      'under_reliance' — 相关通道净反对: (-g1) > tau 即 E_R < -tau
                         含义: 该 token 在相关通道证据下"不该说"，但模型仍输出
      'over_reliance'  — 无关通道净驱动: g3 > tau
                         含义: 该 token 由不该依赖的通道驱动
      'either'         — under_reliance OR over_reliance 的并集
    """
    rho_arr = np.asarray(rho, dtype=np.float64)
    mode = cfg.select_mode

    if mode == "rho":
        marked = rho_arr > cfg.tau
    elif mode == "under_reliance":
        if g1 is None:
            raise ValueError("select_mode='under_reliance' 需要传入 g1 数组")
        marked = (-np.asarray(g1, dtype=np.float64)) > cfg.tau
    elif mode == "over_reliance":
        if g3 is None:
            raise ValueError("select_mode='over_reliance' 需要传入 g3 数组")
        marked = np.asarray(g3, dtype=np.float64) > cfg.tau
    elif mode == "either":
        if g1 is None or g3 is None:
            raise ValueError("select_mode='either' 需要传入 g1 和 g3 数组")
        marked = ((-np.asarray(g1, dtype=np.float64)) > cfg.tau) | \
                 (np.asarray(g3, dtype=np.float64) > cfg.tau)
    elif mode == "topk":
        # Round 7: top-max_spans 个 rho_i token + 最小 ±2 token 窗口，合并重叠窗口
        max_k = max(cfg.max_spans, 1)
        rho_vals = [(i, float(rho_arr[i])) for i in range(len(rho_arr)) if rho_arr[i] > 0]
        rho_vals.sort(key=lambda x: -x[1])
        seeds = [i for i, _ in rho_vals[:max_k]]
        if not seeds:
            # 没有 rho>0 的 token — 退化为空 span 列表
            marked = np.zeros(len(rho_arr), dtype=bool)
            return [], marked
        windows = [(max(0, i - 2), min(len(rho_arr), i + 3)) for i in seeds]
        windows.sort()
        merged: List[List[int]] = []
        for s, e in windows:
            if merged and s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        spans = [(s, e) for s, e in merged]
        marked = np.zeros(len(rho_arr), dtype=bool)
        for s, e in spans:
            marked[s:e] = True
        return spans, marked
    else:
        raise ValueError(f"未知 select_mode={mode!r}; 合法值: rho/under_reliance/over_reliance/either/topk")
    if cfg.content_only and classes is not None:
        is_content = np.array([str(c).startswith("cont") for c in classes], dtype=bool)
        marked = marked & is_content
    marked = expand_to_word_boundaries(tokens, marked)
    spans = merge_spans(marked, gap=cfg.merge_gap)

    # Round 2: 过滤过短的噪声 span
    if cfg.min_span_len > 1:
        spans = [(s, e) for (s, e) in spans if (e - s) >= cfg.min_span_len]

    # Round 2: 每样本最多保留 max_spans 个 span，按 span 内 rho 均值降序取 top-N
    if cfg.max_spans > 0 and len(spans) > cfg.max_spans:
        span_scores = [(s, e, float(rho_arr[s:e].mean())) for (s, e) in spans]
        span_scores.sort(key=lambda x: x[2], reverse=True)
        spans = [(s, e) for (s, e, _) in span_scores[:cfg.max_spans]]
        spans.sort(key=lambda x: x[0])  # 恢复顺序以便正确重建文本

    # 从最终 span 列表重建 marked 数组（与 spans 保持一致）
    if cfg.min_span_len > 1 or cfg.max_spans > 0:
        marked = np.zeros(len(tokens), dtype=bool)
        for (s, e) in spans:
            marked[s:e] = True

    return spans, marked

def _iter_jsonl(path: str):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_ragtruth(path: Optional[str] = None) -> List[dict]:
    """
    ragtruth: R={text}, I=空。P_i^0 用 lp_nopsg (移除 passage) 直接得, 无需重构。
    E_i^R = kappa_text = delta = lp_full - lp_nopsg。I 空 -> g3=0。
    """
    path = path or CACHE_PATHS["ragtruth"]
    d = json.load(open(path))
    out = []
    for r in d["per_sample"]:
        tokens = r["tokens"]
        out.append({
            "id": r["id"],
            "tokens": tokens,
            "classes": r.get("classes"),
            "kappa": {"text": np.asarray(r["delta"], dtype=np.float64)},
            "lp_noR": np.asarray(r["lp_nopsg"], dtype=np.float64),
            "modality": "text",
        })
    return out


def load_haloquest(path: Optional[str] = None) -> List[dict]:
    """
    haloquest: R={visual}, I={text}。
    kappa_visual = delta_img_tok; kappa_text = delta_text_tok。
    lp_noR = lp_noimg = lp_full - delta_img_tok (由 lp_full 与图像 delta 重构)。
    """
    path = path or CACHE_PATHS["haloquest"]
    out = []
    for r in _iter_jsonl(path):
        lp_full = np.asarray(r["lp_full"], dtype=np.float64)
        d_img = np.asarray(r["delta_img_tok"], dtype=np.float64)
        out.append({
            "id": r["id"],
            "tokens": r["tokens"],
            "classes": r.get("classes"),
            "kappa": {
                "visual": d_img,
                "text": np.asarray(r["delta_text_tok"], dtype=np.float64),
            },
            "lp_noR": lp_full - d_img,
            "modality": "visual",
            # 新定位器 (localizer_v2) 所需原始字段 — 直接保留避免重新加载缓存
            "delta_img_tok": list(d_img),
            "delta_text_tok": list(np.asarray(r["delta_text_tok"], dtype=np.float64)),
            "lp_full": list(lp_full),
        })
    return out


def load_avhbench(path: Optional[str] = None, text_path: Optional[str] = None) -> List[dict]:
    """
    avhbench: R={audio,visual}, I={text}。
    kappa_audio=delta_audio, kappa_visual=delta_video, kappa_text=delta_text (从 text 文件按 id join)。
    lp_noR = lp_full - (delta_audio+delta_video)  (双相关模态一阶可加近似; 已标注)。
    text 文件缺失的样本: 无 I 通道 -> g3=0 (照实, 不臆造)。
    """
    path = path or CACHE_PATHS["avhbench"]
    text_path = text_path or CACHE_PATHS["avhbench_text"]
    dtext = {}
    if os.path.exists(text_path):
        for r in _iter_jsonl(text_path):
            dtext[str(r["id"])] = np.asarray(r["delta_text"], dtype=np.float64)
    out = []
    for r in _iter_jsonl(path):
        lp_full = np.asarray(r["lp_full"], dtype=np.float64)
        d_a = np.asarray(r["delta_audio"], dtype=np.float64)
        d_v = np.asarray(r["delta_video"], dtype=np.float64)
        kappa = {"audio": d_a, "visual": d_v}
        dt = dtext.get(str(r["id"]))
        if dt is not None and len(dt) == len(lp_full):
            kappa["text"] = dt
        out.append({
            "id": r["id"],
            "tokens": r["tokens"],
            "classes": r.get("classes"),
            "kappa": kappa,
            "lp_noR": lp_full - (d_a + d_v),
            "modality": "visual+audio",
            "has_text_view": dt is not None,
        })
    return out


LOADERS = {"ragtruth": load_ragtruth, "haloquest": load_haloquest, "avhbench": load_avhbench}

def token_dependency_labels(g1: np.ndarray, g3: np.ndarray, tau: float = 0.0) -> np.ndarray:
    """
    按 g1/g3 给每个 token 打依赖关系标签 (字符串数组):
      'under_reliance' — E_R < -tau: 相关证据反对此 token，模型依赖相关通道不足
      'over_reliance'  — g3 > tau:   无关通道驱动此 token，模型过度依赖无关通道
      'both'           — 两者同时成立 (极端幻觉)
      'normal'         — 两者均不成立 (依赖关系正常)
    """
    under = (-np.asarray(g1, dtype=np.float64)) > tau
    over  = np.asarray(g3, dtype=np.float64) > tau
    labels = np.where(
        under & over, "both",
        np.where(under, "under_reliance",
                 np.where(over, "over_reliance", "normal"))
    )
    return labels


def score_sample(sample: dict, cfg: RepairConfig) -> dict:
    """对单个已加载样本: 算 rho 并选 span。返回打分 + span 结构 (不重生成)。
    select_mode 由 cfg.select_mode 控制:
      'rho'            — 聚合 rho_i > tau (默认)
      'under_reliance' — (-g1) > tau
      'over_reliance'  — g3 > tau
      'either'         — 两者并集
    """
    R, allm, I = relevant_modalities(sample["modality"])
    gr = compute_gamma_rho(sample["kappa"], sample["lp_noR"], R, I, cfg)
    spans, marked = select_spans(
        sample["tokens"], gr["rho"], cfg, sample.get("classes"),
        g1=gr["g1"], g3=gr["g3"],
    )
    dep_labels = token_dependency_labels(gr["g1"], gr["g3"], tau=0.0)
    return {
        "id": sample["id"],
        "n_tokens": len(sample["tokens"]),
        "rho": gr["rho"],
        "marked": marked,
        "spans": spans,
        "gamma": {k: gr[k] for k in ("g1", "g2", "g3", "P")},
        "dep_labels": dep_labels,   # per-token: normal/under_reliance/over_reliance/both
    }


def offline_stats(dataset: str, cfg: RepairConfig, taus: Optional[Sequence[float]] = None) -> dict:
    """
    离线可验证部分: 对整个数据集扫 tau, 汇报真实可测量 (无金标 span 依赖):
      - 被标记 token 比例 mean/分布
      - span 数、平均 span 长度 (token)
      - 有 span 的样本占比 (edit-触发率)
    绝不计算 Corrected-Span-Ratio / Grounded-Preservation / Over-edit (需 GT span)。
    """
    samples = LOADERS[dataset]()
    taus = list(taus) if taus is not None else [cfg.tau]
    rows = []
    for tau in taus:
        c = RepairConfig(tau=tau, prior_baseline=cfg.prior_baseline,
                         vocab_size=cfg.vocab_size, merge_gap=cfg.merge_gap,
                         content_only=cfg.content_only)
        n_tok_total = 0
        n_marked_total = 0
        marked_frac_per_sample = []
        span_counts = []
        span_lens = []
        n_with_span = 0
        for s in samples:
            r = score_sample(s, c)
            n = r["n_tokens"]
            nm = int(r["marked"].sum())
            n_tok_total += n
            n_marked_total += nm
            marked_frac_per_sample.append(nm / n if n else 0.0)
            span_counts.append(len(r["spans"]))
            for (a, b) in r["spans"]:
                span_lens.append(b - a)
            if r["spans"]:
                n_with_span += 1
        rows.append({
            "tau": tau,
            "n_samples": len(samples),
            "marked_token_frac_micro": (n_marked_total / n_tok_total) if n_tok_total else 0.0,
            "marked_token_frac_macro_mean": float(np.mean(marked_frac_per_sample)) if marked_frac_per_sample else 0.0,
            "samples_with_span_frac": n_with_span / len(samples) if samples else 0.0,
            "mean_spans_per_sample": float(np.mean(span_counts)) if span_counts else 0.0,
            "mean_span_len_tok": float(np.mean(span_lens)) if span_lens else 0.0,
            "total_spans": int(np.sum(span_counts)),
        })
    return {"dataset": dataset, "config": {
        "prior_baseline": cfg.prior_baseline, "vocab_size": cfg.vocab_size,
        "merge_gap": cfg.merge_gap, "content_only": cfg.content_only,
    }, "sweep": rows}

def tokens_to_text(tokens: Sequence[str]) -> str:
    """Qwen BPE token 串带前导空格, 直接拼接即还原文本。"""
    return "".join(tokens)


def build_infill_prompt(question: str, answer_marked: str, passage: Optional[str] = None,
                        dep_diagnosis: Optional[str] = None) -> str:
    """
    Rev_theta 的条件重生成提示 (infill): 给完整问题(+证据)与带标记草稿, 只重写标记内文本。
    未标记区间保持不变 -> 满足 "未标记区间原样保留"。媒体证据由 messages 里的媒体 token 提供。

    dep_diagnosis: 由 span_dependency_diagnosis() 生成的依赖归因说明, 告诉模型为什么这个
    span 被怀疑是幻觉, 以及应该往哪个方向修正。若为 None 则用通用提示。
    """
    ev = f"\n\nSource passage:\n{passage}" if passage else ""
    if dep_diagnosis:
        diagnosis_block = f"\n\nDiagnosis of the marked span:\n{dep_diagnosis}"
        instruction = (
            "The marked span is flagged as a potential hallucination based on the diagnosis above. "
            "Rewrite ONLY the text inside <<< >>> to correct it according to the diagnosis and "
            "make it faithful to the evidence. "
            "Keep everything outside the markers identical. "
            "Output only the replacement text for the marked span, nothing else."
        )
    else:
        diagnosis_block = ""
        instruction = (
            "Rewrite ONLY the text inside <<< >>> so it is faithful to the evidence above. "
            "Keep everything outside the markers identical. "
            "Output only the replacement text for the marked span, nothing else."
        )
    return (
        f"Question: {question}{ev}{diagnosis_block}\n\n"
        f"Draft answer (a span is wrapped in <<< >>>):\n{answer_marked}\n\n"
        f"{instruction}"
    )


def span_dependency_diagnosis(
    span_tokens: Sequence[str],
    span_dep_labels: Sequence[str],
    span_g1: np.ndarray,
    span_g3: np.ndarray,
    modality: str,
) -> str:
    """
    给单个 span 生成人类可读的依赖归因诊断，供注入 prompt。

    逻辑：
    1. 统计 span 内各依赖类型的 token 数
    2. 取信号最强的几个 token（g3 最大 / -g1 最大）作为具体例证
    3. 根据主导类型生成针对性说明：
       over_reliance  → 模型在生成这段时依赖了不相关的通道（如文本先验/记忆），
                        而非问题所要求的视觉/听觉证据
       under_reliance → 相关通道的证据明确反对这段内容，但模型仍然输出了它
       both           → 同时存在两种问题（最严重）
    """
    R_name, I_name = _modality_channel_names(modality)

    labels = np.asarray(span_dep_labels)
    g1_arr = np.asarray(span_g1, dtype=np.float64)
    g3_arr = np.asarray(span_g3, dtype=np.float64)
    n = len(labels)

    n_under = int((labels == "under_reliance").sum())
    n_over  = int((labels == "over_reliance").sum())
    n_both  = int((labels == "both").sum())
    n_hal   = n_under + n_over + n_both

    if n_hal == 0:
        return (
            f"This span ({n} tokens) shows no strong dependency anomaly. "
            "Verify it is still consistent with the visual/audio evidence."
        )

    lines = []

    # 主导问题
    dominant = max(
        [("over_reliance", n_over + n_both), ("under_reliance", n_under + n_both)],
        key=lambda x: x[1]
    )[0]

    if dominant == "over_reliance" or n_over + n_both >= n_under + n_both:
        # 找 g3 最大的 token 作证据
        top_idx = int(np.argmax(g3_arr))
        top_tok = repr(span_tokens[top_idx])
        top_g3  = float(g3_arr[top_idx])
        lines.append(
            f"Over-reliance on irrelevant channel ({I_name}): "
            f"{n_over + n_both}/{n} tokens in this span are driven by "
            f"the {I_name} channel (language prior / memory) rather than the "
            f"{R_name} evidence the question asks about. "
            f"The strongest signal is on token {top_tok} (g3={top_g3:.2f}), "
            f"meaning the model's prediction of that word was disproportionately "
            f"influenced by {I_name} context, not by what is actually visible/audible."
        )
        lines.append(
            f"Correction direction: ignore language-prior expectations and "
            f"re-describe only what the {R_name} evidence directly supports."
        )

    if n_under + n_both > 0:
        # 找 -g1 最大的 token
        top_idx2 = int(np.argmax(-g1_arr))
        top_tok2 = repr(span_tokens[top_idx2])
        top_neg_g1 = float(-g1_arr[top_idx2])
        lines.append(
            f"Under-reliance on relevant channel ({R_name}): "
            f"{n_under + n_both}/{n} tokens contradict the {R_name} evidence. "
            f"The strongest signal is on token {top_tok2} (E_R={-top_neg_g1:.2f}), "
            f"meaning the relevant {R_name} evidence actively argues against "
            f"this token being correct — the model said it anyway."
        )
        lines.append(
            f"Correction direction: the {R_name} evidence does not support "
            f"this content; revise to reflect what the {R_name} actually shows."
        )

    return " ".join(lines)


def span_soft_hint(
    span_tokens: Sequence[str],
    span_dep_labels: Sequence[str],
    span_g1: np.ndarray,
    span_g3: np.ndarray,
    modality: str,
) -> str:
    """
    Round 5: 软诊断提示。与 span_dependency_diagnosis 签名相同，但不暴露任何内部机制语言
    (不提 g1/g3/通道名/token计数)，只给模型可直接执行的操作性指令。

    over_reliance  → 告知模型重新仔细观察证据，只描述实际可见/可听到的内容
    under_reliance → 告知模型答案与证据矛盾，需重读证据后修正
    both           → 两者都有，用更强的措辞
    normal         → 不太可能走到这里，给通用核查指令
    """
    labels = np.asarray(span_dep_labels)
    n_under = int((labels == "under_reliance").sum())
    n_over  = int((labels == "over_reliance").sum())
    n_both  = int((labels == "both").sum())

    # 模态 → 人类可读的证据来源名称
    modality_label = {
        "visual":       "the image",
        "audio":        "the audio",
        "text":         "the passage",
        "visual+audio": "the image and audio",
        "audio+visual": "the image and audio",
    }.get((modality or "").strip().lower(), "the provided evidence")

    n_hal = n_under + n_over + n_both

    if n_hal == 0:
        return (
            f"Double-check this span against {modality_label} to make sure it is accurate."
        )

    lines = []

    if n_over + n_both > 0:
        lines.append(
            f"This span may rely on general knowledge or common expectations rather than "
            f"what is actually shown in {modality_label}. "
            f"Look carefully at {modality_label} again and rewrite the span to describe "
            f"only what you can directly observe — do not infer or assume."
        )

    if n_under + n_both > 0:
        lines.append(
            f"This span appears to contradict what {modality_label} shows. "
            f"Re-examine {modality_label} closely and correct the span so it accurately "
            f"reflects what is actually there."
        )

    return " ".join(lines)


def _modality_channel_names(modality: str):
    """返回 (relevant_channel_name, irrelevant_channel_name) 供 diagnosis 文本用。"""
    key = (modality or "").strip().lower()
    mapping = {
        "visual":         ("visual (image)",    "language prior / text memory"),
        "audio":          ("audio",              "language prior / text memory"),
        "text":           ("passage text",       "language prior / memory"),
        "visual+audio":   ("visual and audio",   "language prior / text memory"),
        "audio+visual":   ("visual and audio",   "language prior / text memory"),
    }
    return mapping.get(key, ("relevant modality", "irrelevant channel"))


def _modality_label(modality: str) -> str:
    """人类可读的证据来源名称, 供 evidence query prompt 用。"""
    return {
        "visual":       "the image",
        "audio":        "the audio",
        "text":         "the passage",
        "visual+audio": "the image and audio",
        "audio+visual": "the image and audio",
    }.get((modality or "").strip().lower(), "the provided evidence")


# Round 6: 证据约束修正的辅助常量与函数

import re as _re  # noqa: E402 (延迟导入避免顶层副作用; 实际 re 总在标准库)

REFUSAL_WORDS: frozenset = frozenset({
    "not", "no", "cannot", "can't", "don't", "doesn't", "isn't", "aren't",
    "wasn't", "weren't", "never", "unable", "unknown", "unclear",
    "undetermined", "there is no", "there are no",
})


def _has_refusal(text: str) -> bool:
    """True 当且仅当文本含否定/拒答词汇 (短语优先, 再检查单词)。"""
    t = text.lower()
    for phrase in ("there is no", "there are no", "cannot", "can't", "don't",
                   "doesn't", "isn't", "aren't", "wasn't", "weren't"):
        if phrase in t:
            return True
    words = set(t.split())
    return bool(words & {"not", "no", "never", "unable", "unknown", "unclear", "undetermined"})


def _strip_infill_markers(text: str) -> str:
    """去除模型复读的 <<< / >>> infill 标记。

    优先提取配对 <<<...>>> 中的内容; 若无配对则逐个剥离。
    """
    m = _re.search(r'<<<\s*(.*?)\s*>>>', text, _re.DOTALL)
    if m:
        return m.group(1).strip()
    text = _re.sub(r'<<<+\s*', '', text)
    text = _re.sub(r'\s*>>>+', '', text)
    return text.strip()


def build_evidence_query(
    question: str,
    full_answer: str,
    span_text: str,
    modality_label: str,
) -> str:
    """构造 span 证据核查 prompt。

    呈现完整候选回答 + span 文本, 要求模型以固定格式回答 VERDICT/REASON。
    模型看到完整回答是为了判断 span 是否与上下文一致 (而非孤立核查)。
    """
    return (
        f"You are fact-checking one specific claim within a model's answer "
        f"against the visual evidence.\n\n"
        f"Question: {question}\n"
        f"Full answer: {full_answer}\n"
        f"Claim to verify: \"{span_text}\"\n\n"
        f"Look carefully at {modality_label} and determine whether this specific "
        f"claim is directly supported by what you can observe.\n\n"
        f"Respond in EXACTLY this format (nothing else):\n"
        f"VERDICT: [SUPPORTED | NOT_SUPPORTED | CANNOT_DETERMINE]\n"
        f"REASON: [one sentence describing what {modality_label} shows about this claim]"
    )


def parse_evidence_verdict(response: str) -> Tuple[str, str]:
    """解析证据核查回答, 返回 (verdict, evidence_text)。

    verdict 取值: SUPPORTED / NOT_SUPPORTED / CANNOT_DETERMINE
    evidence_text: REASON 行内容, 供后续 conditioned prompt 使用。
    默认回退: CANNOT_DETERMINE (保守, 不修改)。
    """
    valid = {"SUPPORTED", "NOT_SUPPORTED", "CANNOT_DETERMINE"}

    # 尝试 VERDICT: 行
    vm = _re.search(r'VERDICT\s*:\s*(\S+)', response, _re.IGNORECASE)
    if vm:
        v = vm.group(1).upper().rstrip(".,;")
        if v in valid:
            verdict = v
        elif "NOT" in v:
            verdict = "NOT_SUPPORTED"
        elif "CANNOT" in v or "UNDETERMINED" in v:
            verdict = "CANNOT_DETERMINE"
        else:
            verdict = "SUPPORTED"
    else:
        # 关键词 fallback
        tl = response.lower()
        if "not_supported" in tl or "not supported" in tl:
            verdict = "NOT_SUPPORTED"
        elif "cannot_determine" in tl or "cannot determine" in tl or "undetermined" in tl:
            verdict = "CANNOT_DETERMINE"
        elif "supported" in tl:
            verdict = "SUPPORTED"
        else:
            verdict = "CANNOT_DETERMINE"

    # 提取 REASON
    rm = _re.search(r'REASON\s*:\s*(.+)', response, _re.IGNORECASE | _re.DOTALL)
    evidence_text = rm.group(1).strip()[:300] if rm else response.strip()[:200]

    return verdict, evidence_text


def _base_is_refusal(base_pred: str, old_span: str) -> bool:
    """True 当 old_span 本身含否定词 (span 是拒答措辞的一部分), 而非整个回答的判断。

    只检查 old_span 及其在 base_pred 中前后 40 字符的上下文窗口。
    """
    cs = base_pred.find(old_span)
    if cs >= 0:
        window = base_pred[max(0, cs - 40): cs + len(old_span) + 40]
    else:
        window = old_span
    return _has_refusal(window)


def _should_accept_repl(
    repl: str,
    old_span: str,
    old_verdict: str,
    repl_verdict: str,
    base_pred: str,
    evidence_mode: bool,
    cfg: "RepairConfig",
) -> Tuple[bool, str]:
    """接受判据: 返回 (accept, reason_str)。False = Keep 原 span。

    检查顺序 (任一 False 立即返回):
      1. evidence_mode + SUPPORTED → Keep
      2. evidence_mode + CANNOT_DETERMINE + base 已是拒答 → Keep
      3. repl 为空 → Keep
      4. repl 长度膨胀 → Keep
      5. 否定词翻转 (old_span 含否定词但 repl 不含) → Keep
      6. evidence_mode + repl 被证据否定 → Keep
    """
    if evidence_mode and old_verdict == "SUPPORTED":
        return False, "old_verdict=SUPPORTED"
    if (evidence_mode
            and old_verdict == "CANNOT_DETERMINE"
            and _base_is_refusal(base_pred, old_span)):
        return False, "base_already_hedged"
    if not repl.strip():
        return False, "empty_repl"
    max_len = int(cfg.max_repl_ratio * len(old_span)) + cfg.max_repl_slack
    if len(repl) > max_len:
        return False, f"length_bloat({len(repl)}>{max_len})"
    if _has_refusal(old_span) and not _has_refusal(repl):
        return False, "negation_flip"
    if evidence_mode and repl_verdict == "NOT_SUPPORTED":
        return False, "repl_not_supported"
    return True, "accepted"


def build_evidence_conditioned_prompt(
    question: str,
    marked_text: str,
    evidence_text: str,
    passage: Optional[str] = None,
    dep_diagnosis: Optional[str] = None,
) -> str:
    """infill prompt 变体: 在标记回答前插入 NOT_SUPPORTED 证据块。

    仅在 old_verdict == NOT_SUPPORTED 时调用。
    """
    parts = [f"Question: {question}"]
    if passage:
        parts.append(f"Passage: {passage}")
    if dep_diagnosis:
        parts.append(f"Diagnosis of the marked span: {dep_diagnosis}")
    parts.append(
        f"Fact-check result for the marked span:\n"
        f"The claim is NOT supported by the evidence.\n"
        f"Evidence: {evidence_text}\n\n"
        f"Draft answer (a span is wrapped in <<< >>>):\n{marked_text}\n\n"
        f"Rewrite ONLY the text inside <<< >>> to correct it based on the evidence above.\n"
        f"If the evidence shows the claim is wrong, fix it to match the evidence.\n"
        f"If the content cannot be determined, replace the span with language reflecting uncertainty.\n"
        f"Keep everything outside the markers identical.\n"
        f"Output only the replacement text for the marked span, nothing else."
    )
    return "\n".join(parts)


def regenerate_spans(item: dict, tokens: Sequence[str], spans: List[Tuple[int, int]],
                     processor, model, cfg: RepairConfig,
                     data_path: str, data_root: str,
                     dep_labels: Optional[np.ndarray] = None,
                     gamma: Optional[dict] = None) -> dict:
    """
    GPU 必需。对每个 span 用同 backbone 条件重生成重写并 Replace 回原文, 未标记区间保留。

    item     : methods.shared.data_loader.load_dataset() 归一化后的样本 dict
               (含 question / passage / image_path / video_path / audio_path)。
    processor,model: methods.shared.model_utils.load_model() 的返回。
    data_path, data_root: 供 load_media_for_item 解析媒体绝对路径 (与 pipeline 同一解析)。
    dep_labels: score_sample() 返回的 per-token 依赖标签 (np array of str), 用于生成诊断 prompt。
    gamma    : score_sample() 返回的 gamma dict, 含 g1/g3 数组, 用于定量诊断说明。
    返回 {orig_text, repaired_text, spans, replacements, span_token_coverage}。
    本文件不自跑 GPU; 主控加载模型后调用本函数 (见文末命令)。

    Round 6 新增:
    - Bug fix A1: _strip_infill_markers() 剥离 repl 里的 <<< / >>> 残留
    - Bug fix A2: 字符串层替换 (char-offset), 不再把 decoded repl 插回 token list
    - evidence_mode=True: 两步证据查询 (build_evidence_query → rewrite → post-gen check)
    - span_token_coverage: 记录 Top-2 span 覆盖了多少比例的 answer token
    """
    import torch  # 延迟导入: 离线打分无需 torch
    import sys
    pkg = os.path.join(REPO, "OmniConfess")
    if pkg not in sys.path:
        sys.path.insert(0, pkg)
    from methods.shared.model_utils import (  # noqa: E402
        build_inputs, make_messages, build_user_content_with_media_tokens,
        prepare_for_thinker, decode_outputs,
    )
    from methods.shared.data_loader import load_media_for_item  # noqa: E402
    from methods.shared.prompts import get_system_prompt  # noqa: E402

    pixel_values, audio_values, image_values = load_media_for_item(item, data_path, data_root)
    system_prompt = get_system_prompt(cfg.dataset_name, "v1")
    question = item.get("question", "")
    passage = item.get("passage")
    modality = item.get("modality", "visual")
    mod_label = _modality_label(modality)

    tokens = list(tokens)
    n_tokens = len(tokens)
    replacements = []

    # 单次推理调用链的内部辅助 (闭包捕获 processor/model/system_prompt)
    def _run_generate(prompt_text: str, max_new_tokens: int) -> str:
        user_content = build_user_content_with_media_tokens(
            pixel_values, audio_values, image_values, prompt_text)
        messages = make_messages(system_prompt, user_content)
        inputs = build_inputs(processor, model, messages, pixel_values,
                              audio_values, image_values, cfg.dataset_name)
        inputs = prepare_for_thinker(inputs, model)
        with torch.inference_mode():
            out_ids = model.thinker.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return decode_outputs(processor, out_ids, inputs["input_ids"].shape[1])[0].strip()

    # A2: 字符串层替换 — 预计算所有 span 的 char offset (基于原始 token list, 一次性)
    orig_text = tokens_to_text(tokens)
    char_spans: List[Tuple[int, int]] = []
    for (s, e) in spans:
        cs = sum(len(t) for t in tokens[:s])
        ce = sum(len(t) for t in tokens[:e])
        char_spans.append((cs, ce))

    # span token coverage: Top-2 span 覆盖了多少 answer token
    span_token_coverage = sum(e - s for (s, e) in spans) / n_tokens if n_tokens > 0 else 0.0

    # Round 7: coverage cap — 覆盖率超阈值时跳过修复，直接返回原文
    if cfg.coverage_cap > 0 and span_token_coverage > cfg.coverage_cap:
        return {
            "orig_text": orig_text,
            "repaired_text": orig_text,
            "spans": spans,
            "replacements": [],
            "span_token_coverage": round(span_token_coverage, 4),
            "coverage_cap_skip": True,
        }

    current_text = orig_text
    # 从后往前按 char offset 替换, 保证前面 span 的 offset 不受影响
    sorted_items = sorted(
        zip(spans, char_spans), key=lambda x: x[1][0], reverse=True
    )

    for (s, e), (cs, ce) in sorted_items:
        old_span = orig_text[cs:ce]

        # 依赖诊断 (diag_mode 控制, 与 R4/R5 向后兼容)
        dep_diagnosis = None
        if cfg.diag_mode != "none" and dep_labels is not None and gamma is not None:
            span_labels = dep_labels[s:e]
            span_g1 = np.asarray(gamma["g1"][s:e], dtype=np.float64)
            span_g3 = np.asarray(gamma["g3"][s:e], dtype=np.float64)
            if cfg.diag_mode == "hard":
                dep_diagnosis = span_dependency_diagnosis(
                    tokens[s:e], span_labels, span_g1, span_g3, modality)
            elif cfg.diag_mode == "soft":
                dep_diagnosis = span_soft_hint(
                    tokens[s:e], span_labels, span_g1, span_g3, modality)

        # Round 6 Phase B: 两步证据查询
        old_verdict = "NOT_SUPPORTED"  # 默认: 非证据模式直接走重写
        evidence_text = ""
        repl_verdict = "SUPPORTED"    # 默认: 非证据模式跳过重写后核查

        if cfg.evidence_mode:
            ev_prompt = build_evidence_query(question, orig_text, old_span, mod_label)
            ev_resp = _run_generate(ev_prompt, cfg.evidence_max_new_tokens)
            old_verdict, evidence_text = parse_evidence_verdict(ev_resp)

        # 前置门控: SUPPORTED 或 base 已拒答的 CANNOT_DETERMINE → 直接 Keep
        pre_accept, pre_reason = _should_accept_repl(
            old_span,  # repl=old_span 触发 "不修改" 的下游检查
            old_span, old_verdict, repl_verdict, orig_text,
            cfg.evidence_mode, cfg,
        )
        # 注意: pre_accept 这里只检查 verdict 门和否定 flip (repl==old_span 不会触发空/膨胀)
        # 真正做法: 在生成 repl 之前只判 verdict 相关条件
        pre_accept_verdict = not (
            (cfg.evidence_mode and old_verdict == "SUPPORTED")
            or (cfg.evidence_mode
                and old_verdict == "CANNOT_DETERMINE"
                and _base_is_refusal(orig_text, old_span))
        )

        if not pre_accept_verdict:
            replacements.append({
                "span": (s, e), "char_span": (cs, ce),
                "old": old_span, "new": old_span,
                "dep_diagnosis": dep_diagnosis,
                "verdict": old_verdict, "evidence_text": evidence_text,
                "repl_verdict": "", "accept_reason": "kept_pre_verdict", "kept": True,
            })
            # current_text 不变
            continue

        # 构造 infill prompt
        if cfg.evidence_mode and old_verdict == "NOT_SUPPORTED":
            marked_text = (current_text[:cs] + " <<< " + current_text[cs:ce]
                           + " >>> " + current_text[ce:])
            prompt_text = build_evidence_conditioned_prompt(
                question, marked_text, evidence_text, passage, dep_diagnosis)
        else:
            marked_text = (current_text[:cs] + " <<< " + current_text[cs:ce]
                           + " >>> " + current_text[ce:])
            prompt_text = build_infill_prompt(question, marked_text, passage, dep_diagnosis)

        repl_raw = _run_generate(prompt_text, cfg.max_new_tokens)
        # A1: 剥离 <<< / >>> 残留
        repl = _strip_infill_markers(repl_raw)

        # 重写后证据核查 (Phase B, evidence_mode=True)
        if cfg.evidence_mode:
            # 构造含 repl 的完整回答, 检查 repl 是否也被证据支持
            repl_answer = current_text[:cs] + repl + current_text[ce:]
            ev2_prompt = build_evidence_query(question, repl_answer, repl, mod_label)
            ev2_resp = _run_generate(ev2_prompt, cfg.evidence_max_new_tokens)
            repl_verdict, _ = parse_evidence_verdict(ev2_resp)

        accept, accept_reason = _should_accept_repl(
            repl, old_span, old_verdict, repl_verdict, orig_text,
            cfg.evidence_mode, cfg,
        )
        if not accept:
            repl = old_span  # Keep

        replacements.append({
            "span": (s, e), "char_span": (cs, ce),
            "old": old_span, "new": repl,
            "dep_diagnosis": dep_diagnosis,
            "verdict": old_verdict, "evidence_text": evidence_text,
            "repl_verdict": repl_verdict, "accept_reason": accept_reason,
            "kept": not accept,
        })
        current_text = current_text[:cs] + repl + current_text[ce:]

    return {
        "id": item.get("id"),
        "orig_text": orig_text,
        "repaired_text": current_text,
        "spans": spans,
        "replacements": list(reversed(replacements)),
        "span_token_coverage": round(span_token_coverage, 4),
    }

def _cli():
    ap = argparse.ArgumentParser(description="free-form rho_i 修订风险 + span 选择 (离线)")
    ap.add_argument("--mode", choices=["offline", "dump"], default="offline",
                    help="offline: 扫 tau 出统计; dump: 导出每样本 rho+span (jsonl)")
    ap.add_argument("--dataset", choices=list(LOADERS), required=True)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--taus", type=str, default="",
                    help="逗号分隔 tau 扫描列表, 如 0.5,1.0,2.0,4.0")
    ap.add_argument("--prior-baseline", choices=["uniform", "none"], default="uniform")
    ap.add_argument("--merge-gap", type=int, default=0)
    ap.add_argument("--content-only", action="store_true")
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    cfg = RepairConfig(tau=args.tau, prior_baseline=args.prior_baseline,
                       merge_gap=args.merge_gap, content_only=args.content_only,
                       dataset_name=args.dataset)

    if args.mode == "offline":
        taus = [float(x) for x in args.taus.split(",")] if args.taus else None
        res = offline_stats(args.dataset, cfg, taus)
        print(json.dumps(res, ensure_ascii=False, indent=2))
        if args.out:
            json.dump(res, open(args.out, "w"), ensure_ascii=False, indent=2)
    else:  # dump
        samples = LOADERS[args.dataset]()
        out = args.out or f"freeform_rho_{args.dataset}.jsonl"
        with open(out, "w") as f:
            for s in samples:
                r = score_sample(s, cfg)
                f.write(json.dumps({
                    "id": r["id"], "n_tokens": r["n_tokens"],
                    "tokens": s["tokens"],
                    "rho": [round(float(x), 4) for x in r["rho"]],
                    "marked_idx": [int(i) for i in np.flatnonzero(r["marked"])],
                    "spans": r["spans"],
                }, ensure_ascii=False) + "\n")
        print(f"dumped -> {out} ({len(samples)} samples)")


if __name__ == "__main__":
    _cli()

