"""
localizer_v2.py — Step1-Step3 of HaloQuest freeform repair re-localizer.

背景: Round 7 实验证明旧 rho_i 定位"不确定性"而非"错误"
(base_f1>=66 桶 83 次编辑 72 次退化, corr(base_f1,delta)=-0.524)。
新方案: 直接用缓存里 delta_img_tok (visual_dependence) 作为定位信号。

delta_img_tok[i] = lp_full[i] - lp_no_image[i]
  => 该 token 有多依赖视觉证据；值越大说明图片越关键。
  => 负值说明图片反而降低该 token 的概率 (视觉干扰或文本主导)。

不是 rho 改名: rho = relu(-g1)+g2+g3 是合成修订风险，与 delta_img_tok 语义完全不同。

离线可运行：纯 Python + numpy，不需要 torch/GPU。
"""

from __future__ import annotations

import json
import os
from typing import List, Optional, Sequence, Tuple

import numpy as np

# 从 contract 导入数据契约 (Candidate, LocalizeResult, CACHE_PATH 等)
from localizer_contract import (
    CACHE_PATH,
    Candidate,
    LocalizeResult,
)

# ──────────────────────────────────────────────────────────────────────────────
# 关键词集合 (用于 span_kind 分类, Step3 step d)
# ──────────────────────────────────────────────────────────────────────────────

# 颜色词 (英文常见)
_COLOR_WORDS = {
    "red", "blue", "green", "yellow", "orange", "purple", "pink",
    "brown", "black", "white", "gray", "grey", "cyan", "magenta",
    "violet", "indigo", "beige", "tan", "teal", "gold", "silver",
    "maroon", "navy", "olive", "lime", "turquoise", "coral",
}

# 数词关键词 (数字词 + 形状词放在一起, 因为都是 color_count_shape 类)
_COUNT_SHAPE_WORDS = {
    # 数词
    "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "several", "many", "few",
    "multiple", "single", "pair", "dozen",
    # 形状词
    "round", "square", "rectangular", "circular", "oval", "triangular",
    "cylindrical", "spherical", "flat", "curved", "straight",
}

# 空间/动作词 (action_spatial 类)
_SPATIAL_WORDS = {
    "in", "on", "above", "below", "left", "right", "between",
    "behind", "front", "near", "next", "beside", "under", "over",
    "inside", "outside", "top", "bottom", "around", "along",
    "across", "through", "against", "toward", "towards",
}

_ACTION_WORDS = {
    "is", "are", "was", "were", "has", "have", "had",
    "doing", "holding", "wearing", "standing", "sitting",
    "walking", "running", "looking", "pointing", "facing",
    "showing", "carrying", "placing", "putting", "lying",
}

# 否定/拒答词 (negation_refusal 类)
_NEGATION_WORDS = {
    "not", "no", "never", "cannot", "can't", "cant",
    "impossible", "unclear", "unknown", "unable", "without",
    "nothing", "none", "lack", "lacking", "missing",
}

# 动词关键词 (用于区分 object vs action_spatial)
_VERB_KEYWORDS = _ACTION_WORDS | _SPATIAL_WORDS


# ──────────────────────────────────────────────────────────────────────────────
# 1. 加载缓存
# ──────────────────────────────────────────────────────────────────────────────

def load_cache(path: Optional[str] = None) -> dict[str, dict]:
    """
    加载 jsonl 缓存文件，返回 {id: raw_record}。

    path 默认使用 localizer_contract.CACHE_PATH (相对路径相对于 CWD，
    也支持绝对路径)。每条记录包含 tokens/delta_img_tok/classes/question/id 等字段。
    """
    if path is None:
        path = CACHE_PATH

    # 尝试相对路径，失败则尝试从本文件所在目录推导
    if not os.path.isabs(path) and not os.path.exists(path):
        base = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base, path)

    cache: dict[str, dict] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            record_id = str(record.get("id", ""))
            cache[record_id] = record
    return cache


# ──────────────────────────────────────────────────────────────────────────────
# 2. 从缓存记录提取 visual_dependence
# ──────────────────────────────────────────────────────────────────────────────

def visual_dependence_from_cache(record: dict) -> np.ndarray:
    """
    直接返回 record["delta_img_tok"] 作为 visual_dependence 数组。

    delta_img_tok[i] = lp_full[i] - lp_no_image[i]，即移除图像后该 token
    的 logprob 变化量。正值 = 该 token 依赖视觉证据；值越大依赖越强。
    这不是 rho 改名——rho 是 relu(-g1)+g2+g3 的合成风险，完全不同的量。
    """
    return np.asarray(record["delta_img_tok"], dtype=float)


# ──────────────────────────────────────────────────────────────────────────────
# 3. Token → 字符偏移映射
# ──────────────────────────────────────────────────────────────────────────────

def token_to_char_map(tokens: list[str]) -> list[tuple[int, int]]:
    """
    返回每个 token 的 (char_start, char_end)，与 freeform_repair.tokens_to_text 一致。

    约定: orig_text = "".join(tokens)
          char_start(i) = sum(len(t) for t in tokens[:i])
          char_end(i)   = char_start(i) + len(tokens[i])

    注意: BPE token 里的前导空格 (' mountains') 是 token 本身的字符，会保留在 orig_text 中。
    """
    char_map: list[tuple[int, int]] = []
    pos = 0
    for t in tokens:
        char_map.append((pos, pos + len(t)))
        pos += len(t)
    return char_map


# ──────────────────────────────────────────────────────────────────────────────
# BPE 词边界工具 (复制自 freeform_repair.py，避免循环依赖)
# ──────────────────────────────────────────────────────────────────────────────

def _word_starts(tokens: Sequence[str]) -> np.ndarray:
    """
    标记每个 token 是否为一个词的起始。

    Qwen BPE 约定: 词首 token 带前导空格 (如 ' mountains')，子词续片无前导空格
    (如 'ach' 续 ' po')。标点等前导非字母数字符号也视为独立单元起始。
    """
    starts = np.zeros(len(tokens), dtype=bool)
    for i, t in enumerate(tokens):
        if i == 0:
            # 第一个 token 总是词首
            starts[i] = True
        elif t[:1] in (" ", "\n", "\t"):
            # 前导空白 => 新词
            starts[i] = True
        elif t[:1] and not t[0].isalnum():
            # 前导非字母数字 (标点等) => 视为独立单元
            starts[i] = True
    return starts


def _expand_to_word_boundaries(tokens: Sequence[str], marked: np.ndarray) -> np.ndarray:
    """
    把 marked 中被标记的 token 扩展到完整词边界：
    若某词内任意 sub-token 被标记，该词所有 sub-token 全部标记。
    防止把 BPE 子词切碎成不完整词。
    """
    n = len(tokens)
    ws = _word_starts(tokens)

    # 构建每个 token 所属词的 [start, end) 区间
    word_bounds: list[tuple[int, int]] = []
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


def _merge_spans(marked: np.ndarray, gap: int = 1) -> list[tuple[int, int]]:
    """
    合并相邻标记 token 为连续 span，返回 [(start, end_exclusive), ...]。

    gap=1: 两段之间最多 1 个未标记 token 时合并 (用于合并相邻高依赖词)。
    这比 freeform_repair 中默认 gap=0 宽松一格，避免单个 func token 把语义上连续的词分裂。
    """
    idx = np.flatnonzero(marked)
    if idx.size == 0:
        return []

    spans: list[tuple[int, int]] = []
    s = int(idx[0])
    prev = int(idx[0])
    for k in idx[1:]:
        k = int(k)
        if k - prev - 1 <= gap:
            # gap 内，合并
            prev = k
        else:
            spans.append((s, prev + 1))
            s = k
            prev = k
    spans.append((s, prev + 1))
    return spans


# ──────────────────────────────────────────────────────────────────────────────
# span_kind 分类辅助函数
# ──────────────────────────────────────────────────────────────────────────────

def _classify_span_kind(
    span_tokens: list[str],
    classes: Optional[list[str]],
    tok_start: int,
    preceding_token: str = "",
) -> str:
    """
    根据 span 内 token 文本 + classes 字段推断 span_kind。

    参数:
        span_tokens     : span 内的 token 列表
        classes         : 全句 classes 标签，可为 None
        tok_start       : span 在全句 token 序列中的起始偏移
        preceding_token : span 前一个 token 的文本（用于 gap-merge 后的否定检测，
                          修复 id=601 类 ' not' 因 gap-merge 与后续词分离导致
                          is_refusal 未激活的问题）

    优先级:
      1. negation_refusal: span 本身含否定词，或前驱 token 含否定词
      2. color_count_shape: 包含颜色/数量/形状词
      3. action_spatial: 包含动作/空间关系词
      4. object: content token 占比高且不含动词关键词
      5. other: 其余
    """
    # 提取 span 内所有词的小写文本（去掉 BPE 前导空格）
    words = [t.strip().lower() for t in span_tokens if t.strip()]

    word_set = set(words)

    # 1. 否定/拒答
    if word_set & _NEGATION_WORDS:
        return "negation_refusal"
    # 检查复合否定 "is not" / "are not" 等
    joined = " ".join(words)
    if any(neg in joined for neg in ("is not", "are not", "was not", "were not", "can not", "do not", "does not")):
        return "negation_refusal"
    # A2 修复: 检查前驱 token 是否含否定词（gap-merge 可能把否定词留在相邻 span 外）
    if preceding_token:
        pre = preceding_token.strip().lower()
        if pre in _NEGATION_WORDS:
            return "negation_refusal"
        # 前驱 token 本身就是 "n't" / "not" 的 BPE 片段
        if pre in ("n't", "nt", "not", "no"):
            return "negation_refusal"
        # 前驱 + span 首词构成复合否定
        if words:
            combined = pre + " " + words[0]
            if any(neg in combined for neg in (
                "is not", "are not", "was not", "were not",
                "can not", "do not", "does not", "cannot",
                "isn't", "aren't", "wasn't", "weren't",
                "can't", "won't", "don't", "doesn't",
            )):
                return "negation_refusal"

    # 2. 颜色/数量/形状
    if word_set & (_COLOR_WORDS | _COUNT_SHAPE_WORDS):
        return "color_count_shape"

    # 3. 动作/空间关系
    if word_set & (_SPATIAL_WORDS | _ACTION_WORDS):
        return "action_spatial"

    # 4. object: content 占比 > 0.5 且不含动词关键词
    if classes is not None:
        span_classes = classes[tok_start: tok_start + len(span_tokens)]
        n_content = sum(1 for c in span_classes if c == "content")
        frac = n_content / max(len(span_classes), 1)
        if frac > 0.5 and not (word_set & _VERB_KEYWORDS):
            return "object"

    # 5. 默认
    return "other"


# ──────────────────────────────────────────────────────────────────────────────
# Semantic Claim 解析 (A3)
# ──────────────────────────────────────────────────────────────────────────────

# 常见颜色/形状/大小等属性词集合（用于 claim 解析）
_ATTRIBUTE_WORDS = (
    _COLOR_WORDS
    | {
        # 大小
        "large", "small", "big", "tiny", "huge", "enormous", "little",
        "tall", "short", "wide", "narrow", "thick", "thin",
        # 形状
        "round", "square", "rectangular", "circular", "oval",
        "triangular", "cylindrical", "spherical", "flat", "curved",
        "straight",
        # 材质/状态
        "wooden", "metal", "plastic", "glass", "stone", "leather",
        "bright", "dark", "light", "shiny", "matte", "old", "new",
        "young", "adult", "open", "closed", "empty", "full",
    }
)

# 数词 → 数字映射（仅用于 claim 主语生成）
_WORD_TO_NUM: dict[str, str] = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "eleven": "11", "twelve": "12", "several": "several",
    "many": "many", "few": "a few", "multiple": "multiple",
    "single": "one", "pair": "a pair of", "dozen": "a dozen",
    "1": "1", "2": "2", "3": "3", "4": "4", "5": "5",
    "6": "6", "7": "7", "8": "8", "9": "9", "10": "10",
}

_STOPWORDS = {
    "a", "an", "the", "this", "that", "these", "those",
    "is", "are", "was", "were", "be", "been", "being",
    "in", "on", "at", "of", "to", "for", "with", "by",
    "and", "or", "but", "so", "if", "as",
    "it", "its", "he", "she", "they", "we", "you", "i",
}


def _extract_claim(span_text: str, question: str, base_pred: str) -> dict:
    """
    把 token span 解析为可核查的语义声明（纯启发式，离线，不调用模型）。

    返回:
    {
        "claim_type"      : "object_existence" | "object_attribute" |
                            "object_count" | "action" | "spatial" |
                            "answerability" | "other",
        "primary_claim"   : str,   # 主声明（对象是否存在）
        "attribute_claim" : str,   # 属性声明（可为空串）
        "preserve_hint"   : str,   # 原答案中需要保留的内容提示
    }

    设计原则:
      - primary_claim 始终可被 verifier 以 yes/no 验证
      - attribute_claim 非空时 verifier 可先问存在性再问属性，减少假正例
      - 纯词法分析，不依赖 parse tree；对短 span 足够精确
    """
    # 分词（小写，去 BPE 前导空格）
    raw_words = [w.strip() for w in span_text.split() if w.strip()]
    words_lower = [w.lower().rstrip(".,!?;:") for w in raw_words]
    word_set_lower = set(words_lower)

    # ── 1. 否定/拒答 → answerability ────────────────────────────────────────
    if word_set_lower & _NEGATION_WORDS:
        q_fragment = question.strip().rstrip("?").strip() if question else ""
        return {
            "claim_type": "answerability",
            "primary_claim": (
                f"The image does not contain sufficient information to answer: "
                f"'{q_fragment}'" if q_fragment
                else "The image does not contain sufficient information to answer the question."
            ),
            "attribute_claim": "",
            "preserve_hint": span_text.strip(),
        }

    # ── 2. 抽取名词主语（最后一个非属性词、非数词、非停词内容词）──────────────
    noun_candidates = [
        w for w in raw_words
        if w.lower().rstrip(".,!?;:") not in _STOPWORDS
        and w.lower().rstrip(".,!?;:") not in _ATTRIBUTE_WORDS
        and w.lower().rstrip(".,!?;:") not in _WORD_TO_NUM
        and w.lower().rstrip(".,!?;:") not in _NEGATION_WORDS
        and w.lower().rstrip(".,!?;:") not in _SPATIAL_WORDS
        and w.lower().rstrip(".,!?;:") not in (_ACTION_WORDS - {"holding", "wearing",
                                                                  "standing", "sitting",
                                                                  "walking", "running",
                                                                  "looking", "pointing",
                                                                  "facing", "showing",
                                                                  "carrying"})
    ]
    # 取最后一个名词候选（通常是中心名词）
    noun = noun_candidates[-1].rstrip(".,!?;:") if noun_candidates else span_text.strip()

    # ── 3. 数量声明 ─────────────────────────────────────────────────────────
    count_word = next(
        (w for w in words_lower if w in _WORD_TO_NUM),
        None,
    )
    if count_word is not None:
        count_str = _WORD_TO_NUM[count_word]
        primary = f"There are {count_str} {noun}(s) in the image."
        return {
            "claim_type": "object_count",
            "primary_claim": primary,
            "attribute_claim": "",
            "preserve_hint": span_text.strip(),
        }

    # ── 4. 属性声明（含颜色/形状/大小等）───────────────────────────────────
    attr_words = [w for w in words_lower if w in _ATTRIBUTE_WORDS]
    if attr_words:
        attr_str = " and ".join(attr_words)
        primary = f"A {noun} is present in the image."
        attribute = f"The {noun} is {attr_str}."
        return {
            "claim_type": "object_attribute",
            "primary_claim": primary,
            "attribute_claim": attribute,
            "preserve_hint": span_text.strip(),
        }

    # ── 5. 动作/空间声明 ────────────────────────────────────────────────────
    action_words_in_span = [w for w in words_lower if w in _ACTION_WORDS]
    spatial_words_in_span = [w for w in words_lower if w in _SPATIAL_WORDS]
    if action_words_in_span:
        action = action_words_in_span[0]
        primary = f"The {noun} is {action} as described."
        return {
            "claim_type": "action",
            "primary_claim": primary,
            "attribute_claim": "",
            "preserve_hint": span_text.strip(),
        }
    if spatial_words_in_span:
        spatial = spatial_words_in_span[0]
        primary = f"The {noun} is located {spatial} as described."
        return {
            "claim_type": "spatial",
            "primary_claim": primary,
            "attribute_claim": "",
            "preserve_hint": span_text.strip(),
        }

    # ── 6. 纯存在性声明 ─────────────────────────────────────────────────────
    if noun and noun.lower() not in _STOPWORDS:
        article = "An" if noun[0].lower() in "aeiou" else "A"
        primary = f"{article} {noun} is visible in the image."
        return {
            "claim_type": "object_existence",
            "primary_claim": primary,
            "attribute_claim": "",
            "preserve_hint": span_text.strip(),
        }

    # ── 7. 回退：保留原始 span ───────────────────────────────────────────────
    return {
        "claim_type": "other",
        "primary_claim": span_text.strip(),
        "attribute_claim": "",
        "preserve_hint": span_text.strip(),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 4. 语义 span 提取 (Step 3)
# ──────────────────────────────────────────────────────────────────────────────

def extract_semantic_spans(
    tokens: list[str],
    visual_dep: np.ndarray,
    classes: Optional[list[str]] = None,
    min_vis_dep: float = 0.05,
    question: str = "",
) -> list[dict]:
    """
    Step3: 把 token 序列合并成语义完整的候选 span，返回 list[dict]（与 Candidate 字段对齐）。

    参数:
        tokens      : BPE token 列表 (含前导空格)
        visual_dep  : delta_img_tok 数组，shape=(n_tokens,)
        classes     : 每 token 的词类标签 "func"/"content"，可为 None
        min_vis_dep : 高依赖阈值 (宽松候选入口，不是编辑决策)
        question    : 原始问题文本，用于 claim 生成

    流程:
        a) 标记 visual_dep > min_vis_dep 的 token 为"高依赖"
        b) 扩展到词边界 (不切 BPE 子词)
        c) 合并相邻高依赖词 (gap<=1 token)
        d) 判断 span_kind
        e) 聚合 visual_dependence (mean)
        f) 生成 claim

    重要约束:
        - 不把所有连续高依赖 token 合并成单一大 span (否则退化成 Round7 问题)
        - 若整个答案都是高依赖 (coverage>0.8)，按语义进一步拆分
        - 归一化聚合 (mean)，不跨不同长度 span 比较未归一化分数
    """
    n = len(tokens)
    if n == 0:
        return []

    visual_dep = np.asarray(visual_dep, dtype=float)

    # ── a) 标记高依赖 token ──────────────────────────────────────────────────
    high_dep = visual_dep > min_vis_dep  # shape: (n,)

    if not high_dep.any():
        # 没有任何高依赖 token，返回空列表 (不强行生成候选)
        return []

    # ── b) 扩展到词边界 ──────────────────────────────────────────────────────
    high_dep_word = _expand_to_word_boundaries(tokens, high_dep)

    # ── c) 合并相邻高依赖词 (gap=1) ──────────────────────────────────────────
    raw_spans = _merge_spans(high_dep_word, gap=1)

    # 若整个答案几乎都是高依赖 (coverage>0.8)，对合并结果按词边界进一步拆分，
    # 避免返回覆盖全句的单一 span (Round7 退化根源)
    coverage = float(high_dep_word.sum()) / n
    if coverage > 0.8:
        raw_spans = _split_large_spans(tokens, raw_spans, visual_dep)

    # ── 构建字符映射 ─────────────────────────────────────────────────────────
    char_map = token_to_char_map(tokens)
    orig_text = "".join(tokens)

    # ── d/e/f) 对每个 span 分类 + 聚合 + 生成 claim ─────────────────────────
    results: list[dict] = []
    for (ts, te) in raw_spans:
        span_toks = tokens[ts:te]
        if not span_toks:
            continue

        # 字符偏移
        char_start = char_map[ts][0]
        char_end = char_map[te - 1][1]
        span_text = orig_text[char_start:char_end]

        # 跳过纯空白或极短 span (单个标点等)
        if not span_text.strip():
            continue

        # e) 聚合 visual_dependence (mean)
        tok_vis = visual_dep[ts:te].tolist()
        agg_vis = float(np.mean(visual_dep[ts:te]))

        # d) span_kind 分类 (A2 修复: 传入 preceding_token)
        # 取 span 前一个 token（gap-merge 后可能是否定词）
        preceding_tok = tokens[ts - 1] if ts > 0 else ""
        span_kind = _classify_span_kind(span_toks, classes, ts,
                                         preceding_token=preceding_tok)

        # f) claim 生成 (flat string for Candidate.claim backward compat)
        #    + claim_parsed dict via _extract_claim (A3)
        claim_parsed = _extract_claim(span_text, question, "")
        if span_kind == "negation_refusal":
            claim = (
                "Given the image and question, does the image lack the information "
                "that would be needed to answer this question, making this refusal appropriate?"
            )
        else:
            q_str = question.strip().rstrip("?").strip() if question else ""
            # For object_attribute: surface both primary and attribute as a
            # two-step claim string so downstream verifier can use them.
            if claim_parsed["claim_type"] == "object_attribute":
                p = claim_parsed["primary_claim"]
                a = claim_parsed["attribute_claim"]
                if q_str:
                    claim = (
                        f"Given the image and question '{q_str}', "
                        f"first verify: '{p}' Then if yes, verify: '{a}'"
                    )
                else:
                    claim = f"First verify: '{p}' Then if yes, verify: '{a}'"
            elif q_str:
                claim = (
                    f"Given the image and question '{q_str}', "
                    f"is the following statement supported by the image: '{span_text.strip()}'?"
                )
            else:
                claim = (
                    f"Is the following statement supported by the image: '{span_text.strip()}'?"
                )

        results.append(dict(
            span_text=span_text,
            char_start=char_start,
            char_end=char_end,
            token_start=ts,
            token_end=te,
            visual_dependence=agg_vis,
            agg_method="mean",
            claim=claim,
            claim_parsed=claim_parsed,   # A3: structured semantic claim
            question=question,
            span_kind=span_kind,
            tok_visual_dependence=tok_vis,
            verify_status="NEEDS_VERIFICATION",
            verify_raw="",
            is_refusal=(span_kind == "negation_refusal"),
            decision="PRESERVE",
        ))

    return results


def _split_large_spans(
    tokens: list[str],
    spans: list[tuple[int, int]],
    visual_dep: np.ndarray,
) -> list[tuple[int, int]]:
    """
    当整体 coverage>0.8 时，对大 span 按词边界拆分成更小的独立语义单元。

    策略:
      - 每个 span 内找 visual_dep 的局部极小值点作为分割点 (低依赖的 func 词)
      - 若 span 长度 <= 4 token，不拆
      - 拆后每段 >= 2 token
    """
    ws = _word_starts(tokens)
    result: list[tuple[int, int]] = []

    for (ts, te) in spans:
        span_len = te - ts
        if span_len <= 4:
            result.append((ts, te))
            continue

        # 找该 span 内词边界位置 (除首位)
        word_starts_in_span = [
            i for i in range(ts + 1, te) if ws[i]
        ]

        if len(word_starts_in_span) < 2:
            result.append((ts, te))
            continue

        # 在词边界处找 visual_dep 局部极小值，作为拆分候选
        # 计算每个词边界处前后 mean 差异，差异最大（即该点是局部低谷）的点优先拆
        split_scores = []
        for wb in word_starts_in_span:
            # 该词边界处的 visual_dep 值（作为局部估计）
            split_scores.append((visual_dep[wb], wb))

        # 按 visual_dep 升序排，取前 N//3 个低值点作为分割点
        split_scores.sort()
        n_splits = max(1, span_len // 5)  # 大约每 5 token 拆一刀
        chosen_splits = sorted([wb for _, wb in split_scores[:n_splits]])

        # 按选定分割点拆 span
        seg_starts = [ts] + chosen_splits
        seg_ends = chosen_splits + [te]
        for (ss, se) in zip(seg_starts, seg_ends):
            if se - ss >= 2:  # 至少 2 token
                result.append((ss, se))
            elif se - ss == 1:
                # 太短，合并到前一段
                if result and result[-1][1] == ss:
                    result[-1] = (result[-1][0], se)
                else:
                    result.append((ss, se))

    return result


# ──────────────────────────────────────────────────────────────────────────────
# 5. localize — 主入口
# ──────────────────────────────────────────────────────────────────────────────

def localize(record: dict, cfg=None) -> LocalizeResult:
    """
    主入口：接受缓存里的一条原始 record，返回 LocalizeResult。

    参数:
        record : 缓存 jsonl 中的单条记录 (含 tokens/delta_img_tok/classes/question/id)
        cfg    : 预留配置参数，暂未使用 (接口兼容)

    返回:
        LocalizeResult，其中:
          - base_pred = "".join(tokens)   冻结答案原文
          - candidates = extract_semantic_spans 返回的 Candidate 列表
          - rho_spans_old = []            旧 rho 结果由调用方从 freeform_repair.score_sample 取后传入
          - revision_candidates 属性      = decision=="REVISION_CANDIDATE" 的候选 (初始均为空)
    """
    tokens: list[str] = record.get("tokens", [])
    question: str = record.get("question", "")
    record_id: str = str(record.get("id", ""))
    classes: Optional[list[str]] = record.get("classes", None)

    # Step1: 锚定冻结答案
    base_pred = "".join(tokens)

    # Step2: 提取 visual_dependence
    visual_dep = visual_dependence_from_cache(record)

    # Step3: 提取语义候选 span
    span_dicts = extract_semantic_spans(
        tokens=tokens,
        visual_dep=visual_dep,
        classes=classes,
        question=question,
    )

    # 把 dict 转成 Candidate dataclass
    candidates: list[Candidate] = []
    for d in span_dicts:
        candidates.append(Candidate(
            span_text=d["span_text"],
            char_start=d["char_start"],
            char_end=d["char_end"],
            token_start=d["token_start"],
            token_end=d["token_end"],
            visual_dependence=d["visual_dependence"],
            agg_method=d["agg_method"],
            claim=d["claim"],
            claim_parsed=d.get("claim_parsed", {}),   # A3: structured claim
            question=d["question"],
            span_kind=d["span_kind"],
            tok_visual_dependence=d["tok_visual_dependence"],
            verify_status=d["verify_status"],
            verify_raw=d["verify_raw"],
            is_refusal=d["is_refusal"],
            decision=d["decision"],
        ))

    return LocalizeResult(
        id=record_id,
        question=question,
        base_pred=base_pred,
        n_tokens=len(tokens),
        candidates=candidates,
        rho_spans_old=[],   # 由调用方从 freeform_repair.score_sample 取后填入
        note="",
    )


# ──────────────────────────────────────────────────────────────────────────────
# 6. localize_batch — 批量版本
# ──────────────────────────────────────────────────────────────────────────────

def localize_batch(records: list[dict]) -> list[LocalizeResult]:
    """批量定位，直接 list comprehension。"""
    return [localize(r) for r in records]
