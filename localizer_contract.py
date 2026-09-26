"""
localizer_contract.py — 证据定位重构的共享数据契约 (无业务逻辑, 三个模块共同 import)

背景: Round 7 实测证明旧 rho_i (freeform_repair.compute_gamma_rho 的修订风险) 定位的是
"不确定性"而非"错误"——编辑集中在 base 正确样本 (base_f1>=66 桶 83 次编辑 72 次退化,
mean_delta=-25.26, corr(base_f1,delta)=-0.524)。本次重构改用 teacher-forced 视觉证据
依赖信号重建定位, rho_i 仅保留为诊断基线, 不再直接决定编辑。

关键事实 (可行性已确认):
  缓存 outputs/cf_probe/haloquest_textcf_tokens.jsonl 每条记录字段:
    tokens        : list[str]  贪心冻结答案的 token (带 BPE 前导空格约定)
    classes       : list[str]  每 token 词类 func/content
    lp_full       : list[float] 全证据(图+问+答)下逐 token logprob, teacher-forced 在冻结答案上
    delta_img_tok : list[float] = lp_full - lp_no_image  (即 visual_dependence[i], 移除图像后的 logprob 差)
    delta_text_tok: list[float] = lp_full - lp_no_question
    n_tok, question, neg_item, repro_err
  => Step1/Step2 完全离线可得, 零 GPU。delta_img_tok 就是 visual_dependence, 直接取用,
     不是把 rho 改名 (rho 是 relu(-g1)+g2+g3 的合成风险, 与 delta_img_tok 是不同量)。

token->char 映射约定 (与 freeform_repair.tokens_to_text 一致, 精确可逆):
    orig_text = "".join(tokens)
    char_start(i) = sum(len(t) for t in tokens[:i])
    char_end(i)   = sum(len(t) for t in tokens[:i+1])
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Sequence


# --- Step 3 输出: 语义完整候选 span ---
@dataclass
class Candidate:
    span_text: str                      # 原文中该 span 的精确子串 (char_start:char_end)
    char_start: int
    char_end: int
    token_start: int                    # 半开区间 [token_start, token_end)
    token_end: int
    visual_dependence: float            # 该 span 聚合后的 delta_img_tok (聚合法见 agg_method)
    agg_method: str                     # 聚合方式, 如 "mean" / "len_normalized_sum" — 禁止跨不同长度 span 比未归一化分
    claim: str                          # Step3->Step4: 把 span 转成可对图核验的具体断言（flat string，向后兼容）
    question: str
    span_kind: str = "other"            # object/color_count_shape/action_spatial/negation_refusal/other
    # A3: 结构化语义声明（_extract_claim 输出）
    # claim_type: object_existence | object_attribute | object_count |
    #             action | spatial | answerability | other
    # primary_claim: 主声明（可 yes/no 验证）
    # attribute_claim: 属性声明（非空时 verifier 先验存在性再验属性）
    # preserve_hint: 原答案中需保留的内容
    claim_parsed: dict = field(default_factory=dict)
    # Step2 附带的逐 token 明细 (供诊断, 不直接触发修订)
    tok_visual_dependence: List[float] = field(default_factory=list)
    # Step4 填充:
    verify_status: str = "NEEDS_VERIFICATION"   # SUPPORTED / UNSUPPORTED / INSUFFICIENT_EVIDENCE / NEEDS_VERIFICATION
    verify_raw: str = ""                          # 模型验证原始回答 (可审计)
    is_refusal: bool = False                      # 该 span 是否属于合法拒答语义
    # Step5 填充:
    decision: str = "PRESERVE"          # SUSPICIOUS_BUT_UNVERIFIED / REVISION_CANDIDATE / PRESERVE

    def to_dict(self) -> dict:
        return asdict(self)


# --- 顶层定位结果 ---
@dataclass
class LocalizeResult:
    id: str
    question: str
    base_pred: str                      # = "".join(tokens), 冻结答案原文
    n_tokens: int
    candidates: List[Candidate] = field(default_factory=list)
    # 诊断基线 (仅记录, 不决定编辑):
    rho_spans_old: List = field(default_factory=list)   # 旧 rho_i>tau 选出的 token 区间
    note: str = ""

    @property
    def revision_candidates(self) -> List[Candidate]:
        """Step5: 只有 REVISION_CANDIDATE 才允许进入生成器。可为空。"""
        return [c for c in self.candidates if c.decision == "REVISION_CANDIDATE"]

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


# --- span 语义类别 (Step3 用) ---
SPAN_KINDS = (
    "object",              # 物体名
    "color_count_shape",   # 颜色/数量/形状
    "action_spatial",      # 动作/空间关系
    "negation_refusal",    # 否定/拒答 (有真实事实含义的)
    "other",               # 其他可独立核验的视觉事实
)

VERIFY_STATUSES = ("SUPPORTED", "UNSUPPORTED", "INSUFFICIENT_EVIDENCE", "NEEDS_VERIFICATION")
DECISIONS = ("SUSPICIOUS_BUT_UNVERIFIED", "REVISION_CANDIDATE", "PRESERVE")

CACHE_PATH = "outputs/cf_probe/haloquest_textcf_tokens.jsonl"
