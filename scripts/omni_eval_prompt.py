# -*- coding: utf-8 -*-
"""omni_eval_prompt.py — 七维多模态QA评估 prompt（用户定制版）

模板占位符（调用时用 str.format() 填入）：
  {question}            — 原始问题
  {reference_answers}   — 参考答案（可含多个，用分号分隔）
  {generated_answer}    — 待评估的生成答案
  {ground_truth}        — 参考答案文本，作为 evidence_grounding 的事实依据
                          （原始多模态内容不可用时用 GT 替代；可与 reference_answers 相同）

输出格式：JSON，七个维度各一个 0–10 整数。
"""

SYSTEM_PROMPT = (
    "You are an expert evaluator of question-answering quality.\n\n"
    "Evaluate the generated answer based on the question, the reference\n"
    "answers (which serve as ground-truth evidence), and the additional\n"
    "ground-truth context when provided.\n\n"
    "Score the answer independently across seven dimensions, using\n"
    "integer scores from 0 to 10."
)

USER_TEMPLATE = """\
[INPUT]

Question:
{question}

Reference Answers (treat these as ground-truth evidence):
{reference_answers}

Generated Answer:
{generated_answer}


[EVALUATION DIMENSIONS]

evidence_grounding: Whether the factual claims in the generated answer are
consistent with the reference answers, which serve as ground-truth evidence.
Judge agreement between the generated answer and the reference answers on every
factual claim. Disagreement with, contradiction of, or introduction of facts
absent from the reference answers counts against this score.

Scoring Guide (0–10):
- 10: All factual claims in the generated answer are consistent with the reference
  answers; no contradictions or unsupported assertions.
- 8–9: Strongly consistent, with only minor or peripheral inconsistencies.
- 6–7: Most key claims are consistent, but some details conflict with or are absent
  from the reference answers.
- 4–5: Only part of the answer is consistent; several important claims contradict or
  diverge from the reference answers.
- 1–3: Most claims conflict with or are unsupported by the reference answers.
- 0: Entirely contradicts or ignores the reference answers.

factual_correctness: Whether the answer provides correct information and avoids
factual errors or incorrect conclusions.

Scoring Guide (0–10):
- 10: Fully correct; all conclusions and factual statements are accurate.
- 8–9: Mostly correct, with only minor inaccuracies.
- 6–7: Generally correct, but contains noticeable factual errors.
- 4–5: Partially correct; several important conclusions or statements are incorrect.
- 1–3: Largely incorrect or misleading.
- 0: Entirely incorrect.

relevance: Whether the answer directly addresses the question and avoids
irrelevant or distracting content.

NOTE: Score this dimension based solely on whether the generated answer
addresses the question asked. Do NOT lower the score because the answer is
factually wrong or disagrees with the reference. A factually incorrect answer
that directly addresses the question should still score high here.

Scoring Guide (0–10):
- 10: Fully focused on the question; all content is relevant.
- 8–9: Highly relevant, with only minor digressions.
- 6–7: Generally relevant, but includes some unnecessary or distracting information.
- 4–5: Partially relevant; much of the answer does not directly address the question.
- 1–3: Barely related to the question.
- 0: Entirely irrelevant (e.g., answers a completely different question).

completeness: Whether the answer covers the information required by the question,
without unnecessary elaboration.

Scoring Guide (0–10):
- 10: Fully addresses all answerable aspects of the question, or appropriately
  states that the evidence is insufficient.
- 8–9: Covers nearly all required information, with only minor omissions.
- 6–7: Addresses the main question but omits some relevant information.
- 4–5: Covers only part of the required information.
- 1–3: Provides very limited information and misses most key aspects.
- 0: Fails to address the required information entirely.

logical_coherence: Whether the answer is internally consistent, logically
connected, and free from contradictions.

NOTE: Score this dimension based solely on the internal structure of the
generated answer. Do NOT lower the score because the answer disagrees with
the reference or is factually wrong. A factually incorrect but well-structured
answer should still score high here.

Scoring Guide (0–10):
- 10: Fully coherent and internally consistent.
- 8–9: Clear logical structure with only minor inconsistencies.
- 6–7: Generally coherent, but contains some weak connections or minor contradictions.
- 4–5: Noticeable inconsistencies or poorly connected statements.
- 1–3: Largely incoherent or internally contradictory.
- 0: Completely incoherent or self-contradictory.

uncertainty_calibration: Whether the answer expresses an appropriate level of
certainty given what the reference answers indicate is knowable.

Scoring Guide (0–10):
- 10: Certainty is fully aligned with what the reference answers support; unsupported
  claims are avoided, and uncertainty is expressed whenever the reference answers
  indicate the question cannot be answered definitively.
- 8–9: Well-calibrated, with only minor overconfidence or unnecessary hesitation.
- 6–7: Generally appropriate certainty, but some claims are overstated or excessively hedged.
- 4–5: Noticeable mismatch between expressed certainty and what the reference answers support.
- 1–3: Strong overconfidence or excessive uncertainty that substantially undermines the answer.
- 0: Completely inappropriate certainty, such as confidently asserting a claim that
  directly contradicts the reference answers.

clarity: Whether the answer is clearly expressed, concise, and easy to understand.

NOTE: Score this dimension based solely on the language quality of the generated
answer. Do NOT lower the score because the answer is factually wrong or disagrees
with the reference. A factually incorrect but well-written answer should still
score high here.

Scoring Guide (0–10):
- 10: Exceptionally clear and concise; immediately understandable.
- 8–9: Clear and readable, with only minor wording issues.
- 6–7: Generally understandable, but contains some ambiguity or unnecessary complexity.
- 4–5: Noticeably unclear or difficult to follow.
- 1–3: Poorly expressed and difficult to understand.
- 0: Unreadable or unintelligible.


[IMPORTANT EVALUATION RULES]

1. Evaluate each dimension independently.

2. Do not reward longer answers merely for providing more details.
   A concise answer may receive full scores in all dimensions.

3. When the evidence is insufficient to answer the question, an appropriate
   refusal or expression of uncertainty may receive full scores. Do not
   penalize an answer for refusing to invent unsupported information.

4. Do not reward unsupported details, even if they sound plausible.

5. Reference answers may contain multiple valid alternatives. Accept
   semantically equivalent correct answers.

6. Treat the reference answers as authoritative ground-truth evidence for
   evidence_grounding. Do not introduce visual, audio, or video details that
   go beyond what the reference answers state.

7. When the reference answers are the only available evidence, score
   evidence_grounding based on whether the generated answer's factual claims
   are consistent with them. Do not penalize an answer for omitting details
   that the reference answers contain, as long as it makes no incorrect claims.

8. Evaluate the generated answer without assuming that it was produced by any
   particular model or correction method.

9. Return integer scores only. Do not include explanations or additional text.

10. CRITICAL: relevance, logical_coherence, and clarity measure the QUALITY OF
    THE GENERATED ANSWER ITSELF — they are NEVER based on agreement with the
    reference answer. A factually wrong answer can still be highly relevant to
    the question, internally coherent, and clearly expressed. Score these three
    dimensions purely on the linguistic and structural quality of the generated
    answer:
    - relevance: does the generated answer address the question that was asked?
    - logical_coherence: is the generated answer internally consistent?
    - clarity: is the generated answer clearly and concisely written?
    Example: if the question is "What color are the cat's eyes?" and the generated
    answer is "The cat's eyes are green.", this answer is fully relevant (10),
    fully coherent (10), and fully clear (10), regardless of whether the reference
    says the cat does not exist. Factual accuracy is captured separately by
    factual_correctness and evidence_grounding.


[OUTPUT FORMAT]

{{
  "evidence_grounding": 0,
  "factual_correctness": 0,
  "relevance": 0,
  "completeness": 0,
  "logical_coherence": 0,
  "uncertainty_calibration": 0,
  "clarity": 0
}}"""

# 七个维度名，顺序与 prompt 一致
DIMENSIONS = [
    "evidence_grounding",
    "factual_correctness",
    "relevance",
    "completeness",
    "logical_coherence",
    "uncertainty_calibration",
    "clarity",
]

PROMPT_VERSION = "omni-eval-v3"


def build_messages(question: str, reference_answers: str,
                   generated_answer: str, ground_truth: str = ""):
    """返回 OpenAI-style messages list，可直接传给任意兼容 API。

    ground_truth: 参考答案文本，作为 evidence_grounding 的事实依据。
                  留空时自动沿用 reference_answers。
    """
    ref = reference_answers or ground_truth
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(
            question=question,
            reference_answers=ref,
            generated_answer=generated_answer,
        )},
    ]
