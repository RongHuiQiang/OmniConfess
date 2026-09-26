# -*- coding: utf-8 -*-
"""
Output parsing and result saving utilities.
Output schema is compatible with GUARD/eval/final_eval.py expectations.

final_eval.py reads fields: pred, gt, group_key (and optionally sub_category/task/id/question)
"""
import re
import json
import os


# ================= Prediction Parsing =================

# Datasets that produce binary Yes/No answers
# Only cmm and phd are judgment tasks (yes/no); all others preserve raw_output
BINARY_DATASETS = {"cmm", "phd", "pope"}


def _parse_binary(text):
    """
    Parse Yes/No from model output. Returns 'yes' or 'no'.
    Logic verified from methods_text/base_model.py parse_pred() and
    GUARD/eval/final_eval.py _parse_yesno_answer().
    """
    text_clean = str(text).lower().strip()

    # \\boxed{yes} / \\boxed{no}
    boxed = re.search(r'\\boxed\{(.*?)\}', text_clean)
    if boxed:
        ans = boxed.group(1).strip()
        return 'yes' if 'yes' in ans else 'no'

    # "final answer: yes/no"
    fa = re.search(r'final answer:\s*(yes|no)', text_clean)
    if fa:
        return fa.group(1)

    # Answer:[yes/no]
    ans_tag = re.search(r'answer:\s*\[?(yes|no)\]?', text_clean)
    if ans_tag:
        return ans_tag.group(1)

    # judgment line
    jud = re.search(r'judgment:\s*(yes|no)', text_clean)
    if jud:
        return jud.group(1)

    # First word
    words = text_clean.split()
    if words and words[0] in ('yes', 'no'):
        return words[0]

    # Prefix check: handle "YesThe..." / "NoAs..." (no space after yes/no, Phi-4 style)
    m_prefix = re.match(r'^(Yes|No)(?=[^a-z]|$)', str(text).strip())
    if m_prefix:
        return m_prefix.group(1).lower()

    # Last 30 chars (GUARD final_eval fallback)
    tail = text_clean[-30:]
    if 'yes' in tail:
        return 'yes'
    if 'no' in tail:
        return 'no'

    return 'no'


_BAD_TEMPLATES = [
    "please think step by step, and then provide your final answer.",
    "please provide the final answer:",
    "answer:[your answer here]",
    "[your answer here]",
    "the reasoning steps are:",
    "please generate the next logical reasoning step.",
]


def _is_bad_template(text):
    t = text.lower().strip()
    return any(t == b or t.startswith(b) for b in _BAD_TEMPLATES)


def _strip_brackets(text):
    m = re.fullmatch(r'\[([^\]]{1,200})\]', text.strip())
    if m:
        inner = m.group(1).strip()
        if 'your answer' not in inner.lower():
            return '' if _is_bad_template(inner) else inner
    return text


def _parse_text_answer(text, dataset_name):
    """
    Extract free-text answer for generative datasets.
    Handles SC/PD/ToT prompt-template leakage, bracket placeholders,
    and long reasoning chains by finding the last Answer: marker.
    """
    if not text or not str(text).strip():
        return ''
    text_clean = str(text).strip()

    # Last "Final Answer:" match, first line only
    m = list(re.finditer(r'(?:Final Answer|final answer):\s*(.+)', text_clean, re.DOTALL))
    if m:
        ans = m[-1].group(1).strip().split('\n')[0].strip()
        if ans and not _is_bad_template(ans):
            return _strip_brackets(ans)

    # Last "Answer:" match, first line only
    m = list(re.finditer(r'Answer:\s*(.+)', text_clean, re.DOTALL | re.IGNORECASE))
    if m:
        ans = m[-1].group(1).strip().split('\n')[0].strip()
        if ans and not _is_bad_template(ans):
            return _strip_brackets(ans)

    # "The answer is ..."
    m = re.search(r'[Tt]he answer is[:\s]+(.+)', text_clean)
    if m:
        ans = m.group(1).split('\n')[0].strip().rstrip('.')
        if ans and not _is_bad_template(ans):
            return _strip_brackets(ans)

    # Whole text is a known bad template — nothing extractable
    if _is_bad_template(text_clean):
        return ''

    # Long text: recurse on last non-empty paragraph
    if len(text_clean) > 500:
        paragraphs = [p.strip() for p in text_clean.split('\n\n') if p.strip()]
        if paragraphs and paragraphs[-1] != text_clean:
            return _parse_text_answer(paragraphs[-1], dataset_name)

    return _strip_brackets(text_clean)


def parse_pred(raw_output, dataset_name):
    """
    Parse model raw output into a clean prediction string.
    Binary datasets -> 'yes' / 'no'
    Text datasets   -> extracted answer string
    """
    ds = dataset_name.lower()
    for key in BINARY_DATASETS:
        if key in ds:
            return _parse_binary(raw_output)
    return _parse_text_answer(raw_output, dataset_name)


# ================= Result Saving =================

def save_result(result_dict, output_path):
    """
    Append one result dict as a JSONL line to output_path.
    Flushes immediately so partial results survive crashes.

    Required fields for final_eval.py compatibility:
        pred, gt, group_key

    Full schema saved:
        id, question, dataset, method, pred, gt, group_key,
        raw_output (optional), cot (optional), time_sec (optional)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(result_dict, ensure_ascii=False) + '\n')
        f.flush()


def make_result(
    item,
    raw_output,
    dataset_name,
    method_name,
    time_sec=None,
    cot=None,
):
    """
    Build a result dict ready for save_result().
    item: normalized dict from data_loader.load_dataset()
    """
    pred = parse_pred(raw_output, dataset_name)
    result = {
        "id":          item.get('id', ''),
        "question":    item.get('question', ''),
        "gt":          item.get('gt', ''),
        "pred":        pred,
        "group_key":   item.get('group_key', 'Default'),
        # also store as sub_category so final_eval.py grouping works
        "sub_category": item.get('group_key', 'Default'),
        "dataset":     dataset_name,
        "method":      method_name,
    }
    if raw_output is not None:
        result["raw_output"] = raw_output
    if cot is not None:
        result["cot"] = cot
    if time_sec is not None:
        result["time_sec"] = round(time_sec, 3)
    return result
