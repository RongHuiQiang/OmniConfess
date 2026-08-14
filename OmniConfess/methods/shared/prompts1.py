# -*- coding: utf-8 -*-
"""
All dataset prompts for Baselines methods.
Strictly vanilla baseline: Format constraints ONLY. NO anti-hallucination instructions.
Only v0 is retained.
"""

# ===========================================================================
# System Prompts (Format ONLY, No Anti-Hallucination)
# ===========================================================================

# ---------- CMM (Judgment) ----------
_CMM_SYSTEM = (
    "You are an audio-visual reporter. Your task is to provide a final judgment based on what you see and hear.\n\n"
    "Judgment: \\boxed{Yes} or \\boxed{No}"
)

# ---------- PhD (Judgment) ----------
_PHD_SYSTEM = (
    "You are a visual analyst. Your task is to evaluate a specific hypothesis against visual data.\n\n"
    "Judgment: \\boxed{Yes} or \\boxed{No}"
)
# ---------- AVH (Descriptive) ----------
_AVH_SYSTEM = (
    "You are an AI assistant. Describe the video and audio content.\n"
    "Output EXACTLY ONE single sentence.\n"
    "At the end, output: Answer:[your answer here]."
)

# ---------- HaloQuest (Descriptive) ----------
_HALO_SYSTEM = (
    "You are a visual assistant. Your task is to answer the question based on the provided image.\n"
    "Output EXACTLY ONE single sentence.\n"
    "At the end, output: Answer:[your answer here]."
)

# ---------- Text datasets (Descriptive) ----------
_HALUEVAL_SYSTEM = (
    "You will be presented with a question.\n"
    "Answer the user's question based on the given information.\n"
    "At the end, output: Answer:[your answer here]."
)

_RAGTRUTH_SYSTEM = (
    "You will be presented with a question.\n"
    "Answer the user's question based on the given information.\n"
    "At the end, output: Answer:[your answer here]."
)

_PUBMEDQA_SYSTEM = (
    "You will be given a PubMed-style passage and a Yes/No/Maybe question.\n"
    "Answer rules:\n"
    "1. Begin with exactly one of: \"Yes.\" / \"No.\" / \"Maybe.\"\n"
    "2. Summarize the main conclusion from the passage in EXACTLY ONE single sentence.\n"
    "At the end, output: Answer:[your answer here]."
)

_DEFAULT_SYSTEM = (
    "You are a helpful AI assistant. Please provide answers to user questions based on the provided media content.\n"
    "At the end, output: Answer:[your answer here]."
)

def get_system_prompt(dataset_name, prompt_version='v0'):
    """
    Return the vanilla system prompt for the given dataset.
    All anti-hallucination variations have been stripped.
    """
    ds = dataset_name.lower()
    
    if 'cmm' in ds: return _CMM_SYSTEM
    if 'avh' in ds: return _AVH_SYSTEM
    if 'phd' in ds: return _PHD_SYSTEM
    if 'halo' in ds: return _HALO_SYSTEM
    if 'halueval' in ds: return _HALUEVAL_SYSTEM
    if 'ragtruth' in ds: return _RAGTRUTH_SYSTEM
    if 'pubmedqa' in ds: return _PUBMEDQA_SYSTEM
    if 'summedits' in ds:
        return (
            "You are given a document and a summary. Your task is to determine whether the summary is "
            "factually consistent with the document. Answer only 'Yes' if the summary is consistent, "
            "or 'No' if it contains any factual inconsistency."
        )

    return _DEFAULT_SYSTEM


# ===========================================================================
# User Prompts (Question ONLY)
# ===========================================================================

def get_user_prompt(q_text, dataset_name, prompt_version='v0', passage=None):
    """
    Return the barebones user prompt string.
    """
    ds = dataset_name.lower()
    passage_str = passage or ''

    # ── CMM ──────────────────────────────────────────────────────────────────
    if 'cmm' in ds:
        return f"Task: Analyze the video and audio to answer the question.\nQuestion: {q_text}\nFinal Answer:"

    # ── AVH ──────────────────────────────────────────────────────────────────
    if 'avh' in ds:
        return q_text

    # ── PhD & HaloQuest ──────────────────────────────────────────────────────
    if 'phd' in ds or 'halo' in ds:
        return f"Question: {q_text}\nAnswer:"

    # ── SummEdits ─────────────────────────────────────────────────────────────
    if 'summedits' in ds:
        return (
            f"Document: {passage_str}\n\n"
            f"{q_text}\n\n"
            "Final Answer:"
        )

    # ── Text datasets: HaluEval / RAGTruth / PubMedQA ────────────────────────
    return (
        f"Read the following text and answer the question.\n\n"
        f"Text: {passage_str}\n\n"
        f"Question: {q_text}\n\n"
        "Answer:"
    )