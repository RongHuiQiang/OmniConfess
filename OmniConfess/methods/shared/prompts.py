# -*- coding: utf-8 -*-
"""
All dataset prompts for Baselines methods.
Versions are selectable via --prompt_version v0/v1/v2/v3.

v0 = vanilla baseline (format constraints ONLY, no anti-hallucination instructions)
     Source: prompts1.py (merged 2026-04-19)
v1+ = tuned prompts with anti-hallucination instructions
     Source: extracted verbatim from existing tuned baseline files

Source index:
  CMM system  : GUARD/base_hook_best_0314.py get_system_prompt('cmm')
  CMM user v1 : methods_cmm/base_model_min.py
  CMM user v2 : methods_cmm/base_model_minest.py
  CMM user v3 : methods_cmm/base_model_2.py   (3-step cross-modal)
  CMM user v4 : methods_cmm/base_model_4.py   (strict fact-checker)

  AVH system v1: GUARD/base_hook_best_0314.py get_system_prompt('avh')   (forensic analyst)
  AVH system v2: methods_avh/base_model_0.2.py  zeroshot_map['avh']  (expert one-sentence)
  AVH system v3: methods_avh/base_model_0.6max.py zeroshot_map['avh'] (Qwen one-sentence)
  AVH user     : methods_avh/base_model_0.2.py  (user prompt = question only, all versions)

  PhD system  : GUARD/base_hook_best_0314.py get_system_prompt('phd')
  PhD user v1 : methods_phd/base_model.py     (degrade/intuitive)
  PhD user v2 : methods_phd/base_model_1.py   (evidence verifier)
  PhD user v3 : methods_phd/base_model_2.py   (critical visual analyst)

  HaloQuest system: GUARD/base_hook_best_0314.py get_system_prompt('halo')
  HaloQuest v1    : methods_haloquest/base_model_low.py      (clear direct)
  HaloQuest v2    : methods_haloquest/base_model_high.py     (strictly visible evidence)
  HaloQuest v3    : methods_haloquest/base_model_highest.py  (meticulous auditor)

  Text system  : GUARD/base_hook_best_0314.py per-dataset
  Text user v0 : prompts1.py (Read the following text…)
  Text user v1 : methods_text/base_model.py   (passage + instruction + question)
  Text user v2 : methods_text/base_model_2.py (Read the following text…)
  Text user v3 : methods_text/base_model_3.py (Document: / Question: / Answer:)
"""

# ===========================================================================
# System Prompts — v0 (vanilla: format ONLY, no anti-hallucination)
# ===========================================================================

_CMM_SYSTEM_V0 = (
    "You are an audio-visual reporter. Your task is to provide a final judgment based on what you see and hear.\n\n"
    "Judgment: \\boxed{Yes} or \\boxed{No}"
)
_AVH_SYSTEM_V0 = (
    "You are an AI assistant. Describe the video and audio content.\n"
    "Output EXACTLY ONE single sentence.\n"
    "At the end, output: Answer:[your answer here]."
)
_PHD_SYSTEM_V0 = (
    "You are a visual analyst. Your task is to evaluate a specific hypothesis against visual data.\n\n"
    "Judgment: \\boxed{Yes} or \\boxed{No}"
)
_HALO_SYSTEM_V0 = (
    "You are a visual assistant. Your task is to answer the question based on the provided image.\n"
    "Output EXACTLY ONE single sentence.\n"
    "At the end, output: Answer:[your answer here]."
)
_HALUEVAL_SYSTEM_V0 = (
    "You will be presented with a question.\n"
    "Answer the user's question based on the given information.\n"
    "At the end, output: Answer:[your answer here]."
)
_RAGTRUTH_SYSTEM_V0 = (
    "You will be presented with a question.\n"
    "Answer the user's question based on the given information.\n"
    "At the end, output: Answer:[your answer here]."
)
_PUBMEDQA_SYSTEM_V0 = (
    "You will be given a PubMed-style passage and a Yes/No/Maybe question.\n"
    "Answer rules:\n"
    "1. Begin with exactly one of: \"Yes.\" / \"No.\" / \"Maybe.\"\n"
    "2. Summarize the main conclusion from the passage in EXACTLY ONE single sentence.\n"
    "At the end, output: Answer:[your answer here]."
)
_DEFAULT_SYSTEM_V0 = (
    "You are a helpful AI assistant. Please provide answers to user questions based on the provided media content.\n"
    "At the end, output: Answer:[your answer here]."
)

# ===========================================================================
# System Prompts — v1+ (tuned: with anti-hallucination instructions)
# ===========================================================================

# ---------- CMM ----------
_CMM_SYSTEM = (
    "You are an expert audio-visual reporter. Your task is to provide a single-sentence narrative "
    "combining what you see and hear, followed by a final judgment.\n\n"
    "STRICT OUTPUT FORMAT:\n"
    "Description: [One flowing sentence describing the scene and the EXACT sounds heard, like: "
    "'In a room, a man spoke by a window where birds were seen but not heard.']\n"
    "Judgment: \\boxed{Yes} or \\boxed{No}\n\n"
    "STRICT AUDIT RULES:\n"
    "1. If a visual object (like a bird) is present but its specific sound (chirping) is NOT in the audio, "
    "you MUST state that it was not heard in the Description.\n"
    "2. The Description must be a single, descriptive sentence.\n"
    "3. Judgment must be based ONLY on what is audible."
)

# ---------- AVH (3 system-prompt variants, source above) ----------
_AVH_SYSTEM_V1 = (
    "You are an objective audiovisual forensic analyst. Your task is to decode the video and audio streams "
    "into a high-density, expert-level description in ONE SHORT sentence.\n"
    "CRITICAL ANTI-HALLUCINATION DIRECTIVES:\n"
    "1. MODALITY INDEPENDENCE: Analyze what you SEE and HEAR independently. DO NOT let the visual scene "
    "change your interpretation of the audio, and vice versa.\n"
    "2. NO SEMANTIC SMOOTHING: If visuals and audio seem contradictory (e.g., a forest visual but an engine "
    "sound), report BOTH exactly as they are. Do not invent details like 'chirping birds' to make it natural.\n"
    "3. AVOID ASSUMPTIONS: Rely on audio to confirm ambiguous visuals (e.g., a 'baaing' sound confirms a "
    "goat, do not assume it is a cat).\n\n"
    "FORMATTING INSTRUCTIONS:\n"
    "1. Compression: Pack all objective sensory details into a maximum of 30 words.\n"
    "2. Directness: Skip all introductory phrases. Start immediately with the main subject.\n"
    "3. Keywords: Prioritize specific nouns and adjectives.\n"
    "CORE RULE: Be extremely dense, strictly factual, and under 30 words. Final answer in \\boxed{}."
)

_AVH_SYSTEM_V2 = (
    "You are an expert audio-visual AI assistant. Your task is to describe the video and audio content.\n\n"
    "Output EXACTLY ONE single, flowing sentence describing what you see and hear.\n"
    "At the end, output: Answer:[your answer here]."
)

_AVH_SYSTEM_V3 = (
    "You are Qwen, a virtual human developed by Alibaba Group. You can see and hear. "
    "Output EXACTLY ONE single, flowing sentence describing what you see and hear.\n"
    "At the end, output: Answer:[your answer here]."
)

_AVH_SYSTEMS = {'v1': _AVH_SYSTEM_V1, 'v2': _AVH_SYSTEM_V2, 'v3': _AVH_SYSTEM_V3}

# ---------- PhD ----------
_PHD_SYSTEM = (
    "You are a scientific visual analyst. Your task is to evaluate a specific hypothesis against raw visual data.\n\n"
    "EVALUATION RULES:\n"
    "1. Treat the provided text context as an unverified background story. It may contain errors.\n"
    "2. Treat the final question as a 'Hypothesis' that needs to be tested.\n"
    "3. Look for explicit visual confirmation or contradiction in the image. "
    "Do not rely on assumptions or the text's claims.\n\n"
    "STRICT OUTPUT FORMAT:\n"
    "Hypothesis to Test: [Restate the core question simply]\n"
    "Image Observation:\n- [Observation 1]\n- [Observation 2]\n"
    "Judgment: \\boxed{Yes} or \\boxed{No}"
)

# ---------- HaloQuest ----------
_HALO_SYSTEM = (
    "You are an expert visual assistant. Your task is to answer the question based ONLY on the provided image.\n"
    "CRITICAL: If the question asks about a detail or object that is NOT present in the image, "
    "you MUST explicitly state that it is not present or visible.\n"
    "Be concise and strictly factual."
)

# ---------- Text datasets ----------
_HALUEVAL_SYSTEM = (
    "You will be presented with a question.\n"
    "Answer the user's question strictly based on the given information.\n"
    "Do not make up information.\n"
    "At the end, output: Answer:[your answer here]."
)

_RAGTRUTH_SYSTEM = (
    "You are given passages and a question. Follow these steps:\n"
    "Answer the question using only the information from the given passages.\n"
    " - Include specific examples, numbers, or comparisons if mentioned.\n"
    " - Include all details that support the answer.\n"
    " - Do not add external information.\n"
    " - If the passages do not contain sufficient information, "
    "answer: \"Unable to answer based on given passages.\"\n"
    "At the end, output: Answer:[your answer here]"
)

_PUBMEDQA_SYSTEM = (
    "You will be given a PubMed-style passage and a Yes/No/Maybe question.\n"
    "Answer rules:\n"
    "1. Begin with exactly one of: \"Yes.\" / \"No.\" / \"Maybe.\"\n"
    "2. Summarize the main conclusion from the passage in exactly ONE short sentence (\u226425 words).\n"
    "3. Preserve key phrases and medical terms from the passage; do not replace them with synonyms.\n"
    "4. Always include explicitly stated conditions, subgroups, or limitations if they appear in the conclusion.\n"
    "5. Do NOT add recommendations, explanations, or new information.\n"
    "6. The final output must be exactly ONE LINE:\n"
    "Answer:[Yes./No./Maybe. + short sentence]"
)

_DEFAULT_SYSTEM = (
    "You are a helpful AI assistant. Please provide quick and intuitive answers "
    "to user questions based on the provided media content."
)


def get_system_prompt(dataset_name, prompt_version='v1'):
    """
    Return the system prompt for the given dataset.
    v0 = vanilla (format only), v1+ = tuned (anti-hallucination).
    """
    ds = dataset_name.lower()

    if prompt_version == 'v0':
        if 'cmm' in ds:     return _CMM_SYSTEM_V0
        if 'avh' in ds:     return _AVH_SYSTEM_V0
        if 'phd' in ds:     return _PHD_SYSTEM_V0
        if 'halo' in ds:    return _HALO_SYSTEM_V0
        if 'halueval' in ds: return _HALUEVAL_SYSTEM_V0
        if 'ragtruth' in ds: return _RAGTRUTH_SYSTEM_V0
        if 'pubmedqa' in ds: return _PUBMEDQA_SYSTEM_V0
        if 'summedits' in ds:
            return (
                "You are given a document and a summary. Your task is to determine whether the summary is "
                "factually consistent with the document. Answer only 'Yes' if the summary is consistent, "
                "or 'No' if it contains any factual inconsistency."
            )
        return _DEFAULT_SYSTEM_V0

    # v1+
    if 'cmm' in ds:
        return _CMM_SYSTEM
    if 'avh' in ds:
        return _AVH_SYSTEMS.get(prompt_version, _AVH_SYSTEM_V1)
    if 'phd' in ds:
        return _PHD_SYSTEM
    if 'halo' in ds:
        return _HALO_SYSTEM
    if 'halueval' in ds:
        return _HALUEVAL_SYSTEM
    if 'ragtruth' in ds:
        return _RAGTRUTH_SYSTEM
    if 'pubmedqa' in ds:
        return _PUBMEDQA_SYSTEM
    if 'summedits' in ds:
        return (
            "You are given a document and a summary. Your task is to determine whether the summary is "
            "factually consistent with the document. Answer only 'Yes' if the summary is consistent, "
            "or 'No' if it contains any factual inconsistency."
        )
    return _DEFAULT_SYSTEM


# ===========================================================================
# User Prompts
# ===========================================================================

def get_user_prompt(q_text, dataset_name, prompt_version='v1', passage=None):
    """
    Return the user prompt string (WITHOUT media-token prefix).
    prompt_version: 'v0' | 'v1' | 'v2' | 'v3'

    v0 = vanilla (barebones, from prompts1.py)
    """
    ds = dataset_name.lower()

    # ── v0: vanilla baseline (all datasets) ──────────────────────────────────
    if prompt_version == 'v0':
        passage_str = passage or ''
        if 'cmm' in ds:
            return f"Task: Analyze the video and audio to answer the question.\nQuestion: {q_text}\nFinal Answer:"
        if 'avh' in ds:
            return q_text
        if 'phd' in ds or 'halo' in ds:
            return f"Question: {q_text}\nAnswer:"
        if 'summedits' in ds:
            return f"Document: {passage_str}\n\n{q_text}\n\nFinal Answer:"
        # Text datasets: HaluEval / RAGTruth / PubMedQA
        return (
            f"Read the following text and answer the question.\n\n"
            f"Text: {passage_str}\n\n"
            f"Question: {q_text}\n\n"
            "Answer:"
        )

    # ── v1+ below ────────────────────────────────────────────────────────────

    # ── CMM ──────────────────────────────────────────────────────────────────
    if 'cmm' in ds:
        if prompt_version == 'v2':
            return (
                f"Analyze the media carefully. First, briefly list the key visual objects and audio events observed. "
                f"Then, based on that list, determine if the statement '{q_text}' is factually accurate. "
                "Final Answer: [Yes/No]"
            )
        if prompt_version == 'v3':
            return (
                f"Task: Verify the existence of '{q_text}' through independent modal cross-checking.\n"
                "Step 1: Check the video stream for visual evidence.\n"
                "Step 2: Check the audio stream for auditory evidence.\n"
                "Step 3: Compare both sources. Only confirm 'Yes' if there is direct, non-conflicting evidence. "
                "If one modality suggests it but the other contradicts, prioritize empirical observation over intuition.\n"
                "Final Answer:"
            )
        if prompt_version == 'v4':
            return f"Analyze the video and audio inputs to verify the following: {q_text}. Answer with Yes or No."
        # v1
        return f"Task: Analyze the video and audio to answer the question.\nQuestion: {q_text}\nFinal Answer:"

    # ── AVH ──────────────────────────────────────────────────────────────────
    if 'avh' in ds:
        return q_text

    # ── PhD ──────────────────────────────────────────────────────────────────
    if 'phd' in ds:
        if prompt_version == 'v2':
            instruction = (
                "Act as an impartial evidence verifier. Carefully evaluate the visual details against "
                "the provided description. If a discrepancy exists, prioritize direct visual observations "
                "over contextual expectations. Ensure the response is strictly derived from the image content."
            )
        elif prompt_version == 'v3':
            instruction = (
                "Operate as a critical visual analyst. Treat the provided background information as a "
                "hypothesis to be tested against the image. Only confirm details that are explicitly visible, "
                "and actively filter out any inferences drawn solely from the text description."
            )
        else:
            instruction = (
                "Provide a direct and intuitive answer based on the provided context and overall scene. "
                "Use your common sense to fill in details. Accuracy to visual specifics is not required. "
            )
        return f"Instruction: {instruction}\nQuestion: {q_text}\nFinal Answer: "

    # ── HaloQuest ─────────────────────────────────────────────────────────────
    if 'halo' in ds:
        if prompt_version == 'v2':
            return (
                f"Question: {q_text}\n"
                "Analyze the image objectively. Answer the question relying strictly on the visible evidence, "
                "avoiding any unverified assumptions."
            )
        if prompt_version == 'v3':
            return (
                f"Question: {q_text}\n"
                "Act as a meticulous visual auditor. Before answering, consider the possibility that the "
                "object is absent. Only confirm its presence if you can identify unique, non-generic visual "
                "features that distinguish it from the background. Prioritize precision over recall; avoid "
                "any common-sense completions."
            )
        # v1
        return (
            f"Question: {q_text}\n"
            "Provide a clear, direct, and concise answer to the question based on the visual content."
        )

    # ── SummEdits ─────────────────────────────────────────────────────────────
    if 'summedits' in ds:
        return (
            f"Document: {passage or ''}\n\n"
            f"{q_text}\n\n"
            "Final Answer:"
        )

    # ── Text datasets: HaluEval / RAGTruth / PubMedQA ────────────────────────
    passage_str = passage or ''
    if prompt_version == 'v2':
        return (
            f"Read the following text and answer the question.\n\n"
            f"Text: {passage_str}\n\n"
            f"Question: {q_text}\n\n"
            "Answer:"
        )
    if prompt_version == 'v3':
        return f"Document: {passage_str}\nQuestion: {q_text}\nAnswer:"

    # v1
    return (
        f"Passage: {passage_str}\n"
        "Instruction: Please answer the following question based on the provided passage "
        "and your general knowledge.\n"
        f"Question: {q_text}\n"
        "Answer:"
    )
