# -*- coding: utf-8 -*-
"""
halueval_prompts.py — HaluEval prompt versions for OmniConfess (omniconfess.py)

Dataset profile (halueval_stratified60, 60 samples):
  - yes_no      :  4 (GT = "yes" / "no")
  - single_entity: 17 (GT = 1 word)
  - short_phrase : 33 (GT = 2-4 words)
  - multi_word   :  6 (GT ≥ 5 words)

v0 failure analysis (2026-04-14):
  - yes_no catastrophe: OmniConfess never outputs bare "Yes"/"No", inflates to 25 words avg
    → F1 = 25.48 vs base 80.56 (diff = −55.08), accounts for ~80% of total deficit
  - single_entity verbosity: "19 years old" instead of "19", 6.4x inflation
  - short_phrase occasional hallucination: "0.6 miles", "Paris", "No information provided"

v1 fix strategy:
  - Explicit yes/no rule in system prompt + 4 few-shots covering yes/no and entity types
  - Final prompt: force stop after answer, separate yes/no vs other instruction

Each version defines:
  system   : str — system prompt
  final    : str — final instruction; use {notes} for reasoning_history
  max_tok  : int — max_new_tokens for the final generation step
"""

HALUEVAL_PROMPTS = {

    # ── v0: original baseline ─────────────────────────────────────────────────
    "v0": {
        "system": (
            "You will be presented with a question.\n"
            "Answer the user's question strictly based on the given information.\n"
            "Do not make up information.\n"
            "Output ONLY the answer as a SHORT PHRASE (1-5 words). No full sentences, no explanation."
        ),
        "final": (
            "\n\n[Internal Reasoning]:\n{notes}\n\n"
            "CRITICAL INSTRUCTION:\n"
            "Based on your reasoning above, output ONLY the answer as a short phrase (1-5 words). No explanation, no full sentence.\n"
            "Answer:"
        ),
        "max_tok": 100,
    },

    # ── v1: yes/no fix + few-shot anchoring ──────────────────────────────────
    #
    # Key changes vs v0:
    #   Problem: OmniConfess outputs "No, they did not." (4w) instead of "No" (1w) for yes/no,
    #            or "Yes, Phil Mogg and Dave Peters are musicians." (8w) instead of "Yes".
    #            → Token-F1 collapses from 100 to 20-40 due to precision dilution.
    #   Fix 1 — Explicit yes/no rule: "answer with ONLY 'Yes' or 'No'. Do not add anything."
    #   Fix 2 — 4 few-shots: 2 yes/no + 1 single-entity + 1 short-phrase, covering the
    #            most failure-prone answer types. No multi-word examples (those already work).
    #   Fix 3 — Final prompt: separate yes/no instruction + hard STOP signal.
    #   max_tok: 50
    #     yes_no stops at 1-2 tokens by instruction; max_tok is just a ceiling.
    #     multi_word longest GT: id=686 = 39 tokens (22 words). max_tok=50 covers all 6.
    "v1": {
        "system": (
            "You will be presented with a question.\n"
            "Answer the user's question strictly based on the given information.\n"
            "Do not make up information.\n\n"
            "OUTPUT RULES:\n"
            "• For yes/no questions: answer with ONLY 'Yes' or 'No'. Do not add any explanation.\n"
            "• For all other questions: output the answer as a SHORT PHRASE (1-5 words). No full sentences.\n\n"
            "Examples:\n"
            "  Q: \"Is John a doctor?\" (passage says he is)\n"
            "  A: Yes.\n\n"
            "  Q: \"Did the company merge with X?\" (passage says no)\n"
            "  A: No.\n\n"
            "  Q: \"Who wrote the song?\" (passage gives name)\n"
            "  A: Lindsey Stirling.\n\n"
            "  Q: \"When was it released?\" (passage gives year)\n"
            "  A: 1999."
        ),
        "final": (
            "\n\n[Internal Reasoning]:\n{notes}\n\n"
            "CRITICAL INSTRUCTION:\n"
            "Based on your reasoning above, output ONLY the answer.\n"
            "• If this is a yes/no question: answer with ONLY 'Yes' or 'No'.\n"
            "• Otherwise: output a SHORT PHRASE (1-5 words).\n"
            "Output ONLY the answer. STOP after the answer.\n"
            "Answer:"
        ),
        "max_tok": 50,
    },

    # ── v2: yes/no fix only, no few-shot ─────────────────────────────────────
    #
    # Diagnosis of v1 (2026-04-16):
    #   - yes_no: 100.00  (v1 fixed it perfectly)
    #   - short_phrase: 54.01  (v1 few-shots + heavy OUTPUT RULES caused model to
    #     answer "Michael Haneke" as "No", "pro-life" as "A: Paris")
    #   Root cause: 4 few-shots biased the model toward yes/no format for all answers.
    #
    # v2 strategy: surgical fix only
    #   - System prompt: exact v0 text (no OUTPUT RULES, no few-shots)
    #   - Final prompt: v0 text + one conditional yes/no line
    #   - max_tok: 100 (same as v0)
    "v2": {
        "system": (
            "You will be presented with a question.\n"
            "Answer the user's question strictly based on the given information.\n"
            "Do not make up information.\n"
            "Output ONLY the answer as a SHORT PHRASE (1-5 words). No full sentences, no explanation."
        ),
        "final": (
            "\n\n[Internal Reasoning]:\n{notes}\n\n"
            "CRITICAL INSTRUCTION:\n"
            "Based on your reasoning above, write ONLY the answer, nothing else.\n"
            "If this is a yes/no question, answer with ONLY \"Yes\" or \"No\".\n"
            "Otherwise, output a short phrase (1-5 words). No explanation, no full sentence.\n"
            "Answer:"
        ),
        "max_tok": 100,
    },

}


def get_halueval_prompt(version: str):
    """Return (system_prompt, final_template, max_new_tokens) for a given version."""
    cfg = HALUEVAL_PROMPTS.get(version, HALUEVAL_PROMPTS["v0"])
    return cfg["system"], cfg["final"], cfg["max_tok"]
