# -*- coding: utf-8 -*-
"""
avh_prompts.py — AVHBench prompt versions for GUARD (base_hook_best_0317.py)

Dataset profile (avh_stratified60, 60 samples):
  - All AV Captioning tasks: GT = single sentence describing audio + visual
  - GT avg length: 22.9 words (median 23)
  - base_model avg response: 16.3 words
  - GUARD v0 avg response:   31.1 words  ← 35% longer than GT

v0 failure analysis (2026-04-14):
  - Primary cause: GUARD verbose (31w) vs GT (23w) vs base (16w) → word mismatch hurts precision
  - Secondary: occasional generation crashes (Korean text leak, prompt fragment leak, truncation)
  - Fallback rate 75%, but fallback group F1 ≈ non-fallback group F1 → path routing not the issue
  - Audio/visual coverage OK (9/10 samples cover both modalities)

v1 fix strategy:
  - Hard 25-word cap: GT avg is 23w; base 16w; cap 25w forces brevity without cutting below GT
  - English-only: blocks Korean/non-English transcription leaks
  - No-system-instructions: blocks prompt fragment leaks
  - Final prompt adds "≤25 words, then STOP" as a generation fence
  - NO few-shots: avh v5 experience showed few-shot over-specialization causes regression

Each version defines:
  system   : str — system prompt
  final    : str — final instruction; use {notes} for reasoning_history
  max_tok  : int — max_new_tokens for the final generation step
"""

AVH_PROMPTS = {

    # ── v0: original baseline ─────────────────────────────────────────────────
    "v0": {
        "system": (
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
        ),
        "final": (
            "\n\n[Your observations]:\n{notes}\n\n"
            "Based on your observations, write one sentence describing what is happening in this audio-visual clip.\n"
            "Answer: "
        ),
        "max_tok": 60,
    },

    # ── v2: v0 system (unchanged) + minimal final-prompt constraints ─────────
    #
    # Strategy: keep system prompt identical to v0 (proven to produce good notes),
    # only add soft constraints in the final prompt:
    #   - ~20 words (soft brevity guide)
    #   - English only (blocks language leak)
    #   - No phrase repetition
    "v2": {
        "system": (
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
        ),
        "final": (
            "\n\n[Your observations]:\n{notes}\n\n"
            "Based on your observations, write ONE sentence of approximately 20 words describing what you see and hear. "
            "Use English only. Do not repeat the same phrase.\n"
            "Answer: "
        ),
        "max_tok": 60,
    },

    # ── v2qa: DIAGNOSTIC ONLY — v2 with original Q/A format (for A/B comparison) ──
    "v2qa": {
        "system": (
            "You are an audiovisual analyst. Describe BOTH what you see AND what you hear in ONE concise English sentence.\n\n"
            "Rules:\n"
            "1. Cover both modalities: integrate sight and sound into a single natural description.\n"
            "2. Output only in English. Do not transcribe foreign-language speech; instead describe the speaker's tone or setting.\n"
            "3. Keep it concise — about 20 words. Be direct — no introductory phrases.\n"
            "4. Match the phrasing naturally. Do not repeat the same phrase.\n\n"
            "Examples:\n"
            "Q: Describe what you see and hear in a single sentence.\n"
            "A: The wind whispers through the leaves as a leaf blower hums, while a black snake slithers in the grass near a tree.\n\n"
            "Q: Describe what you see and hear in a single sentence.\n"
            "A: The rhythmic tick-tock of a large clock with visible gears accompanies a man hanging a black and white clock on the wall in his house.\n\n"
            "Q: Describe what you see and hear in a single sentence.\n"
            "A: An old truck idles next to a barn, its engine humming and rattling, while bells ring in the background.\n\n"
            "Q: Describe what you see and hear in a single sentence.\n"
            "A: The tribal drums beat as a green frog sits in a natural environment, surrounded by the sounds of shuffling footsteps, chirping crickets, and wet dirt.\n"
        ),
        "final": (
            "\n\n[Your observations]:\n{notes}\n\n"
            "Based on your observations, write ONE sentence describing what you see and hear. "
            "Match the style of the examples above.\n"
            "Description: "
        ),
        "max_tok": 60,
    },

    # ── v1: 25-word cap + English-only + no-leak constraint ──────────────────
    #
    # Key changes vs v0:
    #   Problem 1 — GUARD response 31.1w >> GT 22.9w → precision penalty
    #               Fix: tighten cap from 50w → 25w (just above GT avg 22.9w)
    #   Problem 2 — Korean text transcription leaking into output (id=51)
    #               Fix: "Use ONLY English." explicit rule
    #   Problem 3 — Prompt fragments leaking ("hear in a single sentence", id=12)
    #               Fix: "DO NOT include any system instructions or non-English text"
    #   Final prompt adds hard fence: "≤25 words, then STOP"
    #   max_tok: 50 (25 words × ~2 tokens/word; leave margin)
    #   NO few-shots: past v5 test showed few-shot over-specialization causes regression
    "v1": {
        "system": (
            "You are an objective audiovisual forensic analyst. Your task is to describe the video and audio streams in ONE sentence.\n"
            "CRITICAL ANTI-HALLUCINATION DIRECTIVES:\n"
            "1. MODALITY INDEPENDENCE: Analyze what you SEE and HEAR independently. DO NOT let the visual scene change your interpretation of the audio, and vice versa.\n"
            "2. NO SEMANTIC SMOOTHING: If visuals and audio seem contradictory, report BOTH exactly as they are. Do not invent details.\n"
            "3. AVOID ASSUMPTIONS: Rely on audio to confirm ambiguous visuals.\n\n"
            "FORMATTING INSTRUCTIONS:\n"
            "1. Describe what you see and hear in ONE sentence, no longer than 25 words.\n"
            "2. Use ONLY English. DO NOT include system instructions, examples, or non-English text in your output. If you start generating non-English text, stop and restart with English only.\n"
            "3. Skip all introductory phrases. Start immediately with the main subject.\n"
            "CORE RULE: ONE English sentence, ≤25 words, strictly factual. No formatting wrappers."
        ),
        "final": (
            "\n\n[Your observations]:\n{notes}\n\n"
            "Answer in ONE sentence (≤25 words), then STOP.\n"
            "Description: "
        ),
        "max_tok": 50,
    },

    # ── v3_simple: modality-separated single-pass description ─────────────
    #
    # Strategy: force the model to describe visual and audio streams separately
    # within a single generation pass, preventing cross-modal contamination.
    # The "(1)/(2)" structure encourages modality independence at prompt level.
    "v3_simple": {
        "system": (
            "You are an audiovisual analyst. Your task is to describe a video clip by listing what you see and what you hear SEPARATELY.\n\n"
            "CRITICAL RULES:\n"
            "1. MODALITY INDEPENDENCE: Report visual observations and audio observations independently. DO NOT let one modality influence the other.\n"
            "2. NO ASSUMPTIONS: Only describe what is directly observable/audible. If you see a dog but hear no barking, do NOT mention barking.\n"
            "3. BREVITY: Each part should be ~10 words. Total output under 30 words.\n"
            "4. English only. No formatting wrappers."
        ),
        "final": (
            "\n\n[Your observations]:\n{notes}\n\n"
            "Based on your observations, list separately:\n"
            "(1) What you SEE in the video.\n"
            "(2) What you HEAR in the audio.\n"
            "Be specific and brief (~10 words each).\n"
            "Answer: "
        ),
        "max_tok": 60,
    },

    # ══════════════════════════════════════════════════════════════════════════
    #
    #
    #   "high-density" / "Pack all objective sensory details into a maximum of 50
    #    words" / "Prioritize specific nouns and adjectives" / "extremely dense"
    #
    #     "while the audio features …" / "as depicted in the visual and auditory
    #
    # ══════════════════════════════════════════════════════════════════════════

    "v4_caption": {
        "system": (
            "You are an audiovisual describer. You write one short, natural caption for a video clip, "
            "covering both what is seen and what is heard.\n"
            "CRITICAL ANTI-HALLUCINATION DIRECTIVES:\n"
            "1. MODALITY INDEPENDENCE: Analyze what you SEE and HEAR independently. DO NOT let the visual scene change your interpretation of the audio, and vice versa.\n"
            "2. NO SEMANTIC SMOOTHING: If visuals and audio seem contradictory, report BOTH exactly as they are. Do not invent details.\n"
            "3. AVOID ASSUMPTIONS: Rely on audio to confirm ambiguous visuals.\n\n"
            "STYLE INSTRUCTIONS:\n"
            "1. Length: ONE sentence of about 18 words. Not a list, not two sentences.\n"
            "2. Caption register: name the main subject, its action, and the main sound. Nothing else.\n"
            "3. Leave out secondary detail: no colours, positions, brands, on-screen text, or barely visible objects, unless one of them IS the main subject.\n"
            "4. No hedging, and no commentary about the clip, the streams, or your own observations.\n"
            "CORE RULE: one plain English sentence, about 18 words, plainly worded. No formatting wrappers."
        ),
        "final": (
            "\n\n[Your observations]:\n{notes}\n\n"
            "Write ONE natural caption sentence of about 18 words describing what happens in this clip.\n"
            "Answer: "
        ),
        "max_tok": 48,
    },

    "v5_integrate": {
        "system": (
            "You are an audiovisual describer. You write one short, natural caption for a video clip, "
            "covering both what is seen and what is heard.\n"
            "CRITICAL ANTI-HALLUCINATION DIRECTIVES:\n"
            "1. MODALITY INDEPENDENCE: Analyze what you SEE and HEAR independently. DO NOT let the visual scene change your interpretation of the audio, and vice versa.\n"
            "2. NO SEMANTIC SMOOTHING: If visuals and audio seem contradictory, report BOTH exactly as they are. Do not invent details.\n"
            "3. AVOID ASSUMPTIONS: Rely on audio to confirm ambiguous visuals.\n\n"
            "STYLE INSTRUCTIONS:\n"
            "1. Length: ONE sentence of about 18 words. Not a list, not two sentences.\n"
            "2. Caption register: name the main subject, its action, and the main sound. Nothing else.\n"
            "3. Leave out secondary detail: no colours, positions, brands, on-screen text, or barely visible objects, unless one of them IS the main subject.\n"
            "4. No hedging, and no commentary about the clip, the streams, or your own observations.\n"
            "5. INTEGRATE, DO NOT LABEL: weave the sights and the sounds into one flowing sentence. "
            "Never write the words 'audio', 'visual', 'video', 'stream', 'observation', or 'clip' in your answer.\n"
            "CORE RULE: one plain English sentence, about 18 words, plainly worded. No formatting wrappers."
        ),
        "final": (
            "\n\n[Your observations, listed by modality]:\n{notes}\n\n"
            "Merge these into ONE natural caption sentence of about 18 words. "
            "Do not label the modalities and do not use the words 'audio', 'visual' or 'stream'.\n"
            "Answer: "
        ),
        "max_tok": 48,
    },

    #     "You are an AI assistant. Describe the video and audio content.
    #      Output EXACTLY ONE single sentence.
    #      At the end, output: Answer:[your answer here]."
    "v6_lean": {
        "system": (
            "You are an AI assistant. Describe the video and audio content.\n"
            "Output EXACTLY ONE natural sentence of about 18 words, covering what is seen and what is heard.\n"
            "Do not invent anything that is not in the video or the audio; if sight and sound conflict, report both as they are.\n"
            "Name the main subject, its action, and the main sound; leave out secondary detail. "
            "Do not use the words 'audio', 'visual' or 'stream'."
        ),
        "final": (
            "\n\n[Your observations]:\n{notes}\n\n"
            "Describe what you see and hear in a single sentence.\n"
            "Answer: "
        ),
        "max_tok": 40,
    },

    #
    #
    "v7_entity": {
        "system": (
            "You are an AI assistant. Describe the video and audio content.\n"
            "Output EXACTLY ONE natural sentence of about 18 words, covering what is seen and what is heard.\n"
            "Do not invent anything that is not in the video or the audio; if sight and sound conflict, report both as they are.\n"
            "Name AT MOST THREE things: the main subject, what it is doing or interacting with, "
            "and whatever is making the main sound. Do not list background or incidental objects — "
            "if something is not part of the main event, leave it out entirely rather than mentioning it briefly.\n"
            "Describing those three things well is worth more than covering everything in the frame.\n"
            "Do not use the words 'audio', 'visual' or 'stream'."
        ),
        "final": (
            "\n\n[Your observations]:\n{notes}\n\n"
            "Describe what you see and hear in a single sentence, naming at most three things.\n"
            "Answer: "
        ),
        "max_tok": 40,
    },

    # ══════════════════════════════════════════════════════════════════════════
    #
    #
    #
    #
    #
    "v8_reuse": {
        "system": (
            "You are an AI assistant. Describe the video and audio content.\n"
            "Output EXACTLY ONE natural sentence of about 18 words, covering what is seen and what is heard.\n"
            "Do not invent anything that is not in your observations; if sight and sound conflict, report both as they are.\n"
            "Your observations below are your only source. Reuse THEIR OWN WORDS for the things, "
            "actions and sounds they name — do not swap in a synonym or a more general term.\n"
            "Read the observations all the way to the END. The later lines carry the sound evidence, "
            "and it is the part most often left out.\n"
            "Your sentence MUST name the sound, using the same word the observations use for it."
        ),
        "final": (
            "\n\n[Your observations]:\n{notes}\n\n"
            "Based on your observations above, write one sentence describing what you see and hear. "
            "Use the observations' own words, and be sure to include the sound described at the end.\n"
            "Answer: "
        ),
        "max_tok": 40,
    },

    "v9_reuse_long": {
        "system": (
            "You are an AI assistant. Describe the video and audio content.\n"
            "Output EXACTLY ONE natural sentence covering what is seen and what is heard.\n"
            "Do not invent anything that is not in your observations; if sight and sound conflict, report both as they are.\n"
            "Your observations below are your only source. Reuse THEIR OWN WORDS for the things, "
            "actions and sounds they name — do not swap in a synonym or a more general term.\n"
            "Read the observations all the way to the END. The later lines carry the sound evidence, "
            "and it is the part most often left out.\n"
            "Your sentence MUST name the sound, using the same word the observations use for it."
        ),
        "final": (
            "\n\n[Your observations]:\n{notes}\n\n"
            "Based on your observations above, write one sentence describing what you see and hear. "
            "Use the observations' own words, and be sure to include the sound described at the end.\n"
            "Answer: "
        ),
        "max_tok": 60,
    },

    #
    #     v0 1.4% | v5_integrate 4.3% | v6_lean 23.3% | v7_entity 34.3% | v4_caption 36.7%
    #     v0 17.42 (K=19.88 h=2.919) | v4 18.60 (13.20/2.595) | v5 20.27 (14.20/2.851)
    #     v6 18.62 (13.88/2.608)     | v7 17.80 (13.05/2.446)
    #
    #
    #
    "v10_int_reuse": {
        "system": (
            "You are an audiovisual describer. You write one short, natural caption for a video clip, "
            "covering both what is seen and what is heard.\n"
            "CRITICAL ANTI-HALLUCINATION DIRECTIVES:\n"
            "1. MODALITY INDEPENDENCE: Analyze what you SEE and HEAR independently. DO NOT let the visual scene change your interpretation of the audio, and vice versa.\n"
            "2. NO SEMANTIC SMOOTHING: If visuals and audio seem contradictory, report BOTH exactly as they are. Do not invent details.\n"
            "3. AVOID ASSUMPTIONS: Rely on audio to confirm ambiguous visuals.\n\n"
            "STYLE INSTRUCTIONS:\n"
            "1. Length: ONE sentence of about 18 words. Not a list, not two sentences.\n"
            "2. Caption register: name the main subject, its action, and the main sound. Nothing else.\n"
            "3. Leave out secondary detail: no colours, positions, brands, on-screen text, or barely visible objects, unless one of them IS the main subject.\n"
            "4. No hedging, and no commentary about the clip, the streams, or your own observations.\n"
            "5. INTEGRATE, DO NOT LABEL: weave the sights and the sounds into one flowing sentence. "
            "Never write the words 'audio', 'visual', 'video', 'stream', 'observation', or 'clip' in your answer.\n"
            "6. REUSE THE WORDING: your observations are your only source. Keep THEIR OWN WORDS for the "
            "things, actions and sounds they name — do not swap in a synonym or a more general term.\n"
            "7. READ TO THE END: the later observations carry the sound evidence, and that is the part most "
            "often left out. Your sentence MUST name the sound, using the same word the observations use for it.\n"
            "CORE RULE: one plain English sentence, about 18 words, plainly worded. No formatting wrappers."
        ),
        "final": (
            "\n\n[Your observations, listed by modality]:\n{notes}\n\n"
            "Merge these into ONE natural caption sentence of about 18 words, keeping the observations' own "
            "wording and including the sound described at the end. "
            "Do not label the modalities and do not use the words 'audio', 'visual' or 'stream'.\n"
            "Answer: "
        ),
        "max_tok": 64,
    },

    "v11_int_long": {
        "system": None,
        "final": None,
        "max_tok": 64,
    },

}

AVH_PROMPTS["v11_int_long"]["system"] = AVH_PROMPTS["v5_integrate"]["system"]
AVH_PROMPTS["v11_int_long"]["final"] = AVH_PROMPTS["v5_integrate"]["final"]


#
#
#
#
#
#
_V12_LEX = (
    "WORDING RULES (these decide whether your sentence matches how such clips are normally described):\n"
    "1. Call people what they are: 'a man', 'a woman', 'two men', 'a boy'. "
    "Never write 'a person', 'an individual', 'a figure' or 'someone' when you can tell which it is.\n"
    "2. If there is any person in your observations, your sentence MUST mention them. "
    "Do not describe only the object they are holding or the sound they are making.\n"
    "3. Never write 'is visible', 'is audible', 'can be heard', 'is seen', 'is shown', "
    "'accompanied by', 'in the background' or 'serves as the backdrop'. "
    "Say what makes the sound instead, e.g. 'the sound of an engine' or 'a dog barks'.\n"
    "4. Keep colours, sizes and materials ('white', 'black', 'red', 'small', 'wooden'). "
    "They belong in the description; do not strip them out.\n"
    "5. Do not quote speech, do not transcribe words, and do not read out text on screen. "
    "If someone is talking, just write that they speak or talk.\n"
    "6. Use the ordinary word for a thing, not the expert one, unless your observations "
    "clearly identify it: 'an instrument' rather than 'a didgeridoo', 'an engine' rather "
    "than 'a portable electric pump'.\n"
)

AVH_PROMPTS["v12_lexicon"] = {
    "system": (
        "You are an audiovisual describer. You write one short, natural caption for a video clip, "
        "covering both what is seen and what is heard.\n"
        "CRITICAL ANTI-HALLUCINATION DIRECTIVES:\n"
        "1. MODALITY INDEPENDENCE: Analyze what you SEE and HEAR independently. DO NOT let the visual scene change your interpretation of the audio, and vice versa.\n"
        "2. NO SEMANTIC SMOOTHING: If visuals and audio seem contradictory, report BOTH exactly as they are. Do not invent details.\n"
        "3. AVOID ASSUMPTIONS: Rely on audio to confirm ambiguous visuals.\n\n"
        "STYLE INSTRUCTIONS:\n"
        "1. Length: ONE sentence of about 18 words. Not a list, not two sentences.\n"
        "2. Caption register: name the main subject, its action, and the main sound.\n"
        "3. Leave out on-screen text, brands, camera angles, and barely visible objects.\n"
        "4. No hedging, and no commentary about the clip, the streams, or your own observations.\n"
        "5. INTEGRATE, DO NOT LABEL: weave the sights and the sounds into one flowing sentence. "
        "Never write the words 'audio', 'visual', 'video', 'stream', 'observation', or 'clip' in your answer.\n"
        "6. REUSE THE WORDING: your observations are your only source. Keep THEIR OWN WORDS for the "
        "things, actions and sounds they name — do not swap in a synonym or a more general term.\n"
        "7. READ TO THE END: the later observations carry the sound evidence, and that is the part most "
        "often left out. Your sentence MUST name the sound.\n\n"
        + _V12_LEX +
        "CORE RULE: one plain English sentence, about 18 words, plainly worded. No formatting wrappers."
    ),
    "final": (
        "\n\n[Your observations, listed by modality]:\n{notes}\n\n"
        "Merge these into ONE natural caption sentence of about 18 words. "
        "Name the people as 'a man' or 'a woman', keep any colours, include the sound, "
        "and do not write 'visible', 'audible', 'heard' or 'in the background'. "
        "Do not label the modalities and do not use the words 'audio', 'visual' or 'stream'.\n"
        "Answer: "
    ),
    "max_tok": 64,
}


_V13_LEX_LEAN_EXTRA = (
    "Word your sentence the way such clips are normally described:\n"
    "- Call people what they are: 'a man', 'a woman', 'two men', 'a boy'. "
    "Never 'a person', 'an individual', 'a figure' or 'someone' when you can tell which it is.\n"
    "- If there is a person in your observations, your sentence MUST mention them, "
    "not only the object they hold or the sound they make.\n"
    "- Never write 'is visible', 'is audible', 'can be heard', 'is seen', 'is shown', "
    "'accompanied by' or 'in the background'. Name what makes the sound instead.\n"
    "- Keep colours, sizes and materials ('white', 'red', 'small', 'wooden').\n"
    "- Do not quote or transcribe speech or on-screen text; if someone talks, just say they talk.\n"
)

AVH_PROMPTS["v13_lex_lean"] = {
    "system": AVH_PROMPTS["v8_reuse"]["system"] + "\n" + _V13_LEX_LEAN_EXTRA,
    "final": (
        "\n\n[Your observations]:\n{notes}\n\n"
        "Based on your observations above, write one sentence describing what you see and hear. "
        "Use the observations' own words, name people as 'a man' or 'a woman', keep any colours, "
        "include the sound described at the end, and do not write 'visible', 'audible', "
        "'heard' or 'in the background'.\n"
        "Answer: "
    ),
    "max_tok": 64,
}

AVH_PROMPTS["v14_colour"] = {
    "system": AVH_PROMPTS["v10_int_reuse"]["system"].replace(
        "3. Leave out secondary detail: no colours, positions, brands, on-screen text, "
        "or barely visible objects, unless one of them IS the main subject.",
        "3. Leave out brands, on-screen text, camera angles, and barely visible objects. "
        "Colours, sizes and materials belong in the description — keep them.",
    ),
    "final": AVH_PROMPTS["v10_int_reuse"]["final"],
    "max_tok": 64,
}

AVH_PROMPTS["v15_person"] = {
    "system": AVH_PROMPTS["v10_int_reuse"]["system"].replace(
        "CORE RULE:",
        "8. NAME THE PEOPLE: call people what they are — 'a man', 'a woman', 'two men', "
        "'a boy' — never 'a person', 'an individual', 'a figure' or 'someone' when you can "
        "tell which it is. If there is a person in your observations, your sentence MUST "
        "mention them, not only the object they hold or the sound they make.\n"
        "CORE RULE:",
    ),
    "final": AVH_PROMPTS["v10_int_reuse"]["final"].replace(
        "Merge these into ONE natural caption sentence of about 18 words,",
        "Merge these into ONE natural caption sentence of about 18 words, "
        "naming any people as 'a man' or 'a woman',",
    ),
    "max_tok": 64,
}


#   r_v10_int_reuse 17.50  16.03  2.519  0.157  −2.08  0.0000     2
#   r_v14_colour    17.55  15.81  2.512  0.159  −2.02  0.5529*    2   *vs v10
#   r_v15_person    18.53  14.98  2.579  0.172  −1.05  0.0001*    2
#   r_v13_lex_lean  18.46  14.52  2.524  0.174  −1.11  0.0002*    3   *vs v8
#   r_v12_lexicon   19.00  15.47  2.692  0.174  −0.58  0.4697     0
#
_V12_SYS = AVH_PROMPTS["v12_lexicon"]["system"]
_V12_FIN = AVH_PROMPTS["v12_lexicon"]["final"]

AVH_PROMPTS["v16_tight"] = {
    "system": (_V12_SYS
               .replace("1. Length: ONE sentence of about 18 words. Not a list, not two sentences.",
                        "1. Length: ONE sentence of AT MOST 14 words. Not a list, not two sentences. "
                        "Every word must carry information; drop any word you could delete "
                        "without losing a thing, an action, or a sound.")
               .replace("CORE RULE: one plain English sentence, about 18 words, plainly worded.",
                        "CORE RULE: one plain English sentence, at most 14 words, plainly worded.")),
    "final": _V12_FIN.replace(
        "Merge these into ONE natural caption sentence of about 18 words.",
        "Merge these into ONE natural caption sentence of AT MOST 14 words."),
    "max_tok": 64,
}

AVH_PROMPTS["v17_noframeban"] = {
    "system": (_V12_SYS
               .replace("3. Never write 'is visible', 'is audible', 'can be heard', 'is seen', "
                        "'is shown', 'accompanied by', 'in the background' or 'serves as the "
                        "backdrop'. Say what makes the sound instead, e.g. 'the sound of an "
                        "engine' or 'a dog barks'.\n", "")
               .replace("4. Keep colours", "3. Keep colours")
               .replace("5. Do not quote speech", "4. Do not quote speech")
               .replace("6. Use the ordinary word", "5. Use the ordinary word")),
    "final": _V12_FIN.replace(
        " and do not write 'visible', 'audible', 'heard' or 'in the background'.", "."),
    "max_tok": 64,
}

AVH_PROMPTS["v18_soundnoun"] = {
    "system": _V12_SYS.replace(
        "CORE RULE:",
        "7. NAME THE SOUND SOURCE: write the specific thing that makes each sound — "
        "'music', 'an engine', 'running water', 'a voice', 'laughter', 'barking' — "
        "not just the bare word 'sound'. If your observations name more than one sound, "
        "put ALL of them in your sentence.\n"
        "CORE RULE:"),
    "final": _V12_FIN.replace(
        "include the sound,",
        "name every sound your observations mention and say what makes it,"),
    "max_tok": 64,
}

AVH_PROMPTS["v19_reuse8"] = {
    "system": _V12_SYS.replace(
        "CORE RULE:",
        "7. COVER YOUR OBSERVATIONS: your sentence must carry over at least EIGHT of the "
        "exact nouns and verbs that appear in your observations, spelled the same way. "
        "Prefer a word your observations actually used over any word of your own.\n"
        "CORE RULE:"),
    "final": _V12_FIN.replace(
        "Name the people as 'a man' or 'a woman',",
        "Carry over at least eight of the observations' own nouns and verbs unchanged, "
        "name the people as 'a man' or 'a woman',"),
    "max_tok": 64,
}


#
#
#
#
AVH_PROMPTS["v20_percept_person"] = {
    "system": (_V12_SYS
               .replace("WORDING RULES (these decide whether your sentence matches how such "
                        "clips are normally described):",
                        "WORDING RULES — these apply to EVERYTHING you write, including your "
                        "perceptual analysis notes, not only to the final sentence:")
               .replace("CORE RULE:",
                        "7. START AS YOU MEAN TO FINISH: obey these wording rules from the very "
                        "first note you write. When you record a person in your perceptual "
                        "analysis, write 'a man', 'a woman', 'two men' or 'a boy' there as well "
                        "— never 'an individual', 'a person', 'a figure' or 'the subject'. Your "
                        "final sentence can only reuse words your own notes already contain, so "
                        "a vague word written now cannot be repaired later.\n"
                        "CORE RULE:")),
    "final": _V12_FIN,
    "max_tok": 64,
}


#
#
#
#   crowing→sing、stone-walled corridor→cave、plain white surface→table、
#   lush garden→grass、mechanical hum→buzzing、electronic track→music、
#   news show→studio、toilet/faucet→bathroom、engine compartment→hood。
#
#
#
#
#

_V21_NOTE_RULE = (
    "HOW TO WRITE YOUR PERCEPTUAL ANALYSIS (this governs the notes you write before "
    "your final sentence):\n"
    "A. Your notes are private working material. Nobody scores them and their length "
    "does not matter, so repeating yourself there costs you nothing.\n"
    "B. Every time you name a thing, a place, a material, a person, an action or a "
    "sound in your notes, put the plainest everyday word for that same thing right "
    "after it in square brackets — the word an ordinary person with no special "
    "knowledge would use. Write 'a beverage [a drink]', 'footwear [shoes]', "
    "'a physician [a doctor]'. If your first word is already the plain one, just "
    "repeat it in the brackets.\n"
    "C. Do this for the sounds too: next to the precise description of a noise, give "
    "the plain word for what makes it and the plain word for the noise itself.\n"
    "D. In your FINAL sentence choose exactly ONE of the two words and drop the "
    "brackets. The final sentence must contain no brackets, no slashes and no "
    "alternatives.\n"
)

AVH_PROMPTS["v21_alias"] = {
    "system": _V12_SYS.replace(
        "CORE RULE:", _V21_NOTE_RULE + "CORE RULE:"),
    "final": _V12_FIN.replace(
        "Answer: ",
        "If your observations give two words for the same thing, pick the plainer one "
        "and use it once; write no brackets.\nAnswer: "),
    "max_tok": 64,
}

_V22_NOTE_RULE = (
    "HOW TO WRITE YOUR PERCEPTUAL ANALYSIS (this governs the notes you write before "
    "your final sentence):\n"
    "A. Write your notes in the plainest everyday English. Use the word an ordinary "
    "person with no special knowledge would use: 'a drink', not 'a beverage'; "
    "'shoes', not 'footwear'; 'a doctor', not 'a physician'.\n"
    "B. This applies to every thing, place, material, person, action and sound you "
    "record. Never reach for the technical, literary or unusual word when a common "
    "one names the same thing.\n"
    "C. Your final sentence can only reuse words your own notes already contain, so "
    "an unusual word written now cannot be repaired later.\n"
)

AVH_PROMPTS["v22_plainnote"] = {
    "system": _V12_SYS.replace(
        "CORE RULE:", _V22_NOTE_RULE + "CORE RULE:"),
    "final": _V12_FIN,
    "max_tok": 64,
}


def get_avh_prompt(version: str):
    """Return (system_prompt, final_template, max_new_tokens) for a given version."""
    cfg = AVH_PROMPTS.get(version, AVH_PROMPTS["v0"])
    return cfg["system"], cfg["final"], cfg["max_tok"]



_STEP_DEFAULT = (
    "Based STRICTLY on the raw visual and auditory evidence provided (do not guess or "
    "assume), analyze this Question: {question}\n"
    "Provide the NEXT reasoning step or description."
)

_STEP_ALIAS = (
    "Based STRICTLY on the raw visual and auditory evidence provided (do not guess or "
    "assume), analyze this Question: {question}\n"
    "Provide the NEXT reasoning step or description.\n"
    "WHILE WRITING THIS NOTE, FOLLOW THIS ONE RULE:\n"
    "Every time you name a thing, a place, a material, a person, an action or a sound, "
    "put the plainest everyday word for that same thing right after it in square "
    "brackets — the word an ordinary person with no special knowledge would use. "
    "Write 'a beverage [a drink]', 'footwear [shoes]', 'a physician [a doctor]', "
    "'a low hum [a buzzing sound]'. If your first word is already the plain one, "
    "just repeat it in the brackets.\n"
    "This note is private working material. Nobody scores it and its length does not "
    "matter, so the brackets cost you nothing."
)

_STEP_PLAIN = (
    "Based STRICTLY on the raw visual and auditory evidence provided (do not guess or "
    "assume), analyze this Question: {question}\n"
    "Provide the NEXT reasoning step or description.\n"
    "WHILE WRITING THIS NOTE, FOLLOW THIS ONE RULE:\n"
    "Write in the plainest everyday English. Use the word an ordinary person with no "
    "special knowledge would use: 'a drink', not 'a beverage'; 'shoes', not 'footwear'; "
    "'a doctor', not 'a physician'. Never reach for the technical, literary or unusual "
    "word when a common one names the same thing. Your final sentence can only reuse "
    "words this note already contains, so an unusual word written now cannot be "
    "repaired later."
)

_FIN_DROP_BRACKETS = _V12_FIN.replace(
    "Answer: ",
    "Your observations put a plain everyday word in square brackets after some terms. "
    "For each of those, pick exactly ONE of the two words for your sentence — whichever "
    "is the more ordinary — and write no brackets, no slashes and no alternatives.\n"
    "Answer: ")

AVH_PROMPTS["v21b_alias"] = {
    "system": _V12_SYS,
    "final": _FIN_DROP_BRACKETS,
    "step": _STEP_ALIAS,
    "max_tok": 64,
}

AVH_PROMPTS["v22b_plainnote"] = {
    "system": _V12_SYS,
    "final": _V12_FIN,
    "step": _STEP_PLAIN,
    "max_tok": 64,
}


def get_avh_step(version: str, question: str) -> str:
    """User instruction for the note-taking stage (the first sampling pass).

        Arms without a `step` field get the default text, which is byte-identical
        to the original hardcoded string in the trunk. Calling this function with
        an old version id therefore does not change its behaviour -- that is the
        precondition for historical readings staying comparable.
    """
    cfg = AVH_PROMPTS.get(version, AVH_PROMPTS["v0"])
    return cfg.get("step", _STEP_DEFAULT).format(question=question)


#     a microphone [a device]、a podium [a piece of furniture]、
#     her right hand [a movement]、blonde [a color]、a pink blazer [a garment]
#
#
_STEP_ALIAS_C = (
    "Based STRICTLY on the raw visual and auditory evidence provided (do not guess or "
    "assume), analyze this Question: {question}\n"
    "Provide the NEXT reasoning step or description.\n"
    "WHILE WRITING THIS NOTE, FOLLOW THIS ONE RULE:\n"
    "After each thing, person, place, action or sound you name, add in square brackets "
    "ONE other everyday word that names THE VERY SAME THING — the word a different "
    "ordinary person might have used for it.\n"
    "It must be the same thing at the same level of detail. Write "
    "'a turbine [a fan]', 'a rooster [a chicken]', 'a corridor [a hallway]', "
    "'a low hum [a buzzing]', 'a physician [a doctor]', 'a beverage [a drink]'.\n"
    "NEVER put a category word in the brackets. Words like 'a device', 'an object', "
    "'an item', 'a piece of furniture', 'a movement', 'a colour', 'a garment', "
    "'a machine', 'a sound', 'an animal', 'a person' name a whole class, not the same "
    "thing, and they are useless here.\n"
    "If your word is ALREADY the ordinary everyday one — man, woman, dog, car, water, "
    "music, rain, door, tree — then just repeat that same word in the brackets. Do NOT "
    "reach for a fancier, vaguer or more literary alternative: 'a woman [a woman]', "
    "not 'a woman [a lady]'.\n"
    "This note is private working material. Nobody scores it and its length does not "
    "matter, so the brackets cost you nothing."
)

AVH_PROMPTS["v21c_alias"] = {
    "system": _V12_SYS,
    "final": _FIN_DROP_BRACKETS,
    "step": _STEP_ALIAS_C,
    "max_tok": 64,
}


#     A train (a locomotive)、a barrier (a gate)、
#     a railroad crossing (a railway crossing)、a train horn (a toot)
#
#
_STEP_ALIAS_D = (
    "Based STRICTLY on the raw visual and auditory evidence provided (do not guess or "
    "assume), analyze this Question: {question}\n"
    "Provide the NEXT reasoning step or description.\n"
    "THEN, on a new line at the very end, write the label 'ALSO CALLED:' followed by a "
    "list of alternative names, separated by semicolons, in the form "
    "'my word = another everyday word'.\n"
    "For every thing, person, place, action and sound you just named, give ONE other "
    "everyday word that names THE VERY SAME THING at the same level of detail — the "
    "word a different ordinary person might have used for it. For example:\n"
    "ALSO CALLED: physician = doctor; sofa = couch; lorry = truck; "
    "trousers = pants; jug = pitcher; commence = begin\n"
    "Two rules for that list:\n"
    "1. Never give a category word. 'device', 'object', 'item', 'piece of furniture', "
    "'movement', 'colour', 'garment', 'machine', 'animal', 'person' name a whole class, "
    "not the same thing, and are useless here.\n"
    "2. Skip any word that is already the ordinary everyday one. Leave those out of the "
    "list entirely rather than reaching for a fancier or vaguer alternative.\n"
    "Write the description in plain prose with no brackets in it; the alternatives go "
    "only on that last line. The note is private working material — nobody scores it and "
    "its length does not matter."
)

_FIN_ALSO_CALLED = _V12_FIN.replace(
    "Answer: ",
    "Your observations may end with a line labelled 'ALSO CALLED:'. That line is not "
    "part of the scene — it only lists other everyday names for the same things. Use "
    "whichever of the two namings is the more ordinary, everyday one. Never write the "
    "words 'ALSO CALLED', and never put two namings, brackets or slashes in your "
    "sentence.\n"
    "Answer: ")

AVH_PROMPTS["v21d_alias"] = {
    "system": _V12_SYS,
    "final": _FIN_ALSO_CALLED,
    "step": _STEP_ALIAS_D,
    "max_tok": 64,
}
