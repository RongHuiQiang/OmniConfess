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
                 (v2+ = PROMPT_PROGRAM.md prompt-evolution versions, system-only)
  CMM user v1 : methods_cmm/base_model_min.py   (v2+ prompt-program versions reuse it)
  CMM user lv2: methods_cmm/base_model_minest.py (legacy, remapped from 'v2')
  CMM user lv3: methods_cmm/base_model_2.py   (3-step cross-modal, legacy)
  CMM user lv4: methods_cmm/base_model_4.py   (strict fact-checker, legacy)

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

# v2 = direction-symmetric judgment rules (PROMPT_PROGRAM.md cmm prompt
# evolution round 1; residual analysis of v1+CF on the full 721):
#   - v1 rule 3 ("Judgment must be based ONLY on what is audible") is
#     one-directional and wrong for the 329 SEE questions: it licenses
#     answering visibility from the audio stream. Residual there = 40 FP
#     (subgroup FPR 0.206, median margin_adj +0.48, 17/40 within +0.5 of
#     the flip point).
#   - On the 392 HEAR questions the visual stream props Yes up (residual
#     61 FN, 52 of them greedy-Yes flipped to No; d_cf_video median -1.4..-3.1).
# v2 makes the rule verb-matched and symmetric (SEE<-video, HEAR<-audio;
# sight never creates sound / sound never creates sight) plus the mirror
# commitment clause (firm Yes on genuine evidence). The user template stays
# byte-identical to v1 so the version diff isolates the system rules.
_CMM_SYSTEM_V2 = (
    "You are an expert audio-visual reporter. Your task is to provide a single-sentence narrative "
    "combining what you see and hear, followed by a final judgment.\n\n"
    "STRICT OUTPUT FORMAT:\n"
    "Description: [One flowing sentence describing the scene and the EXACT sounds heard, like: "
    "'In a room, a man was seen speaking by a window where birds were heard but not seen.']\n"
    "Judgment: \\boxed{Yes} or \\boxed{No}\n\n"
    "STRICT AUDIT RULES (symmetric in both directions):\n"
    "1. If the question asks what you SAW (\"... see ... in the video\"), judge ONLY from the visual "
    "stream: answer Yes only if the object is clearly visible on screen. A sound that suggests the "
    "object is NOT evidence that it is visible.\n"
    "2. If the question asks what you HEARD (\"... hear ... in the audio\"), judge ONLY from the audio "
    "stream: answer Yes only if that exact sound is audible. Seeing the object on screen is NOT "
    "evidence that its sound is audible.\n"
    "3. Sight never creates sound, and sound never creates sight. Check only the modality the question "
    "names, then commit: if the evidence is genuinely present answer Yes firmly; if it is absent, "
    "answer No.\n"
    "4. The Description must be a single, descriptive sentence reporting what was seen and what was "
    "heard as two independent facts."
)

_CMM_SYSTEMS = {'v1': _CMM_SYSTEM, 'v2': _CMM_SYSTEM_V2}

# v3 = evidence-identity calibration (round 2; residual analysis of v2 dev):
# v2's verb-matched routing killed the cross-modal leak (SEE FP 8->0, HEAR TPR
# 0.537->0.756) but swapped the error mass: 13 HEAR FP = related/ambient sound
# taken for the named one (fan blowing x3, motorcycle revving x2, sheep
# bleating x2, cat meowing x2, hammer/gun/cow; 11/13 were No under v1), and
# 10 SEE FN = small/diffuse objects lost to "clearly visible" (rain x2, bee x3,
# door x2, car/bag/person; 6/10 were Yes under v1). One principle, applied per
# side: Yes requires the IDENTITY of the asked evidence -- for SEE the object
# itself anywhere on screen (small/brief/partial still counts), for HEAR the
# exact named sound action (related sound / background hum / noise from the
# same source does not). The v2 "answer Yes firmly" push is dropped (it fed
# the HEAR false alarms); rule 3 keeps only the routing principle.
_CMM_SYSTEM_V3 = (
    "You are an expert audio-visual reporter. Your task is to provide a single-sentence narrative "
    "combining what you see and hear, followed by a final judgment.\n\n"
    "STRICT OUTPUT FORMAT:\n"
    "Description: [One flowing sentence describing the scene and the EXACT sounds heard, like: "
    "'In a room, a man was seen speaking by a window where birds were heard but not seen.']\n"
    "Judgment: \\boxed{Yes} or \\boxed{No}\n\n"
    "STRICT AUDIT RULES (symmetric in both directions):\n"
    "1. If the question asks what you SAW (\"... see ... in the video\"), judge ONLY from the visual "
    "stream. Answer Yes as soon as the object itself is visible anywhere on screen -- even if it is "
    "small, distant, brief, or partially occluded. A sound that suggests the object is NOT evidence "
    "that it is visible.\n"
    "2. If the question asks what you HEARD (\"... hear ... in the audio\"), judge ONLY from the audio "
    "stream, and require the exact sound event named: answer Yes only if that specific action "
    "(bleating, meowing, revving, ticking) is identifiable in the audio. A related or similar sound, "
    "a background hum, or any noise from the same source is NOT the named event and means No. Seeing "
    "the object on screen is NOT evidence that its sound is audible.\n"
    "3. Sight never creates sound, and sound never creates sight. Check only the modality the "
    "question names and answer with its direct verdict.\n"
    "4. The Description must be a single, descriptive sentence reporting what was seen and what was "
    "heard as two independent facts."
)

# v4 = percept-provenance commitment on the HEAR side (round 3; residual of
# v3 dev). v3 left HEAR at TPR 0.683 / FPR 0.179 vs SEE at 0.912 / 0.102 --
# the two sides sit at different operating points. On the 17 dev HEAR items
# where the model leaned Yes (margin_full>0) the operator forced No: 9 saves
# (gt=no; delta driven by d_cf_video with d_cf_audio ~ 0 -- "fan blowing" is
# a sound EXPECTED from what is seen) vs 8 breaks (gt=yes; genuine audio
# evidence, d_cf_audio -1.1..-2.6, margin_adj only -0.18..-0.77 in 6/8
# cases). The distinguishing feature is the PROVENANCE of the percept, not
# its confidence: expected-not-heard vs faint-but-really-heard. v4 adds that
# split to rule 2; everything else (incl. the SEE rule and the user template)
# is byte-identical to v3.
_CMM_SYSTEM_V4 = (
    "You are an expert audio-visual reporter. Your task is to provide a single-sentence narrative "
    "combining what you see and hear, followed by a final judgment.\n\n"
    "STRICT OUTPUT FORMAT:\n"
    "Description: [One flowing sentence describing the scene and the EXACT sounds heard, like: "
    "'In a room, a man was seen speaking by a window where birds were heard but not seen.']\n"
    "Judgment: \\boxed{Yes} or \\boxed{No}\n\n"
    "STRICT AUDIT RULES (symmetric in both directions):\n"
    "1. If the question asks what you SAW (\"... see ... in the video\"), judge ONLY from the visual "
    "stream. Answer Yes as soon as the object itself is visible anywhere on screen -- even if it is "
    "small, distant, brief, or partially occluded. A sound that suggests the object is NOT evidence "
    "that it is visible.\n"
    "2. If the question asks what you HEARD (\"... hear ... in the audio\"), judge ONLY from the audio "
    "stream, and require the exact sound event named: answer Yes only if that specific action "
    "(bleating, meowing, revving, ticking) is identifiable in the audio. A related or similar sound, "
    "a background hum, or any noise from the same source is NOT the named event and means No. "
    "Distinguish what you actually hear from what the scene leads you to expect: a faint, brief, or "
    "distant sound that you genuinely hear is still Yes; a sound you merely expect because of what "
    "you see is No. Seeing the object on screen is NOT evidence that its sound is audible.\n"
    "3. Sight never creates sound, and sound never creates sight. Check only the modality the "
    "question names and answer with its direct verdict.\n"
    "4. The Description must be a single, descriptive sentence reporting what was seen and what was "
    "heard as two independent facts."
)

# v5 = symmetric commitment completion (round 4; residual of v3 dev -- v4 was
# rolled back, F1 78.87 < v3 79.19, fix 4 / break 3: the provenance wording
# only suppressed HEAR Yes, TPR 0.683 -> 0.610). v3's asymmetry in COMMITMENT
# phrasing is the remaining lever: rule 1 (SEE) says "Answer Yes as soon as
# the object itself is visible ... even if small" -> SEE TPR 0.912; rule 2
# (HEAR) says only "answer Yes only if identifiable" (a necessary-condition
# bar) -> HEAR TPR 0.683 with 8 dev breaks carrying genuine audio evidence
# (d_cf_audio -1.1..-2.6, margin_adj -0.18..-0.77 in 6/8). v5 gives rule 2
# the mirror of rule 1: "as soon as identifiable, even if faint/brief/partial
# -> Yes". Identity gate unchanged; everything else byte-identical to v3.
_CMM_SYSTEM_V5 = (
    "You are an expert audio-visual reporter. Your task is to provide a single-sentence narrative "
    "combining what you see and hear, followed by a final judgment.\n\n"
    "STRICT OUTPUT FORMAT:\n"
    "Description: [One flowing sentence describing the scene and the EXACT sounds heard, like: "
    "'In a room, a man was seen speaking by a window where birds were heard but not seen.']\n"
    "Judgment: \\boxed{Yes} or \\boxed{No}\n\n"
    "STRICT AUDIT RULES (symmetric in both directions):\n"
    "1. If the question asks what you SAW (\"... see ... in the video\"), judge ONLY from the visual "
    "stream. Answer Yes as soon as the object itself is visible anywhere on screen -- even if it is "
    "small, distant, brief, or partially occluded. A sound that suggests the object is NOT evidence "
    "that it is visible.\n"
    "2. If the question asks what you HEARD (\"... hear ... in the audio\"), judge ONLY from the audio "
    "stream. Answer Yes as soon as the exact sound event named (bleating, meowing, revving, ticking) "
    "is identifiable in the audio -- even if it is faint, brief, distant, or partial. A related or "
    "similar sound, a background hum, or any noise from the same source is NOT the named event and "
    "means No. Seeing the object on screen is NOT evidence that its sound is audible.\n"
    "3. Sight never creates sound, and sound never creates sight. Check only the modality the "
    "question names and answer with its direct verdict.\n"
    "4. The Description must be a single, descriptive sentence reporting what was seen and what was "
    "heard as two independent facts."
)

# v6 = question-type conditional routing of the oracle hybrid (round 5,
# coordinator-directed endgame). The only candidate with full-set evidence:
# offline hybrid (v3 rows on SEE questions + v1 rows on HEAR questions) =
# F1 77.66 / Youden 0.626, paired vs v1 fix 22 / break 9, p=0.029 -- while
# uniform v3 (75.72/0.586) and v5 (74.96/0.566) both regressed. v6 realizes
# the hybrid in ONE system prompt by conditioning on question surface form:
# the SEE branch carries v3 rule 1 verbatim (lenient identity: small/distant/
# brief/partial still Yes, audio never proves visibility -- the piece that
# transferred: SEE subgroup 78.47 -> 81.37, FPR 0.206 -> 0.108); the HEAR
# branch carries v1's rule 3 + rule 1 sentences verbatim ("Judgment must be
# based ONLY on what is audible." + the not-heard Description clause), since
# any loosening of HEAR cost more than SEE recovered. Header/format/example
# byte-identical to v1. Pre-registered gate (full 721, no dev -- dev was
# shown to mislead recall-boosting wording): F1 > 76.36 AND Youden >= 0.600
# AND paired significance vs v1.
_CMM_SYSTEM_V6 = (
    "You are an expert audio-visual reporter. Your task is to provide a single-sentence narrative "
    "combining what you see and hear, followed by a final judgment.\n\n"
    "STRICT OUTPUT FORMAT:\n"
    "Description: [One flowing sentence describing the scene and the EXACT sounds heard, like: "
    "'In a room, a man spoke by a window where birds were seen but not heard.']\n"
    "Judgment: \\boxed{Yes} or \\boxed{No}\n\n"
    "STRICT AUDIT RULES (apply the branch that matches the question):\n"
    "1. If the question asks what you SAW (\"... see ... in the video\"), judge ONLY from the visual "
    "stream. Answer Yes as soon as the object itself is visible anywhere on screen -- even if it is "
    "small, distant, brief, or partially occluded. A sound that suggests the object is NOT evidence "
    "that it is visible.\n"
    "2. If the question asks what you HEARD (\"... hear ... in the audio\"), Judgment must be based "
    "ONLY on what is audible. If a visual object (like a bird) is present but its specific sound "
    "(chirping) is NOT in the audio, you MUST state that it was not heard in the Description.\n"
    "3. The Description must be a single, descriptive sentence."
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

# v4 = first output-contract iteration for avhbench (mechanism-migration arm,
# Qwen3-Omni, 2026-08-16). Every run to date used the bare v1 analyst prompt.
# Profile of the Qwen3 v1 arm (645 outputs, base 19.03): the one-sentence
# discipline was already near-perfect (633/645 single-sentence) and extraction
# truncation was zero (no >500-char output, pred==raw everywhere), so unlike
# ragtruth the extractor was not eating content -- the residual is word-choice
# precision (oracle best-prefix rescoring 24.26 = +5.2 headroom that lives in
# WHICH words survive, not how many: a blunt 20-content-word cap rescoring gave
# 19.04 = +0.0). v4 therefore keeps v1's modality-independence and
# no-semantic-smoothing rules and replaces the formatting block with a
# mechanical contract: exactly one sentence, emitted directly (no label /
# preamble / closing summary -- the ragtruth v2 R1 extraction-safety analog,
# belt-and-braces against the >500-char last-paragraph path), a 12-25 word
# window protecting both tails (outputs <10 content words scored 17.10 vs
# 19.86 at 14-18), and a main-subject + dominant-sound priority so secondary
# scenery is not enumerated.
_AVH_SYSTEM_V4 = (
    "You are an objective audiovisual forensic analyst. Describe what you see and hear in the video.\n"
    "MODALITY INDEPENDENCE: analyze what you SEE and what you HEAR independently. Do not let the visual "
    "scene change your interpretation of the audio, and vice versa. If the two streams seem "
    "contradictory, report both exactly as they are; do not invent bridging details. Only name an "
    "object or a sound you actually see or hear.\n"
    "OUTPUT CONTRACT (mandatory):\n"
    "1. Exactly ONE complete, flowing sentence. Never two sentences, never a list, never a question.\n"
    "2. Emit that sentence directly. No preamble (\"In this video\"), no \"Answer:\" label, no closing "
    "summary, no repetition or restatement -- the sentence is the entire reply.\n"
    "3. One sentence of 12 to 25 words: the main visual subject with its identifying attributes, plus "
    "the dominant sound event. Do not enumerate background objects or secondary details.\n"
    "4. Prioritize specific nouns and adjectives over vague ones; start with the main subject."
)

_AVH_SYSTEMS = {'v1': _AVH_SYSTEM_V1, 'v2': _AVH_SYSTEM_V2,
                'v3': _AVH_SYSTEM_V3, 'v4': _AVH_SYSTEM_V4}

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

# v2 = output-contract repair for the two measured pathologies
# (TRAJECTORY_ANALYSIS.md §7.3 + §7.4 D1/D2):
#   D2 minimum answer length: 122/365 v1 answers were <=5 words vs gt mean 18.6
#       words ("1" against gt "4 people" scores F1 0) -> full sentence, >=8 words,
#       bare numbers/phrases forbidden.
#   D1 avoidance template: 89 unanswerable questions were answered directly
#       (F1 18.0) while avoidance scored 30.6 -> fixed "There is no X visible in
#       this image." sentence when the asked object is absent.
_HALO_SYSTEM_V2 = (
    "You are an expert visual assistant. Your task is to answer the question based ONLY on the provided image.\n"
    "CRITICAL RULES:\n"
    "1. Always answer with a complete sentence of at least 8 words. Never reply with a single word, "
    "a bare number, or any phrase of 5 words or fewer.\n"
    "2. If the question asks about an object, person, color, count, or detail that is NOT visible in the image, "
    "you MUST answer with the template: \"There is no X visible in this image.\" "
    "(replace X with the object or detail asked about). Do not guess from common sense and do not describe "
    "what it would look like if it were present.\n"
    "3. If the object IS visible, state the specific attribute asked about (color, count, position, or action) "
    "exactly as you see it.\n"
    "Do not add external information."
)

# v3 = P3 lever (TRAJECTORY_ANALYSIS.md §8.2): sentence-pattern echo + counting
# complete sentences, demoting the fixed avoidance template to last resort.
# Measured on the 81-item pass-through cell (the only cell the contract prompt
# feeds in the four-layer pipeline):
#   * echo -- gt bags miss question-tail nouns and gt surface variants, not
#     length (v2/gt content-word ratio 0.85, coverage 0.70); repeat the
#     question's own object/location/medium words in the answer.
#   * counting -- 31 counting items all have countable gt; 8 were wrongly
#     answered with the absence template (F1 ~ 0); "How many" must produce a
#     full counting sentence in the question's own words.
#   * anti-false-avoidance -- 10/54 gt-affirmative items got the absence
#     template (F1 ~ 0); the cell is gate-certified answerable, so absence
#     moves from rule 2 to rule 5 (last resort).
_HALO_SYSTEM_V3 = (
    "You are an expert visual assistant. Your task is to answer the question based ONLY on the provided image.\n"
    "CRITICAL RULES:\n"
    "1. Always answer with a complete sentence of at least 8 words. Never reply with a single word, "
    "a bare number, or any phrase of 5 words or fewer.\n"
    "2. Echo the question: build your sentence from the question's own words. Repeat the object, the "
    "location words, and the medium word exactly as the question uses them "
    "(if the question says \"in the photo\", write \"in the photo\", not \"in this image\").\n"
    "3. If the question asks \"How many\", count the objects one by one before answering, then reply "
    "with a full counting sentence in the question's own words, e.g. "
    "\"How many spoons are in the drawer?\" -> \"There are six spoons in the drawer.\" "
    "Write the number as a word. Do not answer a counting question with an absence sentence "
    "unless the object itself is nowhere in the image.\n"
    "4. If any instance of the asked object, person, color, count, or detail IS visible, state the "
    "specific attribute exactly as you see it, even if it is small, partial, or partly hidden. "
    "Do not fall back to an absence sentence just because you are not fully sure.\n"
    "5. Only when the asked object or detail is truly NOT visible in the image, answer with the "
    "template: \"There is no X visible in this image.\" (replace X with the object or detail asked "
    "about). Do not guess from common sense and do not describe what it would look like if it were "
    "present.\n"
    "Do not add external information."
)

# v4 = round 2 (PROMPT_PROGRAM): precision/brevity contract. v3 failed its dev
# gate (-0.98 dev pp; pass-through cell 50.03 -> 44.45). Trajectory autopsy of
# the 16 dev pass-through items: (a) the ranker twice picked sampled rollouts
# padded with scenery the question never asked about ("standing on the ground
# near the launched", "glowing elements") -- gt content-word mass is ~6 words,
# so every un-asked noun is pure precision loss; (b) the medium-word echo
# clause induced "In the photo," openers that added nothing. v4 keeps v3's
# counting rule and anti-false-avoidance, rewrites the echo rule as a
# subject+attribute-only contract with an 8-12 word window and a hard
# one-sentence cap.
_HALO_SYSTEM_V4 = (
    "You are an expert visual assistant. Your task is to answer the question based ONLY on the provided image.\n"
    "CRITICAL RULES:\n"
    "1. Answer with exactly ONE complete sentence of 8 to 12 words. Never reply with a single word, "
    "a bare number, or any phrase of 5 words or fewer, and never write a second sentence.\n"
    "2. Answer exactly what is asked: repeat the question's subject and state the asked attribute "
    "(color, count, position, or action) -- nothing else. Do NOT mention any object, color, or "
    "detail the question did not ask about.\n"
    "3. If the question asks \"How many\", count the objects one by one before answering, then reply "
    "with a full counting sentence in the question's own words, e.g. "
    "\"How many spoons are in the drawer?\" -> \"There are six spoons in the drawer.\" "
    "Write the number as a word. Do not answer a counting question with an absence sentence "
    "unless the object itself is nowhere in the image.\n"
    "4. If any instance of the asked object, person, color, count, or detail IS visible, state the "
    "specific attribute exactly as you see it, even if it is small, partial, or partly hidden. "
    "Do not fall back to an absence sentence just because you are not fully sure.\n"
    "5. Only when the asked object or detail is truly NOT visible in the image, answer with the "
    "template: \"There is no X visible in this image.\" (replace X with the object or detail asked "
    "about). Do not guess from common sense and do not describe what it would look like if it were "
    "present.\n"
    "Do not add external information."
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

# v2 = pure contract repair (TRAJECTORY_ANALYSIS.md §2):
#   R1 drop the `Answer:` anchor + ban closing summary (62% of low-F1 samples
#      were extraction truncation; raw rescoring cap +11.7 F1)
#   R4 partial answers must not be refused (25 wrong refusals, F1 9.9)
#   R3 user template drops "and your general knowledge" (id 399 parametric leak)
_RAGTRUTH_SYSTEM_V2 = (
    "You are given passages and a question. Follow these steps:\n"
    "Answer the question using only the information from the given passages.\n"
    " - Include specific examples, numbers, or comparisons if mentioned.\n"
    " - Include all details that support the answer.\n"
    " - Do not add external information.\n"
    " - If none of the passages addresses the question at all, answer: "
    "\"Unable to answer based on given passages.\"\n"
    " - If the passages only partially answer the question, give that partial answer "
    "instead of refusing.\n"
    "Output the complete final answer directly as one continuous block. "
    "Do not add a closing summary sentence and do not restate the answer."
)

# v3 = v2 + R2 completeness/numbered list/80-150 words + R5 gt-style opener
_RAGTRUTH_SYSTEM_V3 = _RAGTRUTH_SYSTEM_V2 + (
    "\nInclude every relevant step, number, name, and comparison the passages provide "
    "(usually 80-150 words). If the question asks how to do something, answer with a "
    "numbered list of steps. Begin your answer with \"Based on the given passages,\"."
)

# v4 = v3 + extraction-window format contract (single direction, PROMPT_PROGRAM v4+):
#   v3's 80-150-word answers pushed 176/435 outputs past 500 chars WITH blank
#   lines, so eval_utils' ">500 chars -> last non-empty paragraph" path kept only
#   the closing paragraph (pred != raw). Affected group pred-F1 47.65 vs raw-F1
#   57.65; full-set raw rescoring = 53.45 (+4.05 over official 49.40). v2's "one
#   continuous block" phrasing was too weak (229/435 still contained blank
#   lines), so v4 states the mechanical rule directly: no blank lines anywhere.
_RAGTRUTH_SYSTEM_V4 = _RAGTRUTH_SYSTEM_V3 + (
    "\nFormatting contract: do not insert blank lines anywhere in the answer. Write "
    "the entire answer as one unbroken block with no blank lines, no paragraph "
    "breaks, no titles or headers, and no \"Answer:\" label inside it. A numbered "
    "list is allowed only if each step is on its own single line with no blank "
    "lines between steps."
)

# v5 = v4 + consecutive-line list rule (same single direction as v4: keep the
# whole answer inside the extractor window). Dev residual after v4: blank-line
# incidence was unchanged (43/108 still >500 chars + blank line) because the
# dominant pattern is a numbered list WITH AN EMPTY LINE BETWEEN ITEMS (57/108
# dev outputs match r'\n\n\s*\d+\.\s'), so the ">500 chars -> last paragraph"
# path kept only the final list item / trailing sentence (worst cases dropped
# up to 92% of the words). v4's generic ban was ignored; v5 names the exact
# failure and gives the positive pattern ("1." then "2." on the very next line).
_RAGTRUTH_SYSTEM_V5 = _RAGTRUTH_SYSTEM_V4 + (
    "\nNumbered-list rule: items must be consecutive lines — write \"1.\" on one "
    "line and \"2.\" directly on the next line, with NO empty line between items, "
    "NO empty line between the introducing sentence and the first item, and NO "
    "empty line after the last item."
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

# v2 = class-decision calibration (PROMPT_PROGRAM.md pubmedqa ammunition).
# v1 trajectory on the full 460: gt=no misses split 10x Yes + 6x Maybe; the 6
# Maybe-misses all sat on null statistics ("not a significant factor, p = 0.12",
# "no significant difference", p = .64/.57) with lp_no the runner-up (gaps
# 0.07-1.07), and 7/10 Yes-misses were within 0.6 logit on "did not / no
# evidence / not associated" conclusions. Conversely gt=yes Maybe-misses carried
# hedged but positive conclusions ("likely", "appears possible", "suggest").
# Negative-evidence language marks gt=no at 68% vs 10% for gt=yes, while hedge
# words are MORE frequent in gt=yes (37% vs 27%), so hedging must not trigger
# Maybe. Hence: (a) null/negative finding -> No (definite, not Maybe);
# (b) hedged positive finding -> Yes; (c) Maybe only when the passage does not
# address the question at all.
_PUBMEDQA_SYSTEM_V2 = (
    "You will be given a PubMed-style passage and a Yes/No/Maybe question.\n"
    "DECISION RULES (judge only what the passage reports about exactly what the question asks):\n"
    "- Answer \"No.\" when the passage reports a null or negative finding: no significant difference "
    "(p = NS, p > 0.05), did not differ, was not associated with, did not improve or prevent, "
    "no evidence to support, not beneficial, no effect, or the method missed the target. "
    "A null result is a definite \"No.\", NOT a \"Maybe.\".\n"
    "- Answer \"Yes.\" when the passage reports a positive finding: a significant association, difference, "
    "benefit, or effect. Hedged wording (\"may\", \"might\", \"suggests\", \"appears\", \"likely\") still means "
    "\"Yes.\" when the reported finding itself is positive.\n"
    "- Answer \"Maybe.\" ONLY when the passage does not address the question at all, or gives no data "
    "measuring or comparing it. Whenever any result about the asked relationship is reported, "
    "the answer is \"Yes.\" or \"No.\", never \"Maybe.\".\n"
    "Answer rules:\n"
    "1. Begin with exactly one of: \"Yes.\" / \"No.\" / \"Maybe.\"\n"
    "2. Summarize the main conclusion from the passage in exactly ONE short sentence (\u226425 words).\n"
    "3. Preserve key phrases and medical terms from the passage; do not replace them with synonyms.\n"
    "4. Always include explicitly stated conditions, subgroups, or limitations if they appear in the conclusion.\n"
    "5. Do NOT add recommendations, explanations, or new information.\n"
    "6. The final output must be exactly ONE LINE:\n"
    "Answer:[Yes./No./Maybe. + short sentence]"
)

# v3 = scope repair of v2's No rule. v2 dev: Idx +2.15 but F1 -2.64 (over the
# 0.5 budget). Of its 10 dev flips, 3 won (Maybe->No, Yes->No on gt=no;
# Maybe->Yes on a hedged-positive gt=yes) and 7 lost, all gt=yes -> No. Loss
# autopsy: 4x the passage's negative/null sentence was about a DIFFERENT
# subgroup/dose/agonist/comparison than the asked relationship ("no significant
# difference between the two stent groups" while asked stenting-vs-surgery;
# "butaprost had little effect" while asked PGE2), 2x question polarity was
# inverted ("Is A independent of B?" / "reveal No abnormalities" -- there the
# absence of effect IS the yes-statement), 1x side remark. v3 therefore makes
# the polarity explicit (rule 1), keeps the null->No push only for
# effect-questions (rule 3), and quarantines off-topic negatives (rule 4).
_PUBMEDQA_SYSTEM_V3 = (
    "You will be given a PubMed-style passage and a Yes/No/Maybe question.\n"
    "DECISION RULES:\n"
    "1. First decide what \"Yes.\" means for THIS question: the asked relationship, with its own intervention "
    "or exposure, outcome, and comparison. Beware of question polarity -- for questions of the form "
    "\"Is A independent of B?\" or \"Does A show no difference from C?\", the YES statement is itself an absence "
    "of effect.\n"
    "2. Answer \"Yes.\" when the passage's conclusion supports that statement. Support counts even when it is "
    "hedged (\"may\", \"might\", \"suggests\", \"appears\", \"likely\") and even when it is a null result that "
    "matches an absence-of-effect question.\n"
    "3. Answer \"No.\" when the passage's conclusion contradicts that statement: the asked effect or association "
    "is absent (no significant difference, did not differ, was not associated with, did not improve or prevent, "
    "no evidence to support, missed the target) or goes the other way. For a question of the form "
    "\"Does X improve / predict / associate with Y?\", a null result is a definite \"No.\", NOT a \"Maybe.\".\n"
    "4. Judge only the finding about the asked relationship. A null or negative statement about a different "
    "subgroup, dose, drug, agonist, comparison, or a side remark does NOT answer the question and must not "
    "turn \"Yes.\" into \"No.\".\n"
    "5. Answer \"Maybe.\" ONLY when the passage does not address the asked relationship at all, or gives no "
    "data on it.\n"
    "Answer rules:\n"
    "1. Begin with exactly one of: \"Yes.\" / \"No.\" / \"Maybe.\"\n"
    "2. Summarize the main conclusion from the passage in exactly ONE short sentence (\u226425 words).\n"
    "3. Preserve key phrases and medical terms from the passage; do not replace them with synonyms.\n"
    "4. Always include explicitly stated conditions, subgroups, or limitations if they appear in the conclusion.\n"
    "5. Do NOT add recommendations, explanations, or new information.\n"
    "6. The final output must be exactly ONE LINE:\n"
    "Answer:[Yes./No./Maybe. + short sentence]"
)

# v4 = two lexical patches on v3's rules, aimed at its 7 dev misses.
#   (a) gt=yes Maybe-miss (NK/IL-12): the conclusion was suggestive about the
#       asked relationship ("Our results suggest...", lpM beat lpY by 0.75) --
#       rule 2 now says a suggestive conclusion is a Yes, and rule 5 bans
#       Maybe whenever ANY data on the asked relationship exists.
#   (b) gt=no Yes-miss (TMJ ultrasound): "does not support the recommendation
#       of ultrasound as a conclusive diagnostic tool" -- added to rule 3's
#       contradiction list, together with a negation-scope hint for the AED
#       double-negative ("no group experienced a significant change" = no
#       change anywhere; lpY-lpN was only 0.5 there).
# v3's polarity rule and off-topic quarantine stay byte-identical.
_PUBMEDQA_SYSTEM_V4 = (
    "You will be given a PubMed-style passage and a Yes/No/Maybe question.\n"
    "DECISION RULES:\n"
    "1. First decide what \"Yes.\" means for THIS question: the asked relationship, with its own intervention "
    "or exposure, outcome, and comparison. Beware of question polarity -- for questions of the form "
    "\"Is A independent of B?\" or \"Does A show no difference from C?\", the YES statement is itself an absence "
    "of effect.\n"
    "2. Answer \"Yes.\" when the passage's conclusion supports that statement. Support counts even when it is "
    "hedged or suggestive (\"may\", \"might\", \"suggests\", \"our results suggest\", \"appears\", \"likely\") and "
    "even when it is a null result that matches an absence-of-effect question.\n"
    "3. Answer \"No.\" when the passage's conclusion contradicts that statement: the asked effect or association "
    "is absent (no significant difference, did not differ, was not associated with, did not improve or prevent, "
    "no evidence to support, does not support the use or recommendation of, missed the target) or goes the other "
    "way. Read the scope of negation: \"no group showed a significant change\" means there was no change "
    "anywhere. For a question of the form \"Does X improve / predict / associate with Y?\", a null result is a "
    "definite \"No.\", NOT a \"Maybe.\".\n"
    "4. Judge only the finding about the asked relationship. A null or negative statement about a different "
    "subgroup, dose, drug, agonist, comparison, or a side remark does NOT answer the question and must not "
    "turn \"Yes.\" into \"No.\".\n"
    "5. Answer \"Maybe.\" ONLY when the passage does not address the asked relationship at all, or gives no "
    "data on it. Whenever the passage reports data on the asked relationship -- even a suggestive or indirect "
    "finding -- the answer is \"Yes.\" or \"No.\", never \"Maybe.\".\n"
    "Answer rules:\n"
    "1. Begin with exactly one of: \"Yes.\" / \"No.\" / \"Maybe.\"\n"
    "2. Summarize the main conclusion from the passage in exactly ONE short sentence (≤25 words).\n"
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
        if 'pope' in ds:
            return ("You are a helpful assistant. Look at the image and answer the "
                    "question. Answer with a single word, either Yes or No.")
        return _DEFAULT_SYSTEM_V0

    # v1+
    if 'cmm' in ds:
        return {'v1': _CMM_SYSTEM, 'v2': _CMM_SYSTEM_V2, 'v3': _CMM_SYSTEM_V3,
                'v4': _CMM_SYSTEM_V4, 'v5': _CMM_SYSTEM_V5,
                'v6': _CMM_SYSTEM_V6}.get(prompt_version, _CMM_SYSTEM)
    if 'avh' in ds:
        return _AVH_SYSTEMS.get(prompt_version, _AVH_SYSTEM_V1)
    if 'phd' in ds:
        return _PHD_SYSTEM
    if 'halo' in ds:
        if prompt_version == 'v2':
            return _HALO_SYSTEM_V2
        if prompt_version == 'v3':
            return _HALO_SYSTEM_V3
        if prompt_version == 'v4':
            return _HALO_SYSTEM_V4
        return _HALO_SYSTEM
    if 'halueval' in ds:
        return _HALUEVAL_SYSTEM
    if 'ragtruth' in ds:
        if prompt_version == 'v2':
            return _RAGTRUTH_SYSTEM_V2
        if prompt_version == 'v3':
            return _RAGTRUTH_SYSTEM_V3
        if prompt_version == 'v4':
            return _RAGTRUTH_SYSTEM_V4
        if prompt_version == 'v5':
            return _RAGTRUTH_SYSTEM_V5
        return _RAGTRUTH_SYSTEM
    if 'pubmedqa' in ds:
        if prompt_version == 'v2':
            return _PUBMEDQA_SYSTEM_V2
        if prompt_version == 'v3':
            return _PUBMEDQA_SYSTEM_V3
        if prompt_version == 'v4':
            return _PUBMEDQA_SYSTEM_V4
        if prompt_version == 'v5':
            # v4 was rolled back (dev Idx 77.33 < v3 86.43: its suggestive->Yes
            # patch weakened No on suggestive-negative conclusions with lpY/lpN
            # gaps under 0.1), so v5 builds on the retained v3 system and tests
            # only the user-template fix (drop "and your general knowledge").
            return _PUBMEDQA_SYSTEM_V3
        if prompt_version == 'v6':
            # Attribution arm, NOT a dev-selected version: full-set 2x2 cell
            # (system v1/v3) x (user v1/fixed). On the full 460 the v3 system
            # rules cost Idx -1.62 while the user-template fix gained +3.20
            # (v3->v5), so this cell isolates the user fix on the v1 system.
            return _PUBMEDQA_SYSTEM
        return _PUBMEDQA_SYSTEM
    if 'summedits' in ds:
        return (
            "You are given a document and a summary. Your task is to determine whether the summary is "
            "factually consistent with the document. Answer only 'Yes' if the summary is consistent, "
            "or 'No' if it contains any factual inconsistency."
        )
    # POPE (public-benchmark arm): standard object-hallucination yes/no QA
    # (Li et al., EMNLP 2023). Same template for v0 and v1 -- nothing tuned.
    if 'pope' in ds:
        return ("You are a helpful assistant. Look at the image and answer the "
                "question. Answer with a single word, either Yes or No.")
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
        if 'pope' in ds:
            return f"Question: {q_text}\nAnswer with Yes or No."
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
    # Prompt-program versions (v2+) change the SYSTEM prompt only; the user
    # template is byte-identical to v1 so the version diff isolates one
    # direction. The legacy methods_cmm user templates stay reachable under
    # lv2/lv3/lv4 (they were never run through this module's cmm v2/v3/v4
    # before 2026-08-16; no registered artifact depends on those keys).
    if 'cmm' in ds:
        if prompt_version == 'lv2':
            return (
                f"Analyze the media carefully. First, briefly list the key visual objects and audio events observed. "
                f"Then, based on that list, determine if the statement '{q_text}' is factually accurate. "
                "Final Answer: [Yes/No]"
            )
        if prompt_version == 'lv3':
            return (
                f"Task: Verify the existence of '{q_text}' through independent modal cross-checking.\n"
                "Step 1: Check the video stream for visual evidence.\n"
                "Step 2: Check the audio stream for auditory evidence.\n"
                "Step 3: Compare both sources. Only confirm 'Yes' if there is direct, non-conflicting evidence. "
                "If one modality suggests it but the other contradicts, prioritize empirical observation over intuition.\n"
                "Final Answer:"
            )
        if prompt_version == 'lv4':
            return f"Analyze the video and audio inputs to verify the following: {q_text}. Answer with Yes or No."
        # v1 (and v2+ prompt-program versions)
        return f"Task: Analyze the video and audio to answer the question.\nQuestion: {q_text}\nFinal Answer:"

    # ── AVH ──────────────────────────────────────────────────────────────────
    if 'avh' in ds:
        return q_text

    # ── PhD ──────────────────────────────────────────────────────────────────
    # v2 (2026-08-16, prompt-iteration program): single-direction change -- the
    # v1 user instruction licensed hallucination ("based on the provided
    # context", "common sense", "accuracy to visual specifics is not required"),
    # which explains both the 83% bare-answer rate (decision at step 0, no
    # visual evidence written down) and the context-primed FP errors (39/186
    # residual errors had >=2 question content words pre-mentioned by the
    # context). v2 replaces it with a visual evidence chain; the system prompt
    # stays byte-identical to v1 (_PHD_SYSTEM) so attribution is isolated to
    # the user template. The legacy v2 draft string was never run (registry
    # had no phd v2 entry before this one); v3 draft kept untouched.
    if 'phd' in ds:
        if prompt_version == 'v2':
            instruction = (
                "Judge the hypothesis strictly from what is directly visible in the image. "
                "Do not use the text context or common sense to fill in what you cannot see. "
                "If the hypothesis is about a number of objects, count every visible instance "
                "one by one, including partially visible ones, before answering. "
                "If it is about an attribute, answer Yes only if you can actually see it in the image."
            )
        elif prompt_version == 'v3':
            # v3 (2026-08-16): v2's written evidence chain was REJECTED on dev
            # (F1 78.29 vs 84.39, Youden 0.479 vs 0.550, McNemar p=0.044): the
            # chain hardened perception misses into text ("no chair is visible"
            # -> No on gt=yes; broken 28 vs fixed 14, 23/28 losses were FN) and
            # collapsed the CF intervention rate 21.3% -> 4.8% (committed
            # written evidence leaves no margin for the operator to flip).
            # v3 keeps the v1 bare-answer decision mode (CF's calibrated
            # operating point) and removes ONLY the three licensing phrases:
            # "based on the provided context" (context-following), "common
            # sense to fill in details" (hallucination license), "accuracy to
            # visual specifics is not required" (anti-visual). Symmetric
            # Yes/No clause -- no confirmation-threshold asymmetry.
            instruction = (
                "Answer directly from the image. The text context is an unverified story that often "
                "misdescribes the image; never use it as evidence for what is or is not visible. "
                "Judge the hypothesis on the image alone: answer Yes when the image shows what it "
                "states, and No when it does not."
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
            # Output-contract v2: pairs with _HALO_SYSTEM_V2 (D1 avoidance
            # template + D2 minimum sentence length).
            return (
                f"Question: {q_text}\n"
                "Look at the image carefully and answer with one complete sentence of at least 8 words. "
                "If the object or detail asked about is not visible, answer with the template: "
                "\"There is no X visible in this image.\""
            )
        if prompt_version == 'v3':
            # Output-contract v3 (P3, §8.2): pairs with _HALO_SYSTEM_V3 —
            # sentence-pattern echo + counting complete sentences; the absence
            # template is demoted to last resort. (Supersedes the legacy
            # "meticulous visual auditor" v3 user template, which was never
            # part of the shipped v1->v2 lineage.)
            return (
                f"Question: {q_text}\n"
                "Look at the image carefully and answer with one complete sentence of at least 8 words "
                "built from the question's own words. If the question asks \"How many\", answer with a full "
                "counting sentence such as \"There are two cups on the table.\" Only use the template "
                "\"There is no X visible in this image.\" when the asked object or detail is truly not "
                "visible anywhere in the image."
            )
        if prompt_version == 'v4':
            # Output-contract v4 (round 2): pairs with _HALO_SYSTEM_V4 —
            # precision/brevity: subject + asked attribute only, 8-12 words,
            # no un-asked details; counting + last-resort absence kept from v3.
            return (
                f"Question: {q_text}\n"
                "Look at the image carefully and answer with one complete sentence of 8 to 12 words "
                "that repeats the question's subject and states the asked attribute. Mention nothing "
                "the question did not ask about. If the question asks \"How many\", answer with a full "
                "counting sentence such as \"There are two cups on the table.\" Use the template "
                "\"There is no X visible in this image.\" only when the asked object or detail is truly "
                "not visible."
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
    # RAGTruth v2/v3 output-contract user template (TRAJECTORY_ANALYSIS.md §2):
    # v1 minus the " and your general knowledge" clause (R3), "based on" ->
    # "based only on". Same template serves both v2 and v3 (the system prompt
    # carries the R2/R5 additions).
    if 'ragtruth' in ds and prompt_version in ('v2', 'v3', 'v4', 'v5'):
        return (
            f"Passage: {passage_str}\n"
            "Instruction: Please answer the following question based only on the provided passage.\n"
            f"Question: {q_text}\n"
            "Answer:"
        )
    # PubMedQA prompt versions (v2+) keep the v1 Passage template unless a
    # version explicitly changes it -- without this branch the generic v2/v3
    # text templates below would silently swap the pubmedqa user prompt.
    if 'pubmedqa' in ds:
        # v5 = v4 system + user-template surgery: drop "and your general
        # knowledge" (RAGTruth R3 analog). The v1 full-run yes-misses that
        # never moved under any system prompt sat on deep priors ("Double
        # reading ... is it necessary?" lp_yes -0.62 vs lp_no -1.43);
        # restricting the evidence source to the passage only targets that
        # parametric pull. v2-v4 keep the v1 template byte-identical.
        if prompt_version in ('v5', 'v6'):
            return (
                f"Passage: {passage_str}\n"
                "Instruction: Please answer the following question based only on the provided passage.\n"
                f"Question: {q_text}\n"
                "Answer:"
            )
        return (
            f"Passage: {passage_str}\n"
            "Instruction: Please answer the following question based on the provided passage "
            "and your general knowledge.\n"
            f"Question: {q_text}\n"
            "Answer:"
        )
    if prompt_version == 'v2':
        return (
            f"Read the following text and answer the question.\n\n"
            f"Text: {passage_str}\n\n"
            f"Question: {q_text}\n\n"
            "Answer:"
        )
    if prompt_version == 'v3':
        return f"Document: {passage_str}\nQuestion: {q_text}\nAnswer:"
    if 'pope' in ds:
        return f"Question: {q_text}\nAnswer with Yes or No."

    # v1
    return (
        f"Passage: {passage_str}\n"
        "Instruction: Please answer the following question based on the provided passage "
        "and your general knowledge.\n"
        f"Question: {q_text}\n"
        "Answer:"
    )
