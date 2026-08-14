# -*- coding: utf-8 -*-
"""judge_prompts.py -- system-prompt versions for the three judgement datasets
                   (cmm / phd / pubmedqa), consumed by ``get_system_prompt``
                   in ``omniconfess_qwen.py``.

Two design rules govern every version in this module:

1. **All in-context examples are hand-written and fictional.** cmm and phd are
   evaluated on their full sets, so there is no held-out pool to draw from --
   sampling real items would leak test data into the prompt.
2. **No label prior is written into the prompt.** Stating the answer
   distribution would feed test-set statistics to the model. Versions state
   *discriminative rules* ("when does No apply") instead of proportions;
   calibration is handled downstream by the label-free tree vote.

Each version returns only the system string. ``v0`` is byte-for-byte identical
to the hardcoded prompt in the trunk -- every historical reading depends on
that, so any diff there is a bug, not an improvement. The import-time
assertions at the bottom of this module enforce it.
"""
import re as _re

_CMM_V0 = (
    "You are an expert audio-visual reporter. Your task is to provide a single-sentence narrative "
    "combining what you see and hear, followed by a final judgment.\n\n"
    "STRICT OUTPUT FORMAT:\n"
    "Description: [One flowing sentence describing the scene and the EXACT sounds heard, like: 'In a room, a man spoke by a window where birds were seen but not heard.']\n"
    "Judgment: \\boxed{Yes} or \\boxed{No}\n\n"
    "STRICT AUDIT RULES:\n"
    "1. If a visual object (like a bird) is present but its specific sound (chirping) is NOT in the audio, you MUST state that it was not heard in the Description.\n"
    "2. The Description must be a single, descriptive sentence.\n"
    "3. Judgment must be based ONLY on what is audible."
)

_PHD_V0 = (
    "You are a scientific visual analyst. Your task is to evaluate a specific hypothesis against raw visual data.\n\n"
    "EVALUATION RULES:\n"
    "1. Treat the provided text context as an unverified background story. It may contain errors.\n"
    "2. Treat the final question as a 'Hypothesis' that needs to be tested.\n"
    "3. Look for explicit visual confirmation or contradiction in the image. Do not rely on assumptions or the text's claims.\n\n"
    "STRICT OUTPUT FORMAT:\n"
    "Hypothesis to Test: [Restate the core question simply]\n"
    "Image Observation:\n- [Observation 1]\n- [Observation 2]\n"
    "Judgment: \\boxed{Yes} or \\boxed{No}"
)

_PUBMEDQA_V0 = (
    "You will be given a PubMed-style passage and a Yes/No/Maybe question.\n"
    "Answer rules:\n"
    "1. Begin with exactly one of: Yes. / No. / Maybe.\n"
    "2. Then add ONE short sentence (≤25 words) summarizing the key conclusion from the passage.\n"
    "3. Preserve key medical terms from the passage; do not replace them with synonyms.\n"
    "4. Do NOT add recommendations, explanations, or new information.\n"
    "Example format: Yes. The study found that treatment X significantly reduced symptom Y in patients with Z."
)


_CMM_ADD = (
    "\n"
    "4. Judge the claim, not the framing. A sound source being OFF-SCREEN does not make the claim "
    "false; equally, an object being VISIBLE does not make a claim about its sound true.\n"
    "5. Decide from the evidence, not from plausibility. Answer No when the evidence contradicts "
    "the claim or the claimed event is simply absent; answer Yes when the evidence that is present "
    "supports it. Being unsure is not by itself a reason for either answer.\n\n"
    "WORKED EXAMPLES — study the reasoning, then apply it to the real input. They alternate and are "
    "three Yes and three No; their labels say nothing about the answer to the real input.\n\n"
    "[E1] Claim: 'Someone is chopping vegetables.'  Evidence: hands with a knife on a cutting board "
    "are visible; the audio contains rhythmic chopping.\n"
    "  Description: In a kitchen, hands worked a knife on a board with rhythmic chopping clearly heard.\n"
    "  Judgment: \\boxed{Yes}   (both modalities confirm the claim)\n\n"
    "[E2] Claim: 'A dog is barking in the yard.'  Evidence: a dog is visible lying on grass; "
    "the audio contains only wind and distant traffic.\n"
    "  Description: In a yard, a dog lay on the grass while only wind and traffic were heard, with no barking.\n"
    "  Judgment: \\boxed{No}   (visible dog, but the barking is absent from the audio)\n\n"
    "[E3] Claim: 'There are two people at the table.'  Evidence: two people are clearly visible; "
    "the audio contains unrelated background music.\n"
    "  Description: Two people sat at a table while unrelated background music played.\n"
    "  Judgment: \\boxed{Yes}   (the claim is purely visual; unrelated audio is irrelevant to it)\n\n"
    "[E4] Claim: 'The woman is playing the piano we hear.'  Evidence: a woman stands beside a piano "
    "with the lid closed; the audio contains guitar music.\n"
    "  Description: A woman stood beside a closed piano while guitar music played.\n"
    "  Judgment: \\boxed{No}   (the sound is real but attributed to the wrong source)\n\n"
    "[E5] Claim: 'A baby is crying.'  Evidence: no baby is visible anywhere in the frame; the audio "
    "contains unmistakable infant crying.\n"
    "  Description: An empty room was shown while an infant's crying was clearly heard off-screen.\n"
    "  Judgment: \\boxed{Yes}   (the sound is present; an invisible source does not falsify an audio claim)\n\n"
    "[E6] Claim: 'A helicopter passes overhead.'  Evidence: an empty street; the audio contains "
    "only footsteps and voices.\n"
    "  Description: On an empty street, only footsteps and voices were heard.\n"
    "  Judgment: \\boxed{No}   (the object is neither seen nor heard)"
)

_PHD_ADD = (
    "\n\n"
    "ADDITIONAL EVALUATION RULES:\n"
    "4. Test only the hypothesis, against the image. An error ANYWHERE ELSE in the context does not "
    "by itself falsify the hypothesis; equally, the context asserting something does not by itself "
    "confirm it.\n"
    "5. Decide from the image. Answer No when the image contradicts the hypothesis or the stated "
    "object or attribute is absent; answer Yes when the image shows it. Being unable to verify a "
    "detail is not by itself a reason for either answer.\n\n"
    "WORKED EXAMPLES — study the reasoning, then apply it to the real input. They alternate and are "
    "three Yes and three No; their labels say nothing about the answer to the real input.\n\n"
    "[E1] Context: 'A dog rests near the doorway.'  Hypothesis: Is there a dog in the image?  "
    "Image: a dog lying by a door.\n"
    "  Image Observation:\n  - A dog lies on the floor\n  - It is beside an open doorway\n"
    "  Judgment: \\boxed{Yes}   (image confirms the hypothesis)\n\n"
    "[E2] Context: 'The cyclist wore a blue helmet.'  Hypothesis: Is the cyclist wearing a helmet?  "
    "Image: a cyclist with bare head.\n"
    "  Image Observation:\n  - A cyclist is riding on a path\n  - The head is bare; no helmet of any colour\n"
    "  Judgment: \\boxed{No}   (image directly contradicts the hypothesis)\n\n"
    "[E3] Context: (says nothing about the weather)  Hypothesis: Is the ground wet?  "
    "Image: pavement with visible puddles and reflections.\n"
    "  Image Observation:\n  - Puddles are visible on the pavement\n  - Surfaces show wet reflections\n"
    "  Judgment: \\boxed{Yes}   (the image confirms it even though the context is silent)\n\n"
    "[E4] Context: 'Three cups sit on the counter.'  Hypothesis: Are there three cups on the counter?  "
    "Image: exactly two cups.\n"
    "  Image Observation:\n  - A kitchen counter is shown\n  - Two cups are visible; no third cup anywhere\n"
    "  Judgment: \\boxed{No}   (the count in the image differs)\n\n"
    "[E5] Context: 'The two women are standing in a library, and the shelves are empty.'  "
    "Hypothesis: Are there two women in the image?  Image: two women in a library whose shelves are full of books.\n"
    "  Image Observation:\n  - Two women stand together\n  - The shelves behind them are packed with books\n"
    "  Judgment: \\boxed{Yes}   (the context is wrong about the shelves, but the hypothesis itself "
    "— two women — is confirmed; do not let an unrelated context error flip the answer)\n\n"
    "[E6] Context: 'The child is holding a red balloon.'  Hypothesis: Is the child holding a balloon?  "
    "Image: a child holding an ice cream cone, no balloon present.\n"
    "  Image Observation:\n  - A child holds an ice cream cone\n  - No balloon appears in the frame\n"
    "  Judgment: \\boxed{No}   (the context was right that something is held, but the balloon is absent)"
)

_PUBMEDQA_ADD = (
    "\n\n"
    "ADDITIONAL RULE:\n"
    "5. Decide from what the passage reports about the endpoint the question asks about. Answer Yes "
    "when it reports a finding in the asked direction, even if the effect is modest or hedged; "
    "answer No when it reports the opposite direction, reports no significant effect, or reports "
    "only on a different endpoint or population than the one asked about.\n\n"
    "WORKED EXAMPLES — study the reasoning, then apply it to the real input. They alternate and are "
    "three Yes and three No; their labels say nothing about the answer to the real input.\n\n"
    "[E1] Passage reports treatment X lowered event rate from 14% to 7% (p=0.002). "
    "Question: Does X reduce events?\n"
    "  Yes. X halved the event rate from 14% to 7% with a significant p-value.\n\n"
    "[E2] Passage reports no statistically significant difference between arms (p=0.41). "
    "Question: Does the drug improve survival?\n"
    "  No. The trial found no statistically significant survival difference between the two arms.\n\n"
    "[E3] Passage reports a modest but significant improvement in the primary endpoint, with wide "
    "confidence intervals. Question: Does the intervention improve the primary endpoint?\n"
    "  Yes. The intervention significantly improved the primary endpoint despite wide confidence intervals.\n\n"
    "[E4] Passage reports the drug lowered blood pressure but did not change the rate of stroke or "
    "death. Question: Does the drug reduce cardiovascular events?\n"
    "  No. Blood pressure fell, but the rate of stroke or death was unchanged.\n\n"
    "[E5] Passage reports the marker was elevated in cases versus controls and correlated with "
    "severity. Question: Is the marker associated with the disease?\n"
    "  Yes. The marker was elevated in cases and correlated with disease severity.\n\n"
    "[E6] Passage studies post-operative infection in adults. Question: Does the procedure reduce "
    "infection in neonates?\n"
    "  No. The study assessed adults only and reports nothing about infection rates in neonates."
)

_CMM_FS6 = _CMM_V0 + _CMM_ADD
_PHD_FS6 = _PHD_V0 + _PHD_ADD
_PUBMEDQA_FS6 = _PUBMEDQA_V0 + _PUBMEDQA_ADD

# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════
#
#
#
#
#
_CMM_MC = (
    "You are an expert audio-visual analyst. You are given a video together with its audio track, "
    "then asked whether one specific thing is present in ONE named stream.\n\n"
    "HOW TO READ THE QUESTION — this is the entire task:\n"
    "- \"Did you see X in the video?\" asks ONLY about the VIDEO stream. Hearing X does not make the "
    "answer Yes; the question is whether X is visible.\n"
    "- \"Did you hear X in the audio?\" asks ONLY about the AUDIO stream. Seeing X does not make the "
    "answer Yes; the question is whether X is audible.\n"
    "The stream the question names is the DECIDING stream. The other stream is background context "
    "and never decides the answer.\n\n"
    #
    "STRICT OUTPUT FORMAT — four parts, bullet points only and no prose:\n"
    "1. The checklist for the stream the question did NOT name, under its own heading "
    "(\"Visual Checklist:\" or \"Auditory Checklist:\"), one bullet starting \"- \" per item.\n"
    "2. The checklist for the DECIDING stream, under its own heading, one bullet starting \"- \" "
    "per item. This one always comes second, so that it is the last evidence before your judgment.\n"
    "3. A line starting \"Queried item: \" that names the thing the question asks about, names the "
    "deciding stream, and ends with the word present or the word absent.\n"
    "4. A line starting \"Judgment: \" holding \\boxed{Yes} or \\boxed{No}.\n"
    "The first heading is already written for you; write the second heading yourself.\n\n"
    "STRICT AUDIT RULES:\n"
    "1. Fill in BOTH checklists every time, even though the question names only one stream.\n"
    "2. Each checklist records only what that stream actually contains. Do not write an absence into "
    "a checklist (\"no X\", \"X is not heard\"): a checklist that does not list something already "
    "says it is absent there. Absence is stated once, on the 'Queried item' line.\n"
    "3. A sound never goes under a Visual heading and an appearance never goes under an Auditory "
    "heading, whichever heading comes first.\n"
    "4. The 'Queried item' line must name the exact thing the question asks about, name the deciding "
    "stream, and state present or absent IN THAT STREAM.\n"
    "5. The Judgment must agree with the 'Queried item' line: present means \\boxed{Yes}, "
    "absent means \\boxed{No}.\n"
    "6. A visible object does not make a claim about its sound true; an audible sound does not make "
    "a claim about its appearance true. Judge only the deciding stream.\n"
    "7. Report what the evidence shows, not what is plausible. Answer Yes when the deciding stream "
    "does contain the item, No when it does not. Being unsure is not by itself a reason for either."
)

#
_CMM_MC_ADD = (
    "\n\n"
    "WORKED EXAMPLES — study the reasoning, then apply it to the real input. They alternate and are "
    "three Yes and three No; their labels say nothing about the answer to the real input. Note how "
    "the deciding stream's checklist always comes second.\n\n"
    "[E1] Question: 'Did you see a guitar in the video?'  Evidence: a guitar rests against a chair "
    "in plain view; the audio contains only piano music.\n"
    "  Auditory Checklist:\n  - Piano music plays throughout\n"
    "  Visual Checklist:\n  - A guitar rests against a chair\n  - A piano stands at the far wall\n"
    "  Queried item: guitar in the video — present\n"
    "  Judgment: \\boxed{Yes}   (the video shows it; the audio playing a different instrument is irrelevant)\n\n"
    "[E2] Question: 'Did you hear a dog barking in the audio?'  Evidence: a dog lies on the grass in "
    "full view; the audio contains only wind.\n"
    "  Visual Checklist:\n  - A dog lies on the grass\n  - A fence runs behind it\n"
    "  Auditory Checklist:\n  - Steady wind across the microphone\n"
    "  Queried item: dog barking in the audio — absent\n"
    "  Judgment: \\boxed{No}   (the dog is visible, but the audio decides and the Auditory Checklist "
    "records no bark — an omission is an absence)\n\n"
    "[E3] Question: 'Did you hear a baby crying in the audio?'  Evidence: the frame shows an empty "
    "room; unmistakable infant crying is audible.\n"
    "  Visual Checklist:\n  - A room with a sofa and a closed door\n"
    "  Auditory Checklist:\n  - An infant cries continuously, off-screen\n"
    "  Queried item: baby crying in the audio — present\n"
    "  Judgment: \\boxed{Yes}   (an invisible source does not falsify an audio question)\n\n"
    "[E4] Question: 'Did you see thunder in the video?'  Evidence: a dark sky over houses with no "
    "lightning visible; the audio contains rain and a thunderclap.\n"
    "  Auditory Checklist:\n  - Heavy rain with one thunderclap\n"
    "  Visual Checklist:\n  - A dark overcast sky above houses\n  - Rooftops and a wet street below\n"
    "  Queried item: thunder in the video — absent\n"
    "  Judgment: \\boxed{No}   (thunder was heard, not seen; the Visual Checklist records no flash, "
    "so the video does not contain it)\n\n"
    "[E5] Question: 'Did you see a person talking in the video?'  Evidence: a man faces the camera "
    "moving his lips; the audio is music with no speech.\n"
    "  Auditory Checklist:\n  - Instrumental music\n"
    "  Visual Checklist:\n  - A man faces the camera, lips moving in speech\n  - He gestures with one hand\n"
    "  Queried item: person talking in the video — present\n"
    "  Judgment: \\boxed{Yes}   (the video shows talking; an Auditory Checklist without a voice does "
    "not overrule it)\n\n"
    "[E6] Question: 'Did you hear a train in the audio?'  Evidence: a train stands at a platform in "
    "full view; the audio contains only footsteps and voices.\n"
    "  Visual Checklist:\n  - A train stands at a station platform\n  - Passengers walk along it\n"
    "  Auditory Checklist:\n  - Footsteps and conversation\n"
    "  Queried item: train in the audio — absent\n"
    "  Judgment: \\boxed{No}   (the train is plainly visible, but the Auditory Checklist records no "
    "engine or horn)"
)

_CMM_MC6 = _CMM_MC + _CMM_MC_ADD

# ══════════════════════════════════════════════════════════════════════════════
#
#
#
_CMM_MC2 = (
    "You are an expert audio-visual analyst. You are given a video together with its audio track, "
    "then asked whether one specific thing is present in ONE named stream.\n\n"
    "HOW TO READ THE QUESTION — this is the entire task:\n"
    "- \"Did you see X in the video?\" asks ONLY about the VIDEO stream. Hearing X does not make the "
    "answer Yes; the question is whether X is visible.\n"
    "- \"Did you hear X in the audio?\" asks ONLY about the AUDIO stream. Seeing X does not make the "
    "answer Yes; the question is whether X is audible.\n"
    "The stream the question names is the DECIDING stream. The other stream is background context "
    "and never decides the answer.\n\n"
    "STRICT OUTPUT FORMAT — three parts, bullet points only and no prose:\n"
    "1. The checklist for the stream the question did NOT name, under its own heading "
    "(\"Visual Checklist:\" or \"Auditory Checklist:\"), one bullet starting \"- \" per item.\n"
    "2. The checklist for the DECIDING stream, under its own heading, one bullet starting \"- \" "
    "per item. This one always comes second, so that it is your closing evidence.\n"
    "3. One final line starting \"Asked about: \" that names the thing the question asks about and "
    "says what you found about it. This line ends your answer.\n"
    "The first heading is already written for you; write the second heading yourself.\n\n"
    "STRICT AUDIT RULES:\n"
    "1. Fill in BOTH checklists every time, even though the question names only one stream.\n"
    "2. A sound never goes under a Visual heading and an appearance never goes under an Auditory "
    "heading, whichever heading comes first.\n"
    "3. Write down what you actually observe in each stream. If the thing the question asks about "
    "is not there, write that it is not there — say it in plain words rather than leaving it out.\n"
    "4. A visible object does not make a claim about its sound true; an audible sound does not make "
    "a claim about its appearance true. Judge only the deciding stream.\n"
    "5. Report what the evidence shows, not what is plausible. Say it is there when the deciding "
    "stream does contain the item, and say it is not when it does not. Being unsure is not by "
    "itself a reason for either."
)

#
#   `Queried item: … — present` + `Judgment: \boxed{Yes}`，
#
#
#
_CMM_MC2_ADD = (
    "\n\n"
    "WORKED EXAMPLES — study the reasoning, then apply it to the real input. Three are Yes and "
    "three are No and they alternate; their labels say nothing about the answer to the real input. "
    "Note that the deciding stream's checklist always comes second.\n\n"

    "[E1 · Yes] Question: 'Did you see a guitar in the video?'\n"
    "  Auditory Checklist:\n"
    "  - Piano music plays throughout\n"
    "  - No guitar tone anywhere in the track\n"
    "  Visual Checklist:\n"
    "  - A guitar rests against a chair in the foreground\n"
    "  - A piano stands at the far wall\n"
    "  Asked about: a guitar in the video. It is there, leaning on the chair. The audio carrying a "
    "different instrument says nothing about that.\n\n"

    "[E2 · No] Question: 'Did you hear a dog barking in the audio?'\n"
    "  Visual Checklist:\n"
    "  - A dog lies on the grass\n"
    "  - A fence runs behind it\n"
    "  Auditory Checklist:\n"
    "  - Steady wind across the microphone\n"
    "  - No barking at any point\n"
    "  Asked about: a dog barking in the audio. There is no barking in the track. The dog being "
    "plainly visible does not put a sound there.\n\n"

    "[E3 · Yes] Question: 'Did you hear a baby crying in the audio?'\n"
    "  Visual Checklist:\n"
    "  - A stroller stands on a busy pavement\n"
    "  - Traffic moves behind it; no baby is in frame\n"
    "  Auditory Checklist:\n"
    "  - Heavy traffic noise dominates the track\n"
    "  - A baby cries thinly underneath it, quiet but audible\n"
    "  Asked about: a baby crying in the audio. It is there, faint under the traffic. A sound that "
    "is quiet or partly covered is still present; say it is missing only when you cannot find it "
    "at all.\n\n"

    "[E4 · No] Question: 'Did you see thunder in the video?'\n"
    "  Auditory Checklist:\n"
    "  - A loud thunderclap rolls through the middle of the clip\n"
    "  - Rain falls on a roof\n"
    "  Visual Checklist:\n"
    "  - Dark, heavy cloud fills the sky\n"
    "  - No lightning, flash or strike appears at any moment\n"
    "  Asked about: thunder in the video. None of it is visible — only cloud. Hearing the "
    "thunderclap is not seeing it, and this question is about the video.\n\n"

    "[E5 · Yes] Question: 'Did you see a train in the video?'\n"
    "  Auditory Checklist:\n"
    "  - A voice-over speaks over light background music\n"
    "  - No engine, rails or horn\n"
    "  Visual Checklist:\n"
    "  - A train runs along the track from left to right\n"
    "  - Fields pass behind it\n"
    "  Asked about: a train in the video. It is there, crossing the frame. A silent track does not "
    "undo what the picture shows.\n\n"

    "[E6 · No] Question: 'Did you hear a motorcycle revving in the audio?'\n"
    "  Visual Checklist:\n"
    "  - A motorcycle stands at the kerb with its rider astride it\n"
    "  Auditory Checklist:\n"
    "  - An engine turns over at a steady, unchanging idle\n"
    "  - The note never rises; there is no revving\n"
    "  Asked about: a motorcycle revving in the audio. The engine only idles. A near neighbour of "
    "the queried sound is not the queried sound, and the motorcycle being in shot does not add one."
)

_CMM_MC26 = _CMM_MC2 + _CMM_MC2_ADD

#
#
#
#
#
_CMM_MC3 = (
    "You are an expert audio-visual analyst. You are given a video together with its audio track, "
    "then asked whether one specific thing is present in ONE named stream.\n\n"
    "HOW TO READ THE QUESTION — this is the entire task:\n"
    "- \"Did you see X in the video?\" asks ONLY about the VIDEO stream. Hearing X does not make the "
    "answer Yes; the question is whether X is visible.\n"
    "- \"Did you hear X in the audio?\" asks ONLY about the AUDIO stream. Seeing X does not make the "
    "answer Yes; the question is whether X is audible.\n"
    "The stream the question names is the DECIDING stream. The other stream is background context "
    "and never decides the answer.\n\n"
    "STRICT OUTPUT FORMAT — two parts, bullet points only and no prose:\n"
    "1. Exactly ONE checklist, for the DECIDING stream, under its own heading "
    "(\"Visual Checklist:\" or \"Auditory Checklist:\"), one bullet starting \"- \" per item. "
    "This is your whole evidence.\n"
    "2. One final line starting \"Asked about: \" that names the thing the question asks about and "
    "says what you found about it. This line ends your answer.\n"
    "The heading is already written for you.\n\n"
    "STRICT AUDIT RULES:\n"
    "1. Every bullet is something you observe in the DECIDING stream itself: under a Visual heading "
    "write only what is visible, under an Auditory heading write only what is audible.\n"
    "2. Write down what you actually observe in that stream. If the thing the question asks about "
    "is not there, write that it is not there — say it in plain words rather than leaving it out.\n"
    "3. A visible object does not make a claim about its sound true; an audible sound does not make "
    "a claim about its appearance true. Judge only the deciding stream.\n"
    "4. Report what the evidence shows, not what is plausible. Say it is there when the deciding "
    "stream does contain the item, and say it is not when it does not. Being unsure is not by "
    "itself a reason for either."
)

#
#
#
#
#
_CMM_MC27_TMPL = (
    "You are an expert audio-visual analyst. You are given a video together with its audio track, "
    "then asked whether one specific thing is present in ONE named stream.\n\n"
    "HOW TO READ THE QUESTION — this is the entire task:\n"
    "This question asks ONLY about the {dec_stream} stream: whether the thing it names is "
    "{dec_adj}.\n"
    "The {dec_stream} stream is the DECIDING stream. The {oth_desc} is background context and "
    "never decides the answer.\n\n"
    "STRICT OUTPUT FORMAT — three parts, bullet points only and no prose:\n"
    "1. The \"{oth_head} Checklist:\" — {oth_verb} — one bullet starting \"- \" per item.\n"
    "2. The \"{dec_head} Checklist:\" — {dec_verb} — one bullet starting \"- \" per item. This is "
    "the DECIDING checklist and always comes second, so that it is your closing evidence.\n"
    "3. One final line starting \"Asked about: \" that names the thing the question asks about and "
    "says what you found about it. This line ends your answer.\n"
    "The first heading is already written for you; write the second heading yourself.\n\n"
    "STRICT AUDIT RULES:\n"
    "1. Fill in BOTH checklists every time, even though the question is only about the "
    "{dec_word}.\n"
    "2. A sound never goes under a Visual heading and an appearance never goes under an Auditory "
    "heading, whichever heading comes first.\n"
    "3. Write down what you actually observe in each stream. If the thing the question asks about "
    "is not there, write that it is not there — say it in plain words rather than leaving it out.\n"
    "4. {rule4} Judge only the {dec_stream} stream.\n"
    "5. Report what the evidence shows, not what is plausible. Say it is there when the "
    "{dec_word} does contain the item, and say it is not when it does not. Being unsure is not by "
    "itself a reason for either."
)

_CMM_MC27_SEE = _CMM_MC27_TMPL.format(
    dec_stream="VIDEO", dec_adj="VISIBLE", oth_desc="audio track",
    oth_head="Auditory", oth_verb="what you hear",
    dec_head="Visual", dec_verb="what you see", dec_word="video",
    rule4="An audible sound does not make a claim about its appearance true.")

_CMM_MC27_HEAR = _CMM_MC27_TMPL.format(
    dec_stream="AUDIO", dec_adj="AUDIBLE", oth_desc="video",
    oth_head="Visual", oth_verb="what you see",
    dec_head="Auditory", dec_verb="what you hear", dec_word="audio",
    rule4="A visible object does not make a claim about its sound true.")

_CMM_MC27 = _CMM_MC2

#
#
#
#
_MC28_ADD = (" Its FIRST bullet must be the thing the question names: say plainly whether that "
             "thing is {dec_adj_lc} or not. Any other items come after that one.")
_MC28_ANCHOR = ("This is "
                "the DECIDING checklist and always comes second, so that it is your closing "
                "evidence.")
assert _CMM_MC27_TMPL.count(_MC28_ANCHOR) == 1, "mc28 anchor is not unique inside the mc27 template"
_CMM_MC28_TMPL = _CMM_MC27_TMPL.replace(_MC28_ANCHOR, _MC28_ANCHOR + _MC28_ADD)
assert _CMM_MC28_TMPL.replace(_MC28_ADD, "") == _CMM_MC27_TMPL, "mc28 differs from mc27 by more than that one sentence"

_MC28_KW_SEE = dict(dec_stream="VIDEO", dec_adj="VISIBLE", oth_desc="audio track",
                    oth_head="Auditory", oth_verb="what you hear",
                    dec_head="Visual", dec_verb="what you see", dec_word="video",
                    rule4="An audible sound does not make a claim about its appearance true.")
_MC28_KW_HEAR = dict(dec_stream="AUDIO", dec_adj="AUDIBLE", oth_desc="video",
                     oth_head="Visual", oth_verb="what you see",
                     dec_head="Auditory", dec_verb="what you hear", dec_word="audio",
                     rule4="A visible object does not make a claim about its sound true.")
assert _CMM_MC27_TMPL.format(**_MC28_KW_SEE) == _CMM_MC27_SEE, "mc28/see slots do not come from the mc27 template"
assert _CMM_MC27_TMPL.format(**_MC28_KW_HEAR) == _CMM_MC27_HEAR, "mc28/hear slots do not come from the mc27 template"

_CMM_MC28_SEE = _CMM_MC28_TMPL.format(dec_adj_lc="visible", **_MC28_KW_SEE)
_CMM_MC28_HEAR = _CMM_MC28_TMPL.format(dec_adj_lc="audible", **_MC28_KW_HEAR)

_CMM_MC28 = _CMM_MC2

_MODAL_PROMPTS = {
    ("cmm", "mc27"): {"video": _CMM_MC27_SEE, "audio": _CMM_MC27_HEAR},
    ("cmm", "mc28"): {"video": _CMM_MC28_SEE, "audio": _CMM_MC28_HEAR},
}

JUDGE_PROMPTS = {
    "cmm":      {"v0": _CMM_V0,      "fs6": _CMM_FS6,
                 "mc": _CMM_MC,      "mc6": _CMM_MC6,
                 "mc2": _CMM_MC2,    "mc3": _CMM_MC3,
                 "mc26": _CMM_MC26,  "mc27": _CMM_MC27,
                 "mc28": _CMM_MC28},
    "phd":      {"v0": _PHD_V0,      "fs6": _PHD_FS6},
    "pubmedqa": {"v0": _PUBMEDQA_V0, "fs6": _PUBMEDQA_FS6},
}

#   "\nCRITICAL RULE: Do NOT repeat the context. You MUST use bullet points to list the
#    objective visual facts.\nVisual Checklist:\n- "
#
_MC_S1 = ("\nCRITICAL RULE: answer with the two checklists, the 'Queried item' line and the "
          "'Judgment' line, nothing else. Use short bullet points. Put the {first_word} list first "
          "and the {second_word} list second — the {second_word} list is the one this question is "
          "about, so it must be the last evidence before your judgment.\n"
          "{first_head}:\n- ")

_MC2_S1 = ("\nCRITICAL RULE: answer with the two checklists and the 'Asked about' line, nothing "
           "else. Use short bullet points. Put the {first_word} list first and the {second_word} "
           "list second — the {second_word} list is the one this question is about, so it is your "
           "closing evidence, and the 'Asked about' line ends your answer.\n"
           "{first_head}:\n- ")

_MC3_S1 = ("\nCRITICAL RULE: answer with one checklist and the 'Asked about' line, nothing else. "
           "Use short bullet points. The {first_word} list is the one this question is about, so it "
           "is your only evidence, and the 'Asked about' line ends your answer.\n"
           "{first_head}:\n- ")

_S1_TEMPLATES = {"mc": _MC_S1, "mc6": _MC_S1, "mc2": _MC2_S1, "mc3": _MC3_S1,
                 "mc26": _MC2_S1, "mc27": _MC2_S1, "mc28": _MC2_S1}


def cmm_stage1_heads(question: str, version: str = "mc"):
    """cmm question -> (stage-1 prefix, header used for final assembly).

        Returns (None, None) when the question does not match a known template.

        **The queried stream's checklist is placed second**, because that position
        is closest to the final `\\boxed{` marker: a "did you see" question gets
        audio first then visual, and "did you hear" gets visual first then audio.

        Why the order must flip: originally both question types received the same
        prefix (always headed by `Visual Checklist:`), so on "hear" questions the
        queried stream happened to land second (correct) while on "see" questions
        the second slot held the audio checklist, which should not be deciding.
        Measurements on the full cmm set showed this was the single most expensive
        defect, and its direction matches this fix.

        The returned header must be byte-identical to the end of the prefix --
        final assembly splices the header back into the full chain, and any
        mismatch adds or drops a heading. An import-time assertion enforces this.

        `version` only switches the prefix template (mc/mc6 use _MC_S1, mc2 uses
        _MC2_S1); it never changes the ordering logic. The default "mc" keeps old
        callers (which pass no second argument) byte-for-byte unchanged.
    """
    stream = cmm_stream_of(question)
    if stream is None:
        return None, None
    if version == "mc3":
        first, fw, sw = (("Visual Checklist", "visible", "visible")
                         if stream == "video" else
                         ("Auditory Checklist", "audible", "audible"))
    elif stream == "video":
        first, fw, sw = "Auditory Checklist", "audible", "visible"
    else:
        first, fw, sw = "Visual Checklist", "visible", "audible"
    tmpl = _S1_TEMPLATES.get(version, _MC_S1)
    return (tmpl.format(first_head=first, first_word=fw, second_word=sw),
            f"{first}:\n- ")


def cmm_target_of(question: str):
    """cmm question -> the queried item (e.g. 'bird chirping'), or None.

        Stage 2 names this item explicitly right before the verdict; see `_MC_S2`.
    """
    m = _re.search(r"did you (?:see|hear)\s+(.*?)\s+in the\s+(?:video|audio)",
                   question or "", _re.I)
    return m.group(1).strip() if m else None


#
_MC_S2 = ("\n\nQuestion: {question}\n"
          "The queried item is \"{target}\" and the deciding stream is the {stream} stream — the "
          "other stream must not decide this. Read your {deciding} Checklist above: if it records "
          "the queried item as being in the {stream}, the answer is Yes; if it does not, the answer "
          "is No. Judgment: \\boxed{{")
_MC_S2_NOTGT = ("\n\nQuestion: {question}\n"
                "This question asks about the {stream} stream ONLY — the other stream must not "
                "decide it. Read your {deciding} Checklist above: if it records the queried item "
                "as being in the {stream}, the answer is Yes; if it does not, the answer is No. "
                "Judgment: \\boxed{{")

_MC2_S2 = ("\n\nQuestion: {question}\n"
           "The queried item is \"{target}\" and the deciding stream is the {stream} stream — the "
           "other stream must not decide this. Read your {deciding} Checklist above: if it says "
           "the queried item IS in the {stream}, the answer is Yes; if it says the item is NOT in "
           "the {stream}, or never mentions it at all, the answer is No. Judgment: \\boxed{{")
_MC2_S2_NOTGT = ("\n\nQuestion: {question}\n"
                 "This question asks about the {stream} stream ONLY — the other stream must not "
                 "decide it. Read your {deciding} Checklist above: if it says the queried item IS "
                 "in the {stream}, the answer is Yes; if it says the item is NOT in the {stream}, "
                 "or never mentions it at all, the answer is No. Judgment: \\boxed{{")

_STAGE_SCAFFOLD = {
    ("cmm", "mc"):  {"s1_suffix": _MC_S1.format(first_head="Visual Checklist",
                                                first_word="visible", second_word="audible"),
                     "header": "Visual Checklist:\n- ", "s2_template": _MC_S2,
                     "s1_dynamic": True},
    ("cmm", "mc6"): {"s1_suffix": _MC_S1.format(first_head="Visual Checklist",
                                                first_word="visible", second_word="audible"),
                     "header": "Visual Checklist:\n- ", "s2_template": _MC_S2,
                     "s1_dynamic": True},
    ("cmm", "mc2"): {"s1_suffix": _MC2_S1.format(first_head="Visual Checklist",
                                                 first_word="visible", second_word="audible"),
                     "header": "Visual Checklist:\n- ", "s2_template": _MC2_S2,
                     "s2_template_notgt": _MC2_S2_NOTGT,
                     "s1_dynamic": True},
    ("cmm", "mc26"): {"s1_suffix": _MC2_S1.format(first_head="Visual Checklist",
                                                  first_word="visible", second_word="audible"),
                      "header": "Visual Checklist:\n- ", "s2_template": _MC2_S2,
                      "s2_template_notgt": _MC2_S2_NOTGT,
                      "s1_dynamic": True},
    ("cmm", "mc27"): {"s1_suffix": _MC2_S1.format(first_head="Visual Checklist",
                                                  first_word="visible", second_word="audible"),
                      "header": "Visual Checklist:\n- ", "s2_template": _MC2_S2,
                      "s2_template_notgt": _MC2_S2_NOTGT,
                      "s1_dynamic": True},
    ("cmm", "mc28"): {"s1_suffix": _MC2_S1.format(first_head="Visual Checklist",
                                                  first_word="visible", second_word="audible"),
                      "header": "Visual Checklist:\n- ", "s2_template": _MC2_S2,
                      "s2_template_notgt": _MC2_S2_NOTGT,
                      "s1_dynamic": True},
    ("cmm", "mc3"): {"s1_suffix": _MC3_S1.format(first_head="Auditory Checklist",
                                                 first_word="audible", second_word="audible"),
                     "header": "Auditory Checklist:\n- ", "s2_template": _MC2_S2,
                     "s2_template_notgt": _MC2_S2_NOTGT,
                     "s1_dynamic": True},
}


def cmm_stream_of(question: str):
    """cmm question -> the queried stream ('video' / 'audio'), or None.

        Only the two cmm templates are recognised, with no exceptions across the
        full set. When recognition fails the caller must fall back to the original
        scaffold: leaving the prompt unchanged is always preferable to guessing the
        wrong modality.
    """
    q = (question or "").lower()
    if "did you see" in q:
        return "video"
    if "did you hear" in q:
        return "audio"
    return None


def get_stage1_scaffold(dataset_type: str, version: str = "v0"):
    """(dataset_type, version) -> scaffold override dict, or None.

        None means "use the trunk's original hardcoded scaffold". `dataset_type` is
        matched as a substring, the same convention `get_judge_prompt` uses.
    """
    d = (dataset_type or "").lower()
    for (key, ver), scaf in _STAGE_SCAFFOLD.items():
        if key in d and ver == version:
            return scaf
    return None

_LABEL_SEQ = ["yes", "no", "yes", "no", "yes", "no"]

_FS_PAIRS = [("cmm", "v0", "fs6"), ("phd", "v0", "fs6"), ("pubmedqa", "v0", "fs6"),
             ("cmm", "mc", "mc6"), ("cmm", "mc2", "mc26")]


def _exemplar_labels(prompt: str):
    """Extract the verdict label of each in-context example, split on [E?].

        Three spellings are accepted, in priority order:
          1. `[E1 . Yes]` -- the label lives in the example heading. mc26 uses this:
             the mc2 contract has no verdict line, so an example body containing
             "Yes." would demonstrate an out-of-contract format, whereas the
             heading sits before the stage-1 prefix and cannot be copied.
          2. `\\boxed{Yes}` -- the fs6 / mc6 spelling (their contracts do contain a
             Judgment line).
          3. Two leading spaces plus `Yes.` -- an early spelling, kept for
             compatibility.
    """
    labs = []
    for blk in _re.split(r"\n\n(?=\[E)", prompt):
        if not blk.lstrip().startswith("[E"):
            continue
        m = (_re.match(r"\[E\d+\s*·\s*(Yes|No)\]", blk.lstrip())
             or _re.search(r"\\boxed\{(Yes|No)\}", blk)
             or _re.search(r"^\s{2}(Yes|No)\.", blk, _re.M))
        labs.append(m.group(1).lower() if m else "??")
    return labs


for _ds, _base, _fs in _FS_PAIRS:
    _v = JUDGE_PROMPTS[_ds]
    assert _v[_fs].startswith(_v[_base]), f"{_ds}: {_fs} is not a prefix extension of {_base}"
    assert _v[_fs].count("[E") == 6, f"{_ds}/{_fs}: example count is not 6 ({_v[_fs].count('[E')}）"
    _labs = _exemplar_labels(_v[_fs])
    assert _labs == _LABEL_SEQ, f"{_ds}/{_fs}: example label sequence must be {_LABEL_SEQ}, got {_labs}"

assert "Judgment" not in _CMM_MC26 and "\\boxed" not in _CMM_MC26, \
    "mc26 examples reintroduce Judgment/\\boxed, silently undoing mc2 change 1"
_MODS = [("see" if "Did you see" in _b else "hear")
         for _b in _re.split(r"\n\n(?=\[E)", _CMM_MC26) if _b.lstrip().startswith("[E")]
_LABS26 = _exemplar_labels(_CMM_MC26)
assert _MODS.count("see") == 3 and _MODS.count("hear") == 3, f"mc26 modality split is not 3:3（{_MODS}）"
for _m in ("see", "hear"):
    _l = {lab for mod, lab in zip(_MODS, _LABS26) if mod == _m}
    assert _l == {"yes", "no"}, f"mc26 {_m} examples are single-sided ({_l})-- this would teach a spurious modality-label rule"
del _MODS, _LABS26, _m, _l

for _ds, _vers in JUDGE_PROMPTS.items():
    for _name in ("v0", "mc", "mc2", "mc3"):
        if _name in _vers:
            assert _vers[_name].count("[E") == 0, f"{_ds}/{_name}: must not contain examples"

assert "ONLY on what is audible" not in JUDGE_PROMPTS["cmm"]["mc"], \
    "cmm/mc still carries the v0 audio-only rule, which is the bug being fixed"
assert "ONLY on what is audible" not in JUDGE_PROMPTS["cmm"]["mc2"], \
    "cmm/mc2 still carries the v0 audio-only rule"
assert "ONLY on what is audible" not in JUDGE_PROMPTS["cmm"]["mc3"], \
    "cmm/mc3 still carries the v0 audio-only rule"
assert "ONLY on what is audible" in JUDGE_PROMPTS["cmm"]["v0"], \
    "cmm/v0 was modified -- v0 must stay byte-identical to the trunk hardcoded prompt"

for _ds in ("cmm", "phd", "pubmedqa"):
    for _name in JUDGE_PROMPTS[_ds]:
        _want = (_ds == "cmm" and _name in ("mc", "mc6", "mc2", "mc3", "mc26", "mc27", "mc28"))
        assert (get_stage1_scaffold(_ds, _name) is not None) == _want, \
            f"{_ds}/{_name}: scaffold override lookup is wrong"
del _ds, _base, _fs, _v, _labs, _vers, _name, _want

_M2, _M1 = JUDGE_PROMPTS["cmm"]["mc2"], JUDGE_PROMPTS["cmm"]["mc"]
assert "Judgment" not in _M2, "mc2 system prompt still contains Judgment, so change 1 did not take effect"
assert "Judgment" in _M1, "mc must contain Judgment; if the control arm changed the experiment is void"
assert "Judgment" not in _MC2_S1, "mc2 stage-1 prefix still contains Judgment; changing only the system prompt is a no-op"
assert "Judgment" in _MC_S1, "mc stage-1 prefix was modified, so the control arm is no longer the v1 version"
assert "\\boxed" not in _M2 and "\\boxed" not in _MC2_S1, "mc2 must not contain the \\boxed marker"
assert "Asked about" in _M2 and "Queried" not in _M2, "mc2 labels were not fully renamed"
assert "Asked about" in _MC2_S1 and "Queried" not in _MC2_S1, "mc2 prefix labels were not fully renamed"
for _t in (_MC2_S2, _MC2_S2_NOTGT):
    assert "records the queried item" not in _t, "mc2 stage 2 still uses the mention-based criterion"
    assert "IS in the" in _t and "NOT in the" in _t and "never mentions it" in _t, \
        "mc2 stage 2 is missing one of its three branches"
del _M2, _M1, _t

for _ver in ("mc", "mc6", "mc2", "mc26", "mc27", "mc28"):
    for _q, _want_first, _want_second in (
            ("Did you see clock in the video?",        "Auditory Checklist", "Visual Checklist"),
            ("Did you hear bird churping in the audio?", "Visual Checklist", "Auditory Checklist")):
        _sfx, _hdr = cmm_stage1_heads(_q, _ver)
        assert _sfx and _hdr, f"{_ver}: cmm_stage1_heads failed to parse the question：{_q}"
        assert _sfx.endswith(_hdr), f"{_ver}/{_q}: header does not match the end of the prefix"
        assert _hdr == f"{_want_first}:\n- ", f"{_ver}/{_q}: first checklist header should be {_want_first}"
        assert f"{_want_second}:" not in _sfx, f"{_ver}/{_q}: the queried stream's header must not appear in the prefix"
        _st = cmm_stream_of(_q)
        _tg = cmm_target_of(_q)
        assert _tg, f"{_q}: could not extract the queried item"
        _s2 = _STAGE_SCAFFOLD[("cmm", _ver)]["s2_template"]
        _tail = _s2.format(question=_q, stream=_st, target=_tg,
                           deciding=("Visual" if _st == "video" else "Auditory"))
        assert _tail.endswith("\\boxed{") and _tg in _tail, f"{_ver}/{_q}: stage-2 tail is assembled incorrectly"
assert cmm_stage1_heads("What is in this clip?") == (None, None)
assert cmm_stage1_heads("What is in this clip?", "mc2") == (None, None)
assert cmm_target_of("What is in this clip?") is None
assert cmm_stage1_heads("Did you see clock in the video?") == \
    cmm_stage1_heads("Did you see clock in the video?", "mc")
del _ver, _q, _want_first, _want_second, _sfx, _hdr, _st, _tg, _tail, _s2

assert cmm_stage1_heads("What is in this clip?", "mc3") == (None, None)
for _q, _deciding, _other in (
        ("Did you see clock in the video?",          "Visual Checklist", "Auditory Checklist"),
        ("Did you hear bird churping in the audio?", "Auditory Checklist", "Visual Checklist")):
    _sfx, _hdr = cmm_stage1_heads(_q, "mc3")
    assert _sfx.endswith(_hdr), f"mc3/{_q}: header does not match the end of the prefix"
    assert _hdr == f"{_deciding}:\n- ", f"mc3/{_q}: injected header should be {_deciding}"
    assert _other not in _sfx, f"mc3/{_q}: non-deciding header {_other} leaked into the prefix"
    assert "one checklist" in _sfx, f"mc3/{_q}: prefix does not state that only one checklist is written"
del _q, _deciding, _other, _sfx, _hdr

_M3, _M2 = JUDGE_PROMPTS["cmm"]["mc3"], JUDGE_PROMPTS["cmm"]["mc2"]
assert "Fill in BOTH checklists" in _M2, "the mc2 control arm changed, invalidating the mc3 A/B comparison"
assert "BOTH checklists" not in _M3, "mc3 still says to fill both checklists, so the change did not take effect"
assert "Exactly ONE checklist" in _M3, "mc3 does not state that there is exactly one checklist"
assert ("A visible object does not make a claim about its sound true" in _M3
        and "A visible object does not make a claim about its sound true" in _M2), \
    "mc3 dropped the anti-copy rule, so it is no longer a single-variable control"
assert "Judgment" not in _M3 and "Judgment" not in _MC3_S1, "Judgment reappeared in mc3"
assert "Asked about" in _M3 and "Queried" not in _M3, "mc3 labels did not follow the mc2 rename"
assert "\\boxed" not in _M3 and "\\boxed" not in _MC3_S1, "mc3 must not contain the \\boxed marker"
assert (_STAGE_SCAFFOLD[("cmm", "mc3")]["s2_template"]
        is _STAGE_SCAFFOLD[("cmm", "mc2")]["s2_template"]), "mc3 stage 2 is not the mc2 one"
del _M3, _M2


assert _S1_TEMPLATES["mc26"] is _S1_TEMPLATES["mc2"], "mc26 stage-1 template is not the mc2 one"
for _k in ("s2_template", "s2_template_notgt"):
    assert (_STAGE_SCAFFOLD[("cmm", "mc26")][_k]
            is _STAGE_SCAFFOLD[("cmm", "mc2")][_k]), f"mc26 {_k} is not the mc2 one"
assert (_STAGE_SCAFFOLD[("cmm", "mc26")]["header"]
        == _STAGE_SCAFFOLD[("cmm", "mc2")]["header"]), "mc26 static header differs from mc2"
assert _STAGE_SCAFFOLD[("cmm", "mc26")]["s1_dynamic"] is True, "mc26 does not enable s1_dynamic"
_ADD = JUDGE_PROMPTS["cmm"]["mc26"][len(JUDGE_PROMPTS["cmm"]["mc2"]):]
assert _ADD.count("[E") == 6 and "Fill in BOTH checklists" not in _ADD, \
    "the mc26 delta is more than examples, so it is no longer a single-variable control"
del _k, _ADD

assert _S1_TEMPLATES["mc27"] is _S1_TEMPLATES["mc2"], "mc27 stage-1 template is not the mc2 one"
for _k in ("s2_template", "s2_template_notgt"):
    assert (_STAGE_SCAFFOLD[("cmm", "mc27")][_k]
            is _STAGE_SCAFFOLD[("cmm", "mc2")][_k]), f"mc27 {_k} is not the mc2 one"
assert (_STAGE_SCAFFOLD[("cmm", "mc27")]["header"]
        == _STAGE_SCAFFOLD[("cmm", "mc2")]["header"]), "mc27 static header differs from mc2"
assert _STAGE_SCAFFOLD[("cmm", "mc27")]["s1_dynamic"] is True, "mc27 does not enable s1_dynamic"
assert JUDGE_PROMPTS["cmm"]["mc27"] is JUDGE_PROMPTS["cmm"]["mc2"], \
    "mc27 static fallback is not the mc2 one, so an unparsed modality silently changes the criterion"
_M2 = JUDGE_PROMPTS["cmm"]["mc2"]
for _mp in (_CMM_MC27_SEE, _CMM_MC27_HEAR):
    for _n, _rule in (("2", "2. A sound never goes under a Visual heading and an appearance "
                            "never goes under an Auditory heading, whichever heading comes "
                            "first.\n"),
                      ("3", "3. Write down what you actually observe in each stream. If the "
                            "thing the question asks about is not there, write that it is not "
                            "there — say it in plain words rather than leaving it out.\n")):
        assert _rule in _mp and _rule in _M2, f"mc27 rule {_n} is no longer byte-identical to mc2"
    assert "Fill in BOTH checklists" in _mp, "mc27 dropped the other checklist, collapsing into the dead mc3 route"
    assert "\\boxed" not in _mp and "Judgment" not in _mp, "mc27 reintroduces markers that mc2 removed"
    assert "Asked about" in _mp and "Queried" not in _mp, "mc27 labels did not follow the mc2 rename"
    assert _mp.count("STRICT AUDIT RULES") == 1 and "5. Report what the evidence shows" in _mp, \
        "mc27 rule count differs from mc2"
assert "Did you see" not in _CMM_MC27_SEE and "Did you hear" not in _CMM_MC27_SEE, \
    "mc27/see still branches on the question text; the split is incomplete"
assert "Did you see" not in _CMM_MC27_HEAR and "Did you hear" not in _CMM_MC27_HEAR, \
    "mc27/hear still branches on the question text; the split is incomplete"
assert "VIDEO stream is the DECIDING" in _CMM_MC27_SEE, "mc27/see does not name the visual stream as deciding"
assert "AUDIO stream is the DECIDING" in _CMM_MC27_HEAR, "mc27/hear does not name the audio stream as deciding"
assert _CMM_MC27_SEE != _CMM_MC27_HEAR, "the two mc27 branches are identical, so the split is a no-op"
del _k, _mp, _n, _rule, _M2

assert _S1_TEMPLATES["mc28"] is _S1_TEMPLATES["mc2"], "mc28 stage-1 template is not the mc2 one"
for _k in ("s2_template", "s2_template_notgt"):
    assert (_STAGE_SCAFFOLD[("cmm", "mc28")][_k]
            is _STAGE_SCAFFOLD[("cmm", "mc27")][_k]), f"mc28 {_k} is not the mc27/mc2 one"
assert (_STAGE_SCAFFOLD[("cmm", "mc28")]["s1_suffix"]
        == _STAGE_SCAFFOLD[("cmm", "mc27")]["s1_suffix"]), "mc28 modified the stage-1 prefix, introducing a second independent variable"
assert (_STAGE_SCAFFOLD[("cmm", "mc28")]["header"]
        == _STAGE_SCAFFOLD[("cmm", "mc27")]["header"]), "mc28 static header differs from mc27"
assert _STAGE_SCAFFOLD[("cmm", "mc28")]["s1_dynamic"] is True, "mc28 does not enable s1_dynamic"
assert JUDGE_PROMPTS["cmm"]["mc28"] is JUDGE_PROMPTS["cmm"]["mc2"], \
    "mc28 static fallback is not the mc2 one, so an unparsed modality silently changes the criterion"
assert "FIRST bullet must be the thing the question names" in _CMM_MC28_SEE
assert "FIRST bullet must be the thing the question names" in _CMM_MC28_HEAR
assert "whether that thing is visible or not" in _CMM_MC28_SEE, "mc28/see does not use the adjective 'visible'"
assert "whether that thing is audible or not" in _CMM_MC28_HEAR, "mc28/hear does not use the adjective 'audible'"
assert "audible" not in _CMM_MC28_SEE.split("STRICT OUTPUT FORMAT")[1].split("STRICT AUDIT")[0], \
    "'audible' leaked into the mc28/see format block -- modality crosstalk"
for _mp, _dec in ((_CMM_MC28_SEE, "Visual"), (_CMM_MC28_HEAR, "Auditory")):
    _fmt = _mp.split("STRICT OUTPUT FORMAT")[1].split("STRICT AUDIT")[0]
    assert _fmt.index("FIRST bullet") > _fmt.index(f'"{_dec} Checklist:"'), \
        "the mc28 position rule landed on the non-deciding stream"
    assert _fmt.count("FIRST bullet") == 1, "the mc28 position rule appears more than once"
for _mp, _m27 in ((_CMM_MC28_SEE, _CMM_MC27_SEE), (_CMM_MC28_HEAR, _CMM_MC27_HEAR)):
    assert (_mp.split("STRICT AUDIT RULES")[1]
            == _m27.split("STRICT AUDIT RULES")[1]), "mc28 modified the rules block, so it is no longer single-variable"
    assert "Fill in BOTH checklists" in _mp, "mc28 dropped the other checklist, collapsing into the dead mc3 route"
    assert "\\boxed" not in _mp and "Judgment" not in _mp, "mc28 reintroduces markers that mc2 removed"
    assert "Asked about" in _mp and "Queried" not in _mp, "mc28 labels did not follow the mc2 rename"
    assert "Did you see" not in _mp and "Did you hear" not in _mp, "mc28 still branches on the question text"
assert _CMM_MC28_SEE != _CMM_MC28_HEAR, "the two mc28 branches are identical, so the split is a no-op"
assert len(_CMM_MC28_SEE) > len(_CMM_MC27_SEE) and len(_CMM_MC28_SEE) < len(_CMM_MC2), \
    "mc28 must be one sentence longer than mc27 and still shorter than mc2"
del _k, _mp, _m27, _dec, _fmt


def get_judge_prompt(dataset_type: str, version: str = "v0", question: str = None):
    """(dataset_type, version[, question]) -> system prompt, or None.

        None means the dataset is outside this module's scope. `dataset_type` is
        matched as a substring, matching `get_system_prompt` in the trunk. An
        unknown version falls back to v0 (the trunk's original behaviour) and never
        raises.

        `question` only matters for versions that branch on the question text
        (currently cmm/mc27, registered in `_MODAL_PROMPTS`):
          - question omitted -> the old path; every version is byte-identical, so
            existing callers are unaffected.
          - question given but the version is unregistered -> also byte-identical.
          - question given and the version is registered, but `cmm_stream_of`
            cannot parse the modality -> fall back to that version's static text
            (for mc27, the static text is the mc2 original). Leaving the prompt
            unchanged is always preferable to guessing the wrong modality.
    """
    d = (dataset_type or "").lower()
    for key, versions in JUDGE_PROMPTS.items():
        if key in d:
            modal = _MODAL_PROMPTS.get((key, version)) if question is not None else None
            if modal:
                st = cmm_stream_of(question)
                if st in modal:
                    return modal[st]
            return versions.get(version) or versions["v0"]
    return None
