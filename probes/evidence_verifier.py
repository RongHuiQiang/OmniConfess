"""
evidence_verifier.py — Steps 4-5: per-span visual evidence verification and edit decision.

Pipeline position:
  localizer_v2.py  →  [this module]  →  generator (run_halo_cu121.py)

Input:  LocalizeResult with candidates having verify_status="NEEDS_VERIFICATION"
Output: same LocalizeResult with verify_status, verify_raw, is_refusal, decision filled in

No torch/transformers dependency here — the GPU model is injected via _run_generate_fn.

v3 changes (4-status broad_verify):
  Single broad verify prompt returning one of four statuses:
    SUPPORTED               — image clearly supports the assertion
    CONTRADICTED            — image contains clear counter-evidence
    UNSUPPORTED_BY_EVIDENCE — assertion makes a specific claim but evidence is insufficient
    INSUFFICIENT_TO_DETERMINE — verifier cannot reliably judge

  Key principles:
    - "no clear counter-evidence" ≠ SUPPORTED (must not collapse to SUPPORTED)
    - "insufficient support" ≠ must-edit (UNSUPPORTED_BY_EVIDENCE, let decision layer choose)
    - CONTRADICTED is the strongest edit signal; also triggers B1 step1 observation
    - No candidates are filtered out here — all proceed to regeneration

  Decision mapping:
    CONTRADICTED              → REVISION_CANDIDATE  (strong signal)
    UNSUPPORTED_BY_EVIDENCE   → REVISION_CANDIDATE  (weak signal, still editable)
    SUPPORTED                 → PRESERVE
    INSUFFICIENT_TO_DETERMINE → REVISION_CANDIDATE  (pass to accept layer)

Legacy B1/B2/B3 prompts are retained below for reference/fallback but are no
longer called from verify_candidates().
"""
from __future__ import annotations

import re
from typing import Callable

from localizer_contract import Candidate, LocalizeResult


# ---------------------------------------------------------------------------
# Refusal keyword lists (copied from freeform_repair._has_refusal, not imported
# to avoid circular dependency)
# ---------------------------------------------------------------------------

_REFUSAL_PHRASES = (
    "there is no",
    "there are no",
    "cannot",
    "can't",
    "don't",
    "doesn't",
    "isn't",
    "aren't",
    "wasn't",
    "weren't",
)

_REFUSAL_WORDS = {"not", "no", "never", "unable", "unknown", "unclear", "undetermined"}

# Four-status vocabulary for v3 broad_verify
_V3_STATUSES = (
    "CONTRADICTED",
    "UNSUPPORTED_BY_EVIDENCE",
    "INSUFFICIENT_TO_DETERMINE",
    "SUPPORTED",
)


def _has_refusal_span(span_text: str) -> bool:
    """Return True if span_text contains refusal/negation semantics."""
    t = span_text.lower()
    for phrase in _REFUSAL_PHRASES:
        if phrase in t:
            return True
    words = set(t.split())
    return bool(words & _REFUSAL_WORDS)


# ---------------------------------------------------------------------------
# v3: Single broad_verify — four-status, no filtering
# ---------------------------------------------------------------------------

def build_broad_verify_prompt(candidate: Candidate, item: dict, orig_text: str) -> str:
    """Single-step prompt returning one of four statuses.

    Design choices:
    - CONTRADICTED requires pointing to specific visual counter-evidence.
    - UNSUPPORTED_BY_EVIDENCE means the assertion is specific but evidence is absent/unclear.
    - SUPPORTED requires the image to visibly confirm the claim.
    - INSUFFICIENT_TO_DETERMINE is the catch-all when none of the above are clear.
    - Default bias: when in doubt, lean UNSUPPORTED_BY_EVIDENCE not SUPPORTED.
    """
    question = candidate.question or item.get("question", "")
    claim = candidate.claim or candidate.span_text

    return (
        f'Look at the image carefully.\n\n'
        f'Question context: "{question}"\n\n'
        f'Evaluate this specific claim from the model\'s answer: "{claim}"\n\n'
        f'Choose exactly ONE status:\n'
        f'- SUPPORTED: the image visibly and clearly confirms this claim\n'
        f'- CONTRADICTED: the image contains specific visual evidence that directly '
        f'contradicts this claim (you can point to what you see that proves it wrong)\n'
        f'- UNSUPPORTED_BY_EVIDENCE: the claim makes a specific assertion but the image '
        f'does not provide sufficient evidence to confirm it (absence of support, not '
        f'active contradiction)\n'
        f'- INSUFFICIENT_TO_DETERMINE: you cannot reliably assess this claim from the '
        f'image alone\n\n'
        f'Important:\n'
        f'- "No clear counter-evidence" does NOT mean SUPPORTED\n'
        f'- CONTRADICTED requires you to point to specific visible evidence\n'
        f'- When uncertain between SUPPORTED and UNSUPPORTED_BY_EVIDENCE, choose '
        f'UNSUPPORTED_BY_EVIDENCE\n\n'
        f'Reply with:\n'
        f'STATUS: <one of the four statuses>\n'
        f'Reason: <one sentence citing what you see>'
    )


def parse_broad_verify_status(response: str) -> str:
    """Extract 4-status verdict from a broad_verify response.

    Priority order: CONTRADICTED > UNSUPPORTED_BY_EVIDENCE > SUPPORTED >
    INSUFFICIENT_TO_DETERMINE. Falls back to INSUFFICIENT_TO_DETERMINE.
    """
    upper = response.strip().upper()

    # Look for explicit STATUS: prefix first
    m = re.search(r'STATUS\s*:\s*(\S+)', upper)
    if m:
        token = m.group(1).rstrip('.,;')
        for status in _V3_STATUSES:
            if token.startswith(status):
                return status

    # Fallback: scan full text for any of the four tokens (most-specific first)
    for status in _V3_STATUSES:
        if status in upper:
            return status

    return "INSUFFICIENT_TO_DETERMINE"


def decide_from_broad_verify(verify_status: str) -> str:
    """Map 4-status → decision.

    SUPPORTED        → PRESERVE
    CONTRADICTED     → REVISION_CANDIDATE  (strong signal)
    UNSUPPORTED_BY_EVIDENCE → REVISION_CANDIDATE  (weak signal)
    INSUFFICIENT_TO_DETERMINE → REVISION_CANDIDATE  (pass to accept layer)
    """
    if verify_status == "SUPPORTED":
        return "PRESERVE"
    return "REVISION_CANDIDATE"


# ---------------------------------------------------------------------------
# B1: Two-step verification prompts (legacy, kept for reference)
# ---------------------------------------------------------------------------

def build_step1_observation_prompt(candidate: Candidate, item: dict) -> str:
    """Step 1: Ask the model to describe only what it directly sees.

    Returns a prompt that asks for a one-sentence factual observation about
    the object or attribute mentioned in the candidate's claim, without
    asking the model to evaluate correctness yet.
    """
    question = candidate.question or item.get("question", "")
    claim = candidate.claim or candidate.span_text
    return (
        f'Look at the image carefully.\n\n'
        f'Question context: "{question}"\n\n'
        f'The answer being evaluated mentions: "{claim}"\n\n'
        f'What do you actually see in the image regarding the subject of this claim?\n'
        f'Describe only what is directly visible, without inferring or evaluating.\n'
        f'Answer in one sentence.'
    )


def build_step2_contradiction_prompt(
    candidate: Candidate, item: dict, step1_observation: str
) -> str:
    """Step 2: Given the observation, ask only whether the image contradicts the claim.

    Uses a strict YES/NO format with VERDICT: prefix so parsing is unambiguous.
    Bias is toward NO — YES requires specific pointing-to evidence.
    """
    claim = candidate.claim or candidate.span_text
    return (
        f'Original claim: "{claim}"\n'
        f'Your visual observation: {step1_observation}\n\n'
        f'Does the image directly CONTRADICT the claim?\n'
        f'Answer YES only if you can point to specific visual evidence that '
        f'contradicts it.\n'
        f'Answer NO if the image supports it OR if you cannot determine from '
        f'the image.\n\n'
        f'Format:\n'
        f'VERDICT: YES/NO\n'
        f'Reason: [one sentence citing the specific visual evidence]'
    )


def parse_step2_verdict(response: str) -> str:
    """Return 'YES' or 'NO' from a step-2 response.

    Looks for 'VERDICT: YES' / 'VERDICT: NO' first; falls back to bare YES/NO
    at the start of the response; defaults to NO (conservative).
    """
    upper = response.strip().upper()
    # Explicit VERDICT: token
    m = re.search(r'VERDICT\s*:\s*(YES|NO)', upper)
    if m:
        return m.group(1)
    # Bare first word
    first = upper.split()[0] if upper.split() else ""
    if first in ("YES", "YES.", "YES,"):
        return "YES"
    if first in ("NO", "NO.", "NO,"):
        return "NO"
    # Default: do not flag as contradiction without explicit evidence
    return "NO"


# ---------------------------------------------------------------------------
# B2: Answerability classification for negation_refusal spans
# ---------------------------------------------------------------------------

_ANSWERABILITY_CLASSES = (
    "OBJECT_ABSENT",
    "OBJECT_PRESENT_DETAIL_NOT_VISIBLE",
    "EVIDENCE_AMBIGUOUS",
    "OBJECT_PRESENT_DETAIL_VISIBLE",
)


def build_answerability_prompt(candidate: Candidate, item: dict, orig_text: str) -> str:
    """Classify whether/how the image can answer the question behind the refusal span."""
    question = candidate.question or item.get("question", "")
    return (
        f'Given this image and the question: "{question}"\n\n'
        f'The model\'s answer was: "{orig_text}"\n'
        f'This part of the answer is a refusal/negation: "{candidate.span_text}"\n\n'
        f'Classify the image\'s relationship to this question into exactly ONE category:\n'
        f'- OBJECT_ABSENT: the object/subject the question asks about is simply not '
        f'present in the image at all\n'
        f'- OBJECT_PRESENT_DETAIL_NOT_VISIBLE: the object exists in the image but the '
        f'specific attribute or detail asked about cannot be confirmed\n'
        f'- EVIDENCE_AMBIGUOUS: it is unclear whether the object is present or the '
        f'image is too ambiguous to classify\n'
        f'- OBJECT_PRESENT_DETAIL_VISIBLE: the object exists AND the specific detail '
        f'asked about is clearly visible (meaning the refusal is wrong)\n\n'
        f'Answer with just the category name, then one sentence explaining why.'
    )


def parse_answerability(response: str) -> str:
    """Extract the answerability class from a model response.

    Returns one of the four _ANSWERABILITY_CLASSES, or 'EVIDENCE_AMBIGUOUS'
    as the conservative fallback.
    """
    upper = response.strip().upper()
    # Check most specific first to avoid partial matches
    for cls in (
        "OBJECT_PRESENT_DETAIL_VISIBLE",
        "OBJECT_PRESENT_DETAIL_NOT_VISIBLE",
        "OBJECT_ABSENT",
        "EVIDENCE_AMBIGUOUS",
    ):
        if cls in upper:
            return cls
    return "EVIDENCE_AMBIGUOUS"


# ---------------------------------------------------------------------------
# B3: Reverse consistency check
# ---------------------------------------------------------------------------

def build_reverse_check_prompt(candidate: Candidate, step1_observation: str) -> str:
    """Ask the model to specifically look for evidence supporting the negated claim.

    If the model found the claim unsupported, we ask it to look for evidence
    in the opposite direction.  Contradiction is only reliable when both passes
    agree.
    """
    claim = candidate.claim or candidate.span_text
    # Build a simple negation description by prefixing with "NOT"
    return (
        f'You previously evaluated that the claim "{claim}" is not supported '
        f'by the image.\n\n'
        f'Now specifically look for: Is there any visual evidence in the image '
        f'that directly supports or shows what the claim describes?\n\n'
        f'Your earlier observation was: {step1_observation}\n\n'
        f'Is there any supporting evidence visible?\n'
        f'Answer: YES or NO, with one specific observation.'
    )


def parse_reverse_check(response: str) -> str:
    """Return 'YES' (found supporting evidence) or 'NO' (no supporting evidence).

    Default is NO — i.e., we assume no supporting evidence unless explicitly found,
    which means the original UNSUPPORTED verdict stands.
    """
    upper = response.strip().upper()
    first = upper.split()[0] if upper.split() else ""
    # Explicit YES at start or in a YES/NO pattern
    if first in ("YES", "YES.", "YES,"):
        return "YES"
    m = re.search(r'\b(YES|NO)\b', upper)
    if m:
        return m.group(1)
    return "NO"


# ---------------------------------------------------------------------------
# Original single-step verify prompt (kept for negation_refusal fallback path
# when answerability == OBJECT_ABSENT, where two-step is not useful)
# ---------------------------------------------------------------------------

def build_verify_prompt(candidate: Candidate, item: dict, orig_text: str) -> str:
    """Single-step prompt — used only for the negation_refusal/OBJECT_ABSENT path."""
    question = candidate.question or item.get("question", "")

    if candidate.span_kind == "negation_refusal":
        return (
            f'Given this image and the question: "{question}"\n\n'
            f'The model answered: "{orig_text}"\n\n'
            f'This part of the answer suggests the information is not visible/available: '
            f'"{candidate.span_text}"\n\n'
            f"Does the image actually provide enough information to answer the question "
            f"directly (making this refusal potentially wrong)?\n\n"
            f"Answer with one of:\n"
            f"- YES_IMAGE_ANSWERS: the image clearly provides the answer\n"
            f"- NO_REFUSAL_VALID: the image truly lacks this information\n"
            f"- INSUFFICIENT: cannot determine from the image alone"
        )
    else:
        return (
            f'Given this image and the question: "{question}"\n\n'
            f'The model answered: "{orig_text}"\n\n'
            f'Evaluate this specific claim from the model\'s answer: "{candidate.claim}"\n\n'
            f"Look carefully at the image. Does the image DIRECTLY CONTRADICT this claim "
            f"(i.e., you can see something in the image that proves the claim is wrong)?\n\n"
            f"Answer with one of:\n"
            f"- UNSUPPORTED: the image CLEARLY AND DIRECTLY contradicts this claim "
            f"(describe what you see that proves it wrong)\n"
            f"- SUPPORTED: the image supports or is consistent with this claim\n"
            f"- INSUFFICIENT_EVIDENCE: the image neither clearly supports nor clearly "
            f"contradicts this claim\n\n"
            f"Default to SUPPORTED or INSUFFICIENT_EVIDENCE unless you can point to "
            f"specific visible evidence that contradicts the claim. "
            f"Then briefly explain in one sentence what you see."
        )


# ---------------------------------------------------------------------------
# Step 4b: parse model response (legacy single-step, kept for negation path)
# ---------------------------------------------------------------------------

def parse_verify_response(response: str, span_kind: str) -> tuple[str, str]:
    """Parse the raw model string into (verify_status, raw).

    For negation_refusal spans:
        YES_IMAGE_ANSWERS → UNSUPPORTED  (the refusal is wrong)
        NO_REFUSAL_VALID  → SUPPORTED    (the refusal is correct, preserve)
        INSUFFICIENT/other → INSUFFICIENT_EVIDENCE

    For all other span kinds:
        SUPPORTED / UNSUPPORTED / INSUFFICIENT_EVIDENCE → same
        fallback → INSUFFICIENT_EVIDENCE (conservative)
    """
    raw = response.strip()

    if span_kind == "negation_refusal":
        upper = raw.upper()
        if "YES_IMAGE_ANSWERS" in upper:
            return "UNSUPPORTED", raw
        elif "NO_REFUSAL_VALID" in upper:
            return "SUPPORTED", raw
        else:
            return "INSUFFICIENT_EVIDENCE", raw
    else:
        upper = raw.upper()
        if re.search(r'\bUNSUPPORTED\b', upper):
            return "UNSUPPORTED", raw
        if re.search(r'\bINSUFFICIENT_EVIDENCE\b', upper) or re.search(
            r'\bINSUFFICIENT\b', upper
        ):
            return "INSUFFICIENT_EVIDENCE", raw
        if re.search(r'\bSUPPORTED\b', upper):
            return "SUPPORTED", raw
        lower = raw.lower()
        if "not support" in lower or "contradict" in lower or "does not" in lower:
            return "UNSUPPORTED", raw
        if "cannot determine" in lower or "unclear" in lower or "insufficient" in lower:
            return "INSUFFICIENT_EVIDENCE", raw
        if "support" in lower:
            return "SUPPORTED", raw
        return "INSUFFICIENT_EVIDENCE", raw


# ---------------------------------------------------------------------------
# Step 5: edit decision (legacy — kept for reference, not called by v3 path)
# ---------------------------------------------------------------------------

def decide_from_verify(candidate: Candidate) -> str:
    """Map (verify_status, is_refusal) → decision string (legacy two-status path)."""
    vs = candidate.verify_status

    if candidate.is_refusal and vs == "SUPPORTED":
        return "PRESERVE"
    if vs == "SUPPORTED":
        return "PRESERVE"
    if vs == "UNSUPPORTED":
        return "REVISION_CANDIDATE"
    if vs == "INSUFFICIENT_EVIDENCE":
        return "SUSPICIOUS_BUT_UNVERIFIED"
    return "SUSPICIOUS_BUT_UNVERIFIED"


# ---------------------------------------------------------------------------
# Main entry point — v3: broad_verify, no candidate filtering
# ---------------------------------------------------------------------------

def verify_candidates(
    result: LocalizeResult,
    item: dict,
    _run_generate_fn: Callable[[str, int], str],
) -> LocalizeResult:
    """Steps 4-5: verify each candidate span against the image and decide.

    v3: uses a single broad_verify prompt returning one of four statuses.
    No candidates are filtered out — all remain in result.candidates regardless
    of verdict, letting the downstream accept layer make the final call.

    CONTRADICTED also triggers an additional B1 step1_observation pass to
    obtain a concrete visual observation, which _lv2_evidence_regen() uses to
    ground the correction prompt.

    Each Candidate is updated with:
        candidate.is_refusal        — set from span text
        candidate.verify_status     — SUPPORTED / CONTRADICTED /
                                      UNSUPPORTED_BY_EVIDENCE /
                                      INSUFFICIENT_TO_DETERMINE
        candidate.verify_raw        — raw model string (auditable)
        candidate.decision          — PRESERVE / REVISION_CANDIDATE
        candidate.step1_observation — visual observation string (for all
                                      non-SUPPORTED candidates, else None)
        candidate.answerability     — None (reserved; not used in v3 path)
    """
    orig_text = result.base_pred

    for candidate in result.candidates:
        candidate.is_refusal = _has_refusal_span(candidate.span_text)

        step1_obs: str | None = None
        answerability: str | None = None

        # --- v3 broad verify ---
        broad_prompt = build_broad_verify_prompt(candidate, item, orig_text)
        broad_raw = _run_generate_fn(broad_prompt, 100)
        verify_status = parse_broad_verify_status(broad_raw)
        verify_raw = broad_raw

        # For all non-SUPPORTED candidates, run B1 step1 to get a concrete
        # visual observation that will be injected into the regen prompt.
        # Previously this was CONTRADICTED-only; expanding to all non-SUPPORTED
        # ensures regen always has a visual anchor, not a fallback prompt.
        if verify_status != "SUPPORTED":
            step1_prompt = build_step1_observation_prompt(candidate, item)
            step1_obs = _run_generate_fn(step1_prompt, 80)
            verify_raw = f"[broad={broad_raw}] [step1={step1_obs}]"

        candidate.verify_status = verify_status
        candidate.verify_raw = verify_raw

        # Attach step1_observation and answerability gracefully
        try:
            candidate.step1_observation = step1_obs
        except (AttributeError, TypeError):
            pass
        try:
            candidate.answerability = answerability
        except (AttributeError, TypeError):
            pass

        # Step 5: decision mapping (v3)
        candidate.decision = decide_from_broad_verify(verify_status)

    return result
