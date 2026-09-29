#!/usr/bin/env python3
"""
FactProbe v5 — Modality-Aware Candidate Generation.

Key changes vs v4:
  - arm='audio_only' label is ONLY used for sample selection (delta_audio shim).
    It no longer determines which channel the verifier is told to use.
  - Candidate generation runs TWO separate interrogation passes per sample:
      * Visual pass: pixel_values only (audio_values=None) → visual correction candidate
      * Audio pass:  audio_values only (pixel_values=None)  → audio correction candidate
  - Verification uses diff_modality → channel_note:
      subject/location/object/visual-action dims → VISUAL channel
      sound/audio-action dims                    → AUDIO channel
      both                                       → combined (both channels)
  - step1_obs is NOT passed to the verifier (self-reference chain stays cut).
  - GT is ONLY used in offline oracle computation, never in generation or acceptance.

Goal: Expand Candidate Oracle beyond 28.54%.  Gate precision is secondary.

Usage:
  CUDA_VISIBLE_DEVICES=0 MMHALU_MODEL_ROOT=/tmp HF_HUB_OFFLINE=1 \\
  TRANSFORMERS_OFFLINE=1 PYTHONPATH=/dev/shm/site-packages \\
  LD_LIBRARY_PATH=/dev/shm/so_mirror \\
  /home/tsq/.local/bin/python3.13 scripts/run_avh_factprobe_v5.py \\
    --arm audio_only --limit 50
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

import torch

OUTER_ROOT = Path(__file__).resolve().parent.parent
INNER_ROOT = OUTER_ROOT / "OmniConfess"

assert (INNER_ROOT / "methods/shared/model_utils.py").exists(), \
    f"model_utils not found under {INNER_ROOT}"

sys.path.insert(0, str(INNER_ROOT))
sys.path.insert(0, str(OUTER_ROOT))

AVH_JSON  = OUTER_ROOT / "OmniHalluBench/avhbench/annotations/avhbench.json"
V1_OUT    = OUTER_ROOT / "outputs/lv2_avh_factprobe/audio_only_n50.jsonl"
OUT_DIR   = OUTER_ROOT / "outputs/lv2_avh_factprobe_v5"
OUT_FILE  = OUT_DIR / "audio_only_n50.jsonl"
SUMMARY_F = OUT_DIR / "SUMMARY_factprobe_v5.md"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _token_f1(pred: str, ref: str) -> float:
    pred_toks = set(pred.lower().split())
    ref_toks  = set(ref.lower().split())
    if not pred_toks or not ref_toks:
        return 0.0
    common = pred_toks & ref_toks
    if not common:
        return 0.0
    p = len(common) / len(pred_toks)
    r = len(common) / len(ref_toks)
    return 2 * p * r / (p + r)


STOP = {"a","an","the","is","are","was","were","it","its","in","on","at",
        "to","of","and","or","not","no","do","does","did","be","been",
        "being","have","has","had","this","that","i","you","he","she",
        "we","they","my","your","his","her","our","their"}


def _content_word_overlap(a: str, b: str) -> float:
    wa = {w.strip('.,!?;:"\'') for w in a.lower().split()
          if w.strip('.,!?;:"\'') not in STOP}
    wb = {w.strip('.,!?;:"\'') for w in b.lower().split()
          if w.strip('.,!?;:"\'') not in STOP}
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / min(len(wa), len(wb))


# ---------------------------------------------------------------------------
# Deterministic pre-filter (same as v3/v4)
# ---------------------------------------------------------------------------

_ENTITY_NOUNS = {
    "bird","cat","dog","person","man","woman","people","animal",
    "helicopter","car","truck","vehicle","boat","train","plane","aircraft",
    "toilet","pirate","cage","fire","smoke","tree","guitar","conductor",
    "orchestra","microphone","sign","crowd","clock","baby","seagull","pigeon",
}
_ACTIVITY_NEG_RE = re.compile(
    r"no visible (acceleration|motion|movement|flying|running|playing)"
    r"|no\s+\w+(\s+or\s+\w+(\s+\w+)?)?\s+(is|are)\s+(evident|audible|visible|present)"
    r"|(stationary|no visible motion|not moving|engine running but)"
    r"|no (indication|sign) of (playing|distortion|movement|motion|acceleration|grazing)",
    re.I,
)


def _is_entity_negation(candidate: str) -> bool:
    cl = candidate.lower().strip()
    for noun in _ENTITY_NOUNS:
        if re.search(
            rf"\bno\b[^.]*\b{re.escape(noun)}\b[^.]*\b(visible|present|specified|found|detected|shown)",
            cl,
        ):
            return True
    m = re.match(r"^no\s+(\w+)", cl)
    if m and m.group(1) in _ENTITY_NOUNS:
        return True
    for noun in _ENTITY_NOUNS:
        if re.search(rf"there is no\b[^.]*\b{re.escape(noun)}\b", cl):
            return True
    return False


def deterministic_gate(candidate: str, base_pred: str) -> tuple[bool, list[str]]:
    reasons = []
    if any(x in candidate for x in [r'\boxed', r'\frac', r'\text', r'\begin', r'\end']):
        reasons.append("format_injection")
    if _is_entity_negation(candidate) and not _ACTIVITY_NEG_RE.search(candidate):
        reasons.append("entity_negation")
    base_len = len(base_pred.split())
    cand_len = len(candidate.split())
    if base_len > 0 and cand_len < 0.40 * base_len:
        reasons.append("too_short")
    # v5: low_overlap check is REMOVED — modality-aware generation is expected to
    # produce different content (e.g. visual pass corrects visual subject, which
    # will have low overlap with a wrong base). Oracle expansion is the goal here.
    c_clean = re.sub(r"^(a |an |the )", "", candidate.lower()).rstrip(".")
    b_clean = re.sub(r"^(a |an |the )", "", base_pred.lower()).rstrip(".")
    if c_clean == b_clean:
        reasons.append("no_op")
    return len(reasons) == 0, reasons


# ---------------------------------------------------------------------------
# Diff extraction & modality classification
# ---------------------------------------------------------------------------

_AUDIO_ACTION_WORDS = {
    "bark","barks","barking","chirp","chirping","meow","meowing","baa","baaing",
    "sizzling","buzzing","whirring","hissing","bleating","honking","ringing",
    "whistling","thumping","squealing","snoring","beeping","clanking",
}

_FACT_DIMS = [
    ("subject",  r"^([A-Z][^,\.]+?)(?:,|\s+(?:is|are|was|were|flies|runs|sits|stands|speaks|moves|drives|walks|plays|hovers|perches))"),
    ("location", r"\b(?:in|on|at|near|beside|inside|outside|through|behind|in front of)\s+([^,\.]+)"),
    ("action",   r"\b(accelerates?|flies?|runs?|sits?|stands?|speaks?|moves?|drives?|walks?|plays?|hovers?|perches?|grazes?|barks?|lands?|leads?|conducting|signing|feeding|flushes?|refills?)\b"),
    ("sound",    r"\b(barks?|whistles?|thumping|buzzing|whirring?|squealing?|ringing?|snoring?|meow(?:ing)?|baa(?:ing)?|chirp(?:ing)?|sizzling|hissing|clanking)\b"),
    ("object",   r"\b(helicopter|bird|cat|dog|car|truck|boat|cage|guitar|microphone|blanket|thermometer|clock|barrel|ladder)\b"),
]


def extract_diff(base_pred: str, candidate: str) -> list[dict]:
    diffs = []
    for dim, pat in _FACT_DIMS:
        m_base = re.search(pat, base_pred, re.I)
        m_cand = re.search(pat, candidate, re.I)
        bv = m_base.group(1).strip().lower() if m_base else None
        cv = m_cand.group(1).strip().lower() if m_cand else None
        if bv and cv and bv != cv:
            diffs.append({"dim": dim, "base_val": bv, "cand_val": cv})
    if not diffs:
        base_words = set(base_pred.lower().split()) - STOP
        cand_words = set(candidate.lower().split()) - STOP
        removed = sorted(base_words - cand_words)[:4]
        added   = sorted(cand_words - base_words)[:4]
        if removed or added:
            diffs.append({
                "dim":      "general",
                "base_val": " ".join(removed),
                "cand_val": " ".join(added),
            })
    return diffs


def infer_diff_modality(diffs: list[dict]) -> str:
    """
    Returns 'visual', 'audio', or 'combined'.
    'visual': subject/location/object/non-audio action dims
    'audio':  sound dim, or action dim where the values are audio-action words
    'combined': both types present
    """
    has_visual = False
    has_audio  = False
    for d in diffs:
        dim = d["dim"]
        if dim == "sound":
            has_audio = True
        elif dim == "action":
            bv = d.get("base_val","").lower()
            cv = d.get("cand_val","").lower()
            if any(w in bv or w in cv for w in _AUDIO_ACTION_WORDS):
                has_audio = True
            else:
                has_visual = True
        elif dim in ("subject", "location", "object"):
            has_visual = True
        elif dim == "general":
            # look for audio-related words in the diff values
            combined_text = d.get("base_val","") + " " + d.get("cand_val","")
            if any(w in combined_text.lower() for w in _AUDIO_ACTION_WORDS):
                has_audio = True
            has_visual = True  # general changes are assumed to have a visual component too
    if has_visual and has_audio:
        return "combined"
    if has_audio:
        return "audio"
    return "visual"


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_visual_interrogation_prompt(question: str, base_pred: str) -> str:
    """
    Ask the model to produce the best-grounded visual description, based on VIDEO only.
    No "ACCURATE" escape hatch — always demand a concrete description so the
    downstream overlap filter can decide if it differs meaningfully from base.
    """
    return (
        f"Question: {question}\n\n"
        f"A description of this clip was produced: \"{base_pred}\"\n\n"
        f"Look at the VIDEO channel only (ignore the audio completely). "
        f"Based on what you can actually SEE — the main subject, their appearance, "
        f"the setting, actions, and visible objects — write a single corrected sentence "
        f"that best describes the visual content.\n\n"
        f"Preserve any audio-related facts from the original description (sounds, voices, "
        f"animal calls) unchanged. Fix only what the VIDEO evidence contradicts.\n\n"
        f"Reply with exactly one sentence:"
    )


def build_audio_interrogation_prompt(question: str, base_pred: str) -> str:
    """
    Ask the model to produce the best-grounded audio description, based on AUDIO only.
    No "ACCURATE" escape hatch — always demand a concrete description so the
    downstream overlap filter can decide if it differs meaningfully from base.
    """
    return (
        f"Question: {question}\n\n"
        f"A description of this clip was produced: \"{base_pred}\"\n\n"
        f"Listen to the AUDIO channel only (ignore the video completely). "
        f"Based on what you can actually HEAR — sounds, voices, animal calls, "
        f"music, audio events, and their sources — write a single corrected sentence "
        f"that best describes what is heard in the clip.\n\n"
        f"Preserve any visual facts from the original description (subjects, scenes, "
        f"appearances) unchanged. Fix only what the AUDIO evidence contradicts.\n\n"
        f"Reply with exactly one sentence:"
    )


def parse_candidate_from_response(raw: str, base_pred: str) -> str | None:
    """
    Extract the corrected sentence from the model response.
    Returns None if the model said ACCURATE or the output is indistinguishable from base.
    """
    r = raw.strip()
    # Check for ACCURATE
    if r.upper().startswith("ACCURATE") or r.strip().upper() == "ACCURATE":
        return None
    # Strip leading labels like "Corrected:" or "Corrected sentence:"
    r = re.sub(r"^(corrected[^:]*:|answer:|result:)\s*", "", r, flags=re.I).strip()
    # Remove trailing model artifacts
    r = re.sub(r"<\|im_end\|>.*", "", r).strip()
    # If it's very similar to base (trivial), return None
    if _content_word_overlap(r, base_pred) > 0.98 and abs(len(r.split()) - len(base_pred.split())) <= 1:
        return None
    if len(r.split()) < 5:
        return None
    return r


def build_verify_prompt(question: str, base_claim: str, cand_claim: str, modality: str) -> str:
    """
    Symmetric dual-probe: ask whether the CANDIDATE claim is supported by the
    appropriate channel, and whether the BASE claim is supported.
    channel is determined by diff modality, not by arm label.
    """
    if modality == "visual":
        channel_instr = "Focus on the VIDEO channel (visual evidence only — ignore the audio)."
    elif modality == "audio":
        channel_instr = "Focus on the AUDIO channel (audio evidence only — ignore the video)."
    else:
        channel_instr = "Use both the VIDEO and AUDIO channels."

    return (
        f"Question: {question}\n\n"
        f"{channel_instr}\n\n"
        f"Two claims about this clip:\n"
        f"  BASE:      \"{base_claim}\"\n"
        f"  CANDIDATE: \"{cand_claim}\"\n\n"
        f"Based ONLY on what you directly perceive in the clip right now, "
        f"which claim does the evidence SUPPORT?\n"
        f"  If CANDIDATE is supported and BASE is not: reply CANDIDATE_BETTER\n"
        f"  If BASE is supported and CANDIDATE is not: reply BASE_BETTER\n"
        f"  If both or neither is supported: reply INSUFFICIENT\n\n"
        f"Reply with exactly one phrase: CANDIDATE_BETTER, BASE_BETTER, or INSUFFICIENT.\n\n"
        f"Verdict:"
    )


def parse_verify_verdict(raw: str) -> str:
    r = raw.strip().upper()
    if "CANDIDATE_BETTER" in r or "CANDIDATE BETTER" in r:
        return "CANDIDATE_BETTER"
    if "BASE_BETTER" in r or "BASE BETTER" in r:
        return "BASE_BETTER"
    return "INSUFFICIENT"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm",   default="audio_only",
                        choices=["audio_only", "video_only", "combined"])
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--max-new-tokens-gen",    type=int, default=120)
    parser.add_argument("--max-new-tokens-verify", type=int, default=8)
    parser.add_argument("--v1-pool", default=str(V1_OUT))
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ load v1 pool (baseline)
    print("[v5] Loading v1 baseline pool …")
    v1_rows = {}
    with open(args.v1_pool) as f:
        for line in f:
            try:
                r = json.loads(line)
                v1_rows[r["id"]] = r
            except Exception:
                pass
    v1_ids = [r["id"] for r in sorted(v1_rows.values(), key=lambda x: int(x["id"]))
              if int(r["id"]) < args.limit]
    print(f"[v5] {len(v1_ids)} samples (limit={args.limit})")

    # ------------------------------------------------------------------ load AVHBench + model
    print("[v5] Loading AVHBench data …")
    AVH_DATA      = str(AVH_JSON)
    AVH_DATA_ROOT = str(OUTER_ROOT / "OmniHalluBench/avhbench")
    with open(AVH_JSON) as f:
        avhbench_data = json.load(f)

    print("[v5] Loading model …")
    from methods.shared import model_utils as _mu
    from methods.shared.prompts import get_system_prompt
    from methods.shared.data_loader import load_dataset, load_media_for_item

    build_inputs                         = _mu.build_inputs
    make_messages                        = _mu.make_messages
    build_user_content_with_media_tokens = _mu.build_user_content_with_media_tokens
    prepare_for_thinker                  = _mu.prepare_for_thinker
    decode_outputs                       = _mu.decode_outputs

    _ = torch.zeros(1, device="cuda")
    shm_path = Path("/dev/shm/Qwen2.5-Omni-7B")
    if shm_path.exists():
        print(f"[v5] Using shm model at {shm_path}")
        processor, model = _mu.load_model(str(shm_path))
    else:
        processor, model = _mu.load_model()

    sys_prompt = get_system_prompt("avhbench", "v1")

    avh_items = load_dataset(AVH_DATA, AVH_DATA_ROOT)
    avh_by_uid = {}
    for it in avh_items:
        uid = it.get("uid", "")
        try:
            num = str(int(uid.split("_")[-1]))
            avh_by_uid[num] = it
        except (ValueError, IndexError):
            pass

    # ------------------------------------------------------------------ resume
    done_ids = set()
    if OUT_FILE.exists():
        with open(OUT_FILE) as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["id"])
                except Exception:
                    pass
        print(f"[v5] Resuming — {len(done_ids)} already done")

    out_fh = open(OUT_FILE, "a")

    # ------------------------------------------------------------------ generation helpers
    def _run_gen(prompt_text, pv, av, iv, max_new_tokens):
        uc   = build_user_content_with_media_tokens(pv, av, iv, prompt_text)
        msgs = make_messages(sys_prompt, uc)
        inp  = build_inputs(processor, model, msgs, pv, av, iv, "avhbench")
        out  = model.generate(
            **inp,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            use_cache=True,
        )
        # Qwen2.5-Omni model.generate() returns (sequences_tensor, audio_tensor) tuple
        if isinstance(out, (tuple, list)):
            out = out[0]
        input_len = inp["input_ids"].shape[1]
        texts = decode_outputs(processor, out, input_len)
        # decode_outputs returns a list of strings; return the first element
        if isinstance(texts, (list, tuple)):
            return texts[0] if texts else ""
        return texts

    # ------------------------------------------------------------------ per-sample loop
    for rid in v1_ids:
        if rid in done_ids:
            continue

        v1_row   = v1_rows[rid]
        base_pred = v1_row["base_pred"]
        question  = v1_row.get("question", "")
        gt        = v1_row.get("gt", "")
        base_f1   = v1_row.get("base_f1") or _token_f1(base_pred, gt)

        # Load media — always both channels (same .mp4)
        uid       = f"avhbench_{int(rid):06d}"
        data_item = avh_by_uid.get(rid)
        pv, av, iv = None, None, None
        if data_item is not None:
            try:
                pv, av, iv = load_media_for_item(data_item, AVH_DATA, AVH_DATA_ROOT)
                print(f"  [DBG] id={rid} media loaded: pv={pv is not None} av={av is not None} iv={iv is not None}")
            except Exception as e:
                print(f"  [WARN] media load failed for {uid}: {e}")
        else:
            print(f"  [WARN] data_item not found for uid={uid}")

        # ---- Step 1: Visual interrogation (video only, no audio) ----
        visual_candidate     = None
        visual_det_passed    = None
        visual_det_reasons   = []
        visual_verify        = None
        visual_cand_f1       = None

        visual_prompt = build_visual_interrogation_prompt(question, base_pred)
        try:
            vis_raw = _run_gen(visual_prompt, pv, None, iv,
                               args.max_new_tokens_gen)
            visual_candidate = parse_candidate_from_response(vis_raw, base_pred)
        except Exception as e:
            import traceback as _tb
            print(f"  [WARN] visual gen failed id={rid}: {e}")
            _tb.print_exc()

        if visual_candidate:
            visual_det_passed, visual_det_reasons = deterministic_gate(
                visual_candidate, base_pred)
            visual_cand_f1 = _token_f1(visual_candidate, gt)
            # Verify with VISUAL channel (diff modality for visual candidate)
            if visual_det_passed:
                diffs = extract_diff(base_pred, visual_candidate)
                mod = infer_diff_modality(diffs) if diffs else "visual"
                # Override to visual since this was generated from visual pass
                mod = "visual" if mod != "audio" else "combined"
                vprompt = build_verify_prompt(question, base_pred, visual_candidate, mod)
                try:
                    vraw = _run_gen(vprompt, pv, av, iv,
                                    args.max_new_tokens_verify)
                    visual_verify = parse_verify_verdict(vraw)
                except Exception as e:
                    print(f"  [WARN] visual verify failed id={rid}: {e}")
                    visual_verify = "INSUFFICIENT"

        # ---- Step 2: Audio interrogation (audio only, no video) ----
        audio_candidate     = None
        audio_det_passed    = None
        audio_det_reasons   = []
        audio_verify        = None
        audio_cand_f1       = None

        audio_prompt = build_audio_interrogation_prompt(question, base_pred)
        try:
            aud_raw = _run_gen(audio_prompt, None, av, iv,
                               args.max_new_tokens_gen)
            audio_candidate = parse_candidate_from_response(aud_raw, base_pred)
        except Exception as e:
            print(f"  [WARN] audio gen failed id={rid}: {e}")

        if audio_candidate:
            audio_det_passed, audio_det_reasons = deterministic_gate(
                audio_candidate, base_pred)
            audio_cand_f1 = _token_f1(audio_candidate, gt)
            if audio_det_passed:
                diffs = extract_diff(base_pred, audio_candidate)
                mod = infer_diff_modality(diffs) if diffs else "audio"
                mod = "audio" if mod != "visual" else "combined"
                aprompt = build_verify_prompt(question, base_pred, audio_candidate, mod)
                try:
                    araw = _run_gen(aprompt, pv, av, iv,
                                    args.max_new_tokens_verify)
                    audio_verify = parse_verify_verdict(araw)
                except Exception as e:
                    print(f"  [WARN] audio verify failed id={rid}: {e}")
                    audio_verify = "INSUFFICIENT"

        # ---- Step 3: Accept decision ----
        # Accept visual candidate if it passed gate + verify = CANDIDATE_BETTER
        # Accept audio candidate if it passed gate + verify = CANDIDATE_BETTER
        # If both accepted, pick the one with higher candidate_f1 (offline oracle logic)
        # — but in online path pick the visual one first (it modifies more content)

        vis_accepted = (visual_candidate is not None and
                        visual_det_passed and
                        visual_verify == "CANDIDATE_BETTER")
        aud_accepted = (audio_candidate is not None and
                        audio_det_passed and
                        audio_verify == "CANDIDATE_BETTER")

        if vis_accepted and aud_accepted:
            # Both passed: pick by verify quality; prefer visual if tie
            # (online path: no GT access — just pick visual)
            accepted_candidate = visual_candidate
            accept_source = "visual"
        elif vis_accepted:
            accepted_candidate = visual_candidate
            accept_source = "visual"
        elif aud_accepted:
            accepted_candidate = audio_candidate
            accept_source = "audio"
        else:
            accepted_candidate = None
            accept_source = "none"

        accepted      = accepted_candidate is not None
        repaired_pred = accepted_candidate if accepted else base_pred
        repaired_f1   = _token_f1(repaired_pred, gt) if gt else base_f1

        # Oracle: best available candidate
        candidates_f1 = [f for f in [visual_cand_f1, audio_cand_f1] if f is not None]
        oracle_f1_sample = max([base_f1] + candidates_f1)

        row = dict(
            id=rid, question=question, gt=gt,
            base_pred=base_pred, base_f1=base_f1,
            # Visual pass
            visual_candidate=visual_candidate,
            visual_det_passed=visual_det_passed,
            visual_det_reasons=visual_det_reasons,
            visual_verify=visual_verify,
            visual_cand_f1=visual_cand_f1,
            # Audio pass
            audio_candidate=audio_candidate,
            audio_det_passed=audio_det_passed,
            audio_det_reasons=audio_det_reasons,
            audio_verify=audio_verify,
            audio_cand_f1=audio_cand_f1,
            # Accept
            accepted=accepted,
            accept_source=accept_source,
            repaired_pred=repaired_pred,
            repaired_f1=repaired_f1,
            oracle_f1_sample=oracle_f1_sample,
        )
        out_fh.write(json.dumps(row) + "\n")
        out_fh.flush()

        # Progress
        vis_tag = (f"vis=DET_FAIL({','.join(visual_det_reasons[:2])})"
                   if visual_candidate and not visual_det_passed
                   else f"vis={visual_verify or ('ACCURATE' if not visual_candidate else 'det_fail')}")
        aud_tag = (f"aud=DET_FAIL({','.join(audio_det_reasons[:2])})"
                   if audio_candidate and not audio_det_passed
                   else f"aud={audio_verify or ('ACCURATE' if not audio_candidate else 'det_fail')}")
        print(
            f"  [{str(rid):3s}] {vis_tag:40s} {aud_tag:40s} "
            f"acc={accepted}({accept_source})  "
            f"rep_f1={repaired_f1:.3f}"
        )

    out_fh.close()

    # ------------------------------------------------------------------ summary
    all_rows = []
    with open(OUT_FILE) as f:
        for line in f:
            try:
                all_rows.append(json.loads(line))
            except Exception:
                pass

    def safe_mean(vals):
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else 0.0

    base_f1_mean     = safe_mean([r["base_f1"]     for r in all_rows])
    repaired_f1_mean = safe_mean([r["repaired_f1"] for r in all_rows])
    delta_f1         = repaired_f1_mean - base_f1_mean
    oracle_f1_mean   = safe_mean([r["oracle_f1_sample"] for r in all_rows])
    oracle_delta     = oracle_f1_mean - base_f1_mean

    n_accepted         = sum(1 for r in all_rows if r.get("accepted"))
    n_vis_generated    = sum(1 for r in all_rows if r.get("visual_candidate"))
    n_aud_generated    = sum(1 for r in all_rows if r.get("audio_candidate"))
    n_vis_det_pass     = sum(1 for r in all_rows
                             if r.get("visual_candidate") and r.get("visual_det_passed"))
    n_aud_det_pass     = sum(1 for r in all_rows
                             if r.get("audio_candidate") and r.get("audio_det_passed"))
    n_vis_accepted     = sum(1 for r in all_rows if r.get("accept_source") == "visual")
    n_aud_accepted     = sum(1 for r in all_rows if r.get("accept_source") == "audio")

    # vs v1 known labels (offline)
    v1_good_ids = {"0","7","11","14","15","23","24","25","28","30","32","45","46","49"}
    v1_bad_ids  = {"1","3","6","8","9","12","13","16","20","22","26","27","31",
                   "34","35","37","39","40","42","43","44","47","48"}
    n_pos_acc  = sum(1 for r in all_rows if r.get("accepted") and r["id"] in v1_good_ids)
    n_neg_acc  = sum(1 for r in all_rows if r.get("accepted") and r["id"] in v1_bad_ids)
    n_false_rej= sum(1 for r in all_rows if not r.get("accepted") and r["id"] in v1_good_ids)

    # New candidates vs v1 oracle
    n_vis_improved = sum(1 for r in all_rows
                         if (r.get("visual_cand_f1") or 0) > r["base_f1"] + 0.001)
    n_aud_improved = sum(1 for r in all_rows
                         if (r.get("audio_cand_f1") or 0) > r["base_f1"] + 0.001)
    n_vis_degraded = sum(1 for r in all_rows
                         if r.get("visual_candidate")
                         and (r.get("visual_cand_f1") or 0) < r["base_f1"] - 0.001)
    n_aud_degraded = sum(1 for r in all_rows
                         if r.get("audio_candidate")
                         and (r.get("audio_cand_f1") or 0) < r["base_f1"] - 0.001)

    v1_oracle = 0.2854  # reference: n=50 frozen pool

    lines = [
        "# FactProbe v5 Summary — Modality-Aware Candidate Generation",
        "",
        "## Key results (n=50)",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| base_f1_mean              | {base_f1_mean*100:.2f}% |",
        f"| repaired_f1_mean          | {repaired_f1_mean*100:.2f}% |",
        f"| delta_f1                  | {delta_f1*100:+.2f}% |",
        f"| candidate_oracle_f1       | {oracle_f1_mean*100:.2f}% |",
        f"| oracle_delta_vs_base      | {oracle_delta*100:+.2f}% |",
        f"| v1_oracle (reference)     | {v1_oracle*100:.2f}% |",
        f"| oracle_delta_vs_v1_oracle | {(oracle_f1_mean - v1_oracle)*100:+.2f}% |",
        f"| n_accepted                | {n_accepted}/50 |",
        f"| n_vis_accepted            | {n_vis_accepted} |",
        f"| n_aud_accepted            | {n_aud_accepted} |",
        "",
        "## Candidate generation stats",
        "",
        "| Channel | Generated | Det_pass | oracle_improved | oracle_degraded |",
        "|---|---|---|---|---|",
        f"| Visual | {n_vis_generated}/50 | {n_vis_det_pass} | {n_vis_improved} | {n_vis_degraded} |",
        f"| Audio  | {n_aud_generated}/50 | {n_aud_det_pass} | {n_aud_improved} | {n_aud_degraded} |",
        "",
        "## Gate analysis vs v1 labels (offline only)",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| n_positive_accept (v1-good accepted) | {n_pos_acc}/14 |",
        f"| n_negative_accept (v1-bad accepted)  | {n_neg_acc}/23 |",
        f"| n_false_reject    (v1-good rejected) | {n_false_rej}/14 |",
        "",
        "## Comparison table",
        "",
        "| Version | base_f1 | repaired_f1 | delta | oracle | n_accepted |",
        "|---|---|---|---|---|---|",
        "| v1 (open-ended probe)           | 27.05% | 25.58% | -1.47% | 28.54% | 37/50 |",
        "| v3 (confession-gate)            | 27.05% | 26.94% | -0.11% | 28.54% |  2/50 |",
        "| v4 (raw-evidence, wrong channel)| 27.05% | 27.05% | +0.00% | 28.54% |  1/50 |",
        f"| v5 (modality-aware gen)         | {base_f1_mean*100:.2f}% | "
        f"{repaired_f1_mean*100:.2f}% | {delta_f1*100:+.2f}% | "
        f"{oracle_f1_mean*100:.2f}% | {n_accepted}/50 |",
    ]

    SUMMARY_F.write_text("\n".join(lines))
    print(f"\n[v5] Done. base={base_f1_mean:.4f} repaired={repaired_f1_mean:.4f} "
          f"oracle={oracle_f1_mean:.4f} delta={delta_f1:+.4f} accepted={n_accepted}/50")
    print(f"[v5] oracle_delta_vs_v1={oracle_f1_mean - v1_oracle:+.4f}  "
          f"vis_gen={n_vis_generated} aud_gen={n_aud_generated} "
          f"pos_acc={n_pos_acc} neg_acc={n_neg_acc}")
    print(f"[v5] Summary → {SUMMARY_F}")

    msg = (f"FactProbe v5完成: oracle={oracle_f1_mean*100:.2f}% "
           f"(v1={v1_oracle*100:.2f}%,delta={oracle_f1_mean-v1_oracle:+.4f}) "
           f"repaired_delta={delta_f1*100:+.2f}% accepted={n_accepted}/50 "
           f"vis_gen={n_vis_generated} aud_gen={n_aud_generated}")
    os.system(f'/home/tsq/.claude/feishu/feishu_send.sh "{msg}"')


if __name__ == "__main__":
    main()
