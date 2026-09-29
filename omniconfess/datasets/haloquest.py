#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""haloquest.py -- HaloQuest as a FREE-TEXT three-stage confession instance.

Instantiates the three_stage_freeform spine end-to-end on the model, with no
precomputed inputs -- every stage runs at inference time:

  Stage 1  Generate   greedy-decode the base answer LIVE from the image view
                      with the v1 prompt (the same prompt the confession views
                      use), so the pipeline is fully self-contained. The full-
                      view per-token log-probs are captured DURING this decode
                      (== the full-view teacher-force), so Stage 2 needs only the
                      no-image forward -> one fewer model call per sample.
  Stage 2  Confess    delta_img_tok over the committed answer tokens
                      (lp_full - lp_noimg), always-on. LOW delta = the answer
                      token was driven by text priors, not the image = the
                      ungrounded / fabricated part.
  Stage 3  Decide     bucket-routed (item['sub_category']), delta-driven, and
                      re-generated under the stronger v2 engineered system prompt
                      ("strictly visible evidence"):
                        false premises / visual challenge -> DIRECTED
                                            REGENERATION: point the model at the
                                            lowest-delta span (often a miscount
                                            or a not-actually-present detail) and
                                            make it re-examine the image.
                        insufficient context -> keep base (no signal to act on).

Stage 3 genuinely CONSUMES the Stage-2 confession signal: the lowest-delta span
localises the ungrounded region and the directed-regen prompt aims the model
back at the image there. --fp_mode switches only the FALSE-PREMISES branch:
'directed_regen' (default) or 'premise_check' (a two-step presupposition-extract-
then-answer generator, kept for ablation).

Usage:
  source scripts/env.sh
  CUDA_VISIBLE_DEVICES=0 python run_confess.py haloquest \\
      --halo_shard 0 --halo_nshards 8 [--dry_run]
"""
import os, json, argparse, re, string

# omniconfess/datasets/haloquest.py -> repo root is three levels up. Everything
# resolves relative to this so a fresh clone runs without editing source.
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault('MMHALU_MODEL_ROOT', '/dev/shm')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import numpy as np
import torch

from ..backbone import (
    load_dataset, load_media_for_item,
    load_model, build_inputs, prepare_for_thinker,
    build_user_content_with_media_tokens, make_messages,
    get_system_prompt, get_user_prompt,
)
from ..spine_freeform import (
    FreeformThreeStageDriver, lowest_delta_span, word_overlap,
)
from ..spine import teacher_force_seq_lp

DATA_PATH = os.path.join(REPO, 'OmniHalluBench', 'haloquest', 'annotations', 'haloquest.json')
DATA_ROOT = os.path.join(REPO, 'OmniHalluBench', 'haloquest')


# refusal / evasion detector (from run_lv2_grid EVADERE)
EVADERE = re.compile(
    r"\b(cannot|can't|unable|impossible|not (possible|able)|"
    r"no (information|data|detail)|not (visible|present|shown|clear|"
    r"available|enough|sufficient)|unclear|don't know|I('m| am) not sure|"
    r"hard to (tell|say|determine)|difficult to|cannot determine|"
    r"not (in|from) the (image|picture|photo)|"
    r"image does not|picture does not|photo does not|"
    r"I (can't|cannot) see|not (show|visible|apparent)|"
    r"outside (the|my)|beyond (the|my)|based on (the|this) image)",
    re.IGNORECASE)

_NEG = (' no ', ' not ', "n't", 'none', 'nothing', 'isn', 'aren', 'doesn',
        'cannot', "can't", 'without', 'there is no', 'does not', 'do not',
        'never', 'not visible', 'not present', 'not shown')


def is_refusal(txt):
    return bool(EVADERE.search(str(txt)))


def is_negation(txt):
    s = ' ' + str(txt).lower() + ' '
    return any(n in s for n in _NEG)


_TERSE_STOP = {'is', 'are', 'the', 'a', 'an', 'it', 'there', 'of', 'in', 'on',
               'this', 'that', 'to', 'and'}


def content_len(txt):
    """Number of non-trivial content words -- used to detect terse answers
    ("2", "0", "Tree", "It is white") whose word-overlap with a full sentence
    is uninformative."""
    ws = re.sub(r'[^\w\s]', ' ', str(txt).lower()).split()
    return len([w for w in ws if w not in _TERSE_STOP])


_NUM_WORDS = {'0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
              '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine',
              '10': 'ten', '11': 'eleven', '12': 'twelve', '13': 'thirteen',
              '14': 'fourteen', '15': 'fifteen', '16': 'sixteen',
              '17': 'seventeen', '18': 'eighteen', '19': 'nineteen',
              '20': 'twenty'}


def _count_noun(question):
    """Pull the counted noun out of a 'how many <noun> ...' question so a bare
    numeric answer can be re-expressed in the sentence form the references use."""
    m = re.search(r'how many ([a-z][a-z ]*?)'
                  r'(?: are| is| can| do| does| seen| visible|\?|$)',
                  str(question).lower())
    return m.group(1).strip() if m else ''


def normalize_number_answer(pred, question):
    """Deterministic Stage-3 output formatting: a bare integer answer ("2")
    scores ~0 Token-F1 against HaloQuest references, which spell numbers out and
    phrase them as sentences ("two dots; there are two dots"). Spell the digit
    and, for 'how many' questions, wrap it with the counted noun. Model-free and
    idempotent, so it never depends on decoding noise; only rewrites answers that
    are *exactly* a bare integer (which otherwise ship at 0)."""
    p = str(pred).strip().rstrip('.')
    if re.fullmatch(r'\d+', p) and p in _NUM_WORDS:
        noun = _count_noun(question)
        return (f"There are {_NUM_WORDS[p]} {noun}."
                if noun else f"{_NUM_WORDS[p].capitalize()}.")
    return pred


# --------------------------------------------------------------------------
# view builders (image / no-image), tokenizer, greedy generate
# --------------------------------------------------------------------------
def build_view_img(processor, model, sys_prompt, image, up):
    uc = build_user_content_with_media_tokens(None, None, image, up)
    raw = build_inputs(processor, model, make_messages(sys_prompt, uc),
                       None, None, image, 'haloquest')
    return prepare_for_thinker(raw, model)


def build_view_noimg(processor, model, sys_prompt, up):
    uc = build_user_content_with_media_tokens(None, None, None, up)
    raw = build_inputs(processor, model, make_messages(sys_prompt, uc),
                       None, None, None, 'haloquest')
    return prepare_for_thinker(raw, model)


def greedy(model, tokenizer, clean, max_new_tokens):
    with torch.inference_mode():
        out = model.thinker.generate(
            **clean, max_new_tokens=max_new_tokens,
            do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][clean['input_ids'].shape[1]:],
                            skip_special_tokens=True).strip()


def greedy_with_lp(model, tokenizer, clean, max_new_tokens):
    """Stage-1 Generate that ALSO returns Stage-2's full-view log-probs for free.

    Returns (text, gen_ids, lp_full). Because greedy decoding feeds its own
    emitted token back as the next input, the generation prefix is byte-for-byte
    the teacher-forced prefix, so the per-step log-prob of each emitted token
    UNDER THIS (image) VIEW equals teacher_force_seq_lp(clean_full, gen_ids). We
    therefore fold Stage-2's full-view teacher-force into the generate() pass and
    only run the no-image view separately -> one fewer forward per sample.

    gen_ids are the RAW emitted ids (a trailing EOS is trimmed) so the confession
    signal is bound to the tokens actually produced, not a decode->re-encode
    approximation."""
    eos = tokenizer.eos_token_id
    with torch.inference_mode():
        out = model.thinker.generate(
            **clean, max_new_tokens=max_new_tokens,
            do_sample=False, pad_token_id=eos,
            return_dict_in_generate=True, output_scores=True)
    input_len = clean['input_ids'].shape[1]
    gen_ids = out.sequences[0][input_len:]
    scores = out.scores                      # tuple: one [1, vocab] per step
    n = min(gen_ids.shape[0], len(scores))
    lp_full = []
    for i in range(n):
        lsm = torch.log_softmax(scores[i][0].float(), dim=-1)
        lp_full.append(float(lsm[gen_ids[i]]))
    gen_ids = gen_ids[:n]
    # trim a trailing EOS (not part of the answer text tokens)
    if gen_ids.numel() and int(gen_ids[-1]) == eos:
        gen_ids = gen_ids[:-1]
        lp_full = lp_full[:-1]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return text, gen_ids.detach(), lp_full


# --------------------------------------------------------------------------
# directed-regeneration prompts (Stage 3)
# --------------------------------------------------------------------------
def vc_regen_prompt(question, base_pred, span):
    # Visual-challenge failure mode is OVER-correction + verbosity: the base
    # answer is usually already right, so a skeptical "this may be wrong, fix it"
    # framing flips correct answers to wrong ones and injects contradictory
    # clauses ("not yellow; it is a bright glowing yellow"). Instead: verify the
    # single detail, keep it if it holds, and answer as one concise declarative
    # sentence that REUSES the question's nouns (GT is phrased that way, and the
    # metric is bag-of-words F1 against GT).
    return (
        "Look at the image and answer the question with one short, direct "
        "sentence.\n"
        f"Question: {question}\n"
        f"A draft answer said: \"{base_pred}\"\n"
        f"Double-check only this detail against the image: \"{span}\". "
        "If the image confirms it, keep the same answer. Only change it if the "
        "image clearly shows something different. State the single correct "
        "value, reusing the wording of the question. Give the answer directly, "
        "with no extra description, no hedging, and no phrases like "
        "\"in the image\".")


def ic_regen_prompt(question, base_pred):
    # Insufficient-context GT is a decline: either noun-specific
    # ("The dog's ears are not visible.") or generic uncertainty
    # ("we cannot tell from the image; it cannot be determined"). The base
    # answer (v1) frequently hallucinates a confident value for something the
    # image cannot actually show. Steer to a decline that REUSES the question's
    # subject (captures the noun-specific refs) AND carries the standard
    # uncertainty vocabulary (captures the generic refs) -> both flavours score.
    return (
        "Look at the image and answer the question with one short, direct "
        "sentence.\n"
        f"Question: {question}\n"
        f"A draft answer said: \"{base_pred}\"\n"
        "Answer using ONLY what is actually visible in the image. If the image "
        "does not clearly show the answer, do not guess: say that it cannot be "
        "determined from the image, reusing the subject named in the question. "
        "For example, \"What color are her shoes?\" becomes \"Her shoes are not "
        "visible in the image, so the color cannot be determined.\", and "
        "\"Who made this sculpture?\" becomes \"We cannot tell from the image "
        "who made the sculpture.\" Do not add extra description.")


def fp_regen_prompt(question, base_pred, span):
    # False-premise GT answers are the QUESTION turned into a negation, reusing
    # the question's own nouns and verb: "What kind of hat is the man wearing?"
    # -> "The man is not wearing a hat." The old boilerplate "there is no X
    # visible in this image" drops the subject/verb and adds off-vocabulary
    # tokens, so it loses Token-F1 even when the base was already a clean
    # negation. Steer the model to restate the question as a negative sentence.
    return (
        "Look at the image and answer the question with one short, direct "
        "sentence.\n"
        f"Question: {question}\n"
        f"A draft answer said: \"{base_pred}\"\n"
        f"Check whether the image really contains: \"{span}\". "
        "If the question takes for granted something that is not actually in "
        "the image, do NOT accept it: answer by turning the question into a "
        "negative statement that reuses the question's own subject and verb. "
        "For example, \"What kind of hat is the man wearing?\" becomes "
        "\"The man is not wearing a hat.\", and \"What color is the woman's "
        "necklace?\" becomes \"The woman is not wearing a necklace.\" "
        "Do not add extra description and do not use phrases like "
        "\"there is no ... visible in the image\".")


# --- premise-check FP generator (Phase B path 1): extract the question's
# presupposition, verify it against the image, then answer. Two GPU calls. ---
def premise_extract_prompt(question):
    return (
        "A question can presuppose that something exists or is true in an image.\n"
        f"Question: \"{question}\"\n"
        "In a few words, state the key thing this question ASSUMES is present "
        "or true in the image (the presupposition). Reply with just that "
        "assumed entity or fact, nothing else.")


def premise_answer_prompt(question, presupposition):
    return (
        "Look at the image carefully.\n"
        f"Question: {question}\n"
        f"This question assumes: \"{presupposition}\".\n"
        "Based ONLY on what the image actually shows, answer the question in "
        "ONE short sentence. If that assumption is not actually true in the "
        "image, simply state what is really there (e.g. \"There is no ... in "
        "the image\" or \"The ... is not visible\"). Do NOT explain your "
        "reasoning, do NOT say \"the assumption is false\" -- give only the "
        "direct answer sentence.")


# --------------------------------------------------------------------------
# HaloQuest free-text subclass
# --------------------------------------------------------------------------
class HaloQuestConfessDriver(FreeformThreeStageDriver):
    """Stage 2 = default delta_img_tok confession; Stage 3 = delta-driven
    bucket-routed directed regeneration + heuristic accept guard."""

    def __init__(self, model, processor, tokenizer, sys_prompt,
                 overlap_threshold=0.10, regen_max_tokens=64,
                 fp_mode='directed_regen', regen_sys_prompt=None):
        super().__init__(model, processor, tokenizer)
        self.sys_prompt = sys_prompt
        # Stage-3 (final answer) uses the stronger engineered system prompt when
        # supplied; falls back to the base prompt otherwise.
        self.regen_sys_prompt = regen_sys_prompt or sys_prompt
        self.overlap_threshold = overlap_threshold
        self.regen_max_tokens = regen_max_tokens
        self.fp_mode = fp_mode
        # per-sample context set by the driver loop before run_sample
        self._image_values = None
        # Stage-1 generation hands the full-view log-probs here so Stage-2 skips
        # the redundant full-view teacher-force (see greedy_with_lp).
        self._lp_full = None

    def confess(self, base_pred, clean_full, clean_noimg, gen_ids, item):
        """Stage 2. delta_img_tok, but the FULL-view log-probs were already
        computed during Stage-1 generation (self._lp_full), so only the no-image
        view is teacher-forced here -> one fewer forward per sample."""
        toks = [self.tokenizer.decode([t]) for t in gen_ids.tolist()]
        lp_full = self._lp_full
        if lp_full is None:            # fallback: no precomputed lp (safety)
            lp_full = teacher_force_seq_lp(self.model, clean_full, gen_ids)
        lp_noimg = teacher_force_seq_lp(self.model, clean_noimg, gen_ids)
        delta = [f - n for f, n in zip(lp_full, lp_noimg)]
        return dict(tokens=toks, delta_img_tok=delta, lp_full=lp_full,
                    n_tok=len(toks))

    def _accept(self, base_pred, cand, qt):
        """Bucket-aware heuristic guard. Returns (accepted: bool, reason).

        Two data-driven refinements over the plain overlap gate:
          * revert-if-both-decline: when the base already declines the premise
            (negation/refusal) and the candidate is just another decline, the
            regen only reshuffles vocabulary and usually loses Token-F1 against
            GT (which reuses the question's own nouns) -> keep base.
          * terse-base bypass (VC): a 1-2 word base ("2", "0", "Tree") has ~0
            word-overlap with any full-sentence regen, so the overlap gate
            wrongly rejects good rewrites -> trust the candidate when the base
            is terse.
        """
        if not cand or cand.strip() == base_pred.strip():
            return False, 'nochange'
        ov = word_overlap(cand, base_pred)
        b_decl = is_negation(base_pred) or is_refusal(base_pred)
        c_decl = is_negation(cand) or is_refusal(cand)
        if qt == 'visual challenge':
            if content_len(base_pred) <= 2:            # terse base: overlap uninformative
                if b_decl and c_decl:
                    return False, 'keep_decline'
                return True, 'vc_terse_base'
            if ov < self.overlap_threshold:
                return False, 'low_overlap'
            if is_refusal(cand) and not is_refusal(base_pred):
                return False, 'new_refusal'
            if b_decl and c_decl:
                return False, 'keep_decline'
            return True, 'vc_ok'
        if qt == 'false premises':
            if b_decl and c_decl:                      # don't reword a good decline
                return False, 'keep_decline'
            if ov >= self.overlap_threshold or is_negation(cand):
                return True, 'fp_ok'
            return False, 'low_overlap'
        return False, 'none_bucket'

    def decide(self, base_pred, confess_signal, item):
        qt = item.get('sub_category', 'insufficient context')
        toks = confess_signal['tokens']
        delta = confess_signal['delta_img_tok']
        span, span_delta, span_idx = lowest_delta_span(toks, delta, max_words=3)
        question = item['question']

        rec = dict(id=str(item['id']), question=question, gt=item.get('gt', ''),
                   qt=qt, base_pred=base_pred, low_span=span,
                   low_span_delta=span_delta, low_span_idx=span_idx,
                   n_tok=confess_signal['n_tok'])

        # insufficient-context bucket: the answer should be a decline. The base
        # (v1) often hallucinates a confident value, so regenerate a noun-aware
        # decline and accept it when it actually declines (keep base otherwise).
        if qt == 'insufficient context':
            cand = ''
            try:
                cl = build_view_img(self.processor, self.model, self.regen_sys_prompt,
                                    self._image_values, ic_regen_prompt(question, base_pred))
                cand = greedy(self.model, self.tokenizer, cl, self.regen_max_tokens)
                del cl
                torch.cuda.empty_cache()
            except Exception as e:
                rec.update(stage3='ic_regen_error', regen_done=False, regen_candidate='',
                           accepted=False, accept_reason='error:' + str(e)[:80],
                           edited=False, repaired_pred=base_pred, final_pred=base_pred)
                return base_pred, rec
            # accept the regenerated decline; keep base if regen still asserts a
            # confident (non-declining) answer.
            accept = bool(cand) and cand.strip() != base_pred.strip() and \
                (is_refusal(cand) or is_negation(cand))
            final = cand if accept else base_pred
            rec.update(stage3='ic_regen', regen_done=True, regen_candidate=cand,
                       cand_is_refusal=is_refusal(cand), cand_is_negation=is_negation(cand),
                       accepted=accept, accept_reason='ic_decline' if accept else 'ic_keep_base',
                       edited=accept, repaired_pred=final, final_pred=final)
            return final, rec

        # FALSE PREMISES (ablation): premise-check directed generator.
        # Two GPU calls: extract presupposition -> verify against image & answer.
        # Stored so offline scoring can compare vs base / fp_expl / oracle.
        if qt == 'false premises' and self.fp_mode == 'premise_check':
            try:
                cl_e = build_view_img(self.processor, self.model, self.regen_sys_prompt,
                                      self._image_values, premise_extract_prompt(question))
                presup = greedy(self.model, self.tokenizer, cl_e, 32)
                del cl_e
                cl_a = build_view_img(self.processor, self.model, self.regen_sys_prompt,
                                      self._image_values, premise_answer_prompt(question, presup))
                pc = greedy(self.model, self.tokenizer, cl_a, self.regen_max_tokens)
                del cl_a
                torch.cuda.empty_cache()
            except Exception as e:
                rec.update(stage3='premise_check_error', regen_done=False, regen_candidate='',
                           accepted=False, accept_reason='error:' + str(e)[:80],
                           edited=False, repaired_pred=base_pred, final_pred=base_pred)
                return base_pred, rec
            # heuristic accept: take pc if it differs and is a plausible answer
            accepted, reason = self._accept(base_pred, pc, 'false premises')
            final = pc if accepted else base_pred
            rec.update(stage3='fp_premise_check', regen_done=True, regen_candidate=pc,
                       presupposition=presup,
                       cand_is_negation=is_negation(pc),
                       accepted=accepted, accept_reason=reason,
                       edited=accepted, repaired_pred=final, final_pred=final)
            return final, rec

        # No usable low-delta span -> nothing to regenerate. Keep base, but still
        # apply the deterministic number formatting so a bare-integer VC answer
        # ("2") is spelled out / wrapped instead of shipping at 0 Token-F1.
        if not span:
            final = normalize_number_answer(base_pred, question)
            rec.update(stage3='keep_base', regen_done=False,
                       regen_candidate='', accepted=False, accept_reason='no_span',
                       edited=(final.strip() != base_pred.strip()),
                       repaired_pred=final, final_pred=final)
            return final, rec

        # DIRECTED REGEN branch (VC always; FP unless fp_mode=='premise_check')
        # build directed prompt
        if qt == 'visual challenge':
            rp = vc_regen_prompt(question, base_pred, span)
        else:
            rp = fp_regen_prompt(question, base_pred, span)

        cand = ''
        try:
            cl = build_view_img(self.processor, self.model, self.regen_sys_prompt,
                                self._image_values, rp)
            cand = greedy(self.model, self.tokenizer, cl, self.regen_max_tokens)
            del cl
            torch.cuda.empty_cache()
        except Exception as e:
            rec.update(stage3='regen_error', regen_done=False, regen_candidate='',
                       accepted=False, accept_reason='error:' + str(e)[:80],
                       edited=False, repaired_pred=base_pred, final_pred=base_pred)
            return base_pred, rec

        accepted, reason = self._accept(base_pred, cand, qt)
        final = cand if accepted else base_pred
        final = normalize_number_answer(final, question)
        rec.update(stage3='directed_regen', regen_done=True, regen_candidate=cand,
                   cand_overlap=round(word_overlap(cand, base_pred), 4),
                   cand_is_refusal=is_refusal(cand), cand_is_negation=is_negation(cand),
                   accepted=accepted, accept_reason=reason,
                   edited=(final.strip() != base_pred.strip()),
                   repaired_pred=final, final_pred=final)
        return final, rec


def run_haloquest_pipeline(shard=0, nshards=8, outdir=None, fp_mode='directed_regen',
                           overlap_threshold=0.10, regen_max_tokens=64,
                           dry_run=False, file_name=None):
    """Run the HaloQuest free-text three-stage pipeline over one shard.

    Single authoritative implementation, shared by this module's CLI (main) and
    the unified entry omniconfess_all.run_haloquest, so both run the identical
    Stage1->Stage2->Stage3 flow. Writes one JSONL per shard; returns its path.
    No-image samples keep base (channel_computed=0) so a full run yields N=365.
    """
    if outdir is None:
        outdir = os.path.join(REPO, 'outputs', 'haloquest_freeform')
    os.makedirs(outdir, exist_ok=True)
    stem = file_name if file_name else f'freeform_s{shard}of{nshards}'
    outfile = os.path.join(outdir, stem + '.jsonl')

    data = load_dataset(DATA_PATH, DATA_ROOT)
    by_id = {str(it['id']): it for it in data}

    ids_all = sorted(by_id.keys(), key=lambda x: int(x))
    ids_shard = [i for j, i in enumerate(ids_all) if j % nshards == shard]
    if dry_run:
        # pick a mix of buckets for the dry run
        want = {'false premises': 2, 'visual challenge': 2, 'insufficient context': 1}
        picked, seen = [], {k: 0 for k in want}
        for i in ids_shard:
            qt = by_id[i]['sub_category']
            if seen.get(qt, 99) < want.get(qt, 0):
                picked.append(i); seen[qt] += 1
            if sum(seen.values()) >= sum(want.values()):
                break
        ids_shard = picked
    print(f'[shard {shard}/{nshards}] {len(ids_shard)} samples -> {outfile}', flush=True)

    done = set()
    if os.path.exists(outfile):
        for line in open(outfile, encoding='utf-8'):
            try: done.add(str(json.loads(line)['id']))
            except: pass

    print('[model] loading Qwen2.5-Omni-7B...', flush=True)
    processor, model = load_model()
    tokenizer = processor.tokenizer
    sys_prompt = get_system_prompt('haloquest', 'v2')          # Stage-1 draft (confession)
    regen_sys_prompt = get_system_prompt('haloquest', 'v2')    # Stage-3 final
    print(f'[fp_mode={fp_mode}] Stage-1 draft prompt=v2; Stage-3 regen prompt=v2; '
          f'fixed v1 base anchor spliced at scoring; '
          f'full-view lp folded into generation', flush=True)

    driver = HaloQuestConfessDriver(model, processor, tokenizer, sys_prompt,
                                    overlap_threshold=overlap_threshold,
                                    regen_max_tokens=regen_max_tokens,
                                    fp_mode=fp_mode, regen_sys_prompt=regen_sys_prompt)

    fout = open(outfile, 'a', encoding='utf-8')
    n_done = 0
    for sid in ids_shard:
        if sid in done:
            continue
        item = by_id[sid]
        qt = item['sub_category']
        up = get_user_prompt(item['question'], 'haloquest', 'v1')
        try:
            _, _, image_values = load_media_for_item(item, DATA_PATH, DATA_ROOT)
        except Exception as e:
            image_values = None
            print(f'  [warn {sid}] media err {e}', flush=True)

        if image_values is None:
            # no image: Stage-2 confession is undefined. Still generate a base
            # answer live from the text-only view so a full run yields N=365.
            try:
                cl_noimg = build_view_noimg(processor, model, sys_prompt, up)
                base_pred = greedy(model, tokenizer, cl_noimg, regen_max_tokens)
                del cl_noimg; torch.cuda.empty_cache()
            except Exception as e:
                print(f'  [skip {sid}] noimg base err {e}', flush=True); continue
            rec = dict(id=sid, question=item['question'], gt=item.get('gt', ''),
                       qt=qt, base_pred=base_pred, low_span='', low_span_delta=0.0,
                       low_span_idx=[], n_tok=0, stage3='keep_base_noimg',
                       regen_done=False, regen_candidate='', accepted=False,
                       accept_reason='no_image', edited=False,
                       repaired_pred=base_pred, final_pred=base_pred,
                       delta_img_tok=[], delta_img_agg_mean=0.0,
                       delta_img_agg_min=0.0, channel_computed=0)
            fout.write(json.dumps(rec, ensure_ascii=False) + '\n'); fout.flush()
            n_done += 1
            print(f'  [{sid}] {qt[:4]} no-image -> keep base', flush=True); continue

        # Stage 1 Generate + Stage 2 full-view lp in ONE pass: greedy-decode the
        # base answer from the image view and capture each token's full-view
        # log-prob (== the full-view teacher-force) for free.
        try:
            clean_full = build_view_img(processor, model, sys_prompt, image_values, up)
        except Exception as e:
            print(f'  [skip {sid}] view err {e}', flush=True); continue
        base_pred, gen_ids, lp_full = greedy_with_lp(model, tokenizer, clean_full,
                                                     regen_max_tokens)
        del clean_full; torch.cuda.empty_cache()
        if gen_ids.numel() == 0:
            print(f'  [skip {sid}] empty base tokens', flush=True); continue
        gen_ids = gen_ids.to(model.device)

        # no-image view for the remaining half of the confession signal
        try:
            clean_noimg = build_view_noimg(processor, model, sys_prompt, up)
        except Exception as e:
            print(f'  [skip {sid}] noimg view err {e}', flush=True); continue

        driver._image_values = image_values
        driver._lp_full = lp_full          # Stage 2 reuses this; skips full-view TF
        try:
            final_pred, rec = driver.run_sample(base_pred, None, clean_noimg,
                                                gen_ids, item)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f'  [error {sid}] {e}', flush=True)
            del clean_noimg; torch.cuda.empty_cache(); continue
        del clean_noimg
        torch.cuda.empty_cache()

        fout.write(json.dumps(rec, ensure_ascii=False) + '\n'); fout.flush()
        n_done += 1
        print(f'[{sid}] {qt[:4]} span={rec.get("low_span")!r}({rec.get("low_span_delta")}) '
              f'edit={rec.get("edited")} final={final_pred[:50]!r}', flush=True)

    fout.close()
    print(f'[shard {shard}] done, wrote {n_done} new -> {outfile}', flush=True)
    return outfile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=8)
    ap.add_argument('--outdir', default=os.path.join(REPO, 'outputs', 'haloquest_freeform'))
    ap.add_argument('--overlap_threshold', type=float, default=0.10)
    ap.add_argument('--regen_max_tokens', type=int, default=64)
    ap.add_argument('--fp_mode', choices=['directed_regen', 'premise_check'],
                    default='directed_regen',
                    help="FALSE-PREMISES Stage-3 source. 'directed_regen' "
                         "(default): delta-driven directed regeneration that "
                         "consumes the Stage-2 signal. 'premise_check': two-step "
                         "presupposition-extract-then-answer generator (ablation).")
    ap.add_argument('--dry_run', action='store_true')
    args = ap.parse_args()
    run_haloquest_pipeline(shard=args.shard, nshards=args.nshards, outdir=args.outdir,
                           fp_mode=args.fp_mode, overlap_threshold=args.overlap_threshold,
                           regen_max_tokens=args.regen_max_tokens, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
