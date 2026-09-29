#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""cf_confess_haloquest.py -- HaloQuest ArmD as a FREE-TEXT three-stage subclass.

Formalises the old run_lv2_grid ArmD pipeline into the three_stage_freeform
spine. The three stages, HaloQuest-specific:

  Stage 1  Generate   the v2 engineering-prompt base answer (read from
                      V2_PATH; base=39.37 Token-F1, the shipped baseline).
  Stage 2  Confess    delta_img_tok over the committed answer tokens
                      (lp_full - lp_noimg), always-on. LOW delta = the answer
                      token was driven by text priors, not the image = the
                      ungrounded / fabricated part.
  Stage 3  Decide     bucket-routed (item['sub_category']), HYBRID default:
                        false premises   -> selective2 routing to the precomputed
                                            fp candidate pool (fp_explicit /
                                            fp_neut). NOT delta-regen: the
                                            dedicated fp prompt pool generates
                                            stronger FP candidates (45.53 vs
                                            delta-regen 43.19 on this bucket).
                        visual challenge -> delta-driven DIRECTED REGENERATION:
                                            point the model at the lowest-delta
                                            span (often a mis-counted number)
                                            and make it re-examine the image
                                            (43.50 > always-vc prompt pool 42.67).
                        insufficient context -> keep base (no oil in this bucket).

selective2 (FP): if base_is_evadere -> use fp_expl;
                 elif base has no negation and fp_expl has negation -> use fp_expl;
                 else keep base.

--fp_mode switches the FP branch: 'selective2' (default, hybrid champion) or
'delta_regen' (the original delta-driven directed regen, kept for ablation).

This is the "generate a stronger candidate" paradigm: rather than asking the
same model to self-judge A/B candidates (shown to underperform -- the generator
hits the same grounding wall it did when it produced the wrong answer), each
bucket routes to the strongest available candidate source.

Usage:
  source scripts/env.sh
  CUDA_VISIBLE_DEVICES=0 $PY confession/cf_confess_haloquest.py \\
      --shard 0 --nshards 8 --outdir outputs/haloquest_freeform [--dry_run]
"""
import os, sys, json, argparse, re, string

REPO = '/home/tsq/rhq/omniconfess/OmniConfess'
sys.path.insert(0, os.path.join(REPO, 'OmniConfess', 'methods'))
sys.path.insert(0, os.path.join(REPO, 'confession'))

os.environ.setdefault('MMHALU_MODEL_ROOT', '/dev/shm')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import numpy as np
import torch

from shared import (
    load_dataset, load_media_for_item,
    load_model, build_inputs, prepare_for_thinker,
    build_user_content_with_media_tokens, make_messages,
    get_system_prompt, get_user_prompt,
)
from three_stage_freeform import (
    FreeformThreeStageDriver, lowest_delta_span, word_overlap,
)

DATA_PATH = os.path.join(REPO, 'OmniHalluBench', 'haloquest', 'annotations', 'haloquest.json')
DATA_ROOT = os.path.join(REPO, 'OmniHalluBench', 'haloquest')
V2_PATH   = os.path.join(REPO, 'outputs', 'haloquest_prompt_iter', 'haloquest_prompt_v2_full.jsonl')

# --- FP candidate pools (precomputed, no GPU) + avcd 608-id->question bridge ---
AVCD_PATH = os.path.join(REPO, 'outputs', 'haloquest_base_avcd', 'haloquest_base_avcd_full.jsonl')
FP_EXPL_PATH = '/home/tsq/.claude/jobs/b2990bf5/tmp/haloquest_fp_explicit_stage2.jsonl'
FP_NEUT_PATH = '/home/tsq/.claude/jobs/b2990bf5/tmp/haloquest_fp_neut_stage2.jsonl'


def nq(s):
    return " ".join(str(s).lower().split())


def load_fp_candidate_map():
    """question(normalized) -> {fp_expl, fp_neut, base_is_evadere, base_has_neg}
    via the avcd 608-id bridge. The fp pools carry 608-ids + no question field;
    avcd carries both 608-id and question, so it bridges pool-id -> question.
    Downstream join is by normalized question (pool ids are NOT 365-position)."""
    from collections import defaultdict
    id2q = {}
    for l in open(AVCD_PATH, encoding='utf-8'):
        r = json.loads(l); id2q[int(r['id'])] = nq(r.get('question', ''))
    cand = defaultdict(dict)
    for l in open(FP_EXPL_PATH, encoding='utf-8'):
        r = json.loads(l); q = id2q.get(int(r['id']))
        if q is None: continue
        cand[q]['fp_expl'] = r.get('fp_explicit_stage2_pred', '')
        cand[q]['base_is_evadere'] = r.get('base_is_evadere')
    for l in open(FP_NEUT_PATH, encoding='utf-8'):
        r = json.loads(l); q = id2q.get(int(r['id']))
        if q is None: continue
        cand[q]['fp_neut'] = r.get('fp_neut_stage2_pred', '')
        cand[q]['base_has_neg'] = r.get('base_has_neg')
        cand[q].setdefault('base_is_evadere', r.get('base_is_evadere'))
    return dict(cand)


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


# --------------------------------------------------------------------------
# directed-regeneration prompts (Stage 3)
# --------------------------------------------------------------------------
def vc_regen_prompt(question, base_pred, span):
    return (
        "Look at the image carefully and answer the question.\n"
        f"Question: {question}\n"
        f"A previous answer said: \"{base_pred}\"\n"
        f"The part \"{span}\" may not match what the image actually shows "
        "(for example a miscount or a wrong detail). Re-examine the image, "
        "count and identify carefully, and give the correct answer. "
        "If the image does not support that part, correct it. "
        "Answer in one sentence.")


def fp_regen_prompt(question, base_pred, span):
    return (
        "Look at the image carefully and answer the question.\n"
        f"Question: {question}\n"
        f"A previous answer said: \"{base_pred}\"\n"
        f"The part \"{span}\" describes something that may not actually be "
        "present in the image. If the question assumes something that is not "
        "shown in the image, say so plainly and state what is actually there. "
        "Answer in one sentence based only on what the image shows.")


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
                 fp_mode='selective2', fp_candidates=None):
        super().__init__(model, processor, tokenizer)
        self.sys_prompt = sys_prompt
        self.overlap_threshold = overlap_threshold
        self.regen_max_tokens = regen_max_tokens
        self.fp_mode = fp_mode
        self.fp_candidates = fp_candidates or {}
        # per-sample context set by the driver loop before run_sample
        self._image_values = None

    def _selective2(self, base_pred, question):
        """FP champion: route to fp candidate pool by grounding heuristics.
        Returns (final_pred, reason). No GPU."""
        c = self.fp_candidates.get(nq(question))
        if not c:
            return base_pred, 'no_cand'
        fpe = c.get('fp_expl', '')
        if c.get('base_is_evadere'):
            return (fpe or base_pred), 'evadere->expl'
        if (not is_negation(base_pred)) and is_negation(fpe):
            return fpe, 'neg->expl'
        return base_pred, 'keep_base'

    def _accept(self, base_pred, cand, qt):
        """Bucket-aware heuristic guard. Returns (accepted: bool, reason)."""
        if not cand or cand.strip() == base_pred.strip():
            return False, 'nochange'
        ov = word_overlap(cand, base_pred)
        if qt == 'visual challenge':
            if ov < self.overlap_threshold:
                return False, 'low_overlap'
            if is_refusal(cand) and not is_refusal(base_pred):
                return False, 'new_refusal'
            return True, 'vc_ok'
        if qt == 'false premises':
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

        # insufficient-context bucket: no oil, keep base
        if qt == 'insufficient context':
            rec.update(stage3='keep_base', regen_done=False,
                       regen_candidate='', accepted=False, accept_reason='ic_keep_base',
                       edited=False, repaired_pred=base_pred, final_pred=base_pred)
            return base_pred, rec

        # FALSE PREMISES: hybrid champion routes to fp candidate pool (selective2).
        if qt == 'false premises' and self.fp_mode == 'selective2':
            final, reason = self._selective2(base_pred, question)
            rec.update(stage3='fp_selective2', regen_done=False, regen_candidate='',
                       fp_expl=self.fp_candidates.get(nq(question), {}).get('fp_expl', ''),
                       accepted=(final != base_pred), accept_reason=reason,
                       edited=(final != base_pred), repaired_pred=final, final_pred=final)
            return final, rec

        # FALSE PREMISES (Phase B path 1): premise-check directed generator.
        # Two GPU calls: extract presupposition -> verify against image & answer.
        # Stored so offline scoring can compare vs base / fp_expl / oracle.
        if qt == 'false premises' and self.fp_mode == 'premise_check':
            try:
                cl_e = build_view_img(self.processor, self.model, self.sys_prompt,
                                      self._image_values, premise_extract_prompt(question))
                presup = greedy(self.model, self.tokenizer, cl_e, 32)
                del cl_e
                cl_a = build_view_img(self.processor, self.model, self.sys_prompt,
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
            fpe = self.fp_candidates.get(nq(question), {}).get('fp_expl', '')
            rec.update(stage3='fp_premise_check', regen_done=True, regen_candidate=pc,
                       presupposition=presup, fp_expl=fpe,
                       cand_is_negation=is_negation(pc),
                       accepted=accepted, accept_reason=reason,
                       edited=accepted, repaired_pred=final, final_pred=final)
            return final, rec

        # No usable low-delta span -> nothing to regenerate, keep base.
        if not span:
            rec.update(stage3='keep_base', regen_done=False,
                       regen_candidate='', accepted=False, accept_reason='no_span',
                       edited=False, repaired_pred=base_pred, final_pred=base_pred)
            return base_pred, rec

        # DIRECTED REGEN branch (VC always; FP only when fp_mode=='delta_regen')
        # build directed prompt
        if qt == 'visual challenge':
            rp = vc_regen_prompt(question, base_pred, span)
        else:
            rp = fp_regen_prompt(question, base_pred, span)

        cand = ''
        try:
            cl = build_view_img(self.processor, self.model, self.sys_prompt,
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
        rec.update(stage3='directed_regen', regen_done=True, regen_candidate=cand,
                   cand_overlap=round(word_overlap(cand, base_pred), 4),
                   cand_is_refusal=is_refusal(cand), cand_is_negation=is_negation(cand),
                   accepted=accepted, accept_reason=reason,
                   edited=accepted, repaired_pred=final, final_pred=final)
        return final, rec


def run_haloquest_pipeline(shard=0, nshards=8, outdir=None, fp_mode='selective2',
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
    v2 = {str(r['id']): r for r in map(json.loads, open(V2_PATH))}

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
    processor, model = load_model('/dev/shm/Qwen2.5-Omni-7B')
    tokenizer = processor.tokenizer
    sys_prompt = get_system_prompt('haloquest', 'v1')

    fp_candidates = load_fp_candidate_map() if fp_mode in ('selective2', 'premise_check') else {}
    print(f'[fp_mode={fp_mode}] fp_candidates loaded={len(fp_candidates)}', flush=True)

    driver = HaloQuestConfessDriver(model, processor, tokenizer, sys_prompt,
                                    overlap_threshold=overlap_threshold,
                                    regen_max_tokens=regen_max_tokens,
                                    fp_mode=fp_mode, fp_candidates=fp_candidates)

    fout = open(outfile, 'a', encoding='utf-8')
    n_done = 0
    for sid in ids_shard:
        if sid in done:
            continue
        item = by_id[sid]
        if sid not in v2:
            print(f'  [skip] id={sid} no v2 pred', flush=True); continue
        base_pred = v2[sid]['pred']
        qt = item['sub_category']
        try:
            _, _, image_values = load_media_for_item(item, DATA_PATH, DATA_ROOT)
        except Exception as e:
            image_values = None
            print(f'  [warn {sid}] media err {e}', flush=True)
        if image_values is None:
            # no image: Stage-2 confession is undefined -> keep base (align N=365)
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

        up = get_user_prompt(item['question'], 'haloquest', 'v1')

        # Stage 1 -> tokens; Stage 2 needs full + noimg views
        _, gen_ids = driver.tokenize_pred(base_pred)
        if gen_ids.numel() == 0:
            print(f'  [skip {sid}] empty base tokens', flush=True); continue
        gen_ids = gen_ids.to(model.device)

        try:
            clean_full = build_view_img(processor, model, sys_prompt, image_values, up)
            clean_noimg = build_view_noimg(processor, model, sys_prompt, up)
        except Exception as e:
            print(f'  [skip {sid}] view err {e}', flush=True); continue

        driver._image_values = image_values
        try:
            final_pred, rec = driver.run_sample(base_pred, clean_full, clean_noimg,
                                                gen_ids, item)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f'  [error {sid}] {e}', flush=True)
            del clean_full, clean_noimg; torch.cuda.empty_cache(); continue
        del clean_full, clean_noimg
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
    ap.add_argument('--fp_mode', choices=['selective2', 'delta_regen', 'premise_check'],
                    default='selective2',
                    help="FP-bucket Stage-3 source. 'selective2' (default, "
                         "hybrid champion): route to fp candidate pool. "
                         "'delta_regen': delta-driven directed regen (ablation). "
                         "'premise_check': two-step premise-check generator "
                         "(Phase B path 1, generate a stronger FP candidate).")
    ap.add_argument('--dry_run', action='store_true')
    args = ap.parse_args()
    run_haloquest_pipeline(shard=args.shard, nshards=args.nshards, outdir=args.outdir,
                           fp_mode=args.fp_mode, overlap_threshold=args.overlap_threshold,
                           regen_max_tokens=args.regen_max_tokens, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
