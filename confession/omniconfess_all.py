#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""omniconfess_all.py — OmniConfess: three-stage counterfactual confession
for all six OmniHalluBench datasets in one self-contained file.

The three-stage spine is identical across every dataset:

  Stage 1  Generate   greedy decode under the full input view until the model
                      commits to its answer (decision token for judgement tasks;
                      per-content-word position for free-form generation).
  Stage 2  Confess    teacher-force one or more counterfactual (ablated) views
                      on the same prefix, reading per-class log-probabilities at
                      the committed position.  Every sample that reaches the
                      commit point confesses unconditionally.
  Stage 3  Decide     the per-dataset operator maps the full-view signal and the
                      confession signal to a decision.  For judgement tasks the
                      framework hard-forces the chosen token
                      (scores[:] = -inf; scores[forced] = 0); for free-form tasks
                      it emits corrected text.

Per-dataset instantiation:

  PHD (binary Y/N)
    Ablated channel : text CONTEXT.
    Stage-3 rule    : gamma extrapolation — margin' = margin_full
                      + w*(margin_cf - margin_full); yes iff margin' > tau_p.

  CMM (binary Y/N)
    Ablated channels: audio-masked and video-masked views (both computed every
                      sample; relevant set R chosen per question / group_key).
    Stage-3 rule    : dual-channel grounding gate — keep base-yes when confident
                      (margin_full >= margin_conf) OR grounded in R
                      (min_{m in R} d_cf_m <= tau_m); flip to no otherwise.

  PubMedQA (3-class Y/N/Maybe)
    Ablated channel : PASSAGE (removal gives the parametric prior P^0).
    Stage-3 rule    : candidate-anchored gate — keep base class am unless
                      am==MAYBE and kappa_maybe = lm_full - lm_cf <= tau_maybe,
                      then redecide yes/no by full-view logprob.

  RAGTruth (free-form generation)
    Ablated channel : PASSAGE, lazily synced per content-word step.
    Stage-3 rule    : temperate re-rank — lp_full + lambda*max(Delta,0);
                      guarded by lead-in protection, signal floor, fluency floor.

  HaloQuest (free-form generation)
    Ablated channel : IMAGE, teacher-forced over the committed answer tokens.
    Stage-3 rule    : bucket-routed span correction — VC: directed regeneration
                      of the lowest-delta span; FP: routing to the fp candidate
                      pool; IC: keep base.

  AVHBench (free-form generation)
    Ablated channels: four same-backbone arms at different frame/resolution
                      settings; cross-arm disagreement is the confession signal.
    Stage-3 rule    : majority vote-SC across arms (content words present in
                      >= threshold arms win).

Usage
-----
  python omniconfess_all.py --datasets phd       --data_path <shard.json> ...
  python omniconfess_all.py --datasets cmm       --data_path <shard.json> ...
  python omniconfess_all.py --datasets pubmedqa  --data_path <shard.json> ...
  python omniconfess_all.py --datasets ragtruth  --data_path <shard.json> ...
  python omniconfess_all.py --datasets haloquest --halo_shard 0 --halo_nshards 8 \
                            --output_dir outputs/haloquest_freeform
  python omniconfess_all.py --datasets avhbench  --arm_paths a0.jsonl,a1.jsonl,a2.jsonl,a3.jsonl
"""
import os, sys, json, time, argparse, random, string, re

import numpy as np
import torch
from transformers import LogitsProcessor

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'OmniConfess', 'methods'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shared import (
    load_dataset, load_media_for_item, load_model, build_inputs,
    prepare_for_thinker, build_user_content_with_media_tokens, make_messages,
    get_system_prompt, get_user_prompt, make_result, save_result,
)
from cf_common import (set_seed, token_logprobs, split_question,
                       load_done_ids, attach_record, write_time)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Decision-token ids (Qwen2.5-Omni; shared by all datasets).
YES_ID, NO_ID, MAYBE_ID = 9454, 2753, 21390


# ===========================================================================
# SHARED SPINE: three-stage base class
# ===========================================================================

class ThreeStageConfessProcessor(LogitsProcessor):
    """Base class owning Stage-1 decision detection and Stage-3 token forcing.

    Subclasses implement two hooks:
        confess(prefix_ids) -> dict        # Stage 2
        decide(full_view, confess_signal) -> (token_id, record_dict)  # Stage 3
    """

    def __init__(self, tokenizer, prompt_len):
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len
        self.used = False
        self.record = None

    def confess(self, prefix_ids):
        raise NotImplementedError

    def decide(self, full_view, confess_signal):
        raise NotImplementedError

    def _at_decision(self, gen, am):
        if len(gen) == 0:
            return am in (YES_ID, NO_ID)
        tail = self.tokenizer.decode(gen[-4:])
        return tail.endswith('\\boxed{') or tail.endswith('\\boxed{{')

    @torch.inference_mode()
    def __call__(self, input_ids, scores):
        if self.used:
            return scores
        gen = input_ids[0, self.prompt_len:].tolist()
        logits = scores[0].float()
        am = int(logits.argmax().item())
        if not self._at_decision(gen, am):
            return scores
        lp = torch.log_softmax(logits, dim=-1)
        ly_full, ln_full = lp[YES_ID].item(), lp[NO_ID].item()
        full_view = dict(am=am, ly_full=ly_full, ln_full=ln_full,
                         margin_full=ly_full - ln_full, dec_index=len(gen))
        confess_signal = self.confess(gen)          # Stage 2 — unconditional
        forced, record = self.decide(full_view, confess_signal)  # Stage 3
        scores[0][:] = float('-inf')
        scores[0][forced] = 0.0
        self.used = True
        self.record = record
        return scores


@torch.inference_mode()
def teacher_force_lp(model, clean_view, prefix_ids, token_ids):
    """Teacher-force `clean_view` extended by `prefix_ids`; return log-probs
    for each id in `token_ids` at the last position."""
    ids = torch.cat([clean_view['input_ids'][0],
                     torch.tensor(prefix_ids, dtype=torch.long,
                                  device=clean_view['input_ids'].device)],
                    dim=0).unsqueeze(0)
    kw = dict(input_ids=ids, attention_mask=torch.ones_like(ids))
    for k in ('pixel_values_images', 'pixel_values', 'image_grid_thw',
              'pixel_values_videos', 'video_grid_thw',
              'video_second_per_grid', 'input_features',
              'feature_attention_mask'):
        if k in clean_view:
            kw[k] = clean_view[k]
    logits = model.thinker(**kw).logits[0, -1].float()
    lp = torch.log_softmax(logits, dim=-1)
    return [lp[t].item() for t in token_ids]


# ===========================================================================
# SHARED OPERATORS: gamma / grounding functions
# ===========================================================================

# ---- PHD: single ablatable channel (text context) --------------------------

def gamma_extrapolated_phd(ly_full, ln_full, ly_cf, ln_cf, w=4.0, tau_p=-1.6):
    """PHD Stage-3 rule.

    margin' = margin_full + w*(margin_cf - margin_full)
    yes iff margin' > tau_p.

    w>1 extrapolates past the clean (context-stripped) view: PHD context is
    adversarial misinformation, so amplifying how far removing it moves the
    decision is the correct correction.
    Returns (margin_prime, forced_token).
    """
    margin_full = ly_full - ln_full
    margin_cf   = ly_cf  - ln_cf
    margin_prime = margin_full + w * (margin_cf - margin_full)
    return margin_prime, (YES_ID if margin_prime > tau_p else NO_ID)


# ---- CMM: two ablatable channels (audio, video) ----------------------------

_CMM_AUDIO_RE = re.compile(
    r'\bhear\b|\baudio\b|\bsound\b|\bnoise\b|\bmusic\b|'
    r'\bsaid\b|\bsay\b|\bspeak\b|\bspoke\b|\btalk\b|'
    r'\bvoice\b|\bsinging\b|\bsong\b', re.I)
_CMM_VIDEO_RE = re.compile(
    r'\bsee\b|\bseen\b|\bvideo\b|\bwatch\b|\bvisual\b|'
    r'\blook\b|\bshow\b|\bshown\b|\bappear\b|\bscreen\b|'
    r'\bcolou?r\b|\bframe\b', re.I)
_CMM_GROUP_TO_R = {
    'overrely_audio_ignore_visual': {'video'},
    'overrely_visual_ignore_audio': {'audio'},
}


def relevant_modalities_cmm(question=None, group_key=None):
    """Per-sample relevant modality set R for CMM: {'audio'}, {'video'}, or both."""
    if question:
        a = bool(_CMM_AUDIO_RE.search(question))
        v = bool(_CMM_VIDEO_RE.search(question))
        if a and not v: return {'audio'}, 'question'
        if v and not a: return {'video'}, 'question'
        if a and v:     return {'audio', 'video'}, 'question_both'
    if group_key in _CMM_GROUP_TO_R:
        return set(_CMM_GROUP_TO_R[group_key]), 'group_key'
    return {'audio', 'video'}, 'fallback_both'


def confidence_gated_flip(am, margin_full, flip_token, tau_conf):
    """Anchor to base when |margin_full| >= tau_conf; else take flip_token."""
    return am if abs(margin_full) >= tau_conf else flip_token


def grounded_yes_flip_cmm(am, margin_full, R, d_cf_audio, d_cf_video,
                           margin_conf=2.0, tau_m=-1.6):
    """CMM Stage-3 rule (dual-channel grounding gate).

    For am==YES_ID:
      margin_full >= margin_conf                          -> keep yes  (confident)
      margin_full <  margin_conf, grounded (depR<=tau_m)  -> keep yes
      margin_full <  margin_conf, not grounded            -> flip no
    base-no is never touched.
    """
    if am != YES_ID:
        return am
    if margin_full >= margin_conf:
        return YES_ID
    vals = []
    if 'audio' in R and d_cf_audio is not None: vals.append(d_cf_audio)
    if 'video' in R and d_cf_video is not None: vals.append(d_cf_video)
    if not vals:
        vals = [x for x in (d_cf_audio, d_cf_video) if x is not None]
    if not vals:
        return NO_ID
    return YES_ID if min(vals) <= tau_m else NO_ID


def gamma_margin_cmm(margin_full, margin_noaudio, margin_novideo, R):
    """CMM lambda-delta ablation: E_d^R = sum_{m in R} kappa^(m)."""
    er = 0.0
    if 'audio' in R: er += margin_full - margin_noaudio
    if 'video' in R: er += margin_full - margin_novideo
    return er, (YES_ID if er > 0 else NO_ID)


# ---- PubMedQA: passage is the relevant channel (3-class) -------------------

def gamma_scores_pubmedqa(ly_full, ln_full, lm_full, ly_cf, ln_cf, lm_cf):
    """Literal E^R rule: argmax_d kappa_d^(passage), kappa_d = lp_d_full - lp_d_cf."""
    scores = {YES_ID: ly_full - ly_cf, NO_ID: ln_full - ln_cf,
              MAYBE_ID: lm_full - lm_cf}
    return scores, max(scores, key=scores.get)


def gamma_anchored_pubmedqa(am, ly, ln, lm, ly_cf, ln_cf, lm_cf, tau_maybe=0.5):
    """PubMedQA Stage-3 rule (candidate-anchored + maybe gate).

    Keep base class am, EXCEPT when am==MAYBE and kappa_maybe = lm - lm_cf
    <= tau_maybe (passage does not ground 'maybe'): then redecide yes/no.
    """
    if am == MAYBE_ID and (lm - lm_cf) <= tau_maybe:
        return YES_ID if ly > ln else NO_ID
    return am


# ===========================================================================
# DATASET 1: PHD  (binary Y/N)
# ===========================================================================

def _build_view_phd(processor, model, system_prompt, pv, av, iv, user_prompt, ds='phd'):
    uc = build_user_content_with_media_tokens(pv, av, iv, user_prompt)
    return prepare_for_thinker(
        build_inputs(processor, model, make_messages(system_prompt, uc),
                     pv, av, iv, ds), model)


class PHDConfessProcessor(ThreeStageConfessProcessor):
    """PHD three-stage confession processor.

    Stage 2: teacher-force the context-removed view (image+question only).
    Stage 3: gamma extrapolation — margin' = margin_full + w*(margin_cf - margin_full).
             Ablation rule (--decision lambda): margin' = margin_full + lam*Delta_cf.
    """

    def __init__(self, model, tokenizer, cf_clean, prompt_len,
                 lam=2.0, control_rng=None, decision='gamma',
                 w_extrap=4.0, tau_p=-1.6):
        super().__init__(tokenizer, prompt_len)
        self.model       = model
        self.cf_clean    = cf_clean
        self.lam         = lam
        self.control_rng = control_rng
        self.decision    = decision
        self.w_extrap    = w_extrap  # extrapolation weight
        self.tau_p       = tau_p     # yes decision threshold on margin'

    def confess(self, prefix_ids):
        ly_cf, ln_cf = teacher_force_lp(self.model, self.cf_clean,
                                         prefix_ids, (YES_ID, NO_ID))
        return dict(ly_cf=ly_cf, ln_cf=ln_cf)

    def decide(self, full_view, cs):
        am = full_view['am']
        ly_full, ln_full = full_view['ly_full'], full_view['ln_full']
        ly_cf, ln_cf     = cs['ly_cf'], cs['ln_cf']
        delta = ly_cf - ly_full

        if self.decision == 'gamma' and self.control_rng is None:
            margin_prime, forced = gamma_extrapolated_phd(
                ly_full, ln_full, ly_cf, ln_cf,
                w=self.w_extrap, tau_p=self.tau_p)
            delta_eff = delta
        else:
            delta_eff = delta
            if self.control_rng is not None:
                delta_eff = delta * self.control_rng.choice([-1.0, 1.0])
            margin_prime = (ly_full - ln_full) + self.lam * delta_eff
            forced = YES_ID if margin_prime > 0 else NO_ID

        return forced, dict(
            dec_index=full_view['dec_index'],
            greedy_token=am, forced_token=forced,
            intervened=int(forced != am), decision=self.decision,
            lp_yes_full=round(ly_full, 4), lp_no_full=round(ln_full, 4),
            margin_full=round(full_view['margin_full'], 4),
            lp_yes_cf=round(ly_cf, 4), lp_no_cf=round(ln_cf, 4),
            delta_cf=round(delta, 4), delta_eff=round(delta_eff, 4),
            margin_adj=round(margin_prime, 4),
            w_extrap=self.w_extrap, tau_p=self.tau_p,
        )


def run_phd(args):
    set_seed(args.seed)
    control_rng = random.Random(args.seed + 7) if args.control == 'random_sign' else None
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.time_path, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.file_name + '.jsonl')
    done_ids = load_done_ids(output_path)

    print('[model] loading Qwen2.5-Omni...', flush=True)
    processor, model = load_model()
    tokenizer = processor.tokenizer
    data = load_dataset(args.data_path, args.data_root)
    if args.dry_run: data = data[:3]
    system_prompt = get_system_prompt(args.datasets, args.prompt_version)

    all_out_tok, t_start = 0, time.time()
    for idx, item in enumerate(data):
        if item['id'] in done_ids: continue
        t0 = time.time()
        print(f'\n[{idx+1}/{len(data)}] id={item["id"]}', flush=True)
        rec_extra, gen_ids, logp_list, input_len = {}, None, None, None
        try:
            pv, av, iv = load_media_for_item(item, args.data_path, args.data_root)
            q_full, q_only = split_question(item['question'])
            u_full = get_user_prompt(q_full, args.datasets, args.prompt_version)
            clean_f = _build_view_phd(processor, model, system_prompt,
                                       pv, av, iv, u_full)
            input_len = clean_f['input_ids'].shape[1]
            u_cf = get_user_prompt(q_only, args.datasets, args.prompt_version)
            clean_cf = _build_view_phd(processor, model, system_prompt,
                                        pv, av, iv, u_cf)
            cf_proc = PHDConfessProcessor(model, tokenizer, clean_cf, input_len,
                                           args.lambda_cf, control_rng,
                                           decision=args.decision,
                                           w_extrap=args.w_extrap,
                                           tau_p=args.tau_p)
            with torch.inference_mode():
                out = model.thinker.generate(
                    **clean_f, max_new_tokens=args.max_new_tokens,
                    do_sample=False, pad_token_id=tokenizer.eos_token_id,
                    output_scores=True, return_dict_in_generate=True,
                    logits_processor=[cf_proc])
            gen_ids = out.sequences[0][input_len:].cpu()
            if args.record_process: logp_list = token_logprobs(gen_ids, out.scores)
            response = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            rec_extra = cf_proc.record or dict(dec_index=None, intervened=0)
            rec_extra.update(lambda_cf=args.lambda_cf, control=args.control,
                              decision=args.decision)
            all_out_tok += len(gen_ids)
            print(f'  [{time.time()-t0:.1f}s] interv={rec_extra.get("intervened")}',
                  flush=True)
            del clean_f, clean_cf
        except Exception as e:
            import traceback; traceback.print_exc()
            response = ''
            rec_extra = dict(dec_index=None, intervened=0, error=str(e)[:200],
                              lambda_cf=args.lambda_cf, control=args.control,
                              decision=args.decision)
        torch.cuda.empty_cache()
        method = ('cf_confess_gamma' if args.decision == 'gamma'
                  else f'cf_confess_lam{args.lambda_cf}')
        result = make_result(item, response, args.datasets,
                              method + ('_randctl' if args.control != 'none' else ''),
                              time_sec=time.time() - t0)
        attach_record(result, rec_extra, gen_ids, logp_list, input_len,
                       args.record_process)
        save_result(result, output_path)

    write_time(args.time_path, args.file_name, t_start, all_out_tok)
    print(f'\n[done] phd  output={output_path}', flush=True)


# ===========================================================================
# DATASET 2: CMM  (binary Y/N)
# ===========================================================================

def _build_view_cmm(processor, model, system_prompt, pv, av, iv, user_prompt):
    uc = build_user_content_with_media_tokens(pv, av, iv, user_prompt)
    return prepare_for_thinker(
        build_inputs(processor, model, make_messages(system_prompt, uc),
                     pv, av, iv, 'cmm'), model)


class CMMConfessProcessor(ThreeStageConfessProcessor):
    """CMM three-stage confession processor.

    Stage 2: both masked-view teacher-force forwards, always-on.
        d_cf_i = lp_yes(mask-i view) - lp_yes(full view)  for i in {audio, video}
    Stage 3: dual-channel grounding gate, or lambda-delta (ablation).
    """

    def __init__(self, model, tokenizer, cf_noaudio, cf_novideo, prompt_len,
                 lam=2.0, w_audio=1.0, w_video=1.0, control_rng=None,
                 question=None, group_key=None, gate_tau=2.0,
                 decision='grounded', margin_conf=2.0, tau_m=-1.6):
        super().__init__(tokenizer, prompt_len)
        self.model       = model
        self.cf_noaudio  = cf_noaudio
        self.cf_novideo  = cf_novideo
        self.lam         = lam
        self.w_audio     = w_audio
        self.w_video     = w_video
        self.control_rng = control_rng
        self.gate_tau    = gate_tau
        self.decision    = decision
        self.margin_conf = margin_conf  # confidence gate threshold
        self.tau_m       = tau_m        # grounding threshold
        self.R, self.R_source = relevant_modalities_cmm(question, group_key)

    def confess(self, prefix_ids):
        ya_na = teacher_force_lp(self.model, self.cf_noaudio, prefix_ids, (YES_ID,))[0]
        ya_nv = teacher_force_lp(self.model, self.cf_novideo, prefix_ids, (YES_ID,))[0]
        na_na = teacher_force_lp(self.model, self.cf_noaudio, prefix_ids, (NO_ID,))[0]
        na_nv = teacher_force_lp(self.model, self.cf_novideo, prefix_ids, (NO_ID,))[0]
        return dict(ly_noaudio=ya_na, ly_novideo=ya_nv,
                    ln_noaudio=na_na, ln_novideo=na_nv)

    def decide(self, full_view, cs):
        am         = full_view['am']
        ly_full    = full_view['ly_full']
        ln_full    = full_view['ln_full']
        margin_full = full_view['margin_full']
        dec_index  = full_view['dec_index']
        d_audio    = cs['ly_noaudio'] - ly_full
        d_video    = cs['ly_novideo'] - ly_full

        if self.decision == 'grounded' and self.control_rng is None:
            forced = grounded_yes_flip_cmm(am, margin_full, self.R,
                                            d_audio, d_video,
                                            self.margin_conf, self.tau_m)
            return forced, dict(
                dec_index=dec_index,
                greedy_token=am, forced_token=forced,
                intervened=int(forced != am), decision='grounded',
                lp_yes_full=round(ly_full, 4), lp_no_full=round(ln_full, 4),
                margin_full=round(margin_full, 4),
                d_cf_audio=round(d_audio, 4), d_cf_video=round(d_video, 4),
                R=sorted(self.R), R_source=self.R_source,
                channel_computed=1,
                margin_conf=self.margin_conf, tau_m=self.tau_m,
            )

        # lambda-delta ablation (and random_sign control)
        delta = 0.0
        if 'audio' in self.R: delta += self.w_audio * d_audio
        if 'video' in self.R: delta += self.w_video * d_video
        delta_eff = delta
        if self.control_rng is not None:
            delta_eff = delta * self.control_rng.choice([-1.0, 1.0])
        margin_adj = margin_full + self.lam * delta_eff
        flip   = YES_ID if margin_adj > 0 else NO_ID
        forced = confidence_gated_flip(am, margin_full, flip, self.gate_tau)
        return forced, dict(
            dec_index=dec_index,
            greedy_token=am, forced_token=forced,
            intervened=int(forced != am), decision='lambda_delta',
            lp_yes_full=round(ly_full, 4), lp_no_full=round(ln_full, 4),
            margin_full=round(margin_full, 4),
            d_cf_audio=round(d_audio, 4), d_cf_video=round(d_video, 4),
            w_audio=self.w_audio, w_video=self.w_video,
            R=sorted(self.R), R_source=self.R_source,
            channel_computed=1,
            delta=round(delta, 4), delta_eff=round(delta_eff, 4),
            margin_adj=round(margin_adj, 4), gate_tau=self.gate_tau,
        )


def run_cmm(args):
    set_seed(args.seed)
    control_rng = random.Random(args.seed + 7) if args.control == 'random_sign' else None
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.time_path, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.file_name + '.jsonl')
    done_ids = load_done_ids(output_path)

    print('[model] loading Qwen2.5-Omni...', flush=True)
    processor, model = load_model()
    tokenizer = processor.tokenizer
    data = load_dataset(args.data_path, args.data_root)
    if args.dry_run: data = data[:3]
    system_prompt = get_system_prompt(args.datasets, args.prompt_version)

    all_out_tok, t_start = 0, time.time()
    for idx, item in enumerate(data):
        if item['id'] in done_ids: continue
        t0 = time.time()
        print(f'\n[{idx+1}/{len(data)}] id={item["id"]}', flush=True)
        rec_extra, gen_ids, logp_list, input_len = {}, None, None, None
        try:
            pv, av, iv = load_media_for_item(item, args.data_path, args.data_root)
            u = get_user_prompt(item['question'], args.datasets, args.prompt_version)
            clean_f = _build_view_cmm(processor, model, system_prompt, pv, av, iv, u)
            input_len = clean_f['input_ids'].shape[1]
            clean_a = _build_view_cmm(processor, model, system_prompt, pv, None, iv, u)
            clean_v = _build_view_cmm(processor, model, system_prompt, None, av, iv, u)
            cf_proc = CMMConfessProcessor(
                model, tokenizer, clean_a, clean_v, input_len,
                args.lambda_cf, args.w_audio, args.w_video, control_rng,
                question=item.get('question', ''),
                group_key=item.get('group_key', ''),
                gate_tau=args.gate_tau,
                decision=args.decision,
                margin_conf=args.margin_conf,
                tau_m=args.tau_m)
            with torch.inference_mode():
                out = model.thinker.generate(
                    **clean_f, max_new_tokens=args.max_new_tokens,
                    do_sample=False, pad_token_id=tokenizer.eos_token_id,
                    output_scores=True, return_dict_in_generate=True,
                    logits_processor=[cf_proc])
            gen_ids = out.sequences[0][input_len:].cpu()
            if args.record_process: logp_list = token_logprobs(gen_ids, out.scores)
            response = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            rec_extra = cf_proc.record or dict(dec_index=None, intervened=0)
            rec_extra.update(lambda_cf=args.lambda_cf, control=args.control)
            all_out_tok += len(gen_ids)
            print(f'  [{time.time()-t0:.1f}s] interv={rec_extra.get("intervened")}',
                  flush=True)
            del clean_f, clean_a, clean_v
        except Exception as e:
            import traceback; traceback.print_exc()
            response = ''
            rec_extra = dict(dec_index=None, intervened=0, error=str(e)[:200],
                              lambda_cf=args.lambda_cf, control=args.control)
        torch.cuda.empty_cache()
        result = make_result(item, response, args.datasets,
                              f'cf_confess_lam{args.lambda_cf}'
                              + ('_randctl' if args.control != 'none' else ''),
                              time_sec=time.time() - t0)
        attach_record(result, rec_extra, gen_ids, logp_list, input_len,
                       args.record_process)
        save_result(result, output_path)

    write_time(args.time_path, args.file_name, t_start, all_out_tok)
    print(f'\n[done] cmm   output={output_path}', flush=True)


# ===========================================================================
# DATASET 3: PubMedQA  (3-class Y/N/Maybe)
# ===========================================================================

def _build_view_pubmedqa(processor, model, system_prompt, user_prompt):
    uc = build_user_content_with_media_tokens(None, None, None, user_prompt)
    return prepare_for_thinker(
        build_inputs(processor, model, make_messages(system_prompt, uc),
                     None, None, None, 'pubmedqa'), model)


def _strip_passage_line(user_prompt):
    lines = user_prompt.split('\n')
    if lines and lines[0].startswith('Passage:'):
        return '\n'.join(lines[1:])
    return user_prompt


class PubMedQAConfessProcessor(LogitsProcessor):
    """PubMedQA three-stage confession processor.

    Stage 1: surrounding greedy generate() loop; decision fires at step 0
             (bare Yes/No/Maybe as the very first token).
    Stage 2: teacher-force the passage-removed view on the same prefix;
             read {Yes, No, Maybe} log-probs (the parametric prior P^0).
    Stage 3: candidate-anchored gate — keep base answer am, EXCEPT when
             am==MAYBE and kappa_maybe = lm_full - lm_cf <= tau_maybe.
    """

    def __init__(self, model, tokenizer, cf_clean, prompt_len, lam=2.0,
                 control_rng=None, decision='gamma', gate_tau_maybe=0.5):
        self.model          = model
        self.tokenizer      = tokenizer
        self.cf_clean       = cf_clean
        self.prompt_len     = prompt_len
        self.lam            = lam
        self.control_rng    = control_rng
        self.decision       = decision
        self.gate_tau_maybe = gate_tau_maybe
        self.used           = False
        self.record         = None

    @torch.inference_mode()
    def _confess_lps(self, prefix_ids):
        """Stage 2: teacher-force passage-removed view, return (ly, ln, lm)."""
        ids = torch.cat([self.cf_clean['input_ids'][0],
                         torch.tensor(prefix_ids, dtype=torch.long,
                                      device=self.cf_clean['input_ids'].device)],
                        dim=0).unsqueeze(0)
        kw = dict(input_ids=ids, attention_mask=torch.ones_like(ids))
        for k in ('pixel_values_images', 'pixel_values', 'image_grid_thw',
                  'pixel_values_videos', 'video_grid_thw', 'video_second_per_grid',
                  'input_features', 'feature_attention_mask'):
            if k in self.cf_clean: kw[k] = self.cf_clean[k]
        logits = self.model.thinker(**kw).logits[0, -1].float()
        lp = torch.log_softmax(logits, dim=-1)
        return lp[YES_ID].item(), lp[NO_ID].item(), lp[MAYBE_ID].item()

    def _decide(self, am, ly, ln, lm, ly_cf, ln_cf, lm_cf):
        d_cf = ly_cf - ly
        if self.decision == 'gamma' and self.control_rng is None:
            forced = gamma_anchored_pubmedqa(am, ly, ln, lm, ly_cf, ln_cf, lm_cf,
                                              self.gate_tau_maybe)
            return forced, ly, d_cf
        d_eff = d_cf
        if self.control_rng is not None:
            d_eff = d_cf * self.control_rng.choice([-1.0, 1.0])
        yes_adj = ly + self.lam * d_eff
        choices = {YES_ID: yes_adj, NO_ID: ln, MAYBE_ID: lm}
        return max(choices, key=choices.get), yes_adj, d_eff

    @torch.inference_mode()
    def __call__(self, input_ids, scores):
        if self.used:
            return scores
        gen = input_ids[0, self.prompt_len:].tolist()
        if gen:
            return scores
        logits = scores[0].float()
        am = int(logits.argmax().item())
        if am not in (YES_ID, NO_ID, MAYBE_ID):
            return scores
        lp = torch.log_softmax(logits, dim=-1)
        ly, ln, lm = lp[YES_ID].item(), lp[NO_ID].item(), lp[MAYBE_ID].item()
        ly_cf, ln_cf, lm_cf = self._confess_lps(gen)
        forced, yes_adj, d_eff = self._decide(am, ly, ln, lm, ly_cf, ln_cf, lm_cf)
        scores[0][:] = float('-inf')
        scores[0][forced] = 0.0
        self.used = True
        self.record = dict(
            dec_index=0,
            greedy_token=am, forced_token=forced,
            intervened=int(forced != am), decision=self.decision,
            lp_yes_full=round(ly, 4), lp_no_full=round(ln, 4),
            lp_maybe_full=round(lm, 4),
            lp_yes_cf=round(ly_cf, 4), lp_no_cf=round(ln_cf, 4),
            lp_maybe_cf=round(lm_cf, 4),
            d_cf=round(ly_cf - ly, 4), d_eff=round(d_eff, 4),
            yes_adj=round(yes_adj, 4),
        )
        return scores


def run_pubmedqa(args):
    set_seed(args.seed)
    control_rng = random.Random(args.seed + 7) if args.control == 'random_sign' else None
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.time_path, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.file_name + '.jsonl')
    done_ids = load_done_ids(output_path)

    print('[model] loading Qwen2.5-Omni...', flush=True)
    processor, model = load_model()
    tokenizer = processor.tokenizer
    data = load_dataset(args.data_path, args.data_root)
    if args.dry_run: data = data[:3]
    system_prompt = get_system_prompt(args.datasets, args.prompt_version)

    all_out_tok, t_start = 0, time.time()
    for idx, item in enumerate(data):
        if item['id'] in done_ids: continue
        t0 = time.time()
        print(f'\n[{idx+1}/{len(data)}] id={item["id"]}', flush=True)
        rec_extra, gen_ids, logp_list, input_len = {}, None, None, None
        try:
            u_full = get_user_prompt(item['question'], args.datasets,
                                      args.prompt_version, passage=item.get('passage'))
            clean_f = _build_view_pubmedqa(processor, model, system_prompt, u_full)
            input_len = clean_f['input_ids'].shape[1]
            u_cf = _strip_passage_line(u_full)
            clean_c = _build_view_pubmedqa(processor, model, system_prompt, u_cf)
            cf_proc = PubMedQAConfessProcessor(model, tokenizer, clean_c, input_len,
                                                args.lambda_cf, control_rng,
                                                decision=args.decision,
                                                gate_tau_maybe=args.gate_tau_maybe)
            with torch.inference_mode():
                out = model.thinker.generate(
                    **clean_f, max_new_tokens=args.max_new_tokens,
                    do_sample=False, pad_token_id=tokenizer.eos_token_id,
                    output_scores=True, return_dict_in_generate=True,
                    logits_processor=[cf_proc])
            gen_ids = out.sequences[0][input_len:].cpu()
            if args.record_process: logp_list = token_logprobs(gen_ids, out.scores)
            response = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            rec_extra = cf_proc.record or dict(dec_index=None, intervened=0)
            rec_extra.update(lambda_cf=args.lambda_cf, control=args.control,
                              decision=args.decision)
            all_out_tok += len(gen_ids)
            print(f'  [{time.time()-t0:.1f}s] interv={rec_extra.get("intervened")}',
                  flush=True)
            del clean_f, clean_c
        except Exception as e:
            import traceback; traceback.print_exc()
            response = ''
            rec_extra = dict(dec_index=None, intervened=0, error=str(e)[:200],
                              lambda_cf=args.lambda_cf, control=args.control,
                              decision=args.decision)
        torch.cuda.empty_cache()
        method = ('cf_confess_gamma' if args.decision == 'gamma'
                  else f'cf_confess_lam{args.lambda_cf}')
        result = make_result(item, response, args.datasets,
                              method + ('_randctl' if args.control != 'none' else ''),
                              time_sec=time.time() - t0)
        attach_record(result, rec_extra, gen_ids, logp_list, input_len,
                       args.record_process)
        save_result(result, output_path)

    write_time(args.time_path, args.file_name, t_start, all_out_tok)
    print(f'\n[done] pubmedqa  output={output_path}', flush=True)


# ===========================================================================
# DATASET 4: RAGTruth  (free-form generation)
# ===========================================================================

_RAGTRUTH_STOPWORDS = set("""
a an the and or but if because as what which this that these those then just so
than such both through about for is are was were of while during to in on at by
with it there
""".split())
_RAGTRUTH_FUNC = _RAGTRUTH_STOPWORDS | set("""
i me my myself we us our ours ourselves you your yours yourself yourselves he him
his himself she her hers herself it its itself they them their theirs themselves
what which who whom this that these those am is are was were being been have has
had having do does did doing would should could ought will shall may might must
can cannot not no nor as if because until while of at by for with about against
between into through during before after above below to from up down in out on off
over under again further then once here there when where why how all any both each
few more most other some such only own same s t don now d ll m o re ve y ain
aren couldn didn doesn hadn hasn haven isn ma mightn mustn needn shan shouldn wasn
weren won wouldn also however therefore thus furthermore moreover additionally
another overall
""".split())
_RAGTRUTH_SPECIAL = re.compile(r'<\|.*?\|>')


def _ragtruth_is_word_initial(tok_str, prev_tok):
    if tok_str[:1].isspace():
        return True
    if prev_tok is None:
        return True
    return not (prev_tok[-1] if prev_tok else ' ').isalnum()


def _ragtruth_content_class(tok_str, prev_tok):
    if _RAGTRUTH_SPECIAL.search(tok_str):
        return 'skip'
    s = tok_str.strip()
    if not s or not _ragtruth_is_word_initial(tok_str, prev_tok):
        return 'skip'
    w = s.lower().strip(string.punctuation + ' ')
    if not w or w in _RAGTRUTH_FUNC:
        return 'skip'
    if w.isdigit() and len(w) == 1:
        return 'skip'
    if not (w[0].isalpha() or w[0].isdigit()):
        return 'skip'
    return 'content'


def _ragtruth_token_logprobs(gen_ids, scores):
    out = []
    for tid, logits in zip(gen_ids, scores):
        logits = logits.float()
        if logits.dim() > 1:
            logits = logits[0]
        lp = torch.log_softmax(logits, dim=-1)[tid]
        out.append(round(lp.item(), 4))
    return out


@torch.inference_mode()
def _ragtruth_cf_generate(thinker, tokenizer, clean_full, clean_np, args):
    """RAGTruth three-stage per-content-word loop.

    Stage 1: full-view greedy token.
    Stage 2: lazy-sync no-passage KV cache, read Delta_w = logp_full - logp_np.
    Stage 3: temperate re-rank — lp_full + lambda*max(Delta,0) — guarded by
             lead-in protection, signal floor, and fluency floor.
    """
    device = clean_full['input_ids'].device
    eps, lam, topk = args.eps, args.lambda_cf, args.topk

    out_f = thinker(input_ids=clean_full['input_ids'],
                    attention_mask=torch.ones_like(clean_full['input_ids']),
                    use_cache=True)
    past_f = out_f.past_key_values
    logits_f = out_f.logits[0, -1].float()

    out_n = thinker(input_ids=clean_np['input_ids'],
                    attention_mask=torch.ones_like(clean_np['input_ids']),
                    use_cache=True)
    past_n = out_n.past_key_values

    eos = tokenizer.eos_token_id
    gen_ids, scores, prev_tok = [], [], None
    np_fed, content_seen = 0, 0
    stats = {'content_pos': 0, 'rootless_greedy': 0, 'intervened': 0,
             'changed': 0, 'vetoed': 0, 'np_forwards': 0,
             'veto_lead': 0, 'veto_fluency': 0, 'veto_gap': 0}

    for step in range(args.max_new_tokens):
        lp_full = torch.log_softmax(logits_f, dim=-1)
        w = int(torch.argmax(logits_f).item())
        chosen = w
        tok_str = tokenizer.decode([w])

        if _ragtruth_content_class(tok_str, prev_tok) == 'content':
            stats['content_pos'] += 1
            content_seen += 1
            if np_fed < step:
                chunk = torch.tensor([gen_ids[np_fed:step]], device=device)
                out_n = thinker(input_ids=chunk, past_key_values=past_n,
                                use_cache=True)
                past_n = out_n.past_key_values
                np_fed = step
                stats['np_forwards'] += 1
            lp_np = torch.log_softmax(out_n.logits[0, -1].float(), dim=-1)
            delta_w = float(lp_full[w] - lp_np[w])

            trigger = (random.random() < args.random_rate
                       if args.random_control else delta_w < eps)
            if trigger:
                if not args.random_control:
                    stats['rootless_greedy'] += 1
                cand = torch.topk(logits_f, topk).indices
                lp_f_c = lp_full[cand]
                lp_n_c = lp_np[cand]
                score = lp_f_c + lam * torch.clamp(lp_f_c - lp_n_c, min=0.0)
                j = int(torch.argmax(score).item())
                cand_tok = int(cand[j].item())
                stats['intervened'] += 1
                if content_seen <= args.lead_protect:
                    chosen = w; stats['vetoed'] += 1; stats['veto_lead'] += 1
                elif delta_w > args.sig_floor:
                    chosen = w; stats['vetoed'] += 1; stats['veto_gap'] += 1
                else:
                    fluency_drop = float(lp_full[w] - lp_f_c[j])
                    if fluency_drop > args.fluency_drop:
                        chosen = w; stats['vetoed'] += 1; stats['veto_fluency'] += 1
                    else:
                        chosen = cand_tok
                        if chosen != w: stats['changed'] += 1

        gen_ids.append(chosen)
        scores.append(logits_f.cpu())
        if chosen == eos:
            break
        prev_tok = tokenizer.decode([chosen])
        out_f = thinker(input_ids=torch.tensor([[chosen]], device=device),
                        past_key_values=past_f, use_cache=True)
        past_f = out_f.past_key_values
        logits_f = out_f.logits[0, -1].float()

    return {'gen_ids': gen_ids, 'scores': scores, 'stats': stats}


def run_ragtruth(args):
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.time_path, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.file_name + '.jsonl')

    done_ids = set()
    if os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                try: done_ids.add(json.loads(line.strip())['id'])
                except Exception: pass

    print('[model] loading Qwen2.5-Omni...', flush=True)
    processor, model = load_model()
    tokenizer = processor.tokenizer
    data = load_dataset(args.data_path,
                        os.path.dirname(os.path.abspath(args.data_path)))
    if args.ids_file:
        keep = set(json.load(open(args.ids_file)))
        data = [it for it in data if str(it['id']) in keep]
    if args.dry_run: data = data[:2]
    system_prompt = get_system_prompt(args.datasets, args.prompt_version)

    all_out_tok, t_start = 0, time.time()
    for idx, item in enumerate(data):
        if str(item['id']) in done_ids: continue
        t0 = time.time()
        print(f'\n[{idx+1}/{len(data)}] id={item["id"]}', flush=True)
        passage = item.get('passage') or item.get('context') or ''
        gen_ids, stats = None, None
        try:
            views = {}
            for name, psg in (('full', passage), ('nopsg', '')):
                up = get_user_prompt(item['question'], args.datasets,
                                      args.prompt_version, passage=psg)
                raw = build_inputs(processor, model,
                                    make_messages(system_prompt, up),
                                    None, None, None, args.datasets)
                views[name] = prepare_for_thinker(raw, model)
            out = _ragtruth_cf_generate(model.thinker, tokenizer,
                                         views['full'], views['nopsg'], args)
            gen_ids, stats = out['gen_ids'], out['stats']
            print(f'  [{time.time()-t0:.1f}s] changed={stats["changed"]}/'
                  f'{stats["content_pos"]}', flush=True)
        except Exception as e:
            import traceback; traceback.print_exc()
        torch.cuda.empty_cache()
        response = (tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                    if gen_ids is not None else '')
        if gen_ids is not None: all_out_tok += len(gen_ids)
        result = make_result(item, response, args.datasets,
                              'cf_confess' + ('_rand' if args.random_control else ''),
                              time_sec=time.time() - t0)
        if args.record_process and gen_ids is not None:
            logp_list = _ragtruth_token_logprobs(gen_ids, out['scores'])
            cum = float(np.sum(logp_list))
            result.update({
                'token_ids': gen_ids, 'logp_list': logp_list,
                'cumulative_logprob': round(cum, 4),
                'norm_logp': round(cum / (len(gen_ids) + 1e-8), 4),
                'input_len': int(views['full']['input_ids'].shape[1]),
                'output_tokens': len(gen_ids),
                'cf_stats': stats,
                'cf_config': {'lambda_cf': args.lambda_cf, 'eps': args.eps,
                              'topk': args.topk, 'lead_protect': args.lead_protect,
                              'sig_floor': args.sig_floor,
                              'fluency_drop': args.fluency_drop},
            })
        save_result(result, output_path)

    total = time.time() - t_start
    with open(os.path.join(args.time_path, args.file_name + '.txt'), 'w') as f:
        f.write(f'total_time: {total:.2f}s\ntotal_output_tokens: {all_out_tok}\n')
    print(f'\n[done] ragtruth  output={output_path}', flush=True)


# ===========================================================================
# DATASET 5: HaloQuest  (free-form generation)
# ===========================================================================
# Same three-stage confession narrative; only the Stage-3 output type differs
# (corrected free text, not a forced token):
#
# Stage 1  Generate   the engineering-prompt base answer.
# Stage 2  Confess     delta_img_tok over the committed answer tokens
#                      (lp_full - lp_noimg), always-on. Low delta = the token was
#                      driven by text priors rather than the image (ungrounded).
# Stage 3  Decide      bucket-routed span correction on item['sub_category']:
#                        visual challenge -> delta-driven directed regeneration
#                                            of the lowest-delta span.
#                        false premises   -> route to the fp candidate pool.
#                        insufficient ctx -> keep base.
#
# The authoritative implementation lives in cf_confess_haloquest.py
# (HaloQuestConfessDriver subclassing FreeformThreeStageDriver); this entry is a
# thin wrapper so all six datasets run from one CLI.


def run_haloquest(args):
    """HaloQuest free-text three-stage arm (Stage 1 generate -> Stage 2 confess
    delta_img_tok -> Stage 3 bucket-routed span correction). Delegates to the
    authoritative pipeline in cf_confess_haloquest.run_haloquest_pipeline so the
    unified CLI and the standalone module run the same logic."""
    from cf_confess_haloquest import run_haloquest_pipeline
    run_haloquest_pipeline(
        shard=args.halo_shard, nshards=args.halo_nshards,
        outdir=args.output_dir, fp_mode=args.halo_fp_mode,
        overlap_threshold=args.halo_overlap, regen_max_tokens=args.halo_regen_tokens,
        dry_run=args.dry_run,
        file_name=(args.file_name if args.file_name != 'omniconfess_out' else None),
    )


# ===========================================================================
# DATASET 6: AVHBench  (free-form generation)
# ===========================================================================
# Stage 1: generate the same question under four visual configurations
#          (frame size / num-frames); each is a separate confession view.
# Stage 2: cross-arm disagreement is the confession signal — a content word
#          only a subset of arms produces is less grounded than one all agree on.
# Stage 3: majority vote-SC — content words present in >= threshold arms win.

_AVH_STOP = set('''
a an the and or of to in on at as is are was were be been being it its this that
these those with while by for from into near nearby there here his her their our
your my he she they we you i who which what when where how not no nor but so then
than also very just only one two single describe see hear seen heard video audio
image picture clip scene shows show can cannot could would should may might
'''.split())
_AVH_PUNCT = "!\"#$%&'()*+,-./:;<=>?@[]^_`{|}~\\"


def _avh_norm_gt(s):
    return re.sub(r'\s+', ' ', str(s).strip().lower())


def _avh_clean_boxed(s):
    m = re.search(r'\\boxed\{(.*)\}', str(s), re.S)
    return (m.group(1) if m else str(s)).strip()


def _avh_content_words(s):
    s = _avh_clean_boxed(s).lower().translate(str.maketrans('', '', _AVH_PUNCT))
    return [w for w in s.split() if w not in _AVH_STOP and len(w) > 1]


def _avh_vote_build(arms, keys, armlist, threshold, out_path, method='vote_sc'):
    """Build voted output: keep content words present in >= threshold arms."""
    rows = []
    for key in keys:
        arm_preds = [arms[k][key]['pred'] for k in armlist if key in arms[k]]
        if not arm_preds:
            continue
        word_counts = {}
        for pred in arm_preds:
            for w in _avh_content_words(pred):
                word_counts[w] = word_counts.get(w, 0) + 1
        voted = {w for w, c in word_counts.items() if c >= threshold}
        # Use arm0's sentence as the skeleton for phrasing
        skeleton = arm_preds[0]
        out_words, seen = [], set()
        for tok in skeleton.split():
            w = tok.lower().translate(str.maketrans('', '', _AVH_PUNCT))
            if not w: continue
            if w in _AVH_STOP or len(w) <= 1:
                out_words.append(tok)
            elif w in voted:
                out_words.append(tok)
                seen.add(w)
        missing = [w for w in voted if w not in seen]
        s = ' '.join(out_words + missing)
        s = re.sub(r'\s+([,.!?;:])', r'\1', s)
        s = re.sub(r'\s+', ' ', s).strip()
        if s and s[0].islower(): s = s[0].upper() + s[1:]
        if s and s[-1] not in '.!?': s += '.'
        ref = arms[armlist[0]][key]
        rows.append({'id': ref.get('id', key), 'question': ref.get('question', ''),
                     'gt': ref.get('gt', ''), 'dataset': 'avhbench',
                     'method': method, 'pred': s})
    with open(out_path, 'w') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'[done] avhbench vote_sc  output={out_path} ({len(rows)} rows)', flush=True)


def run_avhbench_vote(args):
    """Stage-3 vote-SC for AVHBench: read pre-generated arm files, write output.

    Required args: arm_paths (comma-separated list of 4 JSONL paths),
                   output_dir, file_name.
    """
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, args.file_name + '.jsonl')
    arm_paths = args.arm_paths.split(',')
    arms, armlist = {}, []
    for k, p in enumerate(arm_paths):
        p = p.strip()
        arm_key = str(k)
        recs = [json.loads(l) for l in open(p)]
        arms[arm_key] = {_avh_norm_gt(r.get('gt', '') + r.get('question', '')): r
                         for r in recs}
        armlist.append(arm_key)
    keys = sorted(set.intersection(*[set(a.keys()) for a in arms.values()]))
    print(f'[vote] {len(arm_paths)} arms, {len(keys)} aligned samples,'
          f' threshold={args.vote_threshold}', flush=True)
    _avh_vote_build(arms, keys, armlist, args.vote_threshold, out_path)


# ===========================================================================
# CLI ROUTER
# ===========================================================================

def _add_common_args(ap):
    ap.add_argument('--data_path',   type=str, required=False)
    ap.add_argument('--data_root',   type=str, default='./OmniHalluBench/')
    ap.add_argument('--output_dir',  type=str, default='./outputs/omniconfess')
    ap.add_argument('--file_name',   type=str, default='omniconfess_out')
    ap.add_argument('--time_path',   type=str, default='./outputs/omniconfess/time')
    ap.add_argument('--seed',        type=int, default=42)
    ap.add_argument('--max_new_tokens', type=int, default=300)
    ap.add_argument('--prompt_version', type=str, default='v1')
    ap.add_argument('--lambda_cf',   type=float, default=2.0)
    ap.add_argument('--control', choices=['none', 'random_sign'], default='none')
    ap.add_argument('--record_process',
                    action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument('--dry_run', action='store_true')


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--datasets', type=str, required=True,
                    help='phd | cmm | pubmedqa | ragtruth | haloquest | avhbench')
    _add_common_args(ap)

    # ---- PHD ----------------------------------------------------------------
    ap.add_argument('--decision',
                    choices=['gamma', 'lambda',          # phd / pubmedqa
                             'grounded', 'lambda_delta', # cmm
                             'rerank', 'abstain'],       # ragtruth
                    default='gamma',
                    help='Stage-3 rule (dataset-specific; see --datasets help).')
    ap.add_argument('--w_extrap', type=float, default=4.0,
                    help='PHD extrapolation weight w.')
    ap.add_argument('--tau_p', type=float, default=-1.6,
                    help='PHD yes decision threshold on margin\'.')

    # ---- CMM ----------------------------------------------------------------
    ap.add_argument('--w_audio',    type=float, default=1.0)
    ap.add_argument('--w_video',    type=float, default=1.0)
    ap.add_argument('--gate_tau',   type=float, default=2.0,
                    help='CMM lambda-delta confidence gate (ablation only).')
    ap.add_argument('--margin_conf', type=float, default=2.0,
                    help='CMM confidence gate: keep base-yes unconditionally '
                         'when margin_full >= margin_conf.')
    ap.add_argument('--tau_m', type=float, default=-1.6,
                    help='CMM grounding threshold.')

    # ---- PubMedQA -----------------------------------------------------------
    ap.add_argument('--gate_tau_maybe', type=float, default=0.5,
                    help='PubMedQA maybe-gate threshold.')

    # ---- RAGTruth -----------------------------------------------------------
    ap.add_argument('--eps',  type=float, default=-0.05,
                    help='RAGTruth rootless threshold (Delta_w < eps triggers).')
    ap.add_argument('--topk', type=int, default=10,
                    help='RAGTruth re-rank candidate pool size.')
    ap.add_argument('--lead_protect', type=int, default=4,
                    help='RAGTruth: guard first N content words from edits.')
    ap.add_argument('--sig_floor', type=float, default=-0.5,
                    help='RAGTruth: require Delta_w <= sig_floor to edit.')
    ap.add_argument('--fluency_drop', type=float, default=0.5,
                    help='RAGTruth: max logp drop to accept a swap.')
    ap.add_argument('--random_control', action='store_true',
                    help='RAGTruth: random-position control arm.')
    ap.add_argument('--random_rate', type=float, default=0.3)
    ap.add_argument('--ids_file', type=str, default='',
                    help='RAGTruth: restrict to ids in this JSON list file.')

    # ---- HaloQuest ----------------------------------------------------------
    ap.add_argument('--halo_shard',   type=int, default=0,
                    help='HaloQuest: shard index for multi-GPU sharding.')
    ap.add_argument('--halo_nshards', type=int, default=8,
                    help='HaloQuest: number of shards.')
    ap.add_argument('--halo_fp_mode',
                    choices=['selective2', 'delta_regen', 'premise_check'],
                    default='selective2',
                    help='HaloQuest FP-bucket Stage-3 source. selective2 is the '
                         'default; delta_regen and premise_check are ablations.')
    ap.add_argument('--halo_overlap', type=float, default=0.10,
                    help='HaloQuest accept-guard word-overlap floor.')
    ap.add_argument('--halo_regen_tokens', type=int, default=64,
                    help='HaloQuest directed-regen max new tokens.')

    # ---- AVHBench -----------------------------------------------------------
    ap.add_argument('--arm_paths', type=str, default='',
                    help='AVHBench: comma-separated paths to arm JSONL files.')
    ap.add_argument('--vote_threshold', type=int, default=2,
                    help='AVHBench: majority threshold (>= N of 4 arms).')

    args = ap.parse_args()

    ds = args.datasets.lower()
    if ds == 'phd':
        run_phd(args)
    elif ds == 'cmm':
        run_cmm(args)
    elif ds == 'pubmedqa':
        run_pubmedqa(args)
    elif ds == 'ragtruth':
        run_ragtruth(args)
    elif ds == 'haloquest':
        run_haloquest(args)
    elif ds == 'avhbench':
        run_avhbench_vote(args)
    else:
        ap.error(f'Unknown --datasets={ds!r}. '
                 f'Choose from: phd cmm pubmedqa ragtruth haloquest avhbench')


if __name__ == '__main__':
    main()
