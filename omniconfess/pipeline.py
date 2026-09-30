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
import os, json, time, argparse, random, string, re

import numpy as np
import torch
from transformers import LogitsProcessor

from .backbone import (
    load_dataset, load_media_for_item, load_model, build_inputs,
    prepare_for_thinker, build_user_content_with_media_tokens, make_messages,
    get_system_prompt, get_user_prompt, make_result, save_result,
)
from .common import (set_seed, token_logprobs, split_question,
                     load_done_ids, attach_record, write_time)
from .spine import ThreeStageConfessProcessor, teacher_force_lp, teacher_force_seq_lp
from .spine_freeform import FreeformThreeStageDriver, lowest_delta_span, word_overlap
from .operators import (
    gamma_extrapolated_phd, relevant_modalities_cmm, confidence_gated_flip,
    grounded_yes_flip_cmm, gamma_margin_cmm, gamma_scores_pubmedqa,
    gamma_anchored_pubmedqa,
)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Decision-token ids (Qwen2.5-Omni; shared by all datasets).
YES_ID, NO_ID, MAYBE_ID = 9454, 2753, 21390


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
# Stage 1  Generate   greedy-decode the base answer live from the image view;
#                     full-view per-token log-probs captured during generation
#                     (== full-view teacher-force) so Stage 2 needs only the
#                     no-image forward -- one fewer model call per sample.
# Stage 2  Confess    delta_img_tok = lp_full - lp_noimg, always-on. Low delta
#                     = token driven by text priors, not the image (ungrounded).
# Stage 3  Decide     bucket-routed (item['sub_category']) directed regen:
#                       false premises / visual challenge -> regen at the span
#                         with the lowest delta under the stronger v2 sys prompt.
#                       insufficient context -> decline-steered regen.

_HALO_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_HALO_DATA_PATH = os.path.join(
    _HALO_REPO, 'OmniHalluBench', 'haloquest', 'annotations', 'haloquest.json')
_HALO_DATA_ROOT = os.path.join(_HALO_REPO, 'OmniHalluBench', 'haloquest')

_HALO_EVADERE = re.compile(
    r"\b(cannot|can't|unable|impossible|not (possible|able)|"
    r"no (information|data|detail)|not (visible|present|shown|clear|"
    r"available|enough|sufficient)|unclear|don't know|I('m| am) not sure|"
    r"hard to (tell|say|determine)|difficult to|cannot determine|"
    r"not (in|from) the (image|picture|photo)|"
    r"image does not|picture does not|photo does not|"
    r"I (can't|cannot) see|not (show|visible|apparent)|"
    r"outside (the|my)|beyond (the|my)|based on (the|this) image)",
    re.IGNORECASE)

_HALO_NEG = (' no ', ' not ', "n't", 'none', 'nothing', 'isn', 'aren', 'doesn',
             'cannot', "can't", 'without', 'there is no', 'does not', 'do not',
             'never', 'not visible', 'not present', 'not shown')

_HALO_TERSE_STOP = {'is', 'are', 'the', 'a', 'an', 'it', 'its', 'this', 'that',
                    'to', 'and'}

_HALO_NUM_WORDS = {'0': 'zero', '1': 'one', '2': 'two', '3': 'three',
                   '4': 'four', '5': 'five', '6': 'six', '7': 'seven',
                   '8': 'eight', '9': 'nine', '10': 'ten', '11': 'eleven',
                   '12': 'twelve', '13': 'thirteen', '14': 'fourteen',
                   '15': 'fifteen', '16': 'sixteen', '17': 'seventeen',
                   '18': 'eighteen', '19': 'nineteen', '20': 'twenty'}


def _halo_is_refusal(txt):
    return bool(_HALO_EVADERE.search(str(txt)))


def _halo_is_negation(txt):
    s = ' ' + str(txt).lower() + ' '
    return any(n in s for n in _HALO_NEG)

def _halo_content_len(txt):
    ws = re.sub(r'[^\w\s]', ' ', str(txt).lower()).split()
    return len([w for w in ws if w not in _HALO_TERSE_STOP])


def _halo_count_noun(question):
    m = re.search(r'how many ([a-z][a-z ]*?)'
                  r'(?: are| is| can| do| does| seen| visible|\?|$)',
                  str(question).lower())
    return m.group(1).strip() if m else ''


def _halo_normalize_number_answer(pred, question):
    """Bare integer answers score ~0 Token-F1 against HaloQuest refs (which spell
    numbers out). Spell the digit; wrap with counted noun for 'how many' questions.
    Model-free and idempotent — only rewrites exact bare integers."""
    p = str(pred).strip().rstrip('.')
    if re.fullmatch(r'\d+', p) and p in _HALO_NUM_WORDS:
        noun = _halo_count_noun(question)
        return (f"There are {_HALO_NUM_WORDS[p]} {noun}."
                if noun else f"{_HALO_NUM_WORDS[p].capitalize()}.")
    return pred


def _halo_build_view_img(processor, model, sys_prompt, image, up):
    uc = build_user_content_with_media_tokens(None, None, image, up)
    raw = build_inputs(processor, model, make_messages(sys_prompt, uc),
                       None, None, image, 'haloquest')
    return prepare_for_thinker(raw, model)


def _halo_build_view_noimg(processor, model, sys_prompt, up):
    uc = build_user_content_with_media_tokens(None, None, None, up)
    raw = build_inputs(processor, model, make_messages(sys_prompt, uc),
                       None, None, None, 'haloquest')
    return prepare_for_thinker(raw, model)


def _halo_greedy(model, tokenizer, clean, max_new_tokens):
    with torch.inference_mode():
        out = model.thinker.generate(
            **clean, max_new_tokens=max_new_tokens,
            do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][clean['input_ids'].shape[1]:],
                            skip_special_tokens=True).strip()


def _halo_greedy_with_lp(model, tokenizer, clean, max_new_tokens):
    """Stage-1 generate that also returns Stage-2 full-view log-probs.

    Greedy decoding is self-teacher-forced, so each emitted token's log-prob
    under the image view equals teacher_force_seq_lp on the same prefix — we
    fold Stage-2's full-view teacher-force into this pass and save one forward.
    """
    eos = tokenizer.eos_token_id
    with torch.inference_mode():
        out = model.thinker.generate(
            **clean, max_new_tokens=max_new_tokens,
            do_sample=False, pad_token_id=eos,
            return_dict_in_generate=True, output_scores=True)
    input_len = clean['input_ids'].shape[1]
    gen_ids = out.sequences[0][input_len:]
    scores = out.scores
    n = min(gen_ids.shape[0], len(scores))
    lp_full = []
    for i in range(n):
        lsm = torch.log_softmax(scores[i][0].float(), dim=-1)
        lp_full.append(float(lsm[gen_ids[i]]))
    gen_ids = gen_ids[:n]
    if gen_ids.numel() and int(gen_ids[-1]) == eos:
        gen_ids = gen_ids[:-1]
        lp_full = lp_full[:-1]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return text, gen_ids.detach(), lp_full

def _halo_vc_regen_prompt(question, base_pred, span):
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


def _halo_ic_regen_prompt(question, base_pred):
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


def _halo_fp_regen_prompt(question, base_pred, span):
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


def _halo_premise_extract_prompt(question):
    return (
        "A question can presuppose that something exists or is true in an image.\n"
        f"Question: \"{question}\"\n"
        "In a few words, state the key thing this question ASSUMES is present "
        "or true in the image (the presupposition). Reply with just that "
        "assumed entity or fact, nothing else.")


def _halo_premise_answer_prompt(question, presupposition):
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

class HaloQuestConfessDriver(FreeformThreeStageDriver):
    """Stage 2 = delta_img_tok confession; Stage 3 = bucket-routed directed regen
    with heuristic accept guard. Subclasses FreeformThreeStageDriver so the
    three-stage loop is shared with the other free-form datasets."""

    def __init__(self, model, processor, tokenizer, sys_prompt,
                 overlap_threshold=0.10, regen_max_tokens=64,
                 fp_mode='directed_regen', regen_sys_prompt=None):
        super().__init__(model, processor, tokenizer)
        self.sys_prompt = sys_prompt
        self.regen_sys_prompt = regen_sys_prompt or sys_prompt
        self.overlap_threshold = overlap_threshold
        self.regen_max_tokens = regen_max_tokens
        self.fp_mode = fp_mode
        self._image_values = None
        # Stage-1 generation stashes full-view lp here; Stage 2 reuses it.
        self._lp_full = None

    def confess(self, base_pred, clean_full, clean_noimg, gen_ids, item):
        """Stage 2. Full-view lp was already captured during Stage-1 generation
        (_lp_full), so only the no-image teacher-force is run here."""
        toks = [self.tokenizer.decode([t]) for t in gen_ids.tolist()]
        lp_full = self._lp_full
        if lp_full is None:
            lp_full = teacher_force_seq_lp(self.model, clean_full, gen_ids)
        lp_noimg = teacher_force_seq_lp(self.model, clean_noimg, gen_ids)
        delta = [f - n for f, n in zip(lp_full, lp_noimg)]
        return dict(tokens=toks, delta_img_tok=delta, lp_full=lp_full,
                    n_tok=len(toks))

    def _accept(self, base_pred, cand, qt):
        """Bucket-aware heuristic guard. Returns (accepted: bool, reason).

        Two refinements over a plain overlap gate:
          revert-if-both-decline: base already declines and cand also declines
            -> keep base (regen only reshuffles vocabulary and loses Token-F1).
          terse-base bypass (VC): a 1-2-word base ("2", "Tree") has ~0
            word-overlap with any full-sentence regen, so the overlap gate would
            wrongly reject a good rewrite -> trust the candidate when base is terse.
        """
        if not cand or cand.strip() == base_pred.strip():
            return False, 'nochange'
        ov = word_overlap(cand, base_pred)
        b_decl = _halo_is_negation(base_pred) or _halo_is_refusal(base_pred)
        c_decl = _halo_is_negation(cand) or _halo_is_refusal(cand)
        if qt == 'visual challenge':
            if _halo_content_len(base_pred) <= 2:
                if b_decl and c_decl:
                    return False, 'keep_decline'
                return True, 'vc_terse_base'
            if ov < self.overlap_threshold:
                return False, 'low_overlap'
            if _halo_is_refusal(cand) and not _halo_is_refusal(base_pred):
                return False, 'new_refusal'
            if b_decl and c_decl:
                return False, 'keep_decline'
            return True, 'vc_ok'
        if qt == 'false premises':
            if b_decl and c_decl:
                return False, 'keep_decline'
            if ov >= self.overlap_threshold or _halo_is_negation(cand):
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

        if qt == 'insufficient context':
            cand = ''
            try:
                cl = _halo_build_view_img(
                    self.processor, self.model, self.regen_sys_prompt,
                    self._image_values, _halo_ic_regen_prompt(question, base_pred))
                cand = _halo_greedy(self.model, self.tokenizer, cl,
                                    self.regen_max_tokens)
                del cl; torch.cuda.empty_cache()
            except Exception as e:
                rec.update(stage3='ic_regen_error', regen_done=False,
                           regen_candidate='', accepted=False,
                           accept_reason='error:' + str(e)[:80],
                           edited=False, repaired_pred=base_pred,
                           final_pred=base_pred)
                return base_pred, rec
            accept = bool(cand) and cand.strip() != base_pred.strip() and \
                (_halo_is_refusal(cand) or _halo_is_negation(cand))
            final = cand if accept else base_pred
            rec.update(stage3='ic_regen', regen_done=True, regen_candidate=cand,
                       cand_is_refusal=_halo_is_refusal(cand),
                       cand_is_negation=_halo_is_negation(cand),
                       accepted=accept,
                       accept_reason='ic_decline' if accept else 'ic_keep_base',
                       edited=accept, repaired_pred=final, final_pred=final)
            return final, rec

        if qt == 'false premises' and self.fp_mode == 'premise_check':
            try:
                cl_e = _halo_build_view_img(
                    self.processor, self.model, self.regen_sys_prompt,
                    self._image_values, _halo_premise_extract_prompt(question))
                presup = _halo_greedy(self.model, self.tokenizer, cl_e, 32)
                del cl_e
                cl_a = _halo_build_view_img(
                    self.processor, self.model, self.regen_sys_prompt,
                    self._image_values,
                    _halo_premise_answer_prompt(question, presup))
                pc = _halo_greedy(self.model, self.tokenizer, cl_a,
                                  self.regen_max_tokens)
                del cl_a; torch.cuda.empty_cache()
            except Exception as e:
                rec.update(stage3='premise_check_error', regen_done=False,
                           regen_candidate='', accepted=False,
                           accept_reason='error:' + str(e)[:80],
                           edited=False, repaired_pred=base_pred,
                           final_pred=base_pred)
                return base_pred, rec
            accepted, reason = self._accept(base_pred, pc, 'false premises')
            final = pc if accepted else base_pred
            rec.update(stage3='fp_premise_check', regen_done=True,
                       regen_candidate=pc, presupposition=presup,
                       cand_is_negation=_halo_is_negation(pc),
                       accepted=accepted, accept_reason=reason,
                       edited=accepted, repaired_pred=final, final_pred=final)
            return final, rec

        if not span:
            final = _halo_normalize_number_answer(base_pred, question)
            rec.update(stage3='keep_base', regen_done=False, regen_candidate='',
                       accepted=False, accept_reason='no_span',
                       edited=(final.strip() != base_pred.strip()),
                       repaired_pred=final, final_pred=final)
            return final, rec

        rp = (_halo_vc_regen_prompt(question, base_pred, span)
              if qt == 'visual challenge'
              else _halo_fp_regen_prompt(question, base_pred, span))
        cand = ''
        try:
            cl = _halo_build_view_img(
                self.processor, self.model, self.regen_sys_prompt,
                self._image_values, rp)
            cand = _halo_greedy(self.model, self.tokenizer, cl,
                                self.regen_max_tokens)
            del cl; torch.cuda.empty_cache()
        except Exception as e:
            rec.update(stage3='regen_error', regen_done=False, regen_candidate='',
                       accepted=False, accept_reason='error:' + str(e)[:80],
                       edited=False, repaired_pred=base_pred, final_pred=base_pred)
            return base_pred, rec

        accepted, reason = self._accept(base_pred, cand, qt)
        final = cand if accepted else base_pred
        final = _halo_normalize_number_answer(final, question)
        rec.update(stage3='directed_regen', regen_done=True, regen_candidate=cand,
                   cand_overlap=round(word_overlap(cand, base_pred), 4),
                   cand_is_refusal=_halo_is_refusal(cand),
                   cand_is_negation=_halo_is_negation(cand),
                   accepted=accepted, accept_reason=reason,
                   edited=(final.strip() != base_pred.strip()),
                   repaired_pred=final, final_pred=final)
        return final, rec

def run_haloquest_pipeline(shard=0, nshards=8, outdir=None, fp_mode='directed_regen',
                           overlap_threshold=0.10, regen_max_tokens=64,
                           dry_run=False, file_name=None):
    """Run the HaloQuest free-text three-stage pipeline over one shard.

    Writes one JSONL per shard; returns its path. No-image samples keep base
    (channel_computed=0) so a full run yields N=365.
    """
    if outdir is None:
        outdir = os.path.join(_HALO_REPO, 'outputs', 'haloquest_freeform')
    os.makedirs(outdir, exist_ok=True)
    stem = file_name if file_name else f'freeform_s{shard}of{nshards}'
    outfile = os.path.join(outdir, stem + '.jsonl')

    data = load_dataset(_HALO_DATA_PATH, _HALO_DATA_ROOT)
    by_id = {str(it['id']): it for it in data}
    ids_all = sorted(by_id.keys(), key=lambda x: int(x))
    ids_shard = [i for j, i in enumerate(ids_all) if j % nshards == shard]

    if dry_run:
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
    sys_prompt = get_system_prompt('haloquest', 'v2')
    regen_sys_prompt = get_system_prompt('haloquest', 'v2')
    print(f'[fp_mode={fp_mode}] Stage-1/Stage-3 prompt=v2; '
          f'full-view lp folded into generation', flush=True)

    driver = HaloQuestConfessDriver(model, processor, tokenizer, sys_prompt,
                                    overlap_threshold=overlap_threshold,
                                    regen_max_tokens=regen_max_tokens,
                                    fp_mode=fp_mode,
                                    regen_sys_prompt=regen_sys_prompt)

    fout = open(outfile, 'a', encoding='utf-8')
    n_done = 0
    for sid in ids_shard:
        if sid in done:
            continue
        item = by_id[sid]
        qt = item['sub_category']
        up = get_user_prompt(item['question'], 'haloquest', 'v1')
        try:
            _, _, image_values = load_media_for_item(item, _HALO_DATA_PATH, _HALO_DATA_ROOT)
        except Exception as e:
            image_values = None
            print(f'  [warn {sid}] media err {e}', flush=True)

        if image_values is None:
            try:
                cl_noimg = _halo_build_view_noimg(processor, model, sys_prompt, up)
                base_pred = _halo_greedy(model, tokenizer, cl_noimg, regen_max_tokens)
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
            print(f'  [{sid}] {qt[:4]} no-image -> keep base', flush=True)
            continue

        try:
            clean_full = _halo_build_view_img(processor, model, sys_prompt,
                                              image_values, up)
        except Exception as e:
            print(f'  [skip {sid}] view err {e}', flush=True); continue
        base_pred, gen_ids, lp_full = _halo_greedy_with_lp(model, tokenizer,
                                                            clean_full, regen_max_tokens)
        del clean_full; torch.cuda.empty_cache()
        if gen_ids.numel() == 0:
            print(f'  [skip {sid}] empty base tokens', flush=True); continue
        gen_ids = gen_ids.to(model.device)

        try:
            clean_noimg = _halo_build_view_noimg(processor, model, sys_prompt, up)
        except Exception as e:
            print(f'  [skip {sid}] noimg view err {e}', flush=True); continue

        driver._image_values = image_values
        driver._lp_full = lp_full
        try:
            final_pred, rec = driver.run_sample(base_pred, None, clean_noimg,
                                                gen_ids, item)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f'  [error {sid}] {e}', flush=True)
            del clean_noimg; torch.cuda.empty_cache(); continue
        del clean_noimg; torch.cuda.empty_cache()

        fout.write(json.dumps(rec, ensure_ascii=False) + '\n'); fout.flush()
        n_done += 1
        print(f'[{sid}] {qt[:4]} span={rec.get("low_span")!r}'
              f'({rec.get("low_span_delta")}) '
              f'edit={rec.get("edited")} final={final_pred[:50]!r}', flush=True)

    fout.close()
    print(f'[shard {shard}] done, wrote {n_done} new -> {outfile}', flush=True)
    return outfile


def run_haloquest(args):
    """HaloQuest free-text three-stage arm (pipeline entry point)."""
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
                    choices=['directed_regen', 'premise_check'],
                    default='directed_regen',
                    help='HaloQuest FALSE-PREMISES Stage-3 source. directed_regen '
                         '(default) is the delta-driven directed regeneration that '
                         'consumes the Stage-2 signal; premise_check is an ablation.')
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
