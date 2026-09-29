#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""cf_confess_cmm.py -- three-stage counterfactual confession for CMM.

Multimodal case: two ablatable channels (audio, video). The three stages:

  Stage 1  Generate   greedy decode under the full A+V view until the decision
                      token (bare Yes/No at step 0, or right after \\boxed{`).
  Stage 2  Confess    teacher-force the two modality-masked views (no-audio,
                      no-video) on the same prefix, giving per-channel pulls
                          d_cf_i = lp_yes(mask-i view) - lp_yes(full view).
                      Restrict to the question-relevant channel set R
                      (relevant_modalities_cmm).
  Stage 3  Decide     dual-channel grounding gate: keep a base-yes when it is
                      either confident (margin_full >= margin_conf) or grounded
                      in R (min_{m in R} d_cf_m <= tau_m); otherwise flip to no.
                      base-no is never touched.

Both masked-view forwards run for every sample (always-on confession).

Controls: --control random_sign flips the sign of delta (seeded, per sample).
Interface compatible with shard_data.py / run_cf_confess_8gpu.sh.
"""
import os, sys, time, argparse, random

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'OmniConfess', 'methods'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from shared import (
    load_dataset, load_media_for_item, load_model, build_inputs,
    prepare_for_thinker, build_user_content_with_media_tokens, make_messages,
    get_system_prompt, get_user_prompt, make_result, save_result,
)
from gamma_operator import (relevant_modalities_cmm, confidence_gated_flip,
                             grounded_yes_flip_cmm, gamma_margin_cmm)
from cf_common import (set_seed, token_logprobs, load_done_ids,
                       attach_record, write_time)
from three_stage import ThreeStageConfessProcessor, teacher_force_lp

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

YES_ID, NO_ID = 9454, 2753


def build_view(processor, model, system_prompt, pv, av, iv, user_prompt):
    uc = build_user_content_with_media_tokens(pv, av, iv, user_prompt)
    return prepare_for_thinker(
        build_inputs(processor, model, make_messages(system_prompt, uc),
                     pv, av, iv, 'cmm'), model)


class CFConfessProcessor(ThreeStageConfessProcessor):
    """CMM three-stage confession processor.

    Stage 2 (confess): both masked-view teacher-force forwards, always-on.
        d_cf_i = lp_yes(mask-i view) - lp_yes(full view)  for i in {audio, video}

    Stage 3 (decide): dual-channel grounding gate (champion rule).
        keep base-yes if confident (margin_full >= margin_conf)
               OR grounded (min_{m in R} d_cf_m <= tau_m)
        flip to no otherwise. base-no is never touched.
    """

    def __init__(self, model, tokenizer, cf_noaudio, cf_novideo, prompt_len,
                 lam, w_audio=1.0, w_video=1.0, control_rng=None,
                 question=None, group_key=None, gate_tau=2.0,
                 decision='grounded', margin_conf=2.0, tau_m=-1.6):
        super().__init__(tokenizer, prompt_len)
        self.model = model
        self.cf_noaudio = cf_noaudio
        self.cf_novideo = cf_novideo
        self.lam = lam
        self.w_audio = w_audio
        self.w_video = w_video
        self.control_rng = control_rng
        self.gate_tau = gate_tau
        self.decision = decision
        self.margin_conf = margin_conf  # confidence gate: keep base-yes when margin_full >= margin_conf
        self.tau_m = tau_m              # grounding threshold: keep low-margin yes iff depR <= tau_m
        self.R, self.R_source = relevant_modalities_cmm(question, group_key)

    def confess(self, prefix_ids):
        """Stage 2: both masked-view teacher-force forwards, always-on."""
        ya_no_audio = teacher_force_lp(self.model, self.cf_noaudio, prefix_ids, (YES_ID,))[0]
        ya_no_video = teacher_force_lp(self.model, self.cf_novideo, prefix_ids, (YES_ID,))[0]
        na_no_audio = teacher_force_lp(self.model, self.cf_noaudio, prefix_ids, (NO_ID,))[0]
        na_no_video = teacher_force_lp(self.model, self.cf_novideo, prefix_ids, (NO_ID,))[0]
        return dict(ly_noaudio=ya_no_audio, ly_novideo=ya_no_video,
                    ln_noaudio=na_no_audio, ln_novideo=na_no_video)

    def decide(self, full_view, confess_signal):
        am = full_view['am']
        ly_full, ln_full = full_view['ly_full'], full_view['ln_full']
        margin_full = full_view['margin_full']
        dec_index = full_view['dec_index']

        d_audio = confess_signal['ly_noaudio'] - ly_full
        d_video = confess_signal['ly_novideo'] - ly_full

        # Champion path: dual-channel grounding gate
        if self.decision == 'grounded' and self.control_rng is None:
            forced = grounded_yes_flip_cmm(am, margin_full, self.R,
                                           d_audio, d_video,
                                           self.margin_conf, self.tau_m)
            record = dict(
                dec_index=dec_index,
                greedy_token=am,
                forced_token=forced,
                intervened=int(forced != am),
                decision='grounded',
                lp_yes_full=round(ly_full, 4), lp_no_full=round(ln_full, 4),
                margin_full=round(margin_full, 4),
                d_cf_audio=round(d_audio, 4), d_cf_video=round(d_video, 4),
                R=sorted(self.R), R_source=self.R_source,
                channel_computed=1,
                margin_conf=self.margin_conf, tau_m=self.tau_m,
            )
            return forced, record

        # lambda_delta (and random_sign control)
        delta = 0.0
        if 'audio' in self.R:
            delta += self.w_audio * d_audio
        if 'video' in self.R:
            delta += self.w_video * d_video
        delta_eff = delta
        if self.control_rng is not None:
            delta_eff = delta * self.control_rng.choice([-1.0, 1.0])
        margin_adj = margin_full + self.lam * delta_eff
        flip = YES_ID if margin_adj > 0 else NO_ID
        forced = confidence_gated_flip(am, margin_full, flip, self.gate_tau)
        record = dict(
            dec_index=dec_index,
            greedy_token=am,
            forced_token=forced,
            intervened=int(forced != am),
            decision='lambda_delta',
            lp_yes_full=round(ly_full, 4), lp_no_full=round(ln_full, 4),
            margin_full=round(margin_full, 4),
            d_cf_audio=round(d_audio, 4), d_cf_video=round(d_video, 4),
            w_audio=self.w_audio, w_video=self.w_video,
            R=sorted(self.R), R_source=self.R_source,
            channel_computed=1,
            delta=round(delta, 4),
            delta_eff=round(delta_eff, 4),
            margin_adj=round(margin_adj, 4),
            gate_tau=self.gate_tau,
        )
        return forced, record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', type=str, default='cmm')
    ap.add_argument('--data_path', type=str, required=True)
    ap.add_argument('--data_root', type=str, default='./OmniHalluBench/')
    ap.add_argument('--output_dir', type=str, default='./outputs/cmm_cf_confess')
    ap.add_argument('--file_name', type=str, default='cf_confess_cmm')
    ap.add_argument('--time_path', type=str, default='./outputs/cmm_cf_confess/time')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--max_new_tokens', type=int, default=300)
    ap.add_argument('--prompt_version', type=str, default='v1')
    ap.add_argument('--lambda_cf', type=float, default=2.0)
    ap.add_argument('--gate_tau', type=float, default=2.0)
    ap.add_argument('--w_audio', type=float, default=1.0)
    ap.add_argument('--w_video', type=float, default=1.0)
    ap.add_argument('--decision', choices=['grounded', 'lambda_delta'],
                    default='grounded',
                    help="Stage-3 rule. 'grounded' (default, champion): "
                         "dual-channel grounding gate (depR<=tau_m). "
                         "'lambda_delta': legacy λ-δ.")
    ap.add_argument('--margin_conf', type=float, default=2.0,
                    help="Confidence gate: keep base-yes unconditionally when "
                         "margin_full >= margin_conf. Champion: 2.0.")
    ap.add_argument('--tau_m', type=float, default=-1.6,
                    help="Grounding threshold: keep a low-margin base-yes iff "
                         "min_{m in R} d_cf_m <= tau_m. Champion: -1.6.")
    ap.add_argument('--control', choices=['none', 'random_sign'], default='none')
    ap.add_argument('--record_process', action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument('--dry_run', action='store_true')
    args = ap.parse_args()

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
    if args.dry_run:
        data = data[:3]

    system_prompt = get_system_prompt(args.datasets, args.prompt_version)

    all_out_tok = 0
    t_start = time.time()

    for idx, item in enumerate(data):
        if item['id'] in done_ids:
            continue
        t0 = time.time()
        print(f'\n[{idx+1}/{len(data)}] id={item["id"]}', flush=True)

        rec_extra, gen_ids, logp_list, input_len = {}, None, None, None
        try:
            pv, av, iv = load_media_for_item(item, args.data_path, args.data_root)
            u = get_user_prompt(item['question'], args.datasets, args.prompt_version)

            clean_f = build_view(processor, model, system_prompt, pv, av, iv, u)
            input_len = clean_f['input_ids'].shape[1]
            clean_a = build_view(processor, model, system_prompt, pv, None, iv, u)
            clean_v = build_view(processor, model, system_prompt, None, av, iv, u)

            cf_proc = CFConfessProcessor(model, tokenizer, clean_a, clean_v,
                                         input_len, args.lambda_cf,
                                         args.w_audio, args.w_video, control_rng,
                                         question=item.get('question', ''),
                                         group_key=item.get('group_key', ''),
                                         gate_tau=args.gate_tau,
                                         decision=args.decision,
                                         margin_conf=args.margin_conf,
                                         tau_m=args.tau_m)

            with torch.inference_mode():
                out = model.thinker.generate(
                    **clean_f,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    output_scores=True,
                    return_dict_in_generate=True,
                    logits_processor=[cf_proc],
                )
            gen_ids = out.sequences[0][input_len:].cpu()
            if args.record_process:
                logp_list = token_logprobs(gen_ids, out.scores)
            response = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            rec_extra = cf_proc.record or dict(dec_index=None, intervened=0)
            rec_extra.update(lambda_cf=args.lambda_cf, control=args.control)
            all_out_tok += len(gen_ids)
            print(f'  response[-60]: {response[-60:]!r} | dec={rec_extra.get("dec_index")} '
                  f'interv={rec_extra.get("intervened")} | {time.time()-t0:.1f}s', flush=True)
            del clean_f, clean_a, clean_v
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f'  [ERROR] {e}', flush=True)
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
    print(f'\n[done] output={output_path}', flush=True)


if __name__ == '__main__':
    main()
