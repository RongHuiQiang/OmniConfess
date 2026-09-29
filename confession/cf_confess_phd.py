#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""cf_confess_phd.py -- three-stage counterfactual confession for PHD.

The misleading text CONTEXT is the irrelevant channel; the image is the
relevant channel and is never ablated. The three stages:

  Stage 1  Generate   greedy decode under the full view (image + context +
                      question) until the decision token.
  Stage 2  Confess    teacher-force the context-removed view (image + question
                      only) on the same prefix and read its Yes/No logprobs.
  Stage 3  Decide     strip the irrelevant context channel; force the token.

Stage 3 rules (--decision):
  gamma  (default, champion)  margin' = margin_full + w*(margin_cf - margin_full);
         force yes iff margin' > tau_p. w>1 extrapolates the context-removal
         signal (phd context is adversarial misinformation).
  lambda (ablation / control) margin' = margin_full + lambda * Delta_cf.
         --control random_sign flips Delta per sample.

Views:
  full : image + context + question   (== base_model input)
  cf   : image + question only        (context removed)

Interface compatible with run_text_8gpu.sh / shard_data.py.
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
from gamma_operator import gamma_extrapolated_phd
from cf_common import (set_seed, token_logprobs, split_question,
                       load_done_ids, attach_record, write_time)
from three_stage import ThreeStageConfessProcessor, teacher_force_lp

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

YES_ID, NO_ID = 9454, 2753


class CFConfessProcessor(ThreeStageConfessProcessor):
    """PHD three-stage confession processor.

    Stage 2 (confess): teacher-force the context-removed view, return
        (lp_yes_cf, lp_no_cf) at the decision token.

    Stage 3 (decide):
        gamma  -- margin' = margin_full + w*(margin_cf - margin_full);
                  force yes iff margin' > tau_p  (champion rule).
        lambda -- margin' = margin_full + lam * Delta_cf  (ablation).
    """

    def __init__(self, model, tokenizer, cf_clean, prompt_len,
                 lam, control_rng=None, decision='gamma', w_extrap=4.0,
                 tau_p=-1.6):
        super().__init__(tokenizer, prompt_len)
        self.model = model
        self.cf_clean = cf_clean
        self.lam = lam
        self.control_rng = control_rng
        self.decision = decision
        self.w_extrap = w_extrap   # context-strip extrapolation weight
        self.tau_p = tau_p         # yes decision threshold on margin'

    def confess(self, prefix_ids):
        """Stage 2: teacher-force the context-removed view on the same prefix."""
        ly_cf, ln_cf = teacher_force_lp(self.model, self.cf_clean, prefix_ids,
                                        (YES_ID, NO_ID))
        return dict(ly_cf=ly_cf, ln_cf=ln_cf)

    def decide(self, full_view, confess_signal):
        """Stage 3: strip the context channel, force the token."""
        am = full_view['am']
        ly_full, ln_full = full_view['ly_full'], full_view['ln_full']
        margin_full = full_view['margin_full']
        ly_cf, ln_cf = confess_signal['ly_cf'], confess_signal['ln_cf']

        delta = ly_cf - ly_full
        if self.decision == 'gamma' and self.control_rng is None:
            margin_prime, forced = gamma_extrapolated_phd(ly_full, ln_full,
                                                          ly_cf, ln_cf,
                                                          w=self.w_extrap,
                                                          tau_p=self.tau_p)
            delta_eff = delta
        else:
            delta_eff = delta
            if self.control_rng is not None:
                delta_eff = delta * self.control_rng.choice([-1.0, 1.0])
            margin_prime = (ly_full - ln_full) + self.lam * delta_eff
            forced = YES_ID if margin_prime > 0 else NO_ID

        record = dict(
            dec_index=full_view['dec_index'],
            greedy_token=am,
            forced_token=forced,
            intervened=int(forced != am),
            decision=self.decision,
            lp_yes_full=round(ly_full, 4), lp_no_full=round(ln_full, 4),
            margin_full=round(margin_full, 4),
            lp_yes_cf=round(ly_cf, 4), lp_no_cf=round(ln_cf, 4),
            delta_cf=round(ly_cf - ly_full, 4),
            delta_eff=round(delta_eff, 4),
            margin_adj=round(margin_prime, 4),
            w_extrap=self.w_extrap,
            tau_p=self.tau_p,
        )
        return forced, record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', type=str, default='phd')
    ap.add_argument('--data_path', type=str, required=True)
    ap.add_argument('--data_root', type=str, default='./OmniHalluBench/')
    ap.add_argument('--output_dir', type=str, default='./outputs/phd_cf_confess')
    ap.add_argument('--file_name', type=str, default='cf_confess_phd')
    ap.add_argument('--time_path', type=str, default='./outputs/phd_cf_confess/time')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--max_new_tokens', type=int, default=300)
    ap.add_argument('--prompt_version', type=str, default='v1')
    ap.add_argument('--lambda_cf', type=float, default=2.0)
    ap.add_argument('--decision', choices=['gamma', 'lambda'], default='gamma',
                    help="Stage-3 rule: 'gamma' = champion lambda-free rule, "
                         "'lambda' = legacy margin+lambda*Delta.")
    ap.add_argument('--w_extrap', type=float, default=4.0,
                    help="Extrapolation weight: margin' = margin_full + "
                         "w*(margin_cf - margin_full). Champion: 4.0.")
    ap.add_argument('--tau_p', type=float, default=-1.6,
                    help="Decision threshold: predict yes iff margin' > tau_p. "
                         "Champion: -1.6.")
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
            q_full, q_only = split_question(item['question'])

            u_full = get_user_prompt(q_full, args.datasets, args.prompt_version)
            uc_full = build_user_content_with_media_tokens(pv, av, iv, u_full)
            clean_f = prepare_for_thinker(
                build_inputs(processor, model,
                             make_messages(system_prompt, uc_full),
                             pv, av, iv, args.datasets), model)
            input_len = clean_f['input_ids'].shape[1]

            u_cf = get_user_prompt(q_only, args.datasets, args.prompt_version)
            uc_cf = build_user_content_with_media_tokens(pv, av, iv, u_cf)
            clean_cf = prepare_for_thinker(
                build_inputs(processor, model,
                             make_messages(system_prompt, uc_cf),
                             pv, av, iv, args.datasets), model)

            cf_proc = CFConfessProcessor(model, tokenizer, clean_cf, input_len,
                                         args.lambda_cf, control_rng,
                                         decision=args.decision,
                                         w_extrap=args.w_extrap,
                                         tau_p=args.tau_p)

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
            rec_extra.update(lambda_cf=args.lambda_cf, control=args.control,
                             decision=args.decision)
            all_out_tok += len(gen_ids)
            print(f'  response[-60]: {response[-60:]!r} | dec={rec_extra.get("dec_index")} '
                  f'interv={rec_extra.get("intervened")} | {time.time()-t0:.1f}s', flush=True)
            del clean_f, clean_cf
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f'  [ERROR] {e}', flush=True)
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
    print(f'\n[done] output={output_path}', flush=True)


if __name__ == '__main__':
    main()
