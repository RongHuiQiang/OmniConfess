#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""cf_confess_phd.py -- three-stage counterfactual confession operator for PhD.

The misleading text CONTEXT is the irrelevant channel that must be stripped;
the image is the relevant channel and is never ablated. The three stages:

  Stage 1  Generate   greedy decode under the full view (image + context +
                      marker + question, identical to base_model) until the
                      decision token (bare Yes/No at step 0, or right after
                      `\\boxed{`).
  Stage 2  Confess     at the decision token, teacher-force the context-removed
                      view (image + question only) on the SAME prefix and read
                      its Yes/No logprobs. This is the model's confession: what
                      it would decide with no misleading text prior.
  Stage 3  Decide      strip the irrelevant context channel and keep the image
                      contribution, then hard-force the decision token.

Two decision rules (Stage 3), selected by --decision:
  gamma  (default, deployed)  margin' = F_d(X^{-context}) = ly_cf - ln_cf,
         force Yes iff margin' > 0. Zero-lambda; this is the accepted Method-A
         deployment decision (offline-validated acc 79.5% vs base 78.5%).
  lambda (ablation / control)  margin' = margin_full + lambda * Delta_cf, with
         Delta_cf = ly_cf - ly_full. The --control random_sign arm flips the
         sign of Delta per sample and therefore always uses this rule.

Views:
  full : image + context + marker + question   (== base_model input)
  cf   : image + question only (context & marker sentence removed)

Interface compatible with run_text_8gpu.sh / shard_data.py
(--data_path/--file_name/--output_dir/--time_path) and base_model.py resume.
"""
import os, sys, time, argparse, random

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'OmniConfess', 'methods'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from transformers import LogitsProcessor

from shared import (
    load_dataset, load_media_for_item, load_model, build_inputs,
    prepare_for_thinker, build_user_content_with_media_tokens, make_messages,
    get_system_prompt, get_user_prompt, make_result, save_result,
)
from gamma_operator import gamma_extrapolated_phd  # Stage-3 v4 (n=200 F1-yes rule)
from cf_common import (set_seed, token_logprobs, split_question,
                       load_done_ids, attach_record, write_time)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

YES_ID, NO_ID = 9454, 2753


class CFConfessProcessor(LogitsProcessor):
    """Runs Stage 2 (Confess) and Stage 3 (Decide) once, at the decision token.

    Stage 1 (Generate) is the surrounding greedy generate() loop. This class
    intervenes exactly once, when generation reaches the decision step:
        (no tokens generated yet and greedy argmax is Yes/No)
        or (generated text so far ends with `\\boxed{`).

    decision:
        'gamma'  -> lambda-free gamma_margin_phd (deployed rule, tau=0).
        'lambda' -> margin_full + lam * Delta_cf (ablation / random_sign arm).
    """

    def __init__(self, model, tokenizer, cf_clean, prompt_len,
                 lam, control_rng=None, decision='gamma', gate_tau=1.4, w_extrap=2.0):
        self.model = model
        self.tokenizer = tokenizer
        self.cf_clean = cf_clean
        self.prompt_len = prompt_len
        self.lam = lam
        self.control_rng = control_rng      # None => real Delta; else sign flip
        self.decision = decision
        self.gate_tau = gate_tau            # legacy v3 gate (unused by v4 rule)
        self.w_extrap = w_extrap            # v4: context-strip extrapolation weight
        self.used = False
        self.record = None

    @torch.inference_mode()
    def _confess_lp_yes_no(self, prefix_ids):
        """Stage 2: teacher-force the context-removed view on the same prefix
        and return its (lp_yes, lp_no) at the decision token."""
        ids = torch.cat([self.cf_clean['input_ids'][0],
                         torch.tensor(prefix_ids, dtype=torch.long,
                                      device=self.cf_clean['input_ids'].device)],
                        dim=0).unsqueeze(0)
        kw = dict(input_ids=ids, attention_mask=torch.ones_like(ids))
        for k in ('pixel_values_images', 'pixel_values', 'image_grid_thw',
                  'pixel_values_videos', 'video_grid_thw',
                  'video_second_per_grid', 'input_features',
                  'feature_attention_mask'):
            if k in self.cf_clean:
                kw[k] = self.cf_clean[k]
        logits = self.model.thinker(**kw).logits[0, -1].float()
        lp = torch.log_softmax(logits, dim=-1)
        return lp[YES_ID].item(), lp[NO_ID].item()

    def _decide(self, am, ly_full, ln_full, ly_cf, ln_cf):
        """Stage 3: return (forced_token, margin', delta_eff).

        gamma  v4 (n=200 F1-yes): margin' = margin_full + w*(margin_cf -
               margin_full) with margin_cf = ly_cf - ln_cf the context-stripped
               margin; force yes iff margin' > 0. w>1 extrapolates past the clean
               view because phd's context is adversarial misinformation, so
               amplifying how far context-removal moves the decision is the right
               correction (base 72.57 -> 82.95 on n=200).
        lambda keeps the full margin plus a scaled confession pull; the
               random_sign control flips Delta before applying it.
        """
        delta = ly_cf - ly_full
        if self.decision == 'gamma' and self.control_rng is None:
            margin_prime, forced = gamma_extrapolated_phd(ly_full, ln_full,
                                                          ly_cf, ln_cf,
                                                          w=self.w_extrap)
            return forced, margin_prime, delta
        delta_eff = delta
        if self.control_rng is not None:
            delta_eff = delta * self.control_rng.choice([-1.0, 1.0])
        margin_adj = (ly_full - ln_full) + self.lam * delta_eff
        forced = YES_ID if margin_adj > 0 else NO_ID
        return forced, margin_adj, delta_eff

    @torch.inference_mode()
    def __call__(self, input_ids, scores):
        if self.used:
            return scores
        gen = input_ids[0, self.prompt_len:].tolist()
        logits = scores[0].float()
        am = int(logits.argmax().item())

        at_decision = False
        if len(gen) == 0:
            at_decision = am in (YES_ID, NO_ID)          # bare immediate answer
        else:
            tail = self.tokenizer.decode(gen[-4:])
            at_decision = tail.endswith('\\boxed{') or tail.endswith('\\boxed{{')

        if not at_decision:
            return scores

        # Stage 2 (Confess): full-view margin + context-removed confession.
        lp = torch.log_softmax(logits, dim=-1)
        ly_full, ln_full = lp[YES_ID].item(), lp[NO_ID].item()
        margin_full = ly_full - ln_full
        ly_cf, ln_cf = self._confess_lp_yes_no(gen)

        # Stage 3 (Decide): strip the irrelevant channel, hard-force the token.
        forced, margin_prime, delta_eff = self._decide(am, ly_full, ln_full,
                                                        ly_cf, ln_cf)
        scores[0][:] = float('-inf')
        scores[0][forced] = 0.0          # hard-force the decision token

        self.used = True
        self.record = dict(
            dec_index=len(gen),
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
            gate_tau=self.gate_tau,
            w_extrap=self.w_extrap,
        )
        return scores


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
                    help="Stage-3 rule: 'gamma' = deployed lambda-free rule "
                         "(default), 'lambda' = legacy margin+lambda*Delta.")
    ap.add_argument('--gate_tau', type=float, default=1.4,
                    help="legacy v3 gate (unused by v4 extrapolation rule).")
    ap.add_argument('--w_extrap', type=float, default=2.0,
                    help="v4 phd rule: margin' = margin_full + w*(margin_cf - "
                         "margin_full); w=1 = literal gamma, w=2 = deployed "
                         "(n=200 F1-yes 82.95).")
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

            # full view (== base_model input)
            u_full = get_user_prompt(q_full, args.datasets, args.prompt_version)
            uc_full = build_user_content_with_media_tokens(pv, av, iv, u_full)
            clean_f = prepare_for_thinker(
                build_inputs(processor, model,
                             make_messages(system_prompt, uc_full),
                             pv, av, iv, args.datasets), model)
            input_len = clean_f['input_ids'].shape[1]

            # counterfactual view: image + question only
            u_cf = get_user_prompt(q_only, args.datasets, args.prompt_version)
            uc_cf = build_user_content_with_media_tokens(pv, av, iv, u_cf)
            clean_cf = prepare_for_thinker(
                build_inputs(processor, model,
                             make_messages(system_prompt, uc_cf),
                             pv, av, iv, args.datasets), model)

            cf_proc = CFConfessProcessor(model, tokenizer, clean_cf, input_len,
                                         args.lambda_cf, control_rng,
                                         decision=args.decision,
                                         gate_tau=args.gate_tau,
                                         w_extrap=args.w_extrap)

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
