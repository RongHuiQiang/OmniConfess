#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""cf_confess_pubmedqa.py -- three-stage counterfactual confession for PubMedQA.

Text-only, 3-class {Yes, No, Maybe}. The PASSAGE is the relevant evidence
channel here; the passage-removed view is the parametric prior P^0 that must
be stripped. pubmedqa base answers begin with a bare 'Yes.' / 'No.' / 'Maybe.'
as the very first generated token (401/57/10 of 468), so the decision step is
step 0.

Three stages:
  Stage 1  Generate   greedy decode under the full view (passage + question)
                      until the step-0 decision token.
  Stage 2  Confess     teacher-force the passage-removed view on the same prefix
                      and read {Yes, No, Maybe} logprobs -- the model's prior
                      answer with no passage evidence.
  Stage 3  Decide      keep the passage contribution per class and strip the
                      no-passage prior, then hard-force the decision token.

Two decision rules (Stage 3), selected by --decision:
  gamma  (default, deployed)  kappa_d = lp_d_full - lp_d_cf for each class,
         decision = argmax_d kappa_d. Zero-lambda literal E^R rule.
  lambda (ablation / control)  yes' = lp_yes_full + lambda * d_cf,
         decision = argmax{yes', lp_no_full, lp_maybe_full}. The random_sign
         control flips d_cf per sample and always uses this rule.

Interface compatible with shard_data.py / run_cf_confess_8gpu.sh and
resume-by-id.
"""
import os, sys, time, argparse, random

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'OmniConfess', 'methods'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from transformers import LogitsProcessor

from shared import (
    load_dataset, load_model, build_inputs,
    prepare_for_thinker, build_user_content_with_media_tokens, make_messages,
    get_system_prompt, get_user_prompt, make_result, save_result,
)
from gamma_operator import gamma_scores_pubmedqa, gamma_anchored_pubmedqa  # Stage-3 rules
from cf_common import (set_seed, token_logprobs, load_done_ids,
                       attach_record, write_time)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

YES_ID, NO_ID, MAYBE_ID = 9454, 2753, 21390


def strip_passage_line(user_prompt):
    lines = user_prompt.split('\n')
    if lines and lines[0].startswith('Passage:'):
        return '\n'.join(lines[1:])
    return user_prompt


def build_view(processor, model, system_prompt, user_prompt):
    uc = build_user_content_with_media_tokens(None, None, None, user_prompt)
    return prepare_for_thinker(
        build_inputs(processor, model, make_messages(system_prompt, uc),
                     None, None, None, 'pubmedqa'), model)


class CFConfessProcessor(LogitsProcessor):
    """Runs Stage 2 (Confess) and Stage 3 (Decide) once, at step 0 if the
    greedy argmax is Yes/No/Maybe (the pubmedqa base output shape, 468/468).

    Stage 1 (Generate) is the surrounding greedy generate() loop.

    decision:
        'gamma'  -> lambda-free gamma_scores_pubmedqa (deployed rule, argmax
                    per-class passage contribution).
        'lambda' -> yes' = lp_yes_full + lam * d_cf (ablation / random_sign arm).
    """

    def __init__(self, model, tokenizer, cf_clean, prompt_len, lam,
                 control_rng=None, decision='gamma', gate_tau_maybe=0.5):
        self.model = model
        self.tokenizer = tokenizer
        self.cf_clean = cf_clean
        self.prompt_len = prompt_len
        self.lam = lam
        self.control_rng = control_rng
        self.decision = decision
        self.gate_tau_maybe = gate_tau_maybe
        self.used = False
        self.record = None

    @torch.inference_mode()
    def _confess_lps(self, prefix_ids):
        """Stage 2: teacher-force the passage-removed view on the same prefix
        and return its (lp_yes, lp_no, lp_maybe) at the decision token."""
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
        return lp[YES_ID].item(), lp[NO_ID].item(), lp[MAYBE_ID].item()

    def _decide(self, am, ly, ln, lm, ly_cf, ln_cf, lm_cf):
        """Stage 3: return (forced_token, yes_adj, d_eff).

        gamma  candidate-anchored + 'maybe' evidence gate (v3): keep the base
               committed class am unless am==maybe and the passage does not
               ground it, then redecide yes/no. (Replaces the ungated argmax_d
               kappa_d, which fit near-zero passage noise and over-flipped.)
        lambda shift only the Yes logit by lam*d_cf, then argmax the 3 classes;
               the random_sign control flips d_cf first.
        """
        d_cf = ly_cf - ly
        if self.decision == 'gamma' and self.control_rng is None:
            forced = gamma_anchored_pubmedqa(am, ly, ln, lm, ly_cf, ln_cf, lm_cf,
                                             self.gate_tau_maybe)
            return forced, ly, d_cf        # yes_adj unused in gamma mode
        d_eff = d_cf
        if self.control_rng is not None:
            d_eff = d_cf * self.control_rng.choice([-1.0, 1.0])
        yes_adj = ly + self.lam * d_eff
        choices = {YES_ID: yes_adj, NO_ID: ln, MAYBE_ID: lm}
        forced = max(choices, key=choices.get)
        return forced, yes_adj, d_eff

    @torch.inference_mode()
    def __call__(self, input_ids, scores):
        if self.used:
            return scores
        gen = input_ids[0, self.prompt_len:].tolist()
        if gen:                       # decision is only ever at step 0 here
            return scores
        logits = scores[0].float()
        am = int(logits.argmax().item())
        if am not in (YES_ID, NO_ID, MAYBE_ID):
            return scores

        # Stage 2 (Confess): full-view + passage-removed logprobs.
        lp = torch.log_softmax(logits, dim=-1)
        ly, ln, lm = lp[YES_ID].item(), lp[NO_ID].item(), lp[MAYBE_ID].item()
        ly_cf, ln_cf, lm_cf = self._confess_lps(gen)

        # Stage 3 (Decide): anchor to base, gate the counterfactual override.
        forced, yes_adj, d_eff = self._decide(am, ly, ln, lm, ly_cf, ln_cf, lm_cf)
        scores[0][:] = float('-inf')
        scores[0][forced] = 0.0

        self.used = True
        self.record = dict(
            dec_index=0,
            greedy_token=am,
            forced_token=forced,
            intervened=int(forced != am),
            decision=self.decision,
            lp_yes_full=round(ly, 4), lp_no_full=round(ln, 4),
            lp_maybe_full=round(lm, 4),
            lp_yes_cf=round(ly_cf, 4), lp_no_cf=round(ln_cf, 4),
            lp_maybe_cf=round(lm_cf, 4),
            d_cf=round(ly_cf - ly, 4), d_eff=round(d_eff, 4),
            yes_adj=round(yes_adj, 4),
        )
        return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', type=str, default='pubmedqa')
    ap.add_argument('--data_path', type=str, required=True)
    ap.add_argument('--data_root', type=str, default='./OmniHalluBench/')
    ap.add_argument('--output_dir', type=str, default='./outputs/pubmedqa_cf_confess')
    ap.add_argument('--file_name', type=str, default='cf_confess_pubmedqa')
    ap.add_argument('--time_path', type=str, default='./outputs/pubmedqa_cf_confess/time')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--max_new_tokens', type=int, default=300)
    ap.add_argument('--prompt_version', type=str, default='v1')
    ap.add_argument('--lambda_cf', type=float, default=2.0)
    ap.add_argument('--decision', choices=['gamma', 'lambda'], default='gamma',
                    help="Stage-3 rule: 'gamma' = deployed lambda-free rule "
                         "(default), 'lambda' = legacy yes+lambda*d_cf.")
    ap.add_argument('--gate_tau_maybe', type=float, default=0.5,
                    help="v3 candidate-anchored gate: keep base answer unless it "
                         "is 'maybe' with passage support kappa_maybe<=tau.")
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
            u_full = get_user_prompt(item['question'], args.datasets,
                                     args.prompt_version, passage=item.get('passage'))
            clean_f = build_view(processor, model, system_prompt, u_full)
            input_len = clean_f['input_ids'].shape[1]

            u_cf = strip_passage_line(u_full)
            clean_c = build_view(processor, model, system_prompt, u_cf)

            cf_proc = CFConfessProcessor(model, tokenizer, clean_c, input_len,
                                         args.lambda_cf, control_rng,
                                         decision=args.decision,
                                         gate_tau_maybe=args.gate_tau_maybe)

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
            print(f'  response[:60]: {response[:60]!r} | dec={rec_extra.get("dec_index")} '
                  f'interv={rec_extra.get("intervened")} | {time.time()-t0:.1f}s', flush=True)
            del clean_f, clean_c
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
