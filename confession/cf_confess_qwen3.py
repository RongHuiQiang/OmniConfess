#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""cf_confess_qwen3.py -- three-stage counterfactual confession on
Qwen3-Omni-30B-A3B (cross-backbone port of cf_confess_phd.py).

Same three stages as the Qwen2.5 phd arm:
  Stage 1  Generate   greedy decode under the full view until the decision
                      token (bare Yes/No at step 0, or right after `\\boxed{`).
  Stage 2  Confess     teacher-force the context-removed view (image + question)
                      on the same prefix, read (lp_yes, lp_no).
  Stage 3  Decide      --decision gamma (default): margin' = ly_cf - ln_cf;
                      --decision lambda: margin_full + lambda * delta_cf.

Qwen3 deltas vs the Qwen2.5 arm: thinker-only load via
shared/qwen3_omni_utils (model IS the thinker), pixel key ``pixel_values``,
and the eos fix -- Qwen3-Omni generation_config carries no eos_token_id, so
generate() must be given eos_token_id=[<|im_end|>,<|endoftext|>] explicitly
(otherwise replies run to the cap, see RESULTS.md 2026-08-16 note).

Controls: --control random_sign (per-sample sign flip of delta, seeded).
Shard interface compatible with shard_data.py / run_text_8gpu.sh.
"""
import os, sys, time, argparse, random

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'OmniConfess', 'methods'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from transformers import LogitsProcessor

from shared import (
    load_dataset, load_media_for_item, get_system_prompt, get_user_prompt,
    make_result, save_result,
)
from shared.qwen3_omni_utils import (
    load_model, build_inputs, prepare_for_thinker,
    build_user_content_with_media_tokens, make_messages,
)
from gamma_operator import gamma_margin_phd    # Stage-3 lambda-free gamma rule
from cf_common import (set_seed, token_logprobs, split_question,
                       load_done_ids, attach_record, write_time)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

YES_ID, NO_ID = 9454, 2753
IM_END, ENDOFTEXT = 151645, 151643          # eos fix (RESULTS.md 2026-08-16)


class CFConfessProcessor(LogitsProcessor):
    """Runs Stage 2 (Confess) and Stage 3 (Decide) once, at the decision token
    (same protocol as the Qwen2.5 phd arm).

    decision:
        'gamma'  -> lambda-free gamma_margin_phd (deployed rule, tau=0).
        'lambda' -> margin_full + lam * Delta_cf (ablation / random_sign arm).
    """

    def __init__(self, model, tokenizer, cf_clean, prompt_len, lam,
                 control_rng=None, decision='gamma'):
        self.model = model
        self.tokenizer = tokenizer
        self.cf_clean = cf_clean
        self.prompt_len = prompt_len
        self.lam = lam
        self.control_rng = control_rng
        self.decision = decision
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
        for k in ('pixel_values', 'image_grid_thw', 'pixel_values_videos',
                  'video_grid_thw', 'video_second_per_grid',
                  'input_features', 'feature_attention_mask'):
            if k in self.cf_clean:
                kw[k] = self.cf_clean[k]
        logits = self.model(**kw).logits[0, -1].float()
        lp = torch.log_softmax(logits, dim=-1)
        return lp[YES_ID].item(), lp[NO_ID].item()

    def _decide(self, ly_full, ln_full, ly_cf, ln_cf):
        """Stage 3: return (forced_token, margin', delta_eff)."""
        delta = ly_cf - ly_full
        if self.decision == 'gamma' and self.control_rng is None:
            margin_prime, forced = gamma_margin_phd(ly_full, ln_full,
                                                    ly_cf, ln_cf)
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
            at_decision = am in (YES_ID, NO_ID)
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
        forced, margin_prime, delta_eff = self._decide(ly_full, ln_full,
                                                        ly_cf, ln_cf)
        scores[0][:] = float('-inf')
        scores[0][forced] = 0.0

        self.used = True
        self.record = dict(
            dec_index=len(gen), greedy_token=am, forced_token=forced,
            intervened=int(forced != am), decision=self.decision,
            lp_yes_full=round(ly_full, 4), lp_no_full=round(ln_full, 4),
            margin_full=round(margin_full, 4),
            lp_yes_cf=round(ly_cf, 4), lp_no_cf=round(ln_cf, 4),
            delta_cf=round(ly_cf - ly_full, 4), delta_eff=round(delta_eff, 4),
            margin_adj=round(margin_prime, 4),
        )
        return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', type=str, default='phd')
    ap.add_argument('--data_path', type=str, required=True)
    ap.add_argument('--data_root', type=str, default='./OmniHalluBench/')
    ap.add_argument('--output_dir', type=str, default='./outputs/phd_cf_confess_qwen3')
    ap.add_argument('--file_name', type=str, default='cf_qwen3')
    ap.add_argument('--time_path', type=str, default='./outputs/phd_cf_confess_qwen3/time')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--max_new_tokens', type=int, default=300)
    ap.add_argument('--prompt_version', type=str, default='v1')
    ap.add_argument('--lambda_cf', type=float, required=True)
    ap.add_argument('--decision', choices=['gamma', 'lambda'], default='gamma',
                    help="Stage-3 rule: 'gamma' = deployed lambda-free rule "
                         "(default), 'lambda' = legacy margin+lambda*Delta.")
    ap.add_argument('--control', choices=['none', 'random_sign'], default='none')
    ap.add_argument('--record_process', action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument('--dry_run', action='store_true')
    ap.add_argument('--limit', type=int, default=0,
                    help='first N items only (smoke; added 2026-08-31 for the '
                         'CF_OVERRIDE unified smoke, run semantics unchanged)')
    args = ap.parse_args()

    set_seed(args.seed)
    control_rng = random.Random(args.seed + 7) if args.control == 'random_sign' else None
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.time_path, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.file_name + '.jsonl')

    done_ids = load_done_ids(output_path)

    print('[model] loading Qwen3-Omni thinker (~60.5GB, needs a free card)...', flush=True)
    processor, model = load_model()
    tokenizer = processor.tokenizer

    data = load_dataset(args.data_path, args.data_root)
    if args.dry_run:
        data = data[:3]
    if args.limit:
        data = data[:args.limit]

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
            clean_f = prepare_for_thinker(build_inputs(
                processor, model, make_messages(
                    system_prompt,
                    build_user_content_with_media_tokens(pv, av, iv, u_full)),
                pv, av, iv, args.datasets))
            input_len = clean_f['input_ids'].shape[1]

            u_cf = get_user_prompt(q_only, args.datasets, args.prompt_version)
            clean_cf = prepare_for_thinker(build_inputs(
                processor, model, make_messages(
                    system_prompt,
                    build_user_content_with_media_tokens(pv, av, iv, u_cf)),
                pv, av, iv, args.datasets))

            cf_proc = CFConfessProcessor(model, tokenizer, clean_cf, input_len,
                                         args.lambda_cf, control_rng,
                                         decision=args.decision)

            with torch.inference_mode():
                out = model.generate(
                    **clean_f,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    eos_token_id=[IM_END, ENDOFTEXT],      # eos fix
                    pad_token_id=IM_END,
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

        method = ('cf_confess_qwen3_gamma' if args.decision == 'gamma'
                  else f'cf_confess_qwen3_lam{args.lambda_cf}')
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
