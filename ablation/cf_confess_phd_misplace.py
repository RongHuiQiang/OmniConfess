#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A1 misplacement-intervention arm: identical CF-Confess operator, WRONG
position (pre-registered criterion: F1 < 80 vs CF-at-commitment-point 84.60).

Everything is copied verbatim from cf_confess_phd.py (frozen operator,
lambda=2.0, same prompt, same decision detection), with ONE change:

  the intervention is deferred by exactly --shift (default 5) generated
  tokens after the would-be decision token. At that later position the
  operator recomputes margin_full from the current logits and delta from a
  teacher-forced cf-view forward on the SAME prefix, then hard-forces the
  Yes/No token per margin' = margin + lambda*delta. The decision token
  itself is left greedy (untouched), so any score change is attributable to
  WHERE the signal is injected, not to the signal value.

Records: would_be_dec_index, fire_index, fired (0 if generation ended before
would_be_dec+shift), margin_full/delta/margin_adj at the intervention point,
and whether the forced token differed from the greedy token there.
"""
import os, sys, time, json, argparse, random

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'OmniConfess', 'methods'))
os.chdir(ROOT)

import numpy as np
import torch
from transformers import LogitsProcessor

from shared import (
    load_dataset, load_media_for_item, load_model, build_inputs,
    prepare_for_thinker, build_user_content_with_media_tokens, make_messages,
    get_system_prompt, get_user_prompt, make_result, save_result,
)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

YES_ID, NO_ID = 9454, 2753
MARKER = ("In case there is an inconsistency between the context and the image "
          "content, you should follow the image.")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class CFMisplaceProcessor(LogitsProcessor):
    """Detect the would-be decision step, do nothing there, intervene +shift."""

    def __init__(self, model, tokenizer, cf_clean, prompt_len, lam, shift=5):
        self.model = model
        self.tokenizer = tokenizer
        self.cf_clean = cf_clean
        self.prompt_len = prompt_len
        self.lam = lam
        self.shift = shift
        self.fire_at = None          # step index where we WILL intervene
        self.used = False
        self.record = None

    @torch.inference_mode()
    def _cf_lp_yes_no(self, prefix_ids):
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

    @torch.inference_mode()
    def __call__(self, input_ids, scores):
        if self.used:
            return scores
        gen = input_ids[0, self.prompt_len:].tolist()
        logits = scores[0].float()
        am = int(logits.argmax().item())
        step = len(gen)

        # 1) detect the would-be decision step (verbatim from the frozen operator)
        if self.fire_at is None:
            at_decision = False
            if step == 0:
                at_decision = am in (YES_ID, NO_ID)
            else:
                tail = self.tokenizer.decode(gen[-4:])
                at_decision = tail.endswith('\\boxed{') or tail.endswith('\\boxed{{')
            if at_decision:
                self.fire_at = step + self.shift
                self.record = dict(would_be_dec_index=step, fire_index=self.fire_at,
                                   fired=0)
            return scores

        # 2) at the deferred position, run the identical operator
        if step != self.fire_at:
            return scores
        lp = torch.log_softmax(logits, dim=-1)
        ly_full, ln_full = lp[YES_ID].item(), lp[NO_ID].item()
        margin_full = ly_full - ln_full
        ly_cf, ln_cf = self._cf_lp_yes_no(gen)
        delta = ly_cf - ly_full
        margin_adj = margin_full + self.lam * delta
        forced = YES_ID if margin_adj > 0 else NO_ID

        scores[0][:] = float('-inf')
        scores[0][forced] = 0.0

        self.used = True
        self.record.update(
            fired=1, fire_step=step, greedy_token=am, forced_token=forced,
            intervened=int(forced != am),
            lp_yes_full=round(ly_full, 4), lp_no_full=round(ln_full, 4),
            margin_full=round(margin_full, 4),
            lp_yes_cf=round(ly_cf, 4), lp_no_cf=round(ln_cf, 4),
            delta_cf=round(delta, 4), margin_adj=round(margin_adj, 4),
            lam=self.lam, shift=self.shift)
        return scores


def split_question(q):
    idx = q.find(MARKER)
    if idx < 0:
        return q, q
    return q, q[idx + len(MARKER):].lstrip()


def token_logprobs(gen_ids, scores):
    out = []
    for tid, logits in zip(gen_ids.tolist(), scores):
        logits = logits.float()
        if logits.dim() > 1:
            logits = logits[0]
        if not torch.isfinite(logits[tid]):
            lp = float('nan')
        else:
            lp = torch.log_softmax(logits, dim=-1)[tid]
        out.append(round(lp.item(), 4))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', type=str, default='phd')
    ap.add_argument('--data_path', type=str, required=True)
    ap.add_argument('--data_root', type=str, default='./OmniHalluBench/')
    ap.add_argument('--output_dir', type=str, default='./outputs/phd_cf_confess_ablation')
    ap.add_argument('--file_name', type=str, default='cf_misplace5')
    ap.add_argument('--time_path', type=str, default='./outputs/phd_cf_confess_ablation/time')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--max_new_tokens', type=int, default=300)
    ap.add_argument('--prompt_version', type=str, default='v1')
    ap.add_argument('--lambda_cf', type=float, default=2.0)
    ap.add_argument('--shift', type=int, default=5)
    ap.add_argument('--record_process', action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument('--dry_run', action='store_true')
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.time_path, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.file_name + '.jsonl')

    done_ids = set()
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            for line in f:
                try:
                    r = json.loads(line.strip())
                    done_ids.add((r['id'], r.get('_q', '')))
                except Exception:
                    pass
        print(f'[resume] skipping {len(done_ids)} already-done samples', flush=True)

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
        if (item['id'], item['question'][:80]) in done_ids:
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

            proc = CFMisplaceProcessor(model, tokenizer, clean_cf, input_len,
                                       args.lambda_cf, args.shift)

            with torch.inference_mode():
                out = model.thinker.generate(
                    **clean_f,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    output_scores=True,
                    return_dict_in_generate=True,
                    logits_processor=[proc],
                )
            gen_ids = out.sequences[0][input_len:].cpu()
            if args.record_process:
                logp_list = token_logprobs(gen_ids, out.scores)
            response = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            rec_extra = proc.record or dict(would_be_dec_index=None, fired=0)
            rec_extra.update(lambda_cf=args.lambda_cf, shift=args.shift,
                             arm='misplaced_intervention')
            all_out_tok += len(gen_ids)
            print(f'  resp[-50]: {response[-50:]!r} | dec={rec_extra.get("would_be_dec_index")} '
                  f'fired={rec_extra.get("fired")} | {time.time()-t0:.1f}s', flush=True)
            del clean_f, clean_cf
        except Exception as e:
            import traceback; traceback.print_exc()
            response = ''
            rec_extra = dict(would_be_dec_index=None, fired=0, error=str(e)[:200])

        torch.cuda.empty_cache()

        result = make_result(item, response, args.datasets, 'cf_misplaced', time_sec=time.time() - t0)
        result['_q'] = item['question'][:80]
        if args.record_process:
            result['record_process'] = True
            result['cf'] = rec_extra
            if gen_ids is not None:
                lps = [x for x in (logp_list or []) if not (isinstance(x, float) and x != x)]
                cum = float(np.nansum(logp_list))
                result['token_ids'] = gen_ids.tolist()
                result['logp_list'] = logp_list
                result['cumulative_logprob'] = round(cum, 4)
                result['norm_logp'] = round(cum / (len(gen_ids) + 1e-8), 4)
                result['input_len'] = int(input_len)
                result['output_tokens'] = int(len(gen_ids))
        save_result(result, output_path)

    with open(os.path.join(args.time_path, args.file_name + '.txt'), 'w') as f:
        f.write(f'total_time: {time.time()-t_start:.2f}s\ntotal_output_tokens: {all_out_tok}\n')
    print(f'\n[done] output={output_path}', flush=True)


if __name__ == '__main__':
    main()
