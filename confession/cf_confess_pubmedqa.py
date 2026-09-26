#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""cf_confess_pubmedqa.py -- CF-Confess (counterfactual confession) operator
for PubMedQA. Text-only twin of cf_confess_phd.py with a 3-class decision.

pubmedqa base answers begin with a bare 'Yes.' / 'No.' / 'Maybe.' as the very
first generated token (401/57/10 of 468), so the decision step is step 0.

Single counterfactual view (passage removed -- the only view a text dataset
has, hence the pre-registered WEAK-effect prediction, no second view to
cross-check):

    d_cf     = log P(Yes | no-passage view, prefix) - log P(Yes | full view, prefix)
    yes'     = lp_yes_full + lambda * d_cf          (Yes logit shifted only,
              the phd margin shift generalised to 3 classes)
    decision = argmax{ yes' , lp_no_full , lp_maybe_full }, hard-forced token

Controls: --control random_sign flips the sign of d_cf per sample (seeded).

Interface compatible with shard_data.py / run_cf_confess_8gpu.sh and
resume-by-id.
"""
import os, sys, time, json, argparse, random

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'OmniConfess', 'methods'))

import numpy as np
import torch
from transformers import LogitsProcessor

from shared import (
    load_dataset, load_model, build_inputs,
    prepare_for_thinker, build_user_content_with_media_tokens, make_messages,
    get_system_prompt, get_user_prompt, make_result, save_result,
)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

YES_ID, NO_ID, MAYBE_ID = 9454, 2753, 21390


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
    """Intervenes exactly once: at step 0 if the greedy argmax is Yes/No/Maybe
    (the pubmedqa base output shape, 468/468)."""

    def __init__(self, model, tokenizer, cf_clean, prompt_len, lam,
                 control_rng=None):
        self.model = model
        self.tokenizer = tokenizer
        self.cf_clean = cf_clean
        self.prompt_len = prompt_len
        self.lam = lam
        self.control_rng = control_rng
        self.used = False
        self.record = None

    @torch.inference_mode()
    def _cf_lps(self, prefix_ids):
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

        lp = torch.log_softmax(logits, dim=-1)
        ly, ln, lm = lp[YES_ID].item(), lp[NO_ID].item(), lp[MAYBE_ID].item()
        ly_cf, ln_cf, lm_cf = self._cf_lps(gen)
        d_cf = ly_cf - ly
        d_eff = d_cf
        if self.control_rng is not None:
            d_eff = d_cf * self.control_rng.choice([-1.0, 1.0])
        yes_adj = ly + self.lam * d_eff
        choices = {YES_ID: yes_adj, NO_ID: ln, MAYBE_ID: lm}
        forced = max(choices, key=choices.get)

        scores[0][:] = float('-inf')
        scores[0][forced] = 0.0

        self.used = True
        self.record = dict(
            dec_index=0,
            greedy_token=am,
            forced_token=forced,
            intervened=int(forced != am),
            lp_yes_full=round(ly, 4), lp_no_full=round(ln, 4),
            lp_maybe_full=round(lm, 4),
            lp_yes_cf=round(ly_cf, 4), lp_no_cf=round(ln_cf, 4),
            lp_maybe_cf=round(lm_cf, 4),
            d_cf=round(d_cf, 4), d_eff=round(d_eff, 4),
            yes_adj=round(yes_adj, 4),
        )
        return scores


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
    ap.add_argument('--control', choices=['none', 'random_sign'], default='none')
    ap.add_argument('--record_process', action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument('--dry_run', action='store_true')
    args = ap.parse_args()

    set_seed(args.seed)
    control_rng = random.Random(args.seed + 7) if args.control == 'random_sign' else None
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.time_path, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.file_name + '.jsonl')

    done_ids = set()
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line.strip())['id'])
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
                                         args.lambda_cf, control_rng)

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
            print(f'  response[:60]: {response[:60]!r} | dec={rec_extra.get("dec_index")} '
                  f'interv={rec_extra.get("intervened")} | {time.time()-t0:.1f}s', flush=True)
            del clean_f, clean_c
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
        if args.record_process:
            result['record_process'] = True
            result['cf'] = rec_extra
            if gen_ids is not None:
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
