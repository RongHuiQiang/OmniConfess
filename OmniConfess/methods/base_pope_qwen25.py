#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""base_pope_qwen25.py -- POPE (public-benchmark arm) runner for Qwen2.5-Omni-7B.

POPE: object-hallucination yes/no probing on COCO val2014 (Li et al., EMNLP
2023, arXiv 2305.10315), three settings: random / popular / adversarial.
Data: /slow_share/tsq/datasets/pope/pope_<setting>.json (500 items each,
stratified 250 yes / 250 no), built by scripts/prep_pope_data.py from the
public lmms-lab/POPE release. This is the external-validity check against the
"self-built benchmark" objection; only the judgement-family arms live here.

Arms
-----
base : single-pass greedy, identical prompt to base_model.py's judgement path
       (datasets name "pope", see shared/prompts.py). A recording logits
       processor stores the decision-token margins lp_yes/lp_no (full view) so
       any CF lambda can be re-decided offline.
cf   : CF-Confess judgement-family operator, POPE reading. In phd/cmm the
       counterfactual view removes the misleading *context* and keeps the
       image; POPE has no context -- the potentially misleading input IS the
       image (the question asks about an object the image may not contain).
       So the cf view masks the image (question-only):
         full : image + question        cf : question only (no media tokens)
         Delta_cf = lp_yes(cf) - lp_yes(full)
         margin' = margin_full + lambda * Delta_cf
       decision token (Yes/No) forced by margin', then greedy continuation.
       lambda: --lambda_cf float. The Z12 rule value
         lambda_hat = 4.41 - 0.0477 * base_yes_rate
       is computed by scripts/pope_score.py from the base arm and passed in;
       the oracle lambda* is swept offline from the recorded margins.

Usage:
  python OmniConfess/methods/base_pope_qwen25.py --arm base \
      --data_path /slow_share/tsq/datasets/pope/pope_random.json \
      --output_dir outputs/pope --file_name pope_base_random
  python OmniConfess/methods/base_pope_qwen25.py --arm cf --lambda_cf 1.7 ...
"""
import os, sys, time, json, argparse, random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from transformers import LogitsProcessor

from shared import (
    load_dataset, load_media_for_item, load_model, build_inputs,
    prepare_for_thinker, build_user_content_with_media_tokens, make_messages,
    get_system_prompt, get_user_prompt, parse_pred, make_result, save_result,
)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

YES_ID, NO_ID = 9454, 2753


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MarginRecorder(LogitsProcessor):
    """Records lp_yes/lp_no at step 0 (the decision token for base greedy)."""

    def __init__(self):
        self.record = None

    @torch.inference_mode()
    def __call__(self, input_ids, scores):
        if self.record is None:
            lp = torch.log_softmax(scores[0].float(), dim=-1)
            self.record = dict(
                lp_yes_full=round(lp[YES_ID].item(), 4),
                lp_no_full=round(lp[NO_ID].item(), 4),
                margin_full=round((lp[YES_ID] - lp[NO_ID]).item(), 4),
            )
        return scores


class CFConfessProcessor(LogitsProcessor):
    """POPE CF operator: intervenes once at the decision token.

    Decision step := no tokens generated yet and greedy argmax is Yes/No, or
    the generated text so far ends with `\\boxed{` (kept for prompt symmetry;
    the pope prompt elicits a bare Yes/No).
    """

    def __init__(self, model, tokenizer, cf_clean, prompt_len, lam):
        self.model = model
        self.tokenizer = tokenizer
        self.cf_clean = cf_clean
        self.prompt_len = prompt_len
        self.lam = lam
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

        at_decision = False
        if len(gen) == 0:
            at_decision = am in (YES_ID, NO_ID)
        else:
            tail = self.tokenizer.decode(gen[-4:])
            at_decision = tail.endswith('\\boxed{') or tail.endswith('\\boxed{{')
        if not at_decision:
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
        self.record = dict(
            dec_index=len(gen),
            greedy_token=am,
            forced_token=forced,
            intervened=int(forced != am),
            lp_yes_full=round(ly_full, 4), lp_no_full=round(ln_full, 4),
            margin_full=round(margin_full, 4),
            lp_yes_cf=round(ly_cf, 4), lp_no_cf=round(ln_cf, 4),
            delta_cf=round(delta, 4),
            margin_adj=round(margin_adj, 4),
        )
        return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=['base', 'cf'], required=True)
    ap.add_argument('--datasets', type=str, default='pope')
    ap.add_argument('--data_path', required=True)
    ap.add_argument('--data_root', type=str, default='/slow_share/tsq/datasets/pope/')
    ap.add_argument('--output_dir', default='./outputs/pope')
    ap.add_argument('--file_name', default=None)
    ap.add_argument('--time_path', default='./outputs/pope/time')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--max_new_tokens', type=int, default=32)
    ap.add_argument('--prompt_version', type=str, default='v1')
    ap.add_argument('--lambda_cf', type=float, default=2.0)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    if args.file_name is None:
        setting = os.path.basename(args.data_path).replace('.json', '').split('_')[-1]
        args.file_name = f'pope_{args.arm}' + (f'_lam{args.lambda_cf}' if args.arm == 'cf' else '') + f'_{setting}'

    set_seed(args.seed)
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
    if args.limit:
        data = data[:args.limit]

    system_prompt = get_system_prompt(args.datasets, args.prompt_version)

    t_start = time.time()
    n_done = 0
    for idx, item in enumerate(data):
        if item['id'] in done_ids:
            continue
        t0 = time.time()
        print(f'\n[{idx+1}/{len(data)}] id={item["id"]}', flush=True)

        rec_extra, response = {}, ''
        try:
            pv, av, iv = load_media_for_item(item, args.data_path, args.data_root)
            u_full = get_user_prompt(item['question'], args.datasets, args.prompt_version)
            uc_full = build_user_content_with_media_tokens(pv, av, iv, u_full)
            clean_f = prepare_for_thinker(
                build_inputs(processor, model,
                             make_messages(system_prompt, uc_full),
                             pv, av, iv, args.datasets), model)
            input_len = clean_f['input_ids'].shape[1]

            if args.arm == 'cf':
                # counterfactual view: question only, image masked
                uc_cf = u_full
                clean_cf = prepare_for_thinker(
                    build_inputs(processor, model,
                                 make_messages(system_prompt, uc_cf),
                                 None, None, None, args.datasets), model)
                proc = CFConfessProcessor(model, tokenizer, clean_cf,
                                          input_len, args.lambda_cf)
            else:
                proc = MarginRecorder()

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
            response = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            rec_extra = proc.record or {}
            rec_extra.update(arm=args.arm,
                             lambda_cf=(args.lambda_cf if args.arm == 'cf' else None))
            print(f'  resp[:60]: {response[:60]!r} | '
                  f'{"interv=" + str(rec_extra.get("intervened")) if args.arm == "cf" else ""} '
                  f'| {time.time()-t0:.1f}s', flush=True)
            if args.arm == 'cf':
                del clean_cf
            del clean_f
        except Exception as e:
            import traceback; traceback.print_exc()
            rec_extra = dict(arm=args.arm, error=str(e)[:200])

        torch.cuda.empty_cache()
        n_done += 1

        result = make_result(item, response, args.datasets,
                             f'pope_{args.arm}' + (f'_lam{args.lambda_cf}' if args.arm == 'cf' else ''),
                             time_sec=time.time() - t0)
        result['cf'] = rec_extra
        result['pope_setting'] = item.get('group_key', '')
        save_result(result, output_path)

    with open(os.path.join(args.time_path, args.file_name + '.txt'), 'w') as f:
        f.write(f'total_time: {time.time()-t_start:.2f}s\nsamples_done: {n_done}\n')
    print(f'\n[done] output={output_path}', flush=True)


if __name__ == '__main__':
    main()
