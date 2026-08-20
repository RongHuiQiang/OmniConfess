#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""cf_confess_qwen3.py -- CF-Confess operator on Qwen3-Omni-30B-A3B (cross-backbone).

Same operator as cf_confess_phd.py (Qwen2.5 arm), ported to the Qwen3 backbone:
  greedy (shared v1 prompt, same as base_model_qwen3_omni) -> at the decision
  token (bare Yes/No at step 0, or right after `\\boxed{`) compute
    margin_full = lp_yes - lp_no            (full view, generate logits)
    delta_cf    = lp_yes(cf view) - lp_yes(full view),  cf = image+question only
    margin'     = margin_full + lambda * delta_cf
  -> force Yes if margin' > 0 else No, continue greedy.

Qwen3 deltas vs the Qwen2.5 arm: thinker-only load via
shared/qwen3_omni_utils (model IS the thinker), pixel key ``pixel_values``,
and the eos fix -- Qwen3-Omni generation_config carries no eos_token_id, so
generate() must be given eos_token_id=[<|im_end|>,<|endoftext|>] explicitly
(otherwise replies run to the cap, see RESULTS.md 2026-08-16 note).

Controls: --control random_sign (per-sample sign flip of delta, seeded).
Shard interface compatible with shard_data.py / run_text_8gpu.sh.
"""
import os, sys, time, json, argparse, random

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'OmniConfess', 'methods'))

import numpy as np
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

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

YES_ID, NO_ID = 9454, 2753
IM_END, ENDOFTEXT = 151645, 151643          # eos fix (RESULTS.md 2026-08-16)
MARKER = ("In case there is an inconsistency between the context and the image "
          "content, you should follow the image.")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class CFConfessProcessor(LogitsProcessor):
    """One-shot intervention at the decision token (same protocol as Qwen2.5 arm)."""

    def __init__(self, model, tokenizer, cf_clean, prompt_len, lam, control_rng=None):
        self.model = model
        self.tokenizer = tokenizer
        self.cf_clean = cf_clean
        self.prompt_len = prompt_len
        self.lam = lam
        self.control_rng = control_rng
        self.used = False
        self.record = None

    @torch.inference_mode()
    def _cf_lp_yes_no(self, prefix_ids):
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
        delta_eff = delta
        if self.control_rng is not None:
            delta_eff = delta * self.control_rng.choice([-1.0, 1.0])
        margin_adj = margin_full + self.lam * delta_eff
        forced = YES_ID if margin_adj > 0 else NO_ID

        scores[0][:] = float('-inf')
        scores[0][forced] = 0.0

        self.used = True
        self.record = dict(
            dec_index=len(gen), greedy_token=am, forced_token=forced,
            intervened=int(forced != am),
            lp_yes_full=round(ly_full, 4), lp_no_full=round(ln_full, 4),
            margin_full=round(margin_full, 4),
            lp_yes_cf=round(ly_cf, 4), lp_no_cf=round(ln_cf, 4),
            delta_cf=round(delta, 4), delta_eff=round(delta_eff, 4),
            margin_adj=round(margin_adj, 4),
        )
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
    ap.add_argument('--output_dir', type=str, default='./outputs/phd_cf_confess_qwen3')
    ap.add_argument('--file_name', type=str, default='cf_qwen3')
    ap.add_argument('--time_path', type=str, default='./outputs/phd_cf_confess_qwen3/time')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--max_new_tokens', type=int, default=300)
    ap.add_argument('--prompt_version', type=str, default='v1')
    ap.add_argument('--lambda_cf', type=float, required=True)
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

    print('[model] loading Qwen3-Omni thinker (~60.5GB, needs a free card)...', flush=True)
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
                                         args.lambda_cf, control_rng)

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
            rec_extra.update(lambda_cf=args.lambda_cf, control=args.control)
            all_out_tok += len(gen_ids)
            print(f'  response[-60]: {response[-60:]!r} | dec={rec_extra.get("dec_index")} '
                  f'interv={rec_extra.get("intervened")} | {time.time()-t0:.1f}s', flush=True)
            del clean_f, clean_cf
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f'  [ERROR] {e}', flush=True)
            response = ''
            rec_extra = dict(dec_index=None, intervened=0, error=str(e)[:200],
                             lambda_cf=args.lambda_cf, control=args.control)

        torch.cuda.empty_cache()

        result = make_result(item, response, args.datasets,
                             f'cf_confess_qwen3_lam{args.lambda_cf}'
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
