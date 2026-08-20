#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""cf_confess_cmm.py -- CF-Confess (counterfactual confession) operator for CMM.

Multimodal twin of cf_confess_phd.py: two counterfactual views (one per
modality) instead of one, combined with per-view weights

    d_cf_i    = log P(Yes | modality-i-masked view, prefix) - log P(Yes | full view, prefix)
    delta     = sum_i w_i * d_cf_i                    (w_audio = w_video = 1, equal)
    margin'   = margin_full + lambda * delta          margin_full = lp_yes - lp_no
    decision  = Yes if margin' > 0 else No            (hard-forced token)

Positive d_cf_i = masking modality i DROPS Yes = modality i was inflating Yes;
the operator therefore pushes Yes DOWN where a single modality props it up
(the cmm failure mode: unimodal-prior yes answers) and up where a modality
suppresses it. delta is the w_i-weighted sum, equal weights first (probe
section of EXPERIMENT_DESIGN.md documents the pre-registration).

Decision step := no tokens generated yet and greedy argmax is Yes/No
(cmm base: 720/721 bare step-0 answers), or generated text ends with
`\\boxed{` (rare). One intervention per sample; trajectory recorded as in
base_model.py plus the cf internals.

Controls: --control random_sign flips the sign of delta (seeded, per sample).

Interface compatible with shard_data.py / run_cf_confess_8gpu.sh
(--data_path/--file_name/--output_dir/--time_path) and resume-by-id.
"""
import os, sys, time, json, argparse, random

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'OmniConfess', 'methods'))

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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_view(processor, model, system_prompt, pv, av, iv, user_prompt):
    uc = build_user_content_with_media_tokens(pv, av, iv, user_prompt)
    return prepare_for_thinker(
        build_inputs(processor, model, make_messages(system_prompt, uc),
                     pv, av, iv, 'cmm'), model)


class CFConfessProcessor(LogitsProcessor):
    """Intervenes exactly once, at the decision token (see module docstring)."""

    def __init__(self, model, tokenizer, cf_noaudio, cf_novideo, prompt_len,
                 lam, w_audio=1.0, w_video=1.0, control_rng=None):
        self.model = model
        self.tokenizer = tokenizer
        self.cf_noaudio = cf_noaudio
        self.cf_novideo = cf_novideo
        self.prompt_len = prompt_len
        self.lam = lam
        self.w_audio = w_audio
        self.w_video = w_video
        self.control_rng = control_rng      # None => real Delta; else sign flip
        self.used = False
        self.record = None

    @torch.inference_mode()
    def _cf_lp_yes(self, clean, prefix_ids):
        ids = torch.cat([clean['input_ids'][0],
                         torch.tensor(prefix_ids, dtype=torch.long,
                                      device=clean['input_ids'].device)],
                        dim=0).unsqueeze(0)
        kw = dict(input_ids=ids, attention_mask=torch.ones_like(ids))
        for k in ('pixel_values_images', 'pixel_values', 'image_grid_thw',
                  'pixel_values_videos', 'video_grid_thw',
                  'video_second_per_grid', 'input_features',
                  'feature_attention_mask'):
            if k in clean:
                kw[k] = clean[k]
        logits = self.model.thinker(**kw).logits[0, -1].float()
        return torch.log_softmax(logits, dim=-1)[YES_ID].item()

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

        lp = torch.log_softmax(logits, dim=-1)
        ly_full, ln_full = lp[YES_ID].item(), lp[NO_ID].item()
        margin_full = ly_full - ln_full

        d_audio = self._cf_lp_yes(self.cf_noaudio, gen) - ly_full
        d_video = self._cf_lp_yes(self.cf_novideo, gen) - ly_full
        delta = self.w_audio * d_audio + self.w_video * d_video
        delta_eff = delta
        if self.control_rng is not None:
            delta_eff = delta * self.control_rng.choice([-1.0, 1.0])
        margin_adj = margin_full + self.lam * delta_eff
        forced = YES_ID if margin_adj > 0 else NO_ID

        scores[0][:] = float('-inf')
        scores[0][forced] = 0.0          # hard-force the decision token

        self.used = True
        self.record = dict(
            dec_index=len(gen),
            greedy_token=am,
            forced_token=forced,
            intervened=int(forced != am),
            lp_yes_full=round(ly_full, 4), lp_no_full=round(ln_full, 4),
            margin_full=round(margin_full, 4),
            d_cf_audio=round(d_audio, 4), d_cf_video=round(d_video, 4),
            w_audio=self.w_audio, w_video=self.w_video,
            delta=round(delta, 4),
            delta_eff=round(delta_eff, 4),
            margin_adj=round(margin_adj, 4),
        )
        return scores


def token_logprobs(gen_ids, scores):
    out = []
    for tid, logits in zip(gen_ids.tolist(), scores):
        logits = logits.float()
        if logits.dim() > 1:
            logits = logits[0]
        if not torch.isfinite(logits[tid]):
            lp = float('nan')      # decision step under intervention
        else:
            lp = torch.log_softmax(logits, dim=-1)[tid]
        out.append(round(lp.item(), 4))
    return out


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
    ap.add_argument('--w_audio', type=float, default=1.0)
    ap.add_argument('--w_video', type=float, default=1.0)
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
            pv, av, iv = load_media_for_item(item, args.data_path, args.data_root)
            u = get_user_prompt(item['question'], args.datasets, args.prompt_version)

            # full view (== base_model input): video + audio
            clean_f = build_view(processor, model, system_prompt, pv, av, iv, u)
            input_len = clean_f['input_ids'].shape[1]

            # counterfactual views: noaudio (video only), novideo (audio only)
            clean_a = build_view(processor, model, system_prompt, pv, None, iv, u)
            clean_v = build_view(processor, model, system_prompt, None, av, iv, u)

            cf_proc = CFConfessProcessor(model, tokenizer, clean_a, clean_v,
                                         input_len, args.lambda_cf,
                                         args.w_audio, args.w_video, control_rng)

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
