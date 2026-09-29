#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""cf_confess_cmm_qwen3.py -- three-stage CF-Confess: CMM x Qwen3-Omni-30B-A3B.

Multimodal twin of cf_confess_qwen3.py (phd arm) with the cmm two-view
construction of cf_confess_cmm.py (Qwen2.5 arm). Like the Qwen2.5 cmm arm it
keeps the lambda-weighted delta rule (lambda-delta); the sign-based gamma rule
does not recover the multimodal case. The three stages:

  Stage 1  Generate   greedy decode under the full A+V view until the decision
                      token (Qwen3 cmm: boxed tail `\\boxed{`; bare step-0
                      Yes/No kept defensively).
  Stage 2  Confess     teacher-force the two modality-masked views (no-audio,
                      no-video) on the same prefix, giving per-channel pulls
                          d_cf_i = lp_yes(mask-i view) - lp_yes(full view).
  Stage 3  Decide      margin' = margin_full + lambda * (w_a d_cf_a + w_v d_cf_v),
                      force Yes iff margin' > 0 (hard-forced token). Equal
                      weights (w=1.0).

Views: noaudio = video+question (pv, None, iv); novideo = audio+question
(None, av, iv); same question text, media tokens skipped for the masked
modality -- byte-isomorphic to the Qwen2.5 cmm arm via the qwen3_omni_utils
adapter (model IS the thinker, video key pixel_values_videos).

Decision step: Qwen3 cmm answers "Description: ...\nJudgment: \\boxed{Yes|No}"
(716/721; 0 bare) -> the boxed-tail branch of the phd-Qwen3 operator
(decode(gen[-4:]).endswith('\\boxed{')); bare step-0 Yes/No kept defensively.
RUNS ONLY IF the pre-registered probe gate passes (AUC(-d_cf_sum) > 0.65 +
Qwen2.5-type direction structure; see probe_cf_signal_cmm_qwen3.py header).

Qwen3 deltas: thinker-only load, and the eos fix -- Qwen3-Omni
generation_config carries no eos_token_id, so generate() gets
eos_token_id=[<|im_end|>,<|endoftext|>] explicitly.

Controls: --control random_sign (per-sample sign flip of delta, seeded).
Shard interface compatible with shard_data.py / run_cf_confess_8gpu.sh.
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
from cf_common import (set_seed, token_logprobs, load_done_ids,
                       attach_record, write_time)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

YES_ID, NO_ID = 9454, 2753
IM_END, ENDOFTEXT = 151645, 151643          # eos fix (RESULTS.md 2026-08-16)


def build_view(processor, model, system_prompt, pv, av, iv, user_prompt):
    uc = build_user_content_with_media_tokens(pv, av, iv, user_prompt)
    return prepare_for_thinker(build_inputs(
        processor, model, make_messages(system_prompt, uc), pv, av, iv, 'cmm'))


class CFConfessProcessor(LogitsProcessor):
    """Runs Stage 2 (Confess) and Stage 3 (Decide) once, at the decision token
    (boxed tail on Qwen3 cmm). Stage 1 (Generate) is the surrounding greedy
    generate() loop. Keeps the lambda-delta rule (decision='lambda_delta')."""

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
        self.control_rng = control_rng      # None => real delta; else sign flip
        self.used = False
        self.record = None

    @torch.inference_mode()
    def _cf_lp_yes(self, clean, prefix_ids):
        ids = torch.cat([clean['input_ids'][0],
                         torch.tensor(prefix_ids, dtype=torch.long,
                                      device=clean['input_ids'].device)],
                        dim=0).unsqueeze(0)
        kw = dict(input_ids=ids, attention_mask=torch.ones_like(ids))
        for k in ('pixel_values', 'image_grid_thw', 'pixel_values_videos',
                  'video_grid_thw', 'video_second_per_grid',
                  'input_features', 'feature_attention_mask'):
            if k in clean:
                kw[k] = clean[k]
        logits = self.model(**kw).logits[0, -1].float()
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

        # Stage 2 (Confess): full-view margin + per-channel modality-masked pulls.
        lp = torch.log_softmax(logits, dim=-1)
        ly_full, ln_full = lp[YES_ID].item(), lp[NO_ID].item()
        margin_full = ly_full - ln_full

        d_audio = self._cf_lp_yes(self.cf_noaudio, gen) - ly_full
        d_video = self._cf_lp_yes(self.cf_novideo, gen) - ly_full

        # Stage 3 (Decide): lambda-weighted delta, hard-force the token.
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
            decision='lambda_delta',
            lp_yes_full=round(ly_full, 4), lp_no_full=round(ln_full, 4),
            margin_full=round(margin_full, 4),
            d_cf_audio=round(d_audio, 4), d_cf_video=round(d_video, 4),
            w_audio=self.w_audio, w_video=self.w_video,
            delta=round(delta, 4),
            delta_eff=round(delta_eff, 4),
            margin_adj=round(margin_adj, 4),
        )
        return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', type=str, default='cmm')
    ap.add_argument('--data_path', type=str, required=True)
    ap.add_argument('--data_root', type=str, default='./OmniHalluBench/')
    ap.add_argument('--output_dir', type=str, default='./outputs/cmm_cf_confess_qwen3')
    ap.add_argument('--file_name', type=str, default='cf_qwen3')
    ap.add_argument('--time_path', type=str, default='./outputs/cmm_cf_confess_qwen3/time')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--max_new_tokens', type=int, default=300)
    ap.add_argument('--prompt_version', type=str, default='v1')
    ap.add_argument('--lambda_cf', type=float, required=True)
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

    done_ids = load_done_ids(output_path)

    print('[model] loading Qwen3-Omni thinker (~60.5GB, needs a free card)...',
          flush=True)
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
            if pv is None or av is None:
                raise RuntimeError('missing video/audio media')
            u = get_user_prompt(item['question'], args.datasets, args.prompt_version)

            # full view (== base input): video + audio
            clean_f = build_view(processor, model, system_prompt, pv, av, iv, u)
            input_len = clean_f['input_ids'].shape[1]

            # counterfactual views: noaudio (video only), novideo (audio only)
            clean_a = build_view(processor, model, system_prompt, pv, None, iv, u)
            clean_v = build_view(processor, model, system_prompt, None, av, iv, u)

            cf_proc = CFConfessProcessor(model, tokenizer, clean_a, clean_v,
                                         input_len, args.lambda_cf,
                                         args.w_audio, args.w_video, control_rng)

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
                             decision='lambda_delta')
            all_out_tok += len(gen_ids)
            print(f'  response[-60]: {response[-60:]!r} | dec={rec_extra.get("dec_index")} '
                  f'interv={rec_extra.get("intervened")} | {time.time()-t0:.1f}s', flush=True)
            del clean_f, clean_a, clean_v
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f'  [ERROR] {e}', flush=True)
            response = ''
            rec_extra = dict(dec_index=None, intervened=0, error=str(e)[:200],
                             lambda_cf=args.lambda_cf, control=args.control,
                             decision='lambda_delta')

        torch.cuda.empty_cache()

        result = make_result(item, response, args.datasets,
                             f'cf_confess_qwen3_lam{args.lambda_cf}'
                             + ('_randctl' if args.control != 'none' else ''),
                             time_sec=time.time() - t0)
        attach_record(result, rec_extra, gen_ids, logp_list, input_len,
                      args.record_process)
        save_result(result, output_path)

    write_time(args.time_path, args.file_name, t_start, all_out_tok)
    print(f'\n[done] output={output_path}', flush=True)


if __name__ == '__main__':
    main()
