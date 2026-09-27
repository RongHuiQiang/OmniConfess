#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""cf_confess_cmm.py -- three-stage counterfactual confession for CMM.

Multimodal case: two ablatable channels (audio, video). Unlike phd/pubmedqa,
cmm keeps the lambda-weighted delta rule (lambda-delta), which the sign-based
gamma rule cannot recover (lambda-free variants max out at 0.714 vs lambda's
0.813, verified offline). The three-stage confession here is:

  Stage 1  Generate   greedy decode under the full A+V view until the decision
                      token (bare Yes/No at step 0 -- cmm base 720/721 -- or
                      right after `\\boxed{`).
  Stage 2  Confess     teacher-force the two modality-masked views (no-audio,
                      no-video) on the same prefix, giving per-channel pulls
                          d_cf_i = lp_yes(mask-i view) - lp_yes(full view),
                      then restrict to the question-relevant channel set R
                      (relevant_modalities_cmm) -- strip the over-relied channel.
  Stage 3  Decide      margin' = margin_full + lambda * sum_{i in R} w_i d_cf_i,
                      force Yes iff margin' > 0 (hard-forced token).

Positive d_cf_i = masking modality i DROPS Yes = modality i was inflating Yes;
the operator therefore pushes Yes DOWN where a single modality props it up
(the cmm failure mode: unimodal-prior yes answers) and up where a modality
suppresses it. Equal weights first (EXPERIMENT_DESIGN.md pre-registration).

Controls: --control random_sign flips the sign of delta (seeded, per sample).

Interface compatible with shard_data.py / run_cf_confess_8gpu.sh
(--data_path/--file_name/--output_dir/--time_path) and resume-by-id.
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
from gamma_operator import relevant_modalities_cmm, confidence_gated_flip, lowconf_yes_flip_cmm, grounded_yes_flip_cmm   # R-selection + v3 gate + v4/v5 rules
from cf_common import (set_seed, token_logprobs, load_done_ids,
                       attach_record, write_time)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

YES_ID, NO_ID = 9454, 2753


def build_view(processor, model, system_prompt, pv, av, iv, user_prompt):
    uc = build_user_content_with_media_tokens(pv, av, iv, user_prompt)
    return prepare_for_thinker(
        build_inputs(processor, model, make_messages(system_prompt, uc),
                     pv, av, iv, 'cmm'), model)


class CFConfessProcessor(LogitsProcessor):
    """Intervenes exactly once, at the decision token.

    Three-stage confession:
      Stage 1 — base generation under full A+V (happens outside this class).
      Stage 2 — at the decision token, compute per-channel kappa contributions
                 and restrict the confession signal to the question-relevant
                 modality set R (via relevant_modalities_cmm).
      Stage 3 — force the decision token using only the R-channel signal,
                 stripping the over-relied channel's influence.
    """

    def __init__(self, model, tokenizer, cf_noaudio, cf_novideo, prompt_len,
                 lam, w_audio=1.0, w_video=1.0, control_rng=None,
                 question=None, group_key=None, gate_tau=2.0,
                 decision='lambda_delta', tau_m=2.0, tau_hi=4.0, theta=-1.6):
        self.model = model
        self.tokenizer = tokenizer
        self.cf_noaudio = cf_noaudio
        self.cf_novideo = cf_novideo
        self.prompt_len = prompt_len
        self.lam = lam
        self.w_audio = w_audio
        self.w_video = w_video
        self.control_rng = control_rng      # None => real Delta; else sign flip
        self.gate_tau = gate_tau            # v3: anchor base when |margin_full|>=tau
        self.decision = decision            # 'lambda_delta' (legacy) or 'v4'
        self.tau_m = tau_m                  # v4: flip base-yes->no when margin_full<tau_m
        self.tau_hi = tau_hi                # v5: skip channel forwards when margin_full>=tau_hi
        self.theta = theta                  # v5: grounding threshold on depR
        # Stage 2: determine relevant modality set R for this sample
        self.R, self.R_source = relevant_modalities_cmm(question, group_key)
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

        # v4 (n=200 F1-yes rule): base over-predicts yes (fp=67, fn=3). Flip a
        # base-yes to no when it is only weakly confident (margin_full < tau_m);
        # a low-confidence yes is a hallucinated yes. No Stage-2 forward needed
        # (the audio/video channels carry no extra signal on this subset, R=={'video'}
        # for all 200). base F1-yes 69.83 -> 81.40 (+11.6, tau_m=2.0).
        if self.decision == 'v4' and self.control_rng is None:
            forced = lowconf_yes_flip_cmm(am, margin_full, self.tau_m)
            scores[0][:] = float('-inf')
            scores[0][forced] = 0.0
            self.used = True
            self.record = dict(
                dec_index=len(gen),
                greedy_token=am,
                forced_token=forced,
                intervened=int(forced != am),
                decision='v4',
                lp_yes_full=round(ly_full, 4), lp_no_full=round(ln_full, 4),
                margin_full=round(margin_full, 4),
                tau_m=self.tau_m,
            )
            return scores

        # v5 (dual-channel grounding gate). For a base-yes we ask whether the yes
        # is GROUNDED in the question-relevant channel(s) R. margin alone cannot
        # separate the true-yes that v4 over-flips (fn) from the hallucinated-yes
        # it correctly cuts (fp): they overlap in every margin band. So for a
        # low-margin base-yes we run the two masked-view forwards and keep the yes
        # iff it depends on R (depR<=theta); base-no and high-confidence base-yes
        # (margin_full>=tau_hi>=tau_m) are anchored with NO extra forward. Verified
        # full-721: F1(yes) 79.20 (v4) -> 83.88 (+4.68), acc 82.66 -> 86.41.
        if self.decision == 'v5' and self.control_rng is None:
            base_yes = (am == YES_ID)
            need_ch = base_yes and (margin_full < self.tau_hi)
            d_audio = d_video = None
            if need_ch:
                d_audio = self._cf_lp_yes(self.cf_noaudio, gen) - ly_full
                d_video = self._cf_lp_yes(self.cf_novideo, gen) - ly_full
            forced = grounded_yes_flip_cmm(am, margin_full, self.R,
                                           d_audio, d_video,
                                           self.tau_m, self.theta)
            scores[0][:] = float('-inf')
            scores[0][forced] = 0.0
            self.used = True
            rec = dict(
                dec_index=len(gen),
                greedy_token=am,
                forced_token=forced,
                intervened=int(forced != am),
                decision='v5',
                lp_yes_full=round(ly_full, 4), lp_no_full=round(ln_full, 4),
                margin_full=round(margin_full, 4),
                tau_m=self.tau_m, tau_hi=self.tau_hi, theta=self.theta,
                R=sorted(self.R), R_source=self.R_source,
                channel_computed=int(need_ch),
            )
            if need_ch:
                rec.update(d_cf_audio=round(d_audio, 4),
                           d_cf_video=round(d_video, 4))
            self.record = rec
            return scores

        # Stage 2: compute per-channel kappa contributions
        d_audio = self._cf_lp_yes(self.cf_noaudio, gen) - ly_full
        d_video = self._cf_lp_yes(self.cf_novideo, gen) - ly_full

        # Restrict confession signal to R (relevant channels only).
        # kappa^(audio) = margin_full - margin_noaudio = -d_audio  (sign: positive
        # kappa means audio raised Yes relative to no-audio baseline)
        # We use the λ-weighted delta formulation (lambda-δ, best for CMM at 81.3%)
        # but sum ONLY over R to strip the over-relied channel's contribution.
        delta = 0.0
        if 'audio' in self.R:
            delta += self.w_audio * d_audio
        if 'video' in self.R:
            delta += self.w_video * d_video

        delta_eff = delta
        if self.control_rng is not None:
            delta_eff = delta * self.control_rng.choice([-1.0, 1.0])
        margin_adj = margin_full + self.lam * delta_eff
        flip = YES_ID if margin_adj > 0 else NO_ID
        # v3 confidence gate: keep the base committed answer when the full-view
        # margin is confident; only let the λ-δ flip fire on low-confidence bases
        # (offline n=20: protects high-confidence-correct Yes from over-flipping).
        forced = confidence_gated_flip(am, margin_full, flip, self.gate_tau)

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
            # Stage 2 confession: which channels are relevant for this sample
            R=sorted(self.R), R_source=self.R_source,
            delta=round(delta, 4),
            delta_eff=round(delta_eff, 4),
            margin_adj=round(margin_adj, 4),
            gate_tau=self.gate_tau,
        )
        return scores


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
    ap.add_argument('--gate_tau', type=float, default=2.0,
                    help="v3 confidence gate: keep base answer when "
                         "|margin_full|>=gate_tau; 0 recovers ungated λ-δ.")
    ap.add_argument('--w_audio', type=float, default=1.0)
    ap.add_argument('--w_video', type=float, default=1.0)
    ap.add_argument('--decision', choices=['lambda_delta', 'v4', 'v5'], default='lambda_delta',
                    help="Stage-3 rule: 'lambda_delta' (legacy λ-δ), 'v4' "
                         "(n=200 F1-yes: flip base-yes->no when margin_full<tau_m), "
                         "or 'v5' (v4 deploy decision + records per-channel d_cf "
                         "for low-margin base-yes so grounding rules can be replayed).")
    ap.add_argument('--tau_m', type=float, default=2.0,
                    help="v4 rule threshold: flip base-yes to no when the base "
                         "yes-margin < tau_m (n=200 F1-yes 81.40 at 2.0).")
    ap.add_argument('--tau_hi', type=float, default=4.0,
                    help="v5: skip the two masked-view forwards when a base-yes "
                         "already has margin_full>=tau_hi (high-confidence anchor). "
                         "Must be >= tau_m so the grounding gate sees channel signal.")
    ap.add_argument('--theta', type=float, default=-1.6,
                    help="v5 grounding threshold: keep a low-margin base-yes iff "
                         "min_{m in R} d_cf_m <= theta (the yes depends on R). "
                         "Full-721 F1(yes) 83.88 at tau_m=2.5, theta=-1.6.")
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
            u = get_user_prompt(item['question'], args.datasets, args.prompt_version)

            # full view (== base_model input): video + audio
            clean_f = build_view(processor, model, system_prompt, pv, av, iv, u)
            input_len = clean_f['input_ids'].shape[1]

            # counterfactual views: noaudio (video only), novideo (audio only).
            # v4 uses only the full-view margin, so skip the extra forwards.
            # v5 builds them (it forwards them only for low-margin base-yes,
            # decided inside the processor).
            if args.decision == 'v4':
                clean_a = clean_v = clean_f
            else:
                clean_a = build_view(processor, model, system_prompt, pv, None, iv, u)
                clean_v = build_view(processor, model, system_prompt, None, av, iv, u)

            cf_proc = CFConfessProcessor(model, tokenizer, clean_a, clean_v,
                                         input_len, args.lambda_cf,
                                         args.w_audio, args.w_video, control_rng,
                                         question=item.get('question', ''),
                                         group_key=item.get('group_key', ''),
                                         gate_tau=args.gate_tau,
                                         decision=args.decision, tau_m=args.tau_m,
                                         tau_hi=args.tau_hi, theta=args.theta)

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
        attach_record(result, rec_extra, gen_ids, logp_list, input_len,
                      args.record_process)
        save_result(result, output_path)

    write_time(args.time_path, args.file_name, t_start, all_out_tok)
    print(f'\n[done] output={output_path}', flush=True)


if __name__ == '__main__':
    main()
