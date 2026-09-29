#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""cf_confess_nemotron_hf.py -- CF-Confess operator on
NVIDIA Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16 (cross-backbone, 2026-08-31
CF_OVERRIDE run: the two withheld grid cells phd x CF lam2.0 and cmm x CF
lam1.0, PI override of the probe gate).

Same three-stage operator as cf_confess_phd.py / cf_confess_cmm.py (Qwen2.5
arms) and cf_confess_qwen3.py (Qwen3 arm). The Nemotron omni wrapper only
exposes generate(), so the one-shot decision-token intervention is run as its
exact offline equivalent (greedy is deterministic). This is the CF_OVERRIDE run
and keeps the lambda-delta rule (decision='lambda') for both withheld cells --
phd x lam2.0 and cmm x lam1.0 -- so the numbers are unchanged.

  Stage 1  Generate   base greedy pass (shared v1 prompt, enable_thinking=
                      False, per-token logprobs recorded) -> full-view
                      trajectory; then locate the decision token in it
                      ('box' 'ed' '{' 'Yes'|'No' tail on this tokenizer,
                      backward scan; bare step-0 Yes/No kept defensively --
                      probe verified argmax==dec 100/100, tf_repro median ~0).
  Stage 2  Confess     teacher-forced per-view rescoring at the decision token
                      (append the base prefix to each view's processor-clean
                      input_ids, single generate step; views probe-identical):
                        phd  full  : image + context + marker + question
                             noctx : image + question only  -> delta_cf
                             noimg : context + question, no image (secondary)
                        cmm  full  : video + audio + question
                             noaudio : video + question      -> d_cf_audio
                             novideo : audio + question       -> d_cf_video
  Stage 3  Decide      margin_full = lp_yes - lp_no (base-pass decision-step
                      logits; == teacher-forced full view, tf_repro ~1e-4);
                        delta (phd) = lp_yes(noctx) - lp_yes(full)
                        delta (cmm) = [lp_yes(noaudio) - lp_yes(full)]
                                      + [lp_yes(novideo) - lp_yes(full)]
                        margin'     = margin_full + lam * delta
                        forced      = Yes if margin' > 0 else No.
                      If forced != greedy decision token, regenerate the
                      continuation from [prompt + prefix + forced_token]
                      greedily (== what the online LogitsProcessor arm does
                      after forcing); else final = base trajectory.

Every sample keeps the cf internals (margin_full / delta_cf / per-view lps /
dec_index / greedy_token / forced_token / intervened) for later analysis, and
greedy_token is the in-run paired base decision (the old Nemotron base
trajectories lived on /ultrafast_share, wiped 2026-08-29).

Model layer identical to base_model_nemotron_hf.py / the probes (transformers
5.6.2 trust_remote_code, six-pitfall repair chain, video fps=2 native frames,
16 kHz mono audio, enable_thinking=False). No per-sample empty_cache (keep the
warm-claim reservation, as the probes did).

Usage (one exclusive card per process):
  CUDA_VISIBLE_DEVICES=1 /slow_share/tsq/venvs/nemotron-hf/bin/python \
      cf_confess_nemotron_hf.py --datasets phd \
      --data_path ./OmniHalluBench/phd/annotations/phd.json \
      --output_dir outputs/phd_nemotron_cf --file_name phd_nemotron_cf_full \
      --lambda_cf 2.0
"""
import os, sys, time, json, argparse, random

REPO = os.environ.get(
    'NEMO_REPO', '/home/tsq/rhq/omniconfess/OmniConfess')
sys.path.insert(0, os.path.join(REPO, 'OmniConfess', 'methods'))

import numpy as np
import torch

from shared import (
    load_dataset, resolve_media_path, get_system_prompt, get_user_prompt,
    make_result, save_result,
)
from shared.data_loader import load_image
from base_model_nemotron_hf import (
    DEFAULT_MODEL, _repair_buffers,
    load_video_frames_native, load_audio_wave, token_logprobs,
)

# Nemotron tokenizer (verified 2026-08-16, same as the probes):
# 'Yes'->16860 'No'->4753 (capitalized, inside \boxed{...}); '{'->1123
YES_ID, NO_ID = 16860, 4753
OPEN_ID = 1123

MARKER = ("In case there is an inconsistency between the context and the image "
          "content, you should follow the image.")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_question(q):
    """phd: peel the misleading context off; keep the marker boundary."""
    idx = q.find(MARKER)
    if idx < 0:
        return q, q
    return q, q[idx + len(MARKER):].lstrip()


def find_decision_index(token_ids):
    """Backward scan for '{' whose successor is Yes/No ('boxed' splits into
    'box'+'ed' on this tokenizer); bare first-token Yes/No kept defensively."""
    ids = list(token_ids)
    if ids and ids[0] in (YES_ID, NO_ID):
        return 0, ids[0]
    for i in range(len(ids) - 2, -1, -1):
        if ids[i] == OPEN_ID and ids[i + 1] in (YES_ID, NO_ID):
            return i + 1, ids[i + 1]
    return None, None


def build_clean(tokenizer, system_prompt, user_text, images=None,
                videos=None, audio=None):
    """Replicate base_model_nemotron_hf.py's input construction exactly."""
    messages = [{'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_text}]
    try:
        text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
            enable_thinking=False)
    except TypeError:
        text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False)
    kw = dict(text=[text], return_tensors='pt')
    if images is not None:
        kw['images'] = images
    if videos is not None:
        kw['videos'] = videos
    if audio is not None:
        kw['audio'] = audio
    return kw


@torch.inference_mode()
def decision_logp(processor, model, clean_kw, prefix_ids, device):
    """Append the base greedy prefix to a view's clean input and read the
    decision-token distribution (single generate step)."""
    inputs = processor(**clean_kw)
    inputs = {k: (v.to(device) if torch.is_tensor(v) else v)
              for k, v in inputs.items()}
    ids = torch.cat([inputs['input_ids'][0],
                     torch.tensor(prefix_ids, dtype=torch.long,
                                  device=inputs['input_ids'].device)],
                    dim=0).unsqueeze(0)
    kw = dict(input_ids=ids, attention_mask=torch.ones_like(ids))
    for k in ('pixel_values', 'pixel_values_videos', 'sound_clips',
              'sound_length'):
        if k in inputs and inputs[k] is not None:
            kw[k] = inputs[k]
    out = model.generate(**kw, max_new_tokens=1, do_sample=False,
                         temperature=None, top_p=None, top_k=None,
                         output_scores=True, return_dict_in_generate=True,
                         pad_token_id=processor.tokenizer.pad_token_id or 0)
    logits = out.scores[0][0].float()
    lp = torch.log_softmax(logits, dim=-1)
    return (lp[YES_ID].item(), lp[NO_ID].item(),
            int(logits.argmax().item()), int(inputs['input_ids'].shape[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', type=str, required=True, choices=['phd', 'cmm'])
    ap.add_argument('--data_path', type=str, required=True)
    ap.add_argument('--data_root', type=str, default=None)
    ap.add_argument('--output_dir', type=str, default='./outputs')
    ap.add_argument('--file_name', type=str, default='cf_nemotron')
    ap.add_argument('--time_path', type=str, default='./outputs/time')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--max_new_tokens', type=int, default=512)
    ap.add_argument('--prompt_version', type=str, default='v1')
    ap.add_argument('--lambda_cf', type=float, required=True)
    ap.add_argument('--control', choices=['none', 'random_sign'], default='none')
    ap.add_argument('--record_process', action=argparse.BooleanOptionalAction,
                    default=True)
    ap.add_argument('--model_path', type=str,
                    default=os.environ.get('NEMOTRON_MODEL_PATH', DEFAULT_MODEL))
    ap.add_argument('--dry_run', action='store_true',
                    help='first 3 items only (smoke)')
    ap.add_argument('--limit', type=int, default=0,
                    help='first N items only (smoke)')
    args = ap.parse_args()
    if not args.data_root:
        args.data_root = os.path.join(REPO, 'OmniHalluBench',
                                      args.datasets) + os.sep

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

    from transformers import AutoModel, AutoProcessor
    print(f'[model] loading {args.model_path} (bf16, one exclusive card)...',
          flush=True)
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(args.model_path,
                                              trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, attn_implementation='eager')
    device = os.environ.get('NEMO_DEVICE', 'cuda:0')
    model.to(device).eval()
    _repair_buffers(model, args.model_path, device)
    if getattr(model, 'video_pruning_rate', 0):
        model.video_pruning_rate = 0.0
    tokenizer = processor.tokenizer
    print(f'[model] loaded in {time.time()-t0:.0f}s', flush=True)

    data = load_dataset(args.data_path, args.data_root)
    if args.dry_run:
        data = data[:3]
    if args.limit:
        data = data[:args.limit]

    system_prompt = get_system_prompt(args.datasets, args.prompt_version)
    json_dir = os.path.dirname(os.path.abspath(args.data_path))

    all_out = 0
    t_start = time.time()

    for idx, item in enumerate(data):
        if item['id'] in done_ids:
            continue
        t0 = time.time()
        print(f'\n[{idx+1}/{len(data)}] id={item["id"]}', flush=True)

        rec_extra = dict(dec_index=None, intervened=0, error=None)
        gen_ids = None
        logp_list = None
        input_len = 0
        has_video = has_audio = has_image = False
        try:
            # ── media (same mapping as base_model_nemotron_hf.py) ──────────
            videos = images = audio = None
            if item.get('video_path'):
                vp = resolve_media_path(item['video_path'], json_dir,
                                        args.data_root)
                videos = load_video_frames_native(vp)
                has_video = videos is not None
                if not has_video:
                    print(f'  [MEDIA WARN] video unreadable: {vp}', flush=True)
            if item.get('audio_path'):
                ap_ = resolve_media_path(item['audio_path'], json_dir,
                                         args.data_root)
                audio = load_audio_wave(ap_)
                has_audio = audio is not None
                if not has_audio:
                    print(f'  [MEDIA WARN] audio unreadable: {ap_}', flush=True)
            if item.get('image_path'):
                ip = resolve_media_path(item['image_path'], json_dir,
                                        args.data_root)
                images = load_image(ip)
                has_image = images is not None
                if not has_image:
                    print(f'  [MEDIA WARN] image unreadable: {ip}', flush=True)

            # ── full-view prompt (== base input construction) ──────────────
            q = item['question']
            u_full = get_user_prompt(q, args.datasets, args.prompt_version,
                                     passage=item.get('passage'))
            media_prefix = ''
            if has_video:
                media_prefix += '<video>\n'
            if has_audio:
                media_prefix += '<so_embedding>\n'   # processor's audio token
            if has_image:
                media_prefix += '<image>\n'
            user_text_full = media_prefix + u_full
            clean_full = build_clean(tokenizer, system_prompt, user_text_full,
                                     images=images, videos=videos, audio=audio)

            inputs = processor(**clean_full)
            inputs = {k: (v.to(device) if torch.is_tensor(v) else v)
                      for k, v in inputs.items()}
            if inputs.get('pixel_values_videos') is not None:
                _pv = inputs['pixel_values_videos']
                print(f'  [video] pv={tuple(_pv.shape)} dtype={_pv.dtype}',
                      flush=True)
            if inputs.get('pixel_values') is not None:
                _pi = inputs['pixel_values']
                print(f'  [image] pv={tuple(_pi.shape)} dtype={_pi.dtype}',
                      flush=True)
            for _mk in ('num_patches', 'num_tokens', 'imgs_sizes'):
                inputs.pop(_mk, None)
            input_len = int(inputs['input_ids'].shape[1])
            if has_audio:
                clips = inputs.get('sound_clips')
                n_s = int(np.asarray(clips[0]).shape[-1]) if clips is not None else 0
                print(f'  [audio] {n_s} samples @16kHz '
                      f'({"non-empty => audio really in" if n_s > 0 else "EMPTY!"})',
                      flush=True)

            # ── Stage 1 (Generate): base greedy pass, scores recorded ──────
            with torch.inference_mode():
                out = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False, temperature=None, top_p=None, top_k=None,
                    pad_token_id=tokenizer.pad_token_id or 0,
                    output_scores=True, return_dict_in_generate=True)
            base_ids = out.sequences[0][input_len:].cpu()
            base_logps = token_logprobs(base_ids, out.scores)

            # ── Stage 1 (Generate) cont.: locate the decision token ────────
            prefix_len, dec_tok = find_decision_index(base_ids.tolist())
            if prefix_len is None:
                print('  [cf] no decision token -> base trajectory kept',
                      flush=True)
                gen_ids = base_ids
                logp_list = base_logps
                rec_extra.update(dec_index=None, intervened=0,
                                 pred_base=None)
            else:
                prefix = base_ids[:prefix_len].tolist()
                dec_scores = out.scores[prefix_len][0].float()
                lp_f = torch.log_softmax(dec_scores, dim=-1)
                ly_full = lp_f[YES_ID].item()
                ln_full = lp_f[NO_ID].item()
                margin_full = ly_full - ln_full

                # ── Stage 2 (Confess): teacher-forced views (probe-ident) ──
                if args.datasets == 'phd':
                    _, q_only = split_question(q)
                    u_nc = get_user_prompt(q_only, args.datasets,
                                           args.prompt_version)
                    ly_nc, ln_nc, _, _ = decision_logp(
                        processor, model, build_clean(
                            tokenizer, system_prompt, '<image>\n' + u_nc,
                            images=images), prefix, device)
                    u_ni = u_full
                    ly_ni, ln_ni, _, _ = decision_logp(
                        processor, model, build_clean(
                            tokenizer, system_prompt, u_ni), prefix, device)
                    delta = ly_nc - ly_full
                    views = dict(
                        lp_yes_noctx=round(ly_nc, 4), lp_no_noctx=round(ln_nc, 4),
                        margin_noctx=round(ly_nc - ln_nc, 4),
                        delta_noctx=round(ly_nc - ly_full, 4),
                        lp_yes_noimg=round(ly_ni, 4), lp_no_noimg=round(ln_ni, 4),
                        margin_noimg=round(ly_ni - ln_ni, 4),
                        delta_noimg=round(ly_ni - ly_full, 4),
                    )
                else:   # cmm: noaudio / novideo (u_full is the BARE user
                        # prompt; each view adds only its own placeholder)
                    u_va = ('<video>\n' if has_video else '') + u_full
                    ly_na, ln_na, _, _ = decision_logp(
                        processor, model, build_clean(
                            tokenizer, system_prompt, u_va, videos=videos),
                        prefix, device)
                    u_au = ('<so_embedding>\n' if has_audio else '') + u_full
                    ly_nv, ln_nv, _, _ = decision_logp(
                        processor, model, build_clean(
                            tokenizer, system_prompt, u_au, audio=audio),
                        prefix, device)
                    d_audio = ly_na - ly_full
                    d_video = ly_nv - ly_full
                    delta = d_audio + d_video
                    views = dict(
                        lp_yes_noaudio=round(ly_na, 4),
                        lp_no_noaudio=round(ln_na, 4),
                        margin_noaudio=round(ly_na - ln_na, 4),
                        delta_audio=round(d_audio, 4),
                        lp_yes_novideo=round(ly_nv, 4),
                        lp_no_novideo=round(ln_nv, 4),
                        margin_novideo=round(ly_nv - ln_nv, 4),
                        delta_video=round(d_video, 4),
                    )

                # ── Stage 3 (Decide): lambda intervention ──────────────
                delta_eff = delta
                if control_rng is not None:
                    delta_eff = delta * control_rng.choice([-1.0, 1.0])
                margin_adj = margin_full + args.lambda_cf * delta_eff
                forced = YES_ID if margin_adj > 0 else NO_ID
                intervened = int(forced != dec_tok)

                # ── Stage 3 (Decide) cont.: forced-token regen when flipped ─
                if intervened:
                    ids2 = torch.cat([
                        inputs['input_ids'][0],
                        torch.tensor(prefix + [forced], dtype=torch.long,
                                     device=device)], dim=0).unsqueeze(0)
                    kw2 = dict(input_ids=ids2, attention_mask=torch.ones_like(ids2))
                    for k in ('pixel_values', 'pixel_values_videos',
                              'sound_clips', 'sound_length'):
                        if k in inputs and inputs[k] is not None:
                            kw2[k] = inputs[k]
                    n_rest = max(8, args.max_new_tokens - (prefix_len + 1))
                    with torch.inference_mode():
                        out2 = model.generate(
                            **kw2, max_new_tokens=n_rest,
                            do_sample=False, temperature=None, top_p=None,
                            top_k=None, pad_token_id=tokenizer.pad_token_id or 0,
                            output_scores=True, return_dict_in_generate=True)
                    cont = out2.sequences[0][ids2.shape[1]:].cpu()
                    gen_ids = torch.cat([
                        base_ids[:prefix_len],
                        torch.tensor([forced], dtype=torch.long), cont])
                    logp_list = (base_logps[:prefix_len]
                                 + [round(lp_f[forced].item(), 4)]
                                 + token_logprobs(cont, out2.scores))
                    print(f'  [cf] INTERVENE {prefix_len}: '
                          f'{"Yes" if dec_tok == YES_ID else "No"}->'
                          f'{"Yes" if forced == YES_ID else "No"} '
                          f'(regen {len(cont)} tokens)', flush=True)
                else:
                    gen_ids = base_ids
                    logp_list = base_logps

                rec_extra.update(
                    dec_index=prefix_len, bare=int(prefix_len == 0),
                    greedy_token=int(dec_tok), forced_token=int(forced),
                    pred_base=('yes' if dec_tok == YES_ID else 'no'),
                    intervened=intervened, decision='lambda',
                    lp_yes_full=round(ly_full, 4), lp_no_full=round(ln_full, 4),
                    margin_full=round(margin_full, 4),
                    delta_cf=round(delta, 4), delta_eff=round(delta_eff, 4),
                    margin_adj=round(margin_adj, 4),
                )
                rec_extra.update(views)
                del out

            response = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            all_out += len(gen_ids)
            rec_extra.update(lambda_cf=args.lambda_cf, control=args.control,
                             decision='lambda')
            print(f'  response[-80]: {response[-80:]!r} | dec={rec_extra.get("dec_index")} '
                  f'interv={rec_extra.get("intervened")} '
                  f'delta={rec_extra.get("delta_cf")} | {time.time()-t0:.1f}s',
                  flush=True)
            del inputs, clean_full
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f'  [ERROR] {e}', flush=True)
            response = ''
            gen_ids = None
            logp_list = None
            rec_extra.update(dec_index=None, intervened=0,
                             error=str(e)[:200],
                             lambda_cf=args.lambda_cf, control=args.control,
                             decision='lambda')

        # no per-sample empty_cache: keep the warm-claim reservation (probes)

        result = make_result(item, response, args.datasets,
                             f'cf_confess_nemotron_lam{args.lambda_cf}'
                             + ('_randctl' if args.control != 'none' else ''),
                             time_sec=time.time() - t0)
        result['nemotron'] = {
            'runtime': 'transformers-5.6.2-trust-remote-code',
            'enable_thinking': False,
            'media': {'video': has_video, 'audio': has_audio,
                      'image': has_image},
        }
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
            else:
                result['token_ids'] = None
                result['logp_list'] = None
                result['input_len'] = int(input_len)
                result['output_tokens'] = 0
        save_result(result, output_path)

    with open(os.path.join(args.time_path, args.file_name + '.txt'), 'w') as f:
        f.write(f'total_time: {time.time()-t_start:.2f}s\n'
                f'total_output_tokens: {all_out}\n')
    print(f'\n[done] output={output_path}', flush=True)


if __name__ == '__main__':
    main()
