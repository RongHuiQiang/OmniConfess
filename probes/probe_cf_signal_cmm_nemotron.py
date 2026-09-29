#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""probe_cf_signal_cmm_nemotron.py -- CF signal probe: cmm x Nemotron-3-Nano-Omni.

Fourth test point of the generational self-healing program and the second on
Nemotron (phd-Nemotron probe = third point). The cmm disease (unimodal-prior
over-Yes; CF@Qwen2.5 +6.29 F1, FPR .555->.124) self-healed on Qwen3 (polarity
INVERTED: d_cf_sum median +0.26 on gt=no rows -- modalities now correctly
SUPPRESS yes; magnitude collapsed 611x). Question here: does the
audio/video-leakage signal exist at the decision token on the NVIDIA omni
backbone, whose behavioral marker is unknown until its base run lands.

Base trajectories (from the [Nemotron 补格] chain, greedy v1 prompt,
enable_thinking=False, per-token logp recorded):
  /ultrafast_share/tsq/tsq/nemotron/outputs/cmm_nemotron/cmm_nemotron_single_scored.jsonl
  (721 rows expected; full-file single process -> ids align with cmm.json
  row order directly, verified by id+question match below).

Views at the decision token (teacher-forced on the base greedy prefix),
identical question text, the masked modality's placeholder AND processor
kwarg both dropped (same construction as probe_cf_signal_cmm_qwen3.py):
  full     : video + audio + question      (== base input)
  noaudio  : video + question              -> d_cf_audio (audio pull on Yes)
  novideo  : audio + question              -> d_cf_video (video pull on Yes)

Decision token: backward scan for '{' whose successor is Yes/No (the
Nemotron '\\boxed{Yes|No}' tail; 'boxed' splits into 'box'+'ed' on this
tokenizer -- see probe_cf_signal_phd_nemotron.py). Bare first-token kept
defensively.

PRE-REGISTERED JUDGMENT (written before running; conjunction copied verbatim
from the cmm-Qwen3 pre-registration, cf. EXPERIMENT_DESIGN.md):
  - Sample: 100 rows, 25 x (gt_yes/no x subgroup{overrely_audio,
    overrely_visual}), seed 42, from scored rows with a locatable decision
    token.
  - Signals: margin_full, d_cf_audio, d_cf_video, d_cf_sum
    (d_cf_i = lp_yes(masked) - lp_yes(full), operator convention).
  - ALIVE := AUC(-d_cf_sum -> gt) > 0.65 AND Qwen2.5-type direction
    structure: (i) d_cf_sum < 0 on the large majority of base-Yes samples
    (a modality is propping Yes up; Q2.5 99.9%), (ii) negative d_cf mean
    among gt=no & base=yes (the FP mass, "audio leakage pushes yes"),
    (iii) error-Yes margins < correct-Yes margins (Q2.5 +1.52 vs +3.42).
    The Qwen3 lesson is baked in: AUC alone can pass on polarity-INVERTED
    samples (Qwen3 0.721 with median d_sum +0.26 on gt=no) -- the direction
    clauses are the decisive conjunct.
  - ALIVE -> full 721 CF@Nemotron-cmm (lambda re-selected on this dev-100:
    max Youden offline grid, +/-5pp yes-rate rule waived for an over-Yes
    base, same cmm exemption as Qwen3; random-sign control arm alongside).
  - AUC <= 0.65 OR direction not same-type -> DEAD: negative result; the
    self-healing program then closes with the disease absent on BOTH
    next-generation omni backbones. Supplementary (reported, non-gate):
    offline lambda grid, per-subgroup |d_cf|, tf_repro distribution.

Usage (one exclusive card):
  CUDA_VISIBLE_DEVICES=0 /ultrafast_share/tsq/venvs/nemotron-hf/bin/python \
      probe_cf_signal_cmm_nemotron.py --n_per_cell 25
"""
import os, sys, json, time, argparse, random

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'OmniConfess', 'methods'))
sys.path.insert(0, '/ultrafast_share/tsq/tsq/nemotron')
os.chdir(HERE)

import torch

from shared import load_dataset, resolve_media_path, get_system_prompt, get_user_prompt
from base_model_nemotron_hf import (
    DEFAULT_MODEL, _repair_buffers,
    load_video_frames_native, load_audio_wave,
)

YES_ID, NO_ID = 16860, 4753
OPEN_ID = 1123


def sub_group(group_key):
    return 'audio' if str(group_key).startswith('overrely_audio') else 'visual'


def find_decision_index(token_ids):
    ids = list(token_ids)
    if ids and ids[0] in (YES_ID, NO_ID):
        return 0, ids[0]
    for i in range(len(ids) - 2, -1, -1):
        if ids[i] == OPEN_ID and ids[i + 1] in (YES_ID, NO_ID):
            return i + 1, ids[i + 1]
    return None, None


def build_clean(tokenizer, system_prompt, user_text, videos=None, audio=None):
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
    if videos is not None:
        kw['videos'] = videos
    if audio is not None:
        kw['audio'] = audio
    return kw


@torch.inference_mode()
def decision_logp(processor, model, clean_kw, prefix_ids, device):
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
    ap.add_argument('--scored', default='/ultrafast_share/tsq/tsq/nemotron/outputs/cmm_nemotron/cmm_nemotron_single_scored.jsonl')
    ap.add_argument('--data_path', default='./OmniHalluBench/cmm/annotations/cmm.json')
    ap.add_argument('--data_root', default='./OmniHalluBench/cmm/')
    ap.add_argument('--n_per_cell', type=int, default=25)
    ap.add_argument('--full', action='store_true')
    ap.add_argument('--out', default='outputs/cf_probe/cmm_probe_nemotron.jsonl')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--model_path', default=os.environ.get(
        'NEMOTRON_MODEL_PATH', DEFAULT_MODEL))
    args = ap.parse_args()

    random.seed(args.seed)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    rows = [json.loads(l) for l in open(args.scored)]
    data = load_dataset(args.data_path, args.data_root)
    by_id = {it['id']: it for it in data}
    for r in rows[:50]:                       # alignment guard (chain = full file)
        assert r['id'] in by_id and by_id[r['id']]['question'] == r['question'], \
            f"misaligned row id={r['id']}"
    json_dir = os.path.dirname(os.path.abspath(args.data_path))

    cells = {(1, 'audio'): [], (1, 'visual'): [], (0, 'audio'): [], (0, 'visual'): []}
    for r in rows:
        if not r.get('token_ids'):
            continue
        gy = 1 if r['gt'] == 'yes' else 0
        cells[(gy, sub_group(r['group_key']))].append(r)
    if args.full:
        sample = [((1 if r['gt'] == 'yes' else 0, sub_group(r['group_key'])), r)
                  for k in cells for r in cells[k]]
    else:
        sample = []
        for k, v in sorted(cells.items()):
            random.shuffle(v)
            sample += [(k, r) for r in v[:args.n_per_cell]]
    random.shuffle(sample)
    if args.limit:
        sample = sample[:args.limit]
    print(f'[probe] sampled {len(sample)} '
          f'({ {k: min(len(v), args.n_per_cell) for k, v in cells.items()} })', flush=True)

    from transformers import AutoModel, AutoProcessor
    print(f'[model] loading {args.model_path} (bf16, one exclusive card)...',
          flush=True)
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(args.model_path,
                                              trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model_path, trust_remote_code=True,
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        attn_implementation='eager')
    device = os.environ.get('NEMO_DEVICE', 'cuda:0')
    model.to(device).eval()
    _repair_buffers(model, args.model_path, device)
    if getattr(model, 'video_pruning_rate', 0):
        model.video_pruning_rate = 0.0
    tokenizer = processor.tokenizer
    print(f'[model] loaded in {time.time()-t0:.0f}s', flush=True)

    system_prompt = get_system_prompt('cmm', 'v1')

    t0 = time.time()
    n_done = 0
    with open(args.out, 'w') as fout:
        for (gy, sg), r in sample:
            try:
                item = by_id[r['id']]
                vp = resolve_media_path(item.get('video_path'), json_dir,
                                        args.data_root)
                ap_ = resolve_media_path(item.get('audio_path'), json_dir,
                                         args.data_root)
                videos = load_video_frames_native(vp)
                audio = load_audio_wave(ap_)
                if videos is None or audio is None:
                    print(f"  [skip] missing media id={r['id']}", flush=True)
                    continue

                prefix_len, dec_tok = find_decision_index(r['token_ids'])
                if prefix_len is None:
                    print(f"  [skip] no decision token id={r['id']}", flush=True)
                    continue
                prefix = r['token_ids'][:prefix_len]

                u = get_user_prompt(r['question'], 'cmm', 'v1')
                rec = dict(id=r['id'], gt=r['gt'], pred=r['pred'], gt_yes=gy,
                           base_correct=r['eval_metrics']['Is_Correct'],
                           group=r['group_key'], sub_group=sg,
                           dec_index=prefix_len, dec_token=dec_tok,
                           bare=int(prefix_len == 0),
                           dec_logp_recorded=r['logp_list'][prefix_len])

                u_full = '<video>\n<so_embedding>\n' + u
                ly, ln, am, il_f = decision_logp(
                    processor, model, build_clean(
                        tokenizer, system_prompt, u_full, videos, audio),
                    prefix, device)
                rec.update(lp_yes_full=round(ly, 4), lp_no_full=round(ln, 4),
                           full_argmax=am,
                           full_argmax_is_dec=int(am == dec_tok),
                           margin_full=round(ly - ln, 4),
                           input_len_full=il_f,
                           input_len_matches=int(
                               il_f == r.get('input_len', -1)))

                u_va = '<video>\n' + u                      # noaudio view
                ly2, ln2, am2, _ = decision_logp(
                    processor, model, build_clean(
                        tokenizer, system_prompt, u_va, videos, None),
                    prefix, device)
                rec.update(lp_yes_noaudio=round(ly2, 4),
                           lp_no_noaudio=round(ln2, 4),
                           margin_noaudio=round(ly2 - ln2, 4),
                           delta_audio=round(ly - ly2, 4))   # full - masked

                u_au = '<so_embedding>\n' + u               # novideo view
                ly3, ln3, am3, _ = decision_logp(
                    processor, model, build_clean(
                        tokenizer, system_prompt, u_au, None, audio),
                    prefix, device)
                rec.update(lp_yes_novideo=round(ly3, 4),
                           lp_no_novideo=round(ln3, 4),
                           margin_novideo=round(ly3 - ln3, 4),
                           delta_video=round(ly - ly3, 4))   # full - masked

                # operator convention (masked - full)
                rec.update(d_cf_audio=round(ly2 - ly, 4),
                           d_cf_video=round(ly3 - ly, 4),
                           d_cf_sum=round(ly2 + ly3 - 2 * ly, 4),
                           tf_repro=round(
                               (ly if dec_tok == YES_ID else ln)
                               - r['logp_list'][prefix_len], 4))

                fout.write(json.dumps(rec, ensure_ascii=False) + '\n')
                fout.flush()
                n_done += 1
                if n_done % 10 == 0:
                    print(f'  [{n_done}/{len(sample)}] {time.time()-t0:.0f}s '
                          f'd_a={rec["delta_audio"]:+.2f} '
                          f'd_v={rec["delta_video"]:+.2f} '
                          f'tf={rec["tf_repro"]:+.4f}', flush=True)
                del videos, audio
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"  [ERROR] id={r['id']}: {e}", flush=True)
            # NB: no per-sample empty_cache() -- keep the warm-claim
            # reservation for the process lifetime (farm contention).

    print(f'[done] {n_done}/{len(sample)} samples in {time.time()-t0:.0f}s '
          f'-> {args.out}', flush=True)


if __name__ == '__main__':
    main()
