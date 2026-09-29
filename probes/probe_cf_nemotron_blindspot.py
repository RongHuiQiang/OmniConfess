#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""probe_cf_nemotron_blindspot.py -- blindspot-axis third backbone:
Nemotron-3-Nano-Omni-30B x {phd, cmm} sample-level rho(conf, frag).

Upgrades the cross-backbone story from two backbones (Qwen2.5 / Qwen3) to
three by replicating the two judgement-family blindspot cells on the NVIDIA
omni backbone. Decision-level, sample-level protocol identical to
probe_cf_signal_{phd,cmm}_nemotron.py (n=100 signal probes) but sampled for
the blindspot axis (n up to 250/family, stratified, resumable).

Definitions COPIED FROM THE QWEN2.5 CALIBERS:
  phd  (cf_confess_phd.py): delta_cf = lp_yes(cf view) - lp_yes(full),
        cf view = image + question only (context & marker removed);
        frag = |delta_cf|.
  cmm  (cf_confess_cmm.py): d_cf_audio = lp_yes(noaudio) - lp_yes(full),
        d_cf_video = lp_yes(novideo) - lp_yes(full); frag = L2
        sqrt(da^2 + dv^2)  (primary aggregation in unified_cf_object.py).
  conf calibers: |margin_full| = |lp_yes - lp_no| at the decision token on
        the full view (primary, same as existing cells), plus norm_logp from
        the base Nemotron greedy run (secondary).

Base trajectories (greedy v1, enable_thinking=False, per-token logp):
  phd /ultrafast_share/tsq/tsq/nemotron/outputs/phd_nemotron/phd_nemotron_single_scored.jsonl (906)
  cmm /ultrafast_share/tsq/tsq/nemotron/outputs/cmm_nemotron/cmm_nemotron_single_scored.jsonl (721)
Decision-token locator: backward scan for '{' whose successor is Yes/No
(Nemotron 'box'+'ed'+'{'; Yes=16860 No=4753 '{'=1123 -- see
probe_cf_signal_phd_nemotron.py). Teacher forcing via generate(max_new_tokens=1,
output_scores=True) on the processor-clean input + base prefix.

Analysis is done separately (zero-GPU, unified_cf_object.blindspot()); this
script only emits per-sample records.

Usage (one exclusive card; nemotron-hf venv):
  CUDA_VISIBLE_DEVICES=7 PYTHONDONTWRITEBYTECODE=1 \
    /ultrafast_share/tsq/venvs/nemotron-hf/bin/python \
    probe_cf_nemotron_blindspot.py --family phd --n 250
"""
import os, sys, json, time, argparse, random, hashlib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'OmniConfess', 'methods'))
sys.path.insert(0, '/ultrafast_share/tsq/tsq/nemotron')
os.chdir(HERE)

import torch

from shared import load_dataset, resolve_media_path, get_system_prompt, get_user_prompt
from shared.data_loader import load_image
from base_model_nemotron_hf import (
    DEFAULT_MODEL, _repair_buffers,
    load_video_frames_native, load_audio_wave,
)

YES_ID, NO_ID = 16860, 4753
OPEN_ID = 1123
MARKER = ("In case there is an inconsistency between the context and the image "
          "content, you should follow the image.")

SCORED = {
    'phd': '/ultrafast_share/tsq/tsq/nemotron/outputs/phd_nemotron/phd_nemotron_single_scored.jsonl',
    'cmm': '/ultrafast_share/tsq/tsq/nemotron/outputs/cmm_nemotron/cmm_nemotron_single_scored.jsonl',
}
DATA = {
    'phd': './OmniHalluBench/phd/annotations/phd.json',
    'cmm': './OmniHalluBench/cmm/annotations/cmm.json',
}


def split_question(q):
    idx = q.find(MARKER)
    if idx < 0:
        return None, q
    return q[:idx].rstrip(), q[idx + len(MARKER):].lstrip()


def find_decision_index(token_ids):
    ids = list(token_ids)
    if ids and ids[0] in (YES_ID, NO_ID):
        return 0, ids[0]
    for i in range(len(ids) - 2, -1, -1):
        if ids[i] == OPEN_ID and ids[i + 1] in (YES_ID, NO_ID):
            return i + 1, ids[i + 1]
    return None, None


def build_clean(tokenizer, system_prompt, user_text, image=None,
                videos=None, audio=None):
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
    if image is not None:
        kw['images'] = image
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
            int(inputs['input_ids'].shape[1]))


def stratified(rows, n, seed):
    """Round-robin over gt x base_correct cells (random order within cell)."""
    rng = random.Random(seed)
    cells = {}
    for r in rows:
        k = (1 if r['gt'] == 'yes' else 0,
             int(r['eval_metrics']['Is_Correct'] == 1))
        cells.setdefault(k, []).append(r)
    for v in cells.values():
        rng.shuffle(v)
    sample, i = [], 0
    while len(sample) < n:
        adv = False
        for v in cells.values():
            if i < len(v) and len(sample) < n:
                sample.append(v[i])
                adv = True
        if not adv:
            break
        i += 1
    rng.shuffle(sample)
    return sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--family', required=True, choices=['phd', 'cmm'])
    ap.add_argument('--n', type=int, default=250)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--limit', type=int, default=0, help='smoke: first N of sample')
    ap.add_argument('--out', default='')
    ap.add_argument('--model_path', default=os.environ.get(
        'NEMOTRON_MODEL_PATH', DEFAULT_MODEL))
    args = ap.parse_args()

    fam = args.family
    out_path = args.out or f'outputs/cf_probe/blindspot_nemotron/{fam}_blindspot_nemotron.jsonl'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    random.seed(args.seed)

    rows = [json.loads(l) for l in open(SCORED[fam])]
    rows = [r for r in rows if r.get('token_ids') and r.get('norm_logp') is not None]
    data = load_dataset(DATA[fam], './OmniHalluBench/')
    if fam == 'phd':
        by_q = {}
        for it in data:
            by_q.setdefault(it['question'], []).append(it)
    else:
        by_id = {it['id']: it for it in data}
        for r in rows[:50]:
            assert r['id'] in by_id and by_id[r['id']]['question'] == r['question']
    json_dir = os.path.dirname(os.path.abspath(DATA[fam]))

    sample = stratified(rows, args.n, args.seed)
    print(f'[probe] {fam}: {len(sample)}/{len(rows)} sampled '
          f'(gt x base_correct round-robin, seed {args.seed})', flush=True)
    if args.limit:
        sample = sample[:args.limit]

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

    system_prompt = get_system_prompt(fam, 'v1')

    done = set()
    if os.path.exists(out_path):
        for l in open(out_path):
            try:
                done.add(json.loads(l)['id'])
            except Exception:
                pass
        print(f'[resume] {len(done)} already done', flush=True)
    mode = 'a' if done else 'w'

    t0 = time.time()
    n_done = 0
    with open(out_path, mode) as fout:
        for r in sample:
            if r['id'] in done:
                continue
            t1 = time.time()
            try:
                prefix_len, dec_tok = find_decision_index(r['token_ids'])
                if prefix_len is None:
                    print(f"  [skip] no decision token id={r['id']}", flush=True)
                    continue
                prefix = r['token_ids'][:prefix_len]

                rec = dict(
                    family=fam, id=r['id'],
                    q_hash=hashlib.md5(r['question'].encode()).hexdigest()[:12],
                    gt=r['gt'], pred=r['pred'],
                    base_correct=int(r['eval_metrics']['Is_Correct'] == 1),
                    dec_index=prefix_len, dec_token=dec_tok,
                    bare=int(prefix_len == 0),
                    norm_logp=r['norm_logp'],
                    dec_logp_recorded=r['logp_list'][prefix_len])

                if fam == 'phd':
                    items = by_q.get(r['question'])
                    if not items:
                        print(f"  [skip] question not found id={r['id']}", flush=True)
                        continue
                    item = items[0]
                    ip = resolve_media_path(item.get('image_path'), json_dir,
                                            './OmniHalluBench/')
                    image = load_image(ip)
                    if image is None:
                        print(f"  [skip] no image id={r['id']}", flush=True)
                        continue
                    _, q_only = split_question(r['question'])

                    # full view (== base input)
                    ly, ln, il_f = decision_logp(
                        processor, model, build_clean(
                            tokenizer, system_prompt,
                            '<image>\n' + get_user_prompt(r['question'], fam, 'v1'),
                            image=image), prefix, device)
                    rec.update(lp_yes_full=round(ly, 4), lp_no_full=round(ln, 4),
                               margin_full=round(ly - ln, 4),
                               input_len_full=il_f)

                    # cf view: image + question only (context & marker removed)
                    ly_c, ln_c, _ = decision_logp(
                        processor, model, build_clean(
                            tokenizer, system_prompt,
                            '<image>\n' + get_user_prompt(q_only, fam, 'v1'),
                            image=image), prefix, device)
                    rec.update(lp_yes_cf=round(ly_c, 4), lp_no_cf=round(ln_c, 4),
                               delta_cf=round(ly_c - ly, 4))   # operator conv.
                else:  # cmm
                    item = by_id[r['id']]
                    vp = resolve_media_path(item.get('video_path'), json_dir,
                                            './OmniHalluBench/')
                    ap_ = resolve_media_path(item.get('audio_path'), json_dir,
                                             './OmniHalluBench/')
                    videos = load_video_frames_native(vp)
                    audio = load_audio_wave(ap_)
                    if videos is None or audio is None:
                        print(f"  [skip] missing media id={r['id']}", flush=True)
                        continue
                    u = get_user_prompt(r['question'], fam, 'v1')

                    ly, ln, il_f = decision_logp(
                        processor, model, build_clean(
                            tokenizer, system_prompt,
                            '<video>\n<so_embedding>\n' + u,
                            videos=videos, audio=audio), prefix, device)
                    rec.update(lp_yes_full=round(ly, 4), lp_no_full=round(ln, 4),
                               margin_full=round(ly - ln, 4),
                               input_len_full=il_f)

                    ly_a, ln_a, _ = decision_logp(       # noaudio: video only
                        processor, model, build_clean(
                            tokenizer, system_prompt, '<video>\n' + u,
                            videos=videos), prefix, device)
                    ly_v, ln_v, _ = decision_logp(       # novideo: audio only
                        processor, model, build_clean(
                            tokenizer, system_prompt, '<so_embedding>\n' + u,
                            audio=audio), prefix, device)
                    rec.update(d_cf_audio=round(ly_a - ly, 4),
                               d_cf_video=round(ly_v - ly, 4))

                rec['tf_repro'] = round(
                    (ly if dec_tok == YES_ID else ln) - r['logp_list'][prefix_len], 4)
                rec['sec'] = round(time.time() - t1, 2)
                fout.write(json.dumps(rec, ensure_ascii=False) + '\n')
                fout.flush()
                n_done += 1
                if n_done % 10 == 0:
                    key = ('delta_cf' if fam == 'phd' else 'd_cf_audio')
                    print(f'  [{n_done}/{len(sample)}] {time.time()-t0:.0f}s '
                          f'{key}={rec[key]:+.2f} m={rec["margin_full"]:+.2f} '
                          f'tf={rec["tf_repro"]:+.4f} {rec["sec"]}s', flush=True)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"  [ERROR] id={r['id']}: {e}", flush=True)
            # NB: no per-sample empty_cache() -- keep the warm-claim
            # reservation for the process lifetime (farm contention).

    print(f'[done] {n_done}/{len(sample)} in {time.time()-t0:.0f}s -> {out_path}',
          flush=True)


if __name__ == '__main__':
    main()
