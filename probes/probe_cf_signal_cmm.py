#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""probe_cf_signal_cmm.py -- CF-Confess signal-existence probe for CMM (Step 1,
zero design commitment), the multimodal twin of probe_cf_signal_phd.py.

At the DECISION token (cmm base: 720/721 samples answer with a BARE 'Yes'/'No'
as the very first generated token, so the decision step is step 0 with an
empty teacher-forced prefix), measure the per-modality counterfactual
sensitivity

    Delta_audio = log P(Yes | full view)       - log P(Yes | video-only view)
    Delta_video = log P(Yes | full view)       - log P(No  ... n/a)
    (phd conv.) d_cf_i        = log P(Yes | modality-i-masked) - log P(Yes | full)

Views (built exactly like base_model.py inputs, media tokens skipped for the
masked modality):
    full     : video + audio + question      (== cmm base_model input)
    noaudio  : video + question              -> Delta_audio ("audio pull on Yes")
    novideo  : audio + question              -> Delta_video ("video pull on Yes")

Subgroups (facets.hallucination_type):
    overrely_audio_ignore_visual  (n=329, "Did you see X" questions, audio prior)
    overrely_visual_ignore_audio  (n=392, "Did you hear X" questions, video prior)

Report (separate analyze step): AUC of each signal for (a) predicting gt,
(b) fix direction among base errors; per-subgroup |Delta| stats.

Usage (dev 100, stratified gt x base-correct):
  CUDA_VISIBLE_DEVICES=1 python probe_cf_signal_cmm.py \
      --scored outputs/cmm_base_model/cmm_base_model_full_single_scored.jsonl \
      --n_per_cell 25 --out outputs/cf_probe/cmm_probe.jsonl
Full-population probe: --full (optionally sharded --worker i --num_workers 8).
"""
import os, sys, json, time, argparse, random

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'OmniConfess', 'methods'))
os.chdir(os.path.dirname(os.path.abspath(__file__)))   # repo root: OmniConfess/

import torch
from shared import (
    load_media_for_item, load_model, build_inputs,
    prepare_for_thinker, build_user_content_with_media_tokens, make_messages,
    get_system_prompt, get_user_prompt,
)

YES_ID, NO_ID = 9454, 2753   # 'Yes', 'No' (no leading space), same as phd
NSHARD = 8


def load_aligned(scored_path, data_dir):
    """Scored full files are shard-local-id concatenations (EXPERIMENT_DESIGN
    phd section 4 pitfall). Re-align each scored row to its dataset item via
    (shard, row index) by re-walking the shard files in order."""
    rows = [json.loads(l) for l in open(scored_path)]
    items = []
    for k in range(NSHARD):
        shard = json.load(open(os.path.join(
            data_dir, f'cmm_shard{k}of{NSHARD}.json')))
        for j, it in enumerate(shard):
            it = dict(it)
            it['id'] = str(j)          # load_dataset uses row index as id
            it['shard'] = k
            items.append(it)
    assert len(rows) == len(items), (len(rows), len(items))
    for r, it in zip(rows, items):
        assert r['id'] == it['id'] and r['question'] == it['question'], \
            f"misaligned row id={r['id']}"
        r['_shard'] = it['shard']
        r['_local_id'] = it['id']
    return rows, items


def find_decision_index(tokenizer, token_ids):
    """cmm base output shapes: bare 'Yes'/'No' as FIRST generated token
    (720/721) -> (0, ids[0]); rare \\boxed{Yes} tail handled like phd."""
    ids = list(token_ids)
    if ids and ids[0] in (YES_ID, NO_ID):
        return 0, ids[0]
    for i in range(len(ids) - 2, -1, -1):
        if tokenizer.decode([ids[i]]).strip() == 'boxed':
            if tokenizer.decode([ids[i + 1]]).strip() == '{':
                return i + 2, ids[i + 2] if i + 2 < len(ids) else None
    return None, None


@torch.inference_mode()
def view_decision_logp(model, clean, prefix_ids):
    """Forward (view inputs + teacher-forced prefix), return lp_yes, lp_no,
    argmax at the position that predicts the decision token."""
    input_ids = torch.cat([clean['input_ids'][0],
                           torch.tensor(prefix_ids, dtype=torch.long,
                                        device=clean['input_ids'].device)],
                          dim=0).unsqueeze(0)
    attn = torch.ones_like(input_ids)
    kw = dict(input_ids=input_ids, attention_mask=attn)
    for k in ('pixel_values_images', 'pixel_values', 'image_grid_thw',
              'pixel_values_videos', 'video_grid_thw', 'video_second_per_grid',
              'input_features', 'feature_attention_mask'):
        if k in clean:
            kw[k] = clean[k]
    logits = model.thinker(**kw).logits[0, -1].float()
    lp = torch.log_softmax(logits, dim=-1)
    return lp[YES_ID].item(), lp[NO_ID].item(), int(logits.argmax().item())


def build_view(processor, model, system_prompt, pv, av, iv, user_prompt):
    uc = build_user_content_with_media_tokens(pv, av, iv, user_prompt)
    clean = prepare_for_thinker(
        build_inputs(processor, model, make_messages(system_prompt, uc),
                     pv, av, iv, 'cmm'), model)
    return clean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scored', default='outputs/cmm_base_model/cmm_base_model_full_single_scored.jsonl')
    ap.add_argument('--data_dir', default='./OmniHalluBench/cmm')
    ap.add_argument('--data_root', default='./OmniHalluBench/')
    ap.add_argument('--n_per_cell', type=int, default=25,
                    help='samples per (gt x base-correct) cell -> 4*n total')
    ap.add_argument('--out', default='outputs/cf_probe/cmm_probe.jsonl')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--limit', type=int, default=0, help='debug: first N only')
    ap.add_argument('--full', action='store_true',
                    help='probe ALL scored rows (population signal stats)')
    ap.add_argument('--worker', type=int, default=0)
    ap.add_argument('--num_workers', type=int, default=1)
    args = ap.parse_args()

    random.seed(args.seed)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    rows, items = load_aligned(args.scored, args.data_dir)
    by_key = {(it['shard'], it['id']): it for it in items}

    cells = {(1, 1): [], (1, 0): [], (0, 1): [], (0, 0): []}   # (gt_yes, base_correct)
    for r in rows:
        if not r.get('token_ids'):
            continue
        gy = 1 if r['gt'] == 'yes' else 0
        bc = 1 if r['eval_metrics']['Is_Correct'] == 1 else 0
        cells[(gy, bc)].append(r)
    sample = []
    if args.full:
        for k in sorted(cells):
            sample += [(k, r) for r in cells[k]]
    else:
        for k, v in sorted(cells.items()):
            random.shuffle(v)
            sample += [(k, r) for r in v[:args.n_per_cell]]
    random.shuffle(sample)
    if args.num_workers > 1:
        sample = sample[args.worker::args.num_workers]
    if args.limit:
        sample = sample[:args.limit]
    print(f'[probe] sampled {len(sample)} '
          f'({ {k: len(v) for k, v in cells.items()} })', flush=True)

    print('[model] loading...', flush=True)
    processor, model = load_model()
    tokenizer = processor.tokenizer

    system_prompt = get_system_prompt('cmm', 'v1')
    data_path = os.path.join(args.data_dir, 'cmm.json')  # for media resolution

    t0 = time.time()
    n_done = 0
    with open(args.out, 'w') as fout:
        for (gy, bc), r in sample:
            item = by_key[(r['_shard'], r['_local_id'])]
            shard_item = dict(item)
            shard_item['video_path'] = (item.get('media') or {}).get('video')
            shard_item['audio_path'] = (item.get('media') or {}).get('audio')
            shard_item['image_path'] = (item.get('media') or {}).get('image')
            t1 = time.time()
            try:
                pv, av, iv = load_media_for_item(shard_item, data_path,
                                                 args.data_root)
                if pv is None or av is None:
                    print(f"  [skip] missing media id={r['id']}", flush=True)
                    continue

                prefix_len, dec_tok = find_decision_index(tokenizer, r['token_ids'])
                if prefix_len is None:
                    print(f"  [skip] no decision token id={r['id']}", flush=True)
                    continue
                prefix = r['token_ids'][:prefix_len]

                u = get_user_prompt(r['question'], 'cmm', 'v1')
                rec = dict(id=r['id'], shard=r['_shard'], local_id=r['_local_id'],
                           uid=item.get('uid'), gt=r['gt'], pred=r['pred'],
                           gt_yes=gy, base_correct=bc,
                           group=r['group_key'], dec_index=prefix_len,
                           dec_token=dec_tok, bare=int(prefix_len == 0),
                           dec_logp_recorded=r['logp_list'][prefix_len])

                # ---- full view: video + audio --------------------------------
                clean_f = build_view(processor, model, system_prompt,
                                     pv, av, iv, u)
                ly, ln, am = view_decision_logp(model, clean_f, prefix)
                rec.update(lp_yes_full=round(ly, 4), lp_no_full=round(ln, 4),
                           full_argmax=am,
                           full_argmax_is_dec=int(am == dec_tok),
                           margin_full=round(ly - ln, 4))

                # ---- noaudio view: video only --------------------------------
                clean_a = build_view(processor, model, system_prompt,
                                     pv, None, iv, u)
                ly2, ln2, am2 = view_decision_logp(model, clean_a, prefix)
                rec.update(lp_yes_noaudio=round(ly2, 4),
                           lp_no_noaudio=round(ln2, 4),
                           margin_noaudio=round(ly2 - ln2, 4),
                           delta_audio=round(ly - ly2, 4))       # full - masked

                # ---- novideo view: audio only --------------------------------
                clean_v = build_view(processor, model, system_prompt,
                                     None, av, iv, u)
                ly3, ln3, am3 = view_decision_logp(model, clean_v, prefix)
                rec.update(lp_yes_novideo=round(ly3, 4),
                           lp_no_novideo=round(ln3, 4),
                           margin_novideo=round(ly3 - ln3, 4),
                           delta_video=round(ly - ly3, 4))       # full - masked

                # phd convention (masked - full), what the operator adds:
                rec.update(d_cf_audio=round(ly2 - ly, 4),
                           d_cf_video=round(ly3 - ly, 4),
                           d_cf_sum=round(ly2 + ly3 - 2 * ly, 4))

                fout.write(json.dumps(rec, ensure_ascii=False) + '\n')
                fout.flush()
                n_done += 1
                if n_done % 10 == 0:
                    print(f'  [{n_done}/{len(sample)}] {time.time()-t0:.0f}s '
                          f'd_a={rec["delta_audio"]:+.2f} d_v={rec["delta_video"]:+.2f}',
                          flush=True)
                del clean_f, clean_a, clean_v
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"  [ERROR] id={r['id']}: {e}", flush=True)
            torch.cuda.empty_cache()

    print(f'[done] {n_done}/{len(sample)} samples in {time.time()-t0:.0f}s -> {args.out}',
          flush=True)


if __name__ == '__main__':
    main()
