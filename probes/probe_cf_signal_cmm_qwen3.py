#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""probe_cf_signal_cmm_qwen3.py -- CF signal probe: CMM x Qwen3-Omni-30B-A3B.

Second test point of the "generational self-healing" hypothesis. The phd
dominance disease self-healed on Qwen3 (probe AUC 0.550 <= 0.65 gate;
operator gains == random-sign control, McNemar p=0.77). The cmm disease
(unimodal-prior over-Yes) has NEVER been probed on Qwen3: base Yes-F1 is
already 74.56 (Q2.5 70.07), but the behavioral marker persists -- yes-rate
60.2% vs gt 41.7% (+18.5pp; overrely_audio subgroup +29.8pp, Q2.5 +34.7pp),
FPR .381 (Q2.5 .555). Question: does the audio/video leakage signal that
carried +6.29 F1 on Qwen2.5 still exist at the decision token?

Same protocol as probe_cf_signal_cmm.py (Qwen2.5 arm) on the Qwen3 backbone
via shared/qwen3_omni_utils.py: thinker-only load (~60.5GB, one free card,
single process), model IS the thinker, video key pixel_values_videos.

Base trajectories (eos-fix applied, verified: outputs stop at <|im_end|>):
  /ultrafast_share/tsq/qwen3omni_outputs/cmm_qwen3omni/
  cmm_qwen3omni_full_single_scored.jsonl  (721 rows, 8-shard concatenation,
  alignment id+question re-verified against shards/cmm/cmm_shardKof8.json)

Views at the decision token (teacher-forced on the base greedy prefix),
identical question text, media tokens skipped for the masked modality
(same construction as cf_confess_cmm.py):
  full     : video + audio + question      (== base input)
  noaudio  : video + question              -> d_cf_audio (audio pull on Yes)
  novideo  : audio + question              -> d_cf_video (video pull on Yes)

Decision token (cmm Final-Answer format, STRICT OUTPUT FORMAT prompt):
  Qwen3 cmm base answers "Description: ...\nJudgment: \\boxed{Yes|No}"
  (716/721 boxed, 0/721 bare) -> located by the backwards 'boxed' '{' scan
  (bare first-token Yes/No kept defensively, as on the phd arm).

PRE-REGISTERED JUDGMENT (written before running; cf. EXPERIMENT_DESIGN.md
"CF cross-backbone" cmm subsection):
  - Sample: 100 rows, 25 x (gt_yes/no x subgroup{overrely_audio,
    overrely_visual}), seed 42, from scored rows with a locatable decision
    token.
  - Signals: margin_full, d_cf_audio, d_cf_video, d_cf_sum
    (d_cf_i = lp_yes(masked) - lp_yes(full), the operator's convention).
  - ALIVE := AUC(-d_cf_sum -> gt) > 0.65 AND Qwen2.5-type direction
    structure: (i) d_cf_sum < 0 on the large majority of base-Yes samples
    (masking a modality drops Yes = a modality is propping Yes up; Q2.5
    99.9%), (ii) negative d_cf means among gt=no & base=yes (the FP mass,
    "audio leakage pushes yes"), (iii) error-Yes margins < correct-Yes
    margins (Q2.5: +1.52 vs +3.42). ALIVE -> run the full 721 CF@cmm-Qwen3
    (lambda re-selected on this dev-100 by the pre-registered cmm exemption
    rule: max Youden on the offline grid, +/-5pp yes-rate rule infeasible
    for a +18.5pp over-Yes base; random-sign control arm alongside).
  - AUC <= 0.65 -> DEAD: negative result, completing the phd+cmm
    "generational self-healing" double evidence. Supplementary (still
    reported, not gate-relevant): offline lambda grid on the probe records
    (exact for the recorded decision step, phd-Qwen3 online/offline 100/100
    precedent), per-subgroup |d_cf|, teacher-forcing reproduction check
    (lp at the decision token == base-recorded dec logp).

Usage (one exclusive free card; GPU0 as of 2026-08-16, GPUs1-7 occupied):
  CUDA_VISIBLE_DEVICES=0 python probe_cf_signal_cmm_qwen3.py --n_per_cell 25
"""
import os, sys, json, time, argparse, random

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'OmniConfess', 'methods'))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import torch
from shared import load_media_for_item, get_system_prompt, get_user_prompt
from shared.qwen3_omni_utils import (
    load_model, build_inputs, prepare_for_thinker,
    build_user_content_with_media_tokens, make_messages,
)

YES_ID, NO_ID = 9454, 2753   # verified identical ids on the Qwen3 tokenizer
NSHARD = 8
SHARD_FMT = {
    'run':  '/ultrafast_share/tsq/qwen3omni_outputs/shards/cmm/cmm_shard{k}of8.json',
    'repo': 'OmniHalluBench/cmm/cmm_shard{k}of8.json',
}


def load_aligned(scored_path):
    """Scored full file = 8-shard concatenation (row-local ids). Re-walk the
    shard INPUT files in order and re-attach (shard, local_id) + media paths.
    GPFS shards are the exact run inputs; repo shards as fallback."""
    rows = [json.loads(l) for l in open(scored_path)]
    items = []
    for k in range(NSHARD):
        src = SHARD_FMT['run'].format(k=k)
        if not os.path.exists(src):
            src = SHARD_FMT['repo'].format(k=k)
        shard = json.load(open(src))
        for j, it in enumerate(shard):
            it = dict(it)
            it['id'] = str(j)
            it['shard'] = k
            items.append(it)
    assert len(rows) == len(items), (len(rows), len(items))
    for r, it in zip(rows, items):
        assert r['id'] == it['id'] and r['question'] == it['question'], \
            f"misaligned row id={r['id']}"
        r['_shard'] = it['shard']
        r['_local_id'] = it['id']
    return rows, items


def sub_group(group_key):
    return 'audio' if group_key.startswith('overrely_audio') else 'visual'


def find_decision_index(tokenizer, token_ids):
    """cmm Final-Answer format on Qwen3: 'Judgment: \\boxed{Yes|No}' tail
    (716/721); bare first-token Yes/No kept defensively (Qwen2.5 shape)."""
    ids = list(token_ids)
    if ids and ids[0] in (YES_ID, NO_ID):
        return 0, ids[0]
    for i in range(len(ids) - 2, -1, -1):
        if tokenizer.decode([ids[i]]).strip() == 'boxed':
            if tokenizer.decode([ids[i + 1]]).strip() == '{':
                return i + 2, ids[i + 2] if i + 2 < len(ids) else None
    return None, None


@torch.inference_mode()
def decision_logp(model, clean, prefix_ids):
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
    logits = model(**kw).logits[0, -1].float()
    lp = torch.log_softmax(logits, dim=-1)
    return lp[YES_ID].item(), lp[NO_ID].item(), int(logits.argmax().item())


def build_view(processor, model, system_prompt, pv, av, iv, user_prompt):
    uc = build_user_content_with_media_tokens(pv, av, iv, user_prompt)
    return prepare_for_thinker(build_inputs(
        processor, model, make_messages(system_prompt, uc),
        pv, av, iv, 'cmm'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scored', default='/ultrafast_share/tsq/qwen3omni_outputs/cmm_qwen3omni/cmm_qwen3omni_full_single_scored.jsonl')
    ap.add_argument('--data_path', default='./OmniHalluBench/cmm/annotations/cmm.json')
    ap.add_argument('--data_root', default='./OmniHalluBench/')
    ap.add_argument('--n_per_cell', type=int, default=25,
                    help='per (gt x subgroup) cell -> 4*n total')
    ap.add_argument('--full', action='store_true', help='all rows (no sampling)')
    ap.add_argument('--out', default='outputs/cf_probe/cmm_probe_qwen3.jsonl')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    rows, items = load_aligned(args.scored)
    by_key = {(it['shard'], it['id']): it for it in items}

    # cells: (gt_yes, subgroup) -- pre-registered stratification
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

    print('[model] loading Qwen3-Omni thinker (~60.5GB, needs an exclusive card)...',
          flush=True)
    processor, model = load_model()
    tokenizer = processor.tokenizer
    system_prompt = get_system_prompt('cmm', 'v1')

    t0 = time.time()
    n_done = 0
    with open(args.out, 'w') as fout:
        for (gy, sg), r in sample:
            item = by_key[(r['_shard'], r['_local_id'])]
            t1 = time.time()
            try:
                shard_item = dict(item)
                media = item.get('media') or {}
                shard_item['video_path'] = media.get('video')
                shard_item['audio_path'] = media.get('audio')
                shard_item['image_path'] = media.get('image')
                pv, av, iv = load_media_for_item(shard_item, args.data_path,
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
                rec = dict(id=r['id'], shard=item['shard'], local_id=r['id'],
                           uid=item.get('uid'), gt=r['gt'], pred=r['pred'],
                           gt_yes=gy, base_correct=r['eval_metrics']['Is_Correct'],
                           group=r['group_key'], sub_group=sg,
                           dec_index=prefix_len, dec_token=dec_tok,
                           bare=int(prefix_len == 0),
                           dec_logp_recorded=r['logp_list'][prefix_len])

                # ---- full view: video + audio --------------------------------
                clean_f = build_view(processor, model, system_prompt,
                                     pv, av, iv, u)
                ly, ln, am = decision_logp(model, clean_f, prefix)
                rec.update(lp_yes_full=round(ly, 4), lp_no_full=round(ln, 4),
                           full_argmax=am,
                           full_argmax_is_dec=int(am == dec_tok),
                           margin_full=round(ly - ln, 4))

                # ---- noaudio view: video only --------------------------------
                clean_a = build_view(processor, model, system_prompt,
                                     pv, None, iv, u)
                ly2, ln2, am2 = decision_logp(model, clean_a, prefix)
                rec.update(lp_yes_noaudio=round(ly2, 4),
                           lp_no_noaudio=round(ln2, 4),
                           margin_noaudio=round(ly2 - ln2, 4),
                           delta_audio=round(ly - ly2, 4))       # full - masked

                # ---- novideo view: audio only --------------------------------
                clean_v = build_view(processor, model, system_prompt,
                                     None, av, iv, u)
                ly3, ln3, am3 = decision_logp(model, clean_v, prefix)
                rec.update(lp_yes_novideo=round(ly3, 4),
                           lp_no_novideo=round(ln3, 4),
                           margin_novideo=round(ly3 - ln3, 4),
                           delta_video=round(ly - ly3, 4))       # full - masked

                # operator convention (masked - full), what the operator adds
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
                          f'd_a={rec["delta_audio"]:+.2f} d_v={rec["delta_video"]:+.2f} '
                          f'tf={rec["tf_repro"]:+.4f}', flush=True)
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
