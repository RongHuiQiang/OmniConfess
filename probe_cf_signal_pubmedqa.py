#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""probe_cf_signal_pubmedqa.py -- CF-Confess signal-existence probe for PubMedQA
(Step 1, zero design commitment), text-only twin of probe_cf_signal_phd.py.

Decision token: pubmedqa base answers begin with a bare 'Yes.' / 'No.' /
'Maybe.' as the very first generated token (401/57/10 of 468), so the
decision step is step 0 with an empty teacher-forced prefix.

Single masked view (the only one a text dataset has -- no second view to
cross-check, hence the pre-registered WEAK-effect prediction):
    Delta_passage = log P(Yes | full) - log P(Yes | passage-removed)
positive Delta_passage = the passage is suppressing Yes (verdict leans on the
prior without it); negative = passage inflates Yes.

The passage-removed view drops the "Passage: ..." line from the shared v1
user prompt (same construction as the ragtruth probe).

Usage (dev 100, stratified gt(yes/no) x base-correct; gt=maybe rows are
excluded exactly like the scorer excludes them):
  CUDA_VISIBLE_DEVICES=2 python probe_cf_signal_pubmedqa.py \
      --scored outputs/pubmedqa_base_model/pubmedqa_base_model_full_single_scored.jsonl \
      --n_per_cell 25 --out outputs/cf_probe/pubmedqa_probe.jsonl
"""
import os, sys, json, time, argparse, random

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'OmniConfess', 'methods'))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import torch
from shared import (
    load_model, build_inputs, prepare_for_thinker,
    build_user_content_with_media_tokens, make_messages,
    get_system_prompt, get_user_prompt,
)

YES_ID, NO_ID, MAYBE_ID = 9454, 2753, 21390   # 'Yes', 'No', 'Maybe' verified
NSHARD = 8


def strip_passage_line(user_prompt):
    """Remove the leading 'Passage: ...' line from the shared v1 user prompt."""
    lines = user_prompt.split('\n')
    if lines and lines[0].startswith('Passage:'):
        return '\n'.join(lines[1:])
    return user_prompt


def load_aligned(scored_path, data_dir):
    """Unlike cmm/phd (row-index ids -> shard-local, join needs shard+id),
    pubmedqa items carry their own globally unique original ids, so a plain
    id join is exact."""
    rows = [json.loads(l) for l in open(scored_path)]
    data = json.load(open(os.path.join(data_dir, 'pubmedqa.json')))
    shard_of = {}
    for k in range(NSHARD):
        for it in json.load(open(os.path.join(
                data_dir, f'pubmedqa_shard{k}of{NSHARD}.json'))):
            shard_of[str(it.get('id') if 'id' in it
                             else (it.get('source') or {}).get('orig_id'))] = k
    items = {}
    for it in data:
        iid = str(it.get('id') if 'id' in it
                  else (it.get('source') or {}).get('orig_id'))
        it = dict(it)
        it['id'] = iid
        it['shard'] = shard_of.get(iid, -1)
        items[iid] = it
    assert len(rows) == len(items) == len(set(items)), (len(rows), len(items))
    for r in rows:
        assert r['question'] == items[r['id']]['question']
        r['_shard'] = items[r['id']]['shard']
        r['_local_id'] = r['id']
    return rows, list(items.values())


def gt_class(gt):
    g = str(gt).lower().strip()
    if g.startswith('yes'):
        return 1
    if g.startswith('no'):
        return 0
    return 'maybe'


@torch.inference_mode()
def view_decision_logp(model, clean, prefix_ids):
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
    return (lp[YES_ID].item(), lp[NO_ID].item(), lp[MAYBE_ID].item(),
            int(logits.argmax().item()))


def build_view(processor, model, system_prompt, user_prompt):
    uc = build_user_content_with_media_tokens(None, None, None, user_prompt)
    clean = prepare_for_thinker(
        build_inputs(processor, model, make_messages(system_prompt, uc),
                     None, None, None, 'pubmedqa'), model)
    return clean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scored', default='outputs/pubmedqa_base_model/pubmedqa_base_model_full_single_scored.jsonl')
    ap.add_argument('--data_dir', default='./OmniHalluBench/pubmedqa')
    ap.add_argument('--n_per_cell', type=int, default=25)
    ap.add_argument('--out', default='outputs/cf_probe/pubmedqa_probe.jsonl')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--full', action='store_true')
    ap.add_argument('--worker', type=int, default=0)
    ap.add_argument('--num_workers', type=int, default=1)
    args = ap.parse_args()

    random.seed(args.seed)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    rows, items = load_aligned(args.scored, args.data_dir)
    by_key = {(it['shard'], it['id']): it for it in items}

    cells = {(1, 1): [], (1, 0): [], (0, 1): [], (0, 0): []}   # (gt_yes, base_correct)
    n_maybe = 0
    for r in rows:
        if not r.get('token_ids'):
            continue
        gy = gt_class(r['gt'])
        if gy == 'maybe':
            n_maybe += 1
            continue                                    # unscoreable, skip
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
    print(f'[probe] sampled {len(sample)} (maybe-gt excluded: {n_maybe}; '
          f'cells { {k: len(v) for k, v in cells.items()} })', flush=True)

    print('[model] loading...', flush=True)
    processor, model = load_model()
    tokenizer = processor.tokenizer

    system_prompt = get_system_prompt('pubmedqa', 'v1')

    t0 = time.time()
    n_done = 0
    with open(args.out, 'w') as fout:
        for (gy, bc), r in sample:
            item = by_key[(r['_shard'], r['_local_id'])]
            passage = item.get('passage')
            t1 = time.time()
            try:
                ids = r['token_ids']
                dec_tok = ids[0]
                if dec_tok not in (YES_ID, NO_ID, MAYBE_ID):
                    print(f"  [skip] first token not Yes/No/Maybe id={r['id']}",
                          flush=True)
                    continue
                prefix = []                     # decision at step 0, bare

                rec = dict(id=r['id'], shard=r['_shard'], local_id=r['_local_id'],
                           uid=item.get('uid'), gt=gt_class(r['gt']),
                           pred_tok=r['eval_metrics']['Prediction'],
                           gt_yes=gy, base_correct=bc,
                           dec_index=0, dec_token=dec_tok,
                           dec_logp_recorded=r['logp_list'][0])

                # ---- full view: passage + instruction + question ------------
                u_full = get_user_prompt(r['question'], 'pubmedqa', 'v1',
                                         passage=passage)
                clean_f = build_view(processor, model, system_prompt, u_full)
                ly, ln, lm, am = view_decision_logp(model, clean_f, prefix)
                rec.update(lp_yes_full=round(ly, 4), lp_no_full=round(ln, 4),
                           lp_maybe_full=round(lm, 4), full_argmax=am,
                           full_argmax_is_dec=int(am == dec_tok))

                # ---- nopassage view ------------------------------------------
                u_cf = strip_passage_line(u_full)
                clean_c = build_view(processor, model, system_prompt, u_cf)
                ly2, ln2, lm2, am2 = view_decision_logp(model, clean_c, prefix)
                rec.update(lp_yes_nopassage=round(ly2, 4),
                           lp_no_nopassage=round(ln2, 4),
                           lp_maybe_nopassage=round(lm2, 4),
                           margin_full=round(ly - ln, 4),
                           margin_nopassage=round(ly2 - ln2, 4),
                           delta_passage=round(ly - ly2, 4),   # full - masked
                           d_cf=round(ly2 - ly, 4))            # phd conv.

                fout.write(json.dumps(rec, ensure_ascii=False) + '\n')
                fout.flush()
                n_done += 1
                if n_done % 20 == 0:
                    print(f'  [{n_done}/{len(sample)}] {time.time()-t0:.0f}s '
                          f'd_p={rec["delta_passage"]:+.2f}', flush=True)
                del clean_f, clean_c
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"  [ERROR] id={r['id']}: {e}", flush=True)
            torch.cuda.empty_cache()

    print(f'[done] {n_done}/{len(sample)} in {time.time()-t0:.0f}s -> {args.out}',
          flush=True)


if __name__ == '__main__':
    main()
