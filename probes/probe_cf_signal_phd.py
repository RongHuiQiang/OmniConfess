#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""probe_cf_signal_phd.py -- CF-Confess signal-existence probe (Step 1, zero design commitment).

For a stratified dev sample of PhD, measure at the DECISION token (the token
right after `\\boxed{`) the counterfactual sensitivity

    Delta_view = log P(Yes | full view, prefix) - log P(Yes | masked view, prefix)

with the SAME generated prefix (teacher-forced from the recorded base_model
greedy trajectory, up to and including `{`). Views:

    full   : image + context + marker + question          (== base_model input)
    noimg  : context + marker + question (image removed)  -> Delta_noimg
    noctx  : image + marker + question (context removed)  -> Delta_noctx

Report: AUC of each signal for (a) predicting gt, (b) predicting the fix
direction among base errors, plus distribution stats and lambda-grid margins.

Usage:
  CUDA_VISIBLE_DEVICES=0 python probe_cf_signal_phd.py \
      --scored outputs/phd_base_model/phd_base_model_full_single_scored.jsonl \
      --n_per_cell 25 --out outputs/cf_probe/phd_probe.jsonl
"""
import os, sys, json, time, argparse, random

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'OmniConfess', 'methods'))
os.chdir(os.path.dirname(os.path.abspath(__file__)))   # repo root: OmniConfess/

import torch
from shared import (
    load_dataset, load_media_for_item, load_model, build_inputs,
    prepare_for_thinker, build_user_content_with_media_tokens, make_messages,
    get_system_prompt, get_user_prompt,
)

MARKER = ("In case there is an inconsistency between the context and the image "
          "content, you should follow the image.")

YES_ID, NO_ID = 9454, 2753   # 'Yes', 'No' (no leading space), verified in probe env


def split_question(q):
    """phd question = context + marker sentence + question sentence."""
    idx = q.find(MARKER)
    if idx < 0:
        return None, None, q
    ctx = q[:idx].rstrip()
    rest = q[idx + len(MARKER):].lstrip()
    return ctx, MARKER, rest


def find_decision_index(tokenizer, token_ids):
    """Return (prefix_len, dec_token_id).

    Two output shapes in the base trajectories:
      bare  : 'Yes'/'No' as the very FIRST generated token (753/906) -> (0, ids[0])
      boxed : '... Judgment: \\boxed{Yes}' -> prefix ends right after '{'
    """
    ids = list(token_ids)
    if ids and ids[0] in (YES_ID, NO_ID):
        return 0, ids[0]
    for i in range(len(ids) - 2, -1, -1):
        if tokenizer.decode([ids[i]]).strip() == 'boxed':
            if tokenizer.decode([ids[i + 1]]).strip() == '{':
                return i + 2, ids[i + 2] if i + 2 < len(ids) else None
    # fallback: incremental decode (rare tokenizer merges)
    for i in range(len(ids) - 1, 0, -1):
        if tokenizer.decode(ids[:i]).endswith('\\boxed{'):
            return i, ids[i]
    return None, None


@torch.inference_mode()
def decision_logp(model, clean, prefix_ids, device):
    """Forward (view inputs + teacher-forced prefix), return lp_yes, lp_no at
    the position that predicts the decision token."""
    input_ids = torch.cat([clean['input_ids'][0],
                           torch.tensor(prefix_ids, dtype=torch.long,
                                        device=clean['input_ids'].device)], dim=0
                          ).unsqueeze(0)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scored', default='outputs/phd_base_model/phd_base_model_full_single_scored.jsonl')
    ap.add_argument('--data_path', default='./OmniHalluBench/phd/annotations/phd.json')
    ap.add_argument('--data_root', default='./OmniHalluBench/')
    ap.add_argument('--n_per_cell', type=int, default=25,
                    help='samples per (gt x base-correct) cell -> 4*n total')
    ap.add_argument('--out', default='outputs/cf_probe/phd_probe.jsonl')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--limit', type=int, default=0, help='debug: first N only')
    ap.add_argument('--full', action='store_true',
                    help='probe ALL scored rows (population signal stats)')
    args = ap.parse_args()

    random.seed(args.seed)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    rows = [json.loads(l) for l in open(args.scored)]
    data = load_dataset(args.data_path, args.data_root)
    by_q = {}
    for it in data:
        by_q.setdefault(it['question'], []).append(it)

    cells = {(1, 1): [], (1, 0): [], (0, 1): [], (0, 0): []}   # (gt_yes, base_correct)
    for r in rows:
        if not r.get('token_ids'):
            continue
        gy = 1 if r['gt'] == 'yes' else 0
        bc = 1 if r['eval_metrics']['Is_Correct'] == 1 else 0
        cells[(gy, bc)].append(r)
    sample = []
    if args.full:
        allrows = [r for k in cells for r in cells[k]]
        sample = [((1 if r['gt'] == 'yes' else 0,
                    1 if r['eval_metrics']['Is_Correct'] == 1 else 0), r)
                  for r in allrows]
    else:
        for k, v in sorted(cells.items()):
            random.shuffle(v)
            sample += [(k, r) for r in v[:args.n_per_cell]]
    random.shuffle(sample)
    if args.limit:
        sample = sample[:args.limit]
    print(f'[probe] sampled {len(sample)} '
          f'({ {k: min(len(v), args.n_per_cell) for k, v in cells.items()} })', flush=True)

    print('[model] loading...', flush=True)
    processor, model = load_model()
    tokenizer = processor.tokenizer
    device = model.device

    system_prompt = get_system_prompt('phd', 'v1')

    t0 = time.time()
    with open(args.out, 'w') as fout:
        for n, ((gy, bc), r) in enumerate(sample):
            items = by_q.get(r['question'])
            if not items:
                print(f'  [skip] question not found id={r["id"]}', flush=True)
                continue
            item = items[0]
            pv, av, iv = load_media_for_item(item, args.data_path, args.data_root)
            if iv is None:
                print(f'  [skip] no image id={r["id"]}', flush=True)
                continue

            prefix_len, dec_tok = find_decision_index(tokenizer, r['token_ids'])
            if prefix_len is None:
                print(f'  [skip] no boxed decision id={r["id"]}', flush=True)
                continue
            prefix = r['token_ids'][:prefix_len]

            ctx, marker, q_only = split_question(r['question'])
            assert q_only and marker

            rec = dict(id=r['id'], gt=r['gt'], pred=r['pred'], gt_yes=gy,
                       base_correct=bc, dec_index=prefix_len,
                       dec_token=dec_tok,
                       dec_logp_recorded=r['logp_list'][prefix_len])

            # ---- full view -------------------------------------------------
            u_full = get_user_prompt(r['question'], 'phd', 'v1')
            uc_full = build_user_content_with_media_tokens(pv, av, iv, u_full)
            clean_f = prepare_for_thinker(
                build_inputs(processor, model, make_messages(system_prompt, uc_full),
                             pv, av, iv, 'phd'), model)
            ly, ln, am = decision_logp(model, clean_f, prefix, device)
            rec.update(lp_yes_full=round(ly, 4), lp_no_full=round(ln, 4),
                       full_argmax=am, full_argmax_is_dec=int(am == dec_tok),
                       margin_full=round(ly - ln, 4))

            # ---- noimg view ------------------------------------------------
            u_txt = get_user_prompt(r['question'], 'phd', 'v1')
            uc_txt = build_user_content_with_media_tokens(None, None, None, u_txt)
            clean_t = prepare_for_thinker(
                build_inputs(processor, model, make_messages(system_prompt, uc_txt),
                             None, None, None, 'phd'), model)
            ly2, ln2, am2 = decision_logp(model, clean_t, prefix, device)
            rec.update(lp_yes_noimg=round(ly2, 4), lp_no_noimg=round(ln2, 4),
                       margin_noimg=round(ly2 - ln2, 4),
                       delta_noimg=round(ly - ly2, 4))

            # ---- noctx view ------------------------------------------------
            u_nc = get_user_prompt(q_only, 'phd', 'v1')
            uc_nc = build_user_content_with_media_tokens(pv, av, iv, u_nc)
            clean_c = prepare_for_thinker(
                build_inputs(processor, model, make_messages(system_prompt, uc_nc),
                             pv, av, iv, 'phd'), model)
            ly3, ln3, am3 = decision_logp(model, clean_c, prefix, device)
            rec.update(lp_yes_noctx=round(ly3, 4), lp_no_noctx=round(ln3, 4),
                       margin_noctx=round(ly3 - ln3, 4),
                       delta_noctx=round(ly - ly3, 4))

            fout.write(json.dumps(rec, ensure_ascii=False) + '\n')
            fout.flush()
            if (n + 1) % 10 == 0:
                print(f'  [{n+1}/{len(sample)}] {time.time()-t0:.0f}s '
                      f'd_noimg={rec["delta_noimg"]:+.2f} d_noctx={rec["delta_noctx"]:+.2f}',
                      flush=True)
            del clean_f, clean_t, clean_c
            torch.cuda.empty_cache()

    print(f'[done] {len(sample)} samples in {time.time()-t0:.0f}s -> {args.out}', flush=True)


if __name__ == '__main__':
    main()
