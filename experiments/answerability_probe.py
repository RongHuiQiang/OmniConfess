# -*- coding: utf-8 -*-
"""answerability_probe.py — Step-1 signal probe for the ANSWERABILITY GATE
(可答性二分类门) on HaloQuest, card 3 of the descriptive four-card race.

Background (see EXPERIMENT_DESIGN.md 描述类 · CF-Confess @ HaloQuest):
  ~89/365 HaloQuest items are unanswerable (asked object absent / insufficient
  context) yet the base model answers affirmatively (Token-F1 ≈ 18 on that
  subset vs ≈ 30+ when an echo-style avoidance sentence is used). The previous
  deletion-style operator failed at flag precision 61% vs the 62% break-even.
  This card asks ONE question only: can a cheap combination of signals push
  unanswerable-item detection precision past 70% at a flag rate ≈ the
  unanswerable rate (~24%)?

Signals (per sample, answer granularity — NOT per claim; the gate is binary):
  S1 grounding  : teacher-forced two-view Delta over the recorded base greedy
                  trajectory, reusing the probe_claim_cf.py haloquest pipeline
                  (full image vs swapped image, A2: generation never perturbed)
                    d_content    = mean Delta over content/num tokens
                    d_content_ne = same, excluding question-echo words
                                   (words whose lemma also occurs in the
                                   question — kills the echo confound)
                    d_attr       = mean Delta over attribute tokens
                                   (color/number; the 0.744-AUC layer)
                    d_top2       = mean of the 2 largest content-token Deltas
                                   ("is ANY part of the answer grounded")
  S2 priors     : zero-cost rules
                    howmany_digit = question contains "how many" AND the
                                    answer contains a digit (false-premise
                                    counting pattern, pred "0"/"2 birds")
                    pred_evade    = answer already avoidance-phrased
                                    (EVADERE on pred — gate hard-skips these)
  S3 (labels)   : neg_gt      = gt is an avoidance sentence (EVADERE on gt)
                   needs_gate = neg_gt AND not pred_evade  (≈ 84-89 items,
                               exactly the subset where replacing the answer
                               with an echo avoidance sentence pays)

Pre-registered build/pass criteria:
  dev  = 100 stratified samples  -> fit logistic combination, pick tau
  test = remaining               -> AUC(needs_gate), precision at
                                    flag_rate ≈ 24% (88/365)
  GATE PASSES if test precision > 70% at that flag rate.

Usage:
    CUDA_VISIBLE_DEVICES=3 python answerability_probe.py \
        --out outputs/answerability/haloquest_answerability_probe.json
"""
import os, sys, json, argparse, random, time
from collections import Counter

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from probe_claim_cf import (  # noqa: E402
    EVADERE, token_class, view_logp, build_view, content_words,
)
sys.path.insert(0, os.path.join(HERE, 'OmniConfess', 'methods'))
from shared import (  # noqa: E402
    load_dataset, load_media_for_item, resolve_media_path,
    load_model, get_system_prompt, get_user_prompt,
)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("MMHALU_MODEL_ROOT", "/ultrafast_share/tsq/MODELS")

COLORS = set("""red blue green yellow black white orange purple pink brown gray grey
gold silver beige teal maroon khaki tan turquoise violet indigo crimson""".split())
NUMW = set('zero one two three four five six seven eight nine ten several many'.split())


def is_attr(w):
    return w in COLORS or any(ch.isdigit() for ch in w) or w in NUMW


def run(args):
    scored = os.path.join(HERE, 'outputs', 'haloquest_base_model',
                          'haloquest_base_model_full_single_scored.jsonl')
    data_path = os.path.join(HERE, 'OmniHalluBench', 'haloquest', 'haloquest.json')
    data_root = os.path.join(HERE, 'OmniHalluBench')

    recs = [json.loads(l) for l in open(scored)]
    recs = [r for r in recs if r.get('token_ids')]
    data = load_dataset(data_path, data_root)
    by_id = {it['id']: it for it in data}

    pool_images = [it for it in data if it.get('image_path') and os.path.exists(
        resolve_media_path(it['image_path'], os.path.dirname(data_path), data_root))]

    processor, model = load_model()
    tokenizer = processor.tokenizer
    thinker = model.thinker
    system_prompt = get_system_prompt('haloquest', 'v1')

    rng = random.Random(args.seed + 7)
    rows, t0 = [], time.time()
    n_done = 0
    if args.limit:
        recs = recs[:args.limit]
    for si, rec in enumerate(recs):
        item = by_id.get(rec['id'])
        if not item or not item.get('image_path') or not os.path.exists(
                resolve_media_path(item['image_path'], os.path.dirname(data_path), data_root)):
            rows.append({'sample_i': si, 'id': rec['id'], 'ok': False,
                         'reason': 'image-missing'})
            continue
        gen_ids = torch.tensor(rec['token_ids'], dtype=torch.long, device=model.device)
        toks = [tokenizer.decode([t], skip_special_tokens=True) for t in rec['token_ids']]
        cls = [token_class(t, toks[i - 1] if i else None) for i, t in enumerate(toks)]

        user_prompt = get_user_prompt(item['question'], 'haloquest', 'v1')
        img = load_media_for_item(item, data_path, data_root)[2]
        partners = []
        while len(partners) < args.partners:
            cand = rng.choice(pool_images)
            if cand['id'] != item['id'] and all(cand['id'] != p['id'] for p in partners):
                partners.append(cand)
        views = {'full': build_view(processor, model, system_prompt, user_prompt,
                                    img, 'haloquest')}
        for pi, p in enumerate(partners):
            pimg = load_media_for_item(p, data_path, data_root)[2]
            views[f'ctrl{pi}'] = build_view(processor, model, system_prompt,
                                            user_prompt, pimg, 'haloquest')
        lp = {n: view_logp(thinker, v, gen_ids) for n, v in views.items()}
        # multi-partner denoised Delta: full vs MEAN over ctrl views
        ctrl = np.mean([lp[f'ctrl{pi}'] for pi in range(args.partners)], axis=0)
        delta = lp['full'] - ctrl
        partner = partners[0]

        # ---- grounding features over the answer tokens ----------------------
        q_words = set(content_words(item['question']))
        idx_content = [j for j in range(len(toks)) if cls[j] in ('content', 'num')]
        idx_attr = [j for j in idx_content
                    if is_attr(toks[j].strip().lower().strip('.,;:!?'))]
        idx_ne = [j for j in idx_content
                  if toks[j].strip().lower().strip('.,;:!?') not in q_words]

        def _mean(idx):
            return float(np.mean([delta[j] for j in idx])) if idx else None

        d_content = _mean(idx_content)
        d_ne = _mean(idx_ne) if idx_ne else d_content
        d_attr = _mean(idx_attr) if idx_attr else None
        d_top2 = float(np.mean(sorted((delta[j] for j in idx_content),
                                      reverse=True)[:2])) if idx_content else None
        lp_full_mean = float(np.mean(lp['full']))

        # ---- prior features --------------------------------------------------
        q_lower = item['question'].lower()
        pred = rec['pred'] or ''
        howmany_digit = ('how many' in q_lower) and any(c.isdigit() for c in pred)
        pred_evade = bool(EVADERE.search(pred))
        neg_gt = bool(EVADERE.search(rec['gt'] or ''))
        needs_gate = neg_gt and not pred_evade

        rows.append({
            'sample_i': si, 'id': rec['id'], 'ok': True,
            'question': item['question'], 'gt': rec['gt'], 'pred': pred,
            'base_f1': rec['eval_metrics']['Token-F1'],
            'group_key': rec.get('group_key'),
            'partner': partner['id'],
            # signals (answerability direction: high d_* = grounded = answerable)
            'd_content': d_content, 'd_content_ne': d_ne,
            'd_attr': d_attr, 'd_top2': d_top2,
            'lp_full_mean': lp_full_mean,
            'n_content': len(idx_content), 'n_attr': len(idx_attr),
            'howmany_digit': bool(howmany_digit),
            'pred_evade': pred_evade,
            # labels
            'neg_gt': neg_gt, 'needs_gate': needs_gate,
            # per-token view for post-hoc diagnostics by the operator
            'delta_tok': [round(float(x), 4) for x in delta],
        })
        n_done += 1
        if n_done % 25 == 0:
            print(f'  [{n_done}/{len(recs)}] {time.time()-t0:.0f}s', flush=True)
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump({'rows': rows, 'meta': {
            'n': len(rows), 'n_ok': n_done, 'seed': args.seed,
            'scored_src': scored, 'generated': time.strftime('%Y-%m-%d %H:%M:%S'),
        }}, f, ensure_ascii=False)
    ok = [r for r in rows if r['ok']]
    print(f'[done] {n_done} ok / {len(rows)} -> {args.out}')
    print(f'  neg_gt={sum(r["neg_gt"] for r in ok)} '
          f'needs_gate={sum(r["needs_gate"] for r in ok)} '
          f'pred_evade={sum(r["pred_evade"] for r in ok)}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(
        HERE, 'outputs', 'answerability', 'haloquest_answerability_probe.json'))
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--partners', type=int, default=1,
                    help='number of swapped-image controls (Delta = full vs mean ctrl)')
    args = ap.parse_args()
    run(args)
