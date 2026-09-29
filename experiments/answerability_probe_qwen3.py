# -*- coding: utf-8 -*-
"""answerability_probe_qwen3.py — Qwen3-Omni port of answerability_probe.py.

Stage-1 signal probe for the ANSWERABILITY GATE (可答性门) on HaloQuest,
ported to Qwen3-Omni-30B-A3B-Instruct (cross-backbone mechanism transfer).

WHAT CHANGES vs the Qwen2.5 original (answerability_probe.py):
  1. Model load: shared.qwen3_shim.load_model  -> returns the Qwen3 thinker
     itself with a self-alias `model.thinker = model`, so `thinker = model.thinker`
     resolves to the model (no .thinker hop on Qwen3).
  2. Media placeholder spelling: build_view here uses the Qwen3 media-token
     helper (<|vision_start|><|image_pad|><|vision_end|>) re-exported by the
     shim, NOT the Qwen2.5 <|IMAGE|> spelling. build_inputs / prepare_for_thinker
     also come from the shim (pixel_values key, scalar fps, eos pins).
  3. Base greedy trajectory: teacher-forces the QWEN3 base_model haloquest
     trajectory (token_ids recorded by base_model_qwen3omni, Token-F1 33.38,
     RESULTS line 70), NOT the Qwen2.5 base trajectory.

Everything else (feature extraction, labels, two-view Delta, A2 = generation
never perturbed) is byte-identical in logic to the Qwen2.5 probe so the gate
scorer / dev-CV protocol (answerability_gate.py) runs unchanged downstream.

Usage:
    CUDA_VISIBLE_DEVICES=7 python answerability_probe_qwen3.py \
        --out outputs/answerability_qwen3/haloquest_answerability_probe_qwen3.json \
        --partners 3
"""
import os, sys, json, argparse, random, time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# backbone-agnostic helpers reused verbatim from the Qwen2.5 probe machinery
from probe_claim_cf import (  # noqa: E402
    EVADERE, token_class, view_logp, content_words,
)
sys.path.insert(0, os.path.join(HERE, 'OmniConfess', 'methods'))
# data/prompt helpers are backbone-agnostic (come from Qwen2.5 shared, fine)
from shared import (  # noqa: E402
    load_dataset, load_media_for_item, resolve_media_path,
    get_system_prompt, get_user_prompt,
)
# Qwen3 model + input builders (self-alias .thinker, Qwen3 media-token spelling)
from shared.qwen3_shim import (  # noqa: E402
    load_model, build_inputs, prepare_for_thinker,
    build_user_content_with_media_tokens, make_messages,
)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("MMHALU_MODEL_ROOT", "/ultrafast_share/tsq/MODELS")

COLORS = set("""red blue green yellow black white orange purple pink brown gray grey
gold silver beige teal maroon khaki tan turquoise violet indigo crimson""".split())
NUMW = set('zero one two three four five six seven eight nine ten several many'.split())

# Qwen3 base_model haloquest trajectory (token_ids + logp_list, 365/365)
Q3_BASE_SCORED = ('/ultrafast_share/tsq/qwen3omni_outputs/haloquest_qwen3omni/'
                  'haloquest_qwen3omni_full_single_scored.jsonl')


def is_attr(w):
    return w in COLORS or any(ch.isdigit() for ch in w) or w in NUMW


def build_view(processor, model, system_prompt, user_prompt, image, dataset_name):
    """Qwen3 two-view builder: Qwen3 media-token spelling + shim build_inputs."""
    user_content = build_user_content_with_media_tokens(None, None, image, user_prompt)
    messages = make_messages(system_prompt, user_content)
    raw = build_inputs(processor, model, messages, None, None, image, dataset_name)
    return prepare_for_thinker(raw, model)


def run(args):
    scored = args.scored or Q3_BASE_SCORED
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
    thinker = model.thinker   # shim self-alias -> model itself on Qwen3
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

        # teacher-forcing reproduction check against the recorded logp_list
        tf_check = None
        if rec.get('logp_list') and len(rec['logp_list']) == len(lp['full']):
            tf_check = float(np.max(np.abs(lp['full'] - np.array(rec['logp_list']))))

        rows.append({
            'sample_i': si, 'id': rec['id'], 'ok': True,
            'question': item['question'], 'gt': rec['gt'], 'pred': pred,
            'base_f1': rec['eval_metrics']['Token-F1'],
            'group_key': rec.get('group_key'),
            'partner': partner['id'],
            'tf_check_maxabsdiff': tf_check,
            # signals (answerability direction: high d_* = grounded = answerable)
            'd_content': d_content, 'd_content_ne': d_ne,
            'd_attr': d_attr, 'd_top2': d_top2,
            'lp_full_mean': lp_full_mean,
            'n_content': len(idx_content), 'n_attr': len(idx_attr),
            'howmany_digit': bool(howmany_digit),
            'pred_evade': pred_evade,
            # labels
            'neg_gt': neg_gt, 'needs_gate': needs_gate,
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
            'partners': args.partners, 'backbone': 'Qwen3-Omni-30B-A3B-Instruct',
            'scored_src': scored, 'generated': time.strftime('%Y-%m-%d %H:%M:%S'),
        }}, f, ensure_ascii=False)
    ok = [r for r in rows if r['ok']]
    tf = [r['tf_check_maxabsdiff'] for r in ok if r.get('tf_check_maxabsdiff') is not None]
    print(f'[done] {n_done} ok / {len(rows)} -> {args.out}')
    print(f'  neg_gt={sum(r["neg_gt"] for r in ok)} '
          f'needs_gate={sum(r["needs_gate"] for r in ok)} '
          f'pred_evade={sum(r["pred_evade"] for r in ok)}')
    if tf:
        print(f'  tf_check median maxabsdiff={np.median(tf):.5f} max={np.max(tf):.5f}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(
        HERE, 'outputs', 'answerability_qwen3',
        'haloquest_answerability_probe_qwen3.json'))
    ap.add_argument('--scored', default=None,
                    help='base greedy trajectory jsonl (default: Qwen3 base)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--partners', type=int, default=3,
                    help='number of swapped-image controls (Delta = full vs mean ctrl)')
    args = ap.parse_args()
    run(args)
