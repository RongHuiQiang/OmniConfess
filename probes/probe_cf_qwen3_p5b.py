#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""probe_cf_qwen3_p5b.py -- P5b blindspot-axis cross-backbone cells:
Qwen3-Omni-30B-A3B x {avhbench, haloquest, ragtruth, pubmedqa} sample-level
counterfactual fragility, to be correlated with Qwen3-base norm_logp in
outputs/problem_structure_fig/unified_cf_robustness_qwen3_full.md.

Condition definitions are COPIES OF THE QWEN2.5 CALIBERS (same contrast views,
same teacher-forcing protocol); only the model layer is swapped to Qwen3-Omni
via shared/qwen3_omni_utils.py (thinker-only load, model IS the thinker):
  avhbench  full / novideo / noaudio  per-token delta_video, delta_audio
            (probe_cf_avhbench.py big-N caliber; frag = sum_t sqrt(dv^2+da^2))
  haloquest full image / swapped image  per-token delta_tok
            (probe_claim_cf.py haloquest caliber; frag = sum_t |delta_tok|)
  ragtruth  own passage / borrowed passage (probe_claim_cf.py ragtruth caliber)
  pubmedqa  own passage / partner passage  (probe_cot_cf_pubmedqa.py caliber)

Base trajectories + confidence (norm_logp) come from the Qwen3 base runs:
  avhbench  outputs/avhbench_qwen3omni_contractv4/..._single_scored.jsonl (pv v4)
  haloquest outputs/haloquest_qwen3omni_contractv2/..._single_scored.jsonl (pv v2)
  ragtruth  outputs/ragtruth_qwen3omni_contractv4/..._single_scored.jsonl (pv v4)
  pubmedqa  outputs/pubmedqa_qwen3omni_promptv5/..._single_scored.jsonl   (pv v5)
Join: avhbench scored id == row index into avhbench.json (verified 645/645
q+gt agree); the other three join on (question, gt) (verified unique, full
coverage -- `id` is provenance, not a join key, per data_loader._alias_legacy).
Every sample carries a tf_repro check (full-view teacher-forced logp vs the
base-recorded logp_list) that validates prompt-version + media reconstruction.

Usage (exclusive free card):
  CUDA_VISIBLE_DEVICES=7 python probe_cf_qwen3_p5b.py --dataset pubmedqa --limit 3
"""
import os, sys, json, argparse, random, string, time
from collections import Counter

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'OmniConfess', 'methods'))
os.chdir(HERE)

from shared import (  # noqa: E402
    load_dataset, load_media_for_item,
    get_system_prompt, get_user_prompt,
)
from shared.qwen3_omni_utils import (  # noqa: E402
    load_model, build_inputs, prepare_for_thinker,
    build_user_content_with_media_tokens, make_messages,
)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("MMHALU_MODEL_ROOT", "/ultrafast_share/tsq/MODELS")

CFG = {
    'avhbench': dict(
        scored='outputs/avhbench_qwen3omni_contractv4/avhbench_qwen3omni_contractv4_full_single_scored.jsonl',
        pv='v4', join='index'),
    'haloquest': dict(
        scored='outputs/haloquest_qwen3omni_contractv2/haloquest_qwen3omni_contractv2_full_single_scored.jsonl',
        pv='v2', join='qgt'),
    'ragtruth': dict(
        scored='outputs/ragtruth_qwen3omni_contractv4/ragtruth_qwen3omni_contractv4_full_single_scored.jsonl',
        pv='v4', join='qgt'),
    'pubmedqa': dict(
        scored='outputs/pubmedqa_qwen3omni_promptv5/pubmedqa_qwen3omni_promptv5_single_scored.jsonl',
        pv='v5', join='qgt'),
}

STOPWORDS = set([
    "a", "an", "the", "and", "or", "but", "if", "because", "as", "what",
    "which", "this", "that", "these", "those", "then", "just", "so", "than",
    "such", "both", "through", "about", "for", "is", "are", "was", "were",
    "of", "while", "during", "to", "in", "on", "at", "by", "with", "it",
    "there",
])


def _f1_tokens(s):
    s = str(s).lower().translate(str.maketrans("", "", string.punctuation))
    return [w for w in s.split() if w not in STOPWORDS and w.strip()]


def token_f1(pred, gt):
    p_t, g_t = _f1_tokens(pred), _f1_tokens(gt)
    if not p_t or not g_t:
        return 0.0
    common = Counter(p_t) & Counter(g_t)
    hit = sum(common.values())
    p = hit / len(p_t)
    r = hit / len(g_t)
    return (2 * p * r) / (p + r) if (p + r) > 0 else 0.0


def stratified_sample(recs, n, seed=42):
    rng = random.Random(seed)
    ordered = sorted(recs, key=lambda r: r[0]['_f1'])
    k = max(1, len(ordered) // 3)
    tiers = [ordered[:k], ordered[k:2 * k], ordered[2 * k:]]
    for t in tiers:
        rng.shuffle(t)
    picked = []
    while len(picked) < n:
        adv = False
        for t in tiers:
            if t and len(picked) < n:
                picked.append(t.pop())
                adv = True
        if not adv:
            break
    return picked


_THINKER_KEYS = (
    'pixel_values', 'image_grid_thw',
    'pixel_values_videos', 'video_grid_thw', 'video_second_per_grid',
    'input_features', 'feature_attention_mask',
)


@torch.inference_mode()
def view_logp(model, clean, gen_ids):
    input_len = clean['input_ids'].shape[1]
    full_ids = torch.cat([clean['input_ids'], gen_ids.unsqueeze(0)], dim=1)
    am = torch.ones_like(full_ids)
    kw = dict(input_ids=full_ids, attention_mask=am, use_cache=False)
    for k in _THINKER_KEYS:
        if k in clean:
            kw[k] = clean[k]
    out = model(**kw)
    logits = out.logits[0].float()
    lsm = torch.log_softmax(logits, dim=-1)
    pos = torch.arange(input_len - 1, full_ids.shape[1] - 1, device=logits.device)
    lp = lsm[pos, gen_ids.to(logits.device)]
    del out, logits, lsm
    return lp.cpu().numpy()


def build_view(processor, model, system_prompt, user_prompt, ds,
               pixel_values=None, audio_values=None, image_values=None):
    uc = build_user_content_with_media_tokens(
        pixel_values, audio_values, image_values, user_prompt)
    messages = make_messages(system_prompt, uc)
    raw = build_inputs(processor, model, messages,
                       pixel_values, audio_values, image_values, ds)
    return prepare_for_thinker(raw, model)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=list(CFG))
    ap.add_argument('--n_samples', type=int, default=0,
                    help='0 = full; avhbench default cap applied separately')
    ap.add_argument('--avh_n', type=int, default=250)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--start', type=int, default=0, help='skip first N of sample (resume)')
    ap.add_argument('--out', default='')
    ap.add_argument('--pv', default='', help='override prompt version (diagnostics)')
    args = ap.parse_args()

    ds = args.dataset
    cfg = CFG[ds]
    pv = args.pv or cfg['pv']
    out_path = args.out or f'outputs/cf_probe/p5b_qwen3/{ds}_qwen3_cf.jsonl'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    data_path = os.path.join(HERE, 'OmniHalluBench', ds, ds + '.json')
    data_root = os.path.dirname(data_path)
    data = load_dataset(data_path, data_root)

    rows = [json.loads(l) for l in open(cfg['scored'])]
    rows = [r for r in rows if r.get('token_ids') and r.get('norm_logp') is not None]
    for r in rows:
        r['_f1'] = token_f1(r['pred'], r['gt'])

    if cfg['join'] == 'index':
        by_key = {str(i): it for i, it in enumerate(data)}
    else:
        by_key = {((it['question'] or '').strip(), str(it.get('gt', '')).strip()): it
                  for it in data}
    pairs = []
    for r in rows:
        key = (str(r['id']) if cfg['join'] == 'index'
               else ((r['question'] or '').strip(), str(r.get('gt', '')).strip()))
        it = by_key.get(key)
        if it is None:
            print(f'[join-miss] id={r["id"]}', flush=True)
            continue
        pairs.append((r, it))
    print(f'[join] {len(pairs)}/{len(rows)} joined', flush=True)

    # ---- sampling -----------------------------------------------------------
    want = args.n_samples or (args.avh_n if ds == 'avhbench' else len(pairs))
    if want < len(pairs):
        sample = stratified_sample(pairs, want, args.seed)
    else:
        sample = list(pairs)
    rng = random.Random(args.seed)
    rng.shuffle(sample)
    if args.start:
        sample = sample[args.start:]
    if args.limit:
        sample = sample[:args.limit]
    print(f'[probe] {ds}: {len(sample)} samples (pv={pv})', flush=True)

    pool_passages = [it['passage'] for _, it in pairs if it.get('passage')]
    pool_images = [it for _, it in pairs
                   if it.get('image_path') and ds == 'haloquest']

    print('[model] loading Qwen3-Omni thinker (~62GB, exclusive card)...', flush=True)
    processor, model = load_model()
    tokenizer = processor.tokenizer
    system_prompt = get_system_prompt(ds, pv)

    # resume support on the output file (skip already-written ids)
    done = set()
    if os.path.exists(out_path):
        for l in open(out_path):
            try:
                done.add(json.loads(l)['uid_key'])
            except Exception:
                pass
        print(f'[resume] {len(done)} already done', flush=True)
    mode = 'a' if done else 'w'
    fout = open(out_path, mode)

    t0 = time.time()
    n_done = 0
    for i, (rec, item) in enumerate(sample):
        uid_key = f"{rec['id']}|{str(rec['gt'])[:50]}"
        if uid_key in done:
            continue
        try:
            gen_ids = torch.tensor(rec['token_ids'], dtype=torch.long,
                                   device=model.device)
            q = item['question']
            rec_out = dict(
                dataset=ds, uid_key=uid_key, id=rec['id'],
                question=q, gt=rec['gt'], conf=rec['norm_logp'],
                f1=rec['_f1'], n_tok=len(rec['token_ids']))

            if ds in ('ragtruth', 'pubmedqa'):
                own = item.get('passage') or ''
                partner = None
                while partner is None:
                    cand = rng.choice(pool_passages)
                    if cand != own:
                        partner = cand
                views = {}
                for name, psg in (('full', own), ('ctrl', partner)):
                    up = get_user_prompt(q, ds, pv, passage=psg)
                    views[name] = build_view(processor, model, system_prompt,
                                             up, ds)
                lp = {n: view_logp(model, v, gen_ids) for n, v in views.items()}
                delta = lp['full'] - lp['ctrl']
                rec_out['delta_tok'] = [round(float(x), 4) for x in delta]
            elif ds == 'haloquest':
                img = load_media_for_item(item, data_path, data_root)[2]
                partner = None
                while partner is None:
                    cand = rng.choice(pool_images)
                    if cand['id'] != item['id']:
                        partner = cand
                pimg = load_media_for_item(partner, data_path, data_root)[2]
                up = get_user_prompt(q, ds, pv, passage=item.get('passage'))
                views = {}
                for name, im in (('full', img), ('ctrl', pimg)):
                    views[name] = build_view(processor, model, system_prompt,
                                             up, ds, image_values=im)
                lp = {n: view_logp(model, v, gen_ids) for n, v in views.items()}
                delta = lp['full'] - lp['ctrl']
                rec_out['delta_tok'] = [round(float(x), 4) for x in delta]
            else:  # avhbench
                pixel_values, audio_values, _ = load_media_for_item(
                    item, data_path, data_root)
                if pixel_values is None or audio_values is None:
                    print(f"  [skip] media missing id={rec['id']}", flush=True)
                    continue
                up = get_user_prompt(q, ds, pv, passage=item.get('passage'))
                views = {}
                for name, pv_flag, av_flag in (('full', True, True),
                                               ('novideo', False, True),
                                               ('noaudio', True, False)):
                    views[name] = build_view(
                        processor, model, system_prompt, up, ds,
                        pixel_values=pixel_values if pv_flag else None,
                        audio_values=audio_values if av_flag else None)
                lp = {n: view_logp(model, v, gen_ids) for n, v in views.items()}
                rec_out['delta_video'] = [round(float(x), 4)
                                          for x in lp['full'] - lp['novideo']]
                rec_out['delta_audio'] = [round(float(x), 4)
                                          for x in lp['full'] - lp['noaudio']]

            base_lp = np.array(rec['logp_list'], dtype=np.float64)
            tf = (float(np.max(np.abs(lp['full'] - base_lp)))
                  if len(base_lp) == len(lp['full']) else float('nan'))
            rec_out['tf_repro'] = round(tf, 4)
            fout.write(json.dumps(rec_out, ensure_ascii=False) + '\n')
            fout.flush()
            n_done += 1
            if n_done % 10 == 0:
                print(f'  [{n_done}/{len(sample)}] {time.time()-t0:.0f}s '
                      f'tf={tf:.4f}', flush=True)
            del views, lp
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f'  [ERROR] id={rec["id"]}: {e}', flush=True)
        torch.cuda.empty_cache()

    fout.close()
    print(f'[done] {n_done}/{len(sample)} in {time.time()-t0:.0f}s -> {out_path}',
          flush=True)


if __name__ == '__main__':
    main()
