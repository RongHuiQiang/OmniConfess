# -*- coding: utf-8 -*-
"""bon_window_avhbench.py — avhbench 攻坚：时间窗候选 BoN + 分模态 VAS.

Coordinator brief (pre-registered). Structural new bet: every previous
candidate pool only varied SAMPLING TEMPERATURE (same evidence -> same wrong
perception in every candidate). Here we vary the EVIDENCE: candidates watch
different TIME WINDOWS of the video, so candidate variance tracks what the
clip actually shows in each window.

  Candidates per sample (all greedy, avhbench v1 prompt, audio always full):
    c0 full-video  linspace-8 (= base_model conditions)
    c1 window front 1/3, linspace-8 inside the window
    c2 window middle 1/3
    c3 window back 1/3

  Features per candidate (one teacher-forced full-view forward WITH
  attentions gives CE + VAS; two contrast forwards give claim-level Delta):
    Delta_v = lp(full) - lp(partner frames + own audio)     [swap video]
    Delta_a = lp(full) - lp(own frames + partner audio)     [swap audio]
    g_content : mean over content tokens of (dv+da)/2
    g_matched : attr-priority claim score (probe_claim_avhbench convention)
    CE        : top-5 truncated entropy, top-5 order statistic
    VAS_v/VAS_a: layer-14 attention share to the VIDEO / AUDIO token regions
    VAS_matched: audio-source words -> audio region, visual attr/content
                 words -> video region (modality-matched, literature-faithful)

  GATE 1 (dispersion): median cross-candidate range of g_content; DEAD if
  < 0.3 (ragtruth v1 pool reference: 1.04).
  GATE 2 (rerank): dev(50)-weighted combo score selection F1 on test(80) vs
  random window candidate vs greedy full-video; PASS iff combo beats BOTH.

Usage (GPUs 4-7, one process each; always source env_ffmpeg.sh first):
    source /home/tsq/rhq/omniconfess/env_ffmpeg.sh
    MMHALU_MODEL_ROOT=... CUDA_VISIBLE_DEVICES=4 python bon_window_avhbench.py \
        --mode probe --shard_idx 0 --n_shards 4
    python bon_window_avhbench.py --mode aggregate        # gates, no GPU
    ... --mode full --shard_idx k --n_shards 4            # if both gates pass
"""
import os, sys, json, argparse, random, string, time, glob
from collections import Counter

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'OmniConfess', 'methods'))
sys.path.insert(0, os.path.join(HERE, 'scripts'))
sys.path.insert(0, HERE)

from shared import (  # noqa: E402
    load_dataset, load_media_for_item, load_model, build_inputs,
    prepare_for_thinker, build_user_content_with_media_tokens, make_messages,
    get_system_prompt, get_user_prompt,
)
from shared.data_loader import load_video_frames, load_audio_data  # noqa: E402
from probe_claim_cf import claim_spans, token_class  # noqa: E402
from probe_cf_avhbench import AUDIO_WORDS, VISUAL_WORDS, norm_word  # noqa: E402
from probe_claim_avhbench import token_f1, stratified_sample  # noqa: E402
from bon_grounding import set_seed  # noqa: E402

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("MMHALU_MODEL_ROOT", "/ultrafast_share/tsq/MODELS")

TOPK, KSTAT = 5, 5
VAS_LAYER = 14
N_FRAMES = 8
WINDOWS = ('front', 'middle', 'back')
COLORS = set("""red blue green yellow black white orange purple pink brown gray
grey gold silver beige teal maroon khaki tan turquoise violet indigo crimson
""".split())
NUMW = set('two three four five six seven eight nine ten several many couple '
           'dozen multiple numerous dozens'.split())


def word_class(w):
    if w in AUDIO_WORDS:
        return 'audio'
    if w in VISUAL_WORDS or w in COLORS or w in NUMW or any(c.isdigit() for c in w):
        return 'visual'
    return None


# ---------------------------------------------------------------------------
# Windowed frame sampling (data_loader.load_video_frames stays untouched).
# ---------------------------------------------------------------------------
def load_video_frames_window(video_path, window, num_frames=N_FRAMES):
    """linspace-8 inside the front/middle/back third of the video."""
    import cv2
    if not video_path or not os.path.exists(video_path):
        return None
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None
    third = max(1, total // 3)
    lo = {'front': 0, 'middle': third, 'back': 2 * third}[window]
    hi = min(total, lo + third + (1 if window == 'back' else 0))
    if hi - lo < num_frames:
        lo, hi = 0, total
    indices = np.linspace(lo, hi - 1, num_frames).astype(int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (224, 224))
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.array(frames) if frames else None


_MEDIA_KEYS = (
    'pixel_values_images', 'pixel_values', 'image_grid_thw',
    'pixel_values_videos', 'video_grid_thw', 'video_second_per_grid',
    'input_features', 'feature_attention_mask',
)


def build_view(processor, model, system_prompt, user_prompt, pixel, audio):
    prefix = ''
    if pixel is not None:
        prefix += "<|vision_bos|><|VIDEO|><|vision_eos|>\n"
    if audio is not None:
        prefix += "<|audio_bos|><|AUDIO|><|audio_eos|>\n"
    messages = make_messages(system_prompt, prefix + user_prompt)
    raw = build_inputs(processor, model, messages, pixel, audio, None, 'avhbench')
    return prepare_for_thinker(raw, model)


def retry(fn, n=3, wait=5, tag=''):
    for i in range(n):
        try:
            return fn()
        except Exception as e:
            print(f'  [retry {i+1}/{n}] {tag}: {type(e).__name__}: {e}', flush=True)
            time.sleep(wait * (i + 1))
    raise RuntimeError(f'retries exhausted: {tag}')


@torch.inference_mode()
def tf_forward(thinker, clean, gen_ids, want_feats, masks=None, content_pos=None):
    """Teacher-forced forward. want_feats=True -> logits+attentions kept for
    CE/VAS (masks: dict region_name -> bool[input_len])."""
    input_len = clean['input_ids'].shape[1]
    full_ids = torch.cat([clean['input_ids'], gen_ids.unsqueeze(0)], dim=1)
    kw = dict(input_ids=full_ids, attention_mask=torch.ones_like(full_ids),
              use_cache=False, output_attentions=want_feats)
    for k in _MEDIA_KEYS:
        if k in clean:
            kw[k] = clean[k]
    out = thinker(**kw)
    logits = out.logits[0].float()
    lsm = torch.log_softmax(logits, dim=-1)
    pos = torch.arange(input_len - 1, full_ids.shape[1] - 1, device=logits.device)
    lp = lsm[pos, gen_ids.to(logits.device)].cpu().numpy()
    res = {'lp': lp}
    if want_feats:
        top = torch.topk(logits[pos], TOPK, dim=-1).values
        pk = torch.softmax(top, dim=-1)
        H = -(pk * torch.log(pk + 1e-9)).sum(-1).cpu().numpy()
        res['CE'] = float(np.mean(np.sort(H)[::-1][:KSTAT]))
        att = out.attentions[VAS_LAYER][0].float().mean(0)
        rows = att[pos]
        denom = rows.sum(-1).clamp(min=1e-9)
        frac = {name: (rows[:, :input_len][:, m].sum(-1) / denom).cpu().numpy()
                for name, m in masks.items()}
        res['frac'] = frac
        res['H'] = H
    del out, logits, lsm
    return res


def region_masks(clean, tokenizer):
    ids = clean['input_ids'][0]
    def m(tokname):
        tid = tokenizer.convert_tokens_to_ids(tokname)
        return (ids == tid)
    return {'video': m('<|VIDEO|>'), 'audio': m('<|AUDIO|>')}


# ---------------------------------------------------------------------------
# Shard runner (probe & full share it; only the sample set differs).
# ---------------------------------------------------------------------------
def run_shard(args):
    scored_path = os.path.join(HERE, 'outputs', 'avhbench_base_model',
                               'avhbench_base_model_full_single_scored.jsonl')
    recs = [json.loads(l) for l in open(scored_path)]
    recs = [r for r in recs if r.get('token_ids')]
    for r in recs:
        r['_f1'] = token_f1(r['pred'], r['gt'])

    if args.mode == 'probe':
        sample = stratified_sample(recs, args.n_samples, args.seed)
    else:
        sample = recs
    sample = [s for i, s in enumerate(sample)
              if i % args.n_shards == args.shard_idx]
    print(f'[{args.mode} shard {args.shard_idx}/{args.n_shards}] {len(sample)} samples',
          flush=True)

    data_path = os.path.join(HERE, 'OmniHalluBench', 'avhbench', 'avhbench.json')
    data_root = os.path.dirname(data_path)
    data = load_dataset(data_path, data_root)
    by_qg = {(d['question'], d['gt']): d for d in data}
    all_items = [d for d in data if d.get('video_path')]

    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
    from shared.paths import resolve_model
    model_path = resolve_model("Qwen/Qwen2.5-Omni-7B")
    processor = Qwen2_5OmniProcessor.from_pretrained(model_path, trust_remote_code=True)
    processor.tokenizer.padding_side = 'left'
    if not processor.tokenizer.pad_token:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map='auto',
        trust_remote_code=True, attn_implementation='eager').eval()
    model.thinker.config.vocab_size = 152064
    tokenizer = processor.tokenizer
    thinker = model.thinker
    system_prompt = get_system_prompt('avhbench', 'v1')

    outdir = os.path.join(HERE, 'outputs', 'avhbench_bon_window')
    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(
        outdir, f'avh_{args.mode}_shard{args.shard_idx}of{args.n_shards}.jsonl')
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            try:
                done.add(json.loads(line)['id'])
            except Exception:
                pass
    fout = open(out_path, 'a')

    rng = random.Random(args.seed + 7 + args.shard_idx)
    t0 = time.time()
    for i, rec in enumerate(sample):
        if rec['id'] in done:
            continue
        item = by_qg.get((rec['question'], rec['gt']))
        if item is None:
            continue
        try:
            own_v_full, own_a, _ = retry(
                lambda: load_media_for_item(item, data_path, data_root),
                tag='own_media')
            if own_v_full is None or own_a is None:
                print(f'  [skip] media missing id={rec["id"]}', flush=True)
                continue
            from shared.data_loader import resolve_media_path
            vpath = resolve_media_path(item['video_path'], data_path, data_root)
            wins = {w: retry(lambda w=w: load_video_frames_window(vpath, w),
                             tag=f'window_{w}') for w in WINDOWS}
            if any(v is None for v in wins.values()):
                print(f'  [skip] window load failed id={rec["id"]}', flush=True)
                continue

            # partner for the contrast views (avhbench same-domain pool)
            partner = None
            for _ in range(6):
                cand = rng.choice(all_items)
                if cand['gt'] == item['gt']:
                    continue
                pv, pa, _ = load_media_for_item(cand, data_path, data_root)
                if pv is not None and pa is not None:
                    partner = (cand, pv, pa)
                    break
            if partner is None:
                print(f'  [skip] no partner id={rec["id"]}', flush=True)
                continue

            user_prompt = get_user_prompt(item['question'], 'avhbench', 'v1')
            cands = []
            gen_specs = [('full', own_v_full)] + [(w, wins[w]) for w in WINDOWS]
            for name, frames in gen_specs:
                clean = retry(lambda f=frames: build_view(
                    processor, model, system_prompt, user_prompt, f, own_a),
                    tag='build_view')
                input_len = clean['input_ids'].shape[1]
                with torch.inference_mode():
                    out = retry(lambda c=clean: thinker.generate(
                        **c, max_new_tokens=args.max_new_tokens, do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                        stop_strings=['<end_of_reasoning>'], tokenizer=tokenizer),
                        tag='generate')
                gen = out[0][input_len:]
                gen = gen[gen != tokenizer.pad_token_id]
                gen_ids_list = gen.tolist()
                text = tokenizer.decode(gen, skip_special_tokens=True).strip()
                from shared import parse_pred
                pred = parse_pred(text, 'avhbench')

                gen_t = torch.tensor(gen_ids_list, dtype=torch.long, device=model.device)
                masks = region_masks(clean, tokenizer)
                if masks['video'].sum() == 0 or masks['audio'].sum() == 0:
                    raise RuntimeError('media region mask empty')
                toks = [tokenizer.decode([t], skip_special_tokens=True)
                        for t in gen_ids_list]
                cls = [token_class(t, toks[k - 1] if k else None)
                       for k, t in enumerate(toks)]
                content_pos = torch.tensor([c in ('content', 'num') for c in cls],
                                           device=model.device)
                # 1) full view WITH feats
                r_full = retry(lambda: tf_forward(
                    thinker, clean, gen_t, True, masks, content_pos), tag='tf_full')
                # 2/3) contrast views (no feats)
                clean_sv = retry(lambda: build_view(
                    processor, model, system_prompt, user_prompt, partner[1], own_a),
                    tag='build_sv')
                r_sv = retry(lambda: tf_forward(thinker, clean_sv, gen_t, False),
                             tag='tf_sv')
                clean_sa = retry(lambda: build_view(
                    processor, model, system_prompt, user_prompt, frames, partner[2]),
                    tag='build_sa')
                r_sa = retry(lambda: tf_forward(thinker, clean_sa, gen_t, False),
                             tag='tf_sa')
                dv = r_full['lp'] - r_sv['lp']
                da = r_full['lp'] - r_sa['lp']
                dmix = (dv + da) / 2.0
                cw = [k for k in range(len(cls)) if cls[k] in ('content', 'num')]
                spans = claim_spans(toks, cls)
                # attr-priority claim score (probe_claim_avhbench convention)
                claim_scores = []
                for lo, hi in spans:
                    attr_d, other_d = [], []
                    for k in range(lo, hi):
                        if cls[k] not in ('content', 'num'):
                            continue
                        w = norm_word(toks[k])
                        wc = word_class(w)
                        if wc == 'audio':
                            attr_d.append(da[k])
                        elif wc == 'visual':
                            attr_d.append(dv[k])
                        else:
                            other_d.append(dmix[k])
                    claim_scores.append(float(np.mean(attr_d if attr_d else other_d))
                                        if (attr_d or other_d) else 0.0)
                # modality-matched VAS over generated positions
                fv, fa = r_full['frac']['video'], r_full['frac']['audio']
                matched_v, matched_a = [], []
                for k in range(len(cls)):
                    w = norm_word(toks[k]) if cls[k] in ('content', 'num') else ''
                    wc = word_class(w) if w else None
                    if wc == 'audio':
                        matched_a.append(fa[k])
                    elif wc == 'visual':
                        matched_v.append(fv[k])
                cands.append({
                    'cand': name, 'pred': pred, 'f1': token_f1(pred, rec['gt']),
                    'n_tokens': len(gen_ids_list), 'n_claims': len(spans),
                    'g_content': float(np.mean(dmix[cw])) if cw else None,
                    'g_matched': float(np.mean(claim_scores)) if claim_scores else None,
                    'CE': r_full['CE'],
                    'vas_v_all': float(np.mean(fv)), 'vas_a_all': float(np.mean(fa)),
                    'vas_v_matched': float(np.mean(matched_v)) if matched_v else None,
                    'vas_a_matched': float(np.mean(matched_a)) if matched_a else None,
                })
                torch.cuda.empty_cache()

            fout.write(json.dumps({'id': rec['id'], 'question': rec['question'],
                                   'gt': rec['gt'], 'partner': partner[0]['id'],
                                   'candidates': cands}, ensure_ascii=False) + '\n')
            fout.flush()
            if (i + 1) % 10 == 0:
                print(f'  [{i+1}/{len(sample)}] {time.time()-t0:.0f}s', flush=True)
        except Exception as e:
            print(f'  [ERROR] id={rec["id"]}: {type(e).__name__}: {e}', flush=True)
            torch.cuda.empty_cache()
    fout.close()
    print(f'[done shard {args.shard_idx}] {time.time()-t0:.0f}s -> {out_path}', flush=True)


# ---------------------------------------------------------------------------
# Aggregate: gates (probe) / final scoring (full). Zero GPU.
# ---------------------------------------------------------------------------
def load_rows(mode):
    rows = []
    for shard in range(4):
        p = os.path.join(HERE, 'outputs', 'avhbench_bon_window',
                         f'avh_{mode}_shard{shard}of4.jsonl')
        if os.path.exists(p):
            rows += [json.loads(l) for l in open(p)]
    return rows


def sel_f1(rows, w, zp):
    f1s = []
    for r in rows:
        best, bv = None, -1e18
        for c in r['candidates']:
            v = 0.0
            for wi, fi in zip(w, range(3)):
                if wi == 0:
                    continue
                x = c[_FEATS[fi]]
                if x is None:
                    x = zp[fi][0]
                v += wi * (x - zp[fi][0]) / zp[fi][1]
            if v > bv:
                bv, best = v, c
        f1s.append(best['f1'])
    return 100 * float(np.mean(f1s))


_FEATS = ('g_matched', 'CE', 'vas_m')


def _vas_m(c):
    vals = []
    if c.get('vas_v_matched') is not None:
        vals.append(c['vas_v_matched'])
    if c.get('vas_a_matched') is not None:
        vals.append(c['vas_a_matched'])
    if vals:
        return float(np.mean(vals))
    return c.get('vas_v_all')


def aggregate(args):
    rows = load_rows('probe' if args.mode == 'aggregate' else 'full')
    for r in rows:
        for c in r['candidates']:
            c['vas_m'] = _vas_m(c)
    print(f'[aggregate] {len(rows)} samples, '
          f'{sum(len(r["candidates"]) for r in rows)} candidates')

    # ---- GATE 1: dispersion --------------------------------------------------
    for key in ('g_content', 'g_matched'):
        rngs = [max(c[key] for c in r['candidates']) -
                min(c[key] for c in r['candidates'])
                for r in rows if all(c[key] is not None for c in r['candidates'])]
        print(f'[gate1] {key}: median range {np.median(rngs):.3f} '
              f'(ragtruth v1 pool ref 1.04; dead if < 0.3)')
    rngs = [max(c['g_content'] for c in r['candidates']) -
            min(c['g_content'] for c in r['candidates']) for r in rows]
    gate1 = float(np.median(rngs)) >= 0.3
    print(f'[gate1] PASS={gate1}')

    # arms that need no weights
    greedy = [r['candidates'][0]['f1'] for r in rows]
    rand = [float(np.mean([c['f1'] for c in r['candidates'][1:]])) for r in rows]
    oracle = [max(c['f1'] for c in r['candidates']) for r in rows]
    print(f'[arms] greedy full-video {100*np.mean(greedy):.2f} | '
          f'random-window {100*np.mean(rand):.2f} | oracle {100*np.mean(oracle):.2f}')

    # ---- GATE 2: dev-weighted combo on test ----------------------------------
    rng = random.Random(42)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    dev = [rows[i] for i in idx[:50]]
    test = [rows[i] for i in idx[50:]]

    def zfit(pool, fi):
        vals = [c[_FEATS[fi]] for r in pool for c in r['candidates']]
        return (float(np.mean(vals)), float(np.std(vals)) + 1e-9)

    zp_dev = [zfit(dev, i) for i in range(3)]
    zp_all = [zfit(rows, i) for i in range(3)]
    best = ((0, 0, 0), -1)
    for be in (0.0, 0.25, 0.5, 1.0, 2.0):
        for ga in (0.0, 0.25, 0.5, 1.0, 2.0, -0.5, -1.0, -2.0):
            w = (1.0, -be, ga)
            fd = sel_f1(dev, w, zp_dev)
            if fd > best[1]:
                best = (w, fd)
    w, fd = best
    print(f'[gate2] dev best {fd:.2f} at w={w}')
    combo_test = sel_f1(test, w, zp_dev)
    delta_test = sel_f1(test, (1, 0, 0), zp_dev)
    greedy_test = 100 * float(np.mean([r['candidates'][0]['f1'] for r in test]))
    rand_test = 100 * float(np.mean([np.mean([c['f1'] for c in r['candidates'][1:]])
                                     for r in test]))
    print(f'[gate2] TEST combo {combo_test:.2f} | pure-delta {delta_test:.2f} | '
          f'greedy-full {greedy_test:.2f} | random-window {rand_test:.2f}')
    gate2 = combo_test > max(greedy_test, rand_test)
    print(f'[gate2] PASS={gate2}')

    if args.mode == 'full':
        combo_full = sel_f1(rows, w, zp_all)
        print(f'[full] combo full-set {combo_full:.2f} '
              f'(judgement: > 19.42 -> {combo_full > 19.42})')
    out = {'n': len(rows), 'gate1': gate1, 'gate2': gate2, 'w': list(w),
           'test': {'combo': round(combo_test, 2), 'delta': round(delta_test, 2),
                    'greedy': round(greedy_test, 2), 'random': round(rand_test, 2)}}
    json.dump(out, open(os.path.join(HERE, 'outputs', 'avhbench_bon_window',
                                     f'{args.mode}_gates.json'), 'w'), indent=1)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True, choices=['probe', 'aggregate', 'full'])
    ap.add_argument('--n_samples', type=int, default=130)
    ap.add_argument('--shard_idx', type=int, default=0)
    ap.add_argument('--n_shards', type=int, default=4)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--max_new_tokens', type=int, default=512)
    args = ap.parse_args()
    set_seed(args.seed)
    if args.mode == 'aggregate':
        aggregate(args)
    else:
        run_shard(args)
