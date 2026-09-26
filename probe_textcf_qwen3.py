#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""probe_textcf_qwen3.py -- Qwen3-Omni modality-axis TEXT-channel probe
(question-text neutralization, placeholder scheme), filling the Qwen3 column
of the modality axis (Qwen2.5 already done; see
outputs/problem_structure_fig/unified_cf_robustness_textcf.md).

Three datasets, one script, resumable, exclusive free card:

  haloquest (pv v2, join qgt, scored=contractv2):
      full : own image + own question
      noq  : own image + NEUTRAL question "(no question provided)"
      delta_text_tok = lp_full - lp_noq  (per output token, teacher-forced on
      the stored base greedy token_ids -- identical positions as the image
      channel in outputs/cf_probe/p5b_qwen3/haloquest_qwen3_cf.jsonl, whose
      delta_tok is copied through as delta_img_tok via uid_key join).

  phd (pv v1, NO stored Qwen3 base run -> generate here):
      native observe-then-judge CoT regenerated greedy (same protocol as
      probe_cot_cf_phd_text.py on Qwen2.5), keep >= 6 tokens; three views
      full / img-swap (partner image) / noq  -> delta_img_tok + delta_text_tok.

  avhbench (pv v4, join index, scored=contractv4):
      rows restricted to those already probed in p5b_qwen3/avhbench (uid_key
      join) so (v,t)/(a,t) pair row-by-row; full/noq views with the fixed
      instruction replaced by "Describe the content in a single sentence."
      delta_video/delta_audio copied through from the p5b row.

Usage (exclusive free card):
  CUDA_VISIBLE_DEVICES=5 /ultrafast_share/tsq/venvs/omniconfess/bin/python \
      probe_textcf_qwen3.py --dataset haloquest
"""
import os, sys, json, argparse, random, string, time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, 'OmniConfess', 'methods'))

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("MMHALU_MODEL_ROOT", "/ultrafast_share/tsq/MODELS")

from shared import (  # noqa: E402
    load_dataset, load_media_for_item,
    get_system_prompt, get_user_prompt, resolve_media_path,
)
from shared.qwen3_omni_utils import (  # noqa: E402
    load_model, build_inputs, prepare_for_thinker,
    build_user_content_with_media_tokens, make_messages,
)
from shared.data_loader import load_image  # noqa: E402
from probe_claim_cf import token_class  # noqa: E402

NEUTRAL_Q = {
    'haloquest': "(no question provided)",
    'phd': "(no question provided)",
    'avhbench': "Describe the content in a single sentence.",
}

CFG = {
    'haloquest': dict(
        scored='outputs/haloquest_qwen3omni_contractv2/haloquest_qwen3omni_contractv2_full_single_scored.jsonl',
        pv='v2', join='qgt'),
    'avhbench': dict(
        scored='outputs/avhbench_qwen3omni_contractv4/avhbench_qwen3omni_contractv4_full_single_scored.jsonl',
        pv='v4', join='index'),
}
P5B = 'outputs/cf_probe/p5b_qwen3'
PHD_DATA = os.path.join(HERE, 'OmniHalluBench/phd/phd.json')

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


def tokens_classes(tokenizer, gen):
    toks = [tokenizer.decode([t], skip_special_tokens=True) for t in gen]
    cls = [token_class(t, toks[i - 1] if i else None) for i, t in enumerate(toks)]
    return toks, cls


def run_halo_or_avh(args, ds):
    cfg = CFG[ds]
    pv = cfg['pv']
    out_path = args.out or f'outputs/cf_probe/qwen3_textcf/{ds}_textcf_tokens.jsonl'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    data_path = os.path.join(HERE, 'OmniHalluBench', ds, ds + '.json')
    data_root = os.path.dirname(data_path)
    data = load_dataset(data_path, data_root)

    rows = [json.loads(l) for l in open(cfg['scored'])]
    rows = [r for r in rows if r.get('token_ids') and r.get('norm_logp') is not None]

    # p5b image/audio-channel rows (join source for the paired channel + sample
    # restriction on avhbench)
    p5b = {}
    p5b_path = os.path.join(P5B, f'{ds}_qwen3_cf.jsonl')
    for l in open(p5b_path):
        r = json.loads(l)
        p5b[r['uid_key']] = r

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
            continue
        uid_key = f"{r['id']}|{str(r['gt'])[:50]}"
        if ds == 'avhbench' and uid_key not in p5b:
            continue  # restrict to the already-probed avh sample
        pairs.append((r, it, uid_key))
    print(f'[join] {ds}: {len(pairs)} samples ({len(p5b)} p5b rows)', flush=True)
    if args.limit:
        pairs = pairs[:args.limit]

    done = set()
    if os.path.exists(out_path):
        for l in open(out_path):
            try:
                done.add(json.loads(l)['uid_key'])
            except Exception:
                pass
        print(f'[resume] {len(done)} already done', flush=True)
    fout = open(out_path, 'a' if done else 'w')

    print('[model] loading Qwen3-Omni thinker (~62GB, exclusive card)...', flush=True)
    processor, model = load_model()
    tokenizer = processor.tokenizer
    system_prompt = get_system_prompt(ds, pv)

    t0, n_done = time.time(), 0
    for i, (rec, item, uid_key) in enumerate(pairs):
        if uid_key in done:
            continue
        try:
            gen = rec['token_ids']
            gen_ids = torch.tensor(gen, dtype=torch.long, device=model.device)
            q = item['question']
            p5b_row = p5b.get(uid_key)
            if p5b_row is not None and len(p5b_row['delta_tok' if ds == 'haloquest'
                                                   else 'delta_video']) != len(gen):
                print(f'  [lenmismatch] {uid_key}: p5b != scored token_ids', flush=True)
                p5b_row = None
            toks, cls = tokens_classes(tokenizer, gen)
            rec_out = dict(
                dataset=ds, uid_key=uid_key, id=rec['id'],
                question=q, gt=rec['gt'], n_tok=len(gen),
                tokens=toks, classes=cls)

            if ds == 'haloquest':
                img = load_media_for_item(item, data_path, data_root)[2]
                up_full = get_user_prompt(q, ds, pv, passage=item.get('passage'))
                up_noq = get_user_prompt(NEUTRAL_Q[ds], ds, pv,
                                         passage=item.get('passage'))
                vf = build_view(processor, model, system_prompt, up_full, ds,
                                image_values=img)
                vn = build_view(processor, model, system_prompt, up_noq, ds,
                                image_values=img)
                lp_f = view_logp(model, vf, gen_ids)
                lp_n = view_logp(model, vn, gen_ids)
                rec_out['delta_text_tok'] = [round(float(x), 4)
                                             for x in lp_f - lp_n]
                if p5b_row is not None:
                    rec_out['delta_img_tok'] = p5b_row['delta_tok']
                del vf, vn
            else:  # avhbench
                pixel_values, audio_values, _ = load_media_for_item(
                    item, data_path, data_root)
                if pixel_values is None or audio_values is None:
                    print(f"  [skip] media missing id={rec['id']}", flush=True)
                    continue
                up_full = get_user_prompt(q, ds, pv, passage=item.get('passage'))
                up_noq = get_user_prompt(NEUTRAL_Q[ds], ds, pv,
                                         passage=item.get('passage'))
                vf = build_view(processor, model, system_prompt, up_full, ds,
                                pixel_values=pixel_values,
                                audio_values=audio_values)
                vn = build_view(processor, model, system_prompt, up_noq, ds,
                                pixel_values=pixel_values,
                                audio_values=audio_values)
                lp_f = view_logp(model, vf, gen_ids)
                lp_n = view_logp(model, vn, gen_ids)
                rec_out['delta_text_tok'] = [round(float(x), 4)
                                             for x in lp_f - lp_n]
                if p5b_row is not None:
                    rec_out['delta_video'] = p5b_row['delta_video']
                    rec_out['delta_audio'] = p5b_row['delta_audio']
                del vf, vn

            base_lp = np.array(rec['logp_list'], dtype=np.float64)
            tf = (float(np.max(np.abs(lp_f - base_lp)))
                  if len(base_lp) == len(lp_f) else float('nan'))
            rec_out['tf_repro'] = round(tf, 4)
            fout.write(json.dumps(rec_out, ensure_ascii=False) + '\n')
            fout.flush()
            n_done += 1
            if n_done % 10 == 0:
                print(f'  [{n_done}/{len(pairs)-len(done)}] {time.time()-t0:.0f}s '
                      f'tf={tf:.4f}', flush=True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f'  [ERROR] uid={uid_key}: {e}', flush=True)
        torch.cuda.empty_cache()

    fout.close()
    print(f'[done] {n_done} new rows in {time.time()-t0:.0f}s -> {out_path}',
          flush=True)


def run_phd(args):
    out_path = args.out or 'outputs/cf_probe/qwen3_textcf/phd_textcf_tokens.jsonl'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    data = json.load(open(PHD_DATA))
    JD = os.path.dirname(PHD_DATA)

    def load_img(item):
        ip = item['media'].get('image')
        rp = resolve_media_path(ip, JD, JD) if ip else None
        if not rp or not os.path.exists(rp):
            return None
        return load_image(rp)

    good = [d for d in data if load_img(d) is not None]
    print(f'[phd] items with image on disk: {len(good)}/{len(data)}', flush=True)
    rng = random.Random(args.seed)
    rng.shuffle(good)
    sample = good[:args.n_samples]
    if args.limit:
        sample = sample[:args.limit]

    done = set()
    if os.path.exists(out_path):
        for l in open(out_path):
            line = l.strip()
            if line:
                done.add(json.loads(line)['question'])
        print(f'[resume] {len(done)} already done', flush=True)
    fout = open(out_path, 'a' if done else 'w')

    print('[model] loading Qwen3-Omni thinker (~62GB, exclusive card)...', flush=True)
    processor, model = load_model()
    tokenizer = processor.tokenizer
    system_prompt = get_system_prompt('phd', 'v1')
    IM_END = tokenizer.convert_tokens_to_ids('<|im_end|>')
    try:
        ENDOFTEXT = tokenizer.convert_tokens_to_ids('<|endoftext|>')
    except Exception:
        ENDOFTEXT = IM_END

    def build_view_img(user_prompt, image):
        uc = build_user_content_with_media_tokens(None, None, image, user_prompt)
        raw = build_inputs(processor, model, make_messages(system_prompt, uc),
                           None, None, image, 'phd')
        return prepare_for_thinker(raw, model)

    t0, n_done, n_short = time.time(), 0, 0
    for si, item in enumerate(sample):
        q = item['question']
        if q in done:
            continue
        try:
            img = load_img(item)
            if img is None:
                continue
            up = get_user_prompt(q, 'phd', 'v1')
            up_noq = get_user_prompt(NEUTRAL_Q['phd'], 'phd', 'v1')
            assert up != up_noq and NEUTRAL_Q['phd'] in up_noq

            full_view = build_view_img(up, img)
            input_len = full_view['input_ids'].shape[1]
            with torch.inference_mode():
                out = model.generate(**full_view, max_new_tokens=args.max_new_tokens,
                                     do_sample=False,
                                     eos_token_id=[IM_END, ENDOFTEXT],
                                     pad_token_id=IM_END)
            gen = out[0, input_len:].tolist()
            while gen and gen[-1] in (IM_END, ENDOFTEXT):
                gen.pop()
            if len(gen) < args.min_tok:
                n_short += 1
                continue
            gen_ids = torch.tensor(gen, dtype=torch.long, device=model.device)
            toks, cls = tokens_classes(tokenizer, gen)

            partner = None
            for _ in range(20):
                cand = rng.choice(good)
                if cand is not item:
                    pim = load_img(cand)
                    if pim is not None:
                        partner = pim
                        break
            if partner is None:
                continue

            noq_view = build_view_img(up_noq, img)
            img_view = build_view_img(up, partner)
            lp_f = view_logp(model, full_view, gen_ids)
            lp_n = view_logp(model, noq_view, gen_ids)
            lp_i = view_logp(model, img_view, gen_ids)

            fout.write(json.dumps({
                'dataset': 'phd', 'sample_i': si, 'question': q,
                'n_tok': len(gen), 'tokens': toks, 'classes': cls,
                'delta_img_tok': [round(float(x), 4) for x in lp_f - lp_i],
                'delta_text_tok': [round(float(x), 4) for x in lp_f - lp_n],
            }, ensure_ascii=False) + '\n')
            fout.flush()
            n_done += 1
            if n_done % 10 == 0:
                print(f'  [{n_done}] {time.time()-t0:.0f}s', flush=True)
            del full_view, noq_view, img_view
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f'  [ERROR] q={q[:50]}: {e}', flush=True)
        torch.cuda.empty_cache()

    fout.close()
    print(f'[done] {n_done} kept ({n_short} short CoT skipped) in '
          f'{time.time()-t0:.0f}s -> {out_path}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True,
                    choices=['haloquest', 'phd', 'avhbench'])
    ap.add_argument('--n_samples', type=int, default=200,
                    help='phd only: pool cap before generation')
    ap.add_argument('--min_tok', type=int, default=6)
    ap.add_argument('--max_new_tokens', type=int, default=300)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--limit', type=int, default=0, help='smoke: first N samples')
    ap.add_argument('--out', default='')
    args = ap.parse_args()
    if args.dataset == 'phd':
        run_phd(args)
    else:
        run_halo_or_avh(args, args.dataset)


if __name__ == '__main__':
    main()
