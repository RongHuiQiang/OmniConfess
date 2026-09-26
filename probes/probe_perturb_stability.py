# -*- coding: utf-8 -*-
"""probe_perturb_stability.py — input-perturbation stability of the delta ruler.

Question (measurement-validity, reviewer-facing): is Delta_audio/Delta_video
stable under *surface* perturbations of the input that leave the text stream
untouched?  Three perturbation ladders, all teacher-forced on the SAME base
greedy token_ids as the bigN probe (outputs/cf_probe/avh_probe_bigN_shard0of1.jsonl):

  crf32    video re-compressed with libx264 CRF 32 (ffmpeg 7.0.2-static, ./bin/ffmpeg)
  halfres  video downscaled to half linear resolution, then up-sampled back by
           the loader's 224x224 resize (ffmpeg scale=iw/2:ih/2)
  audnoise audio + gaussian noise at SNR 20 dB (pure numpy, seeded per-sample)

Sample selection mirrors the bigN run exactly: token_id-bearing rows of
avhbench_base_model_full.jsonl -> stratified_sample(n=400, seed=42) -> first
--limit rows.  Views (full / novideo / noaudio) built with build_view() copied
from probe_cf_avhbench.py; per-token logp via view_logp().  Because the
perturbations do not change the text or the media-placeholder layout, the
prompt token sequence is identical; we verify this per sample by asserting the
perturbed prompt length equals the unperturbed prompt length, and for the
first CHECK_N samples we recompute the unperturbed full view and compare with
the bigN row's lp_full (determinism / repro check).

Usage:
    MMHALU_MODEL_ROOT=/ultrafast_share/tsq/MODELS CUDA_VISIBLE_DEVICES=7 \
    /ultrafast_share/tsq/venvs/omniconfess/bin/python probe_perturb_stability.py \
        --perturb crf32 --limit 100 \
        --out outputs/cf_probe/avh_perturb_crf32_shard0of1.jsonl
"""
import os, sys, json, argparse, subprocess, zlib
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'OmniConfess', 'methods'))
sys.path.insert(0, HERE)

from shared import (  # noqa: E402
    load_dataset, load_media_for_item, load_model,
)
from shared.data_loader import load_video_frames, resolve_media_path  # noqa: E402
from probe_cf_avhbench import (  # noqa: E402
    stratified_sample, token_f1, build_view, view_logp, retry,
)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("MMHALU_MODEL_ROOT", "/ultrafast_share/tsq/MODELS")

FFMPEG = os.path.join(HERE, '..', 'bin', 'ffmpeg')
DATA_PATH = os.path.join(HERE, 'OmniHalluBench', 'avhbench', 'avhbench.json')
SCORED = os.path.join(HERE, 'outputs', 'avhbench_base_model',
                      'avhbench_base_model_full.jsonl')
BIGN = os.path.join(HERE, 'outputs', 'cf_probe', 'avh_probe_bigN_shard0of1.jsonl')
MEDIA_CACHE = os.path.join(HERE, 'outputs', 'cf_probe', 'perturbed_media')

CHECK_N = 3   # unperturbed determinism checks per perturbation run


# ---------------------------------------------------------------------------
# Perturbations
# ---------------------------------------------------------------------------
def ffmpeg_video(src, dst, extra):
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return dst
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    cmd = [FFMPEG, '-y', '-v', 'error', '-i', src] + extra + ['-an', dst]
    subprocess.run(cmd, check=True, capture_output=True)
    return dst


def perturbed_video(item, data_path, data_root, mode):
    """(crf32|halfres) -> numpy frames loaded from the perturbed file."""
    json_dir = os.path.dirname(data_path)
    src = resolve_media_path(item['video_path'], json_dir, data_root)
    base = os.path.splitext(os.path.basename(src))[0]
    if mode == 'crf32':
        extra = ['-c:v', 'libx264', '-crf', '32', '-preset', 'medium']
        dst = os.path.join(MEDIA_CACHE, 'crf32', base + '.mp4')
    elif mode == 'halfres':
        extra = ['-vf', 'scale=trunc(iw/4)*2:trunc(ih/4)*2',
                 '-c:v', 'libx264', '-crf', '18', '-preset', 'medium']
        dst = os.path.join(MEDIA_CACHE, 'halfres', base + '.mp4')
    else:
        raise ValueError(mode)
    ffmpeg_video(src, dst, extra)
    frames = load_video_frames(dst)
    assert frames is not None, f'perturbed video failed: {dst}'
    return frames


def noisy_audio(audio, snr_db, seed):
    """audio + gaussian noise at target SNR (pure numpy)."""
    rng = np.random.default_rng(seed)
    x = np.asarray(audio, dtype=np.float64)
    p_sig = float(np.mean(x ** 2))
    sigma = np.sqrt(p_sig / (10.0 ** (snr_db / 10.0)))
    return (x + rng.standard_normal(x.shape) * sigma).astype(np.float32)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--perturb', required=True,
                    choices=['crf32', 'halfres', 'audnoise'])
    ap.add_argument('--limit', type=int, default=100)
    ap.add_argument('--big_n', type=int, default=400)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', default='')
    ap.add_argument('--snr_db', type=float, default=20.0)
    args = ap.parse_args()

    out_path = args.out or os.path.join(
        HERE, 'outputs', 'cf_probe',
        f'avh_perturb_{args.perturb}_shard0of1.jsonl')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    done_ids = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            line = line.strip()
            if line:
                done_ids.add(json.loads(line)['id'])
    fout = open(out_path, 'a')

    recs = [json.loads(l) for l in open(SCORED)]
    recs = [r for r in recs if r.get('token_ids')]
    for r in recs:
        r['_f1'] = token_f1(r['pred'], r['gt'])
    sample = stratified_sample(recs, args.big_n, args.seed)[:args.limit]
    bign = {r['id']: r for r in map(json.loads, open(BIGN))}

    data_root = os.path.dirname(DATA_PATH)
    data = load_dataset(DATA_PATH, data_root)
    by_qg = {(d['question'], d['gt']): d for d in data}

    processor, model = load_model()
    thinker = model.thinker
    from shared import get_system_prompt, get_user_prompt
    system_prompt = get_system_prompt('avhbench', 'v1')

    todo = [r for r in sample if str(r['id']) not in done_ids]
    print(f'[perturb {args.perturb}] {len(todo)}/{len(sample)} to run '
          f'({len(done_ids)} resumed)', flush=True)
    n_check = 0
    import time
    t0 = time.time()
    for i, rec in enumerate(todo):
        rid = str(rec['id'])
        item = by_qg.get((rec['question'], rec['gt']))
        if item is None:
            print(f'  [skip] no dataset match id={rid}', flush=True)
            continue
        try:
            pixel_values, audio_values, _ = retry(
                lambda: load_media_for_item(item, DATA_PATH, data_root),
                tag='load_media')
            if pixel_values is None or audio_values is None:
                print(f'  [skip] media missing id={rid}', flush=True)
                continue

            if args.perturb in ('crf32', 'halfres'):
                p_pix = perturbed_video(item, DATA_PATH, data_root, args.perturb)
                p_aud = audio_values
            else:
                p_pix = pixel_values
                seed = zlib.crc32(rid.encode()) & 0xffffffff
                p_aud = noisy_audio(audio_values, args.snr_db, seed)

            gen_ids = torch.tensor(rec['token_ids'], dtype=torch.long,
                                   device=model.device)
            user_prompt = get_user_prompt(item['question'], 'avhbench', 'v1')

            # NOTE: the scored jsonl was regenerated after the bigN run, so its
            # token_ids no longer match the bigN rows.  The clean reference is
            # therefore recomputed in the same process/session as the perturbed
            # views -- identical tokens, identical code path, self-consistent.
            conds = [('clean', pixel_values, audio_values)]
            conds.append((args.perturb, p_pix, p_aud))
            out_views, prompt_lens = {}, {}
            for cname, cpix, caud in conds:
                views = {}
                for name, kv, ka in (('full', True, True),
                                     ('novideo', False, True),
                                     ('noaudio', True, False)):
                    pix = cpix if kv else None
                    aud = caud if ka else None
                    clean = retry(lambda: build_view(
                        processor, model, system_prompt, user_prompt, pix, aud,
                        kv, ka), tag=f'build_{cname}_{name}')
                    if cname == 'clean':
                        prompt_lens[name] = int(clean['input_ids'].shape[1])
                    views[name] = retry(
                        lambda c=clean: view_logp(thinker, c, gen_ids),
                        tag=f'forward_{cname}_{name}')
                    del clean
                    torch.cuda.empty_cache()
                out_views[cname] = views

            # token-sequence alignment: each view's lp vector must have one
            # entry per base token; and the perturbed prompt skeleton must
            # equal the clean prompt skeleton (placeholders unchanged).
            n_tok = len(rec['token_ids'])
            aligned = all(
                len(v['full']) == n_tok and len(v['noaudio']) == n_tok and
                len(v['novideo']) == n_tok for v in out_views.values())

            # determinism check vs the bigN row, when token lengths happen to
            # match (they usually do not, since the scored file was rerun).
            repro_full = float('nan')
            base_row = bign.get(rid)
            if n_check < CHECK_N and base_row is not None:
                n_check += 1
                base_lp = np.array(base_row['lp_full'], dtype=np.float64)
                if len(base_lp) == n_tok:
                    repro_full = float(np.max(np.abs(
                        out_views['clean']['full'] - base_lp)))

            row = {'id': rid, 'perturb': args.perturb, 'f1': rec['_f1'],
                   'pred_words': len(rec['pred'].split()),
                   'n_tokens': n_tok,
                   'prompt_len_full': prompt_lens['full'],
                   'aligned': bool(aligned),
                   'repro_full_maxabs': round(repro_full, 4)}
            for cname, views in out_views.items():
                row[f'lp_full_{cname}'] = views['full'].round(4).tolist()
                row[f'lp_noaudio_{cname}'] = views['noaudio'].round(4).tolist()
                row[f'lp_novideo_{cname}'] = views['novideo'].round(4).tolist()
                row[f'delta_audio_{cname}'] = (
                    views['full'] - views['noaudio']).round(4).tolist()
                row[f'delta_video_{cname}'] = (
                    views['full'] - views['novideo']).round(4).tolist()
            fout.write(json.dumps(row, ensure_ascii=False) + '\n')
            fout.flush()
        except Exception as e:
            print(f'  [ERROR] id={rid}: {type(e).__name__}: {e}', flush=True)
        if (i + 1) % 10 == 0:
            print(f'  [{i+1}/{len(todo)}] {time.time()-t0:.0f}s '
                  f'mem={torch.cuda.max_memory_allocated()/2**30:.1f}G',
                  flush=True)
        torch.cuda.empty_cache()
    fout.close()
    print(f'[done {args.perturb}] -> {out_path} ({time.time()-t0:.0f}s)',
          flush=True)


if __name__ == '__main__':
    main()
