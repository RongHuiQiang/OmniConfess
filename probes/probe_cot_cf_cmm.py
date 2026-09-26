# -*- coding: utf-8 -*-
"""probe_cot_cf_cmm.py -- per-token counterfactual (CF) probe over a freshly
generated cmm describe-then-judge CoT, to test whether the problem-structure
**sparsity axis** holds on the 6th (audio-visual JUDGMENT) dataset.

Why generate rather than reuse a stored trajectory:
  - cmm's base_model grid run used a terse prompt -> 2-token yes/no, nothing to
    be sparse over.
  - cmm_base_opera has multi-token traces but they carry the OPERA decoding
    intervention confound -- unfit to characterise the *base model's* structure.
  - cmm's OWN v1 system prompt is already a describe-then-judge format
    ("Description: [one sentence] ... Judgment: \\boxed{Yes/No}"). We generate
    that NATIVE CoT greedily on the plain base model (no OPERA, no VCD, no
    intervention), then probe it. So the trajectory is native to cmm's v1 task
    design, only the base_model grid cell happened to use the terse variant.

Probe (same contrast design as ragtruth wrong-passage / haloquest swapped-image):
    full : own video + own audio
    ctrl : a DIFFERENT item's video + audio (random, != own)
    delta_tok[i] = logP(tok_i | own AV) - logP(tok_i | swapped AV)
  i.e. the per-token audio-visual counterfactual truth-move (lens A). Sparsity of
  |delta_tok| over the CoT content tokens, via leg2 formulas, vs random control.

Reuses shared model machinery + leg2 sparsity formulas verbatim. Single card;
generation is greedy (deterministic); probe is teacher-forced.

    CUDA_VISIBLE_DEVICES=<c> python probe_cot_cf_cmm.py --n_samples 160 \
        --out outputs/problem_structure_fig/cmm_cot_cf_tokens.jsonl
"""
import os, sys, json, argparse, random, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, 'OmniConfess', 'methods'))

os.environ.setdefault("MMHALU_MODEL_ROOT", "/ultrafast_share/tsq/MODELS")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from shared import (  # noqa: E402
    load_model, build_inputs, prepare_for_thinker,
    build_user_content_with_media_tokens, make_messages,
    get_system_prompt, get_user_prompt, resolve_media_path,
)
from shared.data_loader import load_video_frames, load_audio_data  # noqa: E402
from probe_claim_cf import view_logp, token_class  # noqa: E402
import leg2_sparsity_math as L2  # noqa: E402
import torch  # noqa: E402

# numpy 2.x removed np.trapz (renamed np.trapezoid); leg2 still calls np.trapz.
if not hasattr(np, 'trapz') and hasattr(np, 'trapezoid'):
    np.trapz = np.trapezoid

DATA = os.path.join(HERE, 'OmniHalluBench/cmm/cmm.json')
JD = os.path.dirname(DATA)


def load_av(item):
    """Load (video_frames, audio) for a cmm item, or (None,None) if missing."""
    vp = resolve_media_path(item['media'].get('video'), JD, JD)
    ap = resolve_media_path(item['media'].get('audio'), JD, JD)
    px = load_video_frames(vp) if vp else None
    au = load_audio_data(ap) if ap else None
    return px, au


def build_view_av(processor, model, system_prompt, user_prompt, px, au):
    uc = build_user_content_with_media_tokens(px, au, None, user_prompt)
    messages = make_messages(system_prompt, uc)
    raw = build_inputs(processor, model, messages, px, au, None, 'cmm')
    return prepare_for_thinker(raw, model)


def run(args):
    data = json.load(open(DATA))
    # only items whose own video+audio both resolve on disk
    good = []
    for d in data:
        m = d.get('media') or {}
        vp = resolve_media_path(m.get('video'), JD, JD)
        ap = resolve_media_path(m.get('audio'), JD, JD)
        if vp and ap and os.path.exists(vp) and os.path.exists(ap):
            good.append(d)
    print(f'[cmm-cf] items with both media on disk: {len(good)}/{len(data)}', flush=True)

    rng = random.Random(args.seed)
    # stratify by hallucination_type so all cmm failure modes are represented
    buckets = {}
    for d in good:
        k = ((d.get('facets') or {}).get('hallucination_type')) or 'na'
        buckets.setdefault(k, []).append(d)
    for k in buckets:
        rng.shuffle(buckets[k])
    print('[cmm-cf] hallucination_type buckets:',
          {k: len(v) for k, v in buckets.items()}, flush=True)
    order = list(buckets.keys())
    sample = []
    while len(sample) < args.n_samples and any(buckets[k] for k in order):
        for k in order:
            if buckets[k] and len(sample) < args.n_samples:
                sample.append(buckets[k].pop())
    if args.limit:
        sample = sample[:args.limit]

    processor, model = load_model()
    tokenizer = processor.tokenizer
    thinker = model.thinker
    # NOTE: the NATIVE cmm v1 base prompt (get_system_prompt('cmm','v1')) makes
    # Qwen2.5-Omni-7B emit a single decision token ("Yes"/"No") -- verified on
    # card2, gen_len=1 for every sample. There is therefore NO native multi-token
    # CoT to be sparse over (this is exactly why leg2 excluded cmm). To obtain a
    # real AV-grounded trajectory for the per-token sparsity + modality axes we
    # generate a CONSTRUCTED describe-then-judge CoT with an explicit
    # "describe first, then judge" prompt. This is honestly a constructed CoT,
    # NOT the base v1 cell -- every downstream number carries this caveat.
    if args.native_prompt:
        system_prompt = get_system_prompt('cmm', 'v1')
    else:
        system_prompt = (
            "You are an audio-visual reporter. First describe, in one sentence, "
            "what you actually see in the video and hear in the audio. Then give a "
            "final judgment.\n\n"
            "Description: <one sentence of what you see and hear>\n"
            "Judgment: \\boxed{Yes} or \\boxed{No}"
        )
    eos_ids = set(x for x in [tokenizer.eos_token_id,
                              getattr(tokenizer, 'pad_token_id', None)] if x is not None)
    # also stop on <|im_end|>
    try:
        eos_ids.add(tokenizer.convert_tokens_to_ids('<|im_end|>'))
    except Exception:
        pass

    out_f = open(args.out, 'w')
    t0 = time.time()
    n_done = 0
    for si, item in enumerate(sample):
        try:
            px, au = load_av(item)
            if px is None or au is None:
                continue
            if args.native_prompt:
                up = get_user_prompt(item['question'], 'cmm', 'v1')
            else:
                up = ("Task: Analyze the video and audio to answer the question.\n"
                      f"Question: {item['question']}\n"
                      "Description:")

            # --- generate the native describe-then-judge CoT (greedy) ---------
            full_view = build_view_av(processor, model, system_prompt, up, px, au)
            input_len = full_view['input_ids'].shape[1]
            with torch.inference_mode():
                out_ids = thinker.generate(**full_view, max_new_tokens=args.max_new_tokens,
                                           do_sample=False, num_return_sequences=1)
            gen = out_ids[0, input_len:].tolist()
            # trim trailing eos / pad / im_end
            while gen and gen[-1] in eos_ids:
                gen.pop()
            if os.environ.get('CMM_DEBUG'):
                print(f'[cmm-cf DEBUG] si={si} gen_len={len(gen)} text={tokenizer.decode(gen, skip_special_tokens=True)[:120]!r}', flush=True)
            if len(gen) < args.min_tok:
                continue
            gen_ids = torch.tensor(gen, dtype=torch.long, device=model.device)
            toks = [tokenizer.decode([t], skip_special_tokens=True) for t in gen]
            cls = [token_class(t, toks[i - 1] if i else None) for i, t in enumerate(toks)]

            # --- contrast: a DIFFERENT item's video+audio --------------------
            partner = None
            for _ in range(20):
                cand = rng.choice(good)
                if cand is not item:
                    ppx, pau = load_av(cand)
                    if ppx is not None and pau is not None:
                        partner = (ppx, pau); break
            if partner is None:
                continue
            ppx, pau = partner
            # 4 views on the SAME native CoT:
            #   full  = own video + own audio
            #   ctrl  = partner video + partner audio  (both swapped)  -> delta   (sparsity axis)
            #   vswap = partner video + OWN audio       (video swapped) -> delta_v (video marginal)
            #   aswap = OWN video + partner audio       (audio swapped) -> delta_a (audio marginal)
            ctrl_view  = build_view_av(processor, model, system_prompt, up, ppx, pau)
            vswap_view = build_view_av(processor, model, system_prompt, up, ppx, au)
            aswap_view = build_view_av(processor, model, system_prompt, up, px, pau)

            lp_full  = view_logp(thinker, full_view,  gen_ids)
            lp_ctrl  = view_logp(thinker, ctrl_view,  gen_ids)
            lp_vswap = view_logp(thinker, vswap_view, gen_ids)
            lp_aswap = view_logp(thinker, aswap_view, gen_ids)
            delta   = lp_full - lp_ctrl    # both swapped (sparsity axis, unchanged)
            delta_v = lp_full - lp_vswap   # video marginal contribution
            delta_a = lp_full - lp_aswap   # audio marginal contribution

            out_f.write(json.dumps({
                'sample_i': si,
                'question': item['question'],
                'halluc_type': (item.get('facets') or {}).get('hallucination_type'),
                'n_tok': len(gen),
                'text': tokenizer.decode(gen, skip_special_tokens=True),
                'delta_tok': [float(x) for x in delta],
                'delta_v': [float(x) for x in delta_v],
                'delta_a': [float(x) for x in delta_a],
                'classes': cls,
                'full_logp': [float(x) for x in lp_full],
            }) + '\n')
            out_f.flush()
            n_done += 1
        except Exception as e:
            print(f'[cmm-cf] sample {si} ERR {type(e).__name__}: {e}', flush=True)
            continue
        if (si + 1) % 10 == 0:
            el = time.time() - t0
            print(f'[cmm-cf] {si+1}/{len(sample)} done={n_done} {el/(si+1):.1f}s/it', flush=True)
    out_f.close()

    # ---- sparsity statistics (leg2 formulas, lens A = |delta_tok|) -----------
    samples = [json.loads(l) for l in open(args.out)]

    def sample_vec(s):
        d = np.abs(np.asarray(s['delta_tok'], float))
        cl = s['classes']
        keep = np.array([c in ('content', 'num') for c in cl]) if len(cl) == len(d) else np.ones(len(d), bool)
        v = d[keep] if keep.sum() >= 5 else d
        return v

    per_sample, pooled = [], []
    for s in samples:
        v = sample_vec(s)
        if v.size >= 5:
            per_sample.append(L2.participation_ratio(v) / v.size)
            pooled.append(v)
    pooled_vec = np.concatenate(pooled) if pooled else np.array([])
    stats = L2.all_stats(pooled_vec) if pooled_vec.size else {}
    ctrl = L2.random_control(pooled_vec) if pooled_vec.size else {}

    # ---- modality summary (per-token point cloud [delta_v, delta_a]) ---------
    # pooled over content/num tokens; eff_rank/principal_angle use leg1 covariance
    # convention (exp(entropy of normalised covariance eigenvalues); PC1 angle).
    def eff_rank_and_angle(A):
        A = np.asarray(A, float)
        if A.shape[0] < 3:
            return {}
        A = A - A.mean(0)
        C = np.cov(A.T)
        w, V = np.linalg.eigh(C)
        w = np.clip(w, 0, None)
        s = w.sum()
        if s <= 0:
            return {}
        p = w / s
        ent = -(p[p > 0] * np.log(p[p > 0])).sum()
        eff_rank = float(np.exp(ent))
        pc1 = V[:, np.argmax(w)]
        import math as _m
        ang = _m.degrees(_m.atan2(abs(pc1[1]), abs(pc1[0])))
        return {'eff_rank': round(eff_rank, 4),
                'principal_angle_deg': round(float(ang), 4),
                'pc1_energy_frac': round(float(w.max() / s), 4)}

    dv_all, da_all = [], []
    for s in samples:
        dv = s.get('delta_v'); da = s.get('delta_a'); cl = s.get('classes')
        if not dv or not da:
            continue
        dv = np.asarray(dv, float); da = np.asarray(da, float)
        if cl and len(cl) == len(dv):
            keep = np.array([c in ('content', 'num') for c in cl])
        else:
            keep = np.ones(len(dv), bool)
        dv_all.extend(dv[keep].tolist()); da_all.extend(da[keep].tolist())
    modality = {}
    if len(dv_all) >= 3:
        dv_arr = np.asarray(dv_all, float); da_arr = np.asarray(da_all, float)
        if dv_arr.std() > 0 and da_arr.std() > 0:
            pear = float(np.corrcoef(dv_arr, da_arr)[0, 1])
        else:
            pear = float('nan')
        A = np.c_[dv_arr, da_arr]
        modality = {
            'unit': 'per-token (delta_v, delta_a) over content/num tokens',
            'n_tokens': int(len(dv_all)),
            'pearson': round(pear, 4),
            **eff_rank_and_angle(A),
            'mean_abs_delta_v': round(float(np.mean(np.abs(dv_arr))), 4),
            'mean_abs_delta_a': round(float(np.mean(np.abs(da_arr))), 4),
        }

    summary = {
        'dataset': 'cmm',
        'lens': 'A (per-token AV-CF |delta_tok|, video+audio swap contrast)',
        'trajectory': ('CONSTRUCTED describe-then-judge CoT (explicit describe-first prompt), '
                       'greedy on Qwen2.5-Omni base (no OPERA); NOTE base v1 cell emits 1 '
                       'decision token, so this is a constructed CoT not the native base cell'),
        'n_samples_written': len(samples),
        'n_samples_used': len(per_sample),
        'n_tokens_pooled': int(pooled_vec.size),
        'pooled_stats': stats,
        'random_control': ctrl,
        'per_sample_partic_frac_mean': float(np.nanmean(per_sample)) if per_sample else float('nan'),
        'modality_per_token': modality,
    }
    js = args.out.replace('.jsonl', '_summary.json')
    json.dump(summary, open(js, 'w'), indent=2)
    print('[cmm-cf] SUMMARY', json.dumps(summary, indent=2), flush=True)
    print('[cmm-cf] wrote', args.out, 'and', js, flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_samples', type=int, default=140)
    ap.add_argument('--native_prompt', action='store_true',
                    help='use the terse base v1 prompt (emits 1 token; degenerate)')
    ap.add_argument('--max_new_tokens', type=int, default=64)
    ap.add_argument('--min_tok', type=int, default=6)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--out', default=os.path.join(
        HERE, 'outputs/problem_structure_fig/cmm_cot_cf_tokens.jsonl'))
    run(ap.parse_args())
