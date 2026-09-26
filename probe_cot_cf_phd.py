# -*- coding: utf-8 -*-
"""probe_cot_cf_phd.py -- per-token counterfactual (CF) probe over a freshly
generated phd observe-then-judge CoT, to upgrade the problem-structure
**sparsity axis** phd cell from a weak CoT -logp proxy to a NATIVE per-token CF.

Why generate:
  - phd is an image+text hypothesis-judgment set. Its NATIVE base v1 prompt
    ("You are a visual analyst ... Judgment: \\boxed{Yes|No}") produces, for a
    subset of items, a real observe-then-judge CoT:
        "Hypothesis to Test: ...\n\nImage Observation:\n- ...\n\nJudgment: \\boxed{No}"
    (verified: 153/906 stored base rows have >=6 output tokens; median 2 = bare
    "No" for the rest). This is the SAME native prompt, so the multi-token CoT it
    yields on the >=6-token subset is native, not constructed (contrast: cmm's
    base prompt emits a single decision token, so cmm needs a constructed prompt).

Probe (image-swap contrast, exactly like haloquest in probe_claim_cf.py):
    full : own COCO image
    ctrl : a DIFFERENT item's image (random, != own)
    delta_tok[i] = logP(tok_i | own image) - logP(tok_i | swapped image)
  Sparsity of |delta_tok| over CoT content tokens (leg2 formulas) vs random ctrl.

Reuses shared model machinery + probe_claim_cf.view_logp/token_class + leg2.
Single card; generation greedy; probe teacher-forced. Only samples whose freshly
generated CoT has >= min_tok(6) tokens are kept.

    CUDA_VISIBLE_DEVICES=2 python probe_cot_cf_phd.py --n_samples 160 \
        --out outputs/problem_structure_fig/phd_cot_cf_tokens.jsonl
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
from shared.data_loader import load_image  # noqa: E402
from probe_claim_cf import view_logp, token_class  # noqa: E402
import leg2_sparsity_math as L2  # noqa: E402
import torch  # noqa: E402

# numpy 2.x removed np.trapz (renamed np.trapezoid); leg2 still calls np.trapz.
if not hasattr(np, 'trapz') and hasattr(np, 'trapezoid'):
    np.trapz = np.trapezoid

DATA = os.path.join(HERE, 'OmniHalluBench/phd/phd.json')
JD = os.path.dirname(DATA)
SCORED = os.path.join(HERE,
                      'outputs/phd_base_model/phd_base_model_full_single_scored.jsonl')


def load_img(item):
    ip = item['media'].get('image')
    rp = resolve_media_path(ip, JD, JD) if ip else None
    if not rp or not os.path.exists(rp):
        return None
    return load_image(rp)


def build_view_img(processor, model, system_prompt, user_prompt, image):
    uc = build_user_content_with_media_tokens(None, None, image, user_prompt)
    messages = make_messages(system_prompt, uc)
    raw = build_inputs(processor, model, messages, None, None, image, 'phd')
    return prepare_for_thinker(raw, model)


def run(args):
    data = json.load(open(DATA))
    # join phd.json <-> scored by QUESTION TEXT (id may not correspond; verify)
    scored = [json.loads(l) for l in open(SCORED)]
    by_q_scored = {}
    for r in scored:
        by_q_scored.setdefault((r.get('question') or '').strip(), r)
    good = []
    for d in data:
        img = d['media'].get('image')
        rp = resolve_media_path(img, JD, JD) if img else None
        if not (rp and os.path.exists(rp)):
            continue
        good.append(d)
    print(f'[phd-cf] items with image on disk: {len(good)}/{len(data)}', flush=True)
    # verify join coverage
    joined = sum(1 for d in good if (d.get('question') or '').strip() in by_q_scored)
    print(f'[phd-cf] question-join to scored: {joined}/{len(good)}', flush=True)

    rng = random.Random(args.seed)
    # Greedy generation is deterministic, so an item's fresh CoT length matches
    # its stored base CoT length. Only ~153/906 items natively emit a >=6-token
    # observe-then-judge CoT (the rest emit a bare "No"). Bias the sample toward
    # those long-CoT items (via the question-join to the scored file) so we don't
    # waste generation on 2-token items -- this does NOT change the native prompt,
    # only which native items we probe. Fall back to unfiltered if join is thin.
    def stored_len(d):
        r = by_q_scored.get((d.get('question') or '').strip())
        return len(r.get('token_ids') or []) if r else 0
    long_items = [d for d in good if stored_len(d) >= args.min_tok]
    print(f'[phd-cf] items with native stored CoT >= {args.min_tok} tok: {len(long_items)}', flush=True)
    pool = long_items if len(long_items) >= 20 else good
    rng.shuffle(pool)
    sample = pool[:args.n_samples] if not args.limit else pool[:args.limit]

    processor, model = load_model()
    tokenizer = processor.tokenizer
    thinker = model.thinker
    system_prompt = get_system_prompt('phd', 'v1')
    eos_ids = set(x for x in [tokenizer.eos_token_id,
                              getattr(tokenizer, 'pad_token_id', None)] if x is not None)
    try:
        eos_ids.add(tokenizer.convert_tokens_to_ids('<|im_end|>'))
    except Exception:
        pass

    out_f = open(args.out, 'w')
    t0 = time.time()
    n_done = 0
    n_short = 0
    for si, item in enumerate(sample):
        try:
            img = load_img(item)
            if img is None:
                continue
            up = get_user_prompt(item['question'], 'phd', 'v1')

            # --- generate the native observe-then-judge CoT (greedy) ----------
            full_view = build_view_img(processor, model, system_prompt, up, img)
            input_len = full_view['input_ids'].shape[1]
            with torch.inference_mode():
                out_ids = thinker.generate(**full_view, max_new_tokens=args.max_new_tokens,
                                           do_sample=False, num_return_sequences=1)
            gen = out_ids[0, input_len:].tolist()
            while gen and gen[-1] in eos_ids:
                gen.pop()
            if os.environ.get('PHD_DEBUG'):
                print(f'[phd-cf DEBUG] si={si} gen_len={len(gen)} '
                      f'text={tokenizer.decode(gen, skip_special_tokens=True)[:120]!r}', flush=True)
            if len(gen) < args.min_tok:      # keep only multi-token CoT
                n_short += 1
                continue
            gen_ids = torch.tensor(gen, dtype=torch.long, device=model.device)
            toks = [tokenizer.decode([t], skip_special_tokens=True) for t in gen]
            cls = [token_class(t, toks[i - 1] if i else None) for i, t in enumerate(toks)]

            # --- contrast: a DIFFERENT item's image ---------------------------
            partner = None
            for _ in range(20):
                cand = rng.choice(good)
                if cand is not item:
                    pim = load_img(cand)
                    if pim is not None:
                        partner = pim; break
            if partner is None:
                continue
            ctrl_view = build_view_img(processor, model, system_prompt, up, partner)

            lp_full = view_logp(thinker, full_view, gen_ids)
            lp_ctrl = view_logp(thinker, ctrl_view, gen_ids)
            delta = lp_full - lp_ctrl

            out_f.write(json.dumps({
                'sample_i': si,
                'question': item['question'],
                'n_tok': len(gen),
                'text': tokenizer.decode(gen, skip_special_tokens=True),
                'delta_tok': [float(x) for x in delta],
                'classes': cls,
                'full_logp': [float(x) for x in lp_full],
            }) + '\n')
            out_f.flush()
            n_done += 1
        except Exception as e:
            print(f'[phd-cf] sample {si} ERR {type(e).__name__}: {e}', flush=True)
            continue
        if (si + 1) % 10 == 0:
            el = time.time() - t0
            print(f'[phd-cf] {si+1}/{len(sample)} done={n_done} short={n_short} '
                  f'{el/(si+1):.1f}s/it', flush=True)
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
    summary = {
        'dataset': 'phd',
        'lens': 'A (per-token image-CF |delta_tok|, image-swap contrast)',
        'trajectory': ('NATIVE phd v1 observe-then-judge CoT (Hypothesis/Image Observation/'
                       'Judgment), greedy on Qwen2.5-Omni base; kept only freshly generated '
                       'CoT with >= %d tokens' % args.min_tok),
        'n_samples_written': len(samples),
        'n_samples_used': len(per_sample),
        'n_short_dropped': n_short,
        'n_tokens_pooled': int(pooled_vec.size),
        'pooled_stats': stats,
        'random_control': ctrl,
        'per_sample_partic_frac_mean': float(np.nanmean(per_sample)) if per_sample else float('nan'),
    }
    js = args.out.replace('.jsonl', '_summary.json')
    json.dump(summary, open(js, 'w'), indent=2)
    print('[phd-cf] SUMMARY', json.dumps(summary, indent=2), flush=True)
    print('[phd-cf] wrote', args.out, 'and', js, flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_samples', type=int, default=160)
    ap.add_argument('--max_new_tokens', type=int, default=96)
    ap.add_argument('--min_tok', type=int, default=6)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--out', default=os.path.join(
        HERE, 'outputs/problem_structure_fig/phd_cot_cf_tokens.jsonl'))
    run(ap.parse_args())
