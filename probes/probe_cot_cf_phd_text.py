# -*- coding: utf-8 -*-
"""probe_cot_cf_phd_text.py -- per-token TEXT-channel counterfactual probe for
phd, paired with the existing image-swap probe
(outputs/problem_structure_fig/phd_cot_cf_tokens.jsonl from probe_cot_cf_phd.py).

Same sample selection (seed 42, native >=6-token CoT subset) and the same
teacher-forced token positions as the existing probe. Because generation is
greedy/deterministic, the regenerated CoT is identical; this is verified by
joining to the existing output on `question` and comparing full_logp
(see --verify_vs).

Views on the SAME freshly generated CoT:
    full : own image + own question text       (identical to existing probe)
    img  : partner image + own question        -> delta_img_tok (existing axis)
    noq  : own image + NEUTRAL question text   -> delta_text_tok (NEW axis)
           (question replaced by "(no question provided)"; Instruction line
            and image unchanged)

Resumable (skips questions already in the output file).

    CUDA_VISIBLE_DEVICES=2 python probe_cot_cf_phd_text.py --n_samples 160 \
        --out outputs/cf_probe/phd_textcf_tokens.jsonl
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
import torch  # noqa: E402

NEUTRAL_Q = "(no question provided)"


def neutralize_q(question, scheme, key):
    """Return the neutralized question text under the given scheme."""
    if scheme == 'placeholder':
        return NEUTRAL_Q
    if scheme == 'empty':
        return ''
    if scheme == 'shuffle':
        import hashlib, random as _random
        words = question.split()
        if len(words) <= 1:
            return question
        seed = int(hashlib.sha256(str(key).encode('utf-8')).hexdigest(), 16) % (2 ** 31)
        r = _random.Random(seed)
        w = list(words)
        while True:
            r.shuffle(w)
            if w != words:
                break
        return ' '.join(w)
    raise ValueError(scheme)


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
    print(f'[phd-textcf] items with image on disk: {len(good)}/{len(data)}', flush=True)

    rng = random.Random(args.seed)

    def stored_len(d):
        r = by_q_scored.get((d.get('question') or '').strip())
        return len(r.get('token_ids') or []) if r else 0
    long_items = [d for d in good if stored_len(d) >= args.min_tok]
    print(f'[phd-textcf] items with native stored CoT >= {args.min_tok} tok: '
          f'{len(long_items)}', flush=True)
    pool = long_items if len(long_items) >= 20 else good
    rng.shuffle(pool)
    sample = pool[:args.n_samples] if not args.limit else pool[:args.limit]

    # existing image-swap outputs, for a determinism cross-check
    prev = {}
    if args.verify_vs and os.path.exists(args.verify_vs):
        for line in open(args.verify_vs):
            r = json.loads(line)
            prev.setdefault(r['question'], r)
        print(f'[phd-textcf] verify_vs: {len(prev)} existing rows', flush=True)

    done = set()
    if os.path.exists(args.out):
        for line in open(args.out):
            line = line.strip()
            if line:
                done.add(json.loads(line)['question'])
        print(f'[phd-textcf] resume: {len(done)} questions already done', flush=True)

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

    out_f = open(args.out, 'a')
    t0 = time.time()
    n_done = 0
    n_short = 0
    for si, item in enumerate(sample):
        q = item['question']
        if q in done:
            continue
        try:
            img = load_img(item)
            if img is None:
                continue
            up = get_user_prompt(q, 'phd', 'v1')
            up_noq = get_user_prompt(neutralize_q(q, args.scheme, q), 'phd', 'v1')
            assert up != up_noq
            if args.scheme == 'placeholder':
                assert NEUTRAL_Q in up_noq

            # --- regenerate the native observe-then-judge CoT (greedy) --------
            full_view = build_view_img(processor, model, system_prompt, up, img)
            input_len = full_view['input_ids'].shape[1]
            with torch.inference_mode():
                out_ids = thinker.generate(**full_view, max_new_tokens=args.max_new_tokens,
                                           do_sample=False, num_return_sequences=1)
            gen = out_ids[0, input_len:].tolist()
            while gen and gen[-1] in eos_ids:
                gen.pop()
            if len(gen) < args.min_tok:
                n_short += 1
                continue
            gen_ids = torch.tensor(gen, dtype=torch.long, device=model.device)
            toks = [tokenizer.decode([t], skip_special_tokens=True) for t in gen]
            cls = [token_class(t, toks[i - 1] if i else None) for i, t in enumerate(toks)]

            # --- contrast 1: a DIFFERENT item's image ------------------------
            partner = None
            for _ in range(20):
                cand = rng.choice(good)
                if cand is not item:
                    pim = load_img(cand)
                    if pim is not None:
                        partner = pim; break
            if partner is None:
                continue
            # NOTE: the original probe consumed rng draws in the same order
            # (generation is argmax, no rng), so the partner draw sequence
            # matches the existing run sample-by-sample.

            noq_view = build_view_img(processor, model, system_prompt, up_noq, img)
            img_view = build_view_img(processor, model, system_prompt, up, partner)

            lp_full = view_logp(thinker, full_view, gen_ids)
            lp_noq = view_logp(thinker, noq_view, gen_ids)
            lp_img = view_logp(thinker, img_view, gen_ids)
            d_text = lp_full - lp_noq
            d_img = lp_full - lp_img

            verify = None
            if q in prev:
                p = prev[q]
                if len(p.get('full_logp') or []) == len(lp_full):
                    verify = round(float(np.max(np.abs(np.asarray(p['full_logp']) - lp_full))), 4)

            out_f.write(json.dumps({
                'sample_i': si,
                'question': q,
                'n_tok': len(gen),
                'text': tokenizer.decode(gen, skip_special_tokens=True),
                'delta_tok': [float(x) for x in d_img],      # image-swap (existing axis)
                'delta_text_tok': [float(x) for x in d_text],  # NEW text axis
                'scheme': args.scheme,
                'classes': cls,
                'full_logp': [float(x) for x in lp_full],
                'verify_full_logp_maxabsdiff_vs_prev': verify,
            }) + '\n')
            out_f.flush()
            n_done += 1
        except Exception as e:
            print(f'[phd-textcf] sample {si} ERR {type(e).__name__}: {e}', flush=True)
            continue
        if (si + 1) % 10 == 0:
            el = time.time() - t0
            print(f'[phd-textcf] {si+1}/{len(sample)} done={n_done} short={n_short} '
                  f'{el/(si+1):.1f}s/it', flush=True)
    out_f.close()
    n = sum(1 for _ in open(args.out))
    print(f'[phd-textcf] wrote {args.out} ({n} rows)', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--scheme', choices=['placeholder', 'shuffle', 'empty'],
                    default='placeholder')
    ap.add_argument('--n_samples', type=int, default=160)
    ap.add_argument('--max_new_tokens', type=int, default=96)
    ap.add_argument('--min_tok', type=int, default=6)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--verify_vs', default=os.path.join(
        HERE, 'outputs/problem_structure_fig/phd_cot_cf_tokens.jsonl'))
    ap.add_argument('--out', default=os.path.join(
        HERE, 'outputs/cf_probe/phd_textcf_tokens.jsonl'))
    run(ap.parse_args())
