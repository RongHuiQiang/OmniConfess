# -*- coding: utf-8 -*-
"""probe_cot_cf_cmm_text.py -- per-token TEXT-channel counterfactual probe for
cmm, paired with the existing CoT per-token probe
(outputs/problem_structure_fig/cmm_cot_cf_tokens.jsonl from probe_cot_cf_cmm.py).

Same sample selection (stratified by hallucination_type, seed 42, n=140), the
same constructed describe-then-judge prompt, and the same teacher-forced token
positions. Generation is greedy/deterministic so the regenerated CoT is
identical; verified by joining on `question` and comparing full_logp
(--verify_vs).

Views on the SAME freshly generated CoT (video/audio views recomputed so all
three axes sit on one row):
    full  : own video + own audio + own question
    vswap : partner video + own audio           -> delta_v (existing axis)
    aswap : own video + partner audio           -> delta_a (existing axis)
    noq   : own video + own audio + NEUTRAL question text -> delta_t (NEW axis)
            (question replaced by "(no question provided)"; task line unchanged)

Resumable (skips (question, sample_i) already in output).

    CUDA_VISIBLE_DEVICES=4 python probe_cot_cf_cmm_text.py --n_samples 140 \
        --out outputs/cf_probe/cmm_textcf_tokens.jsonl
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
    get_system_prompt, resolve_media_path,
)
from shared.data_loader import load_video_frames, load_audio_data  # noqa: E402
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


DATA = os.path.join(HERE, 'OmniHalluBench/cmm/annotations/cmm.json')
JD = os.path.dirname(DATA)

SYSTEM_PROMPT = (
    "You are an audio-visual reporter. First describe, in one sentence, "
    "what you actually see in the video and hear in the audio. Then give a "
    "final judgment.\n\n"
    "Description: <one sentence of what you see and hear>\n"
    "Judgment: \\boxed{Yes} or \\boxed{No}"
)


def load_av(item):
    vp = resolve_media_path(item['media'].get('video'), JD, JD)
    ap = resolve_media_path(item['media'].get('audio'), JD, JD)
    px = load_video_frames(vp) if vp else None
    au = load_audio_data(ap) if ap else None
    return px, au


def build_view_av(processor, model, user_prompt, px, au):
    uc = build_user_content_with_media_tokens(px, au, None, user_prompt)
    messages = make_messages(SYSTEM_PROMPT, uc)
    raw = build_inputs(processor, model, messages, px, au, None, 'cmm')
    return prepare_for_thinker(raw, model)


def run(args):
    data = json.load(open(DATA))
    good = []
    for d in data:
        m = d.get('media') or {}
        vp = resolve_media_path(m.get('video'), JD, JD)
        ap = resolve_media_path(m.get('audio'), JD, JD)
        if vp and ap and os.path.exists(vp) and os.path.exists(ap):
            good.append(d)
    print(f'[cmm-textcf] items with both media on disk: {len(good)}/{len(data)}', flush=True)

    rng = random.Random(args.seed)
    buckets = {}
    for d in good:
        k = ((d.get('facets') or {}).get('hallucination_type')) or 'na'
        buckets.setdefault(k, []).append(d)
    for k in buckets:
        rng.shuffle(buckets[k])
    order = list(buckets.keys())
    sample = []
    while len(sample) < args.n_samples and any(buckets[k] for k in order):
        for k in order:
            if buckets[k] and len(sample) < args.n_samples:
                sample.append(buckets[k].pop())
    if args.limit:
        sample = sample[:args.limit]

    prev = {}
    if args.verify_vs and os.path.exists(args.verify_vs):
        for line in open(args.verify_vs):
            r = json.loads(line)
            prev.setdefault((r.get('question'), r.get('sample_i')), r)
        print(f'[cmm-textcf] verify_vs: {len(prev)} existing rows', flush=True)

    done = set()
    if os.path.exists(args.out):
        for line in open(args.out):
            line = line.strip()
            if line:
                r = json.loads(line)
                done.add((r['question'], r['sample_i']))
        print(f'[cmm-textcf] resume: {len(done)} already done', flush=True)

    processor, model = load_model()
    tokenizer = processor.tokenizer
    thinker = model.thinker
    eos_ids = set(x for x in [tokenizer.eos_token_id,
                              getattr(tokenizer, 'pad_token_id', None)] if x is not None)
    try:
        eos_ids.add(tokenizer.convert_tokens_to_ids('<|im_end|>'))
    except Exception:
        pass

    out_f = open(args.out, 'a')
    t0 = time.time()
    n_done = 0
    for si, item in enumerate(sample):
        key = (item['question'], si)
        if key in done:
            continue
        try:
            px, au = load_av(item)
            if px is None or au is None:
                continue
            up = ("Task: Analyze the video and audio to answer the question.\n"
                  f"Question: {item['question']}\n"
                  "Description:")
            nq = neutralize_q(item['question'], args.scheme,
                              (item['question'], si))
            up_noq = ("Task: Analyze the video and audio to answer the question.\n"
                      f"Question: {nq}\n"
                      "Description:")
            assert up != up_noq

            full_view = build_view_av(processor, model, up, px, au)
            input_len = full_view['input_ids'].shape[1]
            with torch.inference_mode():
                out_ids = thinker.generate(**full_view, max_new_tokens=args.max_new_tokens,
                                           do_sample=False, num_return_sequences=1)
            gen = out_ids[0, input_len:].tolist()
            while gen and gen[-1] in eos_ids:
                gen.pop()
            if len(gen) < args.min_tok:
                continue
            gen_ids = torch.tensor(gen, dtype=torch.long, device=model.device)
            toks = [tokenizer.decode([t], skip_special_tokens=True) for t in gen]
            cls = [token_class(t, toks[i - 1] if i else None) for i, t in enumerate(toks)]

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
            vswap_view = build_view_av(processor, model, up, ppx, au)
            aswap_view = build_view_av(processor, model, up, px, pau)
            noq_view = build_view_av(processor, model, up_noq, px, au)

            lp_full = view_logp(thinker, full_view, gen_ids)
            lp_vswap = view_logp(thinker, vswap_view, gen_ids)
            lp_aswap = view_logp(thinker, aswap_view, gen_ids)
            lp_noq = view_logp(thinker, noq_view, gen_ids)
            delta_v = lp_full - lp_vswap
            delta_a = lp_full - lp_aswap
            delta_t = lp_full - lp_noq

            verify = None
            p = prev.get((item['question'], si))
            if p and len(p.get('full_logp') or []) == len(lp_full):
                verify = round(float(np.max(np.abs(np.asarray(p['full_logp']) - lp_full))), 4)

            out_f.write(json.dumps({
                'sample_i': si,
                'question': item['question'],
                'halluc_type': (item.get('facets') or {}).get('hallucination_type'),
                'n_tok': len(gen),
                'text': tokenizer.decode(gen, skip_special_tokens=True),
                'delta_v': [float(x) for x in delta_v],
                'delta_a': [float(x) for x in delta_a],
                'delta_text_tok': [float(x) for x in delta_t],
                'scheme': args.scheme,
                'classes': cls,
                'full_logp': [float(x) for x in lp_full],
                'verify_full_logp_maxabsdiff_vs_prev': verify,
            }) + '\n')
            out_f.flush()
            n_done += 1
        except Exception as e:
            print(f'[cmm-textcf] sample {si} ERR {type(e).__name__}: {e}', flush=True)
            continue
        if (si + 1) % 10 == 0:
            print(f'[cmm-textcf] {si+1}/{len(sample)} done={n_done} '
                  f'{(time.time()-t0)/(si+1):.1f}s/it', flush=True)
    out_f.close()
    n = sum(1 for _ in open(args.out))
    print(f'[cmm-textcf] wrote {args.out} ({n} rows)', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--scheme', choices=['placeholder', 'shuffle', 'empty'],
                    default='placeholder')
    ap.add_argument('--n_samples', type=int, default=140)
    ap.add_argument('--max_new_tokens', type=int, default=64)
    ap.add_argument('--min_tok', type=int, default=6)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--verify_vs', default=os.path.join(
        HERE, 'outputs/problem_structure_fig/cmm_cot_cf_tokens.jsonl'))
    ap.add_argument('--out', default=os.path.join(
        HERE, 'outputs/cf_probe/cmm_textcf_tokens.jsonl'))
    run(ap.parse_args())
