# -*- coding: utf-8 -*-
"""probe_cf_avhbench_text.py -- per-token TEXT-channel (noquestion)
counterfactual probe for AVHBench, paired with the existing bigN modality probe
(outputs/cf_probe/avh_probe_bigN_shard0of1.jsonl from probe_cf_avhbench.py).

Same sample selection as the bigN run (stratified_sample(seed=42, n=400) over
the scored base_model file, shard 0 of 1 -> identical sample list) and the same
teacher-forced token positions (stored base greedy token_ids).

NOTE: avhbench's question is a single FIXED captioning instruction
("Describe what you see and hear in a single sentence."), identical for all
645 items. The text ablation therefore replaces this fixed instruction with a
neutral placeholder -- delta_text measures the task-text channel (incl. its
see/hear modality steering), constant-in-form across samples:

    full : own video + own audio + real instruction
    noq  : own video + own audio + NEUTRAL instruction
    delta_text[i] = logP(tok_i|full) - logP(tok_i|noq)

Resumable (skips ids already in output). Output rows join to the bigN rows on
(id, gt, pred) for (delta_video, delta_text) / (delta_audio, delta_text) pairs.

    CUDA_VISIBLE_DEVICES=2 python probe_cf_avhbench_text.py --limit 200 \
        --out outputs/cf_probe/avh_probe_text_shard0of1.jsonl
"""
import os, sys, json, argparse, time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'OmniConfess', 'methods'))

os.environ.setdefault("MMHALU_MODEL_ROOT", "/ultrafast_share/tsq/MODELS")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from shared import (  # noqa: E402
    load_dataset, load_media_for_item, load_model, build_inputs,
    prepare_for_thinker, build_user_content_with_media_tokens, make_messages,
    get_system_prompt, get_user_prompt,
)
from probe_cf_avhbench import token_class, token_f1, stratified_sample, \
    view_logp, retry  # noqa: E402

NEUTRAL_Q = "Describe the content in a single sentence."


def neutralize_q(question, scheme, key):
    """Return the neutralized instruction text under the given scheme."""
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


def build_view(processor, model, system_prompt, user_prompt,
               pixel_values, audio_values):
    prefix = ''
    if pixel_values is not None:
        prefix += "<|vision_bos|><|VIDEO|><|vision_eos|>\n"
    if audio_values is not None:
        prefix += "<|audio_bos|><|AUDIO|><|audio_eos|>\n"
    messages = make_messages(system_prompt, prefix + user_prompt)
    raw = retry(lambda: build_inputs(processor, model, messages,
                                     pixel_values, audio_values, None, 'avhbench'),
                tag='build_inputs')
    return prepare_for_thinker(raw, model)


def run(args):
    recs = [json.loads(l) for l in open(args.scored)]
    recs = [r for r in recs if r.get('token_ids')]
    for r in recs:
        r['_f1'] = token_f1(r['pred'], r['gt'])
    # SAME selection as the bigN run: stratified_sample(..., 400, seed=42),
    # shard 0 of 1, then --limit slices the head.
    sample = stratified_sample(recs, args.big_n, args.seed)
    if args.limit:
        sample = sample[:args.limit]
    print(f'[avh-text] {len(sample)} samples '
          f'(F1 mean {np.mean([s["_f1"] for s in sample]):.3f})', flush=True)

    data_path = os.path.join(HERE, 'OmniHalluBench', 'avhbench', 'avhbench.json')
    data_root = os.path.dirname(data_path)
    data = load_dataset(data_path, data_root)
    by_qg = {(d['question'], d['gt']): d for d in data}

    processor, model = load_model()
    tokenizer = processor.tokenizer
    thinker = model.thinker
    system_prompt = get_system_prompt('avhbench', 'v1')

    done = set()
    if os.path.exists(args.out):
        for line in open(args.out):
            line = line.strip()
            if line:
                done.add(json.loads(line)['id'])
        print(f'[avh-text] resume: {len(done)} ids already done', flush=True)

    fout = open(args.out, 'a')
    t_start = time.time()
    n_new = 0
    for i, rec in enumerate(sample):
        if rec['id'] in done:
            continue
        item = by_qg.get((rec['question'], rec['gt']))
        if item is None:
            continue
        try:
            pixel_values, audio_values, _ = retry(
                lambda: load_media_for_item(item, data_path, data_root),
                tag='load_media')
            if pixel_values is None or audio_values is None:
                continue

            gen_ids = torch.tensor(rec['token_ids'], dtype=torch.long,
                                   device=model.device)
            up_full = get_user_prompt(item['question'], 'avhbench', 'v1')
            up_noq = get_user_prompt(
                neutralize_q(item['question'], args.scheme, rec['id']),
                'avhbench', 'v1')
            assert up_full != up_noq
            if args.scheme == 'placeholder':
                assert NEUTRAL_Q in up_noq

            views = {}
            for name, up in (('full', up_full), ('noq', up_noq)):
                clean = build_view(processor, model, system_prompt, up,
                                   pixel_values, audio_values)
                views[name] = retry(lambda c=clean: view_logp(thinker, c, gen_ids),
                                    tag=f'forward_{name}')
                torch.cuda.empty_cache()

            d_text = views['full'] - views['noq']
            base_lp = np.array(rec['logp_list'], dtype=np.float64)
            repro_err = (float(np.max(np.abs(views['full'] - base_lp)))
                         if len(base_lp) == len(views['full']) else float('nan'))

            toks = [tokenizer.decode([t]) for t in rec['token_ids']]
            cls = [token_class(t, toks[j - 1] if j else None)
                   for j, t in enumerate(toks)]

            fout.write(json.dumps({
                'id': rec['id'], 'gt': rec['gt'], 'pred': rec['pred'],
                'f1': rec['_f1'], 'pred_words': len(rec['pred'].split()),
                'tokens': toks, 'classes': cls,
                'lp_full': views['full'].round(4).tolist(),
                'lp_noq': views['noq'].round(4).tolist(),
                'delta_text': d_text.round(4).tolist(),
                'scheme': args.scheme,
                'repro_err': round(repro_err, 4),
            }, ensure_ascii=False) + '\n')
            fout.flush()
            n_new += 1
        except Exception as e:
            print(f'[avh-text ERROR] id={rec["id"]}: {type(e).__name__}: {e}',
                  flush=True)
        if (i + 1) % 5 == 0:
            print(f'[avh-text {i+1}/{len(sample)}] new={n_new} '
                  f'{time.time()-t_start:.0f}s mem='
                  f'{torch.cuda.max_memory_allocated()/2**30:.1f}G', flush=True)
        torch.cuda.empty_cache()

    fout.close()
    n = sum(1 for _ in open(args.out))
    print(f'[avh-text done] -> {args.out} ({n} rows, {n_new} new)', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--scheme', choices=['placeholder', 'shuffle', 'empty'],
                    default='placeholder')
    ap.add_argument('--scored', default=os.path.join(
        HERE, 'outputs/avhbench_base_model/avhbench_base_model_full.jsonl'))
    ap.add_argument('--big_n', type=int, default=400,
                    help='size of the stratified sample to reproduce the bigN run')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--limit', type=int, default=200)
    ap.add_argument('--out', default=os.path.join(
        HERE, 'outputs/cf_probe/avh_probe_text_shard0of1.jsonl'))
    run(ap.parse_args())
