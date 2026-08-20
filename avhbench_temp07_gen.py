# -*- coding: utf-8 -*-
"""
avhbench_temp07_gen.py — Vote-SC phrasing arms: two temperature-0.7 samples.

Pre-registered extension of the avhbench input-perturbation Vote-SC pool
(outputs/avhbench_vote_sc/, 5-arm token-F1 21.56). Hypothesis: the vote pool's
diversity is the binding constraint, so two *phrasing*-type arms (same 224/8-frame
input, same v1 prompt, temperature=0.7 sampling) may add diversity where the four
*input*-type arms (224/384/448/f16) are blind.

Per item this script produces TWO sampled generations (arm1 / arm2, distinct
seeds) and writes them to separate output dirs:

    outputs/avhbench_vote_sc/temp07_arm1/avhbench_shard{i}.jsonl
    outputs/avhbench_vote_sc/temp07_arm2/avhbench_shard{i}.jsonl

norm_logp is recomputed by a teacher-forced forward over the sampled tokens,
because generate(output_scores=True) under sampling returns *warped* scores
(temperature applied) which are not comparable with the greedy arms' raw
norm_logp. The teacher-forced pass yields the same raw-logit logprob definition
the greedy arms used (token_logprobs in methods/base_model.py).

Seeds are derived per (arm, item index) so a resumed shard reproduces identical
samples: arm1 = 4242*100000 + idx, arm2 = 8484*100000 + idx.

Usage (one shard per GPU):
    python avhbench_temp07_gen.py --shard 0 --shards 8
"""
import os, sys, time, json, argparse, random

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'OmniConfess', 'methods'))
from shared import (  # noqa: E402
    load_dataset, load_media_for_item, load_model, build_inputs,
    prepare_for_thinker, build_user_content_with_media_tokens, make_messages,
    get_system_prompt, get_user_prompt, make_result, save_result,
)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

REPO = '/home/tsq/rhq/omniconfess/OmniConfess'
DATA = REPO + '/OmniHalluBench/avhbench'
OUTROOT = REPO + '/outputs/avhbench_vote_sc'

ARM_SEEDS = {'temp07_arm1': 4242, 'temp07_arm2': 8484}
TEMPERATURE = 0.7
TOP_P = 1.0


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def raw_logp_teacher_forced(model, clean, input_len, gen_ids):
    """Raw (unwarped) per-token logprobs of the sampled continuation.

    One forward pass over [prompt + sampled tokens]; the logits that predict the
    sampled tokens are positions input_len-1 .. -2. Mirrors the greedy arms'
    token_logprobs() definition exactly (log_softmax over raw logits).
    """
    if gen_ids.dim() == 1:
        gen_ids = gen_ids.unsqueeze(0)
    full_ids = torch.cat([clean['input_ids'], gen_ids.to(clean['input_ids'].device)],
                         dim=1)
    attn = torch.ones_like(full_ids)
    fwd = {k: v for k, v in clean.items() if k in ('input_ids', 'attention_mask',
                                                   'pixel_values_videos',
                                                   'video_grid_thw',
                                                   'video_second_per_grid',
                                                   'pixel_values_images',
                                                   'pixel_values', 'image_grid_thw',
                                                   'feature_attention_mask',
                                                   'input_features')}
    fwd['input_ids'] = full_ids
    fwd['attention_mask'] = attn
    with torch.inference_mode():
        logits = model.thinker(**fwd).logits[0]           # [T, vocab]
    lp = torch.log_softmax(logits[input_len - 1:-1].float(), dim=-1)
    tgt = full_ids[0, input_len:]
    tok_lp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    return [round(x, 4) for x in tok_lp.tolist()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard', type=int, required=True)
    ap.add_argument('--shards', type=int, default=8)
    ap.add_argument('--max_new_tokens', type=int, default=512)
    ap.add_argument('--prompt_version', type=str, default='v1')
    ap.add_argument('--dry_run', action='store_true')
    args = ap.parse_args()

    data_path = f'{DATA}/avhbench_shard{args.shard}of{args.shards}.json'
    out_dirs = {a: f'{OUTROOT}/{a}' for a in ARM_SEEDS}
    for d in out_dirs.values():
        os.makedirs(d, exist_ok=True)

    print(f'[shard {args.shard}/{args.shards}] data={data_path}', flush=True)
    print(f'[config] temperature={TEMPERATURE} top_p={TOP_P} '
          f'seeds={ARM_SEEDS} prompt=v1 (224 / 8 frames, base settings)', flush=True)

    done = {a: set() for a in ARM_SEEDS}
    for a, d in out_dirs.items():
        p = f'{d}/avhbench_shard{args.shard}.jsonl'
        if os.path.exists(p):
            for line in open(p, encoding='utf-8'):
                try:
                    done[a].add(str(json.loads(line.strip())['id']))
                except Exception:
                    pass
            print(f'[resume][{a}] {len(done[a])} done', flush=True)

    print('[model] loading Qwen2.5-Omni...', flush=True)
    processor, model = load_model()
    tokenizer = processor.tokenizer

    data = load_dataset(data_path, REPO + '/OmniHalluBench/')
    if args.dry_run:
        data = data[:2]

    system_prompt = get_system_prompt('avhbench', args.prompt_version)

    t_start = time.time()
    for idx, item in enumerate(data):
        if all(str(item['id']) in done[a] for a in ARM_SEEDS):
            continue
        t0 = time.time()
        print(f'\n[{idx+1}/{len(data)}] id={item["id"]}', flush=True)

        try:
            pixel_values, audio_values, image_values = load_media_for_item(
                item, data_path, REPO + '/OmniHalluBench/')
            user_prompt = get_user_prompt(item['question'], 'avhbench',
                                          args.prompt_version,
                                          passage=item.get('passage'))
            user_content = build_user_content_with_media_tokens(
                pixel_values, audio_values, image_values, user_prompt)
            messages = make_messages(system_prompt, user_content)
            raw_inputs = build_inputs(processor, model, messages,
                                      pixel_values, audio_values, image_values,
                                      'avhbench')
            clean = prepare_for_thinker(raw_inputs, model)
            input_len = clean['input_ids'].shape[1]

            for arm, base_seed in ARM_SEEDS.items():
                if str(item['id']) in done[arm]:
                    continue
                seed = base_seed * 100000 + idx
                set_seed(seed)
                gen_kwargs = dict(
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    pad_token_id=tokenizer.eos_token_id,
                )
                with torch.inference_mode():
                    out = model.thinker.generate(**clean, **gen_kwargs)
                gen_ids = out[0][input_len:] if isinstance(out, torch.Tensor) \
                    else out.sequences[0][input_len:]
                response = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

                try:
                    logp_list = raw_logp_teacher_forced(model, clean, input_len,
                                                        gen_ids.cpu())
                    cum = float(np.sum(logp_list))
                    norm_lp = round(cum / (len(gen_ids) + 1e-8), 4)
                except Exception as e:
                    print(f'  [{arm}] logp recompute failed: {e}', flush=True)
                    logp_list, cum, norm_lp = None, None, None

                result = make_result(item, response, 'avhbench', 'temp07_sample')
                result.update(
                    arm=arm, temperature=TEMPERATURE, top_p=TOP_P, seed=seed,
                    token_ids=gen_ids.tolist(), logp_list=logp_list,
                    cumulative_logprob=cum, norm_logp=norm_lp,
                    input_len=int(input_len), output_tokens=int(len(gen_ids)),
                )
                save_result(result, f'{out_dirs[arm]}/avhbench_shard{args.shard}.jsonl')
                print(f'  [{arm}] seed={seed} tokens={len(gen_ids)} '
                      f'norm_logp={norm_lp} | {response[:100]!r}', flush=True)

            del clean, raw_inputs
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f'  [ERROR] {e}', flush=True)
            for arm in ARM_SEEDS:
                if str(item['id']) not in done[arm]:
                    result = make_result(item, '', 'avhbench', 'temp07_sample')
                    result.update(arm=arm, temperature=TEMPERATURE, seed=None,
                                  norm_logp=None, output_tokens=0, error=str(e)[:300])
                    save_result(result, f'{out_dirs[arm]}/avhbench_shard{args.shard}.jsonl')

        torch.cuda.empty_cache()

    print(f'\n[done] shard {args.shard} in {time.time()-t_start:.1f}s', flush=True)


if __name__ == '__main__':
    main()
