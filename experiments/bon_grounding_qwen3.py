# -*- coding: utf-8 -*-
"""bon_grounding_qwen3.py — Qwen3-Omni port of bon_grounding.py.

Best-of-N assertion-grounding rerank，移植到 Qwen3-Omni。生成 1 greedy + 3
temperature-0.7 候选，各在完整图 / 换图对照下 teacher-force 打接地分 Δ，输出
最接地候选。产出 N=4 候选池（供阶段3 特征组合 + 特征探针复用）。

适配（低风险，同门探针）：
  * 模型/输入走 shared.qwen3_shim（.thinker 自别名、Qwen3 媒体占位符、标量 fps、eos pin）
  * generate 用 Qwen3 eos_token_id=[im_end,endoftext]（不用 Qwen2.5 的
    stop_strings=['<end_of_reasoning>']，Qwen3 无此 token；照 base_model_qwen3_omni.py）
  * view_logp / claim_spans / rollout_scores 复用 bon_grounding.py（backbone-agnostic）

Usage:
    CUDA_VISIBLE_DEVICES=7 python bon_grounding_qwen3.py --datasets haloquest \
        --data_path ./OmniHalluBench/haloquest/annotations/haloquest.json \
        --output_dir outputs/haloquest_bon_grounding_qwen3 \
        --file_name haloquest_bon_grounding_qwen3
"""
import os, sys, time, argparse, random, json

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'OmniConfess', 'methods'))
sys.path.insert(0, HERE)
from shared import (  # noqa: E402
    load_dataset, load_media_for_item, resolve_media_path,
    get_system_prompt, get_user_prompt,
    parse_pred, make_result, save_result,
)
from shared.qwen3_shim import (  # noqa: E402
    load_model, build_inputs, prepare_for_thinker,
    build_user_content_with_media_tokens, make_messages,
)
# backbone-agnostic scoring machinery
from bon_grounding import (  # noqa: E402
    token_class, claim_spans, view_logp, rollout_scores, rank_value, set_seed,
)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault("MMHALU_MODEL_ROOT", "/ultrafast_share/tsq/MODELS")


def build_view(processor, model, system_prompt, user_prompt, dataset_name, image=None):
    user_content = build_user_content_with_media_tokens(None, None, image, user_prompt)
    messages = make_messages(system_prompt, user_content)
    raw = build_inputs(processor, model, messages, None, None, image, dataset_name)
    return prepare_for_thinker(raw, model)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets',       type=str, required=True)
    parser.add_argument('--data_path',      type=str, required=True)
    parser.add_argument('--data_root',      type=str, default='./OmniHalluBench/')
    parser.add_argument('--output_dir',     type=str, default='./outputs')
    parser.add_argument('--file_name',      type=str, default='bon_grounding_qwen3_out')
    parser.add_argument('--time_path',      type=str, default='./outputs/time')
    parser.add_argument('--seed',           type=int, default=42)
    parser.add_argument('--num_sampled',    type=int, default=3)
    parser.add_argument('--temperature',    type=float, default=0.7)
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--prompt_version', type=str, default='v1', choices=['v0','v1','v2','v3'])
    parser.add_argument('--rank_score',     type=str, default='g_content',
                        choices=['g_content', 'g_mean_claim', 'g_worst_claim', 'g_traj_mean'])
    parser.add_argument('--dry_run',        action='store_true')
    parser.add_argument('--limit',          type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.time_path, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.file_name + '.jsonl')

    done_ids = set()
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line.strip())['id'])
                except Exception:
                    pass
        print(f'[resume] skipping {len(done_ids)} samples', flush=True)

    print('[model] loading Qwen3-Omni thinker...', flush=True)
    processor, model = load_model()
    tokenizer = processor.tokenizer
    thinker = model.thinker            # shim self-alias -> model
    im_end_id = tokenizer.convert_tokens_to_ids('<|im_end|>')
    endoftext_id = tokenizer.convert_tokens_to_ids('<|endoftext|>')
    eos_ids = [i for i in {im_end_id, endoftext_id} if i is not None]

    data = load_dataset(args.data_path, args.data_root)
    if args.dry_run:
        data = data[:3]
    if args.limit:
        data = data[:args.limit]
    ds = args.datasets
    system_prompt = get_system_prompt(ds, args.prompt_version)

    pool_passages = [it['passage'] for it in data if it.get('passage')]
    pool_images = [it for it in data if it.get('image_path') and os.path.exists(
        resolve_media_path(it['image_path'], os.path.dirname(args.data_path), args.data_root))]
    print(f'[pool] passages={len(pool_passages)} images={len(pool_images)}', flush=True)

    rng = random.Random(args.seed + 7)
    rng_rand = random.Random(args.seed + 13)
    all_input_tokens = all_output_tokens = 0
    start = time.time()

    for idx, item in enumerate(data):
        if item['id'] in done_ids:
            continue
        t0 = time.time()
        print(f'\n[{idx+1}/{len(data)}] id={item["id"]}', flush=True)

        candidate_texts, candidate_ids, candidate_preds = [], [], []
        scores_list, sel_err = [], None
        partner_id = None
        try:
            pixel_values, audio_values, image_values = load_media_for_item(
                item, args.data_path, args.data_root)
            image = image_values if ds.lower().startswith('halo') else None
            user_prompt = get_user_prompt(item['question'], ds, args.prompt_version,
                                          passage=item.get('passage'))
            clean = build_view(processor, model, system_prompt, user_prompt, ds, image)
            input_len = clean['input_ids'].shape[1]
            all_input_tokens += input_len * (1 + args.num_sampled)

            if ds.lower() == 'ragtruth':
                partner_psg = None
                while partner_psg is None:
                    cand_p = rng.choice(pool_passages)
                    if cand_p != (item.get('passage') or ''):
                        partner_psg = cand_p
                up_ctrl = get_user_prompt(item['question'], ds, args.prompt_version,
                                          passage=partner_psg)
                ctrl = build_view(processor, model, system_prompt, up_ctrl, ds, None)
                partner_id = 'psg-borrow'
            else:
                partner = None
                while partner is None:
                    cand_it = rng.choice(pool_images)
                    if cand_it['id'] != item['id']:
                        partner = cand_it
                pimg = load_media_for_item(partner, args.data_path, args.data_root)[2]
                ctrl = build_view(processor, model, system_prompt, user_prompt, ds, pimg)
                partner_id = partner['id']

            _t_gen_start = time.perf_counter()
            gen_specs = [(False, 1.0)] + [(True, args.temperature)] * args.num_sampled
            for s, (do_sample, temp) in enumerate(gen_specs):
                gen_kwargs = dict(
                    max_new_tokens=args.max_new_tokens,
                    do_sample=do_sample,
                    eos_token_id=eos_ids,           # Qwen3 eos fix
                    pad_token_id=endoftext_id,
                )
                if do_sample:
                    gen_kwargs.update(temperature=temp)
                with torch.inference_mode():
                    out = thinker.generate(**clean, **gen_kwargs)
                gen_ids = out[0][input_len:]
                gen_ids = gen_ids[gen_ids != endoftext_id]
                # also strip trailing im_end if present
                gen_ids = gen_ids[gen_ids != im_end_id]
                text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                candidate_texts.append(text)
                candidate_ids.append(gen_ids.tolist())
                candidate_preds.append(parse_pred(text, ds))
                all_output_tokens += len(gen_ids)
                torch.cuda.empty_cache()
            _t_gen = time.perf_counter() - _t_gen_start

            _t_sel_start = time.perf_counter()
            for gen_ids_list in candidate_ids:
                if not gen_ids_list:
                    scores_list.append(None)
                    continue
                gen_t = torch.tensor(gen_ids_list, dtype=torch.long, device=model.device)
                lp_full = view_logp(thinker, clean, gen_t)
                lp_ctrl = view_logp(thinker, ctrl, gen_t)
                delta = lp_full - lp_ctrl
                toks = [tokenizer.decode([t], skip_special_tokens=True) for t in gen_ids_list]
                cls = [token_class(t, toks[i - 1] if i else None) for i, t in enumerate(toks)]
                spans = claim_spans(toks, cls)
                scores_list.append(rollout_scores(delta, spans, cls))
            sel = max(range(len(candidate_ids)),
                      key=lambda i: rank_value(scores_list[i], args.rank_score))
            rand_sel = rng_rand.randrange(len(candidate_ids))
            _t_sel = time.perf_counter() - _t_sel_start
            sel_err = None
        except Exception as e:
            sel_err = str(e)
            print(f'  [ERROR] {e}', flush=True)
            torch.cuda.empty_cache()
            if not candidate_texts:
                candidate_texts, candidate_preds = [''], ['']
                candidate_ids, scores_list = [[]], [None]
                sel = rand_sel = 0
            else:
                sel = rand_sel = 0
            _t_gen = _t_sel = 0.0

        elapsed = time.time() - t0
        final_pred = candidate_preds[sel]
        print(f'  sel={sel}/{len(candidate_texts)-1} pred[:80]={final_pred[:80]!r} '
              f'err={sel_err} time={elapsed:.1f}s', flush=True)

        result = make_result(item, candidate_texts[sel], ds, 'bon_grounding_qwen3',
                             time_sec=elapsed,
                             cot=json.dumps(candidate_texts, ensure_ascii=False))
        result['pred'] = final_pred
        result['generation_time_seconds'] = round(_t_gen, 4)
        result['selection_time_seconds'] = round(_t_sel, 4)
        result['record_process'] = True
        result['trajectories'] = [
            {'rollout': i, 'sampled': bool(i > 0), 'token_ids': candidate_ids[i],
             'pred': candidate_preds[i], 'grounding_scores': scores_list[i]}
            for i in range(len(candidate_texts))]
        result['selected_rollout'] = int(sel)
        result['random_sel_rollout'] = int(rand_sel)
        result['rank_score'] = args.rank_score
        result['partner'] = partner_id
        result['error'] = sel_err
        save_result(result, output_path)
        torch.cuda.empty_cache()

    total_time = time.time() - start
    with open(os.path.join(args.time_path, args.file_name + '.txt'), 'w') as f:
        f.write(f'total_time: {total_time:.2f}s\n')
        f.write(f'greedy + num_sampled: {args.num_sampled} @ T={args.temperature}\n')
        f.write(f'total_input_tokens: {all_input_tokens}\n')
        f.write(f'total_output_tokens: {all_output_tokens}\n')

    ctrl_path = os.path.join(args.output_dir, args.file_name + '_random_sel.jsonl')
    n_lines = 0
    with open(output_path) as fin, open(ctrl_path, 'w') as fout:
        for line in fin:
            r = json.loads(line)
            rs = r.get('random_sel_rollout', 0)
            trajs = r.get('trajectories') or []
            if rs < len(trajs):
                r['raw_output'] = trajs[rs].get('pred', '')
                r['pred'] = trajs[rs].get('pred', '')
            r['method'] = 'bon_grounding_qwen3_random_sel'
            fout.write(json.dumps(r, ensure_ascii=False) + '\n')
            n_lines += 1
    print(f'\n[done] {total_time:.1f}s | {output_path} | control arm {n_lines} lines -> {ctrl_path}')


if __name__ == '__main__':
    main()
