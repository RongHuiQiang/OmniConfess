# -*- coding: utf-8 -*-
"""bon_stacked_haloquest_qwen3.py — Qwen3-Omni port of bon_stacked_haloquest.py.

四层链路第 3 层：契约 v2 × 可答性门 × BoN 接地重排 @ HaloQuest，移植到 Qwen3。

读阶段2 合体文件（base_pred/v2_pred/flagged），按 C1 分支路由：
  R1 v1evade_keep     : base_pred 已回避 → 保留 base_pred
  R2 flagged_T3       : 门 flag → T3 问句回避模板
  R3 passthrough_bon  : v1 肯定 & 门未 flag → 在 v2 契约 prompt 下生成 4 候选，
                        接地分 Δ 选最接地（对照 = 换图）
分支决策仅依赖 base_pred/flagged/v2_pred（不用合体 pred 字段），故与阶段2 选的
A4/C1/A2 无关——四层的 no-BoN 基线 = C1 合体值。

适配（同 bon_grounding_qwen3）：shim 自别名 + Qwen3 媒体占位符 + Qwen3 eos（不用
stop_strings）；v2 系统 prompt 与 Qwen2.5 版逐字节同（system md5 见 RESULTS 行110）。

判据（负责人）：四层 > 阶段2 合体（C1 口径）；对标 Q2.5 39.87→40.06 的 BoN Δ 层 +0.19。

Usage:
    CUDA_VISIBLE_DEVICES=7 python bon_stacked_haloquest_qwen3.py
"""
import os, sys, time, json, random, argparse

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'OmniConfess', 'methods'))
sys.path.insert(0, HERE)
from shared import (  # noqa: E402
    load_dataset, load_media_for_item, resolve_media_path,
    get_system_prompt, get_user_prompt, parse_pred,
)
from shared.qwen3_shim import (  # noqa: E402
    load_model, build_inputs, prepare_for_thinker,
    build_user_content_with_media_tokens, make_messages,
)
from answerability_gate import EVADERE, TEMPLATES  # noqa: E402
from bon_grounding import (  # noqa: E402
    token_class, claim_spans, view_logp, rollout_scores, rank_value, set_seed,
)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("MMHALU_MODEL_ROOT", "/ultrafast_share/tsq/MODELS")

COMBINED_PATH = os.path.join(
    HERE, 'outputs', 'answerability_contract_qwen3',
    'haloquest_answerability_contract_gate_qwen3_full.jsonl')


def build_view(processor, model, system_prompt, user_prompt, dataset_name, image=None):
    user_content = build_user_content_with_media_tokens(None, None, image, user_prompt)
    messages = make_messages(system_prompt, user_content)
    raw = build_inputs(processor, model, messages, None, None, image, dataset_name)
    return prepare_for_thinker(raw, model)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--num_sampled', type=int, default=3)
    ap.add_argument('--temperature', type=float, default=0.7)
    ap.add_argument('--max_new_tokens', type=int, default=512)
    ap.add_argument('--rank_score', type=str, default='g_content',
                    choices=['g_content', 'g_mean_claim', 'g_worst_claim', 'g_traj_mean'])
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out_dir', default=os.path.join(HERE, 'outputs', 'haloquest_bon_stacked_qwen3'))
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    out_main = os.path.join(args.out_dir, 'haloquest_bon_stacked_qwen3_full.jsonl')

    combined = [json.loads(l) for l in open(COMBINED_PATH)]
    done_ids = set()
    if os.path.exists(out_main):
        for line in open(out_main):
            try:
                done_ids.add(json.loads(line)['id'])
            except Exception:
                pass
        print(f'[resume] skipping {len(done_ids)}', flush=True)

    data_path = os.path.join(HERE, 'OmniHalluBench', 'haloquest', 'haloquest.json')
    data_root = os.path.join(HERE, 'OmniHalluBench')
    data = load_dataset(data_path, data_root)
    by_id = {it['id']: it for it in data}
    pool_images = [it for it in data if it.get('image_path') and os.path.exists(
        resolve_media_path(it['image_path'], os.path.dirname(data_path), data_root))]

    processor, model = load_model()
    tokenizer = processor.tokenizer
    thinker = model.thinker
    im_end_id = tokenizer.convert_tokens_to_ids('<|im_end|>')
    endoftext_id = tokenizer.convert_tokens_to_ids('<|endoftext|>')
    eos_ids = [i for i in {im_end_id, endoftext_id} if i is not None]
    system_prompt = get_system_prompt('haloquest', 'v2')   # contract prompt

    rng = random.Random(args.seed + 7)
    rng_rand = random.Random(args.seed + 13)
    T3 = TEMPLATES['T3']
    t0 = time.time()
    n_done = 0
    for rec in combined:
        i = rec['id']
        if i in done_ids:
            continue
        item = by_id[i]
        v1_evade = bool(EVADERE.search(rec['base_pred'] or ''))
        flagged = bool(rec['flagged'])

        entry = {'id': i, 'question': rec['question'], 'gt': rec['gt'],
                 'group_key': rec.get('group_key'), 'dataset': 'haloquest',
                 'method': 'bon_stacked_qwen3', 'base_pred': rec['base_pred'],
                 'v2_pred': rec['v2_pred'], 'flagged': flagged,
                 'v1_evade': v1_evade, 'branch': None, 'rank_score': args.rank_score}

        if v1_evade:
            entry.update(branch='v1evade_keep', pred=rec['base_pred'],
                         trajectories=None, selected_rollout=None, random_sel_rollout=None)
        elif flagged:
            entry.update(branch='flagged_T3', pred=T3(rec['question']),
                         trajectories=None, selected_rollout=None, random_sel_rollout=None)
        else:
            entry['branch'] = 'passthrough_bon'
            cands, cand_ids, scores_list = [], [], []
            try:
                img = load_media_for_item(item, data_path, data_root)[2]
                user_prompt = get_user_prompt(item['question'], 'haloquest', 'v2')
                clean = build_view(processor, model, system_prompt, user_prompt,
                                   'haloquest', img)
                input_len = clean['input_ids'].shape[1]
                partner = None
                while partner is None:
                    cand_it = rng.choice(pool_images)
                    if cand_it['id'] != item['id']:
                        partner = cand_it
                pimg = load_media_for_item(partner, data_path, data_root)[2]
                ctrl = build_view(processor, model, system_prompt, user_prompt,
                                  'haloquest', pimg)
                for s, (do_sample, temp) in enumerate(
                        [(False, 1.0)] + [(True, args.temperature)] * args.num_sampled):
                    gk = dict(max_new_tokens=args.max_new_tokens,
                              do_sample=do_sample,
                              eos_token_id=eos_ids,          # Qwen3 eos
                              pad_token_id=endoftext_id)
                    if do_sample:
                        gk.update(temperature=temp)
                    with torch.inference_mode():
                        out = thinker.generate(**clean, **gk)
                    gen_ids = out[0][input_len:]
                    gen_ids = gen_ids[gen_ids != endoftext_id]
                    gen_ids = gen_ids[gen_ids != im_end_id]
                    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                    cands.append(text)
                    cand_ids.append(gen_ids.tolist())
                    torch.cuda.empty_cache()
                for gl in cand_ids:
                    if not gl:
                        scores_list.append(None)
                        continue
                    gt_t = torch.tensor(gl, dtype=torch.long, device=model.device)
                    delta = (view_logp(thinker, clean, gt_t)
                             - view_logp(thinker, ctrl, gt_t))
                    toks = [tokenizer.decode([t], skip_special_tokens=True) for t in gl]
                    cls = [token_class(t, toks[k - 1] if k else None)
                           for k, t in enumerate(toks)]
                    scores_list.append(rollout_scores(delta, claim_spans(toks, cls), cls))
                sel = max(range(len(cands)),
                          key=lambda k: rank_value(scores_list[k], args.rank_score))
                rand_sel = rng_rand.randrange(len(cands))
                entry.update(pred=parse_pred(cands[sel], 'haloquest'),
                             trajectories=[{'rollout': k, 'sampled': bool(k > 0),
                                            'token_ids': cand_ids[k],
                                            'pred': parse_pred(cands[k], 'haloquest'),
                                            'grounding_scores': scores_list[k]}
                                           for k in range(len(cands))],
                             selected_rollout=int(sel), random_sel_rollout=int(rand_sel),
                             partner=partner['id'])
            except Exception as e:
                print(f'  [ERROR] {i}: {e}', flush=True)
                torch.cuda.empty_cache()
                entry.update(pred=rec['v2_pred'], trajectories=None,
                             selected_rollout=None, random_sel_rollout=None,
                             branch='passthrough_bon_fallback_v2', error=str(e))

        with open(out_main, 'a') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        n_done += 1
        if n_done % 20 == 0:
            print(f'  [{n_done}] {time.time()-t0:.0f}s', flush=True)
        torch.cuda.empty_cache()

    ctrl_path = out_main.replace('_full.jsonl', '_random_sel.jsonl')
    with open(out_main) as fin, open(ctrl_path, 'w') as fout:
        for line in fin:
            r = json.loads(line)
            if r.get('random_sel_rollout') is not None and r.get('trajectories'):
                rs = r['random_sel_rollout']
                r['pred'] = r['trajectories'][rs]['pred']
            r['method'] = 'bon_stacked_qwen3_random_sel'
            fout.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'[done] {time.time()-t0:.0f}s -> {out_main} / {ctrl_path}', flush=True)


if __name__ == '__main__':
    main()
