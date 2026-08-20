# -*- coding: utf-8 -*-
"""probe_bon_features.py — BoN 重排的三特征判别力探针（文献武装升级，第一阶段）.

LITERATURE_FINDINGS.md §二-1（VISTA/VAS）+ §二-3（TTSP critical-token entropy）：

  F1  Δ 接地分（已落盘，bon_grounding.py 的 g_content）
  F2  临界熵 CE：每生成步 top-k(=5) 重归一截断熵 H_t，轨迹分 = k'(=5) 个最大
      H_t 的均值（order statistic，非序列平均——序列平均已被 TRAJECTORY_ANALYSIS
      §5.2 判死，norm_logp pearson 0.104）
  F3  VAS：中间层（28 层取第 14 层；sanity 附带 9/23 层）生成 token 对证据 token
      的注意力占比。haloquest 证据 = 图像 token 区间；ragtruth（纯文本）同构版 =
      passage token 区间（VISTA 的 λ_ins 同位替换）。

全部特征来自同一次 teacher-forced 前向（logits 给 CE，attentions 给 VAS），
作用于已落盘的 BoN 候选池（outputs/{ds}_bon_grounding/，v1 prompt，N=4）。
Δ 复用池内已存的 grounding_scores，无重算。

判据（预注册）：dev(100) 上选权的组合分，test 上选中候选 F1 > 纯 Δ 版 +1.0
→ 启动在线版重跑；否则报告最优权重下的边际。

Usage:
    CUDA_VISIBLE_DEVICES=2 python probe_bon_features.py --dataset haloquest \
        --limit 50 --out outputs/bon_probe/haloquest_feat_sanity.json   # sanity
    CUDA_VISIBLE_DEVICES=2 python probe_bon_features.py --dataset ragtruth \
        --out outputs/bon_probe/ragtruth_bon_feat.json                  # full
"""
import os, sys, json, argparse, time, string

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'OmniConfess', 'methods'))
from shared import (  # noqa: E402
    load_dataset, load_media_for_item,
    load_model, build_inputs, prepare_for_thinker,
    build_user_content_with_media_tokens, make_messages,
    get_system_prompt, get_user_prompt,
)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("MMHALU_MODEL_ROOT", "/ultrafast_share/tsq/MODELS")

TOPK = 5        # truncated-entropy top-k (TTSP)
KSTAT = 5       # order statistic: mean of the KSTAT largest H_t
VAS_LAYERS = (9, 14, 23)   # 28-layer thinker; 14 = middle (primary)


def _tokens(s):
    s = str(s).lower().translate(str.maketrans("", "", string.punctuation))
    STOP = set([
        "a", "an", "the", "and", "or", "but", "if", "because", "as", "what",
        "which", "this", "that", "these", "those", "then", "just", "so", "than",
        "such", "both", "through", "about", "for", "is", "are", "was", "were",
        "of", "while", "during", "to", "in", "on", "at", "by", "with", "it", "there"])
    return [w for w in s.split() if w not in STOP and w.strip()]


def token_f1(pred, gt):
    from collections import Counter
    p_t, g_t = _tokens(pred), _tokens(gt)
    if not p_t or not g_t:
        return 0.0
    common = Counter(p_t) & Counter(g_t)
    hit = sum(common.values())
    p = hit / len(p_t)
    r = hit / len(g_t)
    return (2 * p * r) / (p + r) if (p + r) > 0 else 0.0


@torch.inference_mode()
def forward_with_feats(thinker, clean, gen_ids, ev_mask, content_pos):
    """One teacher-forced forward with attentions.

    Returns dict with critical-entropy stats and VAS at VAS_LAYERS.
    ev_mask: bool tensor [input_len] marking evidence-token positions in the
    prompt (image tokens / passage span).
    content_pos: bool tensor [n_gen] marking content|num generated tokens.
    """
    input_len = clean['input_ids'].shape[1]
    full = dict(clean)
    full['input_ids'] = torch.cat([clean['input_ids'], gen_ids.unsqueeze(0)], dim=1)
    full['attention_mask'] = torch.ones_like(full['input_ids'])
    out = thinker(**full, use_cache=False, output_attentions=True)
    logits = out.logits[0].float()          # [L, V]
    # positions predicting gen token i are input_len-1+i
    pos = torch.arange(input_len - 1, full['input_ids'].shape[1] - 1,
                       device=logits.device)
    lg = logits[pos]                          # [n_gen, V]
    top = torch.topk(lg, TOPK, dim=-1).values
    pk = torch.softmax(top, dim=-1)
    H = -(pk * torch.log(pk + 1e-9)).sum(-1).cpu().numpy()   # [n_gen]
    H_sorted = np.sort(H)[::-1]
    ce = {
        'CE_top5mean': float(np.mean(H_sorted[:KSTAT])),
        'CE_max': float(H_sorted[0]),
        'H_per_tok': [round(float(x), 4) for x in H],
    }
    # VAS from attentions
    vas = {}
    for li, att in enumerate(out.attentions):
        if li not in VAS_LAYERS:
            continue
        a = att[0].float().mean(0)            # [L, L] heads averaged
        rows = a[pos]                          # [n_gen, L]
        denom = rows.sum(-1).clamp(min=1e-9)
        frac = (rows[:, :input_len][:, ev_mask].sum(-1) / denom).cpu().numpy()
        cp = content_pos.cpu().numpy().astype(bool)
        vas[li] = {
            'vas_all': float(np.mean(frac)),
            'vas_content': float(np.mean(frac[cp])) if cp.any() else None,
        }
    del out, logits
    return {'CE': ce, 'VAS': vas}


def build_view(processor, model, system_prompt, user_prompt, dataset_name, image=None):
    user_content = build_user_content_with_media_tokens(None, None, image, user_prompt)
    messages = make_messages(system_prompt, user_content)
    raw = build_inputs(processor, model, messages, None, None, image, dataset_name)
    return prepare_for_thinker(raw, model)


def evidence_mask(tokenizer, processor, clean, dataset, item, messages):
    """Bool mask over prompt positions: image tokens (haloquest) / passage span
    (ragtruth). For ragtruth we re-tokenize the chat-templated string WITH
    offsets and require identity with the processor's input_ids (checked)."""
    ids = clean['input_ids'][0].tolist()
    if dataset == 'haloquest':
        # Omni thinker: media placeholders appear as <|IMAGE|> tokens in
        # input_ids (pixel_values merged at those positions)
        img_id = tokenizer.convert_tokens_to_ids('<|IMAGE|>')
        assert img_id is not None and img_id >= 0, 'no <|IMAGE|> token id'
        return torch.tensor([t == img_id for t in ids], device=clean['input_ids'].device), None
    # ragtruth: passage span via offsets
    p_str = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    p_str = p_str[0] if isinstance(p_str, list) else p_str
    psg = item.get('passage') or ''
    ok_ids = tokenizer(p_str, add_special_tokens=False).input_ids
    assert ok_ids == ids, 'offset mapping precondition failed (tokenization mismatch)'
    enc = tokenizer(p_str, add_special_tokens=False, return_offsets_mapping=True)
    lo_c = p_str.find(psg[:60]) if len(psg) >= 60 else p_str.find(psg)
    hi_c = lo_c + len(psg)
    mask = [lo < hi_c and hi > lo_c for lo, hi in enc.offset_mapping]
    return torch.tensor(mask, device=clean['input_ids'].device), None


def run(args):
    pool_path = args.pool or os.path.join(
        HERE, 'outputs', f'{args.dataset}_bon_grounding',
        f'{args.dataset}_bon_grounding.jsonl')
    recs = [json.loads(l) for l in open(pool_path)]
    recs = [r for r in recs if r.get('trajectories')]
    if args.limit:
        recs = recs[:args.limit]
    data = load_dataset(os.path.join(HERE, 'OmniHalluBench', args.dataset,
                                     args.dataset + '.json'),
                        os.path.join(HERE, 'OmniHalluBench'))
    by_id = {it['id']: it for it in data}

    # eager attention: SDPA returns None attentions; VAS needs the weights
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
    from shared.paths import resolve_model
    model_path = resolve_model("Qwen/Qwen2.5-Omni-7B")
    processor = Qwen2_5OmniProcessor.from_pretrained(model_path, trust_remote_code=True)
    processor.tokenizer.padding_side = 'left'
    if not processor.tokenizer.pad_token:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map='auto',
        trust_remote_code=True, attn_implementation='eager').eval()
    model.thinker.config.vocab_size = 152064
    tokenizer = processor.tokenizer
    thinker = model.thinker
    system_prompt = get_system_prompt(args.dataset, 'v1')

    # token classes for content positions (import from bon_grounding)
    from bon_grounding import token_class
    out_rows = []
    t0 = time.time()
    for si, rec in enumerate(recs):
        item = by_id[rec['id']]
        user_prompt = get_user_prompt(item['question'], args.dataset, 'v1',
                                      passage=item.get('passage'))
        image = None
        if args.dataset == 'haloquest':
            image = load_media_for_item(item, os.path.join(
                HERE, 'OmniHalluBench', args.dataset, args.dataset + '.json'),
                os.path.join(HERE, 'OmniHalluBench'))[2]
        messages = make_messages(system_prompt,
                                 build_user_content_with_media_tokens(None, None, image, user_prompt))
        clean = build_view(processor, model, system_prompt, user_prompt,
                           args.dataset, image)
        ev_mask, _ = evidence_mask(tokenizer, processor, clean, args.dataset, item, messages)
        cands = []
        for tr in rec['trajectories']:
            gen_ids = torch.tensor(tr['token_ids'], dtype=torch.long, device=model.device)
            toks = [tokenizer.decode([t], skip_special_tokens=True) for t in tr['token_ids']]
            cls = [token_class(t, toks[k - 1] if k else None) for k, t in enumerate(toks)]
            content_pos = torch.tensor([c in ('content', 'num') for c in cls],
                                       device=model.device)
            feats = forward_with_feats(thinker, clean, gen_ids, ev_mask, content_pos)
            cands.append({
                'rollout': tr['rollout'], 'sampled': bool(tr['rollout'] > 0),
                'f1': token_f1(tr.get('pred', ''), rec['gt']),
                'n_tokens': len(tr['token_ids']),
                'g_content': (tr.get('grounding_scores') or {}).get('g_content'),
                'g_mean_claim': (tr.get('grounding_scores') or {}).get('g_mean_claim'),
                'CE': feats['CE'],
                'VAS': feats['VAS'],
            })
        out_rows.append({'id': rec['id'], 'gt': rec['gt'][:200], 'question': rec['question'],
                         'selected_rollout': rec.get('selected_rollout'),
                         'candidates': cands})
        if (si + 1) % 20 == 0:
            print(f'  [{si+1}/{len(recs)}] {time.time()-t0:.0f}s', flush=True)
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({'dataset': args.dataset, 'n_samples': len(out_rows), 'rows': out_rows},
              open(args.out, 'w'), ensure_ascii=False)
    print(f'[done] -> {args.out}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=['ragtruth', 'haloquest'])
    ap.add_argument('--pool', default=None)
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()
    run(args)
