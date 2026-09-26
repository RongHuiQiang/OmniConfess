# -*- coding: utf-8 -*-
"""probe_bon_features_qwen3.py — Qwen3-Omni port of probe_bon_features.py.

BoN rerank 三特征（Δ / 临界熵 CE / VAS）判别力探针，移植到 Qwen3-Omni。

关键适配（brief 预警的最大点 = VAS 注意力钩子）：
  1. 深度重定标：Qwen2.5 thinker 28 层 → VAS_LAYERS (9,14,23) primary=14。
     Qwen3 thinker 48 层 → 按比例 VAS_LAYERS (15,24,39) primary=24（0.32/0.5/0.82）。
  2. **注意力捕获方式变了（核心坑）**：Qwen3 的 Qwen3OmniMoeThinkerTextDecoderLayer
     把 self_attn 返回的 attn_weights 直接丢弃（`hidden_states, _ = self.self_attn(...)`），
     且 ThinkerTextModel.forward 不收集 attentions → `output_attentions=True` 在
     Qwen3 上返回 None（已核实 modeling_qwen3_omni_moe.py line 1535 & 1698-1720）。
     Qwen2.5 靠 out.attentions 读，Qwen3 读不到。
     → 改用 **forward hook** 挂在 model.model.layers[li].self_attn 上，捕获
     其返回 tuple 的 [1]（attn_weights，shape [B,H,q,kv]）。eager 加载下
     eager_attention_forward 会算真权重（SDPA 会返回 None）。不猴补模型 math。
  3. 证据掩码 token：Qwen2.5 用 <|IMAGE|>，Qwen3 用 image_token_id=151655
     (<|image_pad|>)。
  4. eager 加载：attn_implementation="eager"，否则 hook 抓到的 attn_weights 是 None。
  5. model.thinker 跳消失：thinker = model 本体；文本骨干 model.model.layers。

Sanity 判据：VAS 判别力 AUC（hit vs miss，对照 Q2.5 halo 的 0.572）。

Usage:
    CUDA_VISIBLE_DEVICES=7 python probe_bon_features_qwen3.py --dataset haloquest \
        --pool outputs/haloquest_bon_grounding_qwen3/haloquest_bon_grounding_qwen3.jsonl \
        --out outputs/bon_probe_qwen3/haloquest_bon_feat_qwen3.json
"""
import os, sys, json, argparse, time, string

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'OmniConfess', 'methods'))
sys.path.insert(0, HERE)
from shared import (  # noqa: E402
    load_dataset, load_media_for_item,
    get_system_prompt, get_user_prompt,
)
# Qwen3 media-token spelling + input builders (via shim re-exports)
from shared.qwen3_shim import (  # noqa: E402
    build_inputs, prepare_for_thinker,
    build_user_content_with_media_tokens, make_messages,
)
from shared.paths import resolve_model  # noqa: E402

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("MMHALU_MODEL_ROOT", "/ultrafast_share/tsq/MODELS")

TOPK = 5        # truncated-entropy top-k (TTSP)
KSTAT = 5       # order statistic: mean of the KSTAT largest H_t
# 48-layer Qwen3 thinker, re-scaled from Qwen2.5's (9,14,23)/28:
#   9/28=0.321 -> 15 ; 14/28=0.5 -> 24 (primary) ; 23/28=0.821 -> 39
VAS_LAYERS = (15, 24, 39)
VAS_PRIMARY = 24
IMAGE_TOKEN_ID = 151655   # Qwen3 <|image_pad|> (thinker_config.image_token_id)

_ATTN_STORE = {}          # {layer_idx: attn_weights tensor} filled by hooks


def _make_hook(li):
    def hook(module, inputs, output):
        # Qwen3OmniMoeThinkerTextAttention.forward returns (attn_output, attn_weights)
        w = output[1] if isinstance(output, (tuple, list)) and len(output) > 1 else None
        _ATTN_STORE[li] = w
    return hook


def register_vas_hooks(thinker, layers):
    """Hook self_attn of target decoder layers to capture attn_weights (Qwen3
    layers discard them, so output_attentions=True is empty)."""
    handles = []
    text_layers = thinker.model.layers   # Qwen3OmniMoeThinkerTextModel.layers
    for li in layers:
        h = text_layers[li].self_attn.register_forward_hook(_make_hook(li))
        handles.append(h)
    return handles


def load_model_eager(model_path=None):
    """Qwen3 thinker, eager attention (so hooks capture real attn_weights),
    with the shim's eos/pad pins. thinker == model (no .thinker hop)."""
    from transformers import (Qwen3OmniMoeThinkerForConditionalGeneration,
                              Qwen3OmniMoeProcessor)
    mp = model_path or resolve_model("Qwen/Qwen3-Omni-30B-A3B-Instruct")
    processor = Qwen3OmniMoeProcessor.from_pretrained(mp, trust_remote_code=True)
    tk = processor.tokenizer
    tk.padding_side = 'left'
    if not tk.pad_token:
        tk.pad_token = tk.eos_token
    device_map = os.environ.get("Q3O_DEVICE_MAP", "cuda:0")
    model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
        mp, torch_dtype=torch.bfloat16, device_map=device_map,
        trust_remote_code=True, attn_implementation="eager").eval()
    # eos/pad pins (Qwen3 known pit)
    im_end = tk.convert_tokens_to_ids("<|im_end|>")
    endoftext = tk.convert_tokens_to_ids("<|endoftext|>")
    if im_end is not None:
        tk.eos_token_id = im_end
    if endoftext is not None:
        tk.pad_token_id = endoftext
    stops = [i for i in {im_end, endoftext} if i is not None]
    try:
        if getattr(model, "generation_config", None) is not None and stops:
            model.generation_config.eos_token_id = stops
            if endoftext is not None:
                model.generation_config.pad_token_id = endoftext
    except Exception:
        pass
    # self-alias for any .thinker access
    try:
        model.thinker = model
    except Exception:
        object.__setattr__(model, "thinker", model)
    return processor, model


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
    """One teacher-forced forward; CE from logits, VAS from hook-captured attn."""
    input_len = clean['input_ids'].shape[1]
    full = dict(clean)
    full['input_ids'] = torch.cat([clean['input_ids'], gen_ids.unsqueeze(0)], dim=1)
    full['attention_mask'] = torch.ones_like(full['input_ids'])
    _ATTN_STORE.clear()
    out = thinker(**full, use_cache=False)   # hooks fill _ATTN_STORE
    logits = out.logits[0].float()
    pos = torch.arange(input_len - 1, full['input_ids'].shape[1] - 1,
                       device=logits.device)
    lg = logits[pos]
    top = torch.topk(lg, TOPK, dim=-1).values
    pk = torch.softmax(top, dim=-1)
    H = -(pk * torch.log(pk + 1e-9)).sum(-1).cpu().numpy()
    H_sorted = np.sort(H)[::-1]
    ce = {
        'CE_top5mean': float(np.mean(H_sorted[:KSTAT])),
        'CE_max': float(H_sorted[0]),
        'H_per_tok': [round(float(x), 4) for x in H],
    }
    vas = {}
    cp = content_pos.cpu().numpy().astype(bool)
    for li in VAS_LAYERS:
        att = _ATTN_STORE.get(li)
        if att is None:
            vas[li] = {'vas_all': None, 'vas_content': None}
            continue
        a = att[0].float().mean(0)            # [q, kv] heads averaged
        rows = a[pos]                          # [n_gen, kv]
        denom = rows.sum(-1).clamp(min=1e-9)
        frac = (rows[:, :input_len][:, ev_mask].sum(-1) / denom).cpu().numpy()
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
    """Bool mask over prompt positions marking image tokens (haloquest).
    Qwen3: image placeholder expands to IMAGE_TOKEN_ID (<|image_pad|>)."""
    ids = clean['input_ids'][0].tolist()
    if dataset == 'haloquest':
        mask = torch.tensor([t == IMAGE_TOKEN_ID for t in ids],
                            device=clean['input_ids'].device)
        assert mask.any(), 'no image_pad tokens found in prompt (evidence mask empty)'
        return mask, int(mask.sum())
    raise NotImplementedError('qwen3 feat probe: haloquest only for stage 3')


def run(args):
    pool_path = args.pool
    recs = [json.loads(l) for l in open(pool_path)]
    recs = [r for r in recs if r.get('trajectories')]
    if args.limit:
        recs = recs[:args.limit]
    data = load_dataset(os.path.join(HERE, 'OmniHalluBench', args.dataset,
                                     args.dataset + '.json'),
                        os.path.join(HERE, 'OmniHalluBench'))
    by_id = {it['id']: it for it in data}

    processor, model = load_model_eager()
    tokenizer = processor.tokenizer
    thinker = model                       # no .thinker hop
    handles = register_vas_hooks(thinker, VAS_LAYERS)
    print(f'[hooks] registered on layers {VAS_LAYERS} (primary {VAS_PRIMARY}) '
          f'of {len(thinker.model.layers)}-layer thinker', flush=True)
    system_prompt = get_system_prompt(args.dataset, 'v1')

    from bon_grounding import token_class
    out_rows = []
    t0 = time.time()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tmp_path = args.out + '.tmp'

    done_ids = set()
    if os.path.exists(tmp_path):
        with open(tmp_path) as f:
            prev = json.load(f)
        out_rows = prev['rows']
        done_ids = {r['id'] for r in out_rows}
        print(f'[resume] {len(done_ids)} rows already in {tmp_path}, skipping them', flush=True)

    def _meta():
        return {'dataset': args.dataset, 'n_samples': len(out_rows),
                'backbone': 'Qwen3-Omni-30B-A3B-Instruct',
                'vas_layers': list(VAS_LAYERS), 'vas_primary': VAS_PRIMARY,
                'rows': out_rows}

    def _flush(path):
        # crash-safe: write to a sibling temp, fsync, atomic replace
        wp = path + '.writing'
        with open(wp, 'w') as f:
            json.dump(_meta(), f, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(wp, path)

    skipped_missing_media = []
    for si, rec in enumerate(recs):
        if rec['id'] in done_ids:
            continue
        item = by_id[rec['id']]
        user_prompt = get_user_prompt(item['question'], args.dataset, 'v1',
                                      passage=item.get('passage'))
        image = None
        if args.dataset == 'haloquest':
            image = load_media_for_item(item, os.path.join(
                HERE, 'OmniHalluBench', args.dataset, args.dataset + '.json'),
                os.path.join(HERE, 'OmniHalluBench'))[2]
            if image is None and item.get('image_path'):
                # dataset asset missing on disk (known gap: 3/365 haloquest images
                # absent) — skip this sample rather than crash the whole run.
                print(f"[SKIP] id={rec['id']} missing image asset "
                      f"{item['image_path']}", flush=True)
                skipped_missing_media.append(rec['id'])
                continue
        messages = make_messages(system_prompt,
                                 build_user_content_with_media_tokens(None, None, image, user_prompt))
        clean = build_view(processor, model, system_prompt, user_prompt,
                           args.dataset, image)
        ev_mask, n_ev = evidence_mask(tokenizer, processor, clean, args.dataset, item, messages)
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
                'g_traj_mean': (tr.get('grounding_scores') or {}).get('g_traj_mean'),
                'CE': feats['CE'],
                'VAS': {str(k): v for k, v in feats['VAS'].items()},
                'n_evidence_tok': n_ev,
            })
        out_rows.append({'id': rec['id'], 'gt': rec['gt'][:200], 'question': rec['question'],
                         'selected_rollout': rec.get('selected_rollout'),
                         'candidates': cands})
        if (si + 1) % 20 == 0:
            print(f'  [{si+1}/{len(recs)}] {time.time()-t0:.0f}s', flush=True)
            _flush(tmp_path)   # periodic crash-safe checkpoint
        torch.cuda.empty_cache()

    for h in handles:
        h.remove()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    _flush(tmp_path)
    os.replace(tmp_path, args.out)   # atomic promote to final
    print(f'[done] -> {args.out}  (n={len(out_rows)}, '
          f'skipped_missing_media={skipped_missing_media})')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='haloquest', choices=['haloquest'])
    ap.add_argument('--pool', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()
    run(args)
