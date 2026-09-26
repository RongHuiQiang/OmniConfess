# -*- coding: utf-8 -*-
"""
base_avcd_nemotron_hf.py -- AVCD (Audio-Visual Contrastive Decoding) baseline
on Nemotron-3-Nano-Omni-30B.

Algorithm from base_avcd_qwen3.py (official AVCD defaults, NOT retuned):
  alpha=0.5, beta(APC)=0.2, entropy gate=0.6, apc_fill=-1e-4;
  3 modalities -> full AVCD (original + 3 masked branches),
  2 modalities -> single-instance CD, text-only -> greedy.

ARCHITECTURE ADAPTATION (recorded in every result row):
  The Nemotron backbone is a hybrid Mamba/Attention stack; only the E
  (attention) blocks -- 18 of 44 on this checkpoint -- carry attention
  weights. AVCD's post-softmax modality masking and the dominance/threshold
  computation are therefore applied on the ATTENTION SUBSET ONLY: masked
  branches mask the hot dominant-modality positions in every attention block
  except the last one; the dominance ranking and threshold are computed from
  the attention blocks' last-query rows (averaged over the subset, as the
  Qwen3 arm averages over its full layer stack). Mamba blocks mix through
  state-space recurrence, which cannot be masked -- this is a structural
  difference of the backbone, not a tuning choice. Additionally the modality
  spans use this checkpoint's placeholder ids: img_context_token_id=18
  (image AND video scatter) and sound_context_token_id=27.

Other Nemotron deltas:
  * Branches prefill on model.language_model over fused inputs_embeds, each
    with its own NemotronHHybridDynamicCache; step decode grows attention
    masks; no mRoPE.
  * NemotronHAttention.forward is replaced class-wide with an eager replica
    that adds AVCD masking + last-query row capture (same "replace forward,
    don't hook" approach as the Qwen3 arm, which the eager attention impl
    makes unnecessary for reading weights but necessary for masking).
"""
import os, sys, time, json, argparse, gc, math
import torch
import torch.nn.functional as F

REPO = os.environ.get('NEMO_REPO', '/home/tsq/rhq/omniconfess/OmniConfess')
sys.path.insert(0, os.path.join(REPO, 'OmniConfess', 'methods'))
from shared import load_dataset, get_system_prompt
from shared.eval_utils import parse_pred
from shared.nemotron_shim import load_nemotron, fuse_inputs_embeds
from shared.nemotron_methods import (
    build_processor_inputs, llm_prefill, llm_prefill_embeds, llm_step, set_seed,
    NEMOTRON_EOS_ID, attention_layer_indices, block_types,
)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

AVCD_ALPHA   = 0.5
APC_BETA     = 0.2
ENTROPY_GATE = 0.6
APC_FILL     = -1e-4

_STATE = {
    "active": False, "attn_rows": None, "mask_span": None,
    "threshold": None, "mask_upto": None,
}


# ---- NemotronHAttention replacement (eager + AVCD masking + row capture) ----

def _repeat_kv(hidden_states, n_rep):
    """GQA head expansion (same layout as transformers' repeat_kv)."""
    if n_rep == 1:
        return hidden_states
    bsz, n_kv_heads, seq_len, head_dim = hidden_states.shape
    return (hidden_states[:, :, None, :, :]
            .expand(bsz, n_kv_heads, n_rep, seq_len, head_dim)
            .reshape(bsz, n_kv_heads * n_rep, seq_len, head_dim))


def _avcd_attn_forward(self, hidden_states, attention_mask=None,
                       past_key_values=None, cache_position=None,
                       position_ids=None, user_cache=None, **kwargs):
    """Replica of NemotronHAttention.forward with AVCD post-softmax masking.
    Mirrors the module's own math (q/k/v proj, cache update via
    past_key_values.update, GQA repeat_kv, eager softmax in fp32).

    GQA note (2026-08-30): this checkpoint has num_attention_heads=32,
    num_key_value_heads=2 -- the module's own forward relies on
    eager_attention_forward to expand kv heads, which the first version of
    this replica omitted: matmul(q[.,32,.], k[.,2,.]^T) raised
    'The size of tensor a (32) must match the size of tensor b (2)' on every
    item (all six avcd cells empty). kv heads are expanded AFTER the cache
    update (the cache stores unexpanded kv, as in the stock module)."""
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    if past_key_values is not None:
        key_states, value_states = past_key_values.update(
            key_states, value_states, self.layer_idx,
            {"cache_position": cache_position})
    n_rep = getattr(self, "num_key_value_groups", 1)
    key_states = _repeat_kv(key_states, n_rep)
    value_states = _repeat_kv(value_states, n_rep)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask[:, :, :, : key_states.shape[-2]] \
            if attention_mask.dim() == 4 else attn_weights
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

    if _STATE["active"] and hidden_states.shape[1] > 1:  # q_len > 1 (prefill)
        row = attn_weights[:, :, -1, :].float()                     # [B,H,S]
        if _STATE["attn_rows"] is not None:
            _STATE["attn_rows"].append(row.detach())
        span = _STATE["mask_span"]
        if span is not None and self.layer_idx < _STATE["mask_upto"]:
            hot = (row > _STATE["threshold"]) & span.view(1, 1, -1)
            keep = (~hot).to(attn_weights.dtype).unsqueeze(-1)
            attn_weights = attn_weights * keep
            attn_weights = attn_weights / attn_weights.sum(-1, keepdim=True).clamp(min=1e-6)

    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


class AttnPatch:
    def __init__(self, attn_cls):
        self.cls = attn_cls
    def __enter__(self):
        self._orig = self.cls.forward
        self.cls.forward = _avcd_attn_forward
        _STATE["active"] = True
        return self
    def __exit__(self, *exc):
        _STATE.update(active=False, attn_rows=None, mask_span=None)
        self.cls.forward = self._orig
        return False


def get_modality_spans(input_ids, model):
    """video(visual)/audio/language spans; image+video share token id 18."""
    ids = input_ids[0]
    vis = (ids == model.img_context_token_id)
    aud = (ids == model.sound_context_token_id) \
        if getattr(model, 'sound_context_token_id', None) is not None \
        else torch.zeros_like(vis)
    return {"video": vis, "audio": aud, "language": ~(vis | aud)}


def compute_dominance(attn_rows, spans):
    """Same as base_avcd_qwen3.compute_dominance but over the attention-subset
    rows (use all captured rows: skip-last handled by caller)."""
    acc = None
    dom = {k: [] for k in spans}
    for row in attn_rows:                                   # [B,H,S]
        r = row.float()
        acc = r if acc is None else acc + r
        mean_h = r.mean(dim=1)[0]
        for name, m in spans.items():
            n = int(m.sum().item())
            dom[name].append((mean_h[m].sum().item() / n) if n > 0 else 0.0)
    acc = acc / len(attn_rows)
    threshold = torch.quantile(acc[0], 0.5, dim=-1).mean().item()
    ranking = sorted(((k, abs(sum(v) / len(v))) for k, v in dom.items()),
                     key=lambda x: x[1], reverse=True)
    return ranking, threshold


def avcd_generate(model, tokenizer, inputs, max_new_tokens=256,
                  alpha=AVCD_ALPHA, beta=APC_BETA,
                  entropy_gate=ENTROPY_GATE, apc_fill=APC_FILL):
    eos_id = tokenizer.eos_token_id or NEMOTRON_EOS_ID
    attn_cls = type(model.language_model.backbone.layers[
        attention_layer_indices(model)[0]].mixer)
    attn_idx = attention_layer_indices(model)
    n_attn = len(attn_idx)
    info = {"n_attn_layers": n_attn}

    spans_all = get_modality_spans(inputs['input_ids'], model)
    spans = {k: v for k, v in spans_all.items() if int(v.sum().item()) > 0}
    info["modalities"] = {k: int(v.sum().item()) for k, v in spans.items()}
    n_mod = len(spans)
    mode = "greedy" if n_mod < 2 else ("single_cd" if n_mod == 2 else "avcd")
    info["avcd_mode"] = mode

    # Fuse the item's media ONCE and reuse the embeds for the original +
    # masked-branch prefills (they differ only in attention masking, not in
    # inputs). Per-branch re-encoding through C-RADIO/Parakeet OOM'd the
    # 3+1-branch AVCD prefill on 80 GB (2026-08-30 empty-run fix).
    fused_embeds, fused_mask, _ = fuse_inputs_embeds(model, inputs)

    def prefill(inp, collect_attn=False, mask_span=None, threshold=None,
                mask_upto=None):
        _STATE["attn_rows"] = [] if collect_attn else None
        _STATE["mask_span"] = mask_span
        _STATE["threshold"] = threshold
        _STATE["mask_upto"] = mask_upto
        try:
            lg, cache, mask = llm_prefill_embeds(model, inp, fused_embeds,
                                                 fused_mask)
        finally:
            rows = _STATE["attn_rows"]
            _STATE["attn_rows"] = None
            _STATE["mask_span"] = None
        torch.cuda.empty_cache()
        return lg, cache, mask, rows

    with AttnPatch(attn_cls):
        logits_o, pkv_o, attn_mask, rows = prefill(
            inputs, collect_attn=(mode != "greedy"))
        branch_logits, branch_pkv, branch_masks = [], [], []
        if mode == "greedy":
            info["dominant"] = None
            print('  [degraded] single modality -> plain greedy', flush=True)
        else:
            if not rows or len(rows) < n_attn:
                raise RuntimeError(
                    f'attention hook captured {len(rows) if rows else 0} rows '
                    f'(expected {n_attn} attention layers); the attention '
                    'interception did not take effect')
            use_rows = rows[:-1] if len(rows) > 1 else rows  # skip last attn layer
            ranking, threshold = compute_dominance(use_rows, spans)
            del rows
            scores = [v for _, v in ranking]
            assert len(ranking) == len(spans), f'degenerate ranking: {ranking}'
            assert max(scores) > 0.0, f'all-zero dominance: {ranking}'
            assert threshold > 0.0, f'non-positive threshold: {threshold}'
            dominant = ranking[0][0]
            others = [k for k, _ in ranking[1:]]
            info.update(dominant=dominant,
                        dominance={k: round(v, 10) for k, v in ranking},
                        attn_threshold=round(threshold, 10))
            print(f"  [avcd] mode={mode} dominant={dominant} "
                  f"rank={[(k, f'{v:.3e}') for k, v in ranking]} "
                  f"thr={threshold:.3e}", flush=True)
            if mode == "avcd":
                mask_sets = [spans[others[0]] | spans[others[1]],
                             spans[others[0]], spans[others[1]]]
                info["masked_modalities"] = ["+".join(others), others[0], others[1]]
            else:
                mask_sets = [spans[others[0]]]
                info["masked_modalities"] = [others[0]]
            for sp in mask_sets:
                lg, pkv, mask, _ = prefill(inputs, collect_attn=False,
                                           mask_span=sp, threshold=threshold,
                                           mask_upto=attn_idx[-1])  # mask all but last attn layer
                branch_logits.append(lg)
                branch_pkv.append(pkv)
                branch_masks.append(mask)

        n_gated = 0
        def _select(lo, lms):
            nonlocal n_gated
            if not lms:
                return lo
            ent = -(F.softmax(lo, dim=-1) * F.log_softmax(lo, dim=-1)).sum(dim=-1)
            if ent.item() < entropy_gate:
                n_gated += 1
                return lo
            if len(lms) == 3:
                contrastive = (2 + 2 * alpha) * lo - 2 * alpha * lms[0] + lms[1] + lms[2]
            else:
                contrastive = (1 + alpha) * lo - alpha * lms[0]
            cutoff = math.log(beta) + lo.max(dim=-1, keepdim=True).values
            return contrastive.masked_fill(lo < cutoff, apc_fill)

        generated = []
        with torch.inference_mode():
            sel = _select(logits_o, branch_logits)
            cur = sel.argmax(dim=-1, keepdim=True)
            tok = cur.item()
            generated.append(tok)
            for _ in range(1, max_new_tokens):
                if tok == eos_id:
                    break
                if len(generated) >= 5 and len(set(generated[-5:])) == 1:
                    break
                logits_o, pkv_o, attn_mask = llm_step(model, cur, attn_mask, pkv_o)
                for i in range(len(branch_pkv)):
                    branch_logits[i], branch_pkv[i], branch_masks[i] = llm_step(
                        model, cur, branch_masks[i], branch_pkv[i])
                sel = _select(logits_o, branch_logits)
                cur = sel.argmax(dim=-1, keepdim=True)
                tok = cur.item()
                generated.append(tok)

    info["n_entropy_gated"] = n_gated
    info["n_tokens"] = len(generated)
    return (tokenizer.decode(generated, skip_special_tokens=True).strip(),
            generated, info)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets',      type=str, required=True)
    parser.add_argument('--data_path',     type=str, required=True)
    parser.add_argument('--data_root',     type=str, default=None)
    parser.add_argument('--output_dir',    type=str, default='./outputs')
    parser.add_argument('--file_name',     type=str, default='avcd_nemotron')
    parser.add_argument('--time_path',     type=str, default='./outputs/time')
    parser.add_argument('--max_new_tokens',type=int, default=256)
    parser.add_argument('--alpha',         type=float, default=AVCD_ALPHA)
    parser.add_argument('--beta',          type=float, default=APC_BETA)
    parser.add_argument('--entropy_gate',  type=float, default=ENTROPY_GATE)
    parser.add_argument('--apc_fill',      type=float, default=APC_FILL)
    parser.add_argument('--prompt_version',type=str, default='v1')
    parser.add_argument('--model_path',    type=str,
                        default=os.environ.get('NEMOTRON_MODEL_PATH'))
    parser.add_argument('--enable_thinking', action='store_true')
    parser.add_argument('--record_process', action=argparse.BooleanOptionalAction,
                        default=False)
    parser.add_argument('--max_samples',   type=int, default=-1)
    parser.add_argument('--start_index',   type=int, default=0)
    parser.add_argument('--dry_run',       action='store_true')
    args = parser.parse_args()
    if not args.data_root:
        args.data_root = os.path.join(REPO, 'OmniHalluBench', args.datasets) + os.sep

    set_seed(42)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.time_path, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.file_name + '.jsonl')
    error_path = os.path.join(args.output_dir, f'ERROR-{args.file_name}.log')

    done_ids = set()
    if os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line.strip())['id'])
                except Exception:
                    pass
        print(f'[resume] skipping {len(done_ids)} done samples', flush=True)

    processor, model, tokenizer, DEVICE = load_nemotron(args.model_path)
    n_attn = len(attention_layer_indices(model))
    print(f'[avcd] alpha={args.alpha} beta={args.beta} gate={args.entropy_gate} '
          f'apc_fill={args.apc_fill} | attention blocks={n_attn}/'
          f'{len(block_types(model))} (masking on attention subset only)', flush=True)

    data = load_dataset(args.data_path, args.data_root)
    if args.start_index > 0:
        data = data[args.start_index:]
    if args.dry_run:
        data = data[:3]
    elif args.max_samples > 0:
        data = data[:args.max_samples]
    total = len(data)
    print(f'[data] {total} samples | dataset={args.datasets}', flush=True)

    system_prompt = get_system_prompt(args.datasets, args.prompt_version)
    start = time.time()

    with open(output_path, 'a', encoding='utf-8') as res_f:
        for idx, item in enumerate(data, start=args.start_index):
            item_id = item.get('id', idx)
            if item_id in done_ids:
                continue
            t0 = time.time()
            print(f'\n[{idx+1}/{args.start_index+total}] id={item_id}', flush=True)
            try:
                inputs, input_len, _ = build_processor_inputs(
                    processor, tokenizer, item, args, system_prompt, DEVICE)
                response, gen_ids, info = avcd_generate(
                    model, tokenizer, inputs,
                    max_new_tokens=args.max_new_tokens,
                    alpha=args.alpha, beta=args.beta,
                    entropy_gate=args.entropy_gate, apc_fill=args.apc_fill)

                elapsed = time.time() - t0
                pred = parse_pred(response, args.datasets)
                result = {
                    'id': item_id, 'pred': pred, 'raw_output': response,
                    'gt': item.get('gt', item.get('answer', item.get('label', ''))),
                    'question': item.get('question', item.get('input', '')),
                    'group_key': item.get('group_key', item.get('sub_category', 'Default')),
                    'sub_category': item.get('sub_category', item.get('group_key', 'Default')),
                    'dataset': args.datasets, 'method': 'avcd',
                    'has_avcd': info['avcd_mode'] != 'greedy',
                    'time_sec': round(elapsed, 3), 'status': 'success',
                    'input_len': input_len, 'output_tokens': len(gen_ids),
                }
                result.update(info)
                if info['avcd_mode'] == 'greedy':
                    result['degraded_reason'] = 'text-only: no AV contrast'
                elif info['avcd_mode'] == 'single_cd':
                    result['degraded_reason'] = ('2 modalities: AVCD self-cancels '
                                                 '-> single-instance CD')
                result['nemotron'] = {
                    'method': 'avcd',
                    'adaptation': ('masking/dominance on attention blocks only '
                                   f'({info.get("n_attn_layers")} of '
                                   f'{len(block_types(model))} hybrid blocks); '
                                   'Mamba blocks are not maskable (see header)'),
                    'runtime': 'transformers-trust-remote-code'}
                res_f.write(json.dumps(result, ensure_ascii=False) + '\n')
                res_f.flush()
                print(f"  pred: {pred!r} raw: {response[:60]!r} ({elapsed:.1f}s)",
                      flush=True)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); gc.collect()
                with open(error_path, 'a') as ef:
                    ef.write(f"{time.strftime('%H:%M:%S')} | id={item_id} | OOM\n")
                res_f.write(json.dumps({'id': item_id, 'status': 'OOM'}) + '\n')
                res_f.flush()
            except Exception as e:
                import traceback
                torch.cuda.empty_cache(); gc.collect()
                print(f'  [ERR] id={item_id}: {repr(e)}', flush=True)
                with open(error_path, 'a') as ef:
                    ef.write(f"{time.strftime('%H:%M:%S')} | id={item_id} | {repr(e)}\n")
                    ef.write(traceback.format_exc() + '\n')
                res_f.write(json.dumps(
                    {'id': item_id, 'status': 'failed',
                     'error': repr(e)[:200]}) + '\n')
                res_f.flush()
            finally:
                gc.collect(); torch.cuda.empty_cache()

    with open(os.path.join(args.time_path, args.file_name + '.txt'), 'w') as tf:
        tf.write(f'time: {time.time()-start:.1f}s\n')
        tf.write(f'total_samples: {total}\nalpha: {args.alpha}\n')
        tf.write(f'beta: {args.beta}\nentropy_gate: {args.entropy_gate}\n')
        tf.write(f'apc_fill: {args.apc_fill}\n')
    print(f'\n[done] {total} samples | {output_path}', flush=True)


if __name__ == '__main__':
    main()
