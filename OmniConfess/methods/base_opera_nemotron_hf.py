# -*- coding: utf-8 -*-
"""
base_opera_nemotron_hf.py -- OPERA baseline (over-trust penalty + partial
rollback beam search) on Nemotron-3-Nano-Omni-30B.

Algorithm 1:1 from base_opera_qwen3.py (official eval-script defaults, NOT
retuned): num_beams=5, scale_factor=50, threshold r=15, n_can=5,
penalty_weights=1.0, window_size=512, greedy (do_sample=False).

Nemotron deltas vs the Qwen3 arm:
  1. Attention access: the loader pins attn_implementation='eager' and the
     remote-code eager attention ALWAYS returns attn_weights as output[1], so
     the last-layer row is read with a plain forward hook on the last
     ATTENTION block's mixer (the backbone is hybrid; the final E block of
     this checkpoint is attention, but the hook is attached to the last
     attention block whichever index it is).
  2. Cache surgery on NemotronHHybridDynamicCache (shared/nemotron_methods):
     key_cache/value_cache lists are sliced/selected/expanded like a plain
     DynamicCache; the Mamba conv/ssm states are batch-first and are
     expanded/selected along dim 0. CRITICAL: probe forwards update the Mamba
     states IN PLACE, so every probe round snapshots them first and restores
     after (the attention kv crop alone would leave SSM state corrupted).
     Rollback replays the prefix through the model, which recomputes both the
     attention kv and the SSM states consistently.
  3. Branch prefill goes through model.language_model over fused
     inputs_embeds; no mRoPE, positions auto-derived from cache_position.
  4. Visual token mask uses img_context_token_id (image AND video share it on
     this checkpoint).
Degenerate behaviour identical: no visual tokens -> the first-10-token image
mass penalty is 0 (plain beam search), has_visual=False recorded.
"""
import os, sys, time, json, argparse, gc
import torch
import torch.nn.functional as F

REPO = os.environ.get('NEMO_REPO', '/home/tsq/rhq/omniconfess/OmniConfess')
sys.path.insert(0, os.path.join(REPO, 'OmniConfess', 'methods'))
from shared import load_dataset, get_system_prompt
from shared.eval_utils import parse_pred
from shared.nemotron_shim import load_nemotron
from shared.nemotron_methods import (
    build_processor_inputs, llm_prefill, set_seed, NEMOTRON_EOS_ID,
    attention_layer_indices, block_types, cache_len, cache_expand,
    cache_select, cache_crop, cache_clone, cache_snapshot_mamba,
    cache_restore_mamba, last_attention_block,
)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

NUM_BEAMS           = 5
SCALE_FACTOR        = 50.0
ROLLBACK_THRESHOLD  = 15
NUM_ATTN_CANDIDATES = 5
PENALTY_WEIGHT      = 1.0
WINDOW_SIZE         = 512
PHI_SCALE           = 1e-7
EARLY_TOKENS        = 10
NEG_INF_FILL        = -999.0

_STATE = {"active": False, "row": None, "n_captured": 0}


class LastLayerAttn:
    """Capture the current query's last-attention-block row (max over heads).
    The eager attention returns (attn_output, attn_weights)."""

    def __init__(self, model):
        self.attn = last_attention_block(model).mixer
        self.handles = []

    def _post(self, module, args, output):
        if not _STATE["active"]:
            return None
        w = output[1] if isinstance(output, (tuple, list)) and len(output) > 1 else None
        if w is not None:
            _STATE["row"] = w[:, :, -1, :].max(dim=1).values.float().detach()
            _STATE["n_captured"] += 1
        return None

    def __enter__(self):
        self.handles.append(self.attn.register_forward_hook(self._post))
        _STATE["active"] = True
        return self

    def __exit__(self, *exc):
        _STATE.update(active=False, row=None)
        for h in self.handles:
            h.remove()
        self.handles = []
        return False


def visual_token_mask(input_ids, model):
    ids = input_ids[0]
    m = (ids == model.img_context_token_id)
    return m, int(m.sum().item())


def column_product_scores(rows, sigma, phi_scale):
    a = rows * sigma
    B, C, T, _ = a.shape
    out = torch.zeros(B, C, T, device=a.device, dtype=torch.float32)
    for j in range(T):
        out[..., j] = phi_scale * a[..., j:, j].prod(-1).float()
    return out


def opera_generate(model, tokenizer, inputs, vis_mask, max_new_tokens=256,
                   num_beams=NUM_BEAMS, n_can=NUM_ATTN_CANDIDATES,
                   sigma=SCALE_FACTOR, alpha=PENALTY_WEIGHT,
                   r=ROLLBACK_THRESHOLD, window_size=WINDOW_SIZE,
                   length_penalty=1.0, enable_rollback=True):
    device = next(model.parameters()).device
    eos_id = tokenizer.eos_token_id or NEMOTRON_EOS_ID
    info = {"n_rollback": 0}

    logits0, cache0, attn_mask0 = llm_prefill(model, inputs)
    prompt_len = int(inputs['input_ids'].shape[1])

    prompt_cache = cache_clone(model, cache0)
    past_kv = cache_expand(cache_clone(model, cache0), num_beams)
    attn_mask = attn_mask0.repeat_interleave(num_beams, dim=0)
    next_logits = logits0.float().repeat(num_beams, 1)
    del cache0, logits0

    vis_idx = torch.nonzero(vis_mask, as_tuple=True)[0]

    beam_scores = torch.full((num_beams,), -1e9, device=device, dtype=torch.float)
    beam_scores[0] = 0.0
    seqs = [[] for _ in range(num_beams)]
    resp_rows = []
    finished = []
    hist_locs, snapshots = [], []
    max_rollback_time, reject = {}, {}

    _STATE["n_captured"] = 0
    with LastLayerAttn(model), torch.inference_mode():
        for _ in range(max_new_tokens):
            T_prev = len(resp_rows)
            cand_lg, cand_id = next_logits.topk(n_can, dim=-1)

            base_len = cache_len(past_kv)
            probe_rows, probe_img = [], []
            probe_mask = torch.cat(
                [attn_mask, torch.ones(num_beams, 1, device=device, dtype=torch.long)],
                dim=-1)
            mamba_snap = cache_snapshot_mamba(past_kv)   # probes mutate SSM state
            for c in range(n_can):
                model.language_model(
                    input_ids=cand_id[:, c:c + 1], attention_mask=probe_mask,
                    past_key_values=past_kv, use_cache=True)
                row = _STATE["row"]
                if row is None:
                    raise RuntimeError(
                        f'OPERA attention hook captured no row on a probe forward '
                        f'(step T_prev={T_prev}, candidate {c}); the attention '
                        'interception did not take effect')
                row = row.float()
                row = row / row.sum(-1, keepdim=True).clamp(min=1e-12)
                probe_rows.append(row[:, prompt_len:])
                probe_img.append(row[:, vis_idx].sum(-1) if vis_idx.numel() > 0
                                 else torch.zeros(num_beams, device=device))
                cache_crop(past_kv, base_len)
            cache_restore_mamba(past_kv, mamba_snap)

            T = T_prev + 1
            square = torch.zeros(num_beams, n_can, T, T, device=device,
                                 dtype=torch.float32)
            for t, rr in enumerate(resp_rows):
                L = min(rr.shape[1], T)
                square[:, :, t, :L] = rr[:, :L].unsqueeze(1)
            for c in range(n_can):
                L = min(probe_rows[c].shape[1], T)
                square[:, c, T - 1, :L] = probe_rows[c][:, :L]

            local_scores = column_product_scores(square, sigma, PHI_SCALE)
            rollback_scores, rollback_locs = local_scores.max(-1)

            attn_i = torch.stack(probe_img, dim=1)
            penalty = (-attn_i) if T <= EARLY_TOKENS else rollback_scores
            cand_scores = cand_lg - alpha * penalty
            full = torch.full_like(next_logits, NEG_INF_FILL)
            full.scatter_(-1, cand_id, cand_scores)

            for (bsrc, tok) in reject.get(T_prev, ()):
                if bsrc < num_beams:
                    full[bsrc, tok] = NEG_INF_FILL

            scores = F.log_softmax(full, dim=-1) + beam_scores.unsqueeze(1)
            flat = scores.view(-1)
            top_sc, top_ix = flat.topk(min(2 * num_beams, flat.numel()))
            V = scores.shape[-1]

            nb_scores, nb_tokens, nb_idx = [], [], []
            for sc, ix in zip(top_sc.tolist(), top_ix.tolist()):
                b, tok = ix // V, ix % V
                if tok == eos_id:
                    if seqs[b]:
                        finished.append((sc / (len(seqs[b]) ** length_penalty),
                                         list(seqs[b])))
                    continue
                nb_scores.append(sc); nb_tokens.append(tok); nb_idx.append(b)
                if len(nb_scores) == num_beams:
                    break
            if not nb_scores:
                break
            while len(nb_scores) < num_beams:
                nb_scores.append(nb_scores[-1]); nb_tokens.append(nb_tokens[-1])
                nb_idx.append(nb_idx[-1])

            snapshots.append({
                "T": T_prev, "seqs": [list(s) for s in seqs],
                "beam_scores": beam_scores.clone(),
                "resp_rows": [rr.clone() for rr in resp_rows],
                "sel": list(zip(nb_idx, nb_tokens)),
            })
            if len(snapshots) > window_size:
                snapshots.pop(0)

            src = torch.tensor(nb_idx, device=device, dtype=torch.long)
            cache_select(past_kv, src)
            resp_rows = [rr.index_select(0, src) for rr in resp_rows]
            seqs = [seqs[b] + [t] for b, t in zip(nb_idx, nb_tokens)]
            beam_scores = torch.tensor(nb_scores, device=device, dtype=torch.float)
            hist_locs = [hl.index_select(0, src) if torch.is_tensor(hl) else hl
                         for hl in hist_locs]
            hist_locs.append(rollback_locs.mode(dim=-1).values.index_select(0, src))
            if len(hist_locs) > window_size:
                hist_locs.pop(0)

            cur = torch.tensor(nb_tokens, device=device, dtype=torch.long).view(num_beams, 1)
            attn_mask = torch.cat(
                [attn_mask, torch.ones(num_beams, 1, device=device, dtype=torch.long)],
                dim=-1)
            outs = model.language_model(
                input_ids=cur, attention_mask=attn_mask,
                past_key_values=past_kv, use_cache=True)
            past_kv = outs.past_key_values
            next_logits = outs.logits[:, -1, :].float()
            row = _STATE["row"].float()
            row = row / row.sum(-1, keepdim=True).clamp(min=1e-12)
            resp_rows.append(row[:, prompt_len:])
            del outs

            if len(finished) >= num_beams:
                break

            if enable_rollback and len(hist_locs) > r:
                gathers = torch.stack(hist_locs, dim=-1)
                glob = gathers.reshape(-1).mode().values
                if bool(((gathers == glob).sum(-1) > r).all()) and int(glob) >= EARLY_TOKENS:
                    pos_rb = int(glob) + 1
                    if max_rollback_time.get(pos_rb, 0) >= n_can:
                        pos_rb -= 1
                    if 0 < pos_rb < len(seqs[0]) and max_rollback_time.get(pos_rb, 0) < n_can:
                        snap = next((s for s in snapshots if s["T"] == pos_rb), None)
                        if snap is not None:
                            try:
                                max_rollback_time[pos_rb] = max_rollback_time.get(pos_rb, 0) + 1
                                reject.setdefault(pos_rb, set()).update(snap["sel"])
                                past_kv = cache_expand(
                                    cache_clone(model, prompt_cache), num_beams)
                                seqs = [list(s) for s in snap["seqs"]]
                                beam_scores = snap["beam_scores"].clone()
                                resp_rows = [rr.clone() for rr in snap["resp_rows"]]
                                hist_locs = hist_locs[:pos_rb] if pos_rb <= len(hist_locs) else hist_locs
                                snapshots = [s for s in snapshots if s["T"] < pos_rb]
                                attn_mask = attn_mask0.repeat_interleave(num_beams, dim=0)
                                if pos_rb > 0:
                                    pre = torch.tensor([s[:pos_rb] for s in seqs],
                                                       device=device, dtype=torch.long)
                                    attn_mask = torch.cat(
                                        [attn_mask, torch.ones(num_beams, pos_rb,
                                                               device=device,
                                                               dtype=torch.long)], dim=-1)
                                    o2 = model.language_model(
                                        input_ids=pre, attention_mask=attn_mask,
                                        past_key_values=past_kv, use_cache=True)
                                    past_kv = o2.past_key_values
                                    next_logits = o2.logits[:, -1, :].float()
                                    del o2
                                info["n_rollback"] += 1
                                print(f"  [opera] rollback -> resp_pos={pos_rb} "
                                      f"(#{info['n_rollback']})", flush=True)
                            except Exception as e:
                                info["rollback_disabled"] = repr(e)[:120]
                                enable_rollback = False

    info["n_attn_rows_captured"] = _STATE["n_captured"]
    for b in range(num_beams):
        if seqs[b]:
            finished.append((float(beam_scores[b]) / (len(seqs[b]) ** length_penalty),
                             list(seqs[b])))
    if not finished:
        return "", [], info
    best = max(finished, key=lambda x: x[0])[1]
    best = [t for t in best if t != eos_id]
    info["n_tokens"] = len(best)
    return tokenizer.decode(best, skip_special_tokens=True).strip(), best, info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets',       type=str, required=True)
    parser.add_argument('--data_path',      type=str, required=True)
    parser.add_argument('--data_root',      type=str, default=None)
    parser.add_argument('--output_dir',     type=str, default='./outputs')
    parser.add_argument('--file_name',      type=str, default='opera_nemotron')
    parser.add_argument('--time_path',      type=str, default='./outputs/time')
    parser.add_argument('--max_new_tokens', type=int, default=256)
    parser.add_argument('--num_beams',      type=int, default=NUM_BEAMS)
    parser.add_argument('--scale_factor',   type=float, default=SCALE_FACTOR)
    parser.add_argument('--num_attn_candidates', type=int, default=NUM_ATTN_CANDIDATES)
    parser.add_argument('--penalty_weights', type=float, default=PENALTY_WEIGHT)
    parser.add_argument('--threshold',      type=int, default=ROLLBACK_THRESHOLD)
    parser.add_argument('--window_size',    type=int, default=WINDOW_SIZE)
    parser.add_argument('--no_rollback',    action='store_true')
    parser.add_argument('--prompt_version', type=str, default='v1')
    parser.add_argument('--model_path',     type=str,
                        default=os.environ.get('NEMOTRON_MODEL_PATH'))
    parser.add_argument('--enable_thinking', action='store_true')
    parser.add_argument('--record_process', action=argparse.BooleanOptionalAction,
                        default=False)
    parser.add_argument('--max_samples',    type=int, default=-1)
    parser.add_argument('--start_index',    type=int, default=0)
    parser.add_argument('--dry_run',        action='store_true')
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
    print(f'[opera] beams={args.num_beams} sigma={args.scale_factor} '
          f'N_can={args.num_attn_candidates} alpha={args.penalty_weights} '
          f'r={args.threshold} window={args.window_size} '
          f'rollback={not args.no_rollback}', flush=True)

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
                vis_mask, n_vis = visual_token_mask(inputs['input_ids'], model)
                if n_vis == 0:
                    print('  [degraded] no image token: first 10 steps carry no '
                          'penalty (plain beam search)', flush=True)
                response, gen_ids, info = opera_generate(
                    model, tokenizer, inputs, vis_mask,
                    max_new_tokens=args.max_new_tokens,
                    num_beams=args.num_beams, n_can=args.num_attn_candidates,
                    sigma=args.scale_factor, alpha=args.penalty_weights,
                    r=args.threshold, window_size=args.window_size,
                    enable_rollback=not args.no_rollback)

                elapsed = time.time() - t0
                pred = parse_pred(response, args.datasets)
                result = {
                    'id': item_id, 'pred': pred, 'raw_output': response,
                    'gt': item.get('gt', item.get('answer', item.get('label', ''))),
                    'question': item.get('question', item.get('input', '')),
                    'group_key': item.get('group_key', item.get('sub_category', 'Default')),
                    'sub_category': item.get('sub_category', item.get('group_key', 'Default')),
                    'dataset': args.datasets, 'method': 'opera',
                    'has_visual': n_vis > 0, 'n_image_tokens': n_vis,
                    'time_sec': round(elapsed, 3), 'status': 'success',
                    'input_len': input_len, 'output_tokens': len(gen_ids),
                }
                result.update(info)
                result['nemotron'] = {
                    'method': 'opera',
                    'adaptation': ('last-attention-block row hook (eager impl '
                                   'returns weights); hybrid-cache surgery with '
                                   'mamba state snapshot/restore around probes '
                                   '(see header)'),
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
        tf.write(f'total_samples: {total}\nnum_beams: {args.num_beams}\n')
        tf.write(f'scale_factor: {args.scale_factor}\n')
        tf.write(f'num_attn_candidates: {args.num_attn_candidates}\n')
        tf.write(f'penalty_weights: {args.penalty_weights}\n')
        tf.write(f'threshold: {args.threshold}\n')
        tf.write(f'window_size: {args.window_size}\n')
    print(f'\n[done] {total} samples | {output_path}', flush=True)


if __name__ == '__main__':
    main()
