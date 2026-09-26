# -*- coding: utf-8 -*-
"""
base_mad_nemotron_hf.py -- MAD (Modality-Adaptive Decoding) baseline on
Nemotron-3-Nano-Omni-30B-A3B (transformers trust_remote_code path).

Algorithm 1:1 from methods/base_mad.py (Qwen2.5-Omni arm; line-by-line
reproduction of the OFFICIAL mechanism, github.com/top-yun/MAD HEAD e6f5072,
qwen-omni/utils.py::mm_contrast_decode_qwen). Official hyperparameters are
NOT retuned:

  gamma = 2.5 (CMM default & README --gamma 2.5, paper 4.1.4)
  fused_logits = a_av * L_av + a_v * L_v + a_a * L_a + a_t * L_t   (argmax)
    alpha_av = 2 + 2 * gamma * both_prob
    alpha_v  = 1 - (both_prob - video_prob) * gamma
    alpha_a  = 1 - (both_prob - audio_prob) * gamma
    alpha_t  =   - (video_prob + audio_prob) * gamma
  Self-assessment: ONE forward on the AV conversation with the question plus
  "\\n" + MODALITY_QUERY_PROMPT appended, reading the last-position logit at
  the three BARE single tokens encode("audio")/encode("video")/
  encode("both")[0] (probed on this checkpoint: 57401 / 24073 / 36203, all
  single-token encodings), softmax over the triple.
  Four independent REAL reduced-modality branches (REMOVAL, not masking):
  av = video+audio, v = video only, a = audio only, t = text only.

Nemotron port deltas vs base_mad.py (backbone adaptation ONLY):
  1. Branch inputs are built with shared/nemotron_methods.inputs_from_text
     (processor chat template + per-branch media prefix <video>/
     <so_embedding>/<image> and media kwargs), then prefilled with
     nemotron_methods.llm_prefill on model.language_model over fused
     inputs_embeds (shared/nemotron_shim.fuse_inputs_embeds, inference_mode --
     the 2026-08-30 OOM fix). NO mRoPE, so no position_ids hooking: positions
     auto-derive from cache_position.
  2. `use_audio_in_video` is a Qwen-Omni-only flag. On Nemotron the av branch
     is simply the input carrying BOTH media streams (<video> + <so_embedding>
     placeholders both expanded); each reduced branch drops the corresponding
     media from the prompt entirely -- the same removal semantics.
  3. Mamba2 hybrid cache: each of the four branches owns the
     NemotronHHybridDynamicCache created by ITS OWN prefill; the lockstep
     decode advances every branch exactly once per generated token (the same
     correctness-critical invariant as base_cad_nemotron_hf). There are NO
     probe forwards on a shared cache in MAD (unlike OPERA), so the
     cache_snapshot_mamba/cache_restore_mamba surgery from nemotron_methods is
     not needed; branches never share state.
  4. eos: tokenizer.eos_token_id (11 = config eos) with NEMOTRON_EOS_ID
     fallback.
  5. Media loading via nemotron_methods.load_item_media
     (load_video_frames_native / load_audio_wave / load_image).

Protocol parity with the REGISTERED Qwen2.5 cells (outputs/cmm_base_mad,
outputs/avhbench_base_mad): shared v1 prompt, --max_new_tokens 256 (the launch
cap used by gpu_batch/run_mad_2gpu.sh and run_avh_gpu5.sh; official eval uses
1 for Yes/No, the registered cross-backbone protocol uses the same 256 cap so
cells are comparable). run_nemotron_grid_cell.sh passes no --max_new_tokens,
hence the default here is 256.

Datasets in scope (AV only; official MAD forces video input):
  cmm, avhbench -> full 4-branch MAD. Items without video AND audio (text-only
  or image-only) are out of official MAD scope; the runner degenerates to
  greedy and records degraded_reason (never claimed as MAD).

Usage:
    CUDA_VISIBLE_DEVICES=6 python base_mad_nemotron_hf.py --datasets cmm \
        --data_path ./OmniHalluBench/cmm/cmm.json \
        --output_dir ./outputs --file_name cmm_nemotron_mad_smoke --max_samples 3
"""
import os, sys, time, json, argparse, gc
import torch
import torch.nn.functional as F

REPO = os.environ.get('NEMO_REPO', '/home/tsq/rhq/omniconfess/OmniConfess')
sys.path.insert(0, os.path.join(REPO, 'OmniConfess', 'methods'))
from shared import load_dataset, get_system_prompt, get_user_prompt
from shared.eval_utils import parse_pred
from shared.nemotron_shim import load_nemotron
from shared.nemotron_methods import (
    inputs_from_text, load_item_media, llm_prefill, llm_step, set_seed,
    NEMOTRON_EOS_ID,
)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ── MAD hyperparameters (official defaults; do NOT tune) ─────────────────────
MAD_GAMMA = 2.5
# Official prompts (utils.py lines 5-6), verbatim:
MODALITY_QUERY_PROMPT = ("To answer this question, which modality is needed "
                         "(audio, video, or both): ")


# ═══════════════════════════════════════════════════════════════════════════
#  Self-assessment modality tokens (official: bare single tokens).
# ═══════════════════════════════════════════════════════════════════════════
def modality_token_ids(tokenizer):
    """Official: encode('audio'/'video'/'both')[0] -- the first (bare) token id.
    Probed on this checkpoint: all three are single-token encodings."""
    return {
        "a":  tokenizer.encode("audio")[0],
        "v":  tokenizer.encode("video")[0],
        "av": tokenizer.encode("both")[0],
    }


def self_assess(model, tokenizer, processor, system_prompt, head_text,
                media, device, mod_ids):
    """One prefill on the AV+modality-query input -> (audio, video, both) probs.

    Mirrors utils.py lines 85-99: read last-position logits at the three bare
    token ids, softmax over [audio, video, both].
    """
    inputs, _ = inputs_from_text(processor, tokenizer, system_prompt,
                                 head_text, media, device)
    logits, _cache, _mask = llm_prefill(model, inputs)   # cache discarded
    logits = logits[0].float()   # llm_prefill already returns last-position [B,V]
    audio_logit = logits[mod_ids["a"]].item()
    video_logit = logits[mod_ids["v"]].item()
    both_logit = logits[mod_ids["av"]].item()
    probs = F.softmax(torch.tensor([audio_logit, video_logit, both_logit]), dim=0)
    del inputs, _cache
    torch.cuda.empty_cache()
    return probs.tolist()   # (audio_prob, video_prob, both_prob)


# ═══════════════════════════════════════════════════════════════════════════
#  MAD generation (official weighted-sum fusion; 4-branch lockstep decode).
# ═══════════════════════════════════════════════════════════════════════════
def mad_generate(model, tokenizer, processor, system_prompt, question_text,
                 media, device, mod_ids, gamma=MAD_GAMMA, max_new_tokens=256,
                 record_process=False, verbose=True):
    """Official mm_contrast_decode_qwen reproduction on the Nemotron backbone.

    `media` = (videos, images, audio) as returned by load_item_media.
    Returns (response, gen_ids, info, logp_list).
    """
    videos, images, audio = media
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        eos_id = NEMOTRON_EOS_ID
    info = {}

    has_video = videos is not None
    has_audio = audio is not None
    info["modalities"] = {"video": bool(has_video), "audio": bool(has_audio),
                          "image": bool(images is not None)}

    if not has_video and not has_audio:
        # Out of official MAD scope (no AV). Degenerate to greedy for parity.
        info["mad_mode"] = "greedy"
        inputs, in_len = inputs_from_text(processor, tokenizer, system_prompt,
                                          question_text, media, device)
        L, pkv, attn = llm_prefill(model, inputs)
        del inputs
        out = _greedy_decode(model, tokenizer, L, pkv, attn, eos_id,
                             max_new_tokens, record_process, info, in_len,
                             device)
        return out

    info["mad_mode"] = "mad"

    # ── Self-assessment forward: AV "head" conversation + modality query ──
    head_text = question_text + "\n" + MODALITY_QUERY_PROMPT
    head_media = (videos, images, audio)          # full AV evidence
    audio_prob, video_prob, both_prob = self_assess(
        model, tokenizer, processor, system_prompt, head_text, head_media,
        device, mod_ids)
    gc.collect(); torch.cuda.empty_cache()

    g = gamma
    alpha_av = 2 + 2 * g * both_prob
    alpha_v = 1 - (both_prob - video_prob) * g
    alpha_a = 1 - (both_prob - audio_prob) * g
    alpha_t = -(video_prob + audio_prob) * g
    weights = {"av": alpha_av, "v": alpha_v, "a": alpha_a, "t": alpha_t}
    info["probs"] = {"audio": round(audio_prob, 6), "video": round(video_prob, 6),
                     "both": round(both_prob, 6)}
    info["alpha"] = {k: round(v, 6) for k, v in weights.items()}
    if verbose:
        print(f"  [mad] probs(a,v,both)=({audio_prob:.3f},{video_prob:.3f},"
              f"{both_prob:.3f}) alpha={info['alpha']} gamma={g}", flush=True)

    # ── Build & prefill the four reduced-modality branches ──
    # (question_text = bare question, NO modality query; REMOVAL not masking)
    branch_specs = {
        "av": (videos, images, audio),   # both streams (Qwen uaiv=True analog)
        "v":  (videos, images, None),    # visual only
        "a":  (None,  None,  audio),     # audio only
        "t":  (None,  None,  None),      # text only
    }
    order = ["av", "v", "a", "t"]
    step_logits, pkvs, attns = {}, {}, {}
    for key in order:
        binp, in_len = inputs_from_text(
            processor, tokenizer, system_prompt, question_text,
            branch_specs[key], device)
        L, pkv, mask = llm_prefill(model, binp)
        step_logits[key] = L
        pkvs[key] = pkv
        attns[key] = mask
        if verbose:  # smoke evidence: real removal (placeholder counts/branch)
            ids_i = binp["input_ids"][0]
            n_vis = int((ids_i == model.img_context_token_id).sum().item())
            n_aud = int((ids_i == model.sound_context_token_id).sum().item()) \
                if getattr(model, "sound_context_token_id", None) is not None else 0
            print(f"  [mad] branch {key}: len={in_len} vis_tok={n_vis} "
                  f"aud_tok={n_aud} logits={tuple(L.shape)}", flush=True)
        if key == "av":
            info["input_len"] = int(in_len)
        del binp, L
        gc.collect(); torch.cuda.empty_cache()
    info["n_branches"] = len(order)

    def _fuse():
        return sum(weights[k] * step_logits[k] for k in order)

    generated, logp_list = [], []
    with torch.inference_mode():
        fused = _fuse()
        cur = fused.argmax(dim=-1, keepdim=True)
        tok = cur.item()
        generated.append(tok)
        if record_process:
            lp = F.log_softmax(fused.float(), dim=-1)[0, tok]
            logp_list.append(round(lp.item(), 4))

        for _ in range(1, max_new_tokens):
            if tok == eos_id:
                break
            if len(generated) >= 5 and len(set(generated[-5:])) == 1:
                break
            # lockstep: every branch's Mamba2 state advances exactly once
            for key in order:
                step_logits[key], pkvs[key], attns[key] = llm_step(
                    model, cur, attns[key], pkvs[key])
            fused = _fuse()
            cur = fused.argmax(dim=-1, keepdim=True)
            tok = cur.item()
            generated.append(tok)
            if record_process:
                lp = F.log_softmax(fused.float(), dim=-1)[0, tok]
                logp_list.append(round(lp.item(), 4))

    del pkvs
    gc.collect(); torch.cuda.empty_cache()
    info["n_tokens"] = len(generated)
    resp = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return resp, generated, info, logp_list


def _greedy_decode(model, tokenizer, L0, pkv0, attn0, eos_id,
                   max_new_tokens, record_process, info, input_len, device):
    """Plain greedy fallback for out-of-scope (no-AV) items."""
    generated, logp_list = [], []
    with torch.inference_mode():
        L = L0
        cur = L.argmax(dim=-1, keepdim=True)
        tok = cur.item()
        generated.append(tok)
        if record_process:
            logp_list.append(round(F.log_softmax(L.float(), dim=-1)[0, tok].item(), 4))
        for _ in range(1, max_new_tokens):
            if tok == eos_id:
                break
            if len(generated) >= 5 and len(set(generated[-5:])) == 1:
                break
            L, pkv0, attn0 = llm_step(model, cur, attn0, pkv0)
            cur = L.argmax(dim=-1, keepdim=True)
            tok = cur.item()
            generated.append(tok)
            if record_process:
                logp_list.append(round(F.log_softmax(L.float(), dim=-1)[0, tok].item(), 4))
    info["n_tokens"] = len(generated)
    info["n_branches"] = 1
    return (tokenizer.decode(generated, skip_special_tokens=True).strip(),
            generated, info, logp_list)


# ═══════════════════════════════════════════════════════════════════════════
def attach_trajectory(result, gen_ids, logp_list, input_len):
    """phi-Decoding style per-token trajectory fields (only when
    --record_process; Nemotron grid cells run with --no-record_process)."""
    result["record_process"] = True
    if gen_ids:
        cum = float(sum(logp_list)) if logp_list else 0.0
        result["token_ids"] = list(gen_ids)
        result["logp_list"] = list(logp_list)
        result["cumulative_logprob"] = round(cum, 4)
        result["norm_logp"] = round(cum / (len(gen_ids) + 1e-8), 4)
        result["input_len"] = int(input_len) if input_len is not None else None
        result["output_tokens"] = len(gen_ids)
    else:
        for k in ("token_ids", "logp_list", "cumulative_logprob",
                  "norm_logp", "input_len"):
            result[k] = None
        result["output_tokens"] = 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets',      type=str, required=True)
    parser.add_argument('--data_path',     type=str, required=True)
    parser.add_argument('--data_root',     type=str, default=None)
    parser.add_argument('--output_dir',    type=str, default='./outputs')
    parser.add_argument('--file_name',     type=str, default='base_mad_nemotron')
    parser.add_argument('--time_path',     type=str, default='./outputs/time')
    # Registered cross-backbone protocol cap: the Qwen2.5 registered MAD cells
    # both launched with 256 (official MAD eval uses 1 for Yes/No).
    parser.add_argument('--max_new_tokens', type=int, default=256)
    parser.add_argument('--gamma',          type=float, default=MAD_GAMMA,
                        help='MAD base contrastive strength (official default 2.5)')
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
    mod_ids = modality_token_ids(tokenizer)
    print(f"[mad] gamma={args.gamma} | modality token ids={mod_ids} "
          f"(bare 'audio'/'video'/'both') | eos={tokenizer.eos_token_id}",
          flush=True)
    print(f"[mad] provenance: base_mad.py (official qwen-omni/utils.py "
          f"mm_contrast_decode_qwen @ e6f5072, weighted-sum of 4 reduced-modality "
          f"forwards) on Nemotron via nemotron_methods prefill pipeline",
          flush=True)

    data = load_dataset(args.data_path, args.data_root)
    if args.start_index > 0:
        data = data[args.start_index:]
    if args.dry_run:
        data = data[:3]
    elif args.max_samples > 0:
        data = data[: args.max_samples]
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
                media = load_item_media(item, args)
                user_prompt = get_user_prompt(
                    item['question'], args.datasets, args.prompt_version,
                    passage=item.get('passage'))

                response, gen_ids, info, logp_list = mad_generate(
                    model, tokenizer, processor, system_prompt, user_prompt,
                    media, DEVICE, mod_ids, gamma=args.gamma,
                    max_new_tokens=args.max_new_tokens,
                    record_process=args.record_process)
                input_len = info.get('input_len')
                if input_len is None:
                    input_len = 0

                elapsed = time.time() - t0
                pred = parse_pred(response, args.datasets)
                result = {
                    'id': item_id, 'pred': pred, 'raw_output': response,
                    'gt': item.get('gt', item.get('answer', item.get('label', ''))),
                    'question': item.get('question', item.get('input', '')),
                    'group_key': item.get('group_key', item.get('sub_category', 'Default')),
                    'sub_category': item.get('group_key', item.get('sub_category', 'Default')),
                    'dataset': args.datasets, 'method': 'mad',
                    'backbone': 'nemotron',
                    'has_mad': info['mad_mode'] == 'mad',
                    'time_sec': round(elapsed, 3), 'status': 'success',
                    'output_tokens': len(gen_ids),
                }
                result.update(info)
                if info['mad_mode'] == 'greedy':
                    result['degraded_reason'] = (
                        'no AV media: out of official MAD scope -> greedy')
                result['nemotron'] = {
                    'method': 'mad',
                    'adaptation': ('4 reduced-modality branches prefilled on '
                                   'model.language_model over fused '
                                   'inputs_embeds, each with its own '
                                   'NemotronHHybridDynamicCache advanced once '
                                   'per token; no use_audio_in_video flag on '
                                   'this backbone (av branch = both media '
                                   'streams in the prompt); no probe forwards '
                                   '-> no Mamba snapshot/restore needed'),
                    'runtime': 'transformers-trust-remote-code',
                }
                if args.record_process:
                    attach_trajectory(result, gen_ids, logp_list, input_len)

                res_f.write(json.dumps(result, ensure_ascii=False) + '\n')
                res_f.flush()
                print(f"  pred: {pred!r}  raw: {response[:60]!r}  "
                      f"gt: {result['gt']!r}  ({elapsed:.1f}s)", flush=True)

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); gc.collect()
                print(f'  [OOM] id={item_id}', flush=True)
                with open(error_path, 'a') as ef:
                    ef.write(f"{time.strftime('%H:%M:%S')} | id={item_id} | OOM\n")
                res_f.write(json.dumps({'id': item_id, 'status': 'OOM',
                                        'error': 'CUDA OOM'}) + '\n')
                res_f.flush()

            except Exception as e:
                import traceback
                torch.cuda.empty_cache(); gc.collect()
                print(f'  [ERR] id={item_id}: {repr(e)}', flush=True)
                with open(error_path, 'a') as ef:
                    ef.write(f"{time.strftime('%H:%M:%S')} | id={item_id} | {repr(e)}\n")
                    ef.write(traceback.format_exc() + '\n')
                res_f.write(json.dumps({'id': item_id, 'status': 'failed',
                                        'error': repr(e)[:200]}) + '\n')
                res_f.flush()

            finally:
                gc.collect(); torch.cuda.empty_cache()

    with open(os.path.join(args.time_path, args.file_name + '.txt'), 'w') as tf:
        tf.write(f'time: {time.time()-start:.1f}s\n')
        tf.write(f'total_samples: {total}\ngamma: {args.gamma}\n')
    print(f'\n[done] {total} samples in {(time.time()-start)/60:.1f} min. '
          f'Results: {output_path}', flush=True)


if __name__ == '__main__':
    main()
