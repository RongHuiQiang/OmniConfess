# -*- coding: utf-8 -*-
"""
base_mad_qwen3.py -- MAD (Modality-Adaptive Decoding) baseline on
Qwen3-Omni-30B-A3B-Instruct (thinker-only, via shared/qwen3_shim.py).

Algorithm port 1:1 from methods/base_mad.py (Qwen2.5-Omni arm), which is a
line-by-line reproduction of the OFFICIAL mechanism
(github.com/top-yun/MAD HEAD e6f5072, qwen-omni/utils.py::
mm_contrast_decode_qwen). Official hyperparameters are NOT retuned:

  gamma = 2.5 (CMM default & README --gamma 2.5, paper 4.1.4)
  fused_logits = a_av * L_av + a_v * L_v + a_a * L_a + a_t * L_t
    alpha_av = 2 + 2 * gamma * both_prob
    alpha_v  = 1 - (both_prob - video_prob) * gamma
    alpha_a  = 1 - (both_prob - audio_prob) * gamma
    alpha_t  =   - (video_prob + audio_prob) * gamma
  Self-assessment: ONE forward on the AV conversation with
  question + "\\n" + MODALITY_QUERY_PROMPT appended after the question, reading
  the last-position logit at the three BARE single tokens
  encode("audio")/encode("video")/encode("both")[0], softmax over the triple.
  Four independent REAL reduced-modality branches (own KV cache), removal not
  masking: av = video+audio (use_audio_in_video=True), v = video only,
  a = audio only, t = text only.

Qwen3-Omni port deltas vs base_mad.py (backbone adaptation ONLY):
  1. Model layer goes through shared/qwen3_shim.py: loads the Qwen3-Omni
     THINKER itself and attaches the self-alias model.thinker == model, so
     every model.thinker.* call site of the Qwen2.5 arm resolves unchanged.
     The shim also pins tokenizer eos/pad (Qwen3 generation_config ships
     without eos_token_id; tokenizer already reports <|im_end|>=151645, so the
     manual decode loop's `tok == eos_id` break is correct).
  2. Inputs come from shared/qwen3_omni_utils: media placeholder spellings
     <|vision_start|><|image_pad|>/<|video_pad|><|vision_end|> and
     <|audio_start|><|audio_pad|><|audio_end|>, thinker image key
     `pixel_values` (not pixel_values_images), video_second_per_grid input,
     24k->16k audio resample, scalar fps. VALID_KEYS below follows the Qwen3
     thinker signature.
  3. The Qwen3-Omni thinker forward natively accepts `use_audio_in_video`
     (modeling_qwen3_omni_moe.py, thinker forward signature), so the official
     av-branch flag is passed exactly as in the Qwen2.5 arm.
  4. mRoPE position bookkeeping on step decode is kept identical to the Qwen2.5
     arm (forward_pre_hook capture on model.thinker.model, +1 per step), the
     same pattern already proven in base_cad_qwen3.py / base_avcd_qwen3.py.
  5. Modality self-assessment bare tokens probed on this checkpoint:
     audio=16736, video=9986, both=21028 (all single-token encodings) --
     computed at runtime via tokenizer.encode(...)[0], official style.

Protocol parity with the REGISTERED Qwen2.5 cells (outputs/cmm_base_mad,
outputs/avhbench_base_mad): shared v1 prompt, --max_new_tokens 256 (the launch
cap used by gpu_batch/run_mad_2gpu.sh and run_avh_gpu5.sh for both registered
cells; official eval uses 1 for Yes/No, our registered cross-backbone protocol
uses the same 256 cap so cells are comparable). run_qwen3_grid_cell.sh passes
no --max_new_tokens, hence the default here is 256.

Datasets in scope (AV only; official MAD forces video input):
  cmm, avhbench (video + audio + text) -> full 4-branch MAD.
  Text-only / image-only sets are out of official MAD scope; the runner
  degenerates to greedy and records degraded_reason (never claimed as MAD).

Usage:
    python base_mad_qwen3.py --datasets cmm \
        --data_path ./OmniHalluBench/cmm/cmm.json \
        --output_dir ./outputs --file_name cmm_qwen3_mad_smoke --max_samples 3
"""
import os, sys, json, time, argparse, gc
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shared import (
    load_dataset, load_media_for_item,
    get_system_prompt, get_user_prompt,
)
from shared.qwen3_shim import (
    load_model, build_inputs, prepare_for_thinker,
    build_user_content_with_media_tokens, make_messages,
)
from shared.eval_utils import parse_pred

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ── MAD hyperparameters (official defaults; do NOT tune) ─────────────────────
MAD_GAMMA = 2.5   # CMM default & README `--gamma 2.5`; paper 4.1.4 (swept 0.5..3.0)
# Official prompts (utils.py lines 5-6), verbatim:
MODALITY_QUERY_PROMPT = ("To answer this question, which modality is needed "
                         "(audio, video, or both): ")
ANSWER_QUERY_PROMPT = (" Answer only 'Yes' or 'No'. Do not include any explanation.")
TEXT_ONLY_DS = {"halueval", "ragtruth", "pubmedqa"}

# Qwen3-Omni thinker input keys (cf. shared/qwen3_omni_utils._THINKER_VALID_KEYS;
# the Qwen2.5 arm's pixel_values_images does not exist here, image = pixel_values)
VALID_KEYS = {
    "input_ids", "attention_mask",
    "pixel_values_videos", "video_grid_thw", "video_second_per_grid",
    "pixel_values", "image_grid_thw",
    "feature_attention_mask", "input_features",
}


# ═══════════════════════════════════════════════════════════════════════════
#  Branch input construction: real reduced-modality inputs (official REMOVAL).
# ═══════════════════════════════════════════════════════════════════════════
def _clean_to_device(d, device):
    out = {k: v for k, v in d.items() if k in VALID_KEYS}
    for k, v in out.items():
        if isinstance(v, torch.Tensor):
            out[k] = (v.to(dtype=torch.bfloat16, device=device)
                      if torch.is_floating_point(v) else v.to(device))
    return out


def build_branch(processor, model, system_prompt, question_text, dataset_name,
                 pixel_values, audio_values, image_values):
    """Tokenise ONE reduced-modality conversation.

    `pixel_values` / `audio_values` / `image_values` are either the loaded media
    array or None; passing None omits that modality's placeholder token and
    features entirely -> a genuine reduced-modality input (official removal).
    Returns the thinker-ready, device-placed input dict.
    """
    user_content = build_user_content_with_media_tokens(
        pixel_values, audio_values, image_values, question_text)
    messages = make_messages(system_prompt, user_content)
    raw = build_inputs(processor, model, messages,
                        pixel_values, audio_values, image_values, dataset_name)
    return _clean_to_device(prepare_for_thinker(raw, model), model.device)


# ═══════════════════════════════════════════════════════════════════════════
#  Self-assessment modality tokens (official: bare single tokens).
# ═══════════════════════════════════════════════════════════════════════════
def modality_token_ids(tokenizer):
    """Official: encode('audio'/'video'/'both')[0] -- the first (bare) token id.
    Probed on Qwen3-Omni: all three are single-token encodings."""
    return {
        "a":  tokenizer.encode("audio")[0],
        "v":  tokenizer.encode("video")[0],
        "av": tokenizer.encode("both")[0],
    }


def self_assess(model, assess_inputs, mod_ids):
    """One forward on the AV+modality-query input -> (audio, video, both) probs.

    Mirrors utils.py lines 85-99: read last-position logits at the three bare
    token ids, softmax over [audio, video, both].
    """
    with torch.inference_mode():
        out = model.thinker(**assess_inputs, use_audio_in_video=True)
        logits = out.logits[0, -1, :].float()
    audio_logit = logits[mod_ids["a"]].item()
    video_logit = logits[mod_ids["v"]].item()
    both_logit = logits[mod_ids["av"]].item()
    probs = F.softmax(torch.tensor([audio_logit, video_logit, both_logit]), dim=0)
    audio_prob, video_prob, both_prob = probs.tolist()
    return audio_prob, video_prob, both_prob


# ═══════════════════════════════════════════════════════════════════════════
#  Prefill / decode helpers (mRoPE position bookkeeping, cf. base_vcd).
# ═══════════════════════════════════════════════════════════════════════════
def prefill_branch(model, inputs, use_audio_in_video):
    """Prefill one branch, capturing its mRoPE position_ids.
    Returns (last_logits, past_key_values, next_position_ids)."""
    captured = {}

    def _pos_hook(module, args, kwargs):
        if kwargs.get("position_ids") is not None and "position_ids" not in captured:
            captured["position_ids"] = kwargs["position_ids"].detach().clone()

    h = model.thinker.model.register_forward_pre_hook(_pos_hook, with_kwargs=True)
    try:
        with torch.inference_mode():
            out = model.thinker(**inputs, past_key_values=None, use_cache=True,
                                use_audio_in_video=use_audio_in_video)
    finally:
        h.remove()
    pos = captured.get("position_ids")
    pos_next = pos[..., -1:] + 1 if pos is not None else None
    return out.logits[:, -1, :], out.past_key_values, pos_next


# ═══════════════════════════════════════════════════════════════════════════
#  MAD generation (official weighted-sum fusion).
# ═══════════════════════════════════════════════════════════════════════════
def mad_generate(model, processor, tokenizer, system_prompt, question_text,
                 dataset_name, pixel_values, audio_values, image_values,
                 mod_ids, gamma=MAD_GAMMA, max_new_tokens=256,
                 record_process=True, verbose=True):
    """Official mm_contrast_decode_qwen reproduction.

    Returns (response, gen_ids, info, logp_list). logp_list holds log_softmax of
    the FUSED logits at each selected token (phi-Decoding trajectory style).
    """
    device = next(model.parameters()).device
    eos_id = tokenizer.eos_token_id
    info = {}

    has_video = pixel_values is not None
    has_audio = audio_values is not None
    info["modalities"] = {"video": bool(has_video), "audio": bool(has_audio),
                          "image": bool(image_values is not None)}

    if not has_video and not has_audio:
        # Out of official MAD scope (no AV). Degenerate to greedy for parity.
        info["mad_mode"] = "greedy"
        assess_inputs = build_branch(processor, model, system_prompt,
                                     question_text, dataset_name,
                                     pixel_values, audio_values, image_values)
        L, pkv, pos = prefill_branch(model, assess_inputs, use_audio_in_video=False)
        attn = assess_inputs["attention_mask"]
        return _greedy_decode(model, tokenizer, L, pkv, pos, attn, device,
                              eos_id, max_new_tokens, record_process, info,
                              assess_inputs["input_ids"].shape[1])

    info["mad_mode"] = "mad"

    # ── Self-assessment forward: AV "head" conversation + modality query ──
    # head text = question (already has ANSWER_QUERY_PROMPT) + "\n" + MODALITY_QUERY_PROMPT
    head_text = question_text + "\n" + MODALITY_QUERY_PROMPT
    assess_inputs = build_branch(processor, model, system_prompt, head_text,
                                 dataset_name, pixel_values, audio_values,
                                 image_values)
    audio_prob, video_prob, both_prob = self_assess(model, assess_inputs, mod_ids)
    del assess_inputs
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
    # (question_text = bare question with ANSWER_QUERY_PROMPT, NO modality query)
    branch_specs = {
        "av": dict(px=pixel_values, au=audio_values, im=image_values, uaiv=True),
        "v":  dict(px=pixel_values, au=None,         im=image_values, uaiv=False),
        "a":  dict(px=None,         au=audio_values, im=None,         uaiv=True),
        "t":  dict(px=None,         au=None,         im=None,         uaiv=False),
    }
    order = ["av", "v", "a", "t"]
    step_logits, pkvs, poss, attns = {}, {}, {}, {}
    for key in order:
        spec = branch_specs[key]
        binp = build_branch(processor, model, system_prompt, question_text,
                            dataset_name, spec["px"], spec["au"], spec["im"])
        L, pkv, pos = prefill_branch(model, binp, use_audio_in_video=spec["uaiv"])
        step_logits[key] = L
        pkvs[key] = pkv
        poss[key] = pos
        attns[key] = binp["attention_mask"]
        if verbose:  # smoke evidence: real removal (media keys per branch)
            media_keys = sorted(k for k in binp
                                if k in ("pixel_values_videos", "pixel_values",
                                         "input_features"))
            print(f"  [mad] branch {key}: len={binp['input_ids'].shape[1]} "
                  f"media={media_keys or 'none'} logits={tuple(L.shape)}",
                  flush=True)
        if key == "av":
            info["input_len"] = int(binp["input_ids"].shape[1])
        del binp
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
            for key in order:
                uaiv = branch_specs[key]["uaiv"]
                attns[key] = torch.cat(
                    [attns[key], torch.ones(1, 1, device=device, dtype=torch.long)],
                    dim=-1)
                kw = dict(input_ids=cur, attention_mask=attns[key],
                          past_key_values=pkvs[key], use_cache=True,
                          use_audio_in_video=uaiv)
                if poss[key] is not None:
                    kw["position_ids"] = poss[key]
                    poss[key] = poss[key] + 1
                out = model.thinker(**kw)
                step_logits[key] = out.logits[:, -1, :]
                pkvs[key] = out.past_key_values
            fused = _fuse()
            cur = fused.argmax(dim=-1, keepdim=True)
            tok = cur.item()
            generated.append(tok)
            if record_process:
                lp = F.log_softmax(fused.float(), dim=-1)[0, tok]
                logp_list.append(round(lp.item(), 4))

    info["n_tokens"] = len(generated)
    resp = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return resp, generated, info, logp_list


def _greedy_decode(model, tokenizer, L0, pkv0, pos0, attn0, device, eos_id,
                   max_new_tokens, record_process, info, input_len):
    """Plain greedy fallback for out-of-scope (no-AV) items."""
    generated, logp_list = [], []
    pkv, pos, attn = pkv0, pos0, attn0
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
            attn = torch.cat([attn, torch.ones(1, 1, device=device, dtype=torch.long)], dim=-1)
            kw = dict(input_ids=cur, attention_mask=attn, past_key_values=pkv, use_cache=True)
            if pos is not None:
                kw["position_ids"] = pos
                pos = pos + 1
            out = model.thinker(**kw)
            L = out.logits[:, -1, :]
            pkv = out.past_key_values
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
    """phi-Decoding style per-token trajectory fields (cf. base_model.py)."""
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
    parser.add_argument("--datasets",       type=str, required=True)
    parser.add_argument("--data_path",      type=str, required=True)
    parser.add_argument("--data_root",      type=str, default="./OmniHalluBench/")
    parser.add_argument("--output_dir",     type=str, default="./results")
    parser.add_argument("--file_name",      type=str, default="base_mad_qwen3_out")
    parser.add_argument("--time_path",      type=str, default="./results/time")
    parser.add_argument("--max_samples",    type=int, default=-1)
    parser.add_argument("--start_index",    type=int, default=0,
                        help="Start from sample start_index (0-based); makes the "
                             "fallback idx global for id-less datasets so shards "
                             "do not collide on 0..N.")
    # Registered cross-backbone protocol cap: the Qwen2.5 registered MAD cells
    # (gpu_batch/run_mad_2gpu.sh, run_avh_gpu5.sh) both launched with 256.
    # Official MAD eval uses 1 (Yes/No); parity with the registered cells wins.
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--gamma",          type=float, default=MAD_GAMMA,
                        help="MAD base contrastive strength (official default 2.5)")
    parser.add_argument("--prompt_version", type=str, default="v1")
    parser.add_argument("--record_process", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Record per-token trajectory (token ids, per-token "
                             "logprobs, cumulative/norm logp) of the FUSED logits")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.time_path,  exist_ok=True)

    output_path = os.path.join(args.output_dir, args.file_name + ".jsonl")
    error_path  = os.path.join(args.output_dir, f"ERROR-{args.file_name}.log")

    done_ids = set()
    if os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line.strip())["id"])
                except Exception:
                    pass
        print(f"[resume] skipping {len(done_ids)} done samples", flush=True)

    print("[model] loading Qwen3-Omni thinker...", flush=True)
    processor, model = load_model()
    tokenizer = processor.tokenizer
    mod_ids = modality_token_ids(tokenizer)
    print(f"[mad] gamma={args.gamma} | modality token ids={mod_ids} "
          f"(bare 'audio'/'video'/'both')", flush=True)
    print(f"[mad] provenance: base_mad.py (official qwen-omni/utils.py "
          f"mm_contrast_decode_qwen @ e6f5072, weighted-sum of 4 reduced-modality "
          f"forwards) on the Qwen3-Omni thinker via shared/qwen3_shim.py",
          flush=True)

    data = load_dataset(args.data_path, args.data_root)
    if args.start_index > 0:
        data = data[args.start_index:]
    if args.max_samples > 0:
        data = data[: args.max_samples]
    total = len(data)
    print(f"[data] {total} samples | dataset={args.datasets}", flush=True)

    system_prompt = get_system_prompt(args.datasets, args.prompt_version)
    is_text_only  = args.datasets.lower() in TEXT_ONLY_DS
    start = time.time()

    with open(output_path, "a", encoding="utf-8") as res_f:
        for idx, item in enumerate(data, start=args.start_index):
            item_id = item.get("id", idx)
            if item_id in done_ids:
                continue

            t0 = time.time()
            print(f"\n[{idx+1}/{args.start_index+total}] id={item_id}", flush=True)

            try:
                pixel_values, audio_values, image_values = load_media_for_item(
                    item, args.data_path, args.data_root)

                user_prompt = get_user_prompt(
                    item["question"], args.datasets, args.prompt_version,
                    passage=item.get("passage"))

                response, gen_ids, info, logp_list = mad_generate(
                    model, processor, tokenizer, system_prompt, user_prompt,
                    args.datasets, pixel_values, audio_values, image_values,
                    mod_ids, gamma=args.gamma,
                    max_new_tokens=args.max_new_tokens,
                    record_process=args.record_process)
                # input_len: the av-branch prompt length (tracked in info).
                input_len = info.get("input_len")

                elapsed = time.time() - t0
                pred_clean = parse_pred(response, args.datasets)
                result = {
                    "id":           item_id,
                    "pred":         pred_clean,
                    "raw_output":   response,
                    "gt":           item.get("gt", item.get("answer", item.get("label", ""))),
                    "question":     item.get("question", item.get("input", "")),
                    "group_key":    item.get("group_key", item.get("sub_category", "Default")),
                    "sub_category": item.get("group_key", item.get("sub_category", "Default")),
                    "dataset":      args.datasets,
                    "method":       "mad",
                    "backbone":     "qwen3_omni",
                    "has_mad":      info["mad_mode"] == "mad",
                    "time_sec":     round(elapsed, 3),
                    "status":       "success",
                }
                result.update(info)
                if info["mad_mode"] == "greedy":
                    result["degraded_reason"] = (
                        "no AV media: out of official MAD scope -> greedy")
                if args.record_process:
                    attach_trajectory(result, gen_ids, logp_list, input_len)

                res_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                res_f.flush()
                print(f"  pred: {pred_clean!r}  raw: {response[:60]!r}  "
                      f"gt: {result['gt']!r}  ({elapsed:.1f}s)", flush=True)

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); gc.collect()
                print(f"  [OOM] id={item_id}", flush=True)
                with open(error_path, "a") as ef:
                    ef.write(f"{time.strftime('%H:%M:%S')} | id={item_id} | OOM\n")
                _err = {"id": item_id, "status": "OOM", "error": "CUDA OOM"}
                if args.record_process:
                    attach_trajectory(_err, None, None, None)
                res_f.write(json.dumps(_err, ensure_ascii=False) + "\n")
                res_f.flush()

            except Exception as e:
                import traceback
                torch.cuda.empty_cache(); gc.collect()
                print(f"  [ERR] id={item_id}: {repr(e)}", flush=True)
                with open(error_path, "a") as ef:
                    ef.write(f"{time.strftime('%H:%M:%S')} | id={item_id} | {repr(e)}\n")
                    ef.write(traceback.format_exc() + "\n")
                _err = {"id": item_id, "status": "failed", "error": repr(e)[:200]}
                if args.record_process:
                    attach_trajectory(_err, None, None, None)
                res_f.write(json.dumps(_err, ensure_ascii=False) + "\n")
                res_f.flush()

            finally:
                gc.collect(); torch.cuda.empty_cache()

    elapsed_total = time.time() - start
    with open(os.path.join(args.time_path, f"{args.file_name}.txt"), "w") as tf:
        tf.write(f"time: {elapsed_total:.1f}s\n")
        tf.write(f"total_samples: {total}\n")
        tf.write(f"gamma: {args.gamma}\n")
    print(f"\n[done] {total} samples in {elapsed_total/60:.1f} min. "
          f"Results: {output_path}", flush=True)


if __name__ == "__main__":
    main()
