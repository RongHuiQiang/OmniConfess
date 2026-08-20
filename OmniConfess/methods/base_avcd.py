# -*- coding: utf-8 -*-
"""
base_avcd.py -- AVCD (Audio-Visual Contrastive Decoding) baseline on Qwen2.5-Omni-7B.

Reference
---------
Chaeyoung Jung, Youngjoon Jang, Joon Son Chung.
"AVCD: Mitigating Hallucinations in Audio-Visual Large Language Models through
Contrastive Decoding."
The reference implementation is the official VideoLLaMA2-based code.

Core formula (matching the official implementation line for line)
-----------------------------------------------------------------
  contrastive = (2+2a)*logit_orig - 2a*logit_m1 + logit_m2 + logit_m3
  next_logits = contrastive.masked_fill(logit_orig < cutoff, APC_FILL)
  cutoff      = log(beta) + max(logit_orig)          # adaptive plausibility

  m1 masks both non-dominant modalities at once; m2 and m3 each mask one of them.
  The dominant modality is obtained by ranking, across layers, the attention mass
  that the last query places on each modality span during prefill.

Hyperparameter provenance (all taken from the official defaults; nothing tuned)
------------------------------------------------------------------------------
  alpha (cd_alpha)      = 0.5
  beta (APC)            = 0.2
  entropy gate          = 0.6
  APC fill value        = -1e-4
  masked layer range    = all but the last layer (VideoLLaMA2 has 28 layers, and
                          the Qwen2.5-Omni thinker likewise has 28, so this is
                          written as n_layers - 1)
  threshold             = the per-layer sum of last-query attention divided by the
                          number of layers, taken as a median over the key
                          dimension and then averaged over heads

  Note that the APC fill value of -1e-4 is the official choice, whereas the
  original VCD uses -inf. Filling with -1e-4 pulls the contrastive logit of an
  implausible token to approximately zero, which is equivalent to masking only
  when the plausible tokens have positive contrastive logits. This implementation
  keeps the official default and exposes --apc_fill so that passing -inf switches
  to the standard VCD behaviour.

Differences from the official implementation (engineering necessities)
----------------------------------------------------------------------
  1) The official code re-runs a full forward pass over the whole sequence for
     every generated token and uses no KV cache at all, so dominance and threshold
     are recomputed at each step. Here dominance and threshold are computed once
     at prefill, each branch keeps its own KV cache, and decoding proceeds step by
     step. The official masking is itself guarded by a check that the query length
     is greater than one, so it never applies during single-step decoding; the
     semantics of "masking applies only during prefill" therefore match.
  2) The official masking lives inside a modified attention kernel, multiplying
     elementwise by zero after the softmax and renormalising. This implementation
     does the same: the model is loaded without an explicit attn_implementation,
     so Qwen2.5-Omni uses eager attention, and Qwen2_5OmniAttention.forward is
     replaced to insert the same masking plus a clamp(sum, 1e-6) renormalisation
     between the softmax and the multiplication by V. Hugging Face selects the
     attention class at construction time from config._attn_implementation, so
     changing the config after loading has no effect; and after o_proj the heads
     are already merged, so per-head masking is no longer possible. Both facts
     force replacing forward rather than attaching a hook.
  3) The official code partitions modalities using hardcoded token ranges. This
     implementation locates them exactly from input_ids using the
     <|IMAGE|>/<|VIDEO|>/<|AUDIO|> token ids, which is robust and dataset-agnostic.
  4) In the official implementation the entropy gate also saves computation, since
     the contrastive branches are skipped below the threshold. Here the branch KV
     caches are already built and the branches are still stepped, but the token
     selection matches the official behaviour: below an entropy of 0.6 the
     original logits are used directly.

Degenerate behaviour
--------------------
  The AVCD formula needs three modalities (visual, audio, language).

   * avhbench, cmm (video + audio + text): three modalities, so full AVCD runs
     with four forward passes.
   * phd, haloquest (image + text, no audio): only two modalities, and the
     official formula cancels itself out here. With language dominant, m1 is "VA",
     m2 is "A" (an empty audio span, so that branch equals the original) and m3 is
     "V"; substituting gives (2+2a)*orig - 2a*o_V + orig + o_V, which at alpha=0.5
     is exactly 4*orig. The contrastive term vanishes and AVCD becomes plain
     greedy. For two modalities this implementation therefore uses the
     single-instance CD variant that the official code carries as a commented
     line, contrastive = (1+a)*orig - a*o_m1 with m1 the sole non-dominant
     modality, with the same APC and entropy gate. The output is marked
     avcd_mode="single_cd".
   * pubmedqa, ragtruth (text only): a single language modality, so there is no
     audio-visual contrast to draw. This degenerates fully to greedy with no extra
     forward pass, marked has_avcd=False, avcd_mode="greedy",
     degraded_reason="text-only: no AV contrast", matching how base_vcd.py
     degenerates on text-only data.

Usage:
    python base_avcd.py \
        --datasets avhbench --data_path ./OmniHalluBench/avhbench/avhbench.json \
        --output_dir ./outputs --file_name base_avcd_avhbench_v0 \
        --prompt_version v0 --max_samples 2
"""
import os, sys, json, time, argparse, gc, math
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from shared import (
    load_dataset, load_media_for_item,
    load_model, build_inputs, prepare_for_thinker,
    build_user_content_with_media_tokens, make_messages,
    get_system_prompt, get_user_prompt,
)
from shared.eval_utils import parse_pred

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# --- AVCD hyperparameters (official defaults; do not tune) ---
AVCD_ALPHA   = 0.5
APC_BETA     = 0.2      # __init__.py:186  beta = 0.2
ENTROPY_GATE = 0.6      # __init__.py:194  if entropy.item() < 0.6
APC_FILL     = -1e-4    # __init__.py:236  -float(1e-4)

VALID_KEYS = {
    "input_ids", "attention_mask",
    "pixel_values_videos", "video_grid_thw", "video_second_per_grid",
    "pixel_values_images", "pixel_values", "image_grid_thw",
    "feature_attention_mask", "input_features",
}


# ═══════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════
#
#
#  （AVCD/videollama2/model/qwen.py:731-777）。
#
#

from transformers.models.qwen2_5_omni import modeling_qwen2_5_omni as _qomni

_STATE = {
    "active":    False,
    "attn_rows": None,
    "mask_span": None,
    "threshold": None,    # float
    "mask_upto": None,
}

_ORIG_ATTN_FORWARD = _qomni.Qwen2_5OmniAttention.forward


def _avcd_attn_forward(self, hidden_states, attention_mask=None, position_ids=None,
                       past_key_values=None, past_key_value=None, output_attentions=False,
                       use_cache=False, cache_position=None, position_embeddings=None, **kwargs):
    # transformers 4.5x compat: the decoder layer unpacks exactly 2 values and
    # passes the KV cache as `past_key_values` (old name kept for older versions).
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states).view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    key_states   = self.k_proj(hidden_states).view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = _qomni.apply_multimodal_rotary_pos_emb(
        query_states, key_states, cos, sin, self.rope_scaling["mrope_section"])

    _pkv = past_key_values if past_key_values is not None else past_key_value
    if _pkv is not None:
        key_states, value_states = _pkv.update(
            key_states, value_states, self.layer_idx,
            {"sin": sin, "cos": cos, "cache_position": cache_position})

    key_states   = _qomni.repeat_kv(key_states, self.num_key_value_groups)
    value_states = _qomni.repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
    # transformers>=4.5x with sdpa config passes attention_mask=None and relies on
    # SDPA's internal is_causal; this hand-written eager math must add the causal
    # mask itself or prefill silently attends to future tokens (bidirectional).
    if attention_mask is None and q_len > 1 and key_states.shape[-2] == q_len:
        min_v = torch.finfo(attn_weights.dtype).min
        causal = torch.full((q_len, q_len), min_v, device=attn_weights.device, dtype=attn_weights.dtype)
        causal = torch.triu(causal, diagonal=1)
        attn_weights = attn_weights + causal[None, None, :, :]
    elif attention_mask is not None:
        attn_weights = attn_weights + attention_mask[:, :, :, : key_states.shape[-2]]
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

    if _STATE["active"] and q_len > 1:
        row = attn_weights[:, :, -1, :].float()                    # [B,H,S]
        if _STATE["attn_rows"] is not None:
            _STATE["attn_rows"].append(row.detach())
        span = _STATE["mask_span"]
        if span is not None and self.layer_idx < _STATE["mask_upto"]:
            hot  = (row > _STATE["threshold"]) & span.view(1, 1, -1)   # [B,H,S]
            keep = (~hot).to(attn_weights.dtype).unsqueeze(-1)         # [B,H,S,1]
            attn_weights = attn_weights * keep
            attn_weights = attn_weights / attn_weights.sum(-1, keepdim=True).clamp(min=1e-6)

    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
    attn_output = self.o_proj(attn_output)
    return attn_output, (attn_weights if output_attentions else None)


class AttnPatch:
    """Context manager that swaps the text backbone's eager attention forward for the AVCD-aware version."""

    def __enter__(self):
        _qomni.Qwen2_5OmniAttention.forward = _avcd_attn_forward
        _STATE["active"] = True
        return self

    def __exit__(self, *exc):
        _STATE.update(active=False, attn_rows=None, mask_span=None)
        _qomni.Qwen2_5OmniAttention.forward = _ORIG_ATTN_FORWARD
        return False


def check_attn_impl(model):
    """Verify the thinker uses the eager attention class this file can intercept, failing early otherwise."""
    cls = type(model.thinker.model.layers[0].self_attn).__name__
    if cls != "Qwen2_5OmniAttention":
        raise RuntimeError(
            f"the thinker's attention class is {cls}, but this implementation intercepts the eager "
            f"Qwen2_5OmniAttention. Check that load_model() does not pass attn_implementation.")
    return cls


# ═══════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════

def get_modality_spans(input_ids, model, tokenizer):
    """
    Return {"video": bool[S], "audio": bool[S], "language": bool[S]}.
    Image and video are merged into "video", i.e. AVCD's visual modality V.
    """
    ids = input_ids[0]
    cfg = model.thinker.config

    def _tid(attr, token_str):
        v = getattr(cfg, attr, None)
        if not isinstance(v, int):
            v = tokenizer.convert_tokens_to_ids(token_str)
        return v if isinstance(v, int) and v >= 0 else None

    vis = torch.zeros_like(ids, dtype=torch.bool)
    for t in (_tid("image_token_index", "<|IMAGE|>"),
              _tid("video_token_index", "<|VIDEO|>")):
        if t is not None:
            vis |= (ids == t)

    aud_id = _tid("audio_token_index", "<|AUDIO|>")
    aud = (ids == aud_id) if aud_id is not None else torch.zeros_like(vis)

    return {"video": vis, "audio": aud, "language": ~(vis | aud)}


# ═══════════════════════════════════════════════════════════════════════════
#  prefill / decode
# ═══════════════════════════════════════════════════════════════════════════

def _clean_to_device(d, device):
    out = {k: v for k, v in d.items() if k in VALID_KEYS}
    for k, v in out.items():
        if isinstance(v, torch.Tensor):
            out[k] = (v.to(dtype=torch.bfloat16, device=device)
                      if torch.is_floating_point(v) else v.to(device))
    return out


def prefill(model, clean, collect_attn=False, mask_span=None,
            threshold=None, mask_upto=None):
    """
    Prefill one branch. Returns (logits_last, past_kv, attn_rows_or_None, pos_ids_next).

    Qwen2.5-Omni uses mRoPE, so during decoding the position_ids must be passed
    explicitly and incremented step by step; otherwise position information is
    scrambled and the output degenerates into repetition. A forward_pre_hook
    captures the position_ids actually used during prefill, and the last column
    plus one becomes the decoding start. The same pattern is used in base_vcd.py
    and base_dola.py.
    """
    captured = {}

    def _pos_hook(module, args, kwargs):
        if kwargs.get("position_ids") is not None and "position_ids" not in captured:
            captured["position_ids"] = kwargs["position_ids"].detach().clone()

    _STATE["attn_rows"] = [] if collect_attn else None
    _STATE["mask_span"] = mask_span
    _STATE["threshold"] = threshold
    _STATE["mask_upto"] = mask_upto

    h = model.thinker.model.register_forward_pre_hook(_pos_hook, with_kwargs=True)
    try:
        with torch.inference_mode():
            out = model.thinker(**clean, past_key_values=None, use_cache=True)
    finally:
        h.remove()
        rows = _STATE["attn_rows"]
        _STATE["attn_rows"] = None
        _STATE["mask_span"] = None

    pos_prefill = captured.get("position_ids")
    pos_next = pos_prefill[..., -1:] + 1 if pos_prefill is not None else None
    return out.logits[:, -1, :], out.past_key_values, rows, pos_next


def compute_dominance(attn_rows, spans, n_layers):
    """
    Reproduction of the official dominance computation.
      * skip the last layer
      * average over heads per layer, then sum within each modality span and
        divide by the number of tokens in that span
      * average across layers and sort descending to get the dominant modality
      * threshold = the per-layer sum of last-query attention divided by the number
        of layers, taken as a median over the key dimension and averaged over heads
    Returns (ranking: [(name, score)], threshold: float).
    """
    use = attn_rows[:max(n_layers - 1, 1)]
    acc = None
    dom = {k: [] for k in spans}
    for row in use:                                   # [B,H,S]
        r = row.float()
        acc = r if acc is None else acc + r
        mean_h = r.mean(dim=1)[0]
        for name, m in spans.items():
            n = int(m.sum().item())
            dom[name].append((mean_h[m].sum().item() / n) if n > 0 else 0.0)

    acc = acc / len(use)                              # [B,H,S]
    threshold = torch.quantile(acc[0], 0.5, dim=-1).mean().item()
    ranking = sorted(((k, abs(sum(v) / len(v))) for k, v in dom.items()),
                     key=lambda x: x[1], reverse=True)
    return ranking, threshold


def _decode_step(model, tok_tensor, attn_mask, past_kv, pos_ids):
    kw = dict(input_ids=tok_tensor, attention_mask=attn_mask,
              past_key_values=past_kv, use_cache=True)
    if pos_ids is not None:
        kw["position_ids"] = pos_ids
    out = model.thinker(**kw)
    return out.logits[:, -1, :], out.past_key_values


# ═══════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════

def avcd_generate(model, tokenizer, clean, max_new_tokens=256,
                  alpha=AVCD_ALPHA, beta=APC_BETA, entropy_gate=ENTROPY_GATE,
                  apc_fill=APC_FILL, verbose=True, record_process=True):
    """Returns (response, gen_ids, info, logp_list).

    When record_process is on, logp_list holds the log-probability of each
    generated token under the distribution actually used for selection, i.e.
    log_softmax of the AVCD formula output after the entropy gate / APC mask
    (phi-Decoding trajectory style, cf. base_model.py). On text-only samples
    the mode degenerates to greedy, so this is the plain model distribution.
    """
    device   = next(model.parameters()).device
    eos_id   = tokenizer.eos_token_id
    n_layers = len(model.thinker.model.layers)
    info     = {}

    spans_all = get_modality_spans(clean["input_ids"], model, tokenizer)
    spans = {k: v for k, v in spans_all.items() if int(v.sum().item()) > 0}
    info["modalities"] = {k: int(v.sum().item()) for k, v in spans.items()}

    n_mod = len(spans)
    mode  = "greedy" if n_mod < 2 else ("single_cd" if n_mod == 2 else "avcd")
    info["avcd_mode"] = mode

    with AttnPatch():
        # -- Prefill the original branch, collecting attention for dominance/threshold --
        logits_o, pkv_o, rows, pos = prefill(
            model, clean, collect_attn=(mode != "greedy"))
        attn_mask = clean["attention_mask"]

        branch_logits, branch_pkv = [], []
        if mode == "greedy":
            info["dominant"] = None
            if verbose:
                print("  [degraded] single modality → plain greedy", flush=True)
        else:
            if not rows or len(rows) < n_layers:
                raise RuntimeError(
                    f"attention hook captured {len(rows) if rows else 0} layers "
                    f"(expected {n_layers}); the attention interception did not take effect, so AVCD cannot run.")
            ranking, threshold = compute_dominance(rows, spans, n_layers)
            del rows
            # credibility assertions (coordinator-requested): dominance must be
            # non-degenerate, otherwise the interception produced meaningless stats
            scores = [v for _, v in ranking]
            assert len(ranking) == len(spans), f"degenerate ranking: {ranking}"
            assert max(scores) > 0.0, f"all-zero dominance: {ranking}"
            assert threshold > 0.0, f"non-positive threshold: {threshold}"
            dominant = ranking[0][0]
            others   = [k for k, _ in ranking[1:]]
            info.update(dominant=dominant,
                        dominance={k: round(v, 10) for k, v in ranking},
                        attn_threshold=round(threshold, 10))
            if verbose:
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
                lg, pkv, _, _ = prefill(model, clean, collect_attn=False,
                                        mask_span=sp, threshold=threshold,
                                        mask_upto=n_layers - 1)
                branch_logits.append(lg)
                branch_pkv.append(pkv)

        # -- Step-by-step decoding --
        n_gated = 0

        def _select(lo, lms):
            """AVCD formula plus entropy gate and adaptive plausibility constraint."""
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
        logp_list = []
        with torch.inference_mode():
            sel_logits = _select(logits_o, branch_logits)
            cur = sel_logits.argmax(dim=-1, keepdim=True)
            tok = cur.item()
            generated.append(tok)
            if record_process:
                lp = F.log_softmax(sel_logits.float(), dim=-1)[0, tok]
                logp_list.append(round(lp.item(), 4))

            for _ in range(1, max_new_tokens):
                if tok == eos_id:
                    break
                if len(generated) >= 5 and len(set(generated[-5:])) == 1:
                    break

                attn_mask = torch.cat(
                    [attn_mask, torch.ones(1, 1, device=device, dtype=torch.long)], dim=-1)

                logits_o, pkv_o = _decode_step(model, cur, attn_mask, pkv_o, pos)
                for i in range(len(branch_pkv)):
                    branch_logits[i], branch_pkv[i] = _decode_step(
                        model, cur, attn_mask, branch_pkv[i], pos)
                if pos is not None:
                    pos = pos + 1

                sel_logits = _select(logits_o, branch_logits)
                cur = sel_logits.argmax(dim=-1, keepdim=True)
                tok = cur.item()
                generated.append(tok)
                if record_process:
                    lp = F.log_softmax(sel_logits.float(), dim=-1)[0, tok]
                    logp_list.append(round(lp.item(), 4))

    info["n_entropy_gated"] = n_gated
    info["n_tokens"] = len(generated)
    return (tokenizer.decode(generated, skip_special_tokens=True).strip(),
            generated, info, logp_list)


# ═══════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════

def attach_trajectory(result, gen_ids, logp_list, input_len):
    """Append phi-Decoding style per-token trajectory fields (see base_model.py)."""
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
    parser.add_argument("--file_name",      type=str, default="base_avcd_out")
    parser.add_argument("--time_path",      type=str, default="./results/time")
    parser.add_argument("--max_samples",    type=int, default=-1)
    parser.add_argument("--start_index",    type=int, default=0,
                        help="Start from sample start_index (0-based); combine with --max_samples to take a range. "
                             "It also makes the fallback idx global for datasets without an id field "
                             "(cmm/phd/avhbench), so it does not collide with an earlier run's 0..199.")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--alpha",          type=float, default=AVCD_ALPHA,
                        help="AVCD contrast weight (official default 0.5)")
    parser.add_argument("--beta",           type=float, default=APC_BETA,
                        help="adaptive plausibility beta (official default 0.2)")
    parser.add_argument("--entropy_gate",   type=float, default=ENTROPY_GATE,
                        help="skip contrast when entropy below this (official 0.6)")
    parser.add_argument("--apc_fill",       type=float, default=APC_FILL,
                        help="APC fill value (official -1e-4; pass -inf for VCD-style)")
    parser.add_argument("--prompt_version", type=str, default="v0")
    parser.add_argument("--record_process", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Record per-token trajectory (token ids, per-token "
                             "logprobs, cumulative/norm logp) alongside the response")
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

    print("[model] loading Qwen2.5-Omni...", flush=True)
    processor, model = load_model()
    tokenizer = processor.tokenizer
    print(f"[attn] {check_attn_impl(model)}", flush=True)

    print(f"[avcd] alpha={args.alpha} beta={args.beta} "
          f"entropy_gate={args.entropy_gate} apc_fill={args.apc_fill}", flush=True)

    data = load_dataset(args.data_path, args.data_root)
    if args.start_index > 0:
        data = data[args.start_index:]
    if args.max_samples > 0:
        data = data[: args.max_samples]
    total = len(data)
    print(f"[data] {total} samples | dataset={args.datasets}", flush=True)

    system_prompt = get_system_prompt(args.datasets, args.prompt_version)
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
                user_content = build_user_content_with_media_tokens(
                    pixel_values, audio_values, image_values, user_prompt)
                messages = make_messages(system_prompt, user_content)

                raw_inputs = build_inputs(
                    processor, model, messages,
                    pixel_values, audio_values, image_values, args.datasets)
                clean = _clean_to_device(prepare_for_thinker(raw_inputs, model),
                                         model.device)

                response, gen_ids, info, logp_list = avcd_generate(
                    model, tokenizer, clean,
                    max_new_tokens=args.max_new_tokens,
                    alpha=args.alpha, beta=args.beta,
                    entropy_gate=args.entropy_gate, apc_fill=args.apc_fill,
                    record_process=args.record_process)
                input_len = clean["input_ids"].shape[1]

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
                    "method":       "avcd",
                    "has_avcd":     info["avcd_mode"] != "greedy",
                    "time_sec":     round(elapsed, 3),
                    "status":       "success",
                }
                result.update(info)
                if info["avcd_mode"] == "greedy":
                    result["degraded_reason"] = "text-only: no AV contrast"
                elif info["avcd_mode"] == "single_cd":
                    result["degraded_reason"] = "2 modalities: AVCD self-cancels → single-instance CD"
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
        tf.write(f"alpha: {args.alpha}\n")
        tf.write(f"beta: {args.beta}\n")
        tf.write(f"entropy_gate: {args.entropy_gate}\n")
        tf.write(f"apc_fill: {args.apc_fill}\n")
    print(f"\n[done] {total} samples in {elapsed_total/60:.1f} min. "
          f"Results: {output_path}", flush=True)


if __name__ == "__main__":
    main()
