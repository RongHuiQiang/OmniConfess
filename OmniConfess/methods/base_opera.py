# -*- coding: utf-8 -*-
"""
base_opera.py -- OPERA baseline on Qwen2.5-Omni-7B.

Reference
---------
Qidong Huang, Xiaoyi Dong, Pan Zhang, Bin Wang, Conghui He, Jiaqi Wang,
Dahua Lin, Weiming Zhang, Nenghai Yu.
"OPERA: Alleviating Hallucination in Multi-Modal Large Language Models via
Over-Trust Penalty and Retrospection-Allocation." CVPR 2024 (Highlight),
pp. 13418-13427.  arXiv:2311.17911.

Official code
-------------
  github.com/shikiw/OPERA, specifically opera_beam_search in
  transformers-4.29.2/src/transformers/generation/utils.py. That file was pulled
  and checked line by line while writing this. The official fork targets
  transformers 4.29.2 whereas this project runs 4.52, and the two generate stacks
  are entirely different, so a minimal-diff port was not possible; the logic was
  reimplemented on Qwen2.5-Omni instead.

Hyperparameter provenance (all from the official eval scripts' argparse defaults)
---------------------------------------------------------------------------------
  num_beams           = 5
  scale_factor sigma  = 50
  threshold r         = 15
  num_attn_candidates = 5
  penalty_weights a   = 1.0
  window_size         = 512
  do_sample           = False   (the official beam search disables sampling)

  Note on window_size: it is not the length of a local attention window but the
  length of the history state buffer, used to size max_rollback_time,
  history_states and reject_token_pos_gather. The actual local window over which
  the column product is taken is the whole generated response region, with no
  fixed k.

Over-trust penalty
------------------
  1. take the top-N_can candidates from the next-token logits
  2. run one forward pass per candidate to obtain the last-layer attention row
     with that candidate as the query, taking the max over heads (not the mean)
  3. stack with the history rows into a square matrix and normalise by row; the
     history rows are max-over-head results whose rows do not sum to one, so this
     normalisation is necessary
  4. slice out the response-region square matrix, scale by sigma, take the column
     product and multiply by 1e-7:
        local_score[..., j] = 1e-7 * (sigma * attn_local)[..., j:, j].prod(-1)
     The 1e-7 factor comes from the official code and is what brings the penalty
     back to a magnitude comparable with the logits; reading the paper text alone
     misses it, which would let the penalty dominate the ranking entirely.
  5. rollback_scores, rollback_locs = local_score.max(-1)
  6. the first ten tokens use a different penalty:
        attn_i  = the attention mass the candidate row places on image tokens
        penalty = -attn_i           if the response length is at most 10
                =  rollback_scores  otherwise
     so the first ten tokens reward attending to the image, and only afterwards is
     knowledge aggregation penalised.
  7. the penalty is subtracted from the raw logits, not from log-probabilities;
     every other vocabulary position is filled with -999 before log_softmax feeds
     the beam search.

Retrospection-allocation
------------------------
  * each step records the mode of rollback_locs into the history
  * the trigger condition is that, for every beam, the number of history entries
    equal to the global mode exceeds r (= 15)
  * a rollback_loc below 10 never triggers a rollback
  * the rollback target is rollback_loc + 1; the (beam, token) pair previously
    chosen at that position is added to a reject list, set to -999, and the beams
    are re-expanded
  * a given position is rolled back at most num_attn_candidates (= 5) times,
    after which the target moves back one position

Differences from the official implementation (engineering necessities)
----------------------------------------------------------------------
  1. Attention access: the official code sets output_attentions=True and takes
     attentions[-1] across all layers. Here the model is loaded without an
     explicit attn_implementation, so Qwen2.5-Omni uses eager attention and
     computes the full attention weights internally anyway. A forward pre-hook
     enables output_attentions on the last layer only, and a forward hook reads
     the last row and takes the max over heads. This is the same quantity the
     official code uses, without materialising [B, H, S, S] for the other layers.
     OPERA only reads it during decoding, where the query length is one, so the
     tensor is tiny.
  2. Candidate probe forwards: the official code runs one forward per candidate
     per step, batched over beams, and so does this. But transformers 4.52 appends
     to a DynamicCache in place, whereas the 4.29 legacy tuple was immutable, so
     the cache must be truncated back to its original length after each probe.
  3. State restoration on rollback: the official code restores input_ids and
     re-runs the whole segment. Here a prompt-only copy of the KV cache is kept and
     used to rebuild the cache for the response prefix, which is equivalent but
     avoids re-running the visual encoder, the most expensive part.
  4. max_new_tokens defaults to 256 to match the other baselines in this project;
     the official CHAIR evaluation uses 512 and the POPE evaluation uses 10. Use
     --max_new_tokens to change it.

Degenerate behaviour
--------------------
  OPERA is a purely decoding-side method that only inspects the language
  backbone's self-attention, and needs no visual or audio contrast branch, so it
  does not degenerate to greedy on the text-only subsets: pubmedqa and ragtruth
  run the full penalty and rollback through the same code path as the multimodal
  subsets. This contrasts with VCD and AVCD, which must degenerate on text.

  There is however one exact semantic degeneration worth stating plainly. Point 6
  above makes the penalty for the first ten tokens equal to minus the attention
  mass on image tokens. A text-only sample has no image tokens, so that mass is
  identically zero and the first ten tokens carry no penalty at all, which makes
  those ten steps ordinary beam search; the column-product penalty only starts at
  the eleventh token. The has_visual and n_image_tokens fields record this so the
  two cases can be reported separately.

  Also note that rollback almost never fires on these subsets. The argmax column
  of the column-product score is 0 at the great majority of steps, i.e. the
  "summary token" sits at position 0 of the response. That is the standard
  attention-sink phenomenon, compounded by a structural bias: column 0 has the
  most factors in its product, which is equally true of the official code. Since
  the official guard refuses to roll back when rollback_loc is below 10, every
  such step is filtered out. This is official behaviour rather than a defect of
  this implementation: on these subsets OPERA is effectively "over-trust penalty
  only, without retrospection-allocation". The n_rollback field records the actual
  number of rollbacks for every sample.

  Finally, beam search produces longer and more fluent output than greedy, so
  Token-F1 is not directly comparable with the greedy base_model baseline. That is
  a property of the method itself, since the official code also uses beam = 5, and
  it is reported as-is rather than compensated for.

Usage:
    python base_opera.py \
        --datasets phd --data_path ./OmniHalluBench/phd/phd.json \
        --output_dir ./outputs --file_name base_opera_phd_v0 \
        --prompt_version v0 --max_samples 2
"""
import os, sys, json, time, argparse, gc, copy
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

# --- OPERA hyperparameters (official eval-script defaults; do not tune) ---
NUM_BEAMS           = 5      # --beam 5
SCALE_FACTOR        = 50.0   # chair_eval.py:85
ROLLBACK_THRESHOLD  = 15     # chair_eval.py:86  (r)
NUM_ATTN_CANDIDATES = 5      # chair_eval.py:87  (N_can)
PENALTY_WEIGHT      = 1.0    # chair_eval.py:88  (α)
WINDOW_SIZE         = 512
PHI_SCALE           = 1e-7
EARLY_TOKENS        = 10
NEG_INF_FILL        = -999.0

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

from transformers.models.qwen2_5_omni import modeling_qwen2_5_omni as _qomni

_STATE = {"active": False, "row": None}


class LastLayerAttn:
    """Enable output_attentions on the last layer and capture the current query's row over all keys."""

    def __init__(self, model):
        self.attn = model.thinker.model.layers[-1].self_attn
        self.handles = []

    def _pre(self, module, args, kwargs):
        kwargs["output_attentions"] = True
        return args, kwargs

    def _post(self, module, args, kwargs, output):
        if not _STATE["active"]:
            return None
        w = output[1] if isinstance(output, tuple) and len(output) > 1 else None
        if w is not None:
            _STATE["row"] = w[:, :, -1, :].max(dim=1).values.float().detach()   # [B,S]
        return None

    def __enter__(self):
        self.handles.append(
            self.attn.register_forward_pre_hook(self._pre, with_kwargs=True))
        self.handles.append(
            self.attn.register_forward_hook(self._post, with_kwargs=True))
        _STATE["active"] = True
        return self

    def __exit__(self, *exc):
        _STATE.update(active=False, row=None)
        for h in self.handles:
            h.remove()
        self.handles = []
        return False


def check_attn_impl(model):
    """Verify the thinker uses the eager attention class, which is the only one that returns attention weights, failing early otherwise."""
    cls = type(model.thinker.model.layers[0].self_attn).__name__
    if cls != "Qwen2_5OmniAttention":
        raise RuntimeError(
            f"the thinker's attention class is {cls}, but this implementation needs the eager Qwen2_5OmniAttention "
            f"to return attn_weights. Check that load_model() does not pass attn_implementation.")
    return cls


# ═══════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════

def _iter_kv(cache):
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        for i in range(len(cache.key_cache)):
            yield i, cache.key_cache[i], cache.value_cache[i]
    elif hasattr(cache, "layers"):
        for i, lyr in enumerate(cache.layers):
            yield i, lyr.keys, lyr.values
    else:
        raise RuntimeError(f"unknown KV cache type: {type(cache)}")


def _set_kv(cache, i, k, v):
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        cache.key_cache[i], cache.value_cache[i] = k, v
    elif hasattr(cache, "layers"):
        cache.layers[i].keys, cache.layers[i].values = k, v
    else:
        raise RuntimeError(f"unknown KV cache type: {type(cache)}")


def cache_len(cache):
    for _, k, _ in _iter_kv(cache):
        return k.shape[-2]
    return 0


def cache_expand(cache, n):
    for i, k, v in list(_iter_kv(cache)):
        _set_kv(cache, i, k.repeat_interleave(n, dim=0), v.repeat_interleave(n, dim=0))
    return cache


def cache_select(cache, idx):
    for i, k, v in list(_iter_kv(cache)):
        _set_kv(cache, i, k.index_select(0, idx), v.index_select(0, idx))
    return cache


def cache_crop(cache, max_len):
    for i, k, v in list(_iter_kv(cache)):
        if k.shape[-2] > max_len:
            _set_kv(cache, i, k[:, :, :max_len, :].contiguous(),
                    v[:, :, :max_len, :].contiguous())
    return cache


def cache_clone(cache):
    new = copy.copy(cache)
    if hasattr(cache, "key_cache"):
        new.key_cache = [k.clone() for k in cache.key_cache]
        new.value_cache = [v.clone() for v in cache.value_cache]
    elif hasattr(cache, "layers"):
        new.layers = copy.deepcopy(cache.layers)
    return new


# ═══════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════

def _clean_to_device(d, device):
    out = {k: v for k, v in d.items() if k in VALID_KEYS}
    for k, v in out.items():
        if isinstance(v, torch.Tensor):
            out[k] = (v.to(dtype=torch.bfloat16, device=device)
                      if torch.is_floating_point(v) else v.to(device))
    return out


def _expand_pos(pos, n):
    if pos is None:
        return None
    return pos.repeat_interleave(n, dim=1) if pos.dim() == 3 else pos.repeat_interleave(n, dim=0)


def visual_token_mask(input_ids, model, tokenizer):
    """Return (bool[S] visual token mask, n_visual). Both image and video count as visual."""
    ids = input_ids[0]
    cfg = model.thinker.config

    def _tid(attr, tok):
        v = getattr(cfg, attr, None)
        if not isinstance(v, int):
            v = tokenizer.convert_tokens_to_ids(tok)
        return v if isinstance(v, int) and v >= 0 else None

    m = torch.zeros_like(ids, dtype=torch.bool)
    for t in (_tid("image_token_index", "<|IMAGE|>"),
              _tid("video_token_index", "<|VIDEO|>")):
        if t is not None:
            m |= (ids == t)
    return m, int(m.sum().item())


def column_product_scores(rows, sigma, phi_scale):
    """
    Reproduction of the official column-product score.
      rows: [B, N_can, T, T], the row-normalised response-region square matrix,
            valid on the lower triangle
      local_score[..., j] = phi_scale * prod_{i>=j} (sigma * rows[..., i, j])
    Returns [B, N_can, T].
    """
    a = rows * sigma
    B, C, T, _ = a.shape
    out = torch.zeros(B, C, T, device=a.device, dtype=torch.float32)
    for j in range(T):
        out[..., j] = phi_scale * a[..., j:, j].prod(-1).float()
    return out


# ═══════════════════════════════════════════════════════════════════════════
#  OPERA beam search
# ═══════════════════════════════════════════════════════════════════════════

def opera_generate(model, tokenizer, clean, vis_mask, max_new_tokens=256,
                   num_beams=NUM_BEAMS, n_can=NUM_ATTN_CANDIDATES,
                   sigma=SCALE_FACTOR, alpha=PENALTY_WEIGHT,
                   r=ROLLBACK_THRESHOLD, window_size=WINDOW_SIZE,
                   length_penalty=1.0, enable_rollback=True, verbose=True,
                   record_process=True):
    """Returns (response, gen_ids, info, logp_list).

    When record_process is on, logp_list holds one logprob per token of the
    winning beam, computed as the incremental beam-score delta of each selected
    token, i.e. its logprob under log_softmax of OPERA's penalised candidate
    distribution (phi-Decoding trajectory style, cf. base_model.py). Rollbacks
    restore the logp lists from the same snapshots as the token sequences, so
    the lists always correspond 1:1 to the returned gen_ids.
    """
    device = next(model.parameters()).device
    eos_id = tokenizer.eos_token_id
    info = {"n_rollback": 0}

    # ── prefill（batch=1）────────────────────────────────────────────────────
    captured = {}

    def _pos_hook(module, args, kwargs):
        if kwargs.get("position_ids") is not None and "position_ids" not in captured:
            captured["position_ids"] = kwargs["position_ids"].detach().clone()

    h = model.thinker.model.register_forward_pre_hook(_pos_hook, with_kwargs=True)
    try:
        with torch.inference_mode():
            out = model.thinker(**clean, past_key_values=None, use_cache=True)
    finally:
        h.remove()

    prompt_len = clean["input_ids"].shape[1]
    pos_prefill = captured.get("position_ids")
    pos0 = pos_prefill[..., -1:] + 1 if pos_prefill is not None else None

    prompt_cache = cache_clone(out.past_key_values)
    past_kv = cache_expand(out.past_key_values, num_beams)
    pos = _expand_pos(pos0, num_beams)
    attn_mask = clean["attention_mask"].repeat_interleave(num_beams, dim=0)
    next_logits = out.logits[:, -1, :].float().repeat(num_beams, 1)   # [B,V]
    del out

    vis_idx = torch.nonzero(vis_mask, as_tuple=True)[0]

    beam_scores = torch.full((num_beams,), -1e9, device=device, dtype=torch.float)
    beam_scores[0] = 0.0
    seqs = [[] for _ in range(num_beams)]
    seq_logps = [[] for _ in range(num_beams)]   # per-beam per-token logp (record)
    resp_rows = []
    finished = []                                # (score, seq[, logp_list])

    hist_locs = []
    snapshots = []
    max_rollback_time = {}
    reject = {}

    with LastLayerAttn(model), torch.inference_mode():
        for _ in range(max_new_tokens):
            T_prev = len(resp_rows)

            # -- 1. take the top-N_can candidates --
            cand_lg, cand_id = next_logits.topk(n_can, dim=-1)        # [B,N_can]

            base_len = cache_len(past_kv)
            probe_rows, probe_img = [], []
            probe_mask = torch.cat(
                [attn_mask, torch.ones(num_beams, 1, device=device, dtype=torch.long)], dim=-1)
            for c in range(n_can):
                kw = dict(input_ids=cand_id[:, c:c + 1], attention_mask=probe_mask,
                          past_key_values=past_kv, use_cache=True)
                if pos is not None:
                    kw["position_ids"] = pos
                model.thinker(**kw)
                row = _STATE["row"].float()                    # [B, base_len+1] max-over-head
                row = row / row.sum(-1, keepdim=True).clamp(min=1e-12)
                probe_rows.append(row[:, prompt_len:])
                probe_img.append(row[:, vis_idx].sum(-1) if vis_idx.numel() > 0
                                 else torch.zeros(num_beams, device=device))
                cache_crop(past_kv, base_len)

            T = T_prev + 1
            # -- 3. assemble the response-region square matrix [B,N_can,T,T], history rows right-padded with 0 --
            square = torch.zeros(num_beams, n_can, T, T, device=device, dtype=torch.float32)
            for t, rr in enumerate(resp_rows):                 # rr: [B, t+1]
                L = min(rr.shape[1], T)
                square[:, :, t, :L] = rr[:, :L].unsqueeze(1)
            for c in range(n_can):
                L = min(probe_rows[c].shape[1], T)
                square[:, c, T - 1, :L] = probe_rows[c][:, :L]

            # -- 4. column product (including the official 1e-7 factor) --
            local_scores = column_product_scores(square, sigma, PHI_SCALE)   # [B,N_can,T]
            rollback_scores, rollback_locs = local_scores.max(-1)            # [B,N_can]

            attn_i = torch.stack(probe_img, dim=1)                           # [B,N_can]
            penalty = (-attn_i) if T <= EARLY_TOKENS else rollback_scores

            cand_scores = cand_lg - alpha * penalty
            full = torch.full_like(next_logits, NEG_INF_FILL)
            full.scatter_(-1, cand_id, cand_scores)

            for (bsrc, tok) in reject.get(T_prev, ()):
                if bsrc < num_beams:
                    full[bsrc, tok] = NEG_INF_FILL

            # -- 7. beam search selection --
            scores = F.log_softmax(full, dim=-1) + beam_scores.unsqueeze(1)
            flat = scores.view(-1)
            top_sc, top_ix = flat.topk(min(2 * num_beams, flat.numel()))
            V = scores.shape[-1]

            nb_scores, nb_tokens, nb_idx = [], [], []
            nb_logps = []                         # per-token logp delta of the selection
            for sc, ix in zip(top_sc.tolist(), top_ix.tolist()):
                b, tok = ix // V, ix % V
                if tok == eos_id:
                    if seqs[b]:
                        if record_process:
                            finished.append((sc / (len(seqs[b]) ** length_penalty),
                                             list(seqs[b]), list(seq_logps[b])))
                        else:
                            finished.append((sc / (len(seqs[b]) ** length_penalty),
                                             list(seqs[b])))
                    continue
                nb_scores.append(sc); nb_tokens.append(tok); nb_idx.append(b)
                if record_process:
                    nb_logps.append(sc - float(beam_scores[b]))
                if len(nb_scores) == num_beams:
                    break
            if not nb_scores:
                break
            while len(nb_scores) < num_beams:
                nb_scores.append(nb_scores[-1]); nb_tokens.append(nb_tokens[-1])
                nb_idx.append(nb_idx[-1])
                if record_process:
                    nb_logps.append(nb_logps[-1])

            snapshots.append({
                "T": T_prev, "seqs": [list(s) for s in seqs],
                "seq_logps": [list(s) for s in seq_logps],
                "beam_scores": beam_scores.clone(),
                "resp_rows": [rr.clone() for rr in resp_rows],
                "sel": list(zip(nb_idx, nb_tokens)),
            })
            if len(snapshots) > window_size:
                snapshots.pop(0)

            # -- 8. commit: reorder cache and state, then forward the selected token --
            src = torch.tensor(nb_idx, device=device, dtype=torch.long)
            cache_select(past_kv, src)
            resp_rows = [rr.index_select(0, src) for rr in resp_rows]
            seqs = [seqs[b] + [t] for b, t in zip(nb_idx, nb_tokens)]
            if record_process:
                seq_logps = [seq_logps[b] + [lp] for b, lp in zip(nb_idx, nb_logps)]
            beam_scores = torch.tensor(nb_scores, device=device, dtype=torch.float)
            hist_locs = [hl.index_select(0, src) if torch.is_tensor(hl) else hl
                         for hl in hist_locs]
            hist_locs.append(rollback_locs.mode(dim=-1).values.index_select(0, src))
            if len(hist_locs) > window_size:
                hist_locs.pop(0)

            cur = torch.tensor(nb_tokens, device=device, dtype=torch.long).view(num_beams, 1)
            attn_mask = torch.cat(
                [attn_mask, torch.ones(num_beams, 1, device=device, dtype=torch.long)], dim=-1)
            kw = dict(input_ids=cur, attention_mask=attn_mask,
                      past_key_values=past_kv, use_cache=True)
            if pos is not None:
                kw["position_ids"] = pos
            outs = model.thinker(**kw)
            past_kv = outs.past_key_values
            next_logits = outs.logits[:, -1, :].float()
            row = _STATE["row"].float()
            row = row / row.sum(-1, keepdim=True).clamp(min=1e-12)
            resp_rows.append(row[:, prompt_len:])
            del outs
            if pos is not None:
                pos = pos + 1

            if len(finished) >= num_beams:
                break

            # -- 9. retrospection-allocation --
            if enable_rollback and len(hist_locs) > r:
                gathers = torch.stack(hist_locs, dim=-1)          # [B, n_steps]
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
                                past_kv = cache_expand(cache_clone(prompt_cache), num_beams)
                                seqs = [list(s) for s in snap["seqs"]]
                                if record_process:
                                    seq_logps = [list(s) for s in snap["seq_logps"]]
                                beam_scores = snap["beam_scores"].clone()
                                resp_rows = [rr.clone() for rr in snap["resp_rows"]]
                                hist_locs = hist_locs[:pos_rb] if pos_rb <= len(hist_locs) else hist_locs
                                snapshots = [s for s in snapshots if s["T"] < pos_rb]
                                attn_mask = clean["attention_mask"].repeat_interleave(
                                    num_beams, dim=0)
                                pos = _expand_pos(pos0, num_beams)
                                if pos_rb > 0:
                                    pre = torch.tensor([s[:pos_rb] for s in seqs],
                                                       device=device, dtype=torch.long)
                                    attn_mask = torch.cat(
                                        [attn_mask, torch.ones(num_beams, pos_rb,
                                                               device=device, dtype=torch.long)], dim=-1)
                                    kw = dict(input_ids=pre, attention_mask=attn_mask,
                                              past_key_values=past_kv, use_cache=True)
                                    if pos is not None:
                                        kw["position_ids"] = pos + torch.arange(
                                            pos_rb, device=device).view(
                                                *([1] * (pos.dim() - 1)), pos_rb)
                                    o2 = model.thinker(**kw)
                                    past_kv = o2.past_key_values
                                    next_logits = o2.logits[:, -1, :].float()
                                    del o2
                                    if pos is not None:
                                        pos = pos + pos_rb
                                info["n_rollback"] += 1
                                if verbose:
                                    print(f"  [opera] rollback → resp_pos={pos_rb} "
                                          f"(#{info['n_rollback']})", flush=True)
                            except Exception as e:
                                info["rollback_disabled"] = repr(e)[:120]
                                enable_rollback = False

    for b in range(num_beams):
        if seqs[b]:
            if record_process:
                finished.append((float(beam_scores[b]) / (len(seqs[b]) ** length_penalty),
                                 list(seqs[b]), list(seq_logps[b])))
            else:
                finished.append((float(beam_scores[b]) / (len(seqs[b]) ** length_penalty),
                                 list(seqs[b])))
    if not finished:
        return "", [], info, ([] if record_process else None)
    best_entry = max(finished, key=lambda x: x[0])
    best, best_logps = best_entry[1], (best_entry[2] if record_process else None)
    if best_logps is not None:
        # drop eos tokens together with their logprobs, keeping 1:1 alignment
        pairs = [(t, lp) for t, lp in zip(best, best_logps) if t != eos_id]
        best = [t for t, _ in pairs]
        best_logps = [round(lp, 4) for _, lp in pairs]
    else:
        best = [t for t in best if t != eos_id]
    info["n_tokens"] = len(best)
    return tokenizer.decode(best, skip_special_tokens=True).strip(), best, info, best_logps


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
    parser.add_argument("--file_name",      type=str, default="base_opera_out")
    parser.add_argument("--time_path",      type=str, default="./results/time")
    parser.add_argument("--max_samples",    type=int, default=-1)
    parser.add_argument("--start_index",    type=int, default=0,
                        help="Start from sample start_index (0-based); combine with --max_samples to take a range. "
                             "It also makes the fallback idx global for datasets without an id field "
                             "(cmm/phd/avhbench), so it does not collide with an earlier run's 0..199.")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--num_beams",      type=int, default=NUM_BEAMS)
    parser.add_argument("--scale_factor",   type=float, default=SCALE_FACTOR)
    parser.add_argument("--num_attn_candidates", type=int, default=NUM_ATTN_CANDIDATES)
    parser.add_argument("--penalty_weights", type=float, default=PENALTY_WEIGHT)
    parser.add_argument("--threshold",      type=int, default=ROLLBACK_THRESHOLD)
    parser.add_argument("--window_size",    type=int, default=WINDOW_SIZE)
    parser.add_argument("--no_rollback",    action="store_true",
                        help="Disable retrospection-allocation (for ablation).")
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
    # transformers>=4.5x auto-selects sdpa, which never returns attn weights;
    # OPERA's LastLayerAttn hook needs eager (the class check below can't see this).
    model.thinker.config._attn_implementation = "eager"
    print(f"[attn] {check_attn_impl(model)} -> forced eager", flush=True)

    print(f"[opera] beams={args.num_beams} sigma={args.scale_factor} "
          f"N_can={args.num_attn_candidates} alpha={args.penalty_weights} "
          f"r={args.threshold} window={args.window_size} "
          f"rollback={not args.no_rollback}", flush=True)

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

                vis_mask, n_vis = visual_token_mask(clean["input_ids"], model, tokenizer)
                if n_vis == 0:
                    print("  [degraded] no image token: the first 10 steps carry no penalty (plain beam search)",
                          flush=True)

                response, gen_ids, info, logp_list = opera_generate(
                    model, tokenizer, clean, vis_mask,
                    max_new_tokens=args.max_new_tokens,
                    num_beams=args.num_beams, n_can=args.num_attn_candidates,
                    sigma=args.scale_factor, alpha=args.penalty_weights,
                    r=args.threshold, window_size=args.window_size,
                    enable_rollback=not args.no_rollback,
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
                    "method":       "opera",
                    "has_visual":   n_vis > 0,
                    "n_image_tokens": n_vis,
                    "time_sec":     round(elapsed, 3),
                    "status":       "success",
                }
                result.update(info)
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
        tf.write(f"num_beams: {args.num_beams}\n")
        tf.write(f"scale_factor: {args.scale_factor}\n")
        tf.write(f"num_attn_candidates: {args.num_attn_candidates}\n")
        tf.write(f"penalty_weights: {args.penalty_weights}\n")
        tf.write(f"threshold: {args.threshold}\n")
        tf.write(f"window_size: {args.window_size}\n")
    print(f"\n[done] {total} samples in {elapsed_total/60:.1f} min. "
          f"Results: {output_path}", flush=True)


if __name__ == "__main__":
    main()
