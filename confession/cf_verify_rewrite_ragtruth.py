# -*- coding: utf-8 -*-
"""cf_verify_rewrite_ragtruth.py — sentence/claim-level "verify -> evidence-
grounded rewrite" three-stage operator for RAGTruth free generation.

Motivation. The token-level counterfactual re-rank (cf_confess_ragtruth.py) is
net-negative on v4 (52.34 vs 53.84). This operator works at the CLAIM level,
the analog of the haloquest LV2 evidence-grounded rewrite that gained +7:

  Stage 1  Generate  v4-prompt greedy decode -> base answer (unchanged base).
  Stage 2  Verify     split the answer into sentences/claims. For each claim,
                      measure passage support by the teacher-forced logprob
                      gap under two views sharing the SAME running prefix:
                          Delta_c = mean_t [ logp(tok_t | passage, prefix)
                                             - logp(tok_t | NO passage, prefix) ]
                      A claim with Delta_c below tau does not lean on the
                      passage -- it is parametric / hallucinated.
  Stage 3  Rewrite    for each unsupported claim, regenerate a passage-grounded
                      replacement claim, strongly conditioned on the passage
                      ("rewrite using ONLY facts stated in the passage"). If the
                      rewrite is itself ungrounded (Delta still low) or empty,
                      keep the original claim (never delete -> protects length /
                      Token-F1 coverage). Supported claims are kept verbatim.

Honesty knobs recorded per sample: n_claims, n_unsupported, n_rewritten,
word length before/after, so length-vs-grounding can be separated in analysis.

Env / interface identical to cf_confess_ragtruth.py.
"""
import os, sys, json, time, argparse, random, string, re

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'OmniConfess', 'methods'))
from shared import (  # noqa: E402
    load_dataset, load_model, build_inputs, prepare_for_thinker,
    make_messages, get_system_prompt, get_user_prompt,
    make_result, save_result,
)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


# ----------------------------- Token-F1 (STOP-word) --------------------------
_STOP = set("a an the and or but if because as what which this that these those "
            "then just so than such both through about for is are was were of "
            "while during to in on at by with it there".split())


def _tf1_toks(s):
    s = str(s).lower().translate(str.maketrans("", "", string.punctuation))
    return [w for w in s.split() if w and w not in _STOP]


# --------------------------- sentence segmentation ---------------------------
# Split into claims on sentence boundaries and numbered-list items, keeping the
# text that belongs to each claim. We keep it simple + deterministic.
_NUM_ITEM = re.compile(r'(?m)(?=^\s*\d+\.\s)')


def split_claims(text):
    text = text.strip()
    if not text:
        return []
    # split numbered-list items first (each "N. ..." is a claim)
    parts = [p.strip() for p in _NUM_ITEM.split(text) if p.strip()]
    claims = []
    for p in parts:
        # further split a non-list block into sentences
        if re.match(r'^\d+\.\s', p):
            claims.append(p)
        else:
            # sentence split on '. ' / '? ' / '! ' followed by capital/space
            sents = re.split(r'(?<=[.!?])\s+', p)
            for s in sents:
                s = s.strip()
                if s:
                    claims.append(s)
    return claims


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


# ------------------------------ core operator --------------------------------
@torch.inference_mode()
def teacher_force_logps(thinker, prefix_ids, cont_ids, device):
    """mean logp of cont_ids given prefix_ids (both 1D python lists / tensors).
    Returns (mean_logp, n_tokens). Single forward pass over prefix+cont."""
    if len(cont_ids) == 0:
        return 0.0, 0
    full = torch.tensor([list(prefix_ids) + list(cont_ids)], device=device)
    out = thinker(input_ids=full, attention_mask=torch.ones_like(full), use_cache=False)
    logits = out.logits[0].float()                 # [L, V]
    lp = torch.log_softmax(logits, dim=-1)
    # position i predicts token i+1; cont tokens sit at [len(prefix)-1 .. L-2]
    start = len(prefix_ids) - 1
    total = 0.0
    for k, tid in enumerate(cont_ids):
        total += float(lp[start + k, tid])
    return total / len(cont_ids), len(cont_ids)


@torch.inference_mode()
def greedy_continue(thinker, tokenizer, prompt_ids, device, max_new, eos):
    """Plain greedy decode from a fully tokenized prompt (1D list). Returns
    list of generated token ids (no prompt)."""
    ids = torch.tensor([list(prompt_ids)], device=device)
    out = thinker(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=True)
    past = out.past_key_values
    logits = out.logits[0, -1]
    gen = []
    for _ in range(max_new):
        t = int(torch.argmax(logits).item())
        if t == eos:
            break
        gen.append(t)
        nxt = torch.tensor([[t]], device=device)
        out = thinker(input_ids=nxt, past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = out.logits[0, -1]
    return gen


def build_view_prefix(processor, model, args, question, passage):
    """Return the tokenized assistant-prefix ids (system+user+gen-prompt) for a
    given passage view, as a python list."""
    sysp = get_system_prompt(args.datasets, args.prompt_version)
    userp = get_user_prompt(question, args.datasets, args.prompt_version, passage=passage)
    messages = make_messages(sysp, userp)
    raw = build_inputs(processor, model, messages, None, None, None, args.datasets)
    clean = prepare_for_thinker(raw, model)
    return clean['input_ids'][0].tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', type=str, default='ragtruth')
    ap.add_argument('--data_path', type=str, required=True)
    ap.add_argument('--output_dir', type=str, required=True)
    ap.add_argument('--file_name', type=str, default='vr_out')
    ap.add_argument('--time_path', type=str, default=None)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--max_new_tokens', type=int, default=512)
    ap.add_argument('--prompt_version', type=str, default='v4')
    ap.add_argument('--n', type=int, default=0, help='limit #samples (0=all)')
    ap.add_argument('--ids_file', type=str, default=None)
    # operator knobs
    ap.add_argument('--tau', type=float, default=0.05,
                    help='claim is unsupported if Delta_c < tau (mean logp gap)')
    ap.add_argument('--rewrite_max_new', type=int, default=90)
    ap.add_argument('--min_claim_toks', type=int, default=3,
                    help='claims shorter than this are always kept (opener etc.)')
    ap.add_argument('--accept_mode', type=str, default='grounding',
                    choices=['grounding', 'force', 'none'],
                    help='grounding=accept rewrite iff more passage-grounded; '
                         'force=always accept rewrite; none=never (diagnostic)')
    ap.add_argument('--mode', type=str, default='rewrite',
                    choices=['base', 'verify_only', 'rewrite'],
                    help='base=v4 greedy only; verify_only=diagnose no edit; '
                         'rewrite=full operator')
    args = ap.parse_args()
    if args.time_path is None:
        args.time_path = os.path.join(args.output_dir, 'time')

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.time_path, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.file_name + '.jsonl')

    done = set()
    if os.path.exists(output_path):
        for line in open(output_path):
            try: done.add(json.loads(line)['id'])
            except Exception: pass
        print(f'[resume] {len(done)} done', flush=True)

    data = load_dataset(args.data_path, os.path.dirname(os.path.abspath(args.data_path)))
    if args.ids_file:
        keep = set(json.load(open(args.ids_file)))
        data = [it for it in data if str(it['id']) in keep]
    if args.n:
        data = data[:args.n]
    print(f'[vr] {len(data)} samples | mode={args.mode} tau={args.tau} pv={args.prompt_version}', flush=True)

    processor, model = load_model()
    tokenizer = processor.tokenizer
    thinker = model.thinker
    device = model.device
    eos = tokenizer.eos_token_id

    start = time.time()
    for idx, item in enumerate(data):
        if str(item['id']) in done:
            continue
        t0 = time.time()
        q = item['question']
        psg = item.get('passage') or ''
        print(f'[{idx+1}/{len(data)}] id={item["id"]}', flush=True)

        try:
            # -------- Stage 1: Generate (v4 greedy) --------
            full_prefix = build_view_prefix(processor, model, args, q, psg)
            base_gen = greedy_continue(thinker, tokenizer, full_prefix, device,
                                       args.max_new_tokens, eos)
            base_text = tokenizer.decode(base_gen, skip_special_tokens=True).strip()

            stats = {'n_claims': 0, 'n_unsupported': 0, 'n_rewritten': 0,
                     'deltas': [], 'base_words': len(base_text.split())}

            if args.mode == 'base':
                final_text = base_text
            else:
                # no-passage prefix (shared for all claims of this sample)
                np_prefix = build_view_prefix(processor, model, args, q, '')

                claims = split_claims(base_text)
                stats['n_claims'] = len(claims)

                # running prefix in each view = view-prefix + tokens of the
                # claims accepted so far. We rebuild claim tokens from text to
                # keep the two views' prefixes textually aligned.
                acc_text = ''      # accepted answer text so far
                out_claims = []
                for ci, claim in enumerate(claims):
                    sep = ' ' if acc_text else ''
                    claim_piece = sep + claim
                    # tokenize the accepted-prefix continuation + this claim,
                    # under each view, to score the claim's own tokens.
                    acc_ids = tokenizer(acc_text, add_special_tokens=False)['input_ids'] if acc_text else []
                    claim_ids = tokenizer(claim_piece, add_special_tokens=False)['input_ids']

                    # never touch the format opener / list-intro line
                    is_opener = ('based on the given passages' in claim.lower()
                                 or claim.rstrip().endswith(':'))
                    if len(claim_ids) < args.min_claim_toks or is_opener:
                        out_claims.append(claim); acc_text += claim_piece
                        continue

                    pre_full = full_prefix + acc_ids
                    pre_np = np_prefix + acc_ids
                    lp_full, _ = teacher_force_logps(thinker, pre_full, claim_ids, device)
                    lp_np, _ = teacher_force_logps(thinker, pre_np, claim_ids, device)
                    delta = lp_full - lp_np
                    stats['deltas'].append(round(delta, 4))

                    supported = delta >= args.tau
                    if supported or args.mode == 'verify_only':
                        out_claims.append(claim); acc_text += claim_piece
                        if not supported:
                            stats['n_unsupported'] += 1
                        continue

                    stats['n_unsupported'] += 1
                    # -------- Stage 3: Rewrite this unsupported claim --------
                    rw_sys = (
                        "You rewrite one sentence so that every fact in it is "
                        "directly stated in the given passage. Use only the "
                        "passage. Keep the same topic as the original sentence. "
                        "Do not add information that is not in the passage. "
                        "Output only the rewritten sentence on one line."
                    )
                    rw_user = (
                        f"Passage: {psg}\n"
                        f"Question: {q}\n"
                        f"Answer so far: {acc_text.strip()}\n"
                        f"Original sentence (may contain unsupported claims): {claim}\n"
                        "Rewrite this sentence using only facts stated in the passage:"
                    )
                    rw_msgs = make_messages(rw_sys, rw_user)
                    rw_raw = build_inputs(processor, model, rw_msgs, None, None, None, args.datasets)
                    rw_clean = prepare_for_thinker(rw_raw, model)
                    rw_prefix = rw_clean['input_ids'][0].tolist()
                    rw_gen = greedy_continue(thinker, tokenizer, rw_prefix, device,
                                             args.rewrite_max_new, eos)
                    rw_text = tokenizer.decode(rw_gen, skip_special_tokens=True).strip()
                    rw_text = rw_text.split('\n')[0].strip()

                    # verify the rewrite is now grounded; else keep original
                    keep_original = True
                    if rw_text:
                        rw_piece = sep + rw_text
                        rw_ids = tokenizer(rw_piece, add_special_tokens=False)['input_ids']
                        if len(rw_ids) >= args.min_claim_toks:
                            lp_full_rw, _ = teacher_force_logps(thinker, full_prefix + acc_ids, rw_ids, device)
                            lp_np_rw, _ = teacher_force_logps(thinker, np_prefix + acc_ids, rw_ids, device)
                            delta_rw = lp_full_rw - lp_np_rw
                            accept = (delta_rw > delta) if args.accept_mode == 'grounding' \
                                     else (args.accept_mode == 'force')
                            stats.setdefault('rewrites', []).append(
                                {'orig': claim, 'rw': rw_text,
                                 'd_orig': round(delta, 3), 'd_rw': round(delta_rw, 3),
                                 'accepted': bool(accept)})
                            if accept:
                                out_claims.append(rw_text)
                                acc_text += rw_piece
                                stats['n_rewritten'] += 1
                                keep_original = False
                    if keep_original:
                        out_claims.append(claim); acc_text += claim_piece

                final_text = ' '.join(out_claims).strip()

            elapsed = time.time() - t0
            stats['final_words'] = len(final_text.split())
            print(f'  claims={stats["n_claims"]} unsup={stats["n_unsupported"]} '
                  f'rewritten={stats["n_rewritten"]} words {stats["base_words"]}->{stats["final_words"]} '
                  f'{elapsed:.1f}s', flush=True)

            result = make_result(item, final_text, args.datasets,
                                 f'vr_{args.mode}', time_sec=elapsed)
            result['base_raw'] = base_text
            result['vr_stats'] = stats
            result['vr_config'] = {'tau': args.tau, 'mode': args.mode,
                                   'prompt_version': args.prompt_version}
        except Exception as e:
            import traceback; traceback.print_exc()
            result = make_result(item, '', args.datasets, f'vr_{args.mode}',
                                 time_sec=time.time() - t0)
            result['error'] = str(e)
        torch.cuda.empty_cache()
        save_result(result, output_path)

    total = time.time() - start
    with open(os.path.join(args.time_path, args.file_name + '.txt'), 'w') as f:
        f.write(f'total_time: {total:.2f}s\n')
    print(f'[done] {total:.1f}s -> {output_path}', flush=True)


if __name__ == '__main__':
    main()
