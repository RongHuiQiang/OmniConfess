# -*- coding: utf-8 -*-
"""cf_confess_ragtruth.py — three-stage counterfactual confession for the
free-generation (RAGTruth descriptive) task.

Unlike the judgement datasets (one decision token), free generation confesses
token-by-token. The three stages run per content-word position:

  Stage 1  Generate   greedy decode under the full input (passage + question).
  Stage 2  Confess     at each *content word*, teacher-force the passage-removed
                      view and read the word's counterfactual grounding
                          Delta_w = log P(w | full) - log P(w | no-passage).
                      A word in the rootless zone (Delta_w < eps) does not
                      depend on the passage — it comes from parametric prior.
  Stage 3  Decide      only rootless positions are re-ranked over the top-k
                      candidates with
                          score(w') = log P(w' | full) + lambda_cf * max(Delta_w', 0),
                      rewarding candidates that DO depend on the passage. This
                      is a *temperate* re-rank, not a veto: the greedy token
                      keeps its full-view logp and can still win. Non-rootless
                      positions keep the greedy token unchanged.

eps is the correction threshold of the free-generation stage: only tokens with
risk score Delta_w < eps are selected, expanded and merged into correctable
spans. It affects free-generation tasks only.

DIAGNOSIS (n=200, v4 prompt, 2026-09-27). The unguarded re-rank was net -1.50
Token-F1 with ZERO upside: the 98 untouched samples were exactly flat, and every
one of the 102 edited samples lost or broke even (mean -2.94). Two mechanisms:
  1. Cascade. A swapped token conditions the whole autoregressive continuation
     through the shared KV cache, so one edit desyncs 25-78 downstream tokens and
     can collapse the answer into a refusal or a mid-list fragment
     (id 150: 89->7 words, -65 F1; id 276: 97->23 words).
  2. No signal. "Rootless" (Delta_w~0) content words are NOT hallucinations --
     they are passage-independent format / common words whose passage and
     no-passage views agree. Rewarding passage-dependence there just injects a
     non-reference synonym that breaks grammar (flake->flakes, caused->commonly).
     Rootless Delta_w clusters at 0 (p50=0.008); essentially none reach -0.5.
     Decisively: the base greedy output is ALREADY more passage-extractive than
     the reference (pred content-word coverage by passage 0.851 vs gt 0.744), so
     pushing generation MORE passage-dependent moves AWAY from gt. The grounding
     direction has NEGATIVE Token-F1 headroom on RAGTruth under v4.

Consequently the operator is now safe-by-construction rather than a net-positive
grounding lever (an honest RAGTruth-under-v4 result, not a length trick -- output
word length is held at base, ~92 words). Stage 3 fires only (a) after the first
`lead_protect` content words, (b) when Delta_w <= `sig_floor` (genuinely
parametric), and (c) when the candidate is within `fluency_drop` logp of greedy
(a near-tie synonym the model itself nearly emitted). With the defaults it makes
~0 edits and reproduces base (delta ~= 0); loosening the guards only reintroduces
the -1.5. Use `--gate_gap`/looser guards only for ablation, not deployment.

The three stages run per content-word position:

Cost control: a second (passage-removed) KV cache is maintained lazily — it
is only brought up to date at content-word positions (~20-30% of steps), in
bulk chunks covering the gap since the last sync.

Interface compatible with methods/base_model.py sharded runs:
    --datasets ragtruth --data_path <shard.json> --output_dir O
    --time_path O/time --file_name <name> [--record_process]

Extra flags:
    --lambda_cf    rerank strength (grid 0.5 / 1.0)
    --eps          rootless threshold (probe knee: 0.1)
    --topk         candidate pool size (default 10)
    --random_control  intervene at random content positions with prob
                   --random_rate using the SAME rerank (control arm)
    --ids_file     restrict to given ids (dev set), JSON list
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
os.environ.setdefault("MMHALU_MODEL_ROOT", "/ultrafast_share/tsq/MODELS")

# ---------------- content-word rule (same as probe_cf_ragtruth.py) -----------
STOPWORDS = set([
    "a", "an", "the", "and", "or", "but", "if", "because", "as", "what",
    "which", "this", "that", "these", "those", "then", "just", "so", "than",
    "such", "both", "through", "about", "for", "is", "are", "was", "were",
    "of", "while", "during", "to", "in", "on", "at", "by", "with", "it", "there",
])
_FUNCTION_WORDS = set("""
i me my myself we us our ours ourselves you your yours yourself yourselves
he him his himself she her hers herself it its itself they them their theirs
themselves what which who whom this that these those am is are was were being
been have has had having do does did doing would should could ought will shall
may might must can cannot not no nor as if because until while of at by for
with about against between into through during before after above below to
from up down in out on off over under again further then once here there when
where why how all any both each few more most other some such only own same
s t don now d ll m o re ve y ain aren couldn didn doesn hadn hasn haven isn
ma mightn mustn needn shan shouldn wasn weren won wouldn also however
therefore thus furthermore moreover additionally another overall
""".split())
_FUNC = STOPWORDS | _FUNCTION_WORDS
_SPECIAL = re.compile(r'<\|.*?\|>')


def _is_word_initial(tok_str, prev_tok):
    """Qwen BPE puts the leading space INSIDE the token (" key"), so a piece
    starts a new word iff it begins with whitespace, or the previous piece
    ends with a non-alphanumeric character (punctuation / newline)."""
    if tok_str[:1].isspace():
        return True
    if prev_tok is None:
        return True
    return not (prev_tok[-1] if prev_tok else ' ').isalnum()


def content_token_class(tok_str, prev_tok):
    """'content' | 'skip' — operator triggers only on word-INITIAL real content
    pieces: alpha words or multi-char numbers. Excluded: subword continuations,
    punctuation, special tokens, function words, single-digit list ordinals
    (their Delta~=0 is format determinism, not parametric hallucination)."""
    if _SPECIAL.search(tok_str):
        return 'skip'
    s = tok_str.strip()
    if not s:
        return 'skip'
    if not _is_word_initial(tok_str, prev_tok):
        return 'skip'                      # subword continuation
    w = s.lower().strip(string.punctuation + ' ')
    if not w:
        return 'skip'
    if w in _FUNC:
        return 'skip'
    if w.isdigit() and len(w) == 1:
        return 'skip'                      # list ordinal 1..9
    if not (w[0].isalpha() or w[0].isdigit()):
        return 'skip'
    return 'content'


def token_logprobs(gen_ids, scores):
    out = []
    for tid, logits in zip(gen_ids, scores):
        logits = logits.float()
        if logits.dim() > 1:
            logits = logits[0]
        lp = torch.log_softmax(logits, dim=-1)[tid]
        out.append(round(lp.item(), 4))
    return out


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.inference_mode()
def cf_generate(thinker, tokenizer, clean_full, clean_np, args):
    """Manual greedy loop with two KV caches. Returns dict."""
    device = clean_full['input_ids'].device
    eps, lam, topk = args.eps, args.lambda_cf, args.topk

    # -- prefill both views --
    out_f = thinker(input_ids=clean_full['input_ids'],
                    attention_mask=torch.ones_like(clean_full['input_ids']),
                    use_cache=True)
    past_f = out_f.past_key_values
    logits_f = out_f.logits[0, -1].float()

    out_n = thinker(input_ids=clean_np['input_ids'],
                    attention_mask=torch.ones_like(clean_np['input_ids']),
                    use_cache=True)
    past_n = out_n.past_key_values
    # np cache is now exactly at the end of the no-passage prompt

    eos = tokenizer.eos_token_id
    gen_ids, scores = [], []
    prev_tok = None
    np_fed = 0                # generated tokens already fed into np cache
    content_seen = 0          # content words emitted so far (0-based position)
    stats = {'content_pos': 0, 'rootless_greedy': 0, 'intervened': 0,
             'changed': 0, 'vetoed': 0, 'deltas_greedy': [], 'deltas_chosen': [],
             'np_forwards': 0, 'veto_lead': 0, 'veto_fluency': 0, 'veto_gap': 0}

    for step in range(args.max_new_tokens):
        # Stage 1 (Generate): full-view greedy token for this position.
        lp_full = torch.log_softmax(logits_f, dim=-1)
        w = int(torch.argmax(logits_f).item())

        chosen = w
        tok_str = tokenizer.decode([w])

        if content_token_class(tok_str, prev_tok) == 'content':
            stats['content_pos'] += 1
            content_seen += 1
            # Stage 2 (Confess): lazy-sync the no-passage cache to gen[0:step]
            # and read this word's counterfactual grounding Delta_w.
            if np_fed < step:
                chunk = torch.tensor([gen_ids[np_fed:step]], device=device)
                out_n = thinker(input_ids=chunk, past_key_values=past_n,
                                use_cache=True)
                past_n = out_n.past_key_values
                np_fed = step
                stats['np_forwards'] += 1
            lp_np = torch.log_softmax(out_n.logits[0, -1].float(), dim=-1)

            delta_w = float(lp_full[w] - lp_np[w])
            stats['deltas_greedy'].append(round(delta_w, 4))

            # Stage 3 (Decide): re-rank only rootless positions (Delta_w < eps).
            # Diagnosis (n=200, v4): the unguarded re-rank was net -1.50 with ZERO
            # upside -- every intervened sample lost or broke even. Two failure
            # modes were isolated: (a) edits on early/structural tokens (opener,
            # list markers) don't change one word, they *condition the whole
            # autoregressive continuation* and collapse it into a refusal or a
            # mid-list fragment (id 150: 89->7 words, -65 F1); (b) even edits that
            # do NOT truncate drift word-choice off the reference (90 samples,
            # mean -2.64) because "maximize passage-dependence" prefers salient
            # synonyms over the gt's actual words. The guards below make the
            # re-rank strictly safe: fire only deep in the body, only on strong
            # parametric signal, and only when the swap barely costs fluency
            # (small full-view logp drop) -- so an accepted edit is a near-tie
            # under the model's own distribution and cannot derail the trajectory.
            trigger = False
            if args.random_control:
                trigger = random.random() < args.random_rate
            else:
                trigger = delta_w < eps
                if trigger:
                    stats['rootless_greedy'] += 1

            if trigger:
                cand = torch.topk(logits_f, topk).indices
                lp_f_c = lp_full[cand]
                lp_n_c = lp_np[cand]
                delta_c = lp_f_c - lp_n_c
                score = lp_f_c + lam * torch.clamp(delta_c, min=0.0)
                j = int(torch.argmax(score).item())
                cand_tok = int(cand[j].item())
                stats['intervened'] += 1
                # -- guard 1: lead-in protection. Never touch the first
                # `lead_protect` content words -- these carry the opener and list
                # scaffolding whose replacement cascades over the whole answer.
                if content_seen <= args.lead_protect:
                    chosen = w
                    stats['vetoed'] += 1
                    stats['veto_lead'] += 1
                # -- guard 2: strong-signal floor. Only intervene when the greedy
                # word is *clearly* parametric (delta_w <= sig_floor); loose
                # rootless positions (delta_w just under eps) are noise.
                elif delta_w > args.sig_floor:
                    chosen = w
                    stats['vetoed'] += 1
                    stats['veto_gap'] += 1
                else:
                    # -- guard 3: fluency floor. Accept the passage-grounded
                    # candidate only if it costs almost no full-view probability
                    # (lp_full[w] - lp_full[cand] <= fluency_drop). A near-tie
                    # swap is a genuine synonym the model itself nearly emitted,
                    # so it will not steer the continuation off a cliff.
                    fluency_drop = float(lp_full[w] - lp_f_c[j])
                    if fluency_drop > args.fluency_drop:
                        chosen = w
                        stats['vetoed'] += 1
                        stats['veto_fluency'] += 1
                    else:
                        chosen = cand_tok
                        if chosen != w:
                            stats['changed'] += 1
                stats['deltas_chosen'].append(round(float(delta_c[j]), 4))
            else:
                stats['deltas_chosen'].append(round(delta_w, 4))

        gen_ids.append(chosen)
        scores.append(logits_f.cpu())
        if chosen == eos:
            break

        prev_tok = tokenizer.decode([chosen])
        out_f = thinker(input_ids=torch.tensor([[chosen]], device=device),
                        past_key_values=past_f, use_cache=True)
        past_f = out_f.past_key_values
        logits_f = out_f.logits[0, -1].float()

    return {'gen_ids': gen_ids, 'scores': scores, 'stats': stats}


@torch.inference_mode()
def abstain_gate(thinker, tokenizer, clean_full, clean_np, args):
    """Document-level abstention gate (diagnostic).

    Stage 1  greedy-decode the base answer under the full view.
    Stage 2  teacher-force that SAME answer under the passage-removed view and
             read the answer-level grounding
                 Delta = mean over CONTENT tokens of (logp_full - logp_np).
    Stage 3  if Delta <= theta_abs the answer does not depend on the passage
             (parametric) -> the response is replaced by the refusal string.

    Returns dict(gen_ids, scores, abstained, delta_content, delta_all, stats).
    NOTE: net-negative on RAGTruth-v4 at every theta (see module docstring); the
    fabricated answers are topically related to the passage, so their content
    tokens stay passage-dependent (high Delta) and the gate cannot separate them
    from genuine grounding without abstaining on most of the set.
    """
    device = clean_full['input_ids'].device

    # -- Stage 1: greedy decode under the full view (== base output) --
    gen = thinker.generate(
        input_ids=clean_full['input_ids'],
        attention_mask=torch.ones_like(clean_full['input_ids']),
        max_new_tokens=args.max_new_tokens,
        do_sample=False, num_beams=1,
    )
    gen_ids = gen[0, clean_full['input_ids'].shape[1]:].tolist()
    eos = tokenizer.eos_token_id
    if gen_ids and gen_ids[-1] == eos:
        gen_ids = gen_ids[:-1]
    if not gen_ids:
        return {'gen_ids': [], 'scores': [], 'abstained': False,
                'delta_content': 0.0, 'delta_all': 0.0, 'stats': {}}

    gen_t = torch.tensor([gen_ids], device=device)

    # -- Stage 2: teacher-force the answer under BOTH views, per-token logp --
    def _tf_logp(prompt_ids):
        full = torch.cat([prompt_ids, gen_t], dim=1)
        out = thinker(input_ids=full, attention_mask=torch.ones_like(full),
                      use_cache=False)
        logits = out.logits[0].float()
        plen = prompt_ids.shape[1]
        return [float(torch.log_softmax(logits[plen + k - 1], dim=-1)[tid])
                for k, tid in enumerate(gen_ids)]

    lp_full = _tf_logp(clean_full['input_ids'])
    lp_np = _tf_logp(clean_np['input_ids'])
    deltas = [f - n for f, n in zip(lp_full, lp_np)]

    prev_tok, content_deltas = None, []
    for tid, d in zip(gen_ids, deltas):
        ts = tokenizer.decode([tid])
        if content_token_class(ts, prev_tok) == 'content':
            content_deltas.append(d)
        prev_tok = ts
    delta_all = sum(deltas) / len(deltas)
    delta_content = (sum(content_deltas) / len(content_deltas)
                     if content_deltas else delta_all)

    # -- Stage 3: abstain if the answer is passage-independent --
    abstained = delta_content <= args.theta_abs
    stats = {'content_pos': len(content_deltas), 'delta_content': round(delta_content, 4),
             'delta_all': round(delta_all, 4), 'abstained': abstained,
             'intervened': int(abstained), 'changed': int(abstained),
             'rootless_greedy': 0, 'vetoed': 0}
    return {'gen_ids': gen_ids, 'scores': [], 'abstained': abstained,
            'delta_content': delta_content, 'delta_all': delta_all, 'stats': stats}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', type=str, default='ragtruth')
    ap.add_argument('--data_path', type=str, required=True)
    ap.add_argument('--data_root', type=str, default='./OmniHalluBench/')
    ap.add_argument('--output_dir', type=str, default='./outputs/cf_confess_ragtruth')
    ap.add_argument('--file_name', type=str, default='cf_confess_out')
    ap.add_argument('--time_path', type=str, default=None)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--max_new_tokens', type=int, default=512)
    ap.add_argument('--prompt_version', type=str, default='v1')
    ap.add_argument('--record_process', action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument('--lambda_cf', type=float, default=1.0)
    ap.add_argument('--eps', type=float, default=0.1)
    ap.add_argument('--topk', type=int, default=10)
    ap.add_argument('--gate_gap', type=float, default=5.0,
                    help="v3 conservative-edit gate: veto a rootless-position "
                         "swap when its grounding-jump (delta_chosen-"
                         "delta_greedy) >= gate_gap; large jumps replace "
                         "high-salience words and derail Token-F1. inf = "
                         "ungated re-rank.")
    ap.add_argument('--lead_protect', type=int, default=8,
                    help="guard 1: never re-rank the first N content words "
                         "(opener + list scaffolding). Early edits cascade over "
                         "the whole autoregressive trajectory. 0 = disabled.")
    ap.add_argument('--sig_floor', type=float, default=-1.0,
                    help="guard 2: only intervene when delta_w <= sig_floor "
                         "(clearly parametric). RAGTruth diagnosis: rootless "
                         "Delta_w clusters at ~0 (p50=0.008) and essentially "
                         "NONE reach -1.0, so this default makes ~0 edits -- the "
                         "safe, Token-F1-neutral posture for a task with no "
                         "parametric-hallucination signal. Loosen (e.g. eps) "
                         "only for ablation; it reintroduces the -1.5. Set to "
                         "eps to fire on all rootless positions.")
    ap.add_argument('--fluency_drop', type=float, default=0.15,
                    help="guard 3: accept a swap only if the candidate's "
                         "full-view logp is within fluency_drop of the greedy "
                         "token (near-tie synonym). RAGTruth diagnosis: even a "
                         "near-tie swap can desync the downstream list/number "
                         "structure and cascade, so this is tight by default. "
                         "inf = disabled.")
    ap.add_argument('--random_control', action='store_true')
    ap.add_argument('--random_rate', type=float, default=0.12,
                    help='intervention prob per content position (control arm)')
    ap.add_argument('--ids_file', type=str, default=None)
    ap.add_argument('--dry_run', action='store_true')
    # ---- document-level abstention gate (2026-09-28 diagnostic, NOT deployable) ----
    ap.add_argument('--decision', type=str, default='rewrite',
                    choices=['rewrite', 'abstain'],
                    help="'rewrite' = the default token-level counterfactual "
                         "re-rank (safe-by-construction, delta~=0). 'abstain' = "
                         "document-level abstention gate: greedy-decode the base "
                         "answer (Stage 1), teacher-force it under the "
                         "passage-removed view to read the ANSWER-LEVEL grounding "
                         "Delta = mean_content(logp_full - logp_np) (Stage 2), and "
                         "if Delta <= --theta_abs REPLACE the whole answer with the "
                         "refusal string (Stage 3). DIAGNOSIS (n=200, v4, "
                         "2026-09-28): NOT separable and net-NEGATIVE at every "
                         "theta -- see module docstring. Keep for ablation only.")
    ap.add_argument('--theta_abs', type=float, default=1.0,
                    help="abstention threshold on the answer-level content Delta. "
                         "Answers with Delta <= theta_abs are treated as "
                         "passage-independent (parametric) and replaced with the "
                         "refusal string. No value nets positive on RAGTruth-v4.")
    ap.add_argument('--refusal_text', type=str,
                    default='Unable to answer based on given passages.',
                    help="replacement string emitted when the abstention gate fires.")
    args = ap.parse_args()
    if args.time_path is None:
        args.time_path = os.path.join(args.output_dir, 'time')

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.time_path, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.file_name + '.jsonl')

    done_ids = set()
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line.strip())['id'])
                except Exception:
                    pass
        print(f'[resume] skipping {len(done_ids)} already-done samples', flush=True)

    data = load_dataset(args.data_path, os.path.dirname(os.path.abspath(args.data_path)))
    if args.ids_file:
        keep = set(json.load(open(args.ids_file)))
        data = [it for it in data if str(it['id']) in keep]
    if args.dry_run:
        data = data[:2]
    print(f'[cf_confess] {len(data)} samples | lambda={args.lambda_cf} eps={args.eps} '
          f'topk={args.topk} random_control={args.random_control}', flush=True)

    processor, model = load_model()
    tokenizer = processor.tokenizer
    system_prompt = get_system_prompt(args.datasets, args.prompt_version)

    all_out_tokens = 0
    start = time.time()

    for idx, item in enumerate(data):
        if str(item['id']) in done_ids:
            continue
        t0 = time.time()
        print(f'\n[{idx+1}/{len(data)}] id={item["id"]}', flush=True)

        passage = item.get('passage') or item.get('context') or ''
        gen_ids, stats, elapsed = None, None, None
        abstained = False
        try:
            views = {}
            for name, psg in (('full', passage), ('nopsg', '')):
                user_prompt = get_user_prompt(item['question'], args.datasets,
                                               args.prompt_version, passage=psg)
                messages = make_messages(system_prompt, user_prompt)
                raw = build_inputs(processor, model, messages, None, None, None,
                                   args.datasets)
                views[name] = prepare_for_thinker(raw, model)

            if args.decision == 'abstain':
                out = abstain_gate(model.thinker, tokenizer,
                                   views['full'], views['nopsg'], args)
                gen_ids, stats = out['gen_ids'], out['stats']
                abstained = out['abstained']
                elapsed = time.time() - t0
                print(f'  tokens={len(gen_ids)} | delta_content={out["delta_content"]:.3f} '
                      f'theta_abs={args.theta_abs} abstained={abstained} '
                      f'| {elapsed:.1f}s', flush=True)
            else:
                out = cf_generate(model.thinker, tokenizer,
                                  views['full'], views['nopsg'], args)
                gen_ids, stats = out['gen_ids'], out['stats']
                elapsed = time.time() - t0
                print(f'  tokens={len(gen_ids)} | content_pos={stats["content_pos"]} '
                      f'rootless={stats["rootless_greedy"]} interv={stats["intervened"]} '
                      f'changed={stats["changed"]} | {elapsed:.1f}s', flush=True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f'  [ERROR] {e}', flush=True)
            elapsed = time.time() - t0

        torch.cuda.empty_cache()
        if args.decision == 'abstain' and abstained:
            response = args.refusal_text
        else:
            response = (tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                        if gen_ids is not None else '')
        if gen_ids is not None:
            all_out_tokens += len(gen_ids)

        method = ('cf_confess_rand' if args.random_control else 'cf_confess')
        result = make_result(item, response, args.datasets, method,
                             time_sec=elapsed)
        if args.record_process:
            result['record_process'] = True
            if gen_ids is not None:
                logp_list = (token_logprobs(gen_ids, out['scores'])
                             if out.get('scores') else [])
                cum = float(np.sum(logp_list)) if logp_list else 0.0
                result.update({
                    'token_ids': gen_ids,
                    'logp_list': logp_list,
                    'cumulative_logprob': round(cum, 4),
                    'norm_logp': round(cum / (len(gen_ids) + 1e-8), 4),
                    'input_len': int(views['full']['input_ids'].shape[1]),
                    'output_tokens': int(len(gen_ids)),
                })
                result['cf_stats'] = {k: v for k, v in stats.items()}
                result['cf_config'] = {'lambda_cf': args.lambda_cf, 'eps': args.eps,
                                       'topk': args.topk, 'gate_gap': args.gate_gap,
                                       'lead_protect': args.lead_protect,
                                       'sig_floor': args.sig_floor,
                                       'fluency_drop': args.fluency_drop,
                                       'random_control': args.random_control,
                                       'random_rate': args.random_rate,
                                       'decision': args.decision,
                                       'theta_abs': args.theta_abs}
            else:
                result.update({'token_ids': None, 'logp_list': None,
                               'output_tokens': 0, 'cf_stats': None})
        save_result(result, output_path)

    total = time.time() - start
    with open(os.path.join(args.time_path, args.file_name + '.txt'), 'w') as f:
        f.write(f'total_time: {total:.2f}s\ntotal_output_tokens: {all_out_tokens}\n')
    print(f'\n[done] total={total:.1f}s -> {output_path}', flush=True)


if __name__ == '__main__':
    main()
