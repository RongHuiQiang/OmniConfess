# -*- coding: utf-8 -*-
"""cf_confess_ragtruth.py — CF-Confess (counterfactual confession) operator,
minimal implementation for RAGTruth descriptive tasks.

Mechanism: greedy decoding as usual, but whenever the next token is about to
be a *content word* whose counterfactual grounding

    Delta_w = log P(w | full input) - log P(w | passage-removed input)

    (rootless zone: Delta_w < eps, i.e. the word's probability does not depend
     on the passage — it comes from parametric prior)

falls in the rootless zone, the position is re-ranked over the top-k
candidates with the score

    score(w') = log P(w' | full) + lambda_cf * max(Delta_w', 0)

which rewards candidates that DO depend on the passage. This is a *temperate*
re-rank, not a veto: the greedy token keeps its full-view logp and can still
win.

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
    stats = {'content_pos': 0, 'rootless_greedy': 0, 'intervened': 0,
             'changed': 0, 'deltas_greedy': [], 'deltas_chosen': [],
             'np_forwards': 0}

    for step in range(args.max_new_tokens):
        lp_full = torch.log_softmax(logits_f, dim=-1)
        w = int(torch.argmax(logits_f).item())

        chosen = w
        tok_str = tokenizer.decode([w])

        if content_token_class(tok_str, prev_tok) == 'content':
            stats['content_pos'] += 1
            # -- lazy sync of no-passage cache up to gen[0:step] --
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
                chosen = int(cand[j].item())
                stats['intervened'] += 1
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
    ap.add_argument('--random_control', action='store_true')
    ap.add_argument('--random_rate', type=float, default=0.12,
                    help='intervention prob per content position (control arm)')
    ap.add_argument('--ids_file', type=str, default=None)
    ap.add_argument('--dry_run', action='store_true')
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
        try:
            views = {}
            for name, psg in (('full', passage), ('nopsg', '')):
                user_prompt = get_user_prompt(item['question'], args.datasets,
                                               args.prompt_version, passage=psg)
                messages = make_messages(system_prompt, user_prompt)
                raw = build_inputs(processor, model, messages, None, None, None,
                                   args.datasets)
                views[name] = prepare_for_thinker(raw, model)

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
                logp_list = token_logprobs(gen_ids, out['scores'])
                cum = float(np.sum(logp_list))
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
                                       'topk': args.topk,
                                       'random_control': args.random_control,
                                       'random_rate': args.random_rate}
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
