# -*- coding: utf-8 -*-
"""probe_cf_ragtruth.py — Step-1 signal probe for CF-Confess (counterfactual
passage-grounding) on RAGTruth.

Mechanism under test (病灶理论): for descriptive tasks the disease is content
words with Delta ~= 0, where

    Delta(tok) = log P(tok | full input)  -  log P(tok | passage-masked input)

A content word whose probability does not move when the passage is removed is
"rootless" — it is produced from parametric prior, not from evidence.

What this script does (no generation — teacher-forced on the base_model greedy
trajectory, so results align exactly with the recorded Token-F1):

 1. Stratified sample of N RAGTruth items (low / mid / high Token-F1 thirds).
 2. Rebuild the exact shared-v1 prompt, plus a counterfactual view with the
    passage removed from the user prompt (re-prefill).
 3. Forward both views over [prompt ; base token_ids]; collect per-token logp.
 4. Tag each generated token as content word / function word / numeric.
 5. Report: |Delta| distribution of content words, knee-point epsilon,
    rootless-rate vs Token-F1 (Spearman, also controlling for pred length),
    Delta distribution comparison (numeric vs entity-ish vs function words).

Usage:
    CUDA_VISIBLE_DEVICES=4 python probe_cf_ragtruth.py \
        --scored outputs/ragtruth_base_model/ragtruth_base_model_full_single_scored.jsonl \
        --n_samples 100 --out outputs/cf_probe/probe_report.json
"""
import os, sys, json, argparse, random, string, time
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'OmniConfess', 'methods'))
from shared import (  # noqa: E402
    load_dataset, load_model, build_inputs, prepare_for_thinker,
    make_messages, get_system_prompt, get_user_prompt,
)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("MMHALU_MODEL_ROOT", "/ultrafast_share/tsq/MODELS")

# ---------------------------------------------------------------------------
# Content-word rule: tokenf1 STOPWORDS (the eval's own notion) + a standard
# English function-word list (pronouns/auxiliaries/prepositions/conjunctions).
# ---------------------------------------------------------------------------
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


def _is_word_initial(tok_str, prev_tok):
    """Qwen BPE puts the leading space INSIDE the token (" key"), so a piece
    starts a new word iff it begins with whitespace, or the previous piece
    ends with a non-alphanumeric character (punctuation / newline)."""
    if tok_str[:1].isspace():
        return True
    if prev_tok is None:
        return True
    return not (prev_tok[-1] if prev_tok else ' ').isalnum()


def token_class(tok_str, prev_tok=None):
    """Classify a decoded token.

    Returns 'num' | 'content' | 'func' | 'punct' | 'cont'.
    Only word-INITIAL pieces are eligible for content/func/num: subword
    continuations ("succ"+"inate") have Delta ~= 0 by construction (the word
    is already determined by its first piece), so counting them would
    mechanically inflate the rootless rate.
    """
    s = tok_str.strip()
    if not s:
        return 'punct'
    if not _is_word_initial(tok_str, prev_tok):
        return 'cont'          # subword continuation, not a new word
    w = s.lower().strip(string.punctuation + ' ')
    if not w:
        return 'punct'
    if any(c.isdigit() for c in w):
        return 'num'
    if not w[0].isalpha():
        return 'punct'
    if w in _FUNC:
        return 'func'
    return 'content'


@torch.inference_mode()
def view_logp(model, thinker, clean, gen_ids):
    """Teacher-forced logp of gen_ids under one input view.

    clean: dict with input_ids/attention_mask (already prepare_for_thinker'ed).
    Returns np.ndarray len == len(gen_ids).
    """
    input_len = clean['input_ids'].shape[1]
    full_ids = torch.cat([clean['input_ids'], gen_ids.unsqueeze(0)], dim=1)
    am = torch.ones_like(full_ids)
    out = thinker(input_ids=full_ids, attention_mask=am, use_cache=False)
    logits = out.logits[0].float()          # [L, V]
    lsm = torch.log_softmax(logits, dim=-1)
    # position input_len-1+i predicts generated token i
    pos = torch.arange(input_len - 1, full_ids.shape[1] - 1, device=logits.device)
    lp = lsm[pos, gen_ids.to(logits.device)]
    return lp.cpu().numpy()


def stratified_sample(recs, n, seed=42):
    """Low/mid/high F1 thirds, round-robin pick to hit n total."""
    rng = random.Random(seed)
    ordered = sorted(recs, key=lambda r: r['eval_metrics']['Token-F1'])
    k = len(ordered) // 3
    tiers = [ordered[:k], ordered[k:2 * k], ordered[2 * k:]]
    for t in tiers:
        rng.shuffle(t)
    picked = []
    while len(picked) < n:
        for t in tiers:
            if t and len(picked) < n:
                picked.append(t.pop())
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scored', required=True)
    ap.add_argument('--data_path', default=None)
    ap.add_argument('--n_samples', type=int, default=100)
    ap.add_argument('--out', default='outputs/cf_probe/probe_report.json')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--limit', type=int, default=0, help='debug: only first k samples')
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(args.scored)]
    recs = [r for r in recs if r.get('token_ids')]
    sample = stratified_sample(recs, args.n_samples, args.seed)
    if args.limit:
        sample = sample[:args.limit]
    print(f'[probe] {len(sample)} samples '
          f'(F1 mean {np.mean([r["eval_metrics"]["Token-F1"] for r in sample]):.1f})', flush=True)

    data_path = args.data_path or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'OmniHalluBench/ragtruth/annotations/ragtruth.json')
    data = {it['id']: it for it in load_dataset(data_path, os.path.dirname(data_path))}

    processor, model = load_model()
    tokenizer = processor.tokenizer
    thinker = model.thinker

    system_prompt = get_system_prompt('ragtruth', 'v1')

    per_sample = []
    t_start = time.time()
    for i, rec in enumerate(sample):
        item = data[rec['id']]
        passage = item.get('passage') or item.get('context') or ''

        gen_ids = torch.tensor(rec['token_ids'], dtype=torch.long,
                                device=model.device)

        views = {}
        for name, psg in (('full', passage), ('nopsg', '')):
            user_prompt = get_user_prompt(item['question'], 'ragtruth', 'v1', passage=psg)
            messages = make_messages(system_prompt, user_prompt)
            raw = build_inputs(processor, model, messages, None, None, None, 'ragtruth')
            clean = prepare_for_thinker(raw, model)
            views[name] = view_logp(model, thinker, clean, gen_ids)

        delta = views['full'] - views['nopsg']
        toks = [tokenizer.decode([t]) for t in rec['token_ids']]
        cls = [token_class(t, toks[i - 1] if i else None)
               for i, t in enumerate(toks)]

        f1 = rec['eval_metrics']['Token-F1']
        per_sample.append({
            'id': rec['id'], 'f1': f1,
            'pred_words': len(rec['pred'].split()),
            'gt_words': len(rec['gt'].split()),
            'n_tokens': len(toks),
            'tokens': toks, 'classes': cls,
            'lp_full': views['full'].round(4).tolist(),
            'lp_nopsg': views['nopsg'].round(4).tolist(),
            'delta': delta.round(4).tolist(),
        })
        if (i + 1) % 10 == 0:
            print(f'  [{i+1}/{len(sample)}] {time.time()-t_start:.0f}s', flush=True)
        torch.cuda.empty_cache()

    # ---------------- aggregate ----------------
    from scipy.stats import spearmanr, pearsonr

    all_content_d = np.concatenate([
        np.array([d for d, c in zip(s['delta'], s['classes']) if c in ('content', 'num')])
        for s in per_sample])
    all_func_d = np.concatenate([
        np.array([d for d, c in zip(s['delta'], s['classes']) if c == 'func'])
        for s in per_sample])
    num_d = np.concatenate([
        np.array([d for d, c in zip(s['delta'], s['classes']) if c == 'num'])
        for s in per_sample])
    cont_d = np.concatenate([
        np.array([d for d, c in zip(s['delta'], s['classes']) if c == 'content'])
        for s in per_sample])

    q = lambda a, p: float(np.quantile(np.abs(a), p)) if len(a) else float('nan')
    abs_cd = np.abs(all_content_d)
    report = {
        'n_samples': len(per_sample),
        'content_absdelta_quantiles': {str(p): round(q(abs_cd, p), 4) for p in
                                       (0.1, 0.25, 0.5, 0.75, 0.9, 0.95)},
        'mean_absdelta': {
            'content+num': round(float(np.mean(abs_cd)), 4),
            'content_only': round(float(np.mean(np.abs(cont_d))), 4),
            'numeric': round(float(np.mean(np.abs(num_d))), 4) if len(num_d) else None,
            'function': round(float(np.mean(np.abs(all_func_d))), 4),
            'n': {'content+num': int(len(abs_cd)), 'numeric': int(len(num_d)),
                  'func': int(len(all_func_d))},
        },
        'delta_positive_frac_content': round(float(np.mean(all_content_d > 0)), 4),
    }

    # epsilon sensitivity + knee
    eps_grid = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 2.0]
    sens = {}
    for eps in eps_grid:
        rates = []
        for s in per_sample:
            cd = [abs(d) for d, c in zip(s['delta'], s['classes'])
                  if c in ('content', 'num')]
            rates.append(np.mean([d < eps for d in cd]) if cd else np.nan)
        f1s = [s['f1'] for s in per_sample]
        m = ~np.isnan(rates)
        rho, p = spearmanr(np.array(rates)[m], np.array(f1s)[m])
        sens[str(eps)] = {
            'rootless_rate_mean': round(float(np.nanmean(rates)), 4),
            'spearman_vs_f1': round(float(rho), 4), 'p_value': round(float(p), 4),
        }
    report['epsilon_sensitivity'] = sens

    # top content words by rootlessness (for interpretability)
    rootless_counter, rooted_counter = Counter(), Counter()
    for s in per_sample:
        for t, c, d in zip(s['tokens'], s['classes'], s['delta']):
            if c in ('content', 'num'):
                (rootless_counter if abs(d) < 0.15 else rooted_counter)[
                    t.strip().lower().strip(string.punctuation)] += 1
    report['top_rootless_words'] = rootless_counter.most_common(25)
    report['top_rooted_words'] = rooted_counter.most_common(25)

    # per-sample record with chosen default eps (filled after knee decision)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump({'report': report, 'per_sample': per_sample}, f, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f'[done] -> {args.out}')


if __name__ == '__main__':
    main()
