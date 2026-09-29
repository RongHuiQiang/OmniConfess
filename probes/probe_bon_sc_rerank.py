# -*- coding: utf-8 -*-
"""probe_bon_sc_rerank.py — Step-1 zero-training probe for the BoN grounding
rerank hypothesis (描述类 · Best-of-N 接地重排 @ RAGTruth).

Hypothesis under test: for descriptive tasks the only viable decoding-side lever
is SELECTION, not editing (six editing rounds already failed): sample N
candidates, score each rollout with assertion-level counterfactual grounding
(Delta vs a WRONG passage), rerank, keep the most grounded one.

This probe is fully offline w.r.t. generation: the rollouts already exist in
outputs/ragtruth_base_sc/ragtruth_base_sc_full.jsonl (4 rollouts/sample,
temperature-0.6 sampling, majority-vote selection). We only teacher-force each
recorded rollout under two views and ask whether assertion-level Delta would
have picked a BETTER rollout than majority vote / random.

Reuse contract (verbatim from probe_claim_cf.py):
  - token_class / claim_spans : assertion segmentation over the token trajectory
  - view_logp                 : teacher-forced per-token logp under a view
  - contrast view             : WRONG passage borrowed from another sample

IMPORTANT prompt detail: base_sc.py appended
    "\\nPlease think step by step, and then provide your final answer."
to the user prompt before sampling, so the teacher-forced views must use the
SAME suffixed prompt or the recorded logp_list will not line up.

Decision gate (fixed in advance):
  ALIVE iff  mean Token-F1(argmax-grounding rollout)
             > mean Token-F1(random rollout) + 1.0
         AND >= mean Token-F1(SC majority-vote rollout)

Usage:
    CUDA_VISIBLE_DEVICES=2 python probe_bon_sc_rerank.py \
        --out outputs/bon_probe/ragtruth_bon_sc_probe.json
"""
import os, sys, json, argparse, random, string, time
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'OmniConfess', 'methods'))
from shared import (  # noqa: E402
    load_dataset, load_model, build_inputs, prepare_for_thinker,
    build_user_content_with_media_tokens, make_messages,
    get_system_prompt, get_user_prompt,
)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("MMHALU_MODEL_ROOT", "/ultrafast_share/tsq/MODELS")

SC_SUFFIX = "\nPlease think step by step, and then provide your final answer."

# ---------------------------------------------------------------------------
# Word classification + assertion segmentation — verbatim from probe_claim_cf.py
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
    if tok_str[:1].isspace():
        return True
    if prev_tok is None:
        return True
    return not (prev_tok[-1] if prev_tok else ' ').isalnum()


def token_class(tok_str, prev_tok=None):
    if not tok_str:
        return 'punct'
    s = tok_str.strip()
    if not s:
        return 'punct'
    if not _is_word_initial(tok_str, prev_tok):
        return 'cont'
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


def _has_content(cls_list, lo, hi, k=1):
    return sum(1 for c in cls_list[lo:hi] if c in ('content', 'num')) >= k


def claim_spans(toks, cls):
    raw, start = [], 0
    for i, t in enumerate(toks):
        if t == '' or '\n' in t:
            if i > start:
                raw.append((start, i))
            start = i + 1
            continue
        s = t.strip()
        if s and s[-1] in '.!?':
            raw.append((start, i + 1))
            start = i + 1
    if start < len(toks):
        raw.append((start, len(toks)))

    out = []
    for lo, hi in raw:
        segs, s0 = [], lo
        for j in range(lo, hi):
            st = toks[j].strip()
            if st and st[-1] in ',;':
                if _has_content(cls, s0, j) and _has_content(cls, j + 1, hi):
                    segs.append((s0, j + 1))
                    s0 = j + 1
        segs.append((s0, hi))
        for a, b in segs:
            if _has_content(cls, a, b, k=2):
                out.append((a, b))
    return out


# ---------------------------------------------------------------------------
# Token-F1 — verbatim from scripts/tokenf1_fullset.py (single_eval footing)
# ---------------------------------------------------------------------------
def _tokens(s):
    s = str(s).lower().translate(str.maketrans("", "", string.punctuation))
    return [w for w in s.split() if w not in STOPWORDS and w.strip()]


def token_f1(pred, gt):
    p_t, g_t = _tokens(pred), _tokens(gt)
    if not p_t or not g_t:
        return 0.0
    common = Counter(p_t) & Counter(g_t)
    hit = sum(common.values())
    p = hit / len(p_t)
    r = hit / len(g_t)
    return (2 * p * r) / (p + r) if (p + r) > 0 else 0.0


# ---------------------------------------------------------------------------
# Teacher-forced view logp — verbatim from probe_claim_cf.py
# ---------------------------------------------------------------------------
@torch.inference_mode()
def view_logp(thinker, clean, gen_ids):
    input_len = clean['input_ids'].shape[1]
    full = dict(clean)
    full['input_ids'] = torch.cat([clean['input_ids'], gen_ids.unsqueeze(0)], dim=1)
    full['attention_mask'] = torch.ones_like(full['input_ids'])
    out = thinker(**full, use_cache=False)
    logits = out.logits[0].float()
    lsm = torch.log_softmax(logits, dim=-1)
    pos = torch.arange(input_len - 1, full['input_ids'].shape[1] - 1, device=logits.device)
    lp = lsm[pos, gen_ids.to(logits.device)]
    return lp.cpu().numpy()


def build_view(processor, model, system_prompt, user_prompt, dataset_name):
    user_content = build_user_content_with_media_tokens(None, None, None, user_prompt)
    messages = make_messages(system_prompt, user_content)
    raw = build_inputs(processor, model, messages, None, None, None, dataset_name)
    return prepare_for_thinker(raw, model)


# ---------------------------------------------------------------------------
# Rollout-level grounding scores (several aggregators, all assertion-level)
# ---------------------------------------------------------------------------
def rollout_scores(delta, spans, cls):
    """Return dict of candidate grounding scores for ONE rollout.

    g_mean_claim : mean over claims of per-token Delta (length-controlled)
    g_worst_claim: min over claims of per-token Delta (weakest-assertion rule)
    g_content    : sum Delta over content tokens / n content tokens (whole traj)
    g_traj_mean  : sum Delta / n tokens (whole trajectory; length-confounded)
    """
    if not spans:
        return None
    claim_dmean = [float(np.sum(delta[lo:hi])) / (hi - lo) for lo, hi in spans]
    cw = [j for j in range(len(delta)) if cls[j] in ('content', 'num')]
    return {
        'g_mean_claim': float(np.mean(claim_dmean)),
        'g_worst_claim': float(np.min(claim_dmean)),
        'g_content': (float(np.sum(delta[cw])) / len(cw)) if cw else None,
        'g_traj_mean': float(np.sum(delta) / len(delta)),
    }


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------
def paired_bootstrap_ci(diffs, n_boot=2000, seed=0, stat=np.mean):
    rng = random.Random(seed)
    d = list(diffs)
    vals = []
    for _ in range(n_boot):
        vals.append(stat([rng.choice(d) for _ in range(len(d))]))
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def run(args):
    here = os.path.dirname(os.path.abspath(__file__))
    sc_path = os.path.join(here, 'outputs', 'ragtruth_base_sc', 'ragtruth_base_sc_full.jsonl')
    data_path = os.path.join(here, 'OmniHalluBench', 'ragtruth', 'ragtruth.json')
    data_root = os.path.join(here, 'OmniHalluBench')

    recs = [json.loads(l) for l in open(sc_path)]
    recs = [r for r in recs if r.get('trajectories') and len(r['trajectories']) >= 2]
    data = load_dataset(data_path, data_root)
    by_id = {it['id']: it for it in data}
    pool_passages = [it['passage'] for it in data
                     if it.get('passage') and it['passage'] != '']
    print(f'[probe] {len(recs)} SC samples, pool of {len(pool_passages)} passages', flush=True)

    processor, model = load_model()
    tokenizer = processor.tokenizer
    thinker = model.thinker
    system_prompt = get_system_prompt('ragtruth', 'v1')

    rng = random.Random(args.seed + 7)
    per_sample = []
    t0 = time.time()
    for si, rec in enumerate(recs):
        if args.limit and si >= args.limit:
            break
        item = by_id[rec['id']]
        # two views: correct passage vs borrowed WRONG passage, both with the
        # exact SC-suffixed user prompt
        up_full = get_user_prompt(item['question'], 'ragtruth', 'v1',
                                  passage=item.get('passage')) + SC_SUFFIX
        partner_psg = None
        while partner_psg is None:
            cand = rng.choice(pool_passages)
            if cand != (item.get('passage') or ''):
                partner_psg = cand
        up_ctrl = get_user_prompt(item['question'], 'ragtruth', 'v1',
                                  passage=partner_psg) + SC_SUFFIX
        views = {'full': build_view(processor, model, system_prompt, up_full, 'ragtruth'),
                 'ctrl': build_view(processor, model, system_prompt, up_ctrl, 'ragtruth')}

        # SC majority vote = rec['pred']; index of the rollout the vote points at
        sc_pick = next((i for i, t in enumerate(rec['trajectories'])
                        if t.get('pred') == rec['pred']), 0)

        rollouts = []
        for ri, tr in enumerate(rec['trajectories']):
            gen_ids = torch.tensor(tr['token_ids'], dtype=torch.long, device=model.device)
            lp = {n: view_logp(thinker, v, gen_ids) for n, v in views.items()}
            delta = lp['full'] - lp['ctrl']
            # teacher-forcing sanity: recorded logp_list came from the TEMPERED
            # (t=0.6) generate scores, so lp_full <= logp_list is expected;
            # report the median gap for the record only.
            lp_gap = float(np.median(lp['full'] - np.array(tr['logp_list']))) \
                if tr.get('logp_list') and len(tr['logp_list']) == len(lp['full']) else None
            toks = [tokenizer.decode([t], skip_special_tokens=True) for t in tr['token_ids']]
            cls = [token_class(t, toks[i - 1] if i else None) for i, t in enumerate(toks)]
            spans = claim_spans(toks, cls)
            sc_ = rollout_scores(delta, spans, cls)
            rollouts.append({
                'rollout': ri,
                'f1': token_f1(tr.get('pred', ''), rec['gt']),
                'n_tokens': len(tr['token_ids']),
                'n_claims': len(spans),
                'scores': sc_,
                'tf_lp_gap_median': lp_gap,
            })
        per_sample.append({
            'sample_i': si, 'id': rec['id'], 'gt_len_words': len((rec['gt'] or '').split()),
            'sc_pick': sc_pick, 'partner_psg': partner_psg[:80],
            'rollouts': rollouts,
        })
        if (si + 1) % 20 == 0:
            print(f'  [{si+1}/{len(recs)}] {time.time()-t0:.0f}s', flush=True)
        torch.cuda.empty_cache()

    # ---- aggregate: dispersion + selection policies -------------------------
    report = {'dataset': 'ragtruth', 'n_samples': len(per_sample),
              'n_rollouts_per_sample': sorted({len(s['rollouts']) for s in per_sample})}

    # 1) dispersion of each score across rollouts (no material -> rerank dead)
    for key in ('g_mean_claim', 'g_worst_claim', 'g_content', 'g_traj_mean'):
        stds, rngs, n_dead = [], [], 0
        for s in per_sample:
            vals = [r['scores'][key] for r in s['rollouts'] if r['scores'] and r['scores'][key] is not None]
            if len(vals) < 2:
                n_dead += 1
                continue
            stds.append(float(np.std(vals)))
            rngs.append(float(max(vals) - min(vals)))
            if rngs[-1] < 0.1:      # near-identical grounding -> nothing to rank
                n_dead += 1
        report[f'dispersion_{key}'] = {
            'mean_std': round(float(np.mean(stds)), 4) if stds else None,
            'median_range': round(float(np.median(rngs)), 4) if rngs else None,
            'frac_range_lt_0.1': round(n_dead / max(1, len(per_sample)), 4),
        }

    # 2) selection policies
    f1_random = [float(np.mean([r['f1'] for r in s['rollouts']])) for s in per_sample]
    f1_sc = [s['rollouts'][s['sc_pick']]['f1'] for s in per_sample]
    f1_oracle = [float(np.max([r['f1'] for r in s['rollouts']])) for s in per_sample]
    policies = {'random': f1_random, 'sc_vote': f1_sc, 'oracle_f1': f1_oracle}
    for key in ('g_mean_claim', 'g_worst_claim', 'g_content', 'g_traj_mean'):
        picked = []
        for s in per_sample:
            best, best_v = None, -1e18
            for r in s['rollouts']:
                v = r['scores']
                if v is not None and v[key] is not None and v[key] > best_v:
                    best_v, best = v[key], r
            picked.append(best['f1'] if best else s['rollouts'][0]['f1'])
        policies[f'argmax_{key}'] = picked

    report['token_f1_x100'] = {k: round(100 * float(np.mean(v)), 2)
                               for k, v in policies.items()}

    # paired comparisons for the primary gate (g_mean_claim) + the others
    gate = {}
    for key in ('g_mean_claim', 'g_worst_claim', 'g_content', 'g_traj_mean'):
        a = policies[f'argmax_{key}']
        d_rand = [x - y for x, y in zip(a, f1_random)]
        d_sc = [x - y for x, y in zip(a, f1_sc)]
        ci_r = paired_bootstrap_ci(d_rand, seed=1)
        ci_s = paired_bootstrap_ci(d_sc, seed=2)
        gate[key] = {
            'vs_random_mean_x100': round(100 * float(np.mean(d_rand)), 2),
            'vs_random_CI95_x100': [round(100 * ci_r[0], 2), round(100 * ci_r[1], 2)],
            'vs_sc_mean_x100': round(100 * float(np.mean(d_sc)), 2),
            'vs_sc_CI95_x100': [round(100 * ci_s[0], 2), round(100 * ci_s[1], 2)],
        }
    report['paired_gains'] = gate

    # does grounding rank F1 at all? per-sample Spearman(score, f1)
    from scipy.stats import spearmanr
    rhos = []
    for s in per_sample:
        xs = [r['scores']['g_mean_claim'] if r['scores'] else None for r in s['rollouts']]
        if any(x is None for x in xs) or len(set(xs)) < 2:
            continue
        rho = spearmanr(xs, [r['f1'] for r in s['rollouts']]).statistic
        if rho == rho:
            rhos.append(rho)
    report['spearman_score_vs_f1_within_sample'] = {
        'n': len(rhos), 'mean_rho': round(float(np.mean(rhos)), 4) if rhos else None}

    # primary verdict (pre-registered gate)
    gain_rand = report['token_f1_x100']['argmax_g_mean_claim'] - report['token_f1_x100']['random']
    gain_sc = report['token_f1_x100']['argmax_g_mean_claim'] - report['token_f1_x100']['sc_vote']
    alive = (gain_rand > 1.0) and (gain_sc >= 0)
    report['verdict_primary_g_mean_claim'] = {
        'gain_vs_random': round(gain_rand, 2), 'gain_vs_sc_vote': round(gain_sc, 2),
        'gate': 'F1(argmax) > random + 1.0 AND >= sc_vote', 'alive': bool(alive)}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump({'report': report, 'samples': per_sample}, f, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f'[done] -> {args.out}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'outputs', 'bon_probe',
        'ragtruth_bon_sc_probe.json'))
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()
    run(args)
