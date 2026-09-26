# -*- coding: utf-8 -*-
"""probe_claim_cf.py — Step-1 signal probe for CLAIM-LEVEL counterfactual
grounding (断言级交叉验证 / 多视图接地性) on RAGTruth + HaloQuest.

Design axioms inherited from the honest failure of the word-level CF probe
(`probe_cf_ragtruth.py`, see EXPERIMENT_DESIGN.md 描述类 · CF-Confess @ RAGTruth):

  A1. Word-level Delta is dominated by word specificity / question-echo, not
      by truth — so granularity moves UP to the assertion (a complete
      entity+predicate declaration, approximated by sentence / comma clause).
  A2. Generation is never perturbed: the base_model greedy trajectory
      (token_ids in *_full_single_scored.jsonl) is teacher-forced under two
      VIEWS and scored post-hoc.
  A3. The contrast view is chosen to kill the specificity confound:
        ragtruth   : WRONG passage (borrowed from another sample) — a word
                     that is merely "specific to this passage's topic" keeps
                     some probability under a wrong passage only if it is a
                     prior; truly grounded words crash.
        haloquest  : SWAPPED image (another sample's picture).

  Delta_claim = logP(claim span | correct view) - logP(claim span | contrast view)

Hypothesis: hallucinated assertions (low Delta_claim, prior-generated) should
be separable from grounded ones. Discriminability test — the exact gate the
ragtruth word-level probe FAILED (word-level direction even reversed):

  claims that HIT gt  (grounded)   vs  claims that MISS gt (unsupported)
  -> AUC(Delta_claim) > 0.65, cluster-bootstrap CI excluding 0.5, and the
     effect must survive length confound control (per-token Delta, length-
     tercile stratified AUC, partial Spearman).

Usage:
    CUDA_VISIBLE_DEVICES=0 python probe_claim_cf.py --dataset ragtruth \
        --out outputs/claim_probe/ragtruth_claim_probe.json
    CUDA_VISIBLE_DEVICES=0 python probe_claim_cf.py --dataset haloquest \
        --out outputs/claim_probe/haloquest_claim_probe.json
"""
import os, sys, json, argparse, random, string, re, time
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'OmniConfess', 'methods'))
from shared import (  # noqa: E402
    load_dataset, load_media_for_item, resolve_media_path,
    load_model, build_inputs, prepare_for_thinker,
    build_user_content_with_media_tokens, make_messages,
    get_system_prompt, get_user_prompt,
)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("MMHALU_MODEL_ROOT", "/ultrafast_share/tsq/MODELS")

# ---------------------------------------------------------------------------
# Word classification — inherited VERBATIM in spirit from probe_cf_ragtruth.py,
# including the BPE leading-space word-initial fix (the bug that inflated the
# rootless rate ~2x in the first word-level probe).
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
    """Qwen BPE puts the leading space INSIDE the token (" key")."""
    if tok_str[:1].isspace():
        return True
    if prev_tok is None:
        return True
    return not (prev_tok[-1] if prev_tok else ' ').isalnum()


def token_class(tok_str, prev_tok=None):
    """'num' | 'content' | 'func' | 'punct' | 'cont' (subword continuation)."""
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


# tokenf1_fullset.py convention (same footing as the reported Token-F1)
def content_words(text):
    s = str(text).lower().translate(str.maketrans("", "", string.punctuation))
    return [w for w in s.split() if w not in STOPWORDS and w.strip()]


def overlap_precision(claim_text, gt_text):
    """Fraction of the claim's content words that also appear in gt
    (multiset intersection / claim count; gt = union over ';'-refs)."""
    c = Counter(content_words(claim_text))
    gt_words = set()
    for ref in str(gt_text).split(';'):
        gt_words.update(content_words(ref))
    if not c:
        return 0.0, 0
    hit = sum(n for w, n in c.items() if w in gt_words)
    tot = sum(c.values())
    return hit / tot, hit


# Avoidance phrasing (haloquest "not visible" ground truths / evasion answers)
EVADERE = re.compile(
    r"(not\s+visible|can\s*not|cannot|unable\s+to|do(es)?\s+not\s+(contain|show|"
    r"provide|appear|seem|determine)|not\s+(shown|seen|present|clearly)|"
    r"there\s+(is|are)\s+no|no\s+one|not\s+wearing|is\s+no\s|hard\s+to\s+"
    r"(see|tell|determine)|insufficient|not\s+enough\s+detail)", re.I)


# ---------------------------------------------------------------------------
# Assertion segmentation over the recorded token trajectory.
# ---------------------------------------------------------------------------
def _has_content(cls_list, lo, hi, k=1):
    return sum(1 for c in cls_list[lo:hi] if c in ('content', 'num')) >= k


def claim_spans(toks, cls):
    """Sentence spans, further split at comma/semicolon boundaries when both
    halves still carry content — approximates entity+predicate assertions.
    Returns list of (lo, hi) token-index ranges (half-open)."""
    raw, start = [], 0
    for i, t in enumerate(toks):
        if t == '' or '\n' in t:                    # special token / newline
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
            # >=2 content/num words: drops bare list ordinals ("1." / "2.")
            # and pure function-word fragments
            if _has_content(cls, a, b, k=2):
                out.append((a, b))
    return out


# ---------------------------------------------------------------------------
# Teacher-forced view logp (generalized to vision inputs; inherits the
# position convention input_len-1+i -> gen token i from probe_cf_ragtruth).
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


def build_view(processor, model, system_prompt, user_prompt, image, dataset_name):
    user_content = build_user_content_with_media_tokens(None, None, image, user_prompt)
    messages = make_messages(system_prompt, user_content)
    raw = build_inputs(processor, model, messages, None, None, image, dataset_name)
    return prepare_for_thinker(raw, model)


def stratified_sample(recs, n, seed=42):
    """Low/mid/high F1 thirds, round-robin (same as probe_cf_ragtruth)."""
    rng = random.Random(seed)
    ordered = sorted(recs, key=lambda r: r['eval_metrics']['Token-F1'])
    k = max(1, len(ordered) // 3)
    tiers = [ordered[:k], ordered[k:2 * k], ordered[2 * k:]]
    for t in tiers:
        rng.shuffle(t)
    picked = []
    while len(picked) < n:
        advanced = False
        for t in tiers:
            if t and len(picked) < n:
                picked.append(t.pop())
                advanced = True
        if not advanced:      # pool exhausted (e.g. fewer neg items than n/2)
            break
    return picked


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def auc_hit_vs_miss(hits, misses):
    """AUC that a score ranks a HIT claim above a MISS claim."""
    from scipy.stats import mannwhitneyu
    if len(hits) < 2 or len(misses) < 2:
        return float('nan'), float('nan')
    labels = np.r_[np.ones(len(hits)), np.zeros(len(misses))]
    scores = np.r_[np.asarray(hits, dtype=float), np.asarray(misses, dtype=float)]
    u, p = mannwhitneyu(scores[labels == 1], scores[labels == 0], alternative='greater')
    return float(u / (len(hits) * len(misses))), float(p)


def cluster_bootstrap_auc(claims, key, n_boot=1000, seed=0):
    """Resample SAMPLES (clusters) with replacement; AUC CI for claims[key]."""
    rng = random.Random(seed)
    by_sample = {}
    for c in claims:
        by_sample.setdefault(c['sample_i'], []).append(c)
    samples = list(by_sample.values())
    aucs = []
    for _ in range(n_boot):
        pool = []
        for _ in range(len(samples)):
            pool.extend(rng.choice(samples))
        h = [c[key] for c in pool if c['hit'] and c.get(key) is not None]
        m = [c[key] for c in pool if not c['hit'] and c.get(key) is not None]
        if len(h) >= 2 and len(m) >= 2:
            a, _ = auc_hit_vs_miss(h, m)
            if not np.isnan(a):
                aucs.append(a)
    if len(aucs) < 200:
        return None
    return (float(np.quantile(aucs, 0.025)), float(np.quantile(aucs, 0.975)))


def length_tercile_auc(claims, key):
    """Stratified AUC: rank only within length terciles (length confound control)."""
    lens = np.array([c['n_tokens'] for c in claims if c.get(key) is not None])
    if len(lens) < 12:
        return None
    q1, q2 = np.quantile(lens, [1 / 3, 2 / 3])
    per_bin, ns = [], []
    for lo, hi in ((0, q1), (q1, q2), (q2, 1e9)):
        sub = [c for c in claims if c.get(key) is not None and lo <= c['n_tokens'] <= hi]
        h = [c[key] for c in sub if c['hit']]
        m = [c[key] for c in sub if not c['hit']]
        if len(h) >= 2 and len(m) >= 2:
            a, _ = auc_hit_vs_miss(h, m)
            per_bin.append(a * len(sub))
            ns.append(len(sub))
    return float(sum(per_bin) / sum(ns)) if ns else None


def partial_spearman(claims, key, n_perm=3000, seed=0):
    """Spearman(claim score, hit) after residualising on rank(log length);
    permutation p-value."""
    from scipy.stats import spearmanr, rankdata
    sub = [c for c in claims if c.get(key) is not None]
    if len(sub) < 20:
        return None
    x = rankdata(np.array([c[key] for c in sub]))
    l = rankdata(np.log(np.array([c['n_tokens'] for c in sub], dtype=float)))
    y = np.array([1.0 if c['hit'] else 0.0 for c in sub])
    # residualise x on l
    A = np.c_[np.ones_like(l), l]
    beta, *_ = np.linalg.lstsq(A, x, rcond=None)
    r = x - A @ beta
    rho0 = spearmanr(r, y).statistic
    rng = random.Random(seed)
    y0 = y.tolist()
    cnt = 0
    for _ in range(n_perm):
        rng.shuffle(y0)
        if abs(spearmanr(r, np.array(y0)).statistic) >= abs(rho0):
            cnt += 1
    return {'rho': round(float(rho0), 4), 'perm_p': round((cnt + 1) / (n_perm + 1), 4)}


# ---------------------------------------------------------------------------
# Main probe
# ---------------------------------------------------------------------------
def run(args):
    here = os.path.dirname(os.path.abspath(__file__))
    scored = {
        'ragtruth': 'outputs/ragtruth_base_model/ragtruth_base_model_full_single_scored.jsonl',
        'haloquest': 'outputs/haloquest_base_model/haloquest_base_model_full_single_scored.jsonl',
    }[args.dataset]
    scored = os.path.join(here, scored) if not os.path.isabs(scored) else scored
    data_path = os.path.join(here, 'OmniHalluBench', args.dataset, args.dataset + '.json')
    data_root = os.path.join(here, 'OmniHalluBench')

    recs = [json.loads(l) for l in open(scored)]
    recs = [r for r in recs if r.get('token_ids')]
    data = load_dataset(data_path, data_root)
    by_id = {it['id']: it for it in data}

    # ---- stratified sampling -------------------------------------------------
    if args.dataset == 'haloquest':
        neg = [r for r in recs if EVADERE.search(r['gt'] or '')]
        pos = [r for r in recs if not EVADERE.search(r['gt'] or '')]
        # drop items whose own image is missing on disk
        def _img_ok(r):
            it = by_id.get(r['id'])
            if not it or not it.get('image_path'):
                return False
            return os.path.exists(resolve_media_path(it['image_path'], os.path.dirname(data_path), data_root))
        neg = [r for r in neg if _img_ok(r)]
        pos = [r for r in pos if _img_ok(r)]
        half = args.n_samples // 2
        sample = stratified_sample(neg, half, args.seed) + stratified_sample(pos, args.n_samples - half, args.seed + 1)
        print(f'[probe] haloquest: neg-type {len(neg)} pos-type {len(pos)} -> {len(sample)} sampled', flush=True)
    else:
        sample = stratified_sample(recs, args.n_samples, args.seed)
    if args.limit:
        sample = sample[:args.limit]

    pool_passages = [by_id[r['id']]['passage'] for r in recs
                     if r['id'] in by_id and (by_id[r['id']].get('passage') or '')]
    pool_images = [it for it in data if it.get('image_path') and os.path.exists(
        resolve_media_path(it['image_path'], os.path.dirname(data_path), data_root))]

    processor, model = load_model()
    tokenizer = processor.tokenizer
    thinker = model.thinker
    system_prompt = get_system_prompt(args.dataset, 'v1')

    rng = random.Random(args.seed + 7)
    claims_all, samples_meta = [], []
    t0 = time.time()
    for si, rec in enumerate(sample):
        item = by_id[rec['id']]
        gen_ids = torch.tensor(rec['token_ids'], dtype=torch.long, device=model.device)
        toks = [tokenizer.decode([t], skip_special_tokens=True) for t in rec['token_ids']]
        cls = [token_class(t, toks[i - 1] if i else None) for i, t in enumerate(toks)]

        # ---- two views -----------------------------------------------------
        user_prompt = get_user_prompt(item['question'], args.dataset, 'v1',
                                      passage=item.get('passage'))
        if args.dataset == 'ragtruth':
            partner_psg = None
            while partner_psg is None:
                cand = rng.choice(pool_passages)
                if cand != (item.get('passage') or ''):
                    partner_psg = cand
            views = {}
            for name, psg in (('full', item.get('passage') or ''),
                              ('ctrl', partner_psg)):
                up = get_user_prompt(item['question'], 'ragtruth', 'v1', passage=psg)
                views[name] = build_view(processor, model, system_prompt, up, None, 'ragtruth')
            partner_id = 'psg-borrow'
        else:
            img = load_media_for_item(item, data_path, data_root)[2]
            partner = None
            while partner is None:
                cand = rng.choice(pool_images)
                if cand['id'] != item['id']:
                    partner = cand
            pimg = load_media_for_item(partner, data_path, data_root)[2]
            views = {}
            for name, im in (('full', img), ('ctrl', pimg)):
                views[name] = build_view(processor, model, system_prompt, user_prompt, im, 'haloquest')
            partner_id = partner['id']

        lp = {n: view_logp(thinker, v, gen_ids) for n, v in views.items()}
        delta = lp['full'] - lp['ctrl']
        tf_check = float(np.max(np.abs(lp['full'] - np.array(rec['logp_list'])))) \
            if rec.get('logp_list') and len(rec['logp_list']) == len(lp['full']) else None

        # ---- claims ---------------------------------------------------------
        # Labeling design note: evasion claims ("there is no bird") are
        # view-INVARIANT by construction (a negative stays true under a random
        # swapped image / wrong passage), so they are excluded from the primary
        # grounded-vs-hallucinated contrast and reported as their own subgroup.
        # Primary contrast (both datasets):
        #   hit  = affirmative claim whose content words overlap gt (grounded)
        #   miss = affirmative claim on a negative-gt item (hallucinated object
        #          / attribute that the gt says is absent), OR affirmative claim
        #          with no gt overlap.
        neg_item = bool(EVADERE.search(rec['gt'] or ''))
        gt = rec['gt']
        for ci, (lo, hi) in enumerate(claim_spans(toks, cls)):
            text = tokenizer.decode(rec['token_ids'][lo:hi], skip_special_tokens=True).strip()
            cw = [j for j in range(lo, hi) if cls[j] in ('content', 'num')]
            is_evade = bool(EVADERE.search(text))
            prec, nhit = overlap_precision(text, gt)
            if is_evade:
                hit, rule, in_primary = False, 'evade(subgroup)', False
            elif neg_item:
                hit, rule, in_primary = False, 'neg-item affirm=halluc', True
            else:
                hit = prec >= args.hit_thresh
                rule, in_primary = f'prec>={args.hit_thresh}', True
            d_sum = float(np.sum(delta[lo:hi]))
            d_con = float(np.sum(delta[cw])) if cw else None
            claims_all.append({
                'sample_i': si, 'id': rec['id'], 'claim_i': ci,
                'text': text, 'lo': lo, 'hi': hi,
                'n_tokens': hi - lo, 'n_content': len(cw),
                'is_evade': is_evade, 'prec': round(prec, 3),
                'hit': bool(hit), 'rule': rule, 'in_primary': in_primary,
                'delta_sum': round(d_sum, 4),
                'delta_mean': round(d_sum / (hi - lo), 4),
                'delta_content': round(d_con, 4) if d_con is not None else None,
                'delta_content_mean': round(d_con / len(cw), 4) if cw else None,
                'lp_full_sum': round(float(np.sum(lp['full'][lo:hi])), 3),
            })
        samples_meta.append({
            'sample_i': si, 'id': rec['id'], 'f1': rec['eval_metrics']['Token-F1'],
            'group_key': rec.get('group_key'), 'neg_item': neg_item,
            'pred_words': len(rec['pred'].split()), 'tf_check_maxabsdiff': tf_check,
            'partner': partner_id, 'n_tokens': len(toks),
            # per-token view for post-hoc diagnostics (attribute words etc.)
            'tokens': toks, 'classes': cls,
            'delta_tok': [round(float(x), 4) for x in delta],
        })
        if (si + 1) % 10 == 0:
            print(f'  [{si+1}/{len(sample)}] {time.time()-t0:.0f}s', flush=True)
        torch.cuda.empty_cache()

    # ---- aggregate ----------------------------------------------------------
    # All headline numbers are computed on PRIMARY claims only (affirmative
    # content assertions); evasion claims are a separate subgroup report.
    prim = [c for c in claims_all if c['in_primary']]
    report = {'dataset': args.dataset, 'n_samples': len(samples_meta),
              'n_claims': len(claims_all), 'n_claims_primary': len(prim),
              'n_claims_evade': len(claims_all) - len(prim),
              'hit_thresh': args.hit_thresh,
              'tf_check_median_maxabsdiff': float(np.median(
                  [m['tf_check_maxabsdiff'] for m in samples_meta
                   if m['tf_check_maxabsdiff'] is not None]))}
    for m in ('delta_sum', 'delta_mean', 'delta_content', 'delta_content_mean'):
        h = [c[m] for c in prim if c['hit'] and c.get(m) is not None]
        mi = [c[m] for c in prim if not c['hit'] and c.get(m) is not None]
        a, p = auc_hit_vs_miss(h, mi)
        ci = cluster_bootstrap_auc(prim, m)
        report[m] = {
            'n_hit': len(h), 'n_miss': len(mi),
            'mean_hit': round(float(np.mean(h)), 3) if h else None,
            'mean_miss': round(float(np.mean(mi)), 3) if mi else None,
            'AUC': round(a, 4) if a == a else None,
            'MWU_p': f'{p:.2e}' if p == p else None,
            'boot_CI95': [round(x, 4) for x in ci] if ci else None,
            'len_tercile_AUC': (lambda v: round(v, 4) if v else None)(length_tercile_auc(prim, m)),
            'partial_spearman': partial_spearman(prim, m),
        }

    # subgroups (on delta_mean)
    def _sub(name, pred):
        sub = [c for c in prim if pred(c)]
        h = [c['delta_mean'] for c in sub if c['hit']]
        mi = [c['delta_mean'] for c in sub if not c['hit']]
        a, p = auc_hit_vs_miss(h, mi)
        report[name] = {'n_hit': len(h), 'n_miss': len(mi),
                        'AUC_delta_mean': round(a, 4) if a == a else None,
                        'MWU_p': f'{p:.2e}' if p == p else None}
    if args.dataset == 'haloquest':
        meta_by_i = {m['sample_i']: m for m in samples_meta}
        _sub('sub_neg_items(halluc=miss)', lambda c: meta_by_i[c['sample_i']]['neg_item'])
        _sub('sub_pos_items', lambda c: not meta_by_i[c['sample_i']]['neg_item'])
        report['sub_evade_claims'] = {
            'n': sum(1 for c in claims_all if c['is_evade']),
            'mean_delta_mean': round(float(np.mean(
                [c['delta_mean'] for c in claims_all if c['is_evade']] or [float('nan')])), 4)
            if any(c['is_evade'] for c in claims_all) else None}
    # hit-threshold sensitivity (0.3 / 0.5 / 0.7); evade/neg-item rules unchanged
    sens = {}
    for th in (0.3, 0.5, 0.7):
        h, mi = [], []
        for c in prim:
            is_hit = c['prec'] >= th if c['rule'] == f'prec>={args.hit_thresh}' else c['hit']
            (h if is_hit else mi).append(c['delta_mean'])
        a, p = auc_hit_vs_miss(h, mi)
        sens[str(th)] = {'AUC': round(a, 4) if a == a else None,
                         'n_hit': len(h), 'n_miss': len(mi)}
    report['hit_thresh_sensitivity'] = sens

    # per-sample contrast (samples owning both hit & miss primary claims)
    both = 0
    d_hits, d_miss = [], []
    for si in range(len(sample)):
        h = [c['delta_mean'] for c in prim if c['sample_i'] == si and c['hit']]
        m = [c['delta_mean'] for c in prim if c['sample_i'] == si and not c['hit']]
        if h and m:
            both += 1
            d_hits.append(np.mean(h))
            d_miss.append(np.mean(m))
    if both:
        from scipy.stats import wilcoxon
        w = wilcoxon(d_hits, d_miss, alternative='greater')
        report['within_sample'] = {
            'n_samples_both': both,
            'mean_hit': round(float(np.mean(d_hits)), 4),
            'mean_miss': round(float(np.mean(d_miss)), 4),
            'wilcoxon_p': round(float(w.pvalue), 5)}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump({'report': report, 'samples': samples_meta, 'claims': claims_all},
                  f, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f'[done] -> {args.out}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=['ragtruth', 'haloquest'])
    ap.add_argument('--n_samples', type=int, default=100)
    ap.add_argument('--hit_thresh', type=float, default=0.5)
    ap.add_argument('--out', default=None)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()
    if args.out is None:
        args.out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'outputs', 'claim_probe', f'{args.dataset}_claim_probe.json')
    run(args)
