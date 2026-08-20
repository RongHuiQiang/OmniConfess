# -*- coding: utf-8 -*-
"""probe_claim_avhbench.py — Claim-level counterfactual grounding probe,
AVHBench port of probe_claim_cf.py (per EXPERIMENT_DESIGN.md §描述类·断言级 CF
§4 移植预案). Runs the two PRE-REGISTERED gates; no online operator is built
unless both pass.

Views (teacher-forced on the base_model greedy trajectory; video+audio come
from the same mp4 in avhbench, so a "swap" replaces only one stream):

    full        : own video frames + own audio
    swapvideo_X : PARTNER's frames + own audio
    swapaudio_X : own frames + PARTNER's audio

    Δ_swapvideo = logP(tok | full) − logP(tok | swapvideo)
    Δ_swapaudio = logP(tok | full) − logP(tok | swapaudio)

Two partner pools (pre-registered in the 移植预案):
    avh  : another avhbench item (same-domain; all 645 items are the same
           task, so this pool tests the homogeneity concern directly)
    mix  : a cmm mp4 (cross-domain pool; some cmm clips lack audio ->
           lazy redraw, max 5 tries)

ATTRIBUTE WORDS (token-level unit, per the avhbench word-level autopsy:
signal lives at attribute words, and the modality structure is clean):
    color  : COLORS set (claim_confess_sim)          -> matched Δ = Δ_swapvideo
    count  : digits / NUMW (two, three, ... many)    -> matched Δ = Δ_swapvideo
    asrc   : audio-source/sound words (AUDIO_WORDS)  -> matched Δ = Δ_swapaudio

PRE-REGISTERED GATE 1 (signal): token-level hit/miss AUC of the MATCHED-
modality Δ over pooled attribute tokens, on the avh (same-domain) pool.
hit := normalized word appears in gt content-word set. PASS := AUC > 0.65
AND cluster-bootstrap 95% CI lower bound > 0.5. Controls reported (not
gating): unmatched-modality Δ, per-attribute-class breakdown, mix pool,
claim-level attr-priority score, non-attribute content tokens.

PRE-REGISTERED GATE 2 (benefit structure, offline sim over the probe
forwards — zero GPU): "flag + evasive replacement (NOT deletion)" on
Token-F1. Claim score = mean matched attribute Δ if the claim has attribute
tokens, else mean (Δ_swapvideo+Δ_swapaudio)/2 over content tokens. Actions:
    delete-claim   (haloquest action; lesson says bag-of-words F1 punishes)
    attr-drop      (remove ONLY the flagged attribute words, keep entities)
    hedge          (wrap flagged claim: "appears to be" insertion)
    evasive-repl   (replace flagged claim with a short hedge clause)
Dev/test split (stratified by F1 tercile), τ grid on dev, paired Wilcoxon on
test vs base, random-flag control arm (same flag rate, random claims).
GATE 2 PASS := some action beats base on test (p<0.05) AND beats the random
control arm. Both gates must pass before any online operator is built.

Usage (GPUs 0-3 shared with the mechanism experiments — single process each):
    source /home/tsq/rhq/omniconfess/env_ffmpeg.sh
    MMHALU_MODEL_ROOT=/ultrafast_share/tsq/MODELS CUDA_VISIBLE_DEVICES=0 \
      python probe_claim_avhbench.py --shard_idx 0 --n_shards 4
    python probe_claim_avhbench.py --aggregate      # gate 1
    python probe_claim_avhbench.py --sim            # gate 2 (no GPU)
"""
import os, sys, json, argparse, random, string, time, glob
from collections import Counter

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'OmniConfess', 'methods'))
sys.path.insert(0, HERE)

from shared import (  # noqa: E402
    load_dataset, load_media_for_item, load_model, build_inputs,
    prepare_for_thinker, build_user_content_with_media_tokens, make_messages,
    get_system_prompt, get_user_prompt,
)
from shared.data_loader import load_video_frames, load_audio_data  # noqa: E402
from probe_claim_cf import (  # noqa: E402
    claim_spans, token_class, content_words, overlap_precision,
    auc_hit_vs_miss, cluster_bootstrap_auc,
)
from probe_cf_avhbench import AUDIO_WORDS, norm_word  # noqa: E402

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("MMHALU_MODEL_ROOT", "/ultrafast_share/tsq/MODELS")

# ---------------------------------------------------------------------------
# Attribute-word classes (colors/counts inherited from claim_confess_sim).
# ---------------------------------------------------------------------------
COLORS = set("""red blue green yellow black white orange purple pink brown gray
grey gold silver beige teal maroon khaki tan turquoise violet indigo crimson
""".split())
NUMW = set('two three four five six seven eight nine ten several many couple '
           'dozen multiple numerous dozens'.split())


def attr_class(w):
    """'color' | 'count' | 'asrc' | None for a normalized word."""
    if w in COLORS:
        return 'color'
    if any(ch.isdigit() for ch in w) or w in NUMW:
        return 'count'
    if w in AUDIO_WORDS:
        return 'asrc'
    return None


MATCHED_DELTA = {'color': 'dv', 'count': 'dv', 'asrc': 'da'}   # pre-registered
UNMATCHED_DELTA = {'color': 'da', 'count': 'da', 'asrc': 'dv'}


# ---------------------------------------------------------------------------
# Token-F1 (same footing as tokenf1_fullset.py / probe_cf_avhbench).
# ---------------------------------------------------------------------------
STOPWORDS = set([
    "a", "an", "the", "and", "or", "but", "if", "because", "as", "what",
    "which", "this", "that", "these", "those", "then", "just", "so", "than",
    "such", "both", "through", "about", "for", "is", "are", "was", "were",
    "of", "while", "during", "to", "in", "on", "at", "by", "with", "it",
    "there",
])


def token_f1(pred, gt):
    def tok(s):
        s = str(s).lower().translate(str.maketrans("", "", string.punctuation))
        return [w for w in s.split() if w not in STOPWORDS and w.strip()]
    p_t, g_t = tok(pred), tok(gt)
    if not p_t or not g_t:
        return 0.0
    common = Counter(p_t) & Counter(g_t)
    hit = sum(common.values())
    p, r = hit / len(p_t), hit / len(g_t)
    return (2 * p * r) / (p + r) if (p + r) > 0 else 0.0


def stratified_sample(recs, n, seed=42):
    rng = random.Random(seed)
    ordered = sorted(recs, key=lambda r: r['_f1'])
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
        if not advanced:
            break
    return picked


# ---------------------------------------------------------------------------
# Forwards.
# ---------------------------------------------------------------------------
_MEDIA_KEYS = (
    'pixel_values_images', 'pixel_values', 'image_grid_thw',
    'pixel_values_videos', 'video_grid_thw', 'video_second_per_grid',
    'input_features', 'feature_attention_mask',
)


@torch.inference_mode()
def view_logp(thinker, clean, gen_ids):
    input_len = clean['input_ids'].shape[1]
    full_ids = torch.cat([clean['input_ids'], gen_ids.unsqueeze(0)], dim=1)
    kw = dict(input_ids=full_ids, attention_mask=torch.ones_like(full_ids),
              use_cache=False)
    for k in _MEDIA_KEYS:
        if k in clean:
            kw[k] = clean[k]
    out = thinker(**kw)
    logits = out.logits[0].float()
    lsm = torch.log_softmax(logits, dim=-1)
    pos = torch.arange(input_len - 1, full_ids.shape[1] - 1, device=logits.device)
    lp = lsm[pos, gen_ids.to(logits.device)]
    del out, logits, lsm
    return lp.cpu().numpy()


def retry(fn, n=3, wait=5, tag=''):
    for i in range(n):
        try:
            return fn()
        except Exception as e:
            print(f'  [retry {i+1}/{n}] {tag}: {type(e).__name__}: {e}',
                  flush=True)
            time.sleep(wait * (i + 1))
    raise RuntimeError(f'retries exhausted: {tag}')


def build_view(processor, model, system_prompt, user_prompt, pixel, audio):
    prefix = ''
    if pixel is not None:
        prefix += "<|vision_bos|><|VIDEO|><|vision_eos|>\n"
    if audio is not None:
        prefix += "<|audio_bos|><|AUDIO|><|audio_eos|>\n"
    messages = make_messages(system_prompt, prefix + user_prompt)
    raw = retry(lambda: build_inputs(processor, model, messages,
                                     pixel, audio, None, 'avhbench'),
                tag='build_inputs')
    return prepare_for_thinker(raw, model)


def load_pair(path):
    """(video_frames, audio) for one cmm mp4 — cmm clips carry NO audio track,
    the audio ships as the sibling .wav. (None, None) if unusable."""
    try:
        v = load_video_frames(path)
        wav = path[:-4] + '.wav' if path.endswith('.mp4') else None
        a = load_audio_data(wav) if wav and os.path.exists(wav) else None
        return v, a
    except Exception:
        return None, None


# ---------------------------------------------------------------------------
# Shard run.
# ---------------------------------------------------------------------------
def run_shard(args):
    scored = args.scored
    recs = [json.loads(l) for l in open(scored)]
    recs = [r for r in recs if r.get('token_ids')]
    for r in recs:
        r['_f1'] = token_f1(r['pred'], r['gt'])
    sample = stratified_sample(recs, args.n_samples, args.seed)
    if args.limit:
        sample = sample[:args.limit]
    sample = [s for i, s in enumerate(sample)
              if i % args.n_shards == args.shard_idx]
    print(f'[probe shard {args.shard_idx}/{args.n_shards}] {len(sample)} samples '
          f'(F1 mean {np.mean([s["_f1"] for s in sample]):.3f})', flush=True)

    data_path = os.path.join(HERE, 'OmniHalluBench', 'avhbench', 'avhbench.json')
    data_root = os.path.dirname(data_path)
    data = load_dataset(data_path, data_root)
    by_qg = {(d['question'], d['gt']): d for d in data}
    all_items = [d for d in data if d.get('video_path')]

    mix_pool = sorted(glob.glob(os.path.join(
        HERE, 'OmniHalluBench', 'cmm', 'reorg_raw_files', '**', '*.mp4'),
        recursive=True))
    print(f'[pools] avhbench items {len(all_items)} | cmm mp4s {len(mix_pool)}',
          flush=True)

    processor, model = load_model()
    tokenizer = processor.tokenizer
    thinker = model.thinker
    system_prompt = get_system_prompt('avhbench', 'v1')

    out_path = args.out_pattern.format(idx=args.shard_idx, n=args.n_shards)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fout = open(out_path, 'w')

    rng = random.Random(args.seed + 7 + args.shard_idx)
    t0 = time.time()
    for i, rec in enumerate(sample):
        item = by_qg.get((rec['question'], rec['gt']))
        if item is None:
            print('  [skip] no dataset match', flush=True)
            continue
        try:
            own_v, own_a, _ = retry(
                lambda: load_media_for_item(item, data_path, data_root),
                tag='own_media')
            if own_v is None or own_a is None:
                print(f'  [skip] own media missing id={rec["id"]}', flush=True)
                continue

            # ---- partners ------------------------------------------------
            partner_avh = None
            while partner_avh is None:
                cand = rng.choice(all_items)
                if cand['gt'] == item['gt']:
                    continue
                pv, pa, _ = load_media_for_item(cand, data_path, data_root)
                if pv is not None and pa is not None:
                    partner_avh = (cand, pv, pa)
                else:
                    partner_avh = None
                    break                     # media missing -> redraw once
            if partner_avh is None:
                # deterministic fallback: retry a few times
                for _ in range(5):
                    cand = rng.choice(all_items)
                    if cand['gt'] == item['gt']:
                        continue
                    pv, pa, _ = load_media_for_item(cand, data_path, data_root)
                    if pv is not None and pa is not None:
                        partner_avh = (cand, pv, pa)
                        break
            if partner_avh is None:
                print(f'  [skip] no avh partner id={rec["id"]}', flush=True)
                continue

            partner_mix = None
            for _ in range(5):
                cand_path = rng.choice(mix_pool)
                mv, ma = load_pair(cand_path)
                if mv is not None and ma is not None:
                    partner_mix = (cand_path, mv, ma)
                    break
            mix_partner_id = partner_mix[0] if partner_mix else None

            gen_ids = torch.tensor(rec['token_ids'], dtype=torch.long,
                                   device=model.device)
            user_prompt = get_user_prompt(item['question'], 'avhbench', 'v1')

            def fwd(pixel, audio):
                clean = build_view(processor, model, system_prompt,
                                   user_prompt, pixel, audio)
                out = retry(lambda: view_logp(thinker, clean, gen_ids),
                            tag='forward')
                torch.cuda.empty_cache()
                return out

            lp = {}
            lp['full'] = fwd(own_v, own_a)
            lp['sv_avh'] = fwd(partner_avh[1], own_a)
            lp['sa_avh'] = fwd(own_v, partner_avh[2])
            if partner_mix is not None:
                lp['sv_mix'] = fwd(partner_mix[1], own_a)
                lp['sa_mix'] = fwd(own_v, partner_mix[2])

            base_lp = np.array(rec['logp_list'], dtype=np.float64)
            repro = (float(np.max(np.abs(lp['full'] - base_lp)))
                     if len(base_lp) == len(lp['full']) else float('nan'))

            toks = [tokenizer.decode([t], skip_special_tokens=True)
                    for t in rec['token_ids']]
            cls = [token_class(t, toks[j - 1] if j else None)
                   for j, t in enumerate(toks)]

            fout.write(json.dumps({
                'id': rec['id'], 'gt': rec['gt'], 'pred': rec['pred'],
                'question': rec['question'], 'f1': rec['_f1'],
                'pred_words': len(rec['pred'].split()),
                'tokens': toks, 'classes': cls,
                'lp_full': lp['full'].round(4).tolist(),
                'lp_sv_avh': lp['sv_avh'].round(4).tolist(),
                'lp_sa_avh': lp['sa_avh'].round(4).tolist(),
                'lp_sv_mix': lp['sv_mix'].round(4).tolist()
                             if 'sv_mix' in lp else None,
                'lp_sa_mix': lp['sa_mix'].round(4).tolist()
                             if 'sa_mix' in lp else None,
                'repro_err': round(repro, 4),
                'partner_avh_gt': partner_avh[0]['gt'][:80],
                'partner_mix': mix_partner_id,
            }, ensure_ascii=False) + '\n')
            fout.flush()
        except Exception as e:
            print(f'  [ERROR] id={rec["id"]}: {type(e).__name__}: {e}',
                  flush=True)
        if (i + 1) % 5 == 0:
            print(f'  [{i+1}/{len(sample)}] {time.time()-t0:.0f}s '
                  f'mem={torch.cuda.max_memory_allocated()/2**30:.1f}G',
                  flush=True)
        torch.cuda.empty_cache()

    fout.close()
    print(f'[done shard {args.shard_idx}] -> {out_path} '
          f'({time.time()-t0:.0f}s)', flush=True)


# ---------------------------------------------------------------------------
# Aggregate: GATE 1.
# ---------------------------------------------------------------------------
def iter_token_rows(s):
    """(word, attr_class, hit, dv_avh, da_avh, dv_mix, da_mix, token_index)."""
    gt_tok = set(content_words(s['gt']))
    dva = np.array(s['lp_full']) - np.array(s['lp_sv_avh'])
    daa = np.array(s['lp_full']) - np.array(s['lp_sa_avh'])
    dvm = (np.array(s['lp_full']) - np.array(s['lp_sv_mix'])
           if s.get('lp_sv_mix') else None)
    dam = (np.array(s['lp_full']) - np.array(s['lp_sa_mix'])
           if s.get('lp_sa_mix') else None)
    for j, (t, c) in enumerate(zip(s['tokens'], s['classes'])):
        if c != 'content' and c != 'num':
            continue
        w = norm_word(t)
        if not w:
            continue
        ac = attr_class(w)
        yield {
            'sample_i': s['sample_i'], 'j': j, 'w': w,
            'cls': c, 'ac': ac, 'hit': w in gt_tok,
            'dv': float(dva[j]), 'da': float(daa[j]),
            'dv_mix': float(dvm[j]) if dvm is not None else None,
            'da_mix': float(dam[j]) if dam is not None else None,
        }


def claim_rows_for(s, rows_by_sample):
    """Claim-level records with attr-priority score (avh pool primary)."""
    spans = claim_spans(s['tokens'], s['classes'])
    rows = rows_by_sample
    out = []
    for ci, (lo, hi) in enumerate(spans):
        cw = [r for r in rows if lo <= r['j'] < hi]
        text = ''.join(s['tokens'][lo:hi]).strip()
        prec, _ = overlap_precision(text, s['gt'])
        attrs = [r for r in cw if r['ac']]
        if attrs:
            score = float(np.mean([r[MATCHED_DELTA[r['ac']]] for r in attrs]))
        elif cw:
            score = float(np.mean([(r['dv'] + r['da']) / 2 for r in cw]))
        else:
            score = None
        # claim hit label: overlap precision >= 0.5 (probe_claim_cf protocol)
        hit = prec >= 0.5
        out.append({
            'sample_i': s['sample_i'], 'claim_i': ci, 'lo': lo, 'hi': hi,
            'text': text, 'n_tokens': hi - lo, 'prec': round(prec, 3),
            'hit': bool(hit), 'score': round(score, 4) if score is not None else None,
            'has_attr': bool(attrs), 'n_attr': len(attrs),
        })
    return out


def aggregate(args):
    files = sorted(glob.glob(args.shard_glob))
    assert files, f'no shard files match {args.shard_glob}'
    per_sample = []
    for f in files:
        for line in open(f):
            line = line.strip()
            if line:
                r = json.loads(line)
                r['sample_i'] = len(per_sample)
                per_sample.append(r)
    print(f'[aggregate] {len(per_sample)} samples from {len(files)} files',
          flush=True)

    # token rows
    token_rows, rows_by_sample = [], {}
    for s in per_sample:
        rs = list(iter_token_rows(s))
        rows_by_sample[s['sample_i']] = rs
        token_rows.extend(rs)
    claim_rows = []
    for s in per_sample:
        claim_rows.extend(claim_rows_for(s, rows_by_sample[s['sample_i']]))

    def sel(ac=None, hit=None, pool='avh', matched=None):
        out = []
        for r in token_rows:
            if ac is not None and r['ac'] != ac:
                continue
            if matched is not None and r['ac'] is None:
                continue          # modality-matched score only for attr words
            if pool == 'avh':
                key = MATCHED_DELTA[r['ac']] if matched else \
                    UNMATCHED_DELTA[r['ac']] if matched is False else 'dv'
                v = r[key]
            else:
                key = {'dv': 'dv_mix', 'da': 'da_mix'}[
                    MATCHED_DELTA[r['ac']] if matched else
                    UNMATCHED_DELTA[r['ac']]]
                v = r[key]
                if v is None:
                    continue
            if hit is None or r['hit'] == hit:
                out.append(v)
        return out

    report = {'n_samples': len(per_sample),
              'repro_err_median': float(np.median(
                  [s['repro_err'] for s in per_sample])),
              'repro_err_max': float(np.max([s['repro_err'] for s in per_sample])),
              'n_token_rows': len(token_rows)}

    def block(name, h, m):
        from scipy.stats import mannwhitneyu
        if len(h) < 2 or len(m) < 2:
            return {'n_hit': len(h), 'n_miss': len(m), 'AUC': None}
        a, p = auc_hit_vs_miss(h, m)
        return {'n_hit': len(h), 'n_miss': len(m),
                'mean_hit': round(float(np.mean(h)), 3),
                'mean_miss': round(float(np.mean(m)), 3),
                'AUC': round(a, 4), 'MWU_p': f'{p:.2e}'}

    # GATE 1 primary: pooled attribute tokens, matched-modality, avh pool
    h = sel(pool='avh', matched=True, hit=True)
    m = sel(pool='avh', matched=True, hit=False)
    # cluster bootstrap needs per-claim rows; emulate via token rows grouped
    attr_rows = [r for r in token_rows if r['ac']]

    def boot_ci(rows, key_fn, n_boot=1000, seed=0):
        rng = random.Random(seed)
        by_s = {}
        for r in rows:
            by_s.setdefault(r['sample_i'], []).append(r)
        samples = list(by_s.values())
        aucs = []
        for _ in range(n_boot):
            pool = []
            for _ in range(len(samples)):
                pool.extend(rng.choice(samples))
            hh = [key_fn(r) for r in pool if r['hit']]
            mm = [key_fn(r) for r in pool if not r['hit']]
            if len(hh) >= 2 and len(mm) >= 2:
                a, _ = auc_hit_vs_miss(hh, mm)
                if a == a:
                    aucs.append(a)
        return [round(float(np.quantile(aucs, 0.025)), 4),
                round(float(np.quantile(aucs, 0.975)), 4)] if len(aucs) > 200 \
            else None

    prim = block('primary', h, m)
    prim['boot_CI95'] = boot_ci(attr_rows, lambda r: r[MATCHED_DELTA[r['ac']]])
    report['gate1_primary_avh_matched'] = prim

    # controls
    report['ctrl_unmatched_avh'] = block(
        'unmatched', sel(pool='avh', matched=False, hit=True),
        sel(pool='avh', matched=False, hit=False))
    report['ctrl_matched_mix'] = block(
        'mix', sel(pool='mix', matched=True, hit=True),
        sel(pool='mix', matched=True, hit=False))
    for ac in ('color', 'count', 'asrc'):
        report[f'by_class_{ac}'] = block(
            ac, sel(ac=ac, pool='avh', matched=True, hit=True),
            sel(ac=ac, pool='avh', matched=True, hit=False))
        report[f'by_class_{ac}_mix'] = block(
            ac, sel(ac=ac, pool='mix', matched=True, hit=True),
            sel(ac=ac, pool='mix', matched=True, hit=False))
    nonattr = [r for r in token_rows if not r['ac'] and r['cls'] == 'content']
    report['ctrl_nonattr_content'] = block(
        'nonattr', [r['dv'] for r in nonattr if r['hit']],
        [r['dv'] for r in nonattr if not r['hit']])

    # claim level (secondary)
    hc = [c['score'] for c in claim_rows if c['hit'] and c['score'] is not None]
    mc = [c['score'] for c in claim_rows if not c['hit'] and c['score'] is not None]
    report['claim_level_attr_priority'] = block('claims', hc, mc)
    ha = [c['score'] for c in claim_rows if c['hit'] and c['has_attr']]
    ma = [c['score'] for c in claim_rows if not c['hit'] and c['has_attr']]
    report['claim_level_attr_only'] = block('claims_attr', ha, ma)

    # coverage
    cov = Counter(r['ac'] for r in token_rows)
    report['attribute_coverage'] = {
        'color': cov.get('color', 0), 'count': cov.get('count', 0),
        'asrc': cov.get('asrc', 0),
        'content_nonattr': sum(1 for r in token_rows if not r['ac']),
        'n_claims': len(claim_rows),
        'n_claims_with_attr': sum(1 for c in claim_rows if c['has_attr']),
    }

    # verdict
    g1_pass = (prim.get('AUC') is not None and prim['AUC'] > 0.65 and
               prim['boot_CI95'] is not None and prim['boot_CI95'][0] > 0.5)
    report['verdict_gate1'] = 'PASS' if g1_pass else 'FAIL'

    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, 'w') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    # dumps for the sim
    with open(args.report.replace('.json', '_tokens.jsonl'), 'w') as f:
        for r in token_rows:
            f.write(json.dumps(r) + '\n')
    with open(args.report.replace('.json', '_claims.jsonl'), 'w') as f:
        for c in claim_rows:
            f.write(json.dumps(c) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f'[done] -> {args.report}')


# ---------------------------------------------------------------------------
# GATE 2: offline benefit-structure simulation (zero GPU).
# ---------------------------------------------------------------------------
HEDGE_SUBS = [
    (' is ', ' appears to be '), (' are ', ' appear to be '),
    (' has ', ' appears to have '), (' have ', ' appear to have '),
    (' wears ', ' appears to wear '),
]


def _span_text(tokens, lo, hi):
    return ''.join(t for t in tokens[lo:hi] if t is not None)


def sim_edit(s, claims, tau, action, rng=None, rand_frac=None):
    tokens = s['tokens'][:]
    mine = [c for c in claims if c['sample_i'] == s['sample_i']]
    flagged = []
    for c in mine:
        if rand_frac is not None:
            if rng.random() < rand_frac:
                flagged.append(c)
        elif c['score'] is not None and c['score'] < tau:
            flagged.append(c)
    out = tokens[:]
    if action == 'delete-claim':
        for c in flagged:
            for j in range(c['lo'], c['hi']):
                out[j] = None
    elif action == 'attr-drop':
        # drop ONLY the flagged attribute words, keep entities/verbs
        for c in flagged:
            for r in c.get('attr_rows', []):
                out[r['j']] = None
    elif action == 'hedge':
        # 降级陈述: copula/verb -> "appears to be" (span-level text rewrite)
        for c in flagged:
            txt = _span_text(out, c['lo'], c['hi'])
            low = ' ' + txt.strip() + ' '
            for a, b in HEDGE_SUBS:
                if a in low:
                    low = low.replace(a, b, 1)
                    break
            out[c['lo']] = low
            for j in range(c['lo'] + 1, c['hi']):
                out[j] = None
    elif action == 'evasive-repl':
        # replace the flagged claim with a short evasive clause (NOT deletion:
        # sentence count/position preserved)
        for c in flagged:
            out[c['lo']] = ' some details are unclear,'
            for j in range(c['lo'] + 1, c['hi']):
                out[j] = None
    text = ''.join(t for t in out if t is not None).strip()
    text = text.strip(' ,;')
    if text and text[-1] not in '.!?':
        text += '.'
    if not text:
        text = 'The video shows unclear content.'
    return text, len(flagged)


def sim(args):
    from scipy.stats import wilcoxon

    # reload full sample+token data (tokens/gt from shard files)
    files = sorted(glob.glob(args.shard_glob))
    per_sample = []
    for f in files:
        for line in open(f):
            if line.strip():
                r = json.loads(line)
                r['sample_i'] = len(per_sample)
                per_sample.append(r)
    claims = [json.loads(l) for l in
              open(args.report.replace('.json', '_claims.jsonl'))]
    tokens_rows = [json.loads(l) for l in
                   open(args.report.replace('.json', '_tokens.jsonl'))]
    rows_by_sample = {}
    for r in tokens_rows:
        rows_by_sample.setdefault(r['sample_i'], []).append(r)
    # attach attribute rows to claims
    for c in claims:
        c['attr_rows'] = [r for r in rows_by_sample.get(c['sample_i'], [])
                          if c['lo'] <= r['j'] < c['hi'] and r['ac']]
    by_si = {}
    for c in claims:
        by_si.setdefault(c['sample_i'], []).append(c)

    # dev/test split stratified by F1 tercile (pre-registered)
    rng = random.Random(args.seed)
    order = sorted(range(len(per_sample)), key=lambda i: per_sample[i]['f1'])
    k = max(1, len(order) // 3)
    tiers = [order[:k], order[k:2 * k], order[2 * k:]]
    dev_idx, test_idx = [], []
    for t in tiers:
        rng.shuffle(t)
        n_dev = round(len(t) * 0.5)
        dev_idx.extend(t[:n_dev]); test_idx.extend(t[n_dev:])

    def base_f1(i):
        return token_f1(per_sample[i]['pred'], per_sample[i]['gt'])

    # tau grid from dev claim scores
    dev_scores = [c['score'] for i in dev_idx for c in by_si.get(i, [])
                  if c['score'] is not None]
    taus = list(np.quantile(dev_scores, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]))

    out = {'n_dev': len(dev_idx), 'n_test': len(test_idx),
           'test_base_f1': round(float(np.mean(
               [base_f1(i) for i in test_idx])), 4)}

    # dev grid: pick best (action, tau)
    grid = {}
    for action in ('delete-claim', 'attr-drop', 'hedge', 'evasive-repl'):
        for tau in taus:
            f1s = [token_f1(sim_edit(per_sample[i], claims, tau, action)[0],
                            per_sample[i]['gt']) for i in dev_idx]
            grid[(action, round(float(tau), 3))] = round(float(np.mean(f1s)), 4)
    best = max(grid, key=grid.get)
    out['dev_grid_top'] = [[str(k), v] for k, v in
                           sorted(grid.items(), key=lambda kv: -kv[1])[:8]]
    action, tau = best
    out['dev_best'] = {'action': action, 'tau': float(tau),
                       'f1': grid[best]}

    # test: main + random control
    f1_main = np.array([token_f1(sim_edit(per_sample[i], claims, tau,
                                          action)[0],
                                 per_sample[i]['gt']) for i in test_idx])
    n_flag = sum(sim_edit(per_sample[i], claims, tau, action)[1]
                 for i in test_idx)
    n_claim_test = sum(len(by_si.get(i, [])) for i in test_idx)
    frac = n_flag / max(1, n_claim_test)
    rng2 = random.Random(49)
    f1_rand = np.array([token_f1(sim_edit(per_sample[i], claims, tau, action,
                                          rng=rng2, rand_frac=frac)[0],
                                 per_sample[i]['gt']) for i in test_idx])
    b = np.array([base_f1(i) for i in test_idx])
    out['action'] = action
    out['tau'] = float(tau)
    out['test_main_f1'] = round(float(f1_main.mean()), 4)
    out['test_main_vs_base'] = {
        'mean_diff': round(float((f1_main - b).mean()), 4),
        'wilcoxon_p': round(float(wilcoxon(f1_main, b).pvalue), 5)}
    out['test_randctrl_f1'] = round(float(f1_rand.mean()), 4)
    out['test_rand_vs_base'] = {
        'mean_diff': round(float((f1_rand - b).mean()), 4),
        'wilcoxon_p': round(float(wilcoxon(f1_rand, b).pvalue), 5)}
    out['main_vs_rand_p'] = round(float(wilcoxon(f1_main, f1_rand).pvalue), 5)
    out['flag_rate'] = round(frac, 4)

    # full action table on test at dev-selected tau (transparency)
    table = {}
    for a2 in ('delete-claim', 'attr-drop', 'hedge', 'evasive-repl'):
        f = np.array([token_f1(sim_edit(per_sample[i], claims, tau, a2)[0],
                               per_sample[i]['gt']) for i in test_idx])
        table[a2] = {'f1': round(float(f.mean()), 4),
                     'vs_base': round(float((f - b).mean()), 4),
                     'p': round(float(wilcoxon(f, b).pvalue), 5)}
    out['action_table'] = table

    g2_pass = (out['test_main_vs_base']['mean_diff'] > 0 and
               out['test_main_vs_base']['wilcoxon_p'] < 0.05 and
               out['main_vs_rand_p'] < 0.05)
    out['verdict_gate2'] = 'PASS' if g2_pass else 'FAIL'

    sim_path = args.report.replace('.json', '_sim.json')
    json.dump(out, open(sim_path, 'w'), ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f'[done] -> {sim_path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scored', default='outputs/avhbench_base_model/'
                                        'avhbench_base_model_full.jsonl')
    ap.add_argument('--n_samples', type=int, default=130)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--shard_idx', type=int, default=0)
    ap.add_argument('--n_shards', type=int, default=1)
    ap.add_argument('--out_pattern',
                    default='outputs/claim_probe/avh_claim_shard{idx}of{n}.jsonl')
    ap.add_argument('--aggregate', action='store_true')
    ap.add_argument('--sim', action='store_true')
    ap.add_argument('--shard_glob',
                    default='outputs/claim_probe/avh_claim_shard*of*.jsonl')
    ap.add_argument('--report',
                    default='outputs/claim_probe/avh_claim_probe_report.json')
    args = ap.parse_args()

    if args.aggregate:
        aggregate(args)
    elif args.sim:
        sim(args)
    else:
        run_shard(args)


if __name__ == '__main__':
    main()
