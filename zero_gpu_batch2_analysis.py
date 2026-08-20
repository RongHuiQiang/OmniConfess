# -*- coding: utf-8 -*-
"""
zero_gpu_batch2_analysis.py -- zero-GPU deep-dive batch 2 (2026-08-17).

Three pre-registered analyses over already-on-disk data only (no GPU, no API).

  P1  Signal horse race (Delta vs entropy proxies vs margin vs VAS) against
      Decision-Tokens' (ICML'26, 2605.16874 app. C.3) claim that a *zero-cost*
      base-entropy trigger already recovers most of the benefit. Scenarios:
        phd@Q2.5  probe outputs/cf_probe/phd_probe_full.jsonl (n=906)
        cmm@Q2.5  probe outputs/cf_probe/cmm_probe_full.jsonl (n=720)
        cmm@Qwen3 probe outputs/cf_probe/cmm_probe_qwen3.jsonl (n=98, dev98)
      Targets:
        (a) base_error = base_correct==0   [literature error-detector frame]
        (b) gt_yes                        [our correction/deployment frame]
      Entropy caveat (checked, negative): the next-token distribution is NOT
      recoverable from trajectories (logp_list stores only the chosen token),
      so Decision-Tokens-style normalized entropy is proxied by
        S1a surprisal(dec) = -dec_logp_recorded  (H >= -log p_max: lower bound)
        S1b -norm_logp (whole output segment)
      Margin: S2 = -|margin_full|.  Delta: pre-registered orientation =
        phd        signed operator-caliber d_noctx  (context pulls Yes)
        cmm / qwen3 signed -d_cf_sum                (registered gt orientation)
      Deployed composite: |margin_full + lam*Delta| with the registered lam
        (phd 2.0 / cmm 1.0); its AUC minus best zero-cost proxy's AUC is the
        *marginal* value of Delta over the literature signals.
      Gate (pre-registered): pure-Delta AUC >= best zero-cost proxy AUC + 0.05
        -> "counterfactual signal has independent discriminative value";
        else "literature zero-cost signals already cover most of the
        discrimination; Delta's value is directionality / evidence attribution".
      VAS (VISTA): not computable for the P1 probes offline (needs mid-layer
      attention), but VAS *was* extracted for the haloquest/ragtruth BoN
      pools -> P1d races VAS vs CE vs Delta at candidate level there.

  P2  haloquest / avhbench error budget: Token-F1 loss mass decomposed into
      mutually exclusive primary-cause buckets, base AND final chain, with
      oracle re-scoring (raw_output caliber, as in ragtruth) per bucket, plus
      a "grounded-but-wrong" split (recall>=0.5 with wrong extra content) to
      interface with TTSP's grounding-paradox narrative, and a Vote-SC
      word-level audit (kept vs dropped word gt-hit; oracle-of-7 recompute).

  P3  Method fingerprint: Qwen2.5 10 methods x 6 datasets matrix recomputed
      with the single_eval.py formulas; pairwise Spearman of method rankings
      (common methods, permutation p), average-linkage clustering on 1-rho;
      plus decision-signal sparsity (Gini / top-decile share of |Delta| per
      dataset, probes) to test whether same-cluster datasets share the
      "sparse decision signal" structure (literature 6: Gini 0.936, 8% tokens).

Usage: /usr/bin/python3 zero_gpu_batch2_analysis.py
Writes outputs/zero_gpu_batch2_analysis.json (full numbers) + prints report.
"""
import json, re, glob, math, random, string, os
from collections import Counter

import numpy as np
from scipy import stats as st

ROOT = '/home/tsq/rhq/omniconfess/OmniConfess'
OUT_JSON = os.path.join(ROOT, 'outputs/zero_gpu_batch2_analysis.json')

# ---------------------------------------------------------------------------
# shared metrics
# ---------------------------------------------------------------------------

def auc(score, label):
    score = np.asarray(score, float); label = np.asarray(label, int)
    pos = score[label == 1]; neg = score[label == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    order = np.argsort(score, kind='mergesort')
    ranks = np.empty(len(score), float)
    sv = score[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    rp = ranks[label == 1].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def auc_ci(score, label, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    score = np.asarray(score, float); label = np.asarray(label, int)
    n = len(score); vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        a = auc(score[idx], label[idx])
        if not math.isnan(a):
            vals.append(a)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals \
        else (float('nan'), float('nan'))


def paired_auc_diff_ci(sa, sb, label, n_boot=2000, seed=1):
    rng = np.random.default_rng(seed)
    sa = np.asarray(sa, float); sb = np.asarray(sb, float); label = np.asarray(label, int)
    n = len(label); vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        d = auc(sa[idx], label[idx]) - auc(sb[idx], label[idx])
        if not math.isnan(d):
            vals.append(d)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals \
        else (float('nan'), float('nan'))


def f1_yes(gt, pr):
    gt = np.asarray(gt, int); pr = np.asarray(pr, int)
    tp = int(((gt == 1) & (pr == 1)).sum()); fp = int(((gt == 0) & (pr == 1)).sum())
    fn = int(((gt == 1) & (pr == 0)).sum())
    if tp == 0:
        return 0.0
    p = tp / (tp + fp); r = tp / (tp + fn)
    return 2 * p * r / (p + r)


def gini(x):
    x = np.sort(np.abs(np.asarray(x, float)))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return float('nan')
    cum = np.cumsum(x)
    return float((n + 1 - 2 * (cum / cum[-1]).sum()) / n)


def jl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


# ===========================================================================
# P1 -- signal horse race (sample level)
# ===========================================================================
# cmm-family trajectories are round-robin 8-shard concatenations; probe uid
# cmm_XXXXXX is the global dataset index -> position via shard offsets.
_CMM_SIZES = [91] + [90] * 7
_CMM_OFF = [0]
for _s in _CMM_SIZES[:-1]:
    _CMM_OFF.append(_CMM_OFF[-1] + _s)


def cmm_gpos(g):
    return _CMM_OFF[g % 8] + g // 8


def load_probe_traj(probe_path, traj_path, cmm_style=False):
    """Join probe rows to base-trajectory rows.

    cmm_style: probe carries uid cmm_XXXXXX = global dataset index; the
    trajectory full file is the round-robin 8-shard concatenation in shard
    order (verified by question/gt/pred equality).
    phd: probe/traj carry shard-LOCAL ids (114 x 8 rows). Join = same id
    group, disambiguated by (gt, pred) and, if still ambiguous, by the
    decision-token logprob (logp_list[dec_index] ~= dec_logp_recorded,
    tf_repro ~ 0). 906/906 unique matches (verified).
    """
    probe = jl(probe_path)
    traj = jl(traj_path)
    if not cmm_style:
        from collections import defaultdict
        by_id = defaultdict(list)
        for i, x in enumerate(traj):
            by_id[str(x['id'])].append(i)
    rows, missing = [], 0
    for r in probe:
        if cmm_style:
            g = int(r['uid'].split('_')[1])
            t = traj[cmm_gpos(g)] if cmm_gpos(g) < len(traj) else None
        else:
            cands = [i for i in by_id[str(r['id'])]
                     if traj[i]['gt'] == r['gt'] and traj[i]['pred'] == r['pred']]
            t = None
            if len(cands) == 1:
                t = traj[cands[0]]
            elif len(cands) > 1:
                di = r.get('dec_index', 0)
                for i in cands:
                    lp = traj[i].get('logp_list') or []
                    if di < len(lp) and abs(lp[di] - r['dec_logp_recorded']) < 0.1:
                        t = traj[i]
                        break
                if t is None:
                    t = traj[cands[0]]
        if t is None:
            missing += 1
            continue
        assert t['pred'] == r['pred'] and t['gt'] == r['gt'], (r['id'],)
        rows.append((r, t))
    return rows, missing


def p1_run(name, pairs, lam, delta_op_of, delta_variant, note):
    rows = [dict(
        base_correct=r['base_correct'], gt_yes=r['gt_yes'], pred=r['pred'],
        margin_full=r['margin_full'], dec_logp=r['dec_logp_recorded'],
        norm_logp=t.get('norm_logp'), delta_op=delta_op_of(r)) for r, t in pairs]
    y_err = np.array([1 - r['base_correct'] for r in rows])
    y_gt = np.array([r['gt_yes'] for r in rows])
    d = np.array([r['delta_op'] for r in rows])
    mf = np.array([r['margin_full'] for r in rows])
    sig = {
        'S1a_surp_dec': -np.array([r['dec_logp'] for r in rows]),
        'S1b_neg_norm_logp': -np.array([r['norm_logp'] if r['norm_logp'] is not None
                                        else np.nan for r in rows]),
        'S2_neg_abs_margin': -np.abs(mf),
        'S3_neg_abs_delta': -np.abs(d),
        'delta_signed_prereg': d,
        'delta_signed_opp': -d,
        'neg_abs_cf_score': -np.abs(mf + lam * d),
    }
    res = dict(n=len(rows), lam=lam, delta_variant=delta_variant, note=note)
    tab = {}
    for k, s in sig.items():
        m = ~np.isnan(np.array(s, float))
        tab[k] = dict(auc=auc(np.array(s, float)[m], y_err[m]),
                      ci=auc_ci(np.array(s, float)[m], y_err[m]), n=int(m.sum()))
    res['auc_error'] = tab
    zero = {k: tab[k] for k in ('S1a_surp_dec', 'S1b_neg_norm_logp', 'S2_neg_abs_margin')}
    bz = max(zero, key=lambda k: zero[k]['auc'])
    res['best_zero_cost_error'] = dict(name=bz, **zero[bz])
    sd = np.array(sig[delta_variant], float); sz = np.array(sig[bz], float)
    m = ~(np.isnan(sd) | np.isnan(sz))
    res['gate_pure_delta'] = dict(
        auc_delta=auc(sd[m], y_err[m]), auc_proxy=auc(sz[m], y_err[m]),
        margin=auc(sd[m], y_err[m]) - auc(sz[m], y_err[m]),
        ci=paired_auc_diff_ci(sd[m], sz[m], y_err[m]),
        passes=bool(auc(sd[m], y_err[m]) - auc(sz[m], y_err[m]) >= 0.05))
    sc = np.array(sig['neg_abs_cf_score'], float)
    res['marginal_value_composite'] = dict(
        auc_composite=auc(sc[m], y_err[m]),
        delta_vs_proxy_ci=paired_auc_diff_ci(sc[m], sz[m], y_err[m]))
    # gt frame (correction)
    tg = {}
    for k in ('delta_signed_prereg', 'delta_signed_opp'):
        tg[k] = dict(auc=auc(sig[k], y_gt), ci=auc_ci(sig[k], y_gt))
    tg['margin_full_signed'] = dict(auc=auc(mf, y_gt), ci=auc_ci(mf, y_gt))
    cf = mf + lam * d
    tg['cf_score'] = dict(auc=auc(cf, y_gt), ci=auc_ci(cf, y_gt))
    tg['margin_plus_cfview_check'] = dict(auc=auc(mf, y_gt))
    res['auc_gt'] = tg
    # conditional within pred=yes (FP-detection deployment view)
    sel = np.array([r['pred'] == 'yes' for r in rows])
    if sel.sum() > 5:
        res['auc_within_predyes_gt'] = dict(
            n=int(sel.sum()), margin_full=auc(mf[sel], y_gt[sel]),
            delta_signed_prereg=auc(sd[sel], y_gt[sel]),
            neg_abs_margin=auc(-np.abs(mf)[sel], y_gt[sel]))
        res['auc_within_predyes_error'] = dict(
            n=int(sel.sum()), margin_full=auc(-np.abs(mf)[sel], y_err[sel]),
            surprisal=auc(sig['S1a_surp_dec'][sel], y_err[sel]),
            neg_norm_logp=auc(np.array(sig['S1b_neg_norm_logp'], float)[sel], y_err[sel]),
            neg_abs_delta=auc(-np.abs(d)[sel], y_err[sel]))
    return res


# ===========================================================================
# P1d -- candidate-level race on the haloquest BoN pool (VAS / CE / Delta)
# ===========================================================================

def p1d_candidate_race():
    d = json.load(open(f'{ROOT}/outputs/bon_probe/haloquest_bon_feat.json'))
    rows = d['rows']
    pool = {r['id']: {t['rollout']: t for t in r['trajectories']}
            for r in jl(f'{ROOT}/outputs/haloquest_bon_grounding/haloquest_bon_grounding.jsonl')}
    V, C, G, GT_, Y, sid = [], [], [], [], [], []
    for r in rows:
        f1s = [c['f1'] for c in r['candidates']]
        med = float(np.median(f1s))
        for c in r['candidates']:
            v = c['VAS']['14']['vas_content']
            if v is None:
                v = c['VAS']['14']['vas_all']
            gs = pool.get(r['id'], {}).get(c['rollout'], {}).get('grounding_scores')
            V.append(v)
            C.append(c['CE']['CE_top5mean'])
            G.append(c['g_content'] if c['g_content'] is not None else 0.0)
            GT_.append(gs['g_traj_mean'] if gs else 0.0)
            Y.append(1 if c['f1'] >= med else 0)     # good-vs-bad, median split
            sid.append(r['id'])
    V = np.array(V, float); C = np.array(C, float)
    G = np.array(G, float); GT_ = np.array(GT_, float); Y = np.array(Y)
    sid = np.array(sid)
    # orientation: hallucinated candidates have LOW VAS (VISTA), LOW grounding
    # delta; CE = top5 mean logp (high = confident). good candidate -> high VAS,
    # high g, high CE.
    sigs = {'VAS(zero-cost, literature)': V, 'CE_top5mean(zero-cost entropy)': C,
            'Delta_g_content(1 extra fwd)': G, 'g_traj_mean': GT_,
            'combo z(D)-.25z(CE)+2z(VAS)': None}
    # z within pool (as registered); recompute combo with dev-free global z for
    # the AUC race only (weights frozen at registered values)
    def z(x):
        x = np.array(x, float)
        sd = x.std()
        return (x - x.mean()) / (sd if sd > 0 else 1.0)
    sigs['combo z(D)-.25z(CE)+2z(VAS)'] = z(G) - 0.25 * z(C) + 2.0 * z(V)
    out = dict(n_cand=len(Y), n_samples=len(set(sid.tolist())), target='cand F1 >= sample median')
    tab = {}
    for k, s in sigs.items():
        tab[k] = dict(auc=auc(s, Y), ci=auc_ci(s, Y))
    out['auc_good_cand'] = tab
    return out


# ===========================================================================
# P2 -- error budget decomposition (Token-F1 caliber of single_eval.py)
# ===========================================================================

STOPWORDS = set([
    "a", "an", "the", "and", "or", "but", "if", "because", "as", "what",
    "which", "this", "that", "these", "those", "then", "just", "so", "than",
    "such", "both", "through", "about", "for", "is", "are", "was", "were",
    "of", "while", "during", "to", "in", "on", "at", "by", "with", "it", "there"
])


def get_tokens(s):
    s = str(s).lower().translate(str.maketrans('', '', string.punctuation))
    return [w for w in s.split() if w not in STOPWORDS and w.strip()]


def token_f1(pred, gt):
    p_t, g_t = get_tokens(pred), get_tokens(gt)
    if not p_t or not g_t:
        return 0.0, 0.0, 0.0
    common = Counter(p_t) & Counter(g_t)
    hit = sum(common.values())
    pr = hit / len(p_t); rc = hit / len(g_t)
    f1 = (2 * pr * rc) / (pr + rc) if (pr + rc) > 0 else 0.0
    return f1, pr, rc


NEG_RE = re.compile(
    r"\b(not|no|cannot|can't|unable|isn't|aren't|doesn't|don't|won't|without|"
    r"neither|nor|nothing|nowhere)\b", re.I)


def halo_bucket(row):
    pred = str(row.get('pred', '')); gt = str(row.get('gt', ''))
    raw = row.get('raw_output', None)
    f1, pr, rc = token_f1(pred, gt)
    raw_f1 = token_f1(str(raw), gt)[0] if raw is not None else f1
    has_raw = raw is not None
    pw, gw = len(str(pred).split()), len(str(gt).split())
    if has_raw and (abs(raw_f1 - f1) > 0.01 or str(pred).strip() != str(raw).strip()):
        cause = 'format_extraction'
    elif pw <= 3:
        cause = 'degenerate_short'
    elif bool(NEG_RE.search(pred)) != bool(NEG_RE.search(gt)):
        cause = 'polarity_mismatch'
    elif gw >= 4 and (pw < 0.5 * gw or pw > 2.0 * gw):
        cause = 'length_mismatch'
    else:
        cause = 'content_error'
    return cause, dict(f1=f1, pr=pr, rc=rc, raw_f1=raw_f1, pw=pw, gw=gw,
                       pred_neg=bool(NEG_RE.search(pred)), gt_neg=bool(NEG_RE.search(gt)))


def p2_halo(path, label):
    rows = jl(path)
    out = []
    for r in rows:
        cause, info = halo_bucket(r)
        info.update(id=r.get('id'), group=r.get('group_key'), cause=cause)
        out.append(info)
    total_loss = sum(100 - 100 * i['f1'] for i in out)
    by = {}
    for i in out:
        b = by.setdefault(i['cause'], dict(n=0, loss=0.0, f1s=[], raw_gain=0.0, groups=Counter()))
        b['n'] += 1; b['loss'] += 100 - 100 * i['f1']; b['f1s'].append(100 * i['f1'])
        b['raw_gain'] += 100 * (i['raw_f1'] - i['f1'])
        b['groups'][i['group']] += 1
    for k, b in by.items():
        b['share_loss'] = b['loss'] / total_loss
        b['mean_f1'] = float(np.mean(b['f1s']))
        b['mean_raw_gain'] = b['raw_gain'] / b['n']
        b['groups'] = dict(b['groups'])
    # oracle ceilings: replace pred by gt (trivial) and by the best same-cause
    # row actually achieved (peer-achievable level)
    peer = {k: max(b['f1s']) for k, b in by.items()}
    return dict(label=label, n=len(out),
                mean_f1=100 * float(np.mean([i['f1'] for i in out])),
                raw_rescore_mean=100 * float(np.mean([i['raw_f1'] for i in out])),
                total_loss_mass=total_loss, buckets=by, peer_best_f1=peer,
                polarity_xtab=dict(
                    pred_neg_gt_neg=int(sum(1 for i in out if i['pred_neg'] and i['gt_neg'])),
                    pred_neg_gt_pos=int(sum(1 for i in out if i['pred_neg'] and not i['gt_neg'])),
                    pred_pos_gt_neg=int(sum(1 for i in out if not i['pred_neg'] and i['gt_neg'])),
                    pred_pos_gt_pos=int(sum(1 for i in out if not i['pred_neg'] and not i['gt_neg']))))


_AUDIO_STEMS = """
sound sounds noise noisy loud louder quiet quietly silence silent music
musical song songs melody tune rhythm beat beats audio audible hear hears
heard hearing voice voices speech speaking speaks spoke talk talks talking
narration narrates narrating scream screams screaming shout shouts shouting
yell yells yelling whisper whispers whispering humming hums hum buzz buzzing
buzzer drone drones droning roar roaring rumble rumbling engine engines
motor motors revving idling horn honk honks honking honked beep beeps beeping
alarm sirens siren ringing ring rings chime chimes chiming bell bells
applause clap claps clapping clapped cheering cheers cheer guitar guitars
piano drum drums drumming violin flutes flute trumpet saxophone keyboard
cymbals tambourine instrumental acoustic chirp chirps chirping chirped tweet
tweets tweeting squawk squawking squawks bark barks barking barked meow
purring purrs moo neigh quack quacking croak croaking hoot hooting howl
howling growl growling hiss hissing sizzle sizzling crackle crackling pop
pops popping whoosh swoosh swish swishing thud thuds thump thumping bang
bangs banging crash crashes crashing screech screeching squeak squeaks
squeaking squeal squealing grunt grunts snort snorts bleat clang clangs
clanging jingle jingles jingling rattle rattles rattling whistle whistles
whistling gasp gasps gasping sigh sighs sighing breath breathes breathing
snore snoring wind winds breeze rustle rustles rustling rustled splash
splashes splashing splashed trickle trickles trickling gurgle gurgles waves
rain rains raining thunder thundering storm footsteps footstep tapping taps
tapped click clicks clicking knock knocks knocking hammered hammering hammer
drilling drill sawing whirring whir vibration vibrating pitched tones
background screaming yelled shouted spoken
""".split()

_VISUAL_STEMS = """
red blue green yellow purple pink brown black white gray grey golden silver
dark bright colorful colored man men woman women person people boy boys girl
girls child children baby guy guys lady crowd audience player players worker
workers shirt jackets jacket dress dresses suit suits pants jeans shorts hat
hats cap helmet helmets shoes boots sweater hoodie uniform tie scarf gloves
glasses sunglasses mask hair beard face hands hand arm arms leg legs table
tables chair chairs car cars truck trucks bus buses motorcycle bicycles
bicycle bike plane airplane helicopter boat ship train road street building
buildings house room kitchen bathroom bedroom wall walls floor ceiling door
doors window windows desk screen phone camera bottle cup glass plate food
pizza cake ball dog cat dogs cats kitten puppy animal animals tree trees
grass rock rocks water river mountain sky cloud clouds forest field flowers
flower leaf leaves background podium microphone stage television laptop book
books paper box boxes striped suitcases
""".split()


def _inflect(stems):
    out = set()
    for s in stems:
        out.update({s, s + 's', s + 'es', s + 'ing', s + 'd', s + 'ed'})
        if s.endswith('e'):
            out.add(s[:-1] + 'ing')
        if s.endswith('y'):
            out.add(s[:-1] + 'ies')
    return {w for w in out if w}


AUDIO_WORDS = _inflect(_AUDIO_STEMS)
VISUAL_WORDS = _inflect(_VISUAL_STEMS)
_ov = AUDIO_WORDS & VISUAL_WORDS
AUDIO_WORDS -= _ov
VISUAL_WORDS -= _ov


def word_mod(w):
    if w in AUDIO_WORDS:
        return 'audio'
    if w in VISUAL_WORDS:
        return 'visual'
    return 'other'


def avh_bucket(row):
    pred = str(row.get('pred', row.get('response', '')))
    gt = str(row.get('gt', row.get('ground_truth', '')))
    raw = row.get('raw_output', None)
    f1, pr, rc = token_f1(pred, gt)
    raw_f1 = token_f1(str(raw), gt)[0] if raw is not None else f1
    p_t, g_t = get_tokens(pred), get_tokens(gt)
    g_a = {w for w in g_t if word_mod(w) == 'audio'}
    g_v = {w for w in g_t if word_mod(w) == 'visual'}
    p_a = {w for w in p_t if word_mod(w) == 'audio'}
    p_v = {w for w in p_t if word_mod(w) == 'visual'}
    rec_a = len(g_a & p_a) / len(g_a) if g_a else None
    rec_v = len(g_v & p_v) / len(g_v) if g_v else None
    if len(p_t) <= 2:
        cause = 'degenerate_empty'
    elif raw is not None and (abs(raw_f1 - f1) > 0.01 or pred.strip() != str(raw).strip()):
        cause = 'format_extraction'
    elif rc < pr - 0.05:
        cause = 'recall_loss_misswords'
    elif pr < rc - 0.05:
        cause = 'precision_loss_wrongwords'
    else:
        cause = 'balanced_loss'
    # TTSP-interface split: grounded-but-wrong (captured >=50% of gt content
    # words yet still losing mass) vs ungrounded-wrong
    grounded = rc >= 0.5
    return cause, dict(f1=f1, pr=pr, rc=rc, raw_f1=raw_f1, n_pred=len(p_t),
                       n_gt=len(g_t), rec_a=rec_a, rec_v=rec_v,
                       has_gt_audio=bool(g_a), has_gt_visual=bool(g_v),
                       grounded=grounded)


def p2_avh(path, label, vote_audit=False):
    rows = jl(path)
    out = []
    for r in rows:
        cause, info = avh_bucket(r)
        info['id'] = r.get('id', r.get('idx'))
        info['cause'] = cause
        out.append(info)
    total_loss = sum(100 - 100 * i['f1'] for i in out)
    by = {}
    for i in out:
        b = by.setdefault(i['cause'], dict(n=0, loss=0.0, f1s=[], raw_gain=0.0,
                                           grounded=0))
        b['n'] += 1; b['loss'] += 100 - 100 * i['f1']; b['f1s'].append(100 * i['f1'])
        b['raw_gain'] += 100 * (i['raw_f1'] - i['f1'])
        b['grounded'] += int(i['grounded'])
    for k, b in by.items():
        b['share_loss'] = b['loss'] / total_loss
        b['mean_f1'] = float(np.mean(b['f1s']))
        b['mean_raw_gain'] = b['raw_gain'] / b['n']
    rl = [i for i in out if i['cause'] == 'recall_loss_misswords']
    am = sum(1 for i in rl if (i['rec_a'] is not None and i['rec_a'] < 0.5) and (i['rec_v'] is None or i['rec_v'] >= 0.5))
    vm = sum(1 for i in rl if (i['rec_v'] is not None and i['rec_v'] < 0.5) and (i['rec_a'] is None or i['rec_a'] >= 0.5))
    bm = sum(1 for i in rl if (i['rec_a'] is not None and i['rec_a'] < 0.5) and (i['rec_v'] is not None and i['rec_v'] < 0.5))
    gw = [i for i in out if i['cause'] in ('precision_loss_wrongwords', 'balanced_loss')]
    res = dict(label=label, n=len(out),
               mean_f1=100 * float(np.mean([i['f1'] for i in out])),
               raw_rescore_mean=100 * float(np.mean([i['raw_f1'] for i in out])),
               total_loss_mass=total_loss, buckets=by,
               audio_recall_mean=float(np.mean([i['rec_a'] for i in out if i['rec_a'] is not None])),
               visual_recall_mean=float(np.mean([i['rec_v'] for i in out if i['rec_v'] is not None])),
               n_gt_with_audio=int(sum(1 for i in out if i['has_gt_audio'])),
               n_gt_with_visual=int(sum(1 for i in out if i['has_gt_visual'])),
               recall_loss_modality=dict(audio_only_miss=am, visual_only_miss=vm,
                                         both_miss=bm, n=len(rl)),
               grounded_but_wrong=dict(n=len(gw), n_grounded=int(sum(1 for i in gw if i['grounded'])),
                                       loss_grounded=100 * sum(1 - i['rc'] if False else 0 for i in gw)))
    # grounded-but-wrong loss share
    lg = sum(100 - 100 * i['f1'] for i in gw if i['grounded'])
    res['grounded_but_wrong'] = dict(n=len(gw), n_grounded=int(sum(1 for i in gw if i['grounded'])),
                                     share_of_loss=lg / total_loss)
    if vote_audit:
        # word-level audit: kept vs dropped-1arm words; oracle-of-arms per sample
        kept_hit, kept_tot, drop_hit, drop_tot = 0, 0, 0, 0
        orac, act = [], []
        for r in rows:
            gt_t = set(get_tokens(r.get('gt', r.get('ground_truth', ''))))
            for w in (r.get('votes') or {}):
                if w in gt_t:
                    kept_hit += 1
                kept_tot += 1
            for w in (r.get('dropped_words_1arm') or {}):
                if w in gt_t:
                    drop_hit += 1
                drop_tot += 1
            af = r.get('arm_f1') or {}
            f1s = [float(v) for v in af.values()]   # arm_f1 is already 0-100
            if f1s:
                orac.append(max(f1s))
                tf1 = (r.get('eval_metrics') or {}).get('Token-F1')
                act.append(float(tf1) if tf1 is not None
                           else 100 * token_f1(r['pred'], r['gt'])[0])
        res['vote_audit'] = dict(
            kept_words_gt_hit=kept_hit / max(kept_tot, 1), kept_words_n=kept_tot,
            dropped1arm_words_gt_hit=drop_hit / max(drop_tot, 1), dropped_words_n=drop_tot,
            oracle_arms_mean=float(np.mean(orac)) if orac else None,
            actual_mean=float(np.mean(act)) if act else None,
            gap=float(np.mean(orac) - np.mean(act)) if orac else None)
    return res


# ===========================================================================
# P3 -- method fingerprint matrix + sparsity
# ===========================================================================

MAYBE = -1


def parse_yesno(text):
    text = str(text).lower().strip()
    bm = re.search(r'\\boxed\{(.*?)\}', text)
    if bm:
        ans = bm.group(1).strip()
        if 'maybe' in ans:
            return MAYBE
        return 1 if 'yes' in ans else 0
    if 'final answer: maybe' in text:
        return MAYBE
    if 'final answer: yes' in text:
        return 1
    if 'final answer: no' in text:
        return 0
    fw = text.split('.')[0].split(',')[0].split()[0] if text.split() else ''
    if fw == 'maybe':
        return MAYBE
    if fw == 'yes':
        return 1
    if fw == 'no':
        return 0
    lc = text[-30:]
    if 'maybe' in lc:
        return MAYBE
    if 'yes' in lc:
        return 1
    if 'no' in lc:
        return 0
    return 0


def parse_gt(text):
    t = str(text).lower().strip()
    if t == 'yes' or t.startswith(('yes.', 'yes,', 'yes ')):
        return 1
    if t == 'no' or t.startswith(('no.', 'no,', 'no ')):
        return 0
    return parse_yesno(text)


def cell_score(ds, path):
    rows = jl(path)
    if ds in ('phd', 'cmm', 'pubmedqa'):
        yt, yp = [], []
        for r in rows:
            g = parse_gt(r.get('gt', r.get('answer', r.get('label'))))
            if g == MAYBE:
                continue
            p = parse_yesno(r.get('pred', r.get('response', r.get('output'))))
            yt.append(g)
            yp.append((1 - g) if p == MAYBE else p)
        return dict(score=round(f1_yes(yt, yp) * 100, 2), n=len(yt), metric='Yes-F1')
    f1s = [token_f1(r.get('pred', r.get('response', '')),
                    r.get('gt', r.get('ground_truth', '')))[0] * 100 for r in rows]
    return dict(score=round(float(np.mean(f1s)), 2), n=len(f1s), metric='Token-F1')


DATASETS = ['phd', 'cmm', 'pubmedqa', 'ragtruth', 'haloquest', 'avhbench']
METHODS = ['base_model', 'base_dola', 'base_cad', 'base_sc', 'base_vcd',
           'base_avcd', 'base_opera', 'base_pd', 'base_tot', 'base_gd']

# registered (RESULTS.md) cells for verification; None = newly computed here
REGISTERED = {
    ('phd', 'base_model'): 71.68, ('phd', 'base_dola'): 72.30, ('phd', 'base_cad'): 79.40,
    ('phd', 'base_sc'): 76.43, ('phd', 'base_vcd'): 72.05, ('phd', 'base_avcd'): 73.81,
    ('phd', 'base_pd'): 32.12, ('phd', 'base_tot'): 70.25, ('phd', 'base_gd'): 21.16,
    ('cmm', 'base_model'): 70.07, ('cmm', 'base_dola'): 70.16, ('cmm', 'base_cad'): 67.37,
    ('cmm', 'base_sc'): 71.96, ('cmm', 'base_vcd'): 69.90, ('cmm', 'base_avcd'): 69.08,
    ('cmm', 'base_opera'): 58.13, ('cmm', 'base_pd'): 58.09, ('cmm', 'base_tot'): 67.85,
    ('cmm', 'base_gd'): 57.54,
    ('pubmedqa', 'base_model'): 95.14, ('pubmedqa', 'base_dola'): 93.52,
    ('pubmedqa', 'base_cad'): 95.70, ('pubmedqa', 'base_sc'): 91.25,
    ('pubmedqa', 'base_opera'): 94.91, ('pubmedqa', 'base_pd'): 79.35,
    ('pubmedqa', 'base_tot'): 68.07, ('pubmedqa', 'base_gd'): 57.37,
    ('ragtruth', 'base_model'): 38.25, ('ragtruth', 'base_dola'): 38.74,
    ('ragtruth', 'base_cad'): 36.29, ('ragtruth', 'base_sc'): 35.38,
    ('ragtruth', 'base_pd'): 12.72, ('ragtruth', 'base_tot'): 12.08,
    ('ragtruth', 'base_gd'): 11.61,
    ('haloquest', 'base_model'): 34.22, ('haloquest', 'base_dola'): 33.62,
    ('haloquest', 'base_cad'): 36.06, ('haloquest', 'base_avcd'): 33.25,
    ('avhbench', 'base_model'): 19.42, ('avhbench', 'base_dola'): 19.15,
    ('avhbench', 'base_cad'): 18.01, ('avhbench', 'base_sc'): 18.29,
    ('avhbench', 'base_avcd'): 19.76,
}


def p3_matrix():
    mat = {}
    for ds in DATASETS:
        for m in METHODS:
            d = os.path.join(ROOT, 'outputs', f'{ds}_{m}')
            if not os.path.isdir(d):
                continue
            cand = sorted(glob.glob(os.path.join(d, f'{ds}_{m}_full.jsonl'))) or \
                sorted(glob.glob(os.path.join(d, '*full*.jsonl')))
            if not cand:
                continue
            try:
                c = cell_score(ds, cand[0])
                c['file'] = os.path.relpath(cand[0], ROOT)
                reg = REGISTERED.get((ds, m), 'new')
                c['registered'] = reg
                c['repro_ok'] = True if reg == 'new' else abs(reg - c['score']) <= 0.3
                mat[(ds, m)] = c
            except Exception as e:
                mat[(ds, m)] = dict(error=str(e))
    return mat


def spearman_perm(x, y, n_perm=20000, seed=0):
    if len(x) < 4:
        return float('nan'), float('nan')
    rho, _ = st.spearmanr(x, y)
    rng = random.Random(seed)
    yl = list(y); cnt = 0
    for _ in range(n_perm):
        rng.shuffle(yl)
        r2, _ = st.spearmanr(x, yl)
        if abs(r2) >= abs(rho) - 1e-12:
            cnt += 1
    return float(rho), (cnt + 1) / (n_perm + 1)


def p3_cluster(mat):
    vals = {ds: {m: mat[(ds, m)]['score'] for m in METHODS if (ds, m) in mat
                 and 'score' in mat[(ds, m)]} for ds in DATASETS}
    pairs = {}
    for i, a in enumerate(DATASETS):
        for b in DATASETS[i + 1:]:
            common = [m for m in METHODS if m in vals[a] and m in vals[b]]
            if len(common) < 4:
                pairs[f'{a}~{b}'] = dict(rho=None, p=None, n=len(common))
                continue
            rho, p = spearman_perm([vals[a][m] for m in common],
                                   [vals[b][m] for m in common])
            pairs[f'{a}~{b}'] = dict(rho=rho, p=p, n=len(common))
    D = {}
    rhos = [v['rho'] for v in pairs.values() if v['rho'] is not None]
    default_d = 1 - (min(rhos) if rhos else 0)
    for k, v in pairs.items():
        a, b = k.split('~')
        dd = max(1 - v['rho'], 0.0) if v['rho'] is not None else default_d
        D[(a, b)] = D[(b, a)] = dd
    clusters = [[ds] for ds in DATASETS]
    merges = []
    while len(clusters) > 1:
        best = None
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                ds_ = [D[(a, b)] for a in clusters[i] for b in clusters[j] if (a, b) in D]
                if not ds_:
                    continue
                dist = float(np.mean(ds_))
                if best is None or dist < best[0]:
                    best = (dist, i, j)
        if best is None:
            break
        dist, i, j = best
        merges.append(dict(distance=round(dist, 3), left=clusters[i][:], right=clusters[j][:]))
        clusters[i] = clusters[i] + clusters[j]
        del clusters[j]
    return pairs, merges, clusters[0]


def p3b_sparsity():
    """|Delta| sparsity per dataset probe (Gini + top-decile share)."""
    out = {}
    phd = jl(f'{ROOT}/outputs/cf_probe/phd_probe_full.jsonl')
    out['phd'] = dict(signal='|d_noctx|', n=len(phd),
                      gini=gini([-r['delta_noctx'] for r in phd]),
                      values=[-r['delta_noctx'] for r in phd])
    cmm = jl(f'{ROOT}/outputs/cf_probe/cmm_probe_full.jsonl')
    out['cmm'] = dict(signal='|d_cf_sum|', n=len(cmm),
                      gini=gini([r['d_cf_sum'] for r in cmm]),
                      values=[r['d_cf_sum'] for r in cmm])
    pm = jl(f'{ROOT}/outputs/cf_probe/pubmedqa_probe_full.jsonl')
    out['pubmedqa'] = dict(signal='|d_cf(passage)|', n=len(pm),
                           gini=gini([r['d_cf'] for r in pm]),
                           values=[r['d_cf'] for r in pm])
    q3 = jl(f'{ROOT}/outputs/cf_probe/cmm_probe_qwen3.jsonl')
    out['cmm_qwen3'] = dict(signal='|d_cf_sum|', n=len(q3),
                            gini=gini([r['d_cf_sum'] for r in q3]),
                            values=[r['d_cf_sum'] for r in q3])
    d = json.load(open(f'{ROOT}/outputs/bon_probe/haloquest_bon_feat.json'))
    gvals = [c['g_content'] for r in d['rows'] for c in r['candidates']
             if c['g_content'] is not None]
    out['haloquest(cand)'] = dict(signal='|g_content|', n=len(gvals),
                                  gini=gini(gvals), values=gvals)
    aw = jl(f'{ROOT}/outputs/cf_probe/avh_probe_report_words.jsonl')
    dv = [abs(w['dv']) for w in aw if w.get('dv') is not None]
    out['avhbench(word)'] = dict(signal='|delta_video| word-level', n=len(dv),
                                 gini=gini(dv), values=dv)
    for k, v in out.items():
        x = np.sort(np.abs(np.array(v.pop('values'), float)))
        dec = max(int(len(x) * 0.1), 1)
        v['top_decile_share'] = float(x[-dec:].sum() / x.sum()) if x.sum() > 0 else None
    return out


# ===========================================================================
# main
# ===========================================================================

def main():
    R = {}

    phd_pairs, miss = load_probe_traj(f'{ROOT}/outputs/cf_probe/phd_probe_full.jsonl',
                                      f'{ROOT}/outputs/phd_base_model/phd_base_model_full.jsonl')
    R['P1_phd_q25'] = p1_run(
        'phd_q25', phd_pairs, lam=2.0,
        delta_op_of=lambda r: -r['delta_noctx'],
        delta_variant='delta_signed_prereg',
        note='CF disease (context suppresses Yes); full-906 probe; repro targets '
             'registered full-set AUCs margin_full 0.739 / delta_noimg 0.753 / '
             'margin_noctx 0.839 / -delta_noctx 0.680')
    R['P1_phd_q25']['n_traj_missing'] = miss

    cmm_pairs, miss = load_probe_traj(f'{ROOT}/outputs/cf_probe/cmm_probe_full.jsonl',
                                      f'{ROOT}/outputs/cmm_base_model/cmm_base_model_full.jsonl',
                                      cmm_style=True)
    R['P1_cmm_q25'] = p1_run(
        'cmm_q25', cmm_pairs, lam=1.0,
        delta_op_of=lambda r: -r['d_cf_sum'],
        delta_variant='delta_signed_prereg',
        note='CF disease (unimodal over-reliance); full-720 probe; repro target '
             '-d_cf_sum -> gt 0.595')
    R['P1_cmm_q25']['n_traj_missing'] = miss

    q3_pairs, miss = load_probe_traj(f'{ROOT}/outputs/cf_probe/cmm_probe_qwen3.jsonl',
                                     '/ultrafast_share/tsq/qwen3omni_outputs/cmm_qwen3omni/cmm_qwen3omni_full.jsonl',
                                     cmm_style=True)
    R['P1_cmm_qwen3'] = p1_run(
        'cmm_qwen3', q3_pairs, lam=1.0,
        delta_op_of=lambda r: -r['d_cf_sum'],
        delta_variant='delta_signed_prereg',
        note='residual calibration disease; dev98 probe; repro target '
             'margin->gt within pred=yes 0.953; n small, CI wide')
    R['P1_cmm_qwen3']['n_traj_missing'] = miss

    R['P1d_halo_candidate_race'] = p1d_candidate_race()

    # ---------------- P2 ----------------
    R['P2_haloquest'] = [
        p2_halo(f'{ROOT}/outputs/haloquest_base_model/haloquest_base_model_full_single_scored.jsonl',
                'base_model 34.22'),
        p2_halo(f'{ROOT}/outputs/haloquest_bon_stacked/haloquest_bon_stacked_feat_combo_single_scored.jsonl',
                'final-4layer 40.14'),
    ]
    cv2 = glob.glob(f'{ROOT}/outputs/haloquest_contract_v2/*single_scored*.jsonl')
    if cv2:
        R['P2_haloquest'].insert(1, p2_halo(cv2[0], 'contract_v2 35.41'))
    R['P2_avhbench'] = [
        p2_avh(f'{ROOT}/outputs/avhbench_base_model/avhbench_base_model_full_single_scored.jsonl',
               'base_model 19.42'),
        p2_avh(f'{ROOT}/outputs/avhbench_base_avcd/avhbench_base_avcd_full_single_scored.jsonl',
               'base_avcd 19.76'),
        p2_avh(f'{ROOT}/outputs/avhbench_vote_sc/avhbench_vote_sc_full_single_scored.jsonl',
               'Vote-SC 5arm@2 21.56', vote_audit=True),
    ]

    # ---------------- P3 ----------------
    mat = p3_matrix()
    R['P3_matrix'] = {f'{ds}|{m}': v for (ds, m), v in mat.items()}
    pairs, merges, order = p3_cluster(mat)
    R['P3_spearman_pairs'] = pairs
    R['P3_dendrogram_merges'] = merges
    R['P3_cluster_order'] = order
    R['P3b_signal_sparsity'] = p3b_sparsity()

    with open(OUT_JSON, 'w') as f:
        json.dump(R, f, indent=1, default=str)
    print('written', OUT_JSON)


if __name__ == '__main__':
    main()
