# -*- coding: utf-8 -*-
"""analyze_textcf_robust.py -- P6: robustness of the text-channel CF probe
(delta^t) to the choice of question-neutralization scheme
(placeholder / shuffle / empty). Zero-GPU analysis.

Outputs the tables written to
outputs/problem_structure_fig/unified_cf_robustness_textcf.md.
"""
import json, os, sys
import numpy as np
from scipy.stats import spearmanr, pearsonr

HERE = os.path.dirname(os.path.abspath(__file__))
CF = os.path.join(HERE, 'outputs/cf_probe')

FILES = {
    'haloquest': {s: os.path.join(CF, f) for s, f in [
        ('placeholder', 'haloquest_textcf_tokens.jsonl'),
        ('shuffle', 'haloquest_textcf_shuffle.jsonl'),
        ('empty', 'haloquest_textcf_empty.jsonl')]},
    'phd': {s: os.path.join(CF, f) for s, f in [
        ('placeholder', 'phd_textcf_tokens.jsonl'),
        ('shuffle', 'phd_textcf_shuffle.jsonl'),
        ('empty', 'phd_textcf_empty.jsonl')]},
    'avh': {s: os.path.join(CF, f) for s, f in [
        ('placeholder', 'avh_probe_text_shard0of1.jsonl'),
        ('shuffle', 'avh_probe_text_shuffle.jsonl'),
        ('empty', 'avh_probe_text_empty.jsonl')]},
    'cmm': {s: os.path.join(CF, f) for s, f in [
        ('placeholder', 'cmm_textcf_tokens.jsonl'),
        ('shuffle', 'cmm_textcf_shuffle.jsonl'),
        ('empty', 'cmm_textcf_empty.jsonl')]},
}

KEY = {'haloquest': lambda r: r['id'],
       'phd': lambda r: r['question'],
       'avh': lambda r: (r['id'], r['gt'], r['pred']),
       'cmm': lambda r: (r['question'], r['sample_i'])}
DT = {'haloquest': 'delta_text_tok', 'phd': 'delta_text_tok',
      'avh': 'delta_text', 'cmm': 'delta_text_tok'}
# modality-channel fields per dataset: name -> field (None => join to bigN)
MODCH = {'haloquest': [('v', 'delta_img_tok')],
         'phd': [('v', 'delta_tok')],
         'cmm': [('v', 'delta_v'), ('a', 'delta_a')],
         'avh': [('v', 'delta_video'), ('a', 'delta_audio')]}
BIGn = os.path.join(CF, 'avh_probe_bigN_shard0of1.jsonl')


def load(ds, scheme):
    rows = {}
    for line in open(FILES[ds][scheme]):
        line = line.strip()
        if line:
            r = json.loads(line)
            rows[KEY[ds](r)] = r
    return rows


def pooled_spearman(a, b, keys):
    x = np.concatenate([a[k][DT[ds]] for k in keys])
    y = np.concatenate([b[k][DT[ds]] for k in keys])
    rho, p = spearmanr(x, y)
    return rho, p, len(x)


def mean_pearson_pairs(rows, keys, mfield, dtfield, bign=None):
    """mean over samples of pearson(modality delta, text delta) for this scheme."""
    rs = []
    for k in keys:
        r = rows[k]
        if bign is not None:
            base = bign.get(k)
            if base is None:
                continue
            m = np.asarray(base[mfield], float)
        else:
            m = np.asarray(r[mfield], float)
        t = np.asarray(r[dtfield], float)
        if len(m) != len(t) or len(m) < 3:
            continue
        if np.std(m) < 1e-9 or np.std(t) < 1e-9:
            continue
        rr = pearsonr(m, t)[0]
        if np.isfinite(rr):
            rs.append(rr)
    return (float(np.mean(rs)), len(rs)) if rs else (float('nan'), 0)


if __name__ == '__main__':
    ds = sys.argv[1]
    ph, sh, em = load(ds, 'placeholder'), load(ds, 'shuffle'), load(ds, 'empty')
    keys = [k for k in ph if k in sh and k in em]
    print(f'== {ds}: joined samples n={len(keys)} '
          f'(ph={len(ph)} sh={len(sh)} em={len(em)})')
    # repro sanity
    if 'repro_err' in ph[keys[0]]:
        for nm, d in (('ph', ph), ('sh', sh), ('em', em)):
            v = [d[k]['repro_err'] for k in keys if 'repro_err' in d[k]]
            if v:
                print(f'  repro_err max ({nm}): {max(v):.4f}')
    if 'verify_full_logp_maxabsdiff_vs_prev' in ph[keys[0]]:
        for nm, d in (('ph', ph), ('sh', sh), ('em', em)):
            v = [d[k]['verify_full_logp_maxabsdiff_vs_prev'] for k in keys
                 if d[k].get('verify_full_logp_maxabsdiff_vs_prev') is not None]
            if v:
                print(f'  verify_maxabs ({nm}): max={max(v):.4f} n={len(v)}')
    # (a) scheme-vs-scheme Spearman of delta^t (pooled tokens)
    rho_sh, p_sh, n = pooled_spearman(ph, sh, keys)
    rho_em, p_em, _ = pooled_spearman(ph, em, keys)
    rho_se, p_se, _ = pooled_spearman(sh, em, keys)
    print(f'  (a) Spearman d^t placeholder~shuffle: rho={rho_sh:.4f} (p={p_sh:.2e}, n_tok={n})')
    print(f'      Spearman d^t placeholder~empty:    rho={rho_em:.4f} (p={p_em:.2e})')
    print(f'      Spearman d^t shuffle~empty:        rho={rho_se:.4f} (p={p_se:.2e})')
    # magnitudes
    for nm, d in (('placeholder', ph), ('shuffle', sh), ('empty', em)):
        m = np.concatenate([d[k][DT[ds]] for k in keys])
        print(f'      |d^t| {nm}: mean={m.mean():.4f} median={np.median(m):.4f} '
              f'p90={np.percentile(m,90):.4f}')
    # (b) channel-pair pearson per scheme
    bign = None
    if ds == 'avh':
        bign = {}
        for line in open(BIGn):
            r = json.loads(line)
            bign[(r['id'], r['gt'], r['pred'])] = r
    for ch, f in MODCH[ds]:
        line = [f'  (b) pearson({ch},t)']
        for nm, d in (('placeholder', ph), ('shuffle', sh), ('empty', em)):
            r_, n_ = mean_pearson_pairs(d, keys, f, DT[ds], bign)
            line.append(f'{nm}={r_:.3f}(n={n_})')
        print(' '.join(line))
