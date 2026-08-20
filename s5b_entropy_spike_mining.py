# -*- coding: utf-8 -*-
"""s5b_entropy_spike_mining.py — S5b 前置一（零 GPU）：Qwen3-Omni 判断类轨迹熵尖峰挖掘.

PRE-REGISTERED criteria (written BEFORE running; 纪律：判据先写后算、负结果照登):
  - Per-token surprisal s_t = -logp_t. Spike := local max (s_t >= neighbors)
    AND s_t > k * median(s), k in {2,3,5} (primary k=3, others sensitivity).
  - Transition point := first token index whose decoded text contains the
    judgment marker ("Judgment" / "Final Answer" / "\\boxed"); obs->judgment
    转折 = that index.
  - PER-SAMPLE feature fed to AUC (error vs correct): spike_ratio =
    max(s)/ (median(s)+1e-9) (primary; spike_count and mean_top2 spike_ratio
    as sensitivity).
  - VERDICT (S5b 在线探针是否值得跑) = "alive" iff BOTH:
      (1) fraction of spike tokens within +-20 tokens of the transition > 0.40
          (computed over all spike tokens of all trajectories, pooled);
      (2) AUC(spike_ratio -> sample wrong) > 0.60 (pooled over the three
          judgement datasets; per-dataset reported alongside).
    Else verdict = "pre_dead" (CF->CoT 扩展无理论基础).
No GPU is used (tokenizer decode only).
"""
import json, os, sys
import numpy as np
from transformers import AutoTokenizer
from sklearn.metrics import roc_auc_score

TOK_PATH = '/ultrafast_share/tsq/MODELS/Qwen3-Omni-30B-A3B-Instruct'
FILES = {
    'phd': '/ultrafast_share/tsq/qwen3omni_outputs/phd_qwen3omni/phd_qwen3omni_full_single_scored.jsonl',
    'cmm': '/ultrafast_share/tsq/qwen3omni_outputs/cmm_qwen3omni/cmm_qwen3omni_full_single_scored.jsonl',
    'pubmedqa': '/ultrafast_share/tsq/qwen3omni_outputs/pubmedqa_qwen3omni/pubmedqa_qwen3omni_full_single_scored.jsonl',
}
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
K_GRID = [2, 3, 5]
PRIMARY_K = 3
WINDOW = 20  # +- tokens around transition
MARKERS = ('Judgment', 'judgment', 'Final Answer', 'final answer', '\\boxed', 'boxed')


def find_transition(tok_texts):
    for i, t in enumerate(tok_texts):
        for m in MARKERS:
            if m in t:
                return i
    return None


def spikes(s, k):
    med = np.median(s)
    thr = k * med
    out = []
    for i in range(1, len(s) - 1):
        if s[i] >= s[i - 1] and s[i] >= s[i + 1] and s[i] > thr:
            out.append(i)
    return out, med


def main():
    tok = AutoTokenizer.from_pretrained(TOK_PATH, trust_remote_code=True)
    per_record = []
    pooled = {k: {'spike_tokens': 0, 'near_trans': 0} for k in K_GRID}
    feats_by_ds = {k: {ds: ([], []) for ds in FILES} for k in K_GRID}  # (vals, wrong)
    no_marker = 0
    for ds, path in FILES.items():
        for line in open(path):
            r = json.loads(line)
            if not r.get('record_process') or not r.get('logp_list'):
                continue
            lp = np.asarray(r['logp_list'], float)
            tids = r['token_ids']
            if len(lp) < 8 or len(lp) != len(tids):
                continue
            s = -lp  # surprisal
            texts = [tok.decode([t]) for t in tids]
            tr = find_transition(texts)
            if tr is None:
                no_marker += 1
                continue
            wrong = 0 if r['eval_metrics'].get('Is_Correct', 1) else 1
            rec = {'dataset': ds, 'id': r['id'], 'n_tok': int(len(s)),
                   'transition_pos': tr, 'rel_trans': tr / len(s), 'wrong': wrong,
                   'surprisal_median': float(np.median(s)),
                   'surprisal_max': float(s.max())}
            for k in K_GRID:
                sp, med = spikes(s, k)
                ratio = (s.max() / (med + 1e-9)) if med > 0 else np.nan
                near = [i for i in sp if abs(i - tr) <= WINDOW]
                if sp:
                    pooled[k]['spike_tokens'] += len(sp)
                    pooled[k]['near_trans'] += len(near)
                feats_by_ds[k][ds][0].append(float(ratio))
                feats_by_ds[k][ds][1].append(wrong)
                rec[f'k{k}'] = {'n_spikes': len(sp),
                                'n_near_trans': len(near),
                                'spike_positions': sp[:50],
                                'spike_ratio': None if np.isnan(ratio) else round(float(ratio), 3)}
            per_record.append(rec)

    summary = {'preregistered_criteria': {
        'spike': 'local max AND s_t > k*median(surprisal), k primary=3',
        'transition': 'first token matching Judgment/Final Answer/boxed',
        'near_trans_window': WINDOW,
        'alive_if': 'near_trans_fraction > 0.40 AND pooled AUC(spike_ratio->wrong) > 0.60'},
        'n_records': len(per_record), 'records_without_marker': no_marker}
    pooled_vals, pooled_wrong = [], []
    for k in K_GRID:
        kk = {}
        for ds in FILES:
            v, w = feats_by_ds[k][ds]
            m = [not np.isnan(x) for x in v]
            v2 = [x for x, mm in zip(v, m) if mm]
            w2 = [x for x, mm in zip(w, m) if mm]
            try:
                auc = float(roc_auc_score(w2, v2)) if len(set(w2)) > 1 else float('nan')
            except ValueError:
                auc = float('nan')
            kk[ds] = {'n': len(v2), 'AUC_spike_ratio_vs_wrong': round(auc, 4),
                      'mean_ratio_wrong': round(float(np.mean([x for x, y in zip(v2, w2) if y])), 3) if any(w2) else None,
                      'mean_ratio_correct': round(float(np.mean([x for x, y in zip(v2, w2) if not y])), 3) if not all(w2) else None}
        p = pooled[k]
        kk['near_trans_fraction'] = round(p['near_trans'] / p['spike_tokens'], 4) if p['spike_tokens'] else None
        kk['n_spike_tokens'] = p['spike_tokens']
        summary[f'k={k}'] = kk
        pooled_vals += [x for x in feats_by_ds[k][FILES and 'phd'][0]]
    # pooled AUC at primary k
    for k in K_GRID:
        v_all, w_all = [], []
        for ds in FILES:
            v, w = feats_by_ds[k][ds]
            v_all += v; w_all += w
        m = [not np.isnan(x) for x in v_all]
        auc = float(roc_auc_score([y for y, mm in zip(w_all, m) if mm],
                                  [x for x, mm in zip(v_all, m) if mm]))
        summary[f'k={k}']['pooled_AUC'] = round(auc, 4)
        if k == PRIMARY_K:
            frac = summary[f'k={k}']['near_trans_fraction']
            verdict = 'alive' if (frac is not None and frac > 0.40 and auc > 0.60) else 'pre_dead'
            summary['verdict'] = verdict
            summary['verdict_numbers'] = {'near_trans_fraction': frac, 'pooled_AUC': round(auc, 4)}

    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump({'summary': summary, 'records': per_record},
              open(os.path.join(OUT_DIR, 's5b_entropy_spike_mining.json'), 'w'),
              ensure_ascii=False)
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
