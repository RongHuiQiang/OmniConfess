#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A1.3 evaluation: the misplacement-intervention arms.

Scores the online misplaced runs with the repo judgement scorer, reports
fire rate / F1 / fired-subset F1 / parse outcomes, paired McNemar vs base
and vs the shipped CF arm. Criterion (pre-registered): F1 < 80 (vs 84.60).
"""
import os
import sys
import json
import numpy as np
from sklearn.metrics import f1_score, recall_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from util import load_jsonl, judgement_preds, pair_by_question, mcnemar_exact

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.getcwd(), 'OmniConfess', 'eval'))
from single_eval import UnifiedEvaluator  # noqa: E402

OUT = os.path.join(HERE, 'a1_misplace.json')


def score(path):
    ev = UnifiedEvaluator()
    ev.evaluate_judgment_tasks(path, save_single=True)
    return load_jsonl(path.replace('.jsonl', '_single_scored.jsonl'))


def metr(prs):
    yt = [g for g, p in prs]
    yp = [p for g, p in prs]
    if not yt:
        return dict(F1=None, Youden=None, n=0)
    f1 = f1_score(yt, yp, zero_division=0) * 100
    tpr = recall_score(yt, yp, pos_label=1, zero_division=0)
    tnr = recall_score(yt, yp, pos_label=0, zero_division=0)
    return dict(F1=round(f1, 2), Youden=round(tpr + tnr - 1, 3), n=len(yt))


def mcn(preds_a, preds_b, gts):
    b = sum(1 for g, x, y in zip(gts, preds_a, preds_b) if x != g and y == g)
    c = sum(1 for g, x, y in zip(gts, preds_a, preds_b) if x == g and y != g)
    return dict(arm_fixes=b, arm_breaks=c, p=mcnemar_exact(b, c))


def main():
    res = {}
    base_p = 'outputs/phd_base_model/phd_base_model_full_single_scored.jsonl'
    cf_p = 'outputs/phd_cf_confess/cf_lam2.0_full_single_scored.jsonl'
    base = load_jsonl(base_p)
    cf = load_jsonl(cf_p)

    for tag, path in [('misplace+5', 'outputs/phd_cf_ablation/cf_misplace5_lam2.0.jsonl'),
                      ('misplace+1', 'outputs/phd_cf_ablation/cf_misplace1_lam2.0.jsonl')]:
        if not os.path.exists(path):
            continue
        rows = score(path)
        cfstat = [r.get('cf') or {} for r in rows]
        fired = [r for r, c in zip(rows, cfstat) if c.get('fired')]
        prs_all = judgement_preds(rows)
        prs_fired = judgement_preds(fired)
        # join with base by (question, gt) occurrence
        bm, cm = {}, {}
        for r in base:
            bm.setdefault((r['question'], str(r['gt'])), []).append(r)
        for r in cf:
            cm.setdefault((r['question'], str(r['gt'])), []).append(r)
        gb, pb, gc, pa, pa_fired = [], [], [], [], []
        fi = 0
        for r, c in zip(rows, cfstat):
            k = (r['question'], str(r['gt']))
            if k not in bm or not bm[k]:
                continue
            b = bm[k].pop(0)
            cc = cm[k].pop(0) if cm.get(k) else None
            em_b = b['eval_metrics']
            em_a = r['eval_metrics']
            if em_b['Ground_Truth'] == 2:
                continue
            gb.append(em_b['Ground_Truth'])
            pb.append((1 - em_b['Ground_Truth']) if em_b['Prediction'] == 2 else em_b['Prediction'])
            pa.append((1 - em_a['Ground_Truth']) if em_a['Prediction'] == 2 else em_a['Prediction'])
            if cc is not None:
                em_c = cc['eval_metrics']
                gc.append(em_c['Ground_Truth'])
                pa_fired.append((1 - em_c['Ground_Truth']) if em_c['Prediction'] == 2 else em_c['Prediction'])
        m_all = metr(prs_all)
        res[tag] = dict(
            n=len(rows),
            fire_rate=round(len(fired) / len(rows) * 100, 1),
            intervened_at_fire=round(sum(1 for c in cfstat if c.get('intervened')) / max(1, len(fired)) * 100, 1),
            F1=m_all['F1'], Youden=m_all['Youden'], n_scored=m_all['n'],
            fired_subset=metr(prs_fired),
            vs_base=mcn(pb, pa, gb),
            criterion='F1 < 80 (vs CF 84.60)')
        if len(pa_fired) == len(gc):
            res[tag]['vs_CF'] = mcn(pa_fired, pa, gc)

    with open(OUT, 'w') as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
