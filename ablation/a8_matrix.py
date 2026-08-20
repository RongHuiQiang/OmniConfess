#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A8 (ABLATION_PLAN.md): pathology-mechanism matching matrix.

Theory under test: hallucination is a heterogeneous disease collection; a
mechanism only works where the disease it treats is present AND the mechanism
is matched to it. Formalized as a cell table:

  rows  = dataset (x backbone where relevant)
  pathology measurements (from disk, cross-referenced to A3/A4 results):
    - dominance AUC  (CF counterfactual-signal AUC; judgement-type disease)
    - output-engineering loss share (pred != raw extraction loss; brevity
      share for haloquest)
    - answerability disease share (needs-gate fraction)
    - perception/coverage headroom (avhbench: window-oracle headroom;
      Look-Again ladder delta ~ 0)
  columns = mechanism gains with paired p

Test: aligned cells (mechanism matches the measured dominant pathology)
vs misaligned cells -> Fisher exact on "significant positive gain (p<0.05)".
Negative results entered as-is. Zero GPU.
"""
import os
import sys
import json
from scipy.stats import fisher_exact

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from util import load_jsonl

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'a8_matrix.json')

# ---------------- pathology measurements ----------------
PATHO = {
    'phd@Q2.5': dict(
        dominance_auc_dev=0.767, dominance_auc_full=0.680,
        extraction_loss='n/a (judgement)', answerability_share=0.0,
        perception='image informative (probe noimg AUC 0.752 probe-conv)'),
    'cmm@Q2.5': dict(
        dominance_auc_full=0.595, margin_full_auc=0.888,
        extraction_loss='n/a (judgement)', answerability_share=0.0,
        perception='per-modality dependence (delta_video >> delta_audio)'),
    'pubmedqa@Q2.5': dict(
        dominance_auc_full=0.585,
        extraction_loss=0.0, answerability_share=0.0, perception='n/a'),
    'phd@Qwen3': dict(
        dominance_auc_dev=0.557,
        pathology_gone='anti-yes bias 50.2%->62.6%, correct-sample delta ~ 0',
        extraction_loss='n/a', answerability_share=0.0, perception='n/a'),
    'ragtruth@Q2.5': dict(
        dominance_auc='claim-level cross-sample AUC ~ 0.52 (judged dead)',
        extraction_loss=0.308, brevity_share=0.0, answerability_share=0.0,
        perception='n/a'),
    'haloquest@Q2.5': dict(
        dominance_auc='answerability gate AUC 0.715 [0.616,0.801]',
        extraction_loss=0.0, brevity_share=0.255,
        answerability_share=83 / 365),
    'avhbench@Q2.5': dict(
        dominance_auc='gate1 variance exists, gate2 dead (Spearman(g,F1)=0.02)',
        extraction_loss=0.0, answervability_share=0.0,
        answerability_share=0.0,
        perception='window-oracle headroom 93/130 = 71.5%; '
                   'Look-Again ladder delta +0.34 p=0.29 (input not bottleneck)'),
}

# ---------------- mechanism cells ----------------
# gain/p values recomputed in A3/A4/A6 jsons or read from the registered
# experiment sections (source noted per cell).
CELLS = [
    # (dataset, mechanism, aligned?, gain, p, note)
    ('phd@Q2.5', 'CF (context-dominance disease)', True, 12.92, 2.3e-17,
     'aligned: misleading-context pull is the measured disease'),
    ('cmm@Q2.5', 'CF (modality-prior disease)', True, 6.28, 1.7e-12,
     'aligned: single-modality over-affirmation is the disease'),
    ('ragtruth@Q2.5', 'output contract (extraction disease)', True, 7.62, 4e-4,
     'aligned: 30.8% of answers lost to extraction; v3 +11.15, v4 +13.33'),
    ('haloquest@Q2.5', 'answerability gate (answerability disease)', True, 3.43, 2.4e-08,
     'aligned: 22.7% needs-gate; subset gain +12.5'),
    ('haloquest@Q2.5', 'output contract (brevity disease)', True, 1.19, 0.164,
     'aligned but weak: brevity share 25.5%, registered "过线但弱"'),
    ('avhbench@Q2.5', 'Vote-SC (multi-arm diversity)', True, 2.14, 4.1e-11,
     'aligned: input-perturbation diversity exists on avhbench'),
    ('ragtruth@Q2.5', 'BoN grounding on v1 pool (variance axis)', True, 3.86, 0.001,
     'aligned: candidate variance = quality axis on the uncontracted pool'),
    # ---- misaligned ----
    ('pubmedqa@Q2.5', 'CF (no dominance signal, AUC 0.585)', False, 0.12, 1.0,
     'misaligned: no prior-dominance disease; pre-registered boundary case'),
    ('phd@Qwen3', 'CF (backbone removed the disease, AUC 0.557)', False, 1.29, 0.358,
     'misaligned: net-of-random-control +0.02 p=0.77'),
    ('ragtruth@Q2.5', 'CF claim-level (AUC ~0.52)', False, None, None,
     'judged dead at probe stage (negative result registered)'),
    ('avhbench@Q2.5', 'Look-Again (perception-input disease absent)', False, 0.34, 0.29,
     'misaligned: pathology is not perception input'),
    ('avhbench@Q2.5', 'window BoN (selection)', False, 0.35, 0.415,
     'misaligned: grounding orthogonal to window quality'),
    ('avhbench@Q2.5', 'window aggregation (coverage)', False, -0.55, None,
     'dead by pre-registered gate (<20.5)'),
    ('ragtruth@Q2.5', 'BoN on v3-contracted pool', False, -0.85, 0.11,
     'misaligned: contract removed the variance axis the ranker needs'),
    ('haloquest@Q2.5', 'BoN grounding overall', False, 0.28, 0.22,
     'CI crosses zero; only the release branch (81/365) is aligned'),
    ('cmm@Q2.5', 'CAD (view misaligned: drops all media)', False, 0.59, 0.41,
     'same-model contrastive family, misaligned view -> ~0'),
    ('phd@Q2.5', 'CF with noimg view (view misaligned)', False, None, None,
     'A1 quantifies: probe AUC 0.752 in probe convention = anti-informative '
     'in operator convention'),
]


def main():
    res = dict(pathology=PATHO, cells=[
        dict(dataset=c[0], mechanism=c[1], aligned=c[2], gain=c[3], p=c[4], note=c[5])
        for c in CELLS])
    # Fisher exact: significant positive gain (p<0.05, gain>0) by alignment
    a_pos = sum(1 for c in CELLS if c[2] and c[3] is not None and c[3] > 0 and c[4] is not None and c[4] < 0.05)
    a_neg = sum(1 for c in CELLS if c[2]) - a_pos
    m_pos = sum(1 for c in CELLS if not c[2] and c[3] is not None and c[3] > 0 and c[4] is not None and c[4] < 0.05)
    m_neg = sum(1 for c in CELLS if not c[2]) - m_pos
    tab = [[a_pos, a_neg], [m_pos, m_neg]]
    odds, p = fisher_exact(tab, alternative='greater')
    res['fisher'] = dict(table=tab,
                         aligned_significant=a_pos, aligned_total=a_pos + a_neg,
                         misaligned_significant=m_pos, misaligned_total=m_pos + m_neg,
                         odds_ratio=None if odds == float('inf') else round(float(odds), 2),
                         p_one_sided=float(p))
    with open(OUT, 'w') as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=str)
    print(json.dumps(res['fisher'], indent=1))
    print(f"aligned cells: {[(c[0], c[1], c[3], c[4]) for c in CELLS if c[2]]}")
    print(f"misaligned significant-positive: {[(c[0], c[1]) for c in CELLS if not c[2] and c[3] and c[3] > 0 and c[4] and c[4] < 0.05]}")


if __name__ == '__main__':
    main()
