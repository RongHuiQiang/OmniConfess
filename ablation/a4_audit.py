#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A4 (ABLATION_PLAN.md): conservatism audit -- final version.

Per mechanism, before/after:
  - judgement-yes-rate displacement toward the ground-truth yes rate
  - refusal / avoidance rate (counting rules below)
  - answer word count / gt coverage
  - correct-answer rewrite rate
Counting rules (fixed before running, reproducing the repo's own counters):
  * refusal (ragtruth/judgement): pred STARTS with a refusal phrase
    (reproduces the registered 30->10/4/2 ragtruth counts)
  * avoidance (haloquest): repo's answerability_contract_gate.is_avoid_text
    = V2_AVOID | EVADERE (reproduces the registered 190/326/284/334 counts)
  * vote-sc tokenization: verbatim STOP/cwords from avhbench_vote_sc_rebuild
"""
import os
import sys
import json
import re
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from util import load_jsonl, judgement_preds, pair_by_question, paired_perm_test

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.getcwd())
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'a4_audit.json')

REFUSE_START = re.compile(
    r"^\s*(unable to answer|unable to|i cannot|i can't|sorry|"
    r"no (answer|information|passage)|i'm unable)", re.I)

from probe_claim_cf import EVADERE  # noqa: E402,E403  (repo's avoidance regex)
V2_AVOID = re.compile(r'there\s+is\s+no\b|not\s+visible|cannot|can\s*not', re.I)
STOP = set('''a an the and or of to in on at as is are was were be been being it its this that these those with while by for from into near nearby
there here his her their our your my he she they we you i who which what when where how not no nor but so then than also very just only
one two single describe see hear seen heard video audio image picture clip scene shows show can cannot could would should may might'''.split())
PUNCT = "!\"#$%&'()*+,-./:;<=>?@[]^_`{|}~\\"


def cwords(s):
    m = re.search(r'\\boxed\{(.*)\}', str(s), re.S)
    s = (m.group(1) if m else str(s)).strip()
    s = s.lower().translate(str.maketrans('', '', PUNCT))
    return [w for w in s.split() if w not in STOP and len(w) > 1]


def wc(s):
    return len(str(s).split())


def is_avoid(t):
    return bool(V2_AVOID.search(t or '') or EVADERE.search(t or ''))


def refusal_stats(rows, start_re=REFUSE_START):
    n = sum(1 for r in rows if start_re.match(str(r.get('pred') or '')))
    return n, n / len(rows) * 100 if rows else 0.0


def avoid_stats(rows):
    n = sum(1 for r in rows if is_avoid(str(r.get('pred') or '')))
    return n, n / len(rows) * 100 if rows else 0.0


def load_pair(base_path, arm_path):
    base, arm = load_jsonl(base_path), load_jsonl(arm_path)
    if [r['question'] for r in base] != [r['question'] for r in arm]:
        base, arm = pair_by_question(base_path, arm_path)
    return base, arm


def judgement_audit(label, base_path, arm_path):
    base, arm = load_pair(base_path, arm_path)
    _b = judgement_preds(base); gb = [x[0] for x in _b]; pb = [x[1] for x in _b]
    _a = judgement_preds(arm); ga = [x[0] for x in _a]; pa = [x[1] for x in _a]
    from sklearn.metrics import recall_score
    gt_yes = float(np.mean(gb)) * 100
    yes_b, yes_a = float(np.mean(pb)) * 100, float(np.mean(pa)) * 100
    rno_b = recall_score(gb, pb, pos_label=0, zero_division=0) * 100
    rno_a = recall_score(ga, pa, pos_label=0, zero_division=0) * 100
    rewrite = sum(1 for g, x, y in zip(gb, pb, pa) if x == g and y != g)
    ncorrect = sum(1 for g, x in zip(gb, pb) if x == g)
    refb, refpb = refusal_stats(base)
    refa, refpa = refusal_stats(arm)
    wlb = float(np.mean([wc(r.get('pred')) for r in base]))
    wla = float(np.mean([wc(r.get('pred')) for r in arm]))
    return dict(label=label, n=len(gb),
                gt_yes_rate=round(gt_yes, 1),
                yes_rate_base=round(yes_b, 1), yes_rate_arm=round(yes_a, 1),
                yes_dist_to_gt_base=round(abs(yes_b - gt_yes), 1),
                yes_dist_to_gt_arm=round(abs(yes_a - gt_yes), 1),
                recall_no_base=round(rno_b, 1), recall_no_arm=round(rno_a, 1),
                refusal_base=refb, refusal_arm=refa,
                refusal_delta_pp=round(refpa - refpb, 2),
                words_base=round(wlb, 1), words_arm=round(wla, 1),
                maybe_pred_base=sum(1 for r in base if r['eval_metrics']['Prediction'] == 2),
                maybe_pred_arm=sum(1 for r in arm if r['eval_metrics']['Prediction'] == 2),
                n_base_correct=ncorrect, correct_rewritten=rewrite,
                correct_rewrite_rate=round(rewrite / ncorrect * 100, 2) if ncorrect else None)


def descriptive_audit(label, base_path, arm_path, avoid=False):
    base, arm = load_pair(base_path, arm_path)
    q = [r['question'] for r in base]
    f1b = {i: r['eval_metrics']['Token-F1'] for i, r in enumerate(base)}
    f1a = {i: r['eval_metrics']['Token-F1'] for i, r in enumerate(arm)}
    d = [f1a[i] - f1b[i] for i in range(len(q))]
    if avoid:
        refb, refpb = avoid_stats(base)
        refa, refpa = avoid_stats(arm)
    else:
        refb, refpb = refusal_stats(base)
        refa, refpa = refusal_stats(arm)
    wlb = float(np.mean([wc(r.get('pred')) for r in base]))
    wla = float(np.mean([wc(r.get('pred')) for r in arm]))

    def cover(rows):
        cs = []
        for r in rows:
            ps = set(cwords(r.get('pred')))
            gs = [w for w in str(r.get('gt')).lower().split() if len(w) > 3]
            if gs:
                cs.append(np.mean([w in ps for w in gs]))
        return float(np.mean(cs)) * 100
    hi = [i for i in range(len(q)) if f1b[i] >= 50]
    dropped = [i for i in hi if f1a[i] < 50]
    obs, p = paired_perm_test(d)
    return dict(label=label, n=len(q),
                tf1_base=round(float(np.mean(list(f1b.values()))), 2),
                tf1_arm=round(float(np.mean(list(f1a.values()))), 2),
                delta=round(obs, 2), p_perm=p,
                refusal_base=refb, refusal_arm=refa,
                refusal_delta_pp=round(refpa - refpb, 2),
                words_base=round(wlb, 1), words_arm=round(wla, 1),
                gt_coverage_base=round(cover(base), 1), gt_coverage_arm=round(cover(arm), 1),
                n_base_good=len(hi), good_dropped=len(dropped),
                correct_rewrite_rate=round(len(dropped) / len(hi) * 100, 2) if hi else None)


def vote_sc_audit():
    rows = load_jsonl('outputs/avhbench_vote_sc/avhbench_vote_sc_full.jsonl')
    base = load_jsonl('outputs/avhbench_base_model/avhbench_base_model_full_single_scored.jsonl')
    bmap = {}
    for r in base:
        bmap.setdefault((r['question'], str(r['gt'])), []).append(r)
    wl_v, wl_b = [], []
    dropped_n, dropped_hit = 0, 0
    kept_n, kept_hit = 0, 0
    cov_b, cov_v = [], []
    for r in rows:
        grp = bmap.get((r['question'], str(r['gt'])), [])
        if not grp:
            continue
        b = grp.pop(0)
        gt_set = set(str(r['gt']).lower().split())
        vw, bw = set(cwords(r.get('pred'))), set(cwords(b.get('pred')))
        wl_v.append(len(str(r['pred']).split()))
        wl_b.append(len(str(b['pred']).split()))
        for w in bw - vw:
            dropped_n += 1
            dropped_hit += int(w in gt_set)
        for w in vw - bw:
            kept_n += 1
            kept_hit += int(w in gt_set)
        gs = [w for w in str(r['gt']).lower().split() if len(w) > 3]
        cov_b.append(np.mean([w in set(str(b['pred']).lower().split()) for w in gs]))
        cov_v.append(np.mean([w in set(str(r['pred']).lower().split()) for w in gs]))
    return dict(label='vote-sc avhbench', n=len(wl_v),
                words_base=round(float(np.mean(wl_b)), 1),
                words_vote=round(float(np.mean(wl_v)), 1),
                dropped_content_words=dropped_n,
                dropped_gt_hit_rate=round(dropped_hit / dropped_n * 100, 1) if dropped_n else None,
                added_content_words=kept_n,
                added_gt_hit_rate=round(kept_hit / kept_n * 100, 1) if kept_n else None,
                gt_coverage_base=round(float(np.mean(cov_b)) * 100, 1),
                gt_coverage_vote=round(float(np.mean(cov_v)) * 100, 1))


def main():
    res = {}
    # ---------------- CF ----------------
    res['cf_phd'] = judgement_audit(
        'CF lam2.0 @ phd',
        'outputs/phd_base_model/phd_base_model_full_single_scored.jsonl',
        'outputs/phd_cf_confess/cf_lam2.0_full_single_scored.jsonl')
    res['cf_cmm'] = judgement_audit(
        'CF lam1.0 @ cmm',
        'outputs/cmm_base_model/cmm_base_model_full_single_scored.jsonl',
        'outputs/cmm_cf_confess/cf_lam1.0_full_single_scored.jsonl')
    res['cf_phd_qwen3'] = judgement_audit(
        'CF lam4.0 @ phd Qwen3',
        '/ultrafast_share/tsq/qwen3omni_outputs/phd_qwen3omni/phd_qwen3omni_full_single_scored.jsonl',
        'outputs/phd_cf_confess_qwen3/cf_qwen3_lam4.0_full_single_scored.jsonl')
    res['cf_pubmedqa'] = judgement_audit(
        'CF lam0.25 @ pubmedqa',
        'outputs/pubmedqa_base_model/pubmedqa_base_model_full_single_scored.jsonl',
        'outputs/pubmedqa_cf_confess/cf_lam0.25_full_single_scored.jsonl')

    # ---------------- gate (haloquest) ----------------
    gate = json.load(open('outputs/answerability/answerability_gate_result.json'))
    gb = load_jsonl('outputs/haloquest_base_model/haloquest_base_model_full.jsonl')
    ga = load_jsonl('outputs/answerability/haloquest_answerability_gate_full.jsonl')
    av_b, avp_b = avoid_stats(gb)
    av_a, avp_a = avoid_stats(ga)
    cg = json.load(open('outputs/answerability_contract/contract_gate_result.json'))
    cga = load_jsonl('outputs/answerability_contract/haloquest_answerability_contract_gate_full.jsonl')
    av_c, avp_c = avoid_stats(cga)
    res['gate_haloquest'] = dict(
        label='answerability gate @ haloquest (avoid = repo is_avoid_text)',
        n=365,
        avoidance_base=av_b, avoidance_gate=av_a, avoidance_combo=av_c,
        avoidance_delta_pp=round(avp_a - avp_b, 2),
        words_base=round(float(np.mean([wc(r.get('pred')) for r in gb])), 1),
        words_gate=round(float(np.mean([wc(r.get('pred')) for r in ga])), 1),
        answerable_subset_f1_base=gate['main']['pos_gt(answerable)']['base'],
        answerable_subset_f1_arm=gate['main']['pos_gt(answerable)']['arm'],
        answerable_delta=gate['main']['pos_gt(answerable)']['delta_pp'],
        answerable_p=gate['main']['pos_gt(answerable)']['wilcoxon_p'],
        needs_gate_delta=gate['main']['needs_gate(unanswerable affirmed)']['delta_pp'],
        plan_criterion='answerable subset F1 must not drop (no suppression of correct answers)')

    # ---------------- contracts ----------------
    res['contract_haloquest_v2'] = descriptive_audit(
        'contract v2 @ haloquest',
        'outputs/haloquest_base_model/haloquest_base_model_full_single_scored.jsonl',
        'outputs/haloquest_contract_v2/haloquest_contract_v2_full_single_scored.jsonl',
        avoid=True)
    for v in ['v2', 'v3', 'v4']:
        res[f'contract_ragtruth_{v}'] = descriptive_audit(
            f'contract {v} @ ragtruth',
            'outputs/ragtruth_base_model/ragtruth_base_model_full_single_scored.jsonl',
            f'outputs/ragtruth_contract_{v}/ragtruth_contract_{v}_full_single_scored.jsonl')

    # ---------------- Vote-SC ----------------
    res['vote_sc_avhbench'] = vote_sc_audit()

    with open(OUT, 'w') as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=str)
    print(json.dumps(res, ensure_ascii=False, indent=1, default=str))


if __name__ == '__main__':
    main()
