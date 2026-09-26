#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Round7 freeform repair offline diagnostics. Pure stdlib, no GPU."""
import json, os, glob, statistics
import sys
sys.path.insert(0, 'scripts')
from tokenf1_fullset import token_f1_multiref

BASE = 'outputs/freeform_grid/round7'
OUT = os.path.join(BASE, 'diagnostics')
os.makedirs(OUT, exist_ok=True)

ARMS = ['COV20','COV40','COV60','MAX1-COV60','MAX1-TAU10','TAU10','TAU20','TOPK']

def f1(pred, gt):
    return round(token_f1_multiref(pred, gt, multiref=True) * 100, 2)

def load(tag):
    path = os.path.join(BASE, f'haloquest_R7-{tag}.jsonl')
    return [json.loads(l) for l in open(path) if l.strip()]

perarm_summary = {}
oracle_summary = {}
merged_edited = []  # collect all edited rows across arms for merged table

for tag in ARMS:
    rows = load(tag)
    detail_path = os.path.join(OUT, f'perarm_{tag}.jsonl')
    n_nonsuccess = 0
    edited_rows = []
    all_base=[]; all_rep=[]; all_oracle=[]
    with open(detail_path,'w') as fo:
        for d in rows:
            base = d.get('base_pred','')
            rep = d.get('repaired_pred','')
            gt = d.get('gt','')
            bf = f1(base, gt)
            rf = f1(rep, gt)
            status = d.get('status')
            if status != 'success':
                n_nonsuccess += 1
            edited = (d.get('n_spans',0) > 0) and (not d.get('coverage_cap_skip', False))
            delta = round(rf - bf, 2)
            rec = {
                'id': d.get('id'),
                'question': d.get('question'),
                'gt': gt,
                'base_pred': base,
                'spans': d.get('spans'),
                'replacements': d.get('replacements'),
                'repaired_pred': rep,
                'base_f1_recompute': bf,
                'repaired_f1_recompute': rf,
                'edited': edited,
                'delta': delta,
                'n_spans': d.get('n_spans'),
                'coverage_cap_skip': d.get('coverage_cap_skip'),
                'span_token_coverage': d.get('span_token_coverage'),
                'tau': d.get('tau'),
                'status': status,
            }
            fo.write(json.dumps(rec, ensure_ascii=False)+'\n')
            all_base.append(bf); all_rep.append(rf); all_oracle.append(max(bf,rf))
            if edited:
                edited_rows.append(rec)
                merged_edited.append(rec)

    # part2: edited-only stats
    n_ed = len(edited_rows)
    deltas = [r['delta'] for r in edited_rows]
    n_imp = sum(1 for x in deltas if x>0)
    n_deg = sum(1 for x in deltas if x<0)
    n_unc = sum(1 for x in deltas if x==0)
    mean_d = round(statistics.mean(deltas),2) if deltas else 0.0
    med_d = round(statistics.median(deltas),2) if deltas else 0.0
    imp_gain = round(statistics.mean([x for x in deltas if x>0]),2) if n_imp else 0.0
    deg_loss = round(statistics.mean([x for x in deltas if x<0]),2) if n_deg else 0.0
    perarm_summary[tag] = dict(n_edited=n_ed, n_improved=n_imp, n_degraded=n_deg,
        n_unchanged=n_unc, mean_delta_edited=mean_d, median_delta_edited=med_d,
        avg_improve_gain=imp_gain, avg_degrade_loss=deg_loss, n_nonsuccess=n_nonsuccess)

    # part3: oracle over full set
    base_avg = round(statistics.mean(all_base),2)
    rep_avg = round(statistics.mean(all_rep),2)
    oracle_avg = round(statistics.mean(all_oracle),2)
    # on edited subset: how many base is actually better (rep<base)
    n_rep_worse = sum(1 for r in edited_rows if r['delta']<0)
    frac_harmful = round(100*n_rep_worse/n_ed,1) if n_ed else 0.0
    oracle_summary[tag] = dict(base_avg=base_avg, repaired_avg=rep_avg, oracle_avg=oracle_avg,
        delta_oracle=round(oracle_avg-base_avg,2), delta_repaired=round(rep_avg-base_avg,2),
        n_edited=n_ed, n_harmful_edits=n_rep_worse, frac_harmful_edits=frac_harmful)

# merged edited table
mdeltas = [r['delta'] for r in merged_edited]
merged = dict(n_edited=len(mdeltas),
    n_improved=sum(1 for x in mdeltas if x>0),
    n_degraded=sum(1 for x in mdeltas if x<0),
    n_unchanged=sum(1 for x in mdeltas if x==0),
    mean_delta_edited=round(statistics.mean(mdeltas),2) if mdeltas else 0.0,
    median_delta_edited=round(statistics.median(mdeltas),2) if mdeltas else 0.0,
    avg_improve_gain=round(statistics.mean([x for x in mdeltas if x>0]),2) if any(x>0 for x in mdeltas) else 0.0,
    avg_degrade_loss=round(statistics.mean([x for x in mdeltas if x<0]),2) if any(x<0 for x in mdeltas) else 0.0)

json.dump(dict(perarm=perarm_summary, merged=merged, oracle=oracle_summary),
          open(os.path.join(OUT,'_summary.json'),'w'), ensure_ascii=False, indent=2)
print("DONE")
print(json.dumps(dict(perarm=perarm_summary, merged=merged, oracle=oracle_summary), ensure_ascii=False, indent=2))
