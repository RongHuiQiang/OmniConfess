#!/usr/bin/env python3
"""
Leg1 modality-separation reinforcement + cross-6-dataset universal directional signals.
Pure read-only. No GPU. python3.
Outputs: tables (md+json) + report to outputs/problem_structure_fig/.
"""
import json, math, os, statistics as st
from collections import Counter, defaultdict

ROOT = "/home/tsq/rhq/omniconfess/OmniConfess"
OUT = os.path.join(ROOT, "outputs", "problem_structure_fig")
os.makedirs(OUT, exist_ok=True)

# ---------- small math helpers (numpy-free to stay dependency-safe) ----------
def mean(x): return sum(x)/len(x) if x else float('nan')
def var(x):
    if len(x)<2: return float('nan')
    m=mean(x); return sum((v-m)**2 for v in x)/(len(x)-1)
def std(x): return math.sqrt(var(x)) if len(x)>1 else float('nan')

def pearson(x,y):
    n=len(x)
    if n<3: return float('nan')
    mx,my=mean(x),mean(y)
    sxy=sum((a-mx)*(b-my) for a,b in zip(x,y))
    sxx=sum((a-mx)**2 for a in x); syy=sum((b-my)**2 for b in y)
    if sxx<=0 or syy<=0: return float('nan')
    return sxy/math.sqrt(sxx*syy)

def cov_matrix(x,y):
    """2x2 covariance for centered [x,y]."""
    mx,my=mean(x),mean(y); n=len(x)
    cxx=sum((a-mx)**2 for a in x)/(n-1)
    cyy=sum((b-my)**2 for b in y)/(n-1)
    cxy=sum((a-mx)*(b-my) for a,b in zip(x,y))/(n-1)
    return cxx,cxy,cyy

def eig2x2(a,b,c):
    """eigenvalues/vectors of [[a,b],[b,c]] symmetric."""
    tr=a+c; det=a*c-b*b
    disc=math.sqrt(max(tr*tr/4-det,0))
    l1=tr/2+disc; l2=tr/2-disc
    # eigenvector for l1
    if abs(b)>1e-12:
        v1=(l1-c, b)  # unnormalized
    else:
        v1=(1.0,0.0) if a>=c else (0.0,1.0)
    nrm=math.hypot(*v1); v1=(v1[0]/nrm, v1[1]/nrm) if nrm>0 else (1.0,0.0)
    return l1,l2,v1

def effective_rank(eigs):
    """participation-ratio effective rank of eigenvalue spectrum."""
    s=sum(eigs)
    if s<=0: return float('nan')
    p=[e/s for e in eigs if e>0]
    ent=-sum(pi*math.log(pi) for pi in p if pi>0)
    return math.exp(ent)

def principal_angle_deg(v):
    """angle of principal axis w.r.t. x-axis, folded to [0,90]."""
    ang=math.degrees(math.atan2(v[1], v[0]))
    ang=abs(ang)%180
    if ang>90: ang=180-ang
    return ang

def gini(vals):
    """Gini of nonnegative 'error mass' concentration. vals sorted internally."""
    v=sorted(x for x in vals)
    n=len(v)
    if n==0 or sum(v)==0: return float('nan')
    cum=0; s=sum(v)
    # standard Gini
    num=sum((i+1)*v[i] for i in range(n))
    return (2*num)/(n*s) - (n+1)/n

def entropy_counts(counter):
    tot=sum(counter.values())
    if tot==0: return float('nan'),float('nan')
    ps=[c/tot for c in counter.values() if c>0]
    H=-sum(p*math.log2(p) for p in ps)
    Hmax=math.log2(len(counter)) if len(counter)>1 else float('nan')
    return H, (H/Hmax if Hmax and Hmax>0 else float('nan'))

def skew(x):
    n=len(x)
    if n<3: return float('nan')
    m=mean(x); s=std(x)
    if s==0: return float('nan')
    return (sum(((v-m)/s)**3 for v in x))*n/((n-1)*(n-2))

def kurtosis_excess(x):
    n=len(x)
    if n<4: return float('nan')
    m=mean(x); s=std(x)
    if s==0: return float('nan')
    g2=sum(((v-m)/s)**4 for v in x)/n - 3
    return g2

def perm_test_corr(x,y,nperm=2000,seed=0):
    import random
    rng=random.Random(seed)
    r0=abs(pearson(x,y))
    yy=list(y); cnt=0
    for _ in range(nperm):
        rng.shuffle(yy)
        if abs(pearson(x,yy))>=r0: cnt+=1
    return (cnt+1)/(nperm+1)

# =========================================================================
# TASK A : Leg-1 modality separation
# =========================================================================
def load_avh_tokens():
    rows=[json.loads(l) for l in open(os.path.join(ROOT,"outputs/claim_probe/avh_claim_probe_report_tokens.jsonl"))]
    return rows

def load_cmm_cf():
    rows=[json.loads(l) for l in open(os.path.join(ROOT,"outputs/cmm_cf_confess/cf_lam1.0_full_single_scored.jsonl"))]
    return rows

def modality_metrics(dv, da, label, n_min=30):
    n=len(dv)
    if n<n_min:
        return {"label":label,"n":n,"skip":True}
    r=pearson(dv,da)
    cxx,cxy,cyy=cov_matrix(dv,da)
    l1,l2,v1=eig2x2(cxx,cxy,cyy)
    eff=effective_rank([l1,l2])
    ang=principal_angle_deg(v1)
    pc1_energy = l1/(l1+l2) if (l1+l2)>0 else float('nan')
    cond = (l1/l2) if l2>0 else float('inf')
    # normalized MI via 2D binning (quartile bins)
    mi = norm_mi(dv,da,bins=4)
    # permutation p for independence (|r|)
    p_perm = perm_test_corr(dv,da,nperm=1500,seed=1)
    return {"label":label,"n":n,"skip":False,
            "pearson":round(r,4),
            "principal_angle_deg":round(ang,2),
            "eff_rank":round(eff,4),
            "pc1_energy_frac":round(pc1_energy,4),
            "cond_number":round(cond,3) if cond!=float('inf') else None,
            "norm_MI":round(mi,4),
            "perm_p_indep":round(p_perm,4)}

def norm_mi(x,y,bins=4):
    """normalized mutual information via quantile binning. NMI = I/sqrt(Hx*Hy)."""
    n=len(x)
    if n<bins*bins: return float('nan')
    def qbin(v):
        s=sorted(v)
        qs=[s[int(k*n/bins)] for k in range(1,bins)]
        out=[]
        for a in v:
            b=0
            for q in qs:
                if a>q: b+=1
            out.append(b)
        return out
    bx=qbin(x); by=qbin(y)
    cx=Counter(bx); cy=Counter(by); cxy=Counter(zip(bx,by))
    Hx=-sum((c/n)*math.log2(c/n) for c in cx.values())
    Hy=-sum((c/n)*math.log2(c/n) for c in cy.values())
    I=0
    for (a,b),c in cxy.items():
        pxy=c/n; px=cx[a]/n; py=cy[b]/n
        I+=pxy*math.log2(pxy/(px*py))
    denom=math.sqrt(Hx*Hy) if Hx>0 and Hy>0 else float('nan')
    return I/denom if denom and denom>0 else float('nan')

def run_task_a():
    results=[]
    # --- AVHBench: token-level dv/da ---
    avh=load_avh_tokens()
    dv=[r['dv'] for r in avh]; da=[r['da'] for r in avh]
    results.append(modality_metrics(dv,da,"AVHBench_all_tokens"))
    # subset by ac (semantic modality tag)
    for tag in ['asrc','color','count']:
        sub=[r for r in avh if r.get('ac')==tag]
        results.append(modality_metrics([r['dv'] for r in sub],[r['da'] for r in sub],f"AVHBench_ac={tag}"))
    # None (generic content) subset
    subN=[r for r in avh if r.get('ac') is None]
    results.append(modality_metrics([r['dv'] for r in subN],[r['da'] for r in subN],"AVHBench_ac=generic"))
    # subset by hit (hallucination vs grounded)
    for hv in [True,False]:
        sub=[r for r in avh if r.get('hit')==hv]
        results.append(modality_metrics([r['dv'] for r in sub],[r['da'] for r in sub],f"AVHBench_hit={hv}"))

    # --- CMM: decision-level d_cf_video / d_cf_audio ---
    cmm=load_cmm_cf()
    dvc=[r['cf']['d_cf_video'] for r in cmm if 'cf' in r and r['cf'].get('d_cf_video') is not None]
    dac=[r['cf']['d_cf_audio'] for r in cmm if 'cf' in r and r['cf'].get('d_cf_audio') is not None]
    # align (same rows)
    pairs=[(r['cf']['d_cf_video'],r['cf']['d_cf_audio']) for r in cmm
           if 'cf' in r and r['cf'].get('d_cf_video') is not None and r['cf'].get('d_cf_audio') is not None]
    dvc=[p[0] for p in pairs]; dac=[p[1] for p in pairs]
    results.append(modality_metrics(dvc,dac,"CMM_all_decisions"))
    # subset by group_key (video-dominant vs audio-dominant task)
    for gk,lab in [("overrely_visual_ignore_audio","CMM_video_dominant"),
                   ("overrely_audio_ignore_visual","CMM_audio_dominant")]:
        sub=[(r['cf']['d_cf_video'],r['cf']['d_cf_audio']) for r in cmm
             if r.get('group_key')==gk and 'cf' in r and r['cf'].get('d_cf_video') is not None]
        results.append(modality_metrics([p[0] for p in sub],[p[1] for p in sub],lab))
    # subset by correctness
    for cv,lab in [(1,"CMM_correct"),(0,"CMM_wrong")]:
        sub=[(r['cf']['d_cf_video'],r['cf']['d_cf_audio']) for r in cmm
             if r.get('eval_metrics',{}).get('Is_Correct')==cv and 'cf' in r and r['cf'].get('d_cf_video') is not None]
        results.append(modality_metrics([p[0] for p in sub],[p[1] for p in sub],lab))
    return results

# =========================================================================
# TASK B : cross-6-dataset universal signals from norm_logp + universal fields
# =========================================================================
DATASETS = {
    "avhbench":  ("outputs/avhbench_base_model/avhbench_base_model_full_single_scored.jsonl","gen"),
    "haloquest": ("outputs/haloquest_base_model/haloquest_base_model_full_single_scored.jsonl","gen"),
    "ragtruth":  ("outputs/ragtruth_base_model/ragtruth_base_model_full_single_scored.jsonl","gen"),
    "cmm":       ("outputs/cmm_cf_confess/cf_lam1.0_full_single_scored.jsonl","judge"),
    "phd":       ("outputs/phd_cf_confess/cf_lam1.0_full_single_scored.jsonl","judge"),
    "pubmedqa":  ("outputs/pubmedqa_cf_confess/cf_lam0.25_full_single_scored.jsonl","judge"),
}
MODALITY = {"avhbench":"AV","cmm":"AV","haloquest":"V","phd":"V","ragtruth":"T","pubmedqa":"T"}

def load_ds(name):
    path,kind=DATASETS[name]
    rows=[json.loads(l) for l in open(os.path.join(ROOT,path))]
    recs=[]
    for r in rows:
        nl=r.get('norm_logp')
        if nl is None:
            cl=r.get('cumulative_logprob'); ot=r.get('output_tokens')
            if cl is not None and ot: nl=cl/ot
        ot=r.get('output_tokens')
        em=r.get('eval_metrics',{})
        if kind=="judge":
            correct=em.get('Is_Correct')
            pred=em.get('Prediction')
        else:
            f1=em.get('Token-F1')
            correct=None; pred=None
            recs_f1=f1
        rec={"norm_logp":nl,"output_tokens":ot,"kind":kind}
        if kind=="judge":
            rec["correct"]=correct; rec["pred"]=pred
        else:
            rec["f1"]=em.get('Token-F1')
        recs.append(rec)
    # binarize correctness for gen via f1>=median
    if kind=="gen":
        f1s=[r["f1"] for r in recs if r["f1"] is not None]
        med=st.median(f1s) if f1s else 0
        for r in recs:
            r["correct"]= (1 if (r["f1"] is not None and r["f1"]>=med) else 0) if r["f1"] is not None else None
        for r in recs: r["_f1_med"]=med
    return recs, kind

def run_task_b():
    per_ds={}
    for name in DATASETS:
        recs,kind=load_ds(name)
        good=[r for r in recs if r["norm_logp"] is not None and r["output_tokens"]]
        n=len(good)
        nl=[r["norm_logp"] for r in good]
        ot=[r["output_tokens"] for r in good]
        corr=[r["correct"] for r in good if r["correct"] is not None]
        acc=mean(corr) if corr else float('nan')

        # 1) length-confidence drift: corr(norm_logp, log output_tokens)
        logot=[math.log(max(v,1)) for v in ot]
        r_nl_len=pearson(nl,logot)
        # confidence-error link: corr(norm_logp, correct)
        pairs_ce=[(r["norm_logp"],r["correct"]) for r in good if r["correct"] is not None]
        r_conf_corr=pearson([p[0] for p in pairs_ce],[p[1] for p in pairs_ce]) if len(pairs_ce)>=30 else float('nan')
        # length-error link
        r_len_corr=pearson([math.log(max(r["output_tokens"],1)) for r in good if r["correct"] is not None],
                           [r["correct"] for r in good if r["correct"] is not None]) if len(pairs_ce)>=30 else float('nan')

        # 2) distribution shape of norm_logp
        sk=skew(nl); ku=kurtosis_excess(nl)

        # 3) error concentration: Gini of error-mass over norm_logp deciles
        # bin norm_logp into deciles, error mass per bin = #wrong; measure concentration
        wrong=[r for r in good if r["correct"]==0]
        gini_err=float('nan'); err_lowconf_frac=float('nan')
        if len(pairs_ce)>=30 and len(wrong)>=10:
            order=sorted(good, key=lambda r:r["norm_logp"])
            k=10
            binsz=max(1,len(order)//k)
            errcounts=[]
            for b in range(k):
                seg=order[b*binsz:(b+1)*binsz] if b<k-1 else order[b*binsz:]
                errcounts.append(sum(1 for r in seg if r["correct"]==0))
            gini_err=gini(errcounts)
            # fraction of errors in the lowest-confidence 30% of samples
            cutidx=int(0.3*len(order))
            lowconf=order[:cutidx]
            err_lowconf_frac=(sum(1 for r in lowconf if r["correct"]==0)/max(1,len(wrong)))

        # 4) answer distribution collapse (judge only)
        maj_frac=float('nan'); pred_ent_norm=float('nan')
        if kind=="judge":
            preds=[r["pred"] for r in good if r["pred"] is not None]
            if preds:
                c=Counter(preds); H,Hn=entropy_counts(c)
                pred_ent_norm=Hn
                maj_frac=max(c.values())/len(preds)

        per_ds[name]={
            "n":n,"modality":MODALITY[name],"kind":kind,"acc":round(acc,4) if acc==acc else None,
            "mean_norm_logp":round(mean(nl),4),"std_norm_logp":round(std(nl),4),
            "mean_output_tokens":round(mean(ot),2),"median_output_tokens":st.median(ot),
            "corr_conf_len(logot)":round(r_nl_len,4),
            "corr_conf_correct":round(r_conf_corr,4) if r_conf_corr==r_conf_corr else None,
            "corr_len_correct":round(r_len_corr,4) if r_len_corr==r_len_corr else None,
            "norm_logp_skew":round(sk,4) if sk==sk else None,
            "norm_logp_kurtosis":round(ku,4) if ku==ku else None,
            "gini_error_over_conf":round(gini_err,4) if gini_err==gini_err else None,
            "err_frac_in_low30conf":round(err_lowconf_frac,4) if err_lowconf_frac==err_lowconf_frac else None,
            "pred_entropy_norm":round(pred_ent_norm,4) if pred_ent_norm==pred_ent_norm else None,
            "majority_pred_frac":round(maj_frac,4) if maj_frac==maj_frac else None,
        }
    return per_ds

# =========================================================================
def fmt(v):
    return "" if v is None else (f"{v}" if not isinstance(v,float) else f"{v:.4f}")

def main():
    a=run_task_a()
    b=run_task_b()

    # --- write JSON ---
    with open(os.path.join(OUT,"leg1_modality_table.json"),"w") as f:
        json.dump(a,f,indent=2)
    with open(os.path.join(OUT,"universal_signal_matrix.json"),"w") as f:
        json.dump(b,f,indent=2)

    # --- markdown table A ---
    colsA=["label","n","pearson","principal_angle_deg","eff_rank","pc1_energy_frac","cond_number","norm_MI","perm_p_indep"]
    linesA=["| "+" | ".join(colsA)+" |","|"+"|".join(["---"]*len(colsA))+"|"]
    for r in a:
        if r.get("skip"):
            linesA.append("| "+r["label"]+" | "+str(r["n"])+" | SKIP(n<30) |"+ " |"*(len(colsA)-3))
        else:
            linesA.append("| "+" | ".join(fmt(r.get(c)) for c in colsA)+" |")
    tblA="\n".join(linesA)

    # --- markdown table B ---
    colsB=["name","modality","kind","n","acc","mean_norm_logp","std_norm_logp","mean_output_tokens",
           "corr_conf_len(logot)","corr_conf_correct","corr_len_correct","norm_logp_skew","norm_logp_kurtosis",
           "gini_error_over_conf","err_frac_in_low30conf","pred_entropy_norm","majority_pred_frac"]
    linesB=["| "+" | ".join(colsB)+" |","|"+"|".join(["---"]*len(colsB))+"|"]
    for name,d in b.items():
        row=[name]+[fmt(d.get(c)) for c in colsB[1:]]
        linesB.append("| "+" | ".join(row)+" |")
    tblB="\n".join(linesB)

    with open(os.path.join(OUT,"leg1_modality_table.md"),"w") as f:
        f.write("# Leg-1 Modality Separation (with subset stratification)\n\n"+tblA+"\n")
    with open(os.path.join(OUT,"universal_signal_matrix.md"),"w") as f:
        f.write("# Cross-6-dataset Universal Signal Matrix\n\n"+tblB+"\n")

    print("=== TABLE A: Leg-1 modality ===")
    print(tblA)
    print("\n=== TABLE B: universal ===")
    print(tblB)
    return a,b

if __name__=="__main__":
    main()
