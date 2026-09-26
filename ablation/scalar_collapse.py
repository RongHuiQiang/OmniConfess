#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Scalar-collapse control arms (H3 closing link) -- zero-GPU offline replay.

Pre-registered design (written before running; task card in
/home/tsq/rhq/omniconfess/RESULTS_pending_register.md):

  Question.  Does collapsing the structured confession C(X) = (d_m)_m into a
  SINGLE SCALAR suffice to drive the correction, or is the full structure
  (per-view sign + channel decomposition) necessary?  If an oracle-tuned
  scalar arm still sits significantly below Arm 0, structured necessity is
  closed; if a scalar arm matches Arm 0, the token x modality story must be
  downgraded and this is reported as-is.

  Data.  outputs/cf_probe/{phd,cmm}_probe_full.jsonl (per-sample margin_full,
  per-view lp_yes, gt, base pred).  Offline replay = online was validated
  bit-for-bit in ablation/a1_offline.json (phd 84.60 n=906, cmm 76.36 n=720).

  Conventions.  d in operator convention lp_yes(view) - lp_yes(full).
  phd: D = d_noctx (shipped single view, w=1); channel set for u = {noctx, noimg}.
  cmm: D = d_cf_sum = d_cf_audio + d_cf_video (shipped w=1,1); channels {audio, video}.
  Anchor lambda (frozen, shipped): phd 2.0, cmm 1.0.

  Arms.
    Arm 0  full structured CF: pred yes iff margin_full + lam*D > 0.
           Zero free params.  Must reproduce 84.60 / 76.36 (method self-check).
    Arm 1  magnitude trigger + blind fixed-direction flip:
           s = |D|; if s > tau: pred = FIXED class (no amplitude, no sign);
           else base.  tau swept over all unique s values; both fixed
           directions swept (prior hint: phd no->yes, cmm yes->no; the label
           prior computed from the probe agrees, see config block).
    Arm 2  magnitude-only delta decision: pred yes iff margin_full - lam2*s > 0
           (s always positive, direction fixed by the sign of lam2; lam2 < 0
           pushes toward yes).  Keeps magnitude, drops sign.
    Arm 3  single scalar u = max_m |d_m|; quantile trigger u > tau_q;
           action fixed-lambda blind repair margin_full - dir*lam3*A with
           A in {1 (constant blind amplitude), u (u-proportional)} and
           dir in {+1 toward-no, -1 toward-yes}; else base.

  Fairness (registered).  Every scalar arm is reported twice:
    (a) oracle  = free params swept to max F1 ON THE FULL SET (upper bound);
    (b) dev     = params picked on the protocol-matched stratified dev probes
                  (phd_probe.jsonl n=100, cmm_probe.jsonl n=87; same 25/cell
                  seed-42 protocol as a3/a13 dev usage), then applied to full.
  Registered grids:
    Arm 1 tau: all unique s values (trigger-all .. trigger-none).
    Arm 2 lam2: 0, +-{0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8}.
    Arm 3 (q, lam3, dir, A): q in {0,.1,...,.9,.95} (tau = quantile(u, q) on
    the selection set), lam3 in {0.5,1,1.5,2,3,4,6,8}, dir in {+1,-1},
    A in {1, u}.
  Metrics per arm: Yes-F1, Youden J, paired exact McNemar vs Arm 0
  (b = Arm0 wrong & arm right, c = Arm0 right & arm wrong), four cells vs
  base (fix / destroy / stay-wrong / stay-right), trigger rate, intervention
  rate (pred != base).  Base = replay base (margin_full > 0); the 1/906 phd
  margin-vs-recorded-pred discrepancy is reported in the validate block.

  Two additions made when the first run showed the cmm Arm 3 optimum
  degenerating to q=0 (trigger ~always, constant amplitude) -- both clearly
  labelled POSTHOC in the output json:
    armT  threshold-only control: pred yes iff margin_full - c > 0, c swept
          -4..+4 step 0.25.  Uses NO confession signal at all; bounds what
          pure decision-threshold recalibration achieves.
    oracle_best_youden: every arm additionally reports the params maximising
          Youden (secondary criterion; primary oracle criterion is F1).

Run: /ultrafast_share/tsq/venvs/omniconfess/bin/python ablation/scalar_collapse.py
Out: ablation/scalar_collapse.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from util import mcnemar_exact  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
CF = os.path.join(ROOT, "outputs", "cf_probe")
OUT = os.path.join(HERE, "scalar_collapse.json")

LAM = {"phd": 2.0, "cmm": 1.0}
LAM2_GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0,
             -0.25, -0.5, -0.75, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0, -8.0]
LAM3_GRID = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]
Q_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
DIRS = {"toward_no": 1.0, "toward_yes": -1.0}  # margin - dir*lam*A


def jl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def load_family(ds):
    full = jl(os.path.join(CF, f"{ds}_probe_full.jsonl"))
    dev = jl(os.path.join(CF, f"{ds}_probe.jsonl"))
    if ds == "phd":
        for r in full + dev:
            r["D"] = r["lp_yes_noctx"] - r["lp_yes_full"]          # operator d, noctx
            r["dvec"] = [r["lp_yes_noctx"] - r["lp_yes_full"],
                         r["lp_yes_noimg"] - r["lp_yes_full"]]      # channels {noctx,noimg}
    else:
        for r in full + dev:
            r["D"] = r["d_cf_sum"]                                  # audio+video, w=1,1
            r["dvec"] = [r["d_cf_audio"], r["d_cf_video"]]
    return full, dev


def margins_arm0(rows, lam):
    return [r["margin_full"] + lam * r["D"] for r in rows]


def preds_from_margins(ms):
    return [1 if m > 0 else 0 for m in ms]


def base_preds(rows):
    return preds_from_margins([r["margin_full"] for r in rows])


def gts(rows):
    return [r["gt_yes"] for r in rows]


def metrics(yt, yp):
    tp = sum(1 for g, p in zip(yt, yp) if g == 1 and p == 1)
    fp = sum(1 for g, p in zip(yt, yp) if g == 0 and p == 1)
    fn = sum(1 for g, p in zip(yt, yp) if g == 1 and p == 0)
    tn = sum(1 for g, p in zip(yt, yp) if g == 0 and p == 0)
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    return dict(f1=round(f1 * 100, 2), youden=round(tpr + tnr - 1, 4),
                tp=tp, fp=fp, fn=fn, tn=tn)


def fourcell_vs_base(yb, ya, yt):
    fix = sum(1 for g, b, a in zip(yt, yb, ya) if b != g and a == g)
    destroy = sum(1 for g, b, a in zip(yt, yb, ya) if b == g and a != g)
    stay_wrong = sum(1 for g, b, a in zip(yt, yb, ya) if b != g and a != g)
    stay_right = sum(1 for g, b, a in zip(yt, yb, ya) if b == g and a == g)
    return dict(fix=fix, destroy=destroy, stay_wrong=stay_wrong,
                stay_right=stay_right)


def mcnemar_vs_arm0(yp0, ya, yt):
    b = sum(1 for g, p0, pa in zip(yt, yp0, ya) if p0 != g and pa == g)
    c = sum(1 for g, p0, pa in zip(yt, yp0, ya) if p0 == g and pa != g)
    return dict(arm0_wrong_arm_right=b, arm0_right_arm_wrong=c,
                p=mcnemar_exact(b, c))


def block(yt, yp, yb, yp0, params, extra=None):
    d = dict(params=params, **metrics(yt, yp))
    d["mcnemar_vs_arm0"] = mcnemar_vs_arm0(yp0, yp, yt)
    d["fourcell_vs_base"] = fourcell_vs_base(yb, yp, yt)
    d["intervention_rate"] = round(
        sum(1 for a, b in zip(yp, yb) if a != b) / len(yb) * 100, 2)
    if extra:
        d.update(extra)
    return d


def pick(cands, key="sel_f1"):
    """Deterministic argmax over evaluated candidates by (f1, youden)."""
    return max(cands, key=lambda c: (c[key], c["sel_youden"]))


# ---------------- arm constructors: params -> yp on given rows ----------------

def arm1_preds(rows, tau, force):
    out = []
    for r in rows:
        if abs(r["D"]) > tau:
            out.append(1 if force == "yes" else 0)
        else:
            out.append(1 if r["margin_full"] > 0 else 0)
    return out


def arm2_preds(rows, lam2):
    return preds_from_margins([r["margin_full"] - lam2 * abs(r["D"])
                               for r in rows])


def arm3_preds(rows, tau, dirsign, lam3, amp):
    out = []
    for r in rows:
        u = max(abs(d) for d in r["dvec"])
        a = lam3 * (u if amp == "u" else 1.0)
        m = r["margin_full"] - dirsign * a if u > tau else r["margin_full"]
        out.append(1 if m > 0 else 0)
    return out


def arm1_tau_grid(rows):
    s = sorted(abs(r["D"]) for r in rows)
    return [s[0] - 1e-9] + sorted(set(s))     # trigger-all .. trigger-none


def arm3_tau_grid(rows):
    u = np.array([max(abs(d) for d in r["dvec"]) for r in rows], float)
    return {q: float(np.quantile(u, q)) for q in Q_GRID}


def armT_preds(rows, c):
    """Post-hoc supplementary control: threshold-only, NO confession signal.
    pred yes iff margin_full - c > 0.  Added during analysis because the cmm
    Arm 3 oracle optimum sat at q=0 (trigger ~always) with A=1, i.e. the arm
    degenerated to a constant margin shift; this control makes that explicit."""
    return preds_from_margins([r["margin_full"] - c for r in rows])


CT_GRID = [x * 0.25 for x in range(-16, 17)]   # -4 .. +4 step 0.25


def run_family(ds):
    full, dev = load_family(ds)
    lam = LAM[ds]
    yt, yb = gts(full), base_preds(full)
    yp0 = preds_from_margins(margins_arm0(full, lam))

    rec = {"n_full": len(full), "n_dev": len(dev), "lam_anchor": lam,
           "D_def": "phd: d_noctx (w=1)" if ds == "phd"
                    else "cmm: d_cf_audio+d_cf_video (w=1,1)",
           "channels_for_u": ["noctx", "noimg"] if ds == "phd"
                             else ["audio", "video"],
           "gt_yes_rate": round(sum(yt) / len(yt), 4),
           "base_yes_rate": round(sum(yb) / len(yb), 4),
           "label_prior_direction_hint": "force_yes" if ds == "phd"
                                         else "force_no",
           "note_prior": ("phd gt majority=yes -> prior flip direction "
                          "no->yes; cmm gt majority=no -> prior yes->no "
                          "(matches the pre-registered hint)")}

    # ---- validation: Arm 0 must reproduce the shipped online arm ----
    rec["validate"] = {
        "arm0_offline_f1": metrics(yt, yp0)["f1"],
        "arm0_offline_youden": metrics(yt, yp0)["youden"],
        "online_f1": 84.60 if ds == "phd" else 76.36,
        "online_youden": 0.542 if ds == "phd" else 0.600,
        "arm0_online_bitwise_match_source": "ablation/a1_offline.json validate",
        "base_replay_f1": metrics(yt, yb)["f1"],
        "base_online_f1": 71.68 if ds == "phd" else 70.07,
        "margin_sign_vs_recorded_pred_mismatch":
            sum(1 for r in full if (r["margin_full"] > 0) != (r["pred"] == "yes")),
    }

    out = {"config": rec, "base": block(yt, yb, yb, yp0, params={}),
           "arm0_full_structured_CF": block(yt, yp0, yb, yp0,
                                            params={"lam": lam})}

    # ---------------- Arm 1 ----------------
    taus_full = arm1_tau_grid(full)
    taus_dev = arm1_tau_grid(dev)
    dir_blocks = {}
    for force in ["yes", "no"]:
        cands = []
        for tau in taus_full:
            yp = arm1_preds(full, tau, force)
            m = metrics(yt, yp)
            cands.append(dict(sel_f1=m["f1"], sel_youden=m["youden"],
                              tau=round(tau, 4), yp=yp))
        best = pick(cands)
        dir_blocks[f"force_{force}"] = block(
            yt, best["yp"], yb, yp0, params={"tau": best["tau"], "force": force},
            extra=dict(trigger_rate=round(
                sum(1 for r in full if abs(r["D"]) > best["tau"]) / len(full) * 100, 2)))
        ybest = pick(cands, key="sel_youden")
        dir_blocks[f"force_{force}"]["oracle_best_youden"] = block(
            yt, ybest["yp"], yb, yp0,
            params={"tau": ybest["tau"], "force": force},
            extra=dict(criterion="max Youden (secondary)"))
        # dev-selected tau for this direction
        dcands = []
        for tau in taus_dev:
            ypd = arm1_preds(dev, tau, force)
            m = metrics(gts(dev), ypd)
            dcands.append(dict(sel_f1=m["f1"], sel_youden=m["youden"], tau=tau))
        dbest = pick(dcands)
        ypa = arm1_preds(full, dbest["tau"], force)
        dir_blocks[f"force_{force}"]["dev_selected"] = block(
            yt, ypa, yb, yp0,
            params={"tau": round(dbest["tau"], 4), "force": force},
            extra=dict(dev_f1=dbest["sel_f1"], trigger_rate=round(
                sum(1 for r in full if abs(r["D"]) > dbest["tau"]) / len(full) * 100, 2)))
    oracle = max(dir_blocks.values(), key=lambda b: b["f1"])
    oracle_j = max((v["oracle_best_youden"] for v in dir_blocks.values()),
                   key=lambda b: b["youden"])
    dev_all = max((v["dev_selected"] for v in dir_blocks.values()),
                  key=lambda b: b["f1"])
    out["arm1_magnitude_trigger_blind_flip"] = dict(
        oracle_best=oracle, oracle_best_youden=oracle_j,
        dev_selected_best=dev_all,
        per_direction=dir_blocks,
        tau_grid="all unique |D| values on selection set (trigger-all..none)")

    # ---------------- Arm 2 ----------------
    cands, grid_f1 = [], {}
    for lam2 in LAM2_GRID:
        yp = arm2_preds(full, lam2)
        m = metrics(yt, yp)
        grid_f1[lam2] = m["f1"]
        cands.append(dict(sel_f1=m["f1"], sel_youden=m["youden"],
                          lam2=lam2, yp=yp))
    best = pick(cands)
    ybest = pick(cands, key="sel_youden")
    out["arm2_magnitude_only_delta"] = dict(
        oracle_best=block(yt, best["yp"], yb, yp0,
                          params={"lam2": best["lam2"]}),
        oracle_best_youden=block(yt, ybest["yp"], yb, yp0,
                                 params={"lam2": ybest["lam2"]},
                                 extra=dict(criterion="max Youden (secondary)")),
        grid_f1=grid_f1)
    dcands = []
    for lam2 in LAM2_GRID:
        m = metrics(gts(dev), arm2_preds(dev, lam2))
        dcands.append(dict(sel_f1=m["f1"], sel_youden=m["youden"], lam2=lam2))
    dbest = pick(dcands)
    out["arm2_magnitude_only_delta"]["dev_selected_best"] = block(
        yt, arm2_preds(full, dbest["lam2"]), yb, yp0,
        params={"lam2": dbest["lam2"]}, extra=dict(dev_f1=dbest["sel_f1"]))

    # ---------------- Arm 3 ----------------
    taus_f = arm3_tau_grid(full)
    taus_d = arm3_tau_grid(dev)
    variants = {}
    for amp in ["1", "u"]:
        for dname, dsig in DIRS.items():
            cands = []
            for q, tau in taus_f.items():
                for lam3 in LAM3_GRID:
                    yp = arm3_preds(full, tau, dsig, lam3, amp)
                    m = metrics(yt, yp)
                    cands.append(dict(sel_f1=m["f1"], sel_youden=m["youden"],
                                      q=q, tau=round(tau, 4), lam3=lam3, yp=yp))
            best = pick(cands)
            ybest = pick(cands, key="sel_youden")
            key = f"A={amp},dir={dname}"
            variants[key] = block(
                yt, best["yp"], yb, yp0,
                params={"q": best["q"], "tau": best["tau"],
                        "lam3": best["lam3"], "dir": dname, "A": amp},
                extra=dict(trigger_rate=round(
                    sum(1 for r in full
                        if max(abs(d) for d in r["dvec"]) > best["tau"])
                    / len(full) * 100, 2)))
            variants[key]["oracle_best_youden"] = block(
                yt, ybest["yp"], yb, yp0,
                params={"q": ybest["q"], "tau": ybest["tau"],
                        "lam3": ybest["lam3"], "dir": dname, "A": amp},
                extra=dict(criterion="max Youden (secondary)"))
            dcands = []
            for q, tau in taus_d.items():
                for lam3 in LAM3_GRID:
                    m = metrics(gts(dev), arm3_preds(dev, tau, dsig, lam3, amp))
                    dcands.append(dict(sel_f1=m["f1"], sel_youden=m["youden"],
                                       q=q, lam3=lam3))
            db = pick(dcands)
            ypa = arm3_preds(full, taus_d[db["q"]], dsig, db["lam3"], amp)
            variants[key]["dev_selected"] = block(
                yt, ypa, yb, yp0,
                params={"q": db["q"], "tau": round(taus_d[db["q"]], 4),
                        "lam3": db["lam3"], "dir": dname, "A": amp},
                extra=dict(dev_f1=db["sel_f1"], trigger_rate=round(
                    sum(1 for r in full
                        if max(abs(d) for d in r["dvec"]) > taus_d[db["q"]])
                    / len(full) * 100, 2)))
    oracle = max(variants.values(), key=lambda b: b["f1"])
    oracle_j = max((v["oracle_best_youden"] for v in variants.values()),
                   key=lambda b: b["youden"])
    dev_all = max((v["dev_selected"] for v in variants.values()),
                  key=lambda b: b["f1"])
    out["arm3_single_scalar_u_trigger_blind_repair"] = dict(
        oracle_best=oracle, oracle_best_youden=oracle_j,
        dev_selected_best=dev_all, per_variant=variants,
        q_grid=Q_GRID, lam3_grid=LAM3_GRID,
        dir_grid=list(DIRS), amp_grid=["1 (constant)", "u (proportional)"])

    # ---- supplementary threshold-only control (post-hoc, labelled) ----
    tcands = []
    for c in CT_GRID:
        yp = armT_preds(full, c)
        m = metrics(yt, yp)
        tcands.append(dict(sel_f1=m["f1"], sel_youden=m["youden"], c=c, yp=yp))
    tbest = pick(tcands)
    tybest = pick(tcands, key="sel_youden")
    tdcands = []
    for c in CT_GRID:
        m = metrics(gts(dev), armT_preds(dev, c))
        tdcands.append(dict(sel_f1=m["f1"], sel_youden=m["youden"], c=c))
    tdbest = pick(tdcands)
    out["armT_threshold_only_control_POSTHOC"] = dict(
        oracle_best=block(yt, tbest["yp"], yb, yp0, params={"c": tbest["c"]},
                          extra=dict(criterion="max F1, no confession signal")),
        oracle_best_youden=block(yt, tybest["yp"], yb, yp0,
                                 params={"c": tybest["c"]},
                                 extra=dict(criterion="max Youden (secondary)")),
        dev_selected_best=block(yt, armT_preds(full, tdbest["c"]), yb, yp0,
                                params={"c": tdbest["c"]},
                                extra=dict(dev_f1=tdbest["sel_f1"])),
        c_grid="-4..+4 step 0.25")

    # ---- mechanism block: sign structure of the confession aggregate ----
    D = np.array([r["D"] for r in full], float)
    M = np.array([r["margin_full"] for r in full], float)
    adj = M + lam * D
    interv = (adj > 0) != (M > 0)
    ch = [np.array([r["dvec"][k] for r in full], float)
          for k in range(len(full[0]["dvec"]))]
    decisive = np.abs(M) < np.abs(D)
    out["mechanism_sign_structure"] = dict(
        D_negative_pct=round(float((D < 0).mean() * 100), 2),
        D_positive_pct=round(float((D > 0).mean() * 100), 2),
        sign_decisive_subset_pct=round(float(decisive.mean() * 100), 2),
        sign_decisive_D_negative_pct=round(float((D[decisive] < 0).mean() * 100), 2),
        rule_margin_plus_D_equals_margin_minus_absD_pct=round(
            float((((M + D) > 0) == ((M - np.abs(D)) > 0)).mean() * 100), 2),
        arm0_interventions=dict(
            n=int(interv.sum()),
            to_yes=int((adj[interv] > 0).sum()),
            to_no=int((adj[interv] <= 0).sum())),
        per_channel_negative_pct=[round(float((c < 0).mean() * 100), 2)
                                  for c in ch],
        per_channel_sign_agree_pct=round(
            float((np.sign(ch[0]) == np.sign(ch[1])).mean() * 100), 2))
    # ---- agreement diagnostics (does the scalar arm's decision set differ
    #      from Arm 0's?) ----
    def agree(ya):
        return dict(pred_agreement_pct=round(
            sum(1 for a, b in zip(ya, yp0) if a == b) / len(yp0) * 100, 2))
    out["agreement_with_arm0"] = {
        "arm1_oracle": agree(arm1_preds(full,
                                        out["arm1_magnitude_trigger_blind_flip"]["oracle_best"]["params"]["tau"],
                                        "yes" if out["arm1_magnitude_trigger_blind_flip"]["oracle_best"]["params"]["force"] == "yes" else "no")),
        "arm2_dev": agree(arm2_preds(full, out["arm2_magnitude_only_delta"]["dev_selected_best"]["params"]["lam2"])),
        "arm3_oracle": agree(arm3_preds(full,
                                        out["arm3_single_scalar_u_trigger_blind_repair"]["oracle_best"]["params"]["tau"],
                                        DIRS[out["arm3_single_scalar_u_trigger_blind_repair"]["oracle_best"]["params"]["dir"]],
                                        out["arm3_single_scalar_u_trigger_blind_repair"]["oracle_best"]["params"]["lam3"],
                                        out["arm3_single_scalar_u_trigger_blind_repair"]["oracle_best"]["params"]["A"])),
        "armT_oracle": agree(armT_preds(full, out["armT_threshold_only_control_POSTHOC"]["oracle_best"]["params"]["c"])),
    }
    return out


def main():
    res = {"_doc": __doc__.strip().splitlines()[:6], "families": {}}
    for ds in ["phd", "cmm"]:
        res["families"][ds] = run_family(ds)
    with open(OUT, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    # compact console summary
    for ds, fam in res["families"].items():
        print(f"==== {ds} (n={fam['config']['n_full']}, lam={fam['config']['lam_anchor']}) ====")
        v = fam["config"]["validate"]
        print(f"  Arm0 replay F1 {v['arm0_offline_f1']} / J {v['arm0_offline_youden']}"
              f"  (online {v['online_f1']} / {v['online_youden']});"
              f" base replay {v['base_replay_f1']} (online {v['base_online_f1']})")
        for k in ["arm1_magnitude_trigger_blind_flip",
                  "arm2_magnitude_only_delta",
                  "arm3_single_scalar_u_trigger_blind_repair"]:
            o, d = fam[k]["oracle_best"], fam[k]["dev_selected_best"]
            print(f"  {k.split('_')[0]}: oracle {o['f1']}/J{o['youden']} "
                  f"{o['params']} p={o['mcnemar_vs_arm0']['p']:.2e} | "
                  f"dev {d['f1']} {d['params']}")
    print("written:", OUT)


if __name__ == "__main__":
    main()
