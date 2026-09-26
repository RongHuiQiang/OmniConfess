#!/usr/bin/env python3
"""
Confidence blind-spot analysis (problem-structure leg 3).

Zero-GPU, read-only over outputs/*scored*.jsonl. For each (dataset x backbone x
method) cell it extracts a surface-confidence signal and a correctness signal,
builds an 8-bin equal-frequency reliability curve, and computes a Blind-Spot
Index (BSI = acc@top-conf-bin - acc@bottom-conf-bin) plus Spearman rho.

Surface confidence口径 (strict, per task prompt):
  - Judgment tasks (cmm/pubmedqa/phd -> yes/no/maybe):
      conf = |cf.margin_full|              (where cf exists: cf_confess runs)
             or |lp_yes_full - lp_no_full| (pubmedqa cf: no margin_full field)
             else conf = norm_logp          (base/other runs: mean token logprob
                                             of the answer = surface confidence)
  - Generation tasks (avhbench/haloquest/ragtruth):
      conf = norm_logp (sequence mean token logprob). Fallback: cumulative_logprob
             / len(token_ids), or min(logp_list).

Correctness口径:
  - Judgment: eval_metrics.Is_Correct (bool).
  - Generation: Token-F1 >= median(Token-F1)  -> 1 else 0  (binarized, noted).

BSI: >=0.15 -> calibrated ; <0.15 -> blind spot.
"""
import json, glob, os, sys
import numpy as np
from scipy.stats import spearmanr

OUT_DIR = "/home/tsq/rhq/omniconfess/outputs/problem_structure_fig"
ROOT = "/home/tsq/rhq/omniconfess/OmniConfess"

JUDGE_DS = {"cmm", "pubmedqa", "phd"}
GEN_DS = {"avhbench", "haloquest", "ragtruth"}


def reliability(conf, correct, K=8):
    conf = np.asarray(conf, float); cor = np.asarray(correct, int)
    m = np.isfinite(conf); conf, cor = conf[m], cor[m]
    if len(conf) < K:  # too few points to bin
        return []
    q = np.quantile(conf, np.linspace(0, 1, K + 1)); out = []
    for i in range(K):
        lo, hi = q[i], q[i + 1]
        sel = (conf >= lo) & (conf <= hi) if i == K - 1 else (conf >= lo) & (conf < hi)
        if sel.sum() < 5:
            continue
        out.append((float(conf[sel].mean()), float(cor[sel].mean()), int(sel.sum())))
    return out


def infer_dataset(path):
    p = path.lower()
    for ds in ("avhbench", "haloquest", "ragtruth", "pubmedqa", "cmm", "phd"):
        if ds in p:
            return ds
    return None


def infer_backbone(path):
    p = path.lower()
    if "nemotron" in p:
        return "nemotron"
    if "qwen3omni" in p or "qwen3" in p:
        return "qwen3"
    return "qwen2.5"  # default backbone


def infer_method(path):
    """Method from directory/file name, e.g. base_model, base_cad, cf_lam1.0, qwen3_sc."""
    fname = os.path.basename(path).replace("_single_scored.jsonl", "").replace(
        "_full", "")
    parent = os.path.basename(os.path.dirname(path))
    # prefer the parent dir tag after dataset name
    ds = infer_dataset(path)
    tag = parent
    if ds and parent.startswith(ds + "_"):
        tag = parent[len(ds) + 1:]
    elif ds and parent == ds:
        tag = fname
    # strip backbone tokens
    for b in ("qwen3omni_", "qwen3_", "nemotron_"):
        tag = tag.replace(b, "")
    # keep lambda token from filename so different-lambda cf runs stay distinct
    import re
    m = re.search(r"lam[\d.]+", fname)
    if m and m.group(0) not in tag:
        tag = f"{tag}_{m.group(0)}"
    return tag or fname


def extract(path, dataset):
    """Return (conf list, correct list, conf_field_used, note)."""
    conf, cor = [], []
    conf_field = None
    # generation tasks need Token-F1 median -> two pass
    if dataset in GEN_DS:
        vals, f1s = [], []
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            em = d.get("eval_metrics") or {}
            f1 = em.get("Token-F1")
            if f1 is None:
                continue
            c = _seq_conf(d)
            if c is None:
                continue
            vals.append(c); f1s.append(float(f1))
        if not vals:
            return [], [], None, "no seq-conf / Token-F1"
        med = float(np.median(f1s))
        cor = [1 if x >= med else 0 for x in f1s]
        conf_field = "norm_logp(seq mean logprob)"
        note = f"correct = Token-F1>=median({med:.2f})"
        return vals, cor, conf_field, note

    # judgment tasks
    has_margin = has_lp = has_norm = False
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        em = d.get("eval_metrics") or {}
        c = em.get("Is_Correct")
        if c is None:
            continue
        cf = d.get("cf") or {}
        v = None
        if cf.get("margin_full") is not None:
            v = abs(cf["margin_full"]); has_margin = True
        elif cf.get("lp_yes_full") is not None and cf.get("lp_no_full") is not None:
            v = abs(cf["lp_yes_full"] - cf["lp_no_full"]); has_lp = True
        elif d.get("norm_logp") is not None:
            v = d["norm_logp"]; has_norm = True
        if v is None:
            continue
        conf.append(v); cor.append(int(c))
    if has_margin:
        conf_field = "|cf.margin_full|"
    elif has_lp:
        conf_field = "|lp_yes_full-lp_no_full|"
    elif has_norm:
        conf_field = "norm_logp(mean answer logprob)"
    else:
        return [], [], None, "no confidence field"
    return conf, cor, conf_field, ""


def _seq_conf(d):
    if d.get("norm_logp") is not None:
        return d["norm_logp"]
    cl = d.get("cumulative_logprob"); tid = d.get("token_ids")
    if cl is not None and tid:
        return cl / max(1, len(tid))
    lp = d.get("logp_list")
    if lp:
        return float(np.mean(lp))
    return None


def analyze(path):
    ds = infer_dataset(path)
    if ds is None:
        return None
    bb = infer_backbone(path)
    method = infer_method(path)
    conf, cor, field, note = extract(path, ds)
    if len(conf) < 20:
        return None
    curve = reliability(conf, cor)
    if len(curve) < 3:
        return None
    bsi = curve[-1][1] - curve[0][1]
    rho = spearmanr(conf, cor).correlation
    if not np.isfinite(rho):
        rho = float("nan")
    return {
        "dataset": ds, "backbone": bb, "method": method,
        "task_type": "judgment" if ds in JUDGE_DS else "generation",
        "n": len(conf), "conf_field": field, "note": note,
        "bsi": round(float(bsi), 4),
        "rho": round(float(rho), 4),
        "n_bins": len(curve),
        "curve": [[round(a, 4), round(b, 4), n] for a, b, n in curve],
        "path": path,
    }


def main():
    files = sorted(glob.glob(os.path.join(ROOT, "outputs", "**", "*scored*.jsonl"),
                             recursive=True))
    results = []
    for fp in files:
        if "shard" in os.path.basename(fp):
            continue
        if "_archived" in fp:
            continue
        try:
            r = analyze(fp)
        except Exception as e:
            sys.stderr.write(f"skip {fp}: {e}\n")
            continue
        if r:
            results.append(r)
    return results


if __name__ == "__main__":
    res = main()
    for r in sorted(res, key=lambda x: x["bsi"]):
        print(f"{r['bsi']:+.3f} rho={r['rho']:+.3f} n={r['n']:4d} bins={r['n_bins']} "
              f"{r['dataset']:9s} {r['backbone']:8s} {r['method']:18s} {r['conf_field']}")
    print(f"\ntotal cells: {len(res)}")
