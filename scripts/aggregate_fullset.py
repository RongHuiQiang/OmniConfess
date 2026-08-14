#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""aggregate_fullset.py -- recompute per-method scores on the unfiltered set.

The scoring conventions are those of ``OmniConfess/eval/single_eval.py``:

  * Judgement datasets: Categorical-F1 = binary ``f1_score(pos_label=1)``, and
    PhD-Index = ``2*r_yes*r_no/(r_yes+r_no)``. Both count only samples whose
    ``status == "success"``.
  * Descriptive datasets: Token-F1 / GAVIE-Rel / GAVIE-Acc are means over
    ``status == "success"`` samples, and GAVIE additionally requires
    ``rel >= 0``.

Two views are produced:

  per-method  each method scored on all of its own samples
  common-ID   all methods scored on the intersection of their ids, which is the
              stricter alignment

Usage:
    python scripts/aggregate_fullset.py --manifest scripts/judge_fullset_manifest.json
"""
import argparse
import json
import os
from collections import defaultdict

from sklearn.metrics import f1_score, recall_score

JUDGEMENT = {"pubmedqa", "phd", "cmm"}
DESCRIPTIVE = {"ragtruth", "haloquest", "avhbench"}


def load_scored(path):
    """Read a ``_single_scored.jsonl`` file into ``{id: eval_metrics}``.

        Ids are normalised to ``str``: baseline files write the string "0" while
        OmniConfess output writes the integer 0, and without normalisation the
        ``--common`` intersection is always empty.
    """
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            em = o.get("eval_metrics")
            if isinstance(em, dict):
                out[str(o.get("id"))] = em
    return out


def score_judgement(metrics_by_id, ids=None):
    yt, yp = [], []
    for i, em in metrics_by_id.items():
        if ids is not None and i not in ids:
            continue
        if str(em.get("Status", "success")).lower() != "success":
            continue
        gt, pred = em.get("Ground_Truth"), em.get("Prediction")
        if gt is None or pred is None:
            continue
        yt.append(gt)
        yp.append(pred)
    if not yt:
        return None
    r_yes = recall_score(yt, yp, pos_label=1, zero_division=0)
    r_no = recall_score(yt, yp, pos_label=0, zero_division=0)
    idx = (2 * r_yes * r_no) / (r_yes + r_no) if (r_yes + r_no) > 0 else 0.0
    return {
        "n_scored": len(yt),
        "F1": round(f1_score(yt, yp, zero_division=0) * 100, 2),
        "Idx": round(idx * 100, 2),
    }


def score_descriptive(metrics_by_id, ids=None):
    f1s, rels, accs = [], [], []
    for i, em in metrics_by_id.items():
        if ids is not None and i not in ids:
            continue
        if str(em.get("Status", "success")).lower() != "success":
            continue
        f1s.append(em.get("Token-F1", 0.0))
        rel = em.get("GAVIE-Rel", -1)
        if rel is not None and rel >= 0:
            rels.append(rel)
            accs.append(em.get("GAVIE-Acc", 0.0))
    if not f1s:
        return None
    mean = lambda xs: sum(xs) / len(xs)
    return {
        "n_scored": len(f1s),
        "F1": round(mean(f1s), 2),
        "GAVIE": round(mean(accs), 2) if accs else 0.0,
        "n_gavie": len(accs),
    }


def status_profile(path):
    """Distribution of the ``status`` field within one file.

        This is where the single_eval convention becomes asymmetric:
        ``str(item.get('status', 'success'))`` means that **a file with no status
        field has every sample counted as a success**, and therefore in the
        denominator. OmniConfess / DoLa / VCD output does carry status, so their
        OOM and other failures are removed from the denominator. Comparing two
        numbers computed over different denominators is not fair, so the
        distribution is recorded explicitly here.
    """
    has = False
    counts = {}
    if not os.path.exists(path):
        return has, counts
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "status" in o:
                has = True
            s = str(o.get("status", "<absent>")).lower()
            counts[s] = counts.get(s, 0) + 1
    return has, counts


def report(ds, files, view_ids=None, label="", csv_rows=None, view=""):
    # Manifest keys carry the universe size ("cmm_795"), so membership is
    # tested by substring -- the same convention get_system_prompt uses for
    # dataset_type. An exact-match test silently routes every judgement
    # dataset to the descriptive scorer and reports F1=0 for all of them.
    is_judge = any(k in ds for k in JUDGEMENT)
    scorer = score_judgement if is_judge else score_descriptive
    rows = []
    for method, path in files.items():
        if method.startswith("_"):   # manifest metadata, not an arm
            continue
        if not os.path.exists(path):
            rows.append((method, None, "FILE MISSING"))
            if csv_rows is not None:
                csv_rows.append({
                    "dataset": ds, "view": view, "method": method,
                    "status": "FILE MISSING", "source_file": path,
                })
            continue
        data = load_scored(path)
        res = scorer(data, ids=view_ids)
        rows.append((method, res, f"n_total={len(data)}"))

        if csv_rows is not None:
            has_st, st_counts = status_profile(path)
            row = {
                "dataset": ds, "view": view, "method": method,
                "task_type": "judgement" if is_judge else "descriptive",
                "n_total": len(data),
                "n_scored": res["n_scored"] if res else 0,
                "n_excluded_by_status": (len(data) - res["n_scored"]) if res else "",
                "has_status_field": has_st,
                "status_counts": json.dumps(st_counts, sort_keys=True),
                "source_file": path,
            }
            if res:
                if is_judge:
                    row["Categorical_F1"] = res["F1"]
                    row["PhD_Index"] = res["Idx"]
                else:
                    row["Token_F1"] = res["F1"]
                    row["GAVIE_Acc"] = res["GAVIE"]
                    row["n_gavie"] = res["n_gavie"]
            csv_rows.append(row)

    metric2 = "Idx" if is_judge else "GAVIE"
    print(f"\n===== {ds} {label} =====")
    print(f"{'Method':<22}{'n_total':>9}{'n_scored':>10}{'F1':>9}{metric2:>9}")
    for method, res, note in rows:
        if res is None:
            print(f"{method:<22}{note:>9}")
            continue
        n_total = note.split("=")[1]
        print(f"{method:<22}{n_total:>9}{res['n_scored']:>10}"
              f"{res['F1']:>9}{res[metric2]:>9}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="scripts/fullset_manifest.json",
                    help="JSON: {dataset: {method: scored_file_path}}")
    ap.add_argument("--common", action="store_true",
                    help="also report aligned scores on the ids common to every method")
    ap.add_argument("--csv", default=None,
                    help="write every number, together with its source file path, to a summary CSV")
    args = ap.parse_args()

    with open(args.manifest, encoding="utf-8") as f:
        manifest = json.load(f)

    csv_rows = [] if args.csv else None

    for ds, files in manifest.items():
        report(ds, files, label="(per-method full set)",
               csv_rows=csv_rows, view="per-method")

        if args.common:
            id_sets = []
            for path in files.values():
                if os.path.exists(path):
                    id_sets.append(set(load_scored(path).keys()))
            if id_sets:
                common = set.intersection(*id_sets)
                report(ds, files, view_ids=common,
                       label=f"(common IDs, n={len(common)})",
                       csv_rows=csv_rows, view=f"common-ID(n={len(common)})")

    if args.csv:
        import csv as _csv
        cols = ["dataset", "view", "method", "task_type", "n_total", "n_scored",
                "n_excluded_by_status", "has_status_field", "status_counts",
                "Categorical_F1", "PhD_Index", "Token_F1", "GAVIE_Acc",
                "n_gavie", "status", "source_file"]
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = _csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(csv_rows)
        print(f"\n✅ summary CSV: {args.csv}  ({len(csv_rows)} rows)")


if __name__ == "__main__":
    main()
