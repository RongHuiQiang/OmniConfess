#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eval_confess_select.py — offline selector analysis over avh_confess_select shards.

GT is used here ONLY for evaluation. No selector below uses F1/GT to choose a
candidate; the confidence/consensus signals are GT-free. Oracle rows are ceilings.
"""
import glob, json, string, sys
from collections import Counter

STOP = set("a an the and or but if because as what which this that these those "
           "then just so than such both through about for is are was were of while "
           "during to in on at by with it there".split())


def _toks(s):
    s = str(s).lower().translate(str.maketrans("", "", string.punctuation))
    return [w for w in s.split() if w and w not in STOP]


def f1(pred, gt):
    p, g = _toks(pred), _toks(gt)
    if not p or not g:
        return 0.0
    c = Counter(p) & Counter(g); h = sum(c.values())
    if h == 0:
        return 0.0
    P, R = h / len(p), h / len(g)
    return 2 * P * R / (P + R)


def consensus_idx(texts):
    best_i, best_s = 0, -1.0
    for i, ti in enumerate(texts):
        s = sum(f1(ti, tj) for j, tj in enumerate(texts) if j != i)
        if s > best_s:
            best_s, best_i = s, i
    return best_i


def main():
    patt = sys.argv[1] if len(sys.argv) > 1 else "outputs/avh_confess_select/dev_shard*of4.jsonl"
    rows = []
    for fp in sorted(glob.glob(patt)):
        for line in open(fp):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows = {r["id"]: r for r in rows}.values()  # dedup by id
    rows = list(rows)
    n = len(rows)
    print(f"n={n} samples  ({patt})")

    names = ["base", "redescribe", "commit", "salient"]
    # per-generator mean F1 (always-use-this-generator)
    print("\n=== per-generator mean F1 (always use) ===")
    gen_means = {}
    for nm in names:
        vals = [r["cands"].get(nm, {}).get("f1", 0.0) for r in rows]
        gen_means[nm] = 100 * sum(vals) / n
        print(f"  {nm:11s} {gen_means[nm]:6.2f}")

    base_f1 = [r["cands"]["base"]["f1"] for r in rows]
    base_mean = 100 * sum(base_f1) / n

    # oracle over all candidates
    oracle = [max(r["cands"][nm]["f1"] for nm in names if nm in r["cands"]) for r in rows]
    print(f"\n  ORACLE(all cands) {100*sum(oracle)/n:6.2f}   (ceiling, +{100*sum(oracle)/n-base_mean:.2f} vs base)")

    # selectors
    def sel_conf(r):  # highest confidence candidate
        cs = [(nm, c) for nm, c in r["cands"].items() if c.get("conf") is not None and c.get("text")]
        if not cs:
            return r["cands"]["base"]["f1"], "base"
        nm = max(cs, key=lambda x: x[1]["conf"])[0]
        return r["cands"][nm]["f1"], nm

    def sel_consensus(r):
        items = [(nm, r["cands"][nm]) for nm in names if nm in r["cands"] and r["cands"][nm].get("text")]
        idx = consensus_idx([c["text"] for _, c in items])
        return items[idx][1]["f1"], items[idx][0]

    def sel_conf_commit(r, margin):  # replace base only if a cand beats base conf by margin
        bconf = r["cands"]["base"].get("conf")
        best_nm, best_conf = "base", bconf if bconf is not None else -1e9
        for nm in names:
            if nm == "base":
                continue
            c = r["cands"].get(nm, {})
            if c.get("conf") is not None and c.get("text") and c["conf"] > best_conf + margin:
                best_conf, best_nm = c["conf"], nm
        return r["cands"][best_nm]["f1"], best_nm

    print("\n=== GT-free selectors (mean F1, delta vs base, accept breakdown) ===")
    print(f"  {'selector':22s}{'F1':>7}{'dvsBase':>9}   good/bad/neut (vs base)")
    def report(name, picker):
        picks = [picker(r) for r in rows]
        m = 100 * sum(p[0] for p in picks) / n
        good = bad = neut = 0
        for r, (fv, nm) in zip(rows, picks):
            b = r["cands"]["base"]["f1"]
            if nm == "base":
                neut += 1
            elif fv > b + 1e-9:
                good += 1
            elif fv < b - 1e-9:
                bad += 1
            else:
                neut += 1
        print(f"  {name:22s}{m:7.2f}{m-base_mean:+9.2f}   {good}/{bad}/{neut}")
        return m

    report("confidence-argmax", sel_conf)
    report("consensus", sel_consensus)
    for mg in (0.0, 0.05, 0.1, 0.2):
        report(f"conf-commit m={mg}", lambda r, mg=mg: sel_conf_commit(r, mg))

    # redescribe-always as a candidate final pipeline
    print(f"\n  redescribe-always delta vs base = {gen_means['redescribe']-base_mean:+.2f}")

    # stored base comparison (validation of media/base config)
    sb = [(r.get("stored_base_f1"), r["cands"]["base"]["f1"]) for r in rows if r.get("stored_base_f1") is not None]
    if sb:
        sm = 100 * sum(x[0] for x in sb) / len(sb)
        gm = 100 * sum(x[1] for x in sb) / len(sb)
        print(f"\n=== base config check (n={len(sb)} with stored base) ===")
        print(f"  stored_base F1={sm:.2f}   regenerated_base F1={gm:.2f}   diff={gm-sm:+.2f}")


if __name__ == "__main__":
    main()
