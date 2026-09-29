#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eval_dualchannel.py — offline analysis of avh_confess_dualchannel shards.

Computes, over the pooled records (GT used ONLY for scoring):
  * mean canonical Token-F1 for: base, pipeline final (accept-gated rewrite),
    rewrite-always, and a pure programmatic-deletion variant (delete every
    unsupported suspect word from base — the most surgical confession edit).
  * accept breakdown (good/bad/neutral vs base) for final and delete.
  * candidate oracle = max(base, rewrite, delete) per sample (ceiling).
  * sign test p-value on the accepted-edit subset.
"""
import glob, json, string, sys
from collections import Counter

STOP = set("a an the and or but if because as what which this that these those "
           "then just so than such both through about for is are was were of while "
           "during to in on at by with it there".split())


def _toks(s):
    s = str(s).lower().translate(str.maketrans("", "", string.punctuation))
    return [w for w in s.split() if w and w not in STOP]


def token_f1(pred, gt):
    p, g = _toks(pred), _toks(gt)
    if not p or not g:
        return 0.0
    c = Counter(p) & Counter(g); h = sum(c.values())
    if h == 0:
        return 0.0
    P, R = h / len(p), h / len(g)
    return 2 * P * R / (P + R)


def delete_variant(base, suspects):
    """Remove every unsupported suspect word from base (case-insensitive, whole
    word). Minimal cleanup of doubled spaces / dangling 'and'/',' artifacts."""
    bad = {s["word"].lower() for s in suspects
           if s.get("verdict") in ("CONTRADICTED", "UNSUPPORTED")}
    if not bad:
        return base
    out = []
    for w in base.split():
        core = w.lower().strip(string.punctuation)
        if core in bad:
            continue
        out.append(w)
    txt = " ".join(out)
    # light cleanup
    txt = txt.replace(" ,", ",").replace(" .", ".")
    while "  " in txt:
        txt = txt.replace("  ", " ")
    for junk in (" and ,", " , and", "and and"):
        txt = txt.replace(junk, " ")
    return txt.strip(" ,")


def sign_test(deltas, eps=1e-9):
    pos = sum(1 for d in deltas if d > eps)
    neg = sum(1 for d in deltas if d < -eps)
    n = pos + neg
    if n == 0:
        return pos, neg, 1.0
    # two-sided exact binomial at p=0.5
    from math import comb
    k = min(pos, neg)
    p = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n) * 2
    return pos, neg, min(p, 1.0)


def main():
    patt = sys.argv[1] if len(sys.argv) > 1 else "outputs/avh_dualchannel/dev_shard*of*.jsonl"
    rows = {}
    for fp in sorted(glob.glob(patt)):
        for line in open(fp):
            line = line.strip()
            if line:
                r = json.loads(line)
                rows[r["id"]] = r
    rows = list(rows.values())
    n = len(rows)
    if n == 0:
        print(f"no records for {patt}")
        return
    print(f"n={n}  ({patt})")

    base = [token_f1(r["base_pred"], r["gt"]) for r in rows]
    final = [token_f1(r["final"], r["gt"]) for r in rows]
    rewrite_always = [token_f1(r["rewrite"] or r["base_pred"], r["gt"]) for r in rows]
    delete = [token_f1(delete_variant(r["base_pred"], r["suspects"]), r["gt"]) for r in rows]
    oracle = [max(b, rw, dl) for b, rw, dl in zip(base, rewrite_always, delete)]

    bm = 100 * sum(base) / n
    def line(name, arr):
        m = 100 * sum(arr) / n
        print(f"  {name:20s} {m:6.2f}  ({m-bm:+.2f} vs base)")
    print("\n=== mean canonical Token-F1 (%) ===")
    print(f"  {'base':20s} {bm:6.2f}")
    line("pipeline final", final)
    line("rewrite-always", rewrite_always)
    line("delete-unsupported", delete)
    line("ORACLE(b/rw/del)", oracle)

    def breakdown(name, arr):
        good = bad = neut = 0
        deltas = []
        for b, x in zip(base, arr):
            d = x - b
            if d > 1e-9:
                good += 1; deltas.append(d)
            elif d < -1e-9:
                bad += 1; deltas.append(d)
            else:
                neut += 1
        pos, neg, p = sign_test(deltas)
        print(f"  {name:20s} good/bad/neut = {good}/{bad}/{neut}   sign-test p={p:.4f}")
    print("\n=== change breakdown vs base ===")
    breakdown("pipeline final", final)
    breakdown("delete-unsupported", delete)

    # decision + suspect stats
    dec = Counter(r["decision"] for r in rows)
    n_sus = sum(r["n_suspects"] for r in rows)
    n_uns = sum(r["n_unsupported"] for r in rows)
    edited = sum(1 for r in rows if r["decision"] == "accept_rewrite")
    print("\n=== pipeline stats ===")
    print(f"  decisions: {dict(dec)}")
    print(f"  total suspects={n_sus}  unsupported={n_uns}  accepted rewrites={edited}")
    # verify verdict distribution
    vv = Counter(s.get("verdict") for r in rows for s in r["suspects"])
    print(f"  verify verdicts: {dict(vv)}")
    # channel routing
    ch = Counter(s["channel"] for r in rows for s in r["suspects"])
    print(f"  suspect channels: {dict(ch)}")


if __name__ == "__main__":
    main()
