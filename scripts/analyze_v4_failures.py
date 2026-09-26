#!/usr/bin/env python3
"""
Failure mode analysis for lv2_avh_v4/audio_only_n50.jsonl
"""
import json
import re
import statistics
from collections import Counter, defaultdict

JSONL_PATH = "/home/tsq/rhq/omniconfess/OmniConfess/outputs/lv2_avh_v4/audio_only_n50.jsonl"
OUT_PATH   = "/home/tsq/rhq/omniconfess/OmniConfess/outputs/lv2_avh_v4/FAILURE_ANALYSIS.md"

# ── helpers ──────────────────────────────────────────────────────────────────

def get_step1_obs(row):
    """Extract step1 observation text from the first candidate's verify_raw."""
    cands = row.get("candidates", [])
    for c in cands:
        vr = c.get("verify_raw", "")
        m = re.search(r'\[step1=(.*?)\]', vr, re.DOTALL)
        if m:
            return m.group(1).strip()
    return ""

def is_vague_obs(obs):
    """Return True if the observation is non-informative."""
    if len(obs) < 30:
        return True
    vague_phrases = [
        "not supported", "cannot verify", "no visual", "no evidence",
        "does not provide", "cannot determine", "unable to", "not visible",
        "no information", "not mentioned", "not confirmed",
    ]
    obs_lower = obs.lower()
    return any(p in obs_lower for p in vague_phrases)

def is_specific_obs(obs):
    """Return True if obs mentions a concrete correctable fact."""
    specific_patterns = [
        r'\b(color|colour|red|blue|green|yellow|white|black|brown|orange|purple|pink|grey|gray)\b',
        r'\b(sound|noise|bark|meow|engine|roar|buzz|hum|music|speech|voice|siren|bird|wind|rain)\b',
        r'\b(presence|absence|visible|not visible|appear|disappear|exist)\b',
        r'\b(person|man|woman|child|dog|cat|car|truck|helicopter|bird|snake|tree|building)\b',
        r'\b(left|right|front|back|top|bottom|behind|beside|near|next to)\b',
        r'\b(number|count|one|two|three|four|five|single|multiple|several)\b',
        r'\b(moving|walking|running|flying|sitting|standing|swimming)\b',
    ]
    obs_lower = obs.lower()
    return any(re.search(p, obs_lower) for p in specific_patterns)

def span_length_group(span_text):
    words = span_text.strip().split()
    n = len(words)
    if n <= 3:
        return "short(1-3)"
    elif n <= 8:
        return "medium(4-8)"
    else:
        return "long(>8)"

# ── load data ────────────────────────────────────────────────────────────────

with open(JSONL_PATH) as f:
    rows = [json.loads(l) for l in f]

# ── per-row categorization ────────────────────────────────────────────────────

def categorize(row):
    base_f1      = row["base_f1"]
    cand_f1      = row["candidate_f1"]          # best candidate
    repaired_f1  = row["repaired_f1"]
    accepted     = row.get("lv2_regen_accepted", False)
    a_scene_ok   = row.get("candidate_A_scene_ok", True)
    b2_scene_ok  = row.get("candidate_B2_scene_ok", True)

    # scene_replace: any scene_ok=False AND candidate degraded
    scene_bad = (not a_scene_ok) or (not b2_scene_ok)
    if scene_bad and cand_f1 < base_f1:
        return "SCENE_REPLACE"

    # no_edit: no candidates selected (n_spans=0 or all decisions not REVISION_CANDIDATE)
    cands = row.get("candidates", [])
    revision_cands = [c for c in cands if c.get("decision") == "REVISION_CANDIDATE"]
    if len(revision_cands) == 0:
        return "NO_SPAN"

    delta = cand_f1 - base_f1

    if abs(delta) < 1.0:
        return "MARGINAL"

    if delta > 1.0 and accepted:
        return "TRUE_IMPROVEMENT"

    if delta > 1.0 and not accepted:
        return "CORRECT_NOT_ACCEPTED"

    if delta < -1.0 and accepted:
        return "ACCEPTED_DEGRADED"

    if delta < -1.0 and not accepted:
        return "WRONG_DIRECTION"

    return "MARGINAL"

categories = [categorize(r) for r in rows]
cat_counts = Counter(categories)

# ── step1_obs quality ─────────────────────────────────────────────────────────

obs_list = [get_step1_obs(r) for r in rows]
vague_count    = sum(1 for o in obs_list if is_vague_obs(o))
specific_count = sum(1 for o in obs_list if not is_vague_obs(o) and is_specific_obs(o))
other_count    = len(obs_list) - vague_count - specific_count

# ── B2 vs A comparison ────────────────────────────────────────────────────────

b2_rows = [r for r in rows if "candidate_B2_f1" in r]
b2_wins = 0
a_wins  = 0
ties    = 0
b2_wins_and_scene_ok = []

for r in b2_rows:
    af1  = r["candidate_A_f1"]
    b2f1 = r["candidate_B2_f1"]
    b2_ok = r.get("candidate_B2_scene_ok", True)
    a_ok  = r.get("candidate_A_scene_ok", True)
    if b2f1 > af1:
        b2_wins += 1
        if b2_ok:
            b2_wins_and_scene_ok.append((af1, b2f1))
    elif af1 > b2f1:
        a_wins += 1
    else:
        ties += 1

b2_scene_ok_count = sum(1 for r in b2_rows if r.get("candidate_B2_scene_ok", True))
a_scene_ok_count  = sum(1 for r in b2_rows if r.get("candidate_A_scene_ok",  True))

mean_b2_f1_when_b2wins = (
    statistics.mean(f2 for _, f2 in b2_wins_and_scene_ok)
    if b2_wins_and_scene_ok else float("nan")
)
mean_a_f1_when_b2wins = (
    statistics.mean(f1 for f1, _ in b2_wins_and_scene_ok)
    if b2_wins_and_scene_ok else float("nan")
)

# ── accept gate impact ────────────────────────────────────────────────────────

def best_candidate_f1(row):
    """Max F1 across A and B2 candidates."""
    f1s = [row["candidate_A_f1"]]
    if "candidate_B2_f1" in row:
        f1s.append(row["candidate_B2_f1"])
    return max(f1s)

virtual_oracle_f1s = [max(best_candidate_f1(r), r["base_f1"]) for r in rows]
actual_oracle_f1s  = [max(r["candidate_f1"], r["base_f1"]) for r in rows]
repaired_f1s       = [r["repaired_f1"] for r in rows]

virtual_oracle_mean = statistics.mean(virtual_oracle_f1s)
actual_oracle_mean  = statistics.mean(actual_oracle_f1s)
repaired_mean       = statistics.mean(repaired_f1s)
base_mean           = statistics.mean(r["base_f1"] for r in rows)

# Good candidates rejected: best_candidate_f1 > base+2 but not accepted
good_rejected = [
    r for r in rows
    if best_candidate_f1(r) > r["base_f1"] + 2.0 and not r.get("lv2_regen_accepted", False)
]

# ── span text quality ─────────────────────────────────────────────────────────

span_groups = defaultdict(list)   # group -> list of (delta_f1, span_text)

for r in rows:
    base_f1 = r["base_f1"]
    cand_f1 = r["candidate_f1"]
    delta   = cand_f1 - base_f1
    # use the span_text of the first REVISION_CANDIDATE
    cands = r.get("candidates", [])
    rev_cands = [c for c in cands if c.get("decision") == "REVISION_CANDIDATE"]
    for c in rev_cands[:1]:
        st = c["span_text"].strip()
        grp = span_length_group(st)
        span_groups[grp].append((delta, st))

span_summary = {}
for grp, items in span_groups.items():
    deltas = [d for d, _ in items]
    span_summary[grp] = {
        "n": len(items),
        "mean_delta": statistics.mean(deltas),
        "pct_positive": 100 * sum(1 for d in deltas if d > 0) / len(deltas),
    }

# ── top-5 improved / worst-5 ──────────────────────────────────────────────────

rows_with_delta = [(r, r["candidate_f1"] - r["base_f1"]) for r in rows]
rows_sorted = sorted(rows_with_delta, key=lambda x: x[1], reverse=True)
top5    = rows_sorted[:5]
worst5  = rows_sorted[-5:]

def summarize_row(r, delta):
    cands = r.get("candidates", [])
    rev_cands = [c for c in cands if c.get("decision") == "REVISION_CANDIDATE"]
    span_text = rev_cands[0]["span_text"].strip() if rev_cands else "(none)"
    obs       = get_step1_obs(r)[:120]
    return {
        "id":        r["id"],
        "base_f1":   r["base_f1"],
        "cand_f1":   r["candidate_f1"],
        "delta":     delta,
        "accepted":  r.get("lv2_regen_accepted", False),
        "path":      r.get("prompt_path", "?"),
        "span_text": span_text,
        "obs":       obs,
        "a_scene_ok": r.get("candidate_A_scene_ok", True),
        "b2_scene_ok": r.get("candidate_B2_scene_ok", "N/A"),
    }

top5_summary    = [summarize_row(r, d) for r, d in top5]
worst5_summary  = [summarize_row(r, d) for r, d in worst5]

# ── build report ──────────────────────────────────────────────────────────────

lines = []
A = lines.append

A("# v4 Audio-Only N=50 Failure Analysis")
A("")
A(f"**File:** `{JSONL_PATH}`  ")
A(f"**N rows:** {len(rows)}  ")
A(f"**Rows with B2:** {len(b2_rows)} / {len(rows)}  ")
A(f"**Prompt path distribution:** A={sum(1 for r in rows if r.get('prompt_path')=='A')}, A+B2={sum(1 for r in rows if r.get('prompt_path')=='A+B2')}")
A("")

# ── Section 1: F1 summary ──
A("## 1. F1 Summary")
A("")
A(f"| Metric | Mean F1 |")
A(f"|--------|---------|")
A(f"| base_f1 | {base_mean:.3f} |")
A(f"| candidate_f1 (best candidate per row) | {statistics.mean(r['candidate_f1'] for r in rows):.3f} |")
A(f"| repaired_f1 (actually accepted) | {repaired_mean:.3f} |")
A(f"| actual oracle (max of cand, base) | {actual_oracle_mean:.3f} |")
A(f"| virtual oracle (always-accept best) | {virtual_oracle_mean:.3f} |")
A("")

# ── Section 2: Failure categories ──
A("## 2. Failure Categories")
A("")
A("Categories assigned per-row (mutually exclusive, priority order):")
A("")
A(f"| Category | Count | % |")
A(f"|----------|-------|---|")
total = len(rows)
for cat in ["TRUE_IMPROVEMENT", "CORRECT_NOT_ACCEPTED", "ACCEPTED_DEGRADED",
            "WRONG_DIRECTION", "SCENE_REPLACE", "NO_SPAN", "MARGINAL"]:
    n = cat_counts.get(cat, 0)
    A(f"| {cat} | {n} | {100*n/total:.1f}% |")
A("")
A("**Definitions:**")
A("- `TRUE_IMPROVEMENT`: candidate_f1 > base_f1 + 1 AND accepted → genuine win")
A("- `CORRECT_NOT_ACCEPTED`: candidate_f1 > base_f1 + 1 but gate rejected → gate is the bottleneck")
A("- `ACCEPTED_DEGRADED`: accepted but repaired_f1 < base_f1 → harmful acceptance")
A("- `WRONG_DIRECTION`: candidate_f1 < base_f1 - 1 AND scene_ok → model corrected wrong span")
A("- `SCENE_REPLACE`: scene_ok=False on A or B2 AND candidate worse → hallucinated scene swap")
A("- `NO_SPAN`: no REVISION_CANDIDATE spans found → verifier found nothing")
A("- `MARGINAL`: |delta| < 1.0 → too small to matter")
A("")

# ── Section 3: step1_obs quality ──
A("## 3. step1_obs Quality")
A("")
A(f"| Quality tier | Count | % |")
A(f"|-------------|-------|---|")
A(f"| Vague / non-informative (<30 chars or generic refusal language) | {vague_count} | {100*vague_count/len(rows):.1f}% |")
A(f"| Specific (mentions concrete correctable fact) | {specific_count} | {100*specific_count/len(rows):.1f}% |")
A(f"| Descriptive but no clear correctable target | {other_count} | {100*other_count/len(rows):.1f}% |")
A("")
A("Sample vague observations:")
vague_samples = [(obs, i) for i, obs in enumerate(obs_list) if is_vague_obs(obs)][:5]
for obs, i in vague_samples:
    A(f"- Row {i}: `{obs[:100]}`")
A("")
A("Sample specific observations:")
specific_samples = [(obs, i) for i, obs in enumerate(obs_list) if not is_vague_obs(obs) and is_specific_obs(obs)][:5]
for obs, i in specific_samples:
    A(f"- Row {i}: `{obs[:120]}`")
A("")

# ── Section 4: B2 vs A ──
A("## 4. B2 vs A Candidate Comparison")
A("")
A(f"Rows with B2: {len(b2_rows)} / {len(rows)} (prompt_path=A+B2)")
A("")
A(f"| Outcome | Count |")
A(f"|---------|-------|")
A(f"| B2 wins (B2_f1 > A_f1) | {b2_wins} |")
A(f"| A wins (A_f1 > B2_f1) | {a_wins} |")
A(f"| Tie | {ties} |")
A("")
A(f"scene_ok rates among B2 rows: A scene_ok={a_scene_ok_count}/{len(b2_rows)} ({100*a_scene_ok_count/len(b2_rows):.1f}%), B2 scene_ok={b2_scene_ok_count}/{len(b2_rows)} ({100*b2_scene_ok_count/len(b2_rows):.1f}%)")
A("")
if b2_wins_and_scene_ok:
    A(f"When B2 wins AND B2 is scene_ok ({len(b2_wins_and_scene_ok)} cases):")
    A(f"- Mean B2_f1: {mean_b2_f1_when_b2wins:.3f}")
    A(f"- Mean A_f1 (same rows): {mean_a_f1_when_b2wins:.3f}")
    A(f"- Mean B2 gain over A: {mean_b2_f1_when_b2wins - mean_a_f1_when_b2wins:.3f}")
A("")

# ── Section 5: Accept gate impact ──
A("## 5. Accept Gate Impact")
A("")
A(f"| F1 Measure | Value |")
A(f"|-----------|-------|")
A(f"| mean base_f1 | {base_mean:.3f} |")
A(f"| mean repaired_f1 (actual system) | {repaired_mean:.3f} |")
A(f"| mean actual oracle (max of cand, base) | {actual_oracle_mean:.3f} |")
A(f"| mean virtual oracle (always-accept best) | {virtual_oracle_mean:.3f} |")
A(f"| gate ceiling gap (virtual - actual oracle) | {virtual_oracle_mean - actual_oracle_mean:.3f} |")
A(f"| actual accept gain over base | {repaired_mean - base_mean:.3f} |")
A(f"| virtual oracle gain over base | {virtual_oracle_mean - base_mean:.3f} |")
A("")
A(f"Rows with a good candidate (best_cand_f1 > base+2) that was NOT accepted: **{len(good_rejected)}**")
if good_rejected:
    A("")
    A("| Row id | base_f1 | best_cand_f1 | delta | path |")
    A("|--------|---------|-------------|-------|------|")
    for r in good_rejected:
        bcf1 = best_candidate_f1(r)
        A(f"| {r['id']} | {r['base_f1']:.2f} | {bcf1:.2f} | +{bcf1-r['base_f1']:.2f} | {r.get('prompt_path','?')} |")
A("")

# ── Section 6: Span text quality ──
A("## 6. Span Text Quality")
A("")
A(f"| Span length group | n | mean delta_f1 | % positive delta |")
A(f"|------------------|---|--------------|-----------------|")
for grp in ["short(1-3)", "medium(4-8)", "long(>8)"]:
    if grp in span_summary:
        s = span_summary[grp]
        A(f"| {grp} | {s['n']} | {s['mean_delta']:.3f} | {s['pct_positive']:.1f}% |")
A("")
# Sample span texts per group
A("Sample span texts by group:")
for grp in ["short(1-3)", "medium(4-8)", "long(>8)"]:
    samples = [(d, st) for d, st in span_groups.get(grp, [])][:5]
    if samples:
        A(f"\n**{grp}:**")
        for d, st in samples:
            A(f"  - `{st}` (delta={d:+.2f})")
A("")

# ── Section 7: Top-5 improved ──
A("## 7. Top-5 Most Improved Samples")
A("")
for s in top5_summary:
    A(f"### Row {s['id']} — delta={s['delta']:+.2f} (base={s['base_f1']:.2f} → cand={s['cand_f1']:.2f})")
    A(f"- **accepted:** {s['accepted']}  **path:** {s['path']}")
    A(f"- **span_text:** `{s['span_text']}`")
    A(f"- **scene_ok A:** {s['a_scene_ok']}  **scene_ok B2:** {s['b2_scene_ok']}")
    A(f"- **step1_obs:** _{s['obs']}_")
    A("")

# ── Section 8: Worst-5 ──
A("## 8. Top-5 Most Degraded Samples")
A("")
for s in worst5_summary:
    A(f"### Row {s['id']} — delta={s['delta']:+.2f} (base={s['base_f1']:.2f} → cand={s['cand_f1']:.2f})")
    A(f"- **accepted:** {s['accepted']}  **path:** {s['path']}")
    A(f"- **span_text:** `{s['span_text']}`")
    A(f"- **scene_ok A:** {s['a_scene_ok']}  **scene_ok B2:** {s['b2_scene_ok']}")
    A(f"- **step1_obs:** _{s['obs']}_")
    # diagnosis
    r_orig = rows[s['id']]
    if not r_orig.get('candidate_A_scene_ok', True) or not r_orig.get('candidate_B2_scene_ok', True):
        diag = "Scene replacement failure: scene_ok=False indicates hallucianted context swap."
    elif s['delta'] < -1 and not s['accepted']:
        diag = "WRONG_DIRECTION: model corrected a span but moved F1 away from GT."
    elif s['delta'] < -1 and s['accepted']:
        diag = "ACCEPTED_DEGRADED: gate accepted a harmful revision."
    else:
        diag = "Marginal or unclear degradation."
    A(f"- **diagnosis:** {diag}")
    A("")

# ── Section 9: Summary ──
A("## 9. Key Findings")
A("")
A(f"1. **Gate is the primary bottleneck**: {len(good_rejected)} rows have a good candidate (delta>+2) that was rejected. "
  f"Virtual oracle would be {virtual_oracle_mean:.3f} vs actual oracle {actual_oracle_mean:.3f} "
  f"(gap={virtual_oracle_mean-actual_oracle_mean:.3f}).")
dominant_cat = cat_counts.most_common(1)[0]
A(f"2. **Dominant failure mode**: `{dominant_cat[0]}` ({dominant_cat[1]}/{len(rows)} rows = {100*dominant_cat[1]/len(rows):.1f}%).")
A(f"3. **B2 vs A**: B2 wins F1 on {b2_wins}/{len(b2_rows)} B2-rows ({100*b2_wins/len(b2_rows):.1f}%), "
  f"A wins on {a_wins} ({100*a_wins/len(b2_rows):.1f}%), ties {ties}.")
A(f"4. **step1_obs quality**: {vague_count}/{len(rows)} observations are vague/non-informative. "
  f"Only {specific_count} mention a specific correctable fact.")
if "short(1-3)" in span_summary and "medium(4-8)" in span_summary:
    s_s = span_summary["short(1-3)"]
    s_m = span_summary["medium(4-8)"]
    A(f"5. **Span length**: short spans (mean delta={s_s['mean_delta']:.2f}, {s_s['pct_positive']:.0f}% positive) "
      f"vs medium spans (mean delta={s_m['mean_delta']:.2f}, {s_m['pct_positive']:.0f}% positive).")
A(f"6. **Accepted repairs**: {sum(1 for r in rows if r.get('lv2_regen_accepted'))}/50 accepted; "
  f"net gain over base = {repaired_mean - base_mean:+.3f} F1.")
A("")

# ── write output ──────────────────────────────────────────────────────────────

report = "\n".join(lines)
with open(OUT_PATH, "w") as f:
    f.write(report)

print(report)

# ── feishu summary values ──────────────────────────────────────────────────────
print(f"\n\n=== FEISHU SUMMARY VALUES ===")
print(f"virtual_oracle={virtual_oracle_mean:.2f}")
print(f"B2 wins A: {b2_wins} times")
