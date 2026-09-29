# -*- coding: utf-8 -*-
"""eval_unified.py -- unified scorer for all six OmniConfess benchmarks.

Routing by filename:
  cmm / phd / pubmedqa  ->  evaluate_judgment_tasks   (Categorical-F1)
  ragtruth / haloquest / avhbench  ->  evaluate_descriptive_tasks  (Token-F1)

Both methods auto-detect confession format fields (final_pred / repaired_pred)
and fall back to generic pred / response / output, so the same call works for
baseline and confession outputs alike.

Usage
-----
# score a folder of outputs (auto-routed by filename):
python evaluation/eval_unified.py --target_folder outputs/

# score specific files:
python evaluation/eval_unified.py --target_files outputs/haloquest.jsonl

# GAVIE scoring requires a Zhipu API key in $ZHIPU_API_KEY
"""
import os
import json
import re
import csv
import glob
import math
import random
import string
import numpy as np
from tqdm import tqdm
from collections import Counter
from sklearn.metrics import f1_score, recall_score

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "because", "as", "what",
    "which", "this", "that", "these", "those", "then", "just", "so", "than",
    "such", "both", "through", "about", "for", "is", "are", "was", "were",
    "of", "while", "during", "to", "in", "on", "at", "by", "with", "it", "there"
}


class UnifiedEvaluator:
    def __init__(self, zhipu_key=None):
        self._zhipu_key = zhipu_key
        self._client = None

    @property
    def client(self):
        if self._client is None and self._zhipu_key:
            from zhipuai import ZhipuAI
            self._client = ZhipuAI(api_key=self._zhipu_key)
        return self._client

    # ---- shared utilities ------------------------------------------------

    def _load_data(self, file_path):
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    m = re.search(r'(\{.*\})', line)
                    if m:
                        data.append(json.loads(m.group(1)))
                except Exception:
                    continue
        if not data:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content.startswith('['):
                        return json.loads(content)
            except Exception:
                pass
        return data

    def _get_value(self, item, keys):
        for k in keys:
            if k in item and item[k] is not None:
                return str(item[k])
        return ""

    @staticmethod
    def _bootstrap_ci(scores, n_boot=2000, alpha=0.05, seed=42):
        rng = random.Random(seed)
        n = len(scores)
        means = sorted(sum(rng.choices(scores, k=n)) / n for _ in range(n_boot))
        return means[int(alpha / 2 * n_boot)], means[int((1 - alpha / 2) * n_boot)]

    # ---- judgment tasks: CMM / PHD / PubMedQA ----------------------------
    # PubMedQA has a three-way label set (yes / no / maybe). MAYBE rows are
    # excluded from the binary metrics; a MAYBE prediction on a yes/no row
    # is scored as a miss.
    MAYBE = 2

    def _parse_yesno_answer(self, text):
        text = str(text).lower().strip()
        m = re.search(r'\\boxed\{(.*?)\}', text)
        if m:
            ans = m.group(1).strip()
            if 'maybe' in ans: return self.MAYBE
            return 1 if 'yes' in ans else 0
        if 'final answer: maybe' in text: return self.MAYBE
        if 'final answer: yes'  in text: return 1
        if 'final answer: no'   in text: return 0
        first = text.split('.')[0].split(',')[0].split()[0] if text.split() else ''
        if first == 'maybe': return self.MAYBE
        if first == 'yes':   return 1
        if first == 'no':    return 0
        tail = text[-30:]
        if 'maybe' in tail: return self.MAYBE
        if 'yes'   in tail: return 1
        if 'no'    in tail: return 0
        return 0

    def evaluate_judgment_tasks(self, file_path, save_single=False):
        data = self._load_data(file_path)
        name = os.path.basename(file_path)
        print(f"\n[eval] {name}")
        grouped = {"Overall": {"y_true": [], "y_pred": []}}
        scored, n_undecidable = [], 0

        for item in data:
            status = str(item.get('status', 'success')).lower()
            gt_val  = self._get_value(item, ['gt', 'answer', 'label', 'ground_truth'])
            pred_raw = self._get_value(item, ['pred', 'response', 'output'])
            gt_lower = gt_val.lower().strip()
            if gt_lower in ('yes',) or gt_lower.startswith(('yes.', 'yes,', 'yes ')):
                gt = 1
            elif gt_lower in ('no',) or gt_lower.startswith(('no.', 'no,', 'no ')):
                gt = 0
            else:
                gt = self._parse_yesno_answer(gt_val)
            pred = self._parse_yesno_answer(pred_raw)
            group_key = self._get_value(item, ['sub_category', 'task']) or "Default"
            undecidable = (gt == self.MAYBE)
            item["eval_metrics"] = {
                "Ground_Truth": gt, "Prediction": pred,
                "Is_Correct": 0 if (undecidable or gt != pred) else 1,
                "Group": group_key, "Status": status,
                "Undecidable": int(undecidable),
            }
            scored.append(item)
            if status == "success" and not undecidable:
                pred_bin = (1 - gt) if pred == self.MAYBE else pred
                grouped["Overall"]["y_true"].append(gt)
                grouped["Overall"]["y_pred"].append(pred_bin)
                if group_key not in grouped:
                    grouped[group_key] = {"y_true": [], "y_pred": []}
                grouped[group_key]["y_true"].append(gt)
                grouped[group_key]["y_pred"].append(pred_bin)
            elif undecidable:
                n_undecidable += 1

        if n_undecidable:
            print(f"[eval] {n_undecidable} row(s) excluded: ground truth is 'maybe'.")

        if save_single:
            out = file_path.rsplit('.', 1)[0] + "_single_scored.jsonl"
            with open(out, 'w', encoding='utf-8') as f:
                for d in scored:
                    f.write(json.dumps(d, ensure_ascii=False) + '\n')
            print(f"[ok] per-sample results -> {out}")

        results = []
        for cat in sorted(grouped, key=lambda x: (x != "Overall", x)):
            yt, yp = grouped[cat]["y_true"], grouped[cat]["y_pred"]
            if not yt:
                continue
            r_yes = recall_score(yt, yp, pos_label=1, zero_division=0)
            r_no  = recall_score(yt, yp, pos_label=0, zero_division=0)
            phd   = (2 * r_yes * r_no) / (r_yes + r_no) if (r_yes + r_no) > 0 else 0
            results.append({
                "File Name": f"{name} [{cat}]",
                "Samples": len(yt),
                "Categorical-F1": round(f1_score(yt, yp, zero_division=0) * 100, 2),
                "PhD-Index": round(phd * 100, 2),
            })
        return results

    # ---- descriptive tasks: RAGTruth / HaloQuest / AVHBench --------------

    def _compute_token_f1(self, pred, gt):
        def tokens(s):
            s = str(s).lower().translate(str.maketrans('', '', string.punctuation))
            return [w for w in s.split() if w not in STOPWORDS and w.strip()]
        p_t, g_t = tokens(pred), tokens(gt)
        if not p_t or not g_t:
            return 0.0
        common = Counter(p_t) & Counter(g_t)
        hit = sum(common.values())
        p = hit / len(p_t)
        r = hit / len(g_t)
        return (2 * p * r) / (p + r) if (p + r) > 0 else 0.0

    def _get_gavie_score(self, gt, pred, timeout_sec=25, debug=False):
        prompt = (
            f"Reference: {gt}\nModel Prediction: {pred}\n"
            "Score the prediction (0.0-10.0) on:\n"
            "1. relevancy: Main event coverage.\n"
            "2. accuracy: Detail correctness.\n"
            "Use decimal points for partial correctness (e.g. 7.4). "
            "Output strictly in this format: relevancy: X.X, accuracy: Y.Y"
        )
        import concurrent.futures
        def _call():
            judge = os.environ.get("ZHIPU_JUDGE_MODEL", "glm-4")
            return self.client.chat.completions.create(
                model=judge,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                res = ex.submit(_call).result(timeout=timeout_sec)
            content = res.choices[0].message.content.lower()
            if debug:
                print(f"   [judge]: {content.strip()}")
            rel = re.search(r'relevancy:\s*(\d+(?:\.\d+)?)', content)
            acc = re.search(r'accuracy:\s*(\d+(?:\.\d+)?)', content)
            r = min(float(rel.group(1)), 10.0) if rel else -1.0
            a = min(float(acc.group(1)), 10.0) if acc else -1.0
            return r, a
        except Exception as e:
            if debug:
                print(f"   [API error]: {e}")
            return -1.0, -1.0

    def evaluate_descriptive_tasks(self, file_path, save_single=False):
        """Token-F1 (+ optional GAVIE) for RAGTruth, HaloQuest, and AVHBench.

        Confession outputs carry base_pred / repaired_pred; a bootstrap CI and
        edited-subset delta are reported automatically when base_pred is present.
        Generic outputs use pred / response / output and report Token-F1 only.
        """
        data = self._load_data(file_path)
        name = os.path.basename(file_path)
        is_confession = any('base_pred' in r for r in data)

        base_f1s, rep_f1s, edited_flags = [], [], []
        rel_scores, acc_scores = [], []
        scored = []

        for idx, item in enumerate(tqdm(data, desc=f"Token-F1 ({name})")):
            status = str(item.get('status', 'success')).lower()
            gt = self._get_value(item, ['gt', 'ground_truth', 'answer'])

            if is_confession:
                base_p = self._get_value(item, ['base_pred'])
                rep_p  = self._get_value(item, ['repaired_pred', 'final_pred']) or base_p
            else:
                base_p = rep_p = self._get_value(
                    item, ['final_pred', 'repaired_pred', 'pred', 'response', 'output'])

            bf = self._compute_token_f1(base_p, gt)
            rf = self._compute_token_f1(rep_p,  gt)
            edited = bool(item.get('edited', False))

            item_rel, item_acc = -1.0, -1.0
            if self.client:
                item_rel, item_acc = self._get_gavie_score(gt, rep_p, debug=(idx < 5))

            item["eval_metrics"] = {
                "Token-F1": round(rf * 100, 2),
                "GAVIE-Rel": item_rel,
                "GAVIE-Acc": item_acc,
                "Status": status,
            }
            scored.append(item)

            if status == "success":
                base_f1s.append(bf)
                rep_f1s.append(rf)
                edited_flags.append(edited)
                if item_rel >= 0:
                    rel_scores.append(item_rel)
                    acc_scores.append(item_acc)

        if save_single:
            out = file_path.rsplit('.', 1)[0] + "_single_scored.jsonl"
            with open(out, 'w', encoding='utf-8') as f:
                for d in scored:
                    f.write(json.dumps(d, ensure_ascii=False) + '\n')
            print(f"[ok] per-sample results -> {out}")

        n = len(rep_f1s)
        mean_rep  = 100 * sum(rep_f1s)  / n if n else 0.0
        mean_base = 100 * sum(base_f1s) / n if n else 0.0
        result = {
            "File Name": name,
            "Samples":   n,
            "Token-F1":  round(mean_rep, 2),
            "GAVIE-Rel": round(float(np.mean(rel_scores)), 2) if rel_scores else 0.0,
            "GAVIE-Acc": round(float(np.mean(acc_scores)), 2) if acc_scores else 0.0,
        }

        if is_confession and n:
            delta = mean_rep - mean_base
            per   = [100 * (r - b) for r, b in zip(rep_f1s, base_f1s)]
            lo, hi = self._bootstrap_ci(per)
            result.update({
                "base Token-F1": round(mean_base, 2),
                "Delta": round(delta, 2),
                "CI-lo": round(lo, 2),
                "CI-hi": round(hi, 2),
                "Edited": sum(edited_flags),
            })
            # edited-subset delta
            eb = [b for b, e in zip(base_f1s, edited_flags) if e]
            er = [r for r, e in zip(rep_f1s,  edited_flags) if e]
            if eb:
                ed_delta = 100 * sum(er) / len(er) - 100 * sum(eb) / len(eb)
                lo_ed, hi_ed = self._bootstrap_ci(
                    [100 * (r - b) for r, b in zip(er, eb)])
                result["Delta-edited"] = round(ed_delta, 2)
                result["CI-lo-edited"] = round(lo_ed, 2)
                result["CI-hi-edited"] = round(hi_ed, 2)

        return [result]


# ---- CLI -----------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Unified scorer for OmniConfess outputs.")
    parser.add_argument('--target_folder', type=str, default=None,
                        help="Score every file in this folder (auto-routed by name).")
    parser.add_argument('--target_files', type=str, nargs='+', default=None,
                        help="Score one or more specific files.")
    args = parser.parse_args()

    if not args.target_folder and not args.target_files:
        parser.error("provide --target_folder or --target_files")

    evaluator = UnifiedEvaluator(zhipu_key=os.environ.get("ZHIPU_API_KEY"))

    files = (args.target_files or
             glob.glob(os.path.join(args.target_folder, "*.json*")))

    all_reports: dict = {}

    for f in files:
        if "_scored" in f:
            continue
        n = os.path.basename(f).lower()
        if   any(k in n for k in ("cmm",)):       mode, cat = "judgement", "cmm"
        elif any(k in n for k in ("phd",)):        mode, cat = "judgement", "phd"
        elif any(k in n for k in ("pubmedqa",)):   mode, cat = "judgement", "pubmedqa"
        elif any(k in n for k in ("ragtruth",)):   mode, cat = "descriptive", "ragtruth"
        elif any(k in n for k in ("halo",)):       mode, cat = "descriptive", "haloquest"
        elif any(k in n for k in ("avh",)):        mode, cat = "descriptive", "avhbench"
        else:
            continue

        key = (mode, cat)
        all_reports.setdefault(key, [])
        try:
            if mode == "judgement":
                rows = evaluator.evaluate_judgment_tasks(f, save_single=True)
            else:
                rows = evaluator.evaluate_descriptive_tasks(f, save_single=True)
            all_reports[key].extend(rows)
        except Exception as e:
            print(f"[error] {f}: {e}")

    for (mode, cat), rows in all_reports.items():
        if not rows:
            continue
        fname = f"{mode}_{cat}.csv"
        if mode == "judgement":
            headers = ["File Name", "Samples", "Categorical-F1", "PhD-Index"]
        else:
            # write all keys present across rows
            headers = list(dict.fromkeys(k for r in rows for k in r))
        with open(fname, 'a', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=headers, extrasaction='ignore')
            if fh.tell() == 0:
                w.writeheader()
            w.writerows(rows)
        print(f"[ok] {fname}")

