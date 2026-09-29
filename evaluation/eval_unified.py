import os
import json
import re
import csv
import glob
import math
import random
import numpy as np
import string
import io
import contextlib
from tqdm import tqdm
from collections import Counter
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from zhipuai import ZhipuAI

STOPWORDS = set([
    "a", "an", "the", "and", "or", "but", "if", "because", "as", "what", 
    "which", "this", "that", "these", "those", "then", "just", "so", "than", 
    "such", "both", "through", "about", "for", "is", "are", "was", "were", 
    "of", "while", "during", "to", "in", "on", "at", "by", "with", "it", "there"
])

class UnifiedEvaluator:
    def __init__(self, zhipu_key=None):
        self.client = ZhipuAI(api_key=zhipu_key) if zhipu_key else None

    # ================= [utilities] =================
    def _load_data(self, file_path):
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    json_match = re.search(r'(\{.*\})', line)
                    if json_match: data.append(json.loads(json_match.group(1)))
                except: continue
        if not data:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content.startswith('['): return json.loads(content)
            except: pass
        return data

    def _get_value(self, item, possible_keys):
        for k in possible_keys:
            if k in item and item[k] is not None: return str(item[k])
        return ""

    # ================= [1. judgement tasks (CMM/PhD)] =================
    # PubMedQA's label set is yes / no / maybe, so the parser needs a third
    # outcome. MAYBE is a sentinel rather than a class index: nothing downstream
    # can place it on a yes-or-no axis, so evaluate_judgment_tasks excludes rows
    # whose ground truth is MAYBE and counts a MAYBE prediction as a miss.
    # Without the branches below, "Maybe. <long answer>" fell through to the
    # last_chunk heuristic and was labelled from whatever 'yes' or 'no' happened
    # to appear in its final 30 characters.
    MAYBE = 2

    def _parse_yesno_answer(self, text):
        text = str(text).lower().strip()
        boxed_match = re.search(r'\\boxed\{(.*?)\}', text)
        if boxed_match:
            ans = boxed_match.group(1).strip()
            if 'maybe' in ans: return self.MAYBE
            return 1 if 'yes' in ans else 0
        if 'final answer: maybe' in text: return self.MAYBE
        if 'final answer: yes' in text: return 1
        if 'final answer: no' in text: return 0
        # Check first word (handles "Yes. ..." or "No, ..." style responses)
        first_word = text.split('.')[0].split(',')[0].split()[0] if text.split() else ''
        if first_word == 'maybe': return self.MAYBE
        if first_word == 'yes': return 1
        if first_word == 'no': return 0
        last_chunk = text[-30:]
        if 'maybe' in last_chunk: return self.MAYBE
        if 'yes' in last_chunk: return 1
        if 'no' in last_chunk: return 0
        return 0

    def evaluate_judgment_tasks(self, file_path, save_single=False):
        data = self._load_data(file_path)
        file_name = os.path.basename(file_path)
        
        print(f"\n[eval] file: {file_name}")
        grouped = {"Overall": {"y_true": [], "y_pred": []}}
        
        scored_data = []
        n_undecidable = 0

        for item in data:
            status = str(item.get('status', 'success')).lower()
            
            gt_val = self._get_value(item, ['gt', 'answer', 'label', 'ground_truth'])
            pred_raw = self._get_value(item, ['pred', 'response', 'output'])
            
            gt_lower = gt_val.lower().strip()
            if gt_lower == 'yes' or gt_lower.startswith('yes.') or gt_lower.startswith('yes,') or gt_lower.startswith('yes '):
                gt = 1
            elif gt_lower == 'no' or gt_lower.startswith('no.') or gt_lower.startswith('no,') or gt_lower.startswith('no '):
                gt = 0
            else:
                gt = self._parse_yesno_answer(gt_val)
            pred = self._parse_yesno_answer(pred_raw)
            group_key = self._get_value(item, ['sub_category', 'task']) or "Default"
            
            undecidable = (gt == self.MAYBE)
            item["eval_metrics"] = {
                "Ground_Truth": gt,
                "Prediction": pred,
                "Is_Correct": 0 if (undecidable or gt != pred) else 1,
                "Group": group_key,
                "Status": status,
                "Undecidable": 1 if undecidable else 0
            }
            scored_data.append(item)

            if status == "success" and not undecidable:
                # A MAYBE prediction has no place on the binary axis, so score it
                # as a miss by flipping it away from the ground truth.
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
            print(f"[eval] {n_undecidable} sample(s) excluded: ground truth is "
                  f"'maybe', which the yes/no metrics cannot score. PubMedQA is "
                  f"the only subset where this occurs.")

        if save_single:
            out_path = file_path.rsplit('.', 1)[0] + "_single_scored.jsonl"
            with open(out_path, 'w', encoding='utf-8') as f:
                for d in scored_data:
                    f.write(json.dumps(d, ensure_ascii=False) + '\n')
            print(f"[ok] per-sample results written to: {out_path}")

        results = []
        sorted_keys = sorted(grouped.keys(), key=lambda x: (x != "Overall", x))
        
        for cat in sorted_keys:
            val = grouped[cat]
            yt, yp = val["y_true"], val["y_pred"]
            
            if not yt:
                continue
                
            r_yes = recall_score(yt, yp, pos_label=1, zero_division=0)
            r_no = recall_score(yt, yp, pos_label=0, zero_division=0)
            phd_idx = (2 * r_yes * r_no) / (r_yes + r_no) if (r_yes + r_no) > 0 else 0
            
            results.append({
                "File Name": f"{file_name} [{cat}]",
                "Samples": len(yt),
                "Categorical-F1": round(f1_score(yt, yp, zero_division=0) * 100, 2),
                "PhD-Index": round(phd_idx * 100, 2)
            })
        
        return results

    # ================= [2. descriptive tasks (AVH/Halo)] =================
    def _compute_token_f1(self, pred, gt):
        def get_tokens(s):
            s = str(s).lower().translate(str.maketrans('', '', string.punctuation))
            return [w for w in s.split() if w not in STOPWORDS and w.strip()]
        
        p_t, g_t = get_tokens(pred), get_tokens(gt)
        if not p_t or not g_t: return 0.0
        common = Counter(p_t) & Counter(g_t)
        hit = sum(common.values())
        p = hit / len(p_t)
        r = hit / len(g_t)
        return (2 * p * r) / (p + r) if (p + r) > 0 else 0

    def _get_gavie_score(self, gt, pred, timeout_sec=25, debug=False):
        prompt = (
            f"Reference: {gt}\nModel Prediction: {pred}\n"
            "Score the prediction (0.0-10.0) on:\n"
            "1. relevancy: Main event coverage.\n"
            "2. accuracy: Detail correctness.\n"
            "Use decimal points for partial correctness (e.g. 7.4). Output strictly in this format: relevancy: X.X, accuracy: Y.Y"
        )
        import concurrent.futures
        def _call():
            # Judge model is configurable: the Zhipu coding-plan key only covers
            # glm-4.5-air / glm-4-flash, and cross-judge consistency is reported
            # separately, so runs must record which judge scored them.
            judge = os.environ.get("ZHIPU_JUDGE_MODEL", "glm-4")
            return self.client.chat.completions.create(model=judge, messages=[{"role": "user", "content": prompt}], temperature=0.1)
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(_call)
                res = future.result(timeout=timeout_sec)
            content = res.choices[0].message.content.lower()
            
            if debug:
                print(f"   [judge API raw reply]: {content.strip()}")

            rel_match = re.search(r'relevancy:\s*(\d+(?:\.\d+)?)', content)
            acc_match = re.search(r'accuracy:\s*(\d+(?:\.\d+)?)', content)
            r = float(rel_match.group(1)) if rel_match else -1.0
            a = float(acc_match.group(1)) if acc_match else -1.0
            if r > 10.0: r = 10.0
            if a > 10.0: a = 10.0
            return r, a
        except Exception as e:
            if debug:
                print(f"   [API error/timeout]: {e}")
            return -1.0, -1.0

    def evaluate_descriptive_tasks(self, file_path, save_single=False):
        data = self._load_data(file_path)
        metrics = {"f1": [], "rel": [], "acc": []}
        scored_data = []

        for idx, item in enumerate(tqdm(data, desc=f"NLP eval ({os.path.basename(file_path)})", disable=not save_single)):
            status = str(item.get('status', 'success')).lower()
            gt = self._get_value(item, ['gt', 'ground_truth', 'answer'])
            # confession outputs use final_pred / repaired_pred; generic outputs use pred
            pred = self._get_value(item, ['final_pred', 'repaired_pred', 'pred', 'response', 'output'])
            
            item_f1 = self._compute_token_f1(pred, gt)
            item_rel, item_acc = -1.0, -1.0
            
            if self.client:
                is_debug = (idx < 5)
                item_rel, item_acc = self._get_gavie_score(gt, pred, debug=is_debug)

            item["eval_metrics"] = {
                "Token-F1": round(item_f1 * 100, 2),
                "GAVIE-Rel": item_rel,
                "GAVIE-Acc": item_acc,
                "Status": status
            }
            scored_data.append(item)

            if status == "success":
                metrics["f1"].append(item_f1)
                if item_rel >= 0: 
                    metrics["rel"].append(item_rel)
                    metrics["acc"].append(item_acc)

        if save_single:
            out_path = file_path.rsplit('.', 1)[0] + "_single_scored.jsonl"
            with open(out_path, 'w', encoding='utf-8') as f:
                for d in scored_data:
                    f.write(json.dumps(d, ensure_ascii=False) + '\n')
            print(f"[ok] per-sample results written to: {out_path}")

        return [{
            "File Name": os.path.basename(file_path),
            "Samples": len(metrics["f1"]),
            "Token-F1": round(np.mean(metrics["f1"]) * 100, 2) if metrics["f1"] else 0.0,
            "GAVIE-Rel": round(np.mean(metrics["rel"]), 2) if metrics["rel"] else 0.0,
            "GAVIE-Acc": round(np.mean(metrics["acc"]), 2) if metrics["acc"] else 0.0
        }]
        # ================= stage 2: compute CIDEr globally =================
        individual_cider_scores = [0.0] * len(data)
        
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                g_tok = tokenizer.tokenize(g_dict)
                p_tok = tokenizer.tokenize(p_dict)
                if g_tok and p_tok and len(g_tok[0]) > 0 and len(p_tok[0]) > 0:
                    cider_scorer = Cider()
                    avg_score, scores_list = cider_scorer.compute_score(g_tok, p_tok)
                    cider_score = avg_score
                    individual_cider_scores = scores_list
            except Exception as e:
                print(f"\n[error] CIDEr computation failed: {e}")

        for idx, item in enumerate(scored_data):
            item["eval_metrics"]["CIDEr"] = round(float(individual_cider_scores[idx]), 4)

        # ================= stage 3: save and return =================
        if save_single:
            out_path = file_path.rsplit('.', 1)[0] + "_single_scored.jsonl"
            with open(out_path, 'w', encoding='utf-8') as f:
                for d in scored_data:
                    f.write(json.dumps(d, ensure_ascii=False) + '\n')
            print(f"[ok] per-sample results written to: {out_path}")

        if self.client and len(metrics["rel"]) == 0:
            print(f"\n[warn] the GAVIE score list is empty: every judge API request failed. Check the network or the account balance.")

        return [{
            "File Name": os.path.basename(file_path),
            "Samples": len(data),
            "Token-F1": round(np.mean(metrics["f1"]) * 100, 2) if metrics["f1"] else 0.0,
            "CIDEr": round(cider_score * 100, 2),
            "GAVIE-Rel": round(np.mean(metrics["rel"]), 2) if metrics["rel"] else 0.0,
            "GAVIE-Acc": round(np.mean(metrics["acc"]), 2) if metrics["acc"] else 0.0
        }]

    # ================= [3. HaloQuest confession evaluation] =================
    @staticmethod
    def _bootstrap_ci(scores, n_boot=2000, alpha=0.05, seed=42):
        rng = random.Random(seed)
        n = len(scores)
        means = [sum(rng.choices(scores, k=n)) / n for _ in range(n_boot)]
        means.sort()
        return means[int(alpha / 2 * n_boot)], means[int((1 - alpha / 2) * n_boot)]

    def evaluate_haloquest_confession(self, file_path, csv_out=False):
        """Evaluate a HaloQuest confession JSONL (base_pred vs repaired_pred).

        Reports Token-F1 for base and confession outputs, the delta with a
        95% bootstrap CI, and the edited-only subset delta. Mirrors the logic
        of the former standalone eval_haloquest.py.
        """
        from pathlib import Path
        path = Path(file_path)
        rows = [json.loads(l) for l in path.open(encoding='utf-8') if l.strip()]
        n_total = len(rows)

        base_f1s, rep_f1s, edited_flags, ids = [], [], [], []
        for r in rows:
            gt = r.get('gt', '')
            bp = r.get('base_pred', '')
            rp = r.get('repaired_pred', bp)
            base_f1s.append(self._compute_token_f1(bp, gt))
            rep_f1s.append(self._compute_token_f1(rp, gt))
            edited_flags.append(bool(r.get('edited', False)))
            ids.append(r.get('id', ''))

        n_edited = sum(edited_flags)
        mean_base = 100 * sum(base_f1s) / n_total
        mean_rep = 100 * sum(rep_f1s) / n_total
        delta = mean_rep - mean_base
        delta_per = [100 * (r - b) for r, b in zip(rep_f1s, base_f1s)]
        lo_d, hi_d = self._bootstrap_ci(delta_per)

        edited_base = [b for b, e in zip(base_f1s, edited_flags) if e]
        edited_rep = [r for r, e in zip(rep_f1s, edited_flags) if e]
        if edited_base:
            mean_base_ed = 100 * sum(edited_base) / len(edited_base)
            mean_rep_ed = 100 * sum(edited_rep) / len(edited_rep)
            delta_ed = mean_rep_ed - mean_base_ed
            lo_ed, hi_ed = self._bootstrap_ci(
                [100 * (r - b) for r, b in zip(edited_rep, edited_base)])
        else:
            mean_base_ed = mean_rep_ed = delta_ed = lo_ed = hi_ed = float('nan')

        if csv_out:
            print('id,base_f1,rep_f1,delta,edited')
            for i, (bf, rf, ef) in enumerate(zip(base_f1s, rep_f1s, edited_flags)):
                print(f'{ids[i]},{100*bf:.4f},{100*rf:.4f},{100*(rf-bf):.4f},{int(ef)}')
            return []

        print('=' * 60)
        print('HaloQuest confession Token-F1 evaluation')
        print(f'  Input : {path}')
        print(f'  N     : {n_total}')
        print('=' * 60)
        print(f'  base_pred    Token-F1 : {mean_base:.2f}')
        print(f'  confession   Token-F1 : {mean_rep:.2f}')
        print(f'  Delta                 : {delta:+.2f}  95% CI [{lo_d:+.2f}, {hi_d:+.2f}]')
        print()
        print(f'  Edited samples        : {n_edited} / {n_total} ({100*n_edited/n_total:.1f}%)')
        if not math.isnan(delta_ed):
            print(f'  Edited-only delta     : {delta_ed:+.2f}  95% CI [{lo_ed:+.2f}, {hi_ed:+.2f}]')
        print('=' * 60)
        return [{
            'File Name': os.path.basename(file_path),
            'Samples': n_total,
            'base Token-F1': round(mean_base, 2),
            'Token-F1': round(mean_rep, 2),
            'Delta': round(delta, 2),
            'CI-lo': round(lo_d, 2),
            'CI-hi': round(hi_d, 2),
            'Edited': n_edited,
        }]

# ================= [main: dynamic grouping and report generation] =================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_folder', type=str, default=None,
                        help="Aggregate evaluation over every file in the folder")
    parser.add_argument('--target_files', type=str, nargs='+', default=None,
                        help="Per-sample evaluation of one or more files")
    parser.add_argument('--csv', action='store_true',
                        help="Dump per-sample CSV (haloquest confession only)")
    args = parser.parse_args()

    if not args.target_folder and not args.target_files:
        print("[error] provide either --target_folder or --target_files")
        exit(1)

    zhipu_key = os.environ.get("ZHIPU_API_KEY")
    evaluator = UnifiedEvaluator(zhipu_key=zhipu_key)

    files_to_eval = []
    is_single_file_mode = False

    if args.target_files:
        files_to_eval = args.target_files
        is_single_file_mode = True
    elif args.target_folder:
        files_to_eval = glob.glob(os.path.join(args.target_folder, "*.json*"))

    all_reports = {}

    for f in files_to_eval:
        if "_scored" in f:
            continue
        name = os.path.basename(f).lower()

        if "cmm" in name:
            target_cat, mode = "cmm", "judgement"
        elif "phd" in name:
            target_cat, mode = "phd", "judgement"
        elif "pubmedqa" in name:
            target_cat, mode = "pubmedqa", "judgement"
        elif "summedits" in name:
            target_cat, mode = "summedits", "judgement"
        elif "halo" in name:
            target_cat, mode = "halo", "halo_confession"
        elif "avh" in name:
            target_cat, mode = "avh", "descriptive"
        else:
            detected = False
            for other in ["halueval", "ragtruth", "drop"]:
                if other in name:
                    target_cat, mode, detected = other, "descriptive", True
                    break
            if not detected:
                continue

        report_key = (mode, target_cat)
        if report_key not in all_reports:
            all_reports[report_key] = []

        try:
            if mode == "judgement":
                results = evaluator.evaluate_judgment_tasks(f, save_single=True)
            elif mode == "halo_confession":
                results = evaluator.evaluate_haloquest_confession(f, csv_out=args.csv)
            else:
                results = evaluator.evaluate_descriptive_tasks(f, save_single=True)
            all_reports[report_key].extend(results)
        except Exception as e:
            print(f"[error] evaluation failed for {f}: {e}")

    for (mode, cat), data in all_reports.items():
        if not data:
            continue
        if mode == "halo_confession":
            filename = f"halo_confession.csv"
            headers = ["File Name", "Samples", "base Token-F1", "Token-F1",
                       "Delta", "CI-lo", "CI-hi", "Edited"]
        elif mode == "judgement":
            filename = f"{mode}_{cat}.csv"
            headers = ["File Name", "Samples", "Categorical-F1", "PhD-Index"]
        else:
            filename = f"{mode}_{cat}.csv"
            headers = ["File Name", "Samples", "Token-F1", "GAVIE-Rel", "GAVIE-Acc"]
        with open(filename, 'a', newline='') as csvfile:
            w = csv.DictWriter(csvfile, fieldnames=headers)
            if csvfile.tell() == 0:
                w.writeheader()
            w.writerows(data)
        print(f"[ok] aggregate report updated: {filename}")