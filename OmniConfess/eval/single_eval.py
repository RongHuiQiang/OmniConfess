import os
import json
import re
import csv
import glob
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

    def _parse_yesno_answer(self, text):
        text = str(text).lower().strip()
        boxed_match = re.search(r'\\boxed\{(.*?)\}', text)
        if boxed_match:
            ans = boxed_match.group(1).strip()
            return 1 if 'yes' in ans else 0
        if 'final answer: yes' in text: return 1
        if 'final answer: no' in text: return 0
        # Check first word (handles "Yes. ..." or "No, ..." style responses)
        first_word = text.split('.')[0].split(',')[0].split()[0] if text.split() else ''
        if first_word == 'yes': return 1
        if first_word == 'no': return 0
        last_chunk = text[-30:]
        if 'yes' in last_chunk: return 1
        if 'no' in last_chunk: return 0
        return 0

    def evaluate_judgment_tasks(self, file_path, save_single=False):
        data = self._load_data(file_path)
        file_name = os.path.basename(file_path)
        
        print(f"\n [] : {file_name}")
        grouped = {"Overall": {"y_true": [], "y_pred": []}}
        
        scored_data = []

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
            
            item["eval_metrics"] = {
                "Ground_Truth": gt,
                "Prediction": pred,
                "Is_Correct": 1 if gt == pred else 0,
                "Group": group_key,
                "Status": status
            }
            scored_data.append(item)

            if status == "success":
                grouped["Overall"]["y_true"].append(gt)
                grouped["Overall"]["y_pred"].append(pred)
                
                if group_key not in grouped:
                    grouped[group_key] = {"y_true": [], "y_pred": []}
                grouped[group_key]["y_true"].append(gt)
                grouped[group_key]["y_pred"].append(pred)

        if save_single:
            out_path = file_path.rsplit('.', 1)[0] + "_single_scored.jsonl"
            with open(out_path, 'w', encoding='utf-8') as f:
                for d in scored_data:
                    f.write(json.dumps(d, ensure_ascii=False) + '\n')
            print(f" : {out_path}")

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
            return self.client.chat.completions.create(model="glm-4", messages=[{"role":"user","content":prompt}], temperature=0.1)
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(_call)
                res = future.result(timeout=timeout_sec)
            content = res.choices[0].message.content.lower()
            
            if debug:
                print(f"    [ API ]: {content.strip()}")

            rel_match = re.search(r'relevancy:\s*(\d+(?:\.\d+)?)', content)
            acc_match = re.search(r'accuracy:\s*(\d+(?:\.\d+)?)', content)
            r = float(rel_match.group(1)) if rel_match else -1.0
            a = float(acc_match.group(1)) if acc_match else -1.0
            if r > 10.0: r = 10.0
            if a > 10.0: a = 10.0
            return r, a
        except Exception as e:
            if debug:
                print(f"    [API /]: {e}")
            return -1.0, -1.0

    def evaluate_descriptive_tasks(self, file_path, save_single=False):
        data = self._load_data(file_path)
        metrics = {"f1": [], "rel": [], "acc": []}
        scored_data = [] 

        for idx, item in enumerate(tqdm(data, desc=f"Eval ({os.path.basename(file_path)})", disable=not save_single)): 
            status = str(item.get('status', 'success')).lower()
            gt = self._get_value(item, ['gt', 'ground_truth', 'answer'])
            pred = self._get_value(item, ['pred', 'response', 'output'])
            
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
            print(f" : {out_path}")

        return [{
            "File Name": os.path.basename(file_path),
            "Samples": len(metrics["f1"]),
            "Token-F1": round(np.mean(metrics["f1"]) * 100, 2) if metrics["f1"] else 0.0,
            "GAVIE-Rel": round(np.mean(metrics["rel"]), 2) if metrics["rel"] else 0.0,
            "GAVIE-Acc": round(np.mean(metrics["acc"]), 2) if metrics["acc"] else 0.0
        }]
        cider_score = 0.0
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
                print(f"\n CIDEr : {e}")

        for idx, item in enumerate(scored_data):
            item["eval_metrics"]["CIDEr"] = round(float(individual_cider_scores[idx]), 4)

        if save_single:
            out_path = file_path.rsplit('.', 1)[0] + "_single_scored.jsonl"
            with open(out_path, 'w', encoding='utf-8') as f:
                for d in scored_data:
                    f.write(json.dumps(d, ensure_ascii=False) + '\n')
            print(f" : {out_path}")

        if self.client and len(metrics["rel"]) == 0:
            print(f"\n️ : GAVIE  API 。")

        return [{
            "File Name": os.path.basename(file_path),
            "Samples": len(data),
            "Token-F1": round(np.mean(metrics["f1"]) * 100, 2) if metrics["f1"] else 0.0,
            "CIDEr": round(cider_score * 100, 2),
            "GAVIE-Rel": round(np.mean(metrics["rel"]), 2) if metrics["rel"] else 0.0,
            "GAVIE-Acc": round(np.mean(metrics["acc"]), 2) if metrics["acc"] else 0.0
        }]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_folder', type=str, default=None, help="Aggregate evaluation on all files in a folder")
    parser.add_argument('--target_files', type=str, nargs='+', default=None, help="Per-sample evaluation on one or more files")
    args = parser.parse_args()

    if not args.target_folder and not args.target_files:
        print("  --target_folder  --target_files")
        exit(1)

    zhipu_key = "YOUR_ZHIPU_API_KEY"  # set via env ZHIPU_API_KEY
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
        if "_scored" in f: continue
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
            target_cat, mode = "halo", "descriptive"
        elif "avh" in name:
            target_cat, mode = "avh", "descriptive"
        else:
            detected = False
            for other in ["halueval", "ragtruth", "drop"]:
                if other in name:
                    target_cat, mode, detected = other, "descriptive", True
                    break
            if not detected: continue

        report_key = (mode, target_cat)
        if report_key not in all_reports:
            all_reports[report_key] = []

        try:
            if mode == "judgement":
                results = evaluator.evaluate_judgment_tasks(f, save_single=True)
            else:
                results = evaluator.evaluate_descriptive_tasks(f, save_single=True)
            all_reports[report_key].extend(results)
        except Exception as e:
            print(f"  {f}: {e}")

    for (mode, cat), data in all_reports.items():
        if not data: continue
        filename = f"{mode}_{cat}.csv"
        headers = ["File Name", "Samples", "Categorical-F1", "PhD-Index"] if mode == "judgement" else \
                  ["File Name", "Samples", "Token-F1", "GAVIE-Rel", "GAVIE-Acc"]
        with open(filename, 'a', newline='') as csvfile:
            w = csv.DictWriter(csvfile, fieldnames=headers)
            if csvfile.tell() == 0:
                w.writeheader()
            w.writerows(data)
        print(f" : {filename}")