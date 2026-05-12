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
        """ 1(Yes)  0(No)"""
        text = str(text).lower().strip()
        boxed_match = re.search(r'\\boxed\{(.*?)\}', text)
        if boxed_match:
            ans = boxed_match.group(1).strip()
            return 1 if 'yes' in ans else 0
        if 'final answer: yes' in text: return 1
        if 'final answer: no' in text: return 0
        last_chunk = text[-30:]
        if 'yes' in last_chunk: return 1
        if 'no' in last_chunk: return 0
        return 0

    def evaluate_judgment_tasks(self, file_path):
        """Evaluate judgment-type tasks with per-group accuracy."""
        data = self._load_data(file_path)
        file_name = os.path.basename(file_path)
        is_phd = "phd" in file_name.lower()
        
        print(f"\n[Eval] File: {file_name}")
        print(f"[Eval] {'PhD mode (group by PhD-xxx)' if is_phd else 'Standard mode'}")
        
        grouped = {"Overall": {"y_true": [], "y_pred": []}}
        
        debug_count = 0

        for item in data:
            # gt_val = self._get_value(item, ['gt', 'answer', 'label'])
            gt_val = self._get_value(item, ['gt', 'answer', 'label', 'ground_truth'])
            pred_raw = self._get_value(item, ['pred', 'response', 'output'])
            
            gt = 1 if gt_val.lower() == 'yes' else 0
            pred = self._parse_yesno_answer(pred_raw)
            
            group_key = self._get_value(item, ['sub_category', 'task']) or "Default"
            
            if debug_count < 3:
                print(f"   [-{debug_count}] ID: {item.get('id','N/A')} | : {group_key}")
                debug_count += 1
            
            grouped["Overall"]["y_true"].append(gt)
            grouped["Overall"]["y_pred"].append(pred)
            
            if group_key not in grouped:
                grouped[group_key] = {"y_true": [], "y_pred": []}
            grouped[group_key]["y_true"].append(gt)
            grouped[group_key]["y_pred"].append(pred)

        found_cats = [k for k in grouped.keys() if k != "Overall"]
        print(f" [] : {found_cats}")

        results = []
        sorted_keys = sorted(grouped.keys(), key=lambda x: (x != "Overall", x))
        
        for cat in sorted_keys:
            val = grouped[cat]
            yt, yp = val["y_true"], val["y_pred"]
            if not yt: continue
            
            r_yes = recall_score(yt, yp, pos_label=1, zero_division=0)
            r_no = recall_score(yt, yp, pos_label=0, zero_division=0)
            phd_idx = (2 * r_yes * r_no) / (r_yes + r_no) if (r_yes + r_no) > 0 else 0
            
            results.append({
                "File Name": f"{file_name} [{cat}]",
                "Samples": len(yt),
                "Accuracy": round(accuracy_score(yt, yp) * 100, 2),
                "Categorical-F1": round(f1_score(yt, yp, zero_division=0) * 100, 2),
                "PhD-Index": round(phd_idx * 100, 2)
            })
        
        return results
    def _compute_token_f1(self, pred, gt):
        """ F1 ()"""
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

    def _get_gavie_score(self, gt, pred):
        """ ()"""
        prompt = (
            f"Reference: {gt}\nModel Prediction: {pred}\n"
            "Score the prediction (0.0-10.0) on:\n"
            "1. relevancy: Main event coverage.\n"
            "2. accuracy: Detail correctness.\n"
            "Use decimal points for partial correctness (e.g. 7.4). Output strictly in this format: relevancy: X.X, accuracy: Y.Y"
        )
        try:
            res = self.client.chat.completions.create(model="glm-4", messages=[{"role":"user","content":prompt}], temperature=0.1)
            content = res.choices[0].message.content.lower()
            
            rel_match = re.search(r'relevancy:\s*(\d+(?:\.\d+)?)', content)
            acc_match = re.search(r'accuracy:\s*(\d+(?:\.\d+)?)', content)
            
            r = float(rel_match.group(1)) if rel_match else -1.0
            a = float(acc_match.group(1)) if acc_match else -1.0
            
            if r > 10.0: r = 10.0
            if a > 10.0: a = 10.0
            
            return r, a
        except: 
            return -1.0, -1.0
    def evaluate_descriptive_tasks(self, file_path):
        """ AVH/Halo NLP  GAVIE"""
        data = self._load_data(file_path)
        tokenizer = PTBTokenizer()
        scorers = [(Bleu(4),["B1","B2","B3","B4"]),(Meteor(),"METEOR"),(Rouge(),"ROUGE_L"),(Cider(),"CIDEr")]
        
        metrics = {"f1": [], "cider": [], "rel": [], "acc": []}
        
        for item in tqdm(data, desc="NLP Eval", disable=True):
            gt = self._get_value(item, ['gt', 'ground_truth', 'answer'])
            pred = self._get_value(item, ['pred', 'response', 'output'])
            
            metrics["f1"].append(self._compute_token_f1(pred, gt))
            
            g_dict = {0: [{"caption": gt}]}; p_dict = {0: [{"caption": pred}]}
            
            with contextlib.redirect_stdout(io.StringIO()):
                try:
                    g_tok = tokenizer.tokenize(g_dict)
                    p_tok = tokenizer.tokenize(p_dict)
                    
                    if not g_tok or not p_tok or len(g_tok[0]) == 0 or len(p_tok[0]) == 0:
                        continue

                    for scorer, _ in scorers:
                        try:
                            s, _ = scorer.compute_score(g_tok, p_tok)
                            if isinstance(scorer, Cider): 
                                metrics["cider"].append(s)
                        except Exception:
                            if isinstance(scorer, Cider): metrics["cider"].append(0.0)
                            continue
                except Exception:
                    continue
            
            if self.client:
                r, a = self._get_gavie_score(gt, pred)
                if r >= 0: metrics["rel"].append(r); metrics["acc"].append(a)

        return [{
            "File Name": os.path.basename(file_path),
            "Samples": len(data),
            "Token-F1": round(np.mean(metrics["f1"]) * 100, 2) if metrics["f1"] else 0,
            "CIDEr": round(np.mean(metrics["cider"]), 2) if metrics["cider"] else 0,
            "GAVIE-Rel": round(np.mean(metrics["rel"]), 2) if metrics["rel"] else 0,
            "GAVIE-Acc": round(np.mean(metrics["acc"]), 2) if metrics["acc"] else 0
        }]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_folder', type=str, required=True)
    args = parser.parse_args()

    evaluator = UnifiedEvaluator(zhipu_key=os.environ.get("ZHIPU_API_KEY"))
    
    all_reports = {}
    files = glob.glob(os.path.join(args.target_folder, "*.json*"))

    for f in files:
        if "_scored" in f: continue
        name = os.path.basename(f).lower()
        
        if "cmm" in name:
            target_cat, mode = "cmm", "judgement"
        elif "phd" in name:
            target_cat, mode = "phd", "judgement"
        elif "halo" in name:
            target_cat, mode = "halo", "descriptive"
        elif "avh" in name:
            target_cat, mode = "avh", "descriptive"
        else:
            detected = False
            for other in ["halueval", "ragtruth", "drop", "pubmedqa"]:
                if other in name:
                    target_cat, mode, detected = other, "descriptive", True
                    break
            if not detected: continue

        report_key = (mode, target_cat)
        if report_key not in all_reports:
            all_reports[report_key] = []

        try:
            if mode == "judgement":
                results = evaluator.evaluate_judgment_tasks(f)
            else:
                results = evaluator.evaluate_descriptive_tasks(f)
            all_reports[report_key].extend(results)
        except Exception as e:
            print(f"  {f}: {e}")

    for (mode, cat), data in all_reports.items():
        if not data: continue
        filename = f"{mode}_{cat}.csv"
        headers = ["File Name", "Samples", "Accuracy", "Categorical-F1", "PhD-Index"] if mode == "judgement" else \
                  ["File Name", "Samples", "Token-F1", "CIDEr", "GAVIE-Rel", "GAVIE-Acc"]
        
        with open(filename, 'a', newline='') as csvfile:
            w = csv.DictWriter(csvfile, fieldnames=headers)
            w.writeheader()
            w.writerows(data)
        print(f" : {filename}")