# -*- coding: utf-8 -*-
"""gavie_flash_backfill.py — GAVIE flash 队列补齐 (job #20).

Reconstructs the pipeline3 flash-judge queue (judge=glm-4.5-flash,
thinking=disabled, GAVIE_WORKERS=2, 429/1302 progressive backoff) for every
descriptive *_single_scored.jsonl whose eval_metrics.GAVIE-Rel/Acc still
carry the -1 sentinel. Only the GAVIE fields are filled; Token-F1 and all
other fields are left untouched. Resumable: rows already >=0 are skipped,
per-file progress is checkpointed to gavie_flash_backfill_state.json.
A file is marked done only if >=90% of its success rows got a valid score.
"""
import os, sys, json, re, time, glob
from concurrent.futures import ThreadPoolExecutor

from zhipuai import ZhipuAI

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
STATE = os.path.join(HERE, 'gavie_flash_backfill_state.json')
KEY = os.environ.get('ZHIPU_API_KEY')
JUDGE = os.environ.get('ZHIPU_JUDGE_MODEL', 'glm-4.5-flash')
assert KEY, 'source /home/tsq/rhq/omniconfess/.env_zhipu first'

# gavie-judge-v2 (four-element audit 2026-08-24; identical text to
# gavie_eval_hedge.GAVIE_PROMPT). The queue this script fills was pending on
# zhipu balance, so no row scored by this script has shipped under v1; every
# row it writes is tagged eval_metrics['GAVIE-Prompt'] so v1-scored legacy
# rows (from eval/single_eval.py) stay distinguishable inside the same file.
PROMPT = (
    "Reference: {gt}\nModel Prediction: {pred}\n"
    "You are a strict judge. The Reference is the ONLY ground truth; the "
    "Model Prediction is the answer under review: check it only against the "
    "Reference, never against your own knowledge, and never reward fluency "
    "or plausibility alone.\n"
    "Score the prediction (0.0-10.0) on:\n"
    "1. relevancy: how much of the Reference's main content the Model "
    "Prediction covers.\n"
    "2. accuracy: how many details in the Model Prediction are consistent "
    "with the Reference; any detail that contradicts or is absent in the "
    "Reference is a hallucination and lowers accuracy even if plausible.\n"
    "Use decimal points for partial correctness (e.g. 7.4). "
    "Output strictly in this format: relevancy: X.X, accuracy: Y.Y"
)
PROMPT_VERSION = 'gavie-judge-v2'

client = ZhipuAI(api_key=KEY)


def gavie_call(gt, pred, retries=5):
    prompt = PROMPT.format(gt=gt, pred=pred)
    for i in range(retries):
        try:
            r = client.chat.completions.create(
                model=JUDGE,
                messages=[{'role': 'user', 'content': prompt}],
                temperature=0.1,
                thinking={'type': 'disabled'})
            c = r.choices[0].message.content.lower()
            rm = re.search(r'relevancy:\s*(\d+(?:\.\d+)?)', c)
            am = re.search(r'accuracy:\s*(\d+(?:\.\d+)?)', c)
            if rm and am:
                return (min(float(rm.group(1)), 10.0),
                        min(float(am.group(1)), 10.0))
        except Exception as e:
            s = str(e)
            if '429' in s or '1302' in s:
                time.sleep(60 * (i + 1))   # quota/rate: long progressive backoff
            else:
                time.sleep(3 * (i + 1))
    return -1.0, -1.0


def gt_of(d):
    for k in ('gt', 'ground_truth', 'answer'):
        if d.get(k) is not None:
            return str(d[k])
    return ''


def pred_of(d):
    for k in ('pred', 'response', 'output'):
        if d.get(k) is not None:
            return str(d[k])
    return ''


def main():
    files = []
    for f in sorted(glob.glob(os.path.join(ROOT, 'outputs', '**', '*_scored.jsonl'),
                            recursive=True)):
        if '_archived' in f or '_invalid_' in f:
            continue
        try:
            rows = [json.loads(l) for l in open(f) if l.strip()]
        except Exception:
            continue
        need = [i for i, d in enumerate(rows)
                if isinstance(d.get('eval_metrics'), dict)
                and 'GAVIE-Rel' in d['eval_metrics']
                and (d['eval_metrics']['GAVIE-Rel'] == -1
                     or d['eval_metrics']['GAVIE-Acc'] == -1)]
        if need:
            files.append((f, rows, need))
    state = json.load(open(STATE)) if os.path.exists(STATE) else {}
    total = sum(len(n) for _, _, n in files)
    print(f'{len(files)} files, {total} rows to score (judge={JUDGE})', flush=True)

    for f, rows, need in files:
        if state.get(f, {}).get('done'):
            continue
        t0 = time.time()
        done_ct = [0]
        lock_rows = need[:]

        def work(i):
            r, a = gavie_call(gt_of(rows[i]), pred_of(rows[i]))
            return i, r, a

        with ThreadPoolExecutor(max_workers=2) as ex:
            for i, r, a in ex.map(work, lock_rows):
                rows[i]['eval_metrics']['GAVIE-Rel'] = r
                rows[i]['eval_metrics']['GAVIE-Acc'] = a
                rows[i]['eval_metrics']['GAVIE-Judge'] = JUDGE
                rows[i]['eval_metrics']['GAVIE-Prompt'] = PROMPT_VERSION
                done_ct[0] += 1
                if done_ct[0] % 100 == 0:
                    print(f'  {os.path.basename(f)} {done_ct[0]}/{len(need)} '
                          f'{time.time()-t0:.0f}s', flush=True)

        ok = [d for d in rows
              if str(d.get('status', 'success')).lower() == 'success'
              and d.get('eval_metrics', {}).get('GAVIE-Rel', -1) >= 0
              and d['eval_metrics'].get('GAVIE-Acc', -1) >= 0]
        nsucc = sum(1 for d in rows
                    if str(d.get('status', 'success')).lower() == 'success')
        rel = round(sum(d['eval_metrics']['GAVIE-Rel'] for d in ok) / len(ok), 3) if ok else None
        acc = round(sum(d['eval_metrics']['GAVIE-Acc'] for d in ok) / len(ok), 3) if ok else None
        rate = len(ok) / nsucc if nsucc else 0
        entry = {'done': rate >= 0.9, 'valid_rate': round(rate, 4),
                 'n_success': nsucc, 'n_valid': len(ok),
                 'GAVIE-Rel': rel, 'GAVIE-Acc': acc,
                 'judge': JUDGE, 'sec': round(time.time() - t0)}
        tmp = f + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as fh:
            for d in rows:
                fh.write(json.dumps(d, ensure_ascii=False) + '\n')
        os.replace(tmp, f)
        state[f] = entry
        json.dump(state, open(STATE, 'w'), ensure_ascii=False, indent=1)
        print(f'[{"DONE" if entry["done"] else "PARTIAL"}] {os.path.relpath(f, ROOT)} '
              f'Rel={rel} Acc={acc} valid={len(ok)}/{nsucc} {entry["sec"]}s', flush=True)

    print('ALL FILES PROCESSED', flush=True)


if __name__ == '__main__':
    main()
