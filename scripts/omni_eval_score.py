# -*- coding: utf-8 -*-
"""omni_eval_score.py — 七维多模态QA评估，对 Round 7 freeform 结果打分。

对每个样本分别给 base_pred 和 repaired_pred 打分，输出到
outputs/freeform_grid/round7/omni_eval/ 目录，每个输入文件对应一个输出文件。

用法:
  source /home/tsq/rhq/omniconfess/.env_zhipu
  python3 scripts/omni_eval_score.py [--workers 12] [--dry-run] [--files R7-COV60]

断点续跑: 输出文件已存在则跳过已打分行 (omni_eval_base 非空即跳过)。
"""
import os, sys, json, re, time, glob, argparse, threading
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

KEY  = os.environ.get('ZHIPU_API_KEY') or os.environ.get('GLM_API_KEY')
MODEL = os.environ.get('OMNI_EVAL_MODEL', 'glm-4.5-flash')
WORKERS = int(os.environ.get('OMNI_EVAL_WORKERS', '12'))

INDIR  = os.path.join(ROOT, 'outputs', 'freeform_grid', 'round7')
OUTDIR = os.path.join(INDIR, 'omni_eval')

# 七维顺序必须与 prompt 一致
DIMENSIONS = [
    'evidence_grounding', 'factual_correctness', 'relevance',
    'completeness', 'logical_coherence', 'uncertainty_calibration', 'clarity',
]

_tls = threading.local()


def _client():
    c = getattr(_tls, 'c', None)
    if c is None:
        from zhipuai import ZhipuAI
        _tls.c = ZhipuAI(api_key=KEY)
        c = _tls.c
    return c


def _build_prompt(question, gt, pred):
    from omni_eval_prompt import build_messages
    return build_messages(
        question=question,
        reference_answers=gt,
        generated_answer=pred,
        ground_truth=gt,
    )


def _parse_scores(text):
    """从模型输出中提取七维 JSON。返回 dict 或 None（解析失败）。"""
    # 尝试直接解析
    try:
        obj = json.loads(text.strip())
        if all(k in obj for k in DIMENSIONS):
            return {k: int(obj[k]) for k in DIMENSIONS}
    except Exception:
        pass
    # 从文本中找 {...} 块
    m = re.search(r'\{[^{}]+\}', text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            if all(k in obj for k in DIMENSIONS):
                return {k: int(obj[k]) for k in DIMENSIONS}
        except Exception:
            pass
    return None


def score_one(question, gt, pred, retries=5):
    """调用 API 给单个答案打七维分。失败返回全 -1。"""
    msgs = _build_prompt(question, gt, pred)
    for attempt in range(retries):
        try:
            resp = _client().chat.completions.create(
                model=MODEL,
                messages=msgs,
                temperature=0.05,
                thinking={'type': 'disabled'},
            )
            text = resp.choices[0].message.content
            parsed = _parse_scores(text)
            if parsed:
                return parsed
            time.sleep(2 * (attempt + 1))
        except Exception as e:
            s = str(e)
            if '429' in s or '1302' in s:
                time.sleep(60 * (attempt + 1))
            else:
                time.sleep(4 * (attempt + 1))
    return {k: -1 for k in DIMENSIONS}


def _score_row(args):
    """Worker 函数：对一行打 base + repaired 两组分。返回 (idx, base_scores, rep_scores)。"""
    idx, row = args
    q  = row.get('question', '')
    gt = row.get('gt', '')
    base_pred    = row.get('base_pred', '')
    repaired_pred = row.get('repaired_pred', base_pred)
    base_scores    = score_one(q, gt, base_pred)
    repaired_scores = score_one(q, gt, repaired_pred)
    return idx, base_scores, repaired_scores


def process_file(fpath, outpath, dry_run=False):
    rows = [json.loads(l) for l in open(fpath, encoding='utf-8') if l.strip()]

    # 断点续跑：读已有输出
    done = {}
    if os.path.exists(outpath):
        for l in open(outpath, encoding='utf-8'):
            if l.strip():
                try:
                    r = json.loads(l)
                    done[r['id']] = r
                except Exception:
                    pass

    need = [(i, r) for i, r in enumerate(rows)
            if r.get('status') == 'success'
            and r['id'] not in done]

    tag = os.path.basename(fpath).replace('haloquest_', '').replace('.jsonl', '')
    print(f'[{tag}] total={len(rows)} need={len(need)} done={len(done)}', flush=True)

    if dry_run:
        print(f'[{tag}] --dry-run, skip API calls', flush=True)
        return

    workers = WORKERS
    out_rows = list(done.values())

    futures_map = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for item in need:
            futures_map[pool.submit(_score_row, item)] = item[0]

        done_count = 0
        futures_results = []
        for fut in as_completed(futures_map):
            try:
                result = fut.result()
            except Exception as e:
                idx = futures_map[fut]
                print(f'[{tag}] ERROR row idx={idx}: {e}', flush=True)
                result = (idx, {k: -1 for k in DIMENSIONS}, {k: -1 for k in DIMENSIONS})
            futures_results.append(result)
            done_count += 1
            if done_count % 5 == 0 or done_count == len(need):
                print(f'[{tag}] scored {done_count}/{len(need)}', flush=True)

    for idx, base_scores, rep_scores in futures_results:
        r = dict(rows[idx])
        r['omni_eval_base']     = base_scores
        r['omni_eval_repaired'] = rep_scores
        r['omni_eval_model']    = MODEL
        r['omni_eval_version']  = 'omni-eval-v1'
        out_rows.append(r)

    # 按原始 id 顺序写出
    id_order = {r['id']: i for i, r in enumerate(rows)}
    out_rows.sort(key=lambda r: id_order.get(r['id'], 9999))

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, 'w', encoding='utf-8') as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'[{tag}] wrote {len(out_rows)} rows → {outpath}', flush=True)


def print_summary(outdir):
    files = sorted(glob.glob(os.path.join(outdir, '*.jsonl')))
    if not files:
        print('No scored files found.', flush=True)
        return

    header = f"{'tag':<18} {'n':>4}  " + '  '.join(f'{d[:8]:>8}' for d in DIMENSIONS) + \
             '  base_avg  rep_avg   delta'
    print('\n===== OMNI-EVAL SUMMARY =====')
    print(header)
    print('-' * len(header))

    for fpath in files:
        tag = os.path.basename(fpath).replace('haloquest_', '').replace('.jsonl', '')
        rows = [json.loads(l) for l in open(fpath, encoding='utf-8') if l.strip()]
        ok = [r for r in rows
              if isinstance(r.get('omni_eval_base'), dict)
              and isinstance(r.get('omni_eval_repaired'), dict)
              and all(r['omni_eval_base'].get(d, -1) >= 0 for d in DIMENSIONS)]
        if not ok:
            print(f'{tag:<18} NO_SCORED_ROWS')
            continue

        base_means = {d: sum(r['omni_eval_base'][d] for r in ok) / len(ok) for d in DIMENSIONS}
        rep_means  = {d: sum(r['omni_eval_repaired'][d] for r in ok) / len(ok) for d in DIMENSIONS}
        base_avg = sum(base_means.values()) / len(DIMENSIONS)
        rep_avg  = sum(rep_means.values())  / len(DIMENSIONS)
        delta    = rep_avg - base_avg

        dim_str = '  '.join(f'{rep_means[d]:8.2f}' for d in DIMENSIONS)
        print(f'{tag:<18} {len(ok):>4}  {dim_str}  {base_avg:8.2f}  {rep_avg:8.2f}  {delta:+7.2f}')


def main():
    global WORKERS
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=WORKERS)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--files', nargs='*', default=None,
                    help='只处理含这些关键词的文件（如 R7-COV60 R7-TOPK）')
    ap.add_argument('--summary-only', action='store_true',
                    help='只打印已有结果的汇总表，不调用 API')
    args = ap.parse_args()

    WORKERS = args.workers

    if args.summary_only:
        print_summary(OUTDIR)
        return

    assert KEY, 'source /home/tsq/rhq/omniconfess/.env_zhipu first'

    all_files = sorted(glob.glob(os.path.join(INDIR, 'haloquest_R7-*.jsonl')))
    if args.files:
        kws = args.files
        all_files = [f for f in all_files if any(kw in f for kw in kws)]

    print(f'[omni_eval] model={MODEL} workers={WORKERS} files={len(all_files)}', flush=True)

    for fpath in all_files:
        tag  = os.path.basename(fpath).replace('haloquest_', '').replace('.jsonl', '')
        outpath = os.path.join(OUTDIR, f'haloquest_{tag}.jsonl')
        process_file(fpath, outpath, dry_run=args.dry_run)

    print_summary(OUTDIR)


if __name__ == '__main__':
    main()
