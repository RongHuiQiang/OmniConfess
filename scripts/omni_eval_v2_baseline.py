# -*- coding: utf-8 -*-
"""omni_eval_v2_baseline.py — 对 results_shipped/ 所有 baseline 方法跑 omni-eval-v2 七维打分。

输出: results_shipped/<ds>/{method}_v2scored.jsonl
      每行追加 omni_eval, omni_eval_model, omni_eval_version="omni-eval-v2" 字段

用法:
  ZHIPU_API_KEY=xxx python3 scripts/omni_eval_v2_baseline.py [--workers 8] [--dry-run]

断点续跑: {method}_v2scored.jsonl 已存在且行数 >= 原文件则跳过。
每个文件最多打 MAX_PER_FILE 条，超出部分写 omni_eval_skipped=True。
"""

import os, sys, json, re, time, glob, argparse, threading
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE    = os.path.dirname(os.path.abspath(__file__))
ROOT    = os.path.dirname(HERE)
SHIPPED = os.path.join(ROOT, 'results_shipped')

KEY     = os.environ.get('ZHIPU_API_KEY') or os.environ.get('GLM_API_KEY')
MODEL   = os.environ.get('OMNI_EVAL_MODEL', 'glm-4-flash')
WORKERS = int(os.environ.get('OMNI_EVAL_WORKERS', '8'))

MAX_PER_FILE = 200   # 每文件最多打分条数

DIMENSIONS = [
    'evidence_grounding', 'factual_correctness', 'relevance',
    'completeness', 'logical_coherence', 'uncertainty_calibration', 'clarity',
]

# 执行顺序（从小到大）
DATASET_ORDER = [
    'pubmedqa_500',
    'phd_566',
    'haloquest_608',
    'avhbench_645',
    'ragtruth_740',
    'cmm_795',
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
    sys.path.insert(0, HERE)
    from omni_eval_prompt import build_messages
    return build_messages(
        question=question,
        reference_answers=str(gt),
        generated_answer=str(pred),
        ground_truth=str(gt),
    )


def _parse_scores(text):
    try:
        obj = json.loads(text.strip())
        if all(k in obj for k in DIMENSIONS):
            return {k: int(obj[k]) for k in DIMENSIONS}
    except Exception:
        pass
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
    msgs = _build_prompt(question, gt, pred)
    for attempt in range(retries):
        try:
            resp = _client().chat.completions.create(
                model=MODEL,
                messages=msgs,
                temperature=0,
            )
            text = resp.choices[0].message.content
            parsed = _parse_scores(text)
            if parsed:
                time.sleep(0.1)
                return parsed
            time.sleep(2 * (attempt + 1))
        except Exception as e:
            s = str(e)
            if '429' in s or '1302' in s:
                time.sleep(60 * (attempt + 1))
            else:
                time.sleep(4 * (attempt + 1))
    return {k: -1 for k in DIMENSIONS}


def _score_row_worker(args):
    idx, row, pred_field, gt_field = args
    q    = row.get('question', '')
    gt   = row.get(gt_field) or row.get('gt') or row.get('ground_truth') or row.get('reference', '')
    pred = row.get(pred_field) or row.get('pred') or row.get('response') or row.get('output', '')
    if not pred:
        return idx, {k: -1 for k in DIMENSIONS}
    scores = score_one(q, gt, pred)
    return idx, scores


def detect_pred_field(rows):
    """检测 pred 字段名。"""
    for field in ('pred', 'response', 'output'):
        if any(r.get(field) for r in rows[:5]):
            return field
    return 'pred'


def detect_gt_field(rows):
    """检测 gt 字段名。"""
    for field in ('gt', 'ground_truth', 'reference'):
        if any(r.get(field) for r in rows[:5]):
            return field
    return 'gt'


def validate_scores(scores):
    """
    检测无效打分。返回 (is_invalid, reason)。
    标准：
    1. 任何维度 == -1（解析失败）
    2. 所有维度都是 0（binary 错误）
    3. 所有维度是同一个值（copy-paste 错误）
    4. 任何维度超出 0-10 范围
    """
    vals = [scores.get(d, -1) for d in DIMENSIONS]
    if any(v == -1 for v in vals):
        return True, 'PARSE_FAIL'
    if all(v == 0 for v in vals):
        return True, 'ALL_ZERO'
    if len(set(vals)) == 1:
        return True, f'ALL_SAME({vals[0]})'
    if any(v < 0 or v > 10 for v in vals):
        return True, f'OUT_OF_RANGE({vals})'
    return False, None


def process_file(fpath, outpath, dry_run=False):
    """对单个 jsonl 文件打分，写到 outpath。返回 (n_total, n_scored, n_invalid, dim_means)。"""
    rows = [json.loads(l) for l in open(fpath, encoding='utf-8') if l.strip()]
    n_total = len(rows)

    # 断点续跑
    done_ids = set()
    done_rows = {}
    if os.path.exists(outpath):
        for l in open(outpath, encoding='utf-8'):
            if l.strip():
                try:
                    r = json.loads(l)
                    ev = r.get('omni_eval', {})
                    if ev and r.get('omni_eval_version') == 'omni-eval-v2':
                        done_ids.add(str(r['id']))
                        done_rows[str(r['id'])] = r
                except Exception:
                    pass

    pred_field = detect_pred_field(rows)
    gt_field   = detect_gt_field(rows)

    # 最多打 MAX_PER_FILE 条，其余标记 skipped
    to_score = []
    skipped_rows = []
    scored_count = 0
    for i, r in enumerate(rows):
        rid = str(r['id'])
        if rid in done_ids:
            scored_count += 1
            continue
        if scored_count < MAX_PER_FILE:
            to_score.append((i, r))
            scored_count += 1
        else:
            skipped_rows.append((i, r))

    label = os.path.relpath(fpath, SHIPPED)
    print(f'[{label}] total={n_total} need={len(to_score)} done={len(done_ids)} skip={len(skipped_rows)}', flush=True)

    if dry_run:
        print(f'[{label}] --dry-run, skip API calls', flush=True)
        return n_total, 0, 0, {}

    # 并发打分
    futures_results = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures_map = {
            pool.submit(_score_row_worker, (i, r, pred_field, gt_field)): i
            for i, r in to_score
        }
        done_count = 0
        for fut in as_completed(futures_map):
            try:
                result = fut.result()
            except Exception as e:
                idx = futures_map[fut]
                print(f'[{label}] ERROR idx={idx}: {e}', flush=True)
                result = (idx, {k: -1 for k in DIMENSIONS})
            futures_results.append(result)
            done_count += 1
            if done_count % 20 == 0 or done_count == len(to_score):
                print(f'[{label}] scored {done_count}/{len(to_score)}', flush=True)

    # 检测无效
    invalid_rows = []
    new_scored = {}
    for idx, scores in futures_results:
        r = dict(rows[idx])
        is_inv, reason = validate_scores(scores)
        if is_inv:
            invalid_rows.append((r.get('id'), reason,
                                  str(r.get(pred_field) or r.get('pred', ''))[:60]))
        r['omni_eval']         = scores
        r['omni_eval_model']   = MODEL
        r['omni_eval_version'] = 'omni-eval-v2'
        new_scored[str(r['id'])] = r

    n_invalid   = len(invalid_rows)
    n_new_valid = len(futures_results) - n_invalid
    invalid_rate = n_invalid / len(futures_results) if futures_results else 0.0

    if invalid_rate > 0.05 and len(futures_results) > 0:
        print(f'[{label}] WARNING: invalid_rate={invalid_rate:.1%} > 5%, pausing — NOT writing output', flush=True)
        for rid, reason, pred_preview in invalid_rows[:10]:
            print(f'  {reason}  id={rid}  pred="{pred_preview}"', flush=True)
        return n_total, len(futures_results), n_invalid, {}

    # 报告无效详情
    for rid, reason, pred_preview in invalid_rows:
        print(f'  [INVALID] {reason}  id={rid}  pred="{pred_preview}"', flush=True)

    # 合并 done_rows + new_scored + skipped
    all_out = dict(done_rows)
    all_out.update(new_scored)
    for idx, r in skipped_rows:
        rc = dict(r)
        rc['omni_eval_skipped'] = True
        all_out[str(rc['id'])] = rc

    # 按原顺序排序
    id_order = {str(r['id']): i for i, r in enumerate(rows)}
    out_list = sorted(all_out.values(), key=lambda r: id_order.get(str(r['id']), 9999))

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, 'w', encoding='utf-8') as f:
        for r in out_list:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    # 计算 dim_means（只用有效分数）
    valid_scored = [r for r in out_list
                    if isinstance(r.get('omni_eval'), dict)
                    and r.get('omni_eval_version') == 'omni-eval-v2'
                    and all(r['omni_eval'].get(d, -1) >= 0 for d in DIMENSIONS)]
    dim_means = {}
    if valid_scored:
        dim_means = {d: sum(r['omni_eval'][d] for r in valid_scored) / len(valid_scored)
                     for d in DIMENSIONS}

    n_scored_total = len(done_ids) + len(futures_results)
    print(f'[DONE] {label}', flush=True)
    print(f'  n={n_total} n_scored={n_scored_total} n_invalid={n_invalid} '
          f'invalid_rate={invalid_rate:.2%}', flush=True)
    if dim_means:
        short = {'evidence_grounding': 'ev', 'factual_correctness': 'fc',
                 'relevance': 'rel', 'completeness': 'comp',
                 'logical_coherence': 'lc', 'uncertainty_calibration': 'uc', 'clarity': 'cla'}
        dm_str = '  '.join(f'{short[d]}={dim_means[d]:.2f}' for d in DIMENSIONS)
        print(f'  dim_means: {dm_str}', flush=True)

    return n_total, n_scored_total, n_invalid, dim_means


def discover_files():
    """
    找出所有需要打分的 (fpath, outpath) 对。
    跳过已有 v2scored.jsonl 且行数 >= 原文件的情况。
    返回按 DATASET_ORDER 排序的列表。
    """
    entries = []
    for ds in DATASET_ORDER:
        ds_dir = os.path.join(SHIPPED, ds)
        if not os.path.isdir(ds_dir):
            continue
        for fpath in sorted(glob.glob(os.path.join(ds_dir, '*.jsonl')) +
                            glob.glob(os.path.join(ds_dir, '*.json'))):
            basename = os.path.basename(fpath)
            # 跳过旧 scored 文件和 omni_eval 子目录
            if basename.endswith('_scored.jsonl') or basename.endswith('_v2scored.jsonl'):
                continue
            if 'omni_eval' in fpath:
                continue

            # 检查是否已有 v2 分数
            stem = os.path.splitext(basename)[0]
            outpath = os.path.join(ds_dir, f'{stem}_v2scored.jsonl')

            n_orig = sum(1 for _ in open(fpath, encoding='utf-8') if _.strip())
            if os.path.exists(outpath):
                n_out = sum(1 for _ in open(outpath, encoding='utf-8') if _.strip())
                if n_out >= n_orig:
                    print(f'[SKIP] {os.path.relpath(fpath, SHIPPED)}  '
                          f'(v2scored exists, {n_out}>={n_orig} rows)', flush=True)
                    continue

            entries.append((fpath, outpath))

    return entries


def print_summary(entries):
    print('\n===== OMNI-EVAL-V2 SUMMARY =====', flush=True)
    header = (f"{'file':<40} {'n':>5}  " +
              '  '.join(f'{d[:5]:>5}' for d in DIMENSIONS) + '   avg')
    print(header, flush=True)
    print('-' * len(header), flush=True)

    for fpath, outpath in entries:
        label = os.path.relpath(fpath, SHIPPED)
        if not os.path.exists(outpath):
            print(f'{label:<40} NOT_RUN', flush=True)
            continue
        rows = [json.loads(l) for l in open(outpath, encoding='utf-8') if l.strip()]
        ok = [r for r in rows
              if isinstance(r.get('omni_eval'), dict)
              and r.get('omni_eval_version') == 'omni-eval-v2'
              and all(r['omni_eval'].get(d, -1) >= 0 for d in DIMENSIONS)]
        if not ok:
            print(f'{label:<40} NO_VALID_SCORES ({len(rows)} rows)', flush=True)
            continue
        means = {d: sum(r['omni_eval'][d] for r in ok) / len(ok) for d in DIMENSIONS}
        avg = sum(means.values()) / len(DIMENSIONS)
        dim_str = '  '.join(f'{means[d]:5.2f}' for d in DIMENSIONS)
        print(f'{label:<40} {len(ok):>5}  {dim_str}  {avg:5.2f}', flush=True)


def main():
    global WORKERS, MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=WORKERS)
    ap.add_argument('--model', default=MODEL)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--summary-only', action='store_true')
    ap.add_argument('--datasets', nargs='*', default=None,
                    help='只处理这些数据集，如 pubmedqa_500 haloquest_608')
    ap.add_argument('--methods', nargs='*', default=None,
                    help='只处理含这些关键词的方法文件，如 BaseModel GD')
    args = ap.parse_args()

    WORKERS = args.workers
    MODEL   = args.model

    entries = discover_files()

    if args.datasets:
        entries = [(f, o) for f, o in entries
                   if any(ds in f for ds in args.datasets)]
    if args.methods:
        entries = [(f, o) for f, o in entries
                   if any(m in os.path.basename(f) for m in args.methods)]

    if args.summary_only:
        print_summary(entries)
        return

    if not KEY:
        print('ERROR: ZHIPU_API_KEY 未设置', flush=True)
        sys.exit(1)

    print(f'[omni_eval_v2] model={MODEL} workers={WORKERS} files={len(entries)}', flush=True)

    for fpath, outpath in entries:
        process_file(fpath, outpath, dry_run=args.dry_run)

    print_summary(entries)


if __name__ == '__main__':
    main()
