# -*- coding: utf-8 -*-
"""omni_eval_shipped.py — 对 results_shipped/ 里 freeform 数据集的 baseline 和
OmniConfess 结果跑七维 GLM 评分。

目标文件:
  haloquest_608/BaseModel.jsonl        pred 字段
  haloquest_608/OmniConfess.json       response 字段
  avhbench_645/BaseModel.jsonl         pred 字段
  avhbench_645/OmniConfess.jsonl       response 字段

输出: results_shipped/<ds>/omni_eval/<arm>_scored.jsonl
      每行追加 omni_eval, omni_eval_model, omni_eval_version 字段

用法:
  source /home/tsq/rhq/omniconfess/.env_zhipu
  python3 scripts/omni_eval_shipped.py [--workers 12] [--dry-run] [--arms haloquest/BaseModel ...]

断点续跑: 输出文件已存在则跳过已打分行 (omni_eval 非空且非全-1 即跳过)。
"""
import os, sys, json, re, time, argparse, threading
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE  = os.path.dirname(os.path.abspath(__file__))
ROOT  = os.path.dirname(HERE)
SHIPPED = os.path.join(ROOT, 'results_shipped')

KEY   = os.environ.get('ZHIPU_API_KEY') or os.environ.get('GLM_API_KEY')
MODEL = os.environ.get('OMNI_EVAL_MODEL', 'glm-4.5-flash')
WORKERS = int(os.environ.get('OMNI_EVAL_WORKERS', '12'))

DIMENSIONS = [
    'evidence_grounding', 'factual_correctness', 'relevance',
    'completeness', 'logical_coherence', 'uncertainty_calibration', 'clarity',
]

# freeform 数据集 + 评估臂配置
# arm_name: (jsonl_path_relative_to_SHIPPED, pred_field, gt_field)
ARMS = {
    'haloquest/BaseModel':    ('haloquest_608/BaseModel.jsonl',  'pred',     'gt'),
    'haloquest/OmniConfess':  ('haloquest_608/OmniConfess.json', 'response', 'ground_truth'),
    'avhbench/BaseModel':     ('avhbench_645/BaseModel.jsonl',   'pred',     'gt'),
    'avhbench/OmniConfess':   ('avhbench_645/OmniConfess.jsonl', 'response', 'ground_truth'),
}

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
        reference_answers=gt,
        generated_answer=pred,
        ground_truth=gt,
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


def _score_row_single(args):
    """Worker: 对单行打一组分（只打 pred，不打 base/repaired 对比）。"""
    idx, row, pred_field, gt_field = args
    q    = row.get('question', '')
    gt   = row.get(gt_field, '')
    pred = row.get(pred_field, '')
    if not pred:
        return idx, {k: -1 for k in DIMENSIONS}
    return idx, score_one(q, gt, pred)


def process_arm(arm_name, dry_run=False):
    rel_path, pred_field, gt_field = ARMS[arm_name]
    fpath = os.path.join(SHIPPED, rel_path)
    if not os.path.exists(fpath):
        print(f'[{arm_name}] file not found: {fpath}', flush=True)
        return

    # 输出目录
    ds_dir = os.path.dirname(fpath)
    out_dir = os.path.join(ds_dir, 'omni_eval')
    os.makedirs(out_dir, exist_ok=True)
    arm_slug = arm_name.replace('/', '_')
    outpath = os.path.join(out_dir, f'{arm_slug}_scored.jsonl')

    rows = [json.loads(l) for l in open(fpath, encoding='utf-8') if l.strip()]

    # 断点续跑
    done = {}
    if os.path.exists(outpath):
        for l in open(outpath, encoding='utf-8'):
            if l.strip():
                try:
                    r = json.loads(l)
                    ev = r.get('omni_eval', {})
                    if ev and all(ev.get(d, -1) >= 0 for d in DIMENSIONS):
                        done[str(r['id'])] = r
                except Exception:
                    pass

    need = [(i, r) for i, r in enumerate(rows)
            if r.get(pred_field) and str(r['id']) not in done]

    print(f'[{arm_name}] total={len(rows)} need={len(need)} done={len(done)}', flush=True)

    if dry_run:
        print(f'[{arm_name}] --dry-run, skip API calls', flush=True)
        return

    futures_results = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures_map = {
            pool.submit(_score_row_single, (i, r, pred_field, gt_field)): i
            for i, r in need
        }
        done_count = 0
        for fut in as_completed(futures_map):
            try:
                result = fut.result()
            except Exception as e:
                idx = futures_map[fut]
                print(f'[{arm_name}] ERROR idx={idx}: {e}', flush=True)
                result = (idx, {k: -1 for k in DIMENSIONS})
            futures_results.append(result)
            done_count += 1
            if done_count % 10 == 0 or done_count == len(need):
                print(f'[{arm_name}] scored {done_count}/{len(need)}', flush=True)

    # 合并旧结果
    out_rows = list(done.values())
    for idx, scores in futures_results:
        r = dict(rows[idx])
        r['omni_eval']         = scores
        r['omni_eval_model']   = MODEL
        r['omni_eval_version'] = 'shipped-eval-v1'
        out_rows.append(r)

    id_order = {str(r['id']): i for i, r in enumerate(rows)}
    out_rows.sort(key=lambda r: id_order.get(str(r['id']), 9999))

    with open(outpath, 'w', encoding='utf-8') as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'[{arm_name}] wrote {len(out_rows)} rows → {outpath}', flush=True)


def print_summary():
    print('\n===== OMNI-EVAL SUMMARY (results_shipped) =====')
    header = f"{'arm':<28} {'n':>4}  " + '  '.join(f'{d[:8]:>8}' for d in DIMENSIONS) + '   avg'
    print(header)
    print('-' * len(header))
    for arm_name in ARMS:
        ds_dir = os.path.dirname(os.path.join(SHIPPED, ARMS[arm_name][0]))
        arm_slug = arm_name.replace('/', '_')
        outpath = os.path.join(ds_dir, 'omni_eval', f'{arm_slug}_scored.jsonl')
        if not os.path.exists(outpath):
            print(f'{arm_name:<28} NOT_RUN')
            continue
        rows = [json.loads(l) for l in open(outpath, encoding='utf-8') if l.strip()]
        ok = [r for r in rows
              if isinstance(r.get('omni_eval'), dict)
              and all(r['omni_eval'].get(d, -1) >= 0 for d in DIMENSIONS)]
        if not ok:
            print(f'{arm_name:<28} NO_SCORED_ROWS ({len(rows)} total, all -1)')
            continue
        means = {d: sum(r['omni_eval'][d] for r in ok) / len(ok) for d in DIMENSIONS}
        avg = sum(means.values()) / len(DIMENSIONS)
        dim_str = '  '.join(f'{means[d]:8.2f}' for d in DIMENSIONS)
        print(f'{arm_name:<28} {len(ok):>4}  {dim_str}  {avg:6.2f}')


def main():
    global WORKERS
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=WORKERS)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--summary-only', action='store_true')
    ap.add_argument('--arms', nargs='*', default=None,
                    help='要处理的臂，如 haloquest/BaseModel avhbench/OmniConfess；默认全部')
    args = ap.parse_args()

    WORKERS = args.workers

    if args.summary_only:
        print_summary()
        return

    arms = args.arms if args.arms else list(ARMS.keys())
    invalid = [a for a in arms if a not in ARMS]
    if invalid:
        print(f'未知臂: {invalid}，合法值: {list(ARMS.keys())}')
        sys.exit(1)

    if not KEY:
        print('ERROR: ZHIPU_API_KEY / GLM_API_KEY 未设置，请先 source .env_zhipu', flush=True)
        sys.exit(1)

    for arm in arms:
        process_arm(arm, dry_run=args.dry_run)

    print_summary()


if __name__ == '__main__':
    main()
