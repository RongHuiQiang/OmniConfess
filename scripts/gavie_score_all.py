# -*- coding: utf-8 -*-
"""gavie_score_all.py — 用单一裁判(默认 GLM-5.3)对全部描述族结果重打 GAVIE。

与 gavie_flash_backfill.py 的区别:
  * 直连 REST(requests),不依赖未安装的 zhipuai SDK;base_url / model / key
    全部可配置,可指向 glm-5.3。
  * glm-5.3 是"始终思考"模型:thinking 只能 low/high/max,不能 disabled
    (env GLM_THINKING, 默认 low —— 打分任务省 token 最快)。
  * 两种范围:
      --mode all   : 全部描述族行重打(默认),旧的 GAVIE-Rel/Acc 移到
                     GAVIE-Rel-prev/GAVIE-Acc-prev(仅当 prev 不存在时),
                     首次处理某文件前整文件备份到 GAVIE_BACKUP_DIR。
      --mode fill  : 只补 GAVIE-Rel/Acc 为 -1 或缺键的行。
  * 全局线程池(env GAVIE_WORKERS, 默认 24),429/1302 递增退避。
  * 断点续跑:每文件处理后原子写回 + 记 state;--shard i/N 可多进程并行,
    文件按路径排序后取模分片,互不冲突。
  * 6 小时窗口友好:每完成一个文件就落盘;中断后重跑自动跳过已完成文件,
    fill/all 都能续(all 模式靠 GAVIE-Judge 标签跳过已用本裁判打过的行)。

用法:
  source /path/.env_glm   # 里面 export GLM_API_KEY=...
  python3 gavie_score_all.py --dry-run
  python3 gavie_score_all.py --mode all --shard 0/4
"""
import os, sys, json, re, time, glob, shutil, argparse, threading
from concurrent.futures import ThreadPoolExecutor

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                       # .../OmniConfess
OUTPUTS = os.path.join(ROOT, 'outputs')

KEY = os.environ.get('GLM_API_KEY') or os.environ.get('ZHIPU_API_KEY')
MODEL = os.environ.get('GLM_JUDGE_MODEL', 'glm-5.3')
BASE_URL = os.environ.get('GLM_BASE_URL',
                          'https://open.bigmodel.cn/api/paas/v4/chat/completions')
THINKING = os.environ.get('GLM_THINKING', 'low')   # low|high|max ('' -> omit)
WORKERS = int(os.environ.get('GAVIE_WORKERS', '24'))
MAXTOK = int(os.environ.get('GAVIE_MAXTOK', '2048'))
BACKUP_DIR = os.environ.get('GAVIE_BACKUP_DIR',
                            os.path.join(ROOT, 'outputs_gavie_prev_backup'))
STATE = os.path.join(HERE, 'gavie_score_all_state.json')

# 只对自由生成的描述族打 GAVIE;判断族(phd/cmm/pubmedqa)用 F1/Index,不打。
DESC_DATASETS = ('ragtruth', 'haloquest', 'avhbench')
SKIP_MARKERS = ('_invalid_', '_archived', '_dead', '_aborted', 'smoke',
                '_failed', '_bak', '/backup')

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

_session = threading.local()


def _sess():
    s = getattr(_session, 's', None)
    if s is None:
        s = requests.Session()
        s.headers.update({'Authorization': 'Bearer ' + KEY,
                          'Content-Type': 'application/json'})
        _session.s = s
    return s


def gavie_call(gt, pred, retries=6):
    body = {'model': MODEL,
            'messages': [{'role': 'user',
                          'content': PROMPT.format(gt=gt, pred=pred)}],
            'temperature': 0.1, 'max_tokens': MAXTOK}
    if THINKING:
        body['thinking'] = {'type': THINKING}
    for i in range(retries):
        try:
            r = _sess().post(BASE_URL, json=body, timeout=120)
            if r.status_code == 200:
                c = (r.json()['choices'][0]['message'].get('content') or '').lower()
                rm = re.search(r'relevancy:\s*(\d+(?:\.\d+)?)', c)
                am = re.search(r'accuracy:\s*(\d+(?:\.\d+)?)', c)
                if rm and am:
                    return (min(float(rm.group(1)), 10.0),
                            min(float(am.group(1)), 10.0))
                time.sleep(2 * (i + 1))          # 200 但格式不对,重试
                continue
            txt = r.text
            if r.status_code == 429 or '1302' in txt or '1301' in txt or '1113' in txt:
                time.sleep(30 * (i + 1))          # 限流/余额:长退避
            else:
                time.sleep(3 * (i + 1))
        except Exception:
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


def is_desc(path):
    b = os.path.basename(path)
    if not any(b.startswith(ds) for ds in DESC_DATASETS):
        return False
    return not any(m in path for m in SKIP_MARKERS)


def row_needs(d, mode):
    em = d.get('eval_metrics')
    if not isinstance(em, dict):
        return False
    if str(d.get('status', em.get('Status', 'success'))).lower() != 'success':
        return False
    if not gt_of(d) or not pred_of(d):
        return False
    if mode == 'fill':
        return ('GAVIE-Rel' not in em or em.get('GAVIE-Rel') == -1
                or em.get('GAVIE-Acc') == -1)
    # mode all: 跳过已被本裁判打过的行(续跑用),其余全部重打
    return em.get('GAVIE-Judge') != MODEL


def discover(mode, shard):
    files = []
    for f in sorted(glob.glob(os.path.join(OUTPUTS, '**', '*_scored.jsonl'),
                              recursive=True)):
        if not is_desc(f):
            continue
        try:
            rows = [json.loads(l) for l in open(f) if l.strip()]
        except Exception:
            continue
        need = [i for i, d in enumerate(rows) if row_needs(d, mode)]
        if need:
            files.append((f, rows, need))
    if shard:
        i, n = shard
        files = [t for k, t in enumerate(files) if k % n == i]
    return files


def backup_once(f):
    if not os.path.isdir(BACKUP_DIR):
        os.makedirs(BACKUP_DIR, exist_ok=True)
    rel = os.path.relpath(f, OUTPUTS).replace('/', '__')
    dst = os.path.join(BACKUP_DIR, rel + '.orig')
    if not os.path.exists(dst):
        shutil.copy2(f, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['all', 'fill'], default='all')
    ap.add_argument('--shard', default='')          # "i/N"
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--limit', type=int, default=0)  # 每文件最多打几行(冒烟)
    args = ap.parse_args()

    assert KEY, 'set GLM_API_KEY (source .env_glm)'
    shard = None
    if args.shard:
        i, n = args.shard.split('/'); shard = (int(i), int(n))

    files = discover(args.mode, shard)
    total = sum(len(n) for _, _, n in files)
    print(f'model={MODEL} thinking={THINKING or "off"} mode={args.mode} '
          f'shard={args.shard or "-"} workers={WORKERS}', flush=True)
    print(f'{len(files)} files, {total} rows to score', flush=True)
    if args.dry_run:
        for f, _, need in files[:40]:
            print(f'  {len(need):5d}  {os.path.relpath(f, OUTPUTS)}')
        if len(files) > 40:
            print(f'  ... (+{len(files)-40} more files)')
        return

    state = json.load(open(STATE)) if os.path.exists(STATE) else {}
    grand = [0]; t_all = time.time()
    for f, rows, need in files:
        if state.get(f, {}).get('done') and args.mode != 'all':
            continue
        if args.mode == 'all':
            backup_once(f)
        if args.limit:
            need = need[:args.limit]
        t0 = time.time(); done = [0]

        def work(i):
            r, a = gavie_call(gt_of(rows[i]), pred_of(rows[i]))
            return i, r, a

        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            for i, r, a in ex.map(work, need):
                em = rows[i].setdefault('eval_metrics', {})
                if args.mode == 'all' and em.get('GAVIE-Rel', -1) != -1 \
                        and 'GAVIE-Rel-prev' not in em:
                    em['GAVIE-Rel-prev'] = em.get('GAVIE-Rel')
                    em['GAVIE-Acc-prev'] = em.get('GAVIE-Acc')
                    em['GAVIE-Judge-prev'] = em.get('GAVIE-Judge')
                em['GAVIE-Rel'] = r
                em['GAVIE-Acc'] = a
                em['GAVIE-Judge'] = MODEL
                em['GAVIE-Prompt'] = PROMPT_VERSION
                done[0] += 1; grand[0] += 1
                if grand[0] % 100 == 0:
                    rate = grand[0] / (time.time() - t_all)
                    eta = (total - grand[0]) / rate if rate else 0
                    print(f'  {grand[0]}/{total} {rate:.1f}/s eta={eta/60:.0f}m',
                          flush=True)

        ok = [d for d in rows
              if str(d.get('status', 'success')).lower() == 'success'
              and d.get('eval_metrics', {}).get('GAVIE-Rel', -1) >= 0
              and d['eval_metrics'].get('GAVIE-Acc', -1) >= 0]
        nsucc = sum(1 for d in rows
                    if str(d.get('status', 'success')).lower() == 'success')
        rel = round(sum(d['eval_metrics']['GAVIE-Rel'] for d in ok) / len(ok), 3) if ok else None
        acc = round(sum(d['eval_metrics']['GAVIE-Acc'] for d in ok) / len(ok), 3) if ok else None
        rate = len(ok) / nsucc if nsucc else 0
        tmp = f + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as fh:
            for d in rows:
                fh.write(json.dumps(d, ensure_ascii=False) + '\n')
        os.replace(tmp, f)
        state[f] = {'done': rate >= 0.9, 'valid_rate': round(rate, 4),
                    'n_success': nsucc, 'n_valid': len(ok),
                    'GAVIE-Rel': rel, 'GAVIE-Acc': acc, 'judge': MODEL,
                    'sec': round(time.time() - t0)}
        json.dump(state, open(STATE, 'w'), ensure_ascii=False, indent=1)
        print(f'[{"DONE" if state[f]["done"] else "PARTIAL"}] '
              f'{os.path.relpath(f, OUTPUTS)} Rel={rel} Acc={acc} '
              f'valid={len(ok)}/{nsucc} {state[f]["sec"]}s', flush=True)

    print(f'ALL DONE {grand[0]} rows in {(time.time()-t_all)/60:.1f}m', flush=True)


if __name__ == '__main__':
    main()
