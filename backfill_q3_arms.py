# -*- coding: utf-8 -*-
"""
backfill_q3_arms.py — elastic single-card scheduler for the Qwen3 input-perturbation
arms on avhbench.

Why this exists: the 8-way launch of both arms collided with *other experiment
sessions* grabbing the same cards (ragtruth x Qwen3 x5, nemotron, base_tot), so
most shard processes died at model load with CUDA OOM. The machine is now
shared; this scheduler polls for any card with >66GB free, runs one pending
shard process per free card, and is fully resume-safe (the runner skips
already-done ids; a shard is "done" when its jsonl reaches the shard size).

Tasks are derived from what is actually missing on disk, so rerunning this
script at any time is safe. After all shards complete it re-merges each arm's
avhbench_<variant>_full.jsonl.

Usage:  python backfill_q3_arms.py
"""
import json, os, subprocess, time, sys

ROOT = '/home/tsq/rhq/omniconfess/OmniConfess'
SHARDS = '/ultrafast_share/tsq/qwen3omni_outputs/shards/avhbench'
PY = '/ultrafast_share/tsq/venvs/omniconfess/bin/python'
RUNNER = ROOT + '/OmniConfess/methods/shared/base_model_qwen3_omni_var.py'
VARIANTS = {
    'q3_res384': ('res384', 384, 8),
    'q3_f16':    ('frames16', 224, 16),
}
FREE_MIB = 66000          # thinker-only 30B needs ~62GB + activations
POLL_S = 20
MAX_TASK_S = 45 * 60      # kill + requeue a shard that stalls
OVERALL_DEADLINE = time.time() + 4.5 * 3600


def shard_size(i):
    p = f'{SHARDS}/avhbench_shard{i}of8.json'
    with open(p) as f:
        return len(json.load(f))


def nlines(p):
    if not os.path.exists(p):
        return 0
    n = 0
    with open(p, encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def free_gpus():
    out = subprocess.run(
        ['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'],
        capture_output=True, text=True).stdout
    d = {}
    for row in out.strip().splitlines():
        idx, free = row.split(',')
        d[int(idx)] = int(free)
    return d


def main():
    os.chdir(ROOT)
    tasks = []
    for variant, (tag, size, nfr) in VARIANTS.items():
        for i in range(8):
            need = shard_size(i)
            have = nlines(f'{ROOT}/outputs/avhbench_vote_sc/{variant}/avhbench_{variant}_shard{i}.jsonl')
            if have < need:
                tasks.append(dict(variant=variant, tag=tag, size=size, nfr=nfr,
                                  shard=i, need=need, have=have))
    print(f'[backfill] {len(tasks)} incomplete shard tasks:', flush=True)
    for t in tasks:
        print(f'   {t["variant"]} shard{t["shard"]}: {t["have"]}/{t["need"]}', flush=True)
    if not tasks:
        merge()
        return

    running = {}          # gpu -> (proc, task, t0)

    def launch(t, gpu):
        env = dict(os.environ)
        env.update(CUDA_VISIBLE_DEVICES=str(gpu),
                   OMNI_FRAME_SIZE=str(t['size']), OMNI_NUM_FRAMES=str(t['nfr']),
                   MMHALU_MODEL_ROOT='/ultrafast_share/tsq/MODELS',
                   TMPDIR='/ultrafast_share/tsq/qwen3omni_outputs/tmp',
                   PYTHONDONTWRITEBYTECODE='1',
                   PATH='/home/tsq/rhq/omniconfess/bin:' + env['PATH'])
        out = f'{ROOT}/outputs/avhbench_vote_sc/{t["variant"]}/shard{t["shard"]}_bf.log'
        cmd = [PY, RUNNER,
               '--datasets', 'avhbench',
               '--data_path', f'{SHARDS}/avhbench_shard{t["shard"]}of8.json',
               '--data_root', './OmniHalluBench/avhbench/',
               '--output_dir', f'{ROOT}/outputs/avhbench_vote_sc/{t["variant"]}',
               '--time_path', f'{ROOT}/outputs/avhbench_vote_sc/{t["variant"]}/time',
               '--file_name', f'avhbench_{t["variant"]}_shard{t["shard"]}',
               '--model_tag', 'Qwen3-Omni-30B-A3B-Instruct',
               '--variant', t['tag']]
        with open(out, 'a') as f:
            f.write(f'\n===== backfill launch gpu={gpu} {time.ctime()} =====\n')
            proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
        running[gpu] = (proc, t, time.time())
        print(f'[backfill] gpu{gpu} -> {t["variant"]} shard{t["shard"]} '
              f'({t["have"]}/{t["need"]} on disk)', flush=True)

    while tasks or running:
        if time.time() > OVERALL_DEADLINE:
            print('[backfill] overall deadline hit; merging what exists', flush=True)
            for gpu, (proc, t, t0) in list(running.items()):
                if proc.poll() is None:
                    proc.terminate()
            time.sleep(5)
            break

        # reap finished / stalled processes
        for gpu, (proc, t, t0) in list(running.items()):
            done = nlines(f'{ROOT}/outputs/avhbench_vote_sc/{t["variant"]}/avhbench_{t["variant"]}_shard{t["shard"]}.jsonl')
            if done >= t['need']:
                print(f'[backfill] gpu{gpu} DONE {t["variant"]} shard{t["shard"]} ({done}/{t["need"]})', flush=True)
                proc.terminate()
                running.pop(gpu)
            elif proc.poll() is not None:
                print(f'[backfill] gpu{gpu} EXITED rc={proc.returncode} {t["variant"]} '
                      f'shard{t["shard"]} at {done}/{t["need"]} -> requeue', flush=True)
                t['have'] = done
                tasks.append(t)
                running.pop(gpu)
            elif time.time() - t0 > MAX_TASK_S:
                print(f'[backfill] gpu{gpu} STALLED {t["variant"]} shard{t["shard"]} -> kill+requeue', flush=True)
                proc.terminate()
                time.sleep(3)
                t['have'] = done
                tasks.append(t)
                running.pop(gpu)

        # never double-drive a shard some other process already owns
        live = subprocess.run(['pgrep', '-af', 'base_model_qwen3_omni_var'],
                              capture_output=True, text=True).stdout
        for t in [t for t in tasks if f"avhbench_{t['variant']}_shard{t['shard']}" in live]:
            tasks.remove(t)
            print(f'[backfill] {t["variant"]} shard{t["shard"]} owned by external '
                  f'process, leaving it alone', flush=True)

        # assign pending tasks to free cards
        if tasks:
            fg = free_gpus()
            for gpu, free in sorted(fg.items()):
                if gpu in running or not tasks:
                    continue
                if free > FREE_MIB:
                    launch(tasks.pop(0), gpu)

        time.sleep(POLL_S)

    merge()


def merge():
    for variant in VARIANTS:
        out = f'{ROOT}/outputs/avhbench_vote_sc/{variant}/avhbench_{variant}_full.jsonl'
        n = 0
        with open(out, 'w', encoding='utf-8') as f:
            for i in range(8):
                p = f'{ROOT}/outputs/avhbench_vote_sc/{variant}/avhbench_{variant}_shard{i}.jsonl'
                if not os.path.exists(p):
                    continue
                for line in open(p, encoding='utf-8', errors='ignore'):
                    if line.strip():
                        f.write(line)
                        n += 1
        print(f'[merge] {variant}: {n} lines -> {out}', flush=True)


if __name__ == '__main__':
    main()
