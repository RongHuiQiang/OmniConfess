#!/usr/bin/env python3.10
# -*- coding: utf-8 -*-
"""log_vram_peak.py -- VRAM peak logger for ANY runner (wrapper mode).

Wraps an arbitrary runner command, samples while it runs, and reports:
  1. torch-level peak: torch.cuda.max_memory_allocated() / reserved, via a
     lightweight sitecustomize hook injected into the child (if the child
     imports torch and uses CUDA; otherwise reported as n/a), and
  2. driver-level peak: nvidia-smi polling of this process tree's GPU memory
     (attribute via compute-apps PID list; catches allocator reserve, KV
     cache, fragmentation -- everything torch counters miss).

Usage (from repo root):
  python scripts/log_vram_peak.py --label cf --out outputs/vram/vram_cf.json \
      -- python OmniConfess/cf_confess_phd.py --limit 3 [runner args...]

  Everything after `--` is the runner command. `--label` names the method in
  the report; `--out` writes JSON (one line per invocation, append-mode, so
  repeated runs accumulate); `--interval` sets the nvidia-smi poll period
  (default 0.5 s).

Typical session (after GPUs 0-3 are free, per SOP; check nvidia-smi first):
  for m in base cf avcd opera; do
    python scripts/log_vram_peak.py --label $m --out outputs/vram/vram_peaks.jsonl \
        -- <runner command for method $m, 3 samples only: --limit 3>
  done
  python scripts/log_vram_peak.py --summarize outputs/vram/vram_peaks.jsonl

`--summarize` prints the per-method peak table (max over runs of
peak MiB: torch_alloc, torch_reserved, driver) without running anything.

The child torch hook works by prepending a temp dir to PYTHONPATH containing
a sitecustomize.py that registers an atexit hook printing
__VRAM_TORCH__ {json}__ to stdout of the child; the wrapper scrapes it.
NO GPU WORK IS DONE BY THIS SCRIPT ITSELF.
"""
import argparse, json, os, subprocess, sys, tempfile, threading, time

HOOK = r'''
# auto-injected by log_vram_peak.py (sitecustomize)
import atexit, json
def _vram_dump():
    try:
        import torch
        if torch.cuda.is_available():
            stats = {"max_memory_allocated": torch.cuda.max_memory_allocated(),
                     "max_memory_reserved": torch.cuda.max_memory_reserved()}
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
            print("__VRAM_TORCH__ " + json.dumps(stats) + "__", flush=True)
    except Exception:
        pass
atexit.register(_vram_dump)
'''


def poll_driver(proc, interval, stop, sink):
    """Poll nvidia-smi for GPU memory attributable to the child process tree."""
    while not stop.is_set():
        try:
            out = subprocess.run(
                ['nvidia-smi', '--query-compute-apps=pid,used_memory',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=10).stdout
            pids = {int(p) for p in
                    subprocess.run(['ps', '-o', 'pid=', '--ppid', str(proc.pid),
                                    '-p', str(proc.pid)],
                                   capture_output=True, text=True
                                   ).stdout.split()}
            for line in out.strip().splitlines():
                pid_s, _, mem_s = line.partition(',')
                try:
                    pid, mem = int(pid_s.strip()), float(mem_s.strip())
                except ValueError:
                    continue
                if pid in pids:
                    sink.append(mem)
        except Exception:
            pass
        stop.wait(interval)


def run(args):
    tmp = tempfile.mkdtemp(prefix='vramhook_')
    with open(os.path.join(tmp, 'sitecustomize.py'), 'w') as f:
        f.write(HOOK)
    env = dict(os.environ)
    env['PYTHONPATH'] = tmp + os.pathsep + env.get('PYTHONPATH', '')

    t0 = time.time()
    proc = subprocess.Popen(args.cmd, env=env)
    samples, stop = [], threading.Event()
    th = threading.Thread(target=poll_driver,
                          args=(proc, args.interval, stop, samples), daemon=True)
    th.start()
    out, _ = proc.communicate()
    stop.set(); th.join(timeout=5)
    elapsed = time.time() - t0

    torch_stats = None
    if out:
        tag = '__VRAM_TORCH__ '
        if tag in out:
            frag = out.split(tag, 1)[1].split('__', 1)[0]
            try:
                torch_stats = json.loads(frag)
            except Exception:
                pass
    rec = dict(label=args.label, cmd=args.cmd, returncode=proc.returncode,
               elapsed_sec=round(elapsed, 1),
               driver_peak_mib=round(max(samples), 1) if samples else None,
               driver_n_samples=len(samples),
               torch_max_allocated_mib=round(torch_stats['max_memory_allocated'] / 2**20, 1)
               if torch_stats else None,
               torch_max_reserved_mib=round(torch_stats['max_memory_reserved'] / 2**20, 1)
               if torch_stats else None)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'a') as f:
        f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    print(json.dumps(rec, indent=1))
    return proc.returncode


def summarize(path):
    rows = [json.loads(l) for l in open(path)]
    best = {}
    for r in rows:
        b = best.setdefault(r['label'], dict(label=r['label'], runs=0,
                                             driver_peak_mib=0,
                                             torch_max_allocated_mib=0,
                                             torch_max_reserved_mib=0))
        b['runs'] += 1
        for k in ('driver_peak_mib', 'torch_max_allocated_mib',
                  'torch_max_reserved_mib'):
            if r.get(k):
                b[k] = max(b[k], r[k])
    print('| method | runs | torch alloc peak (MiB) | torch reserved peak (MiB) '
          '| driver peak (MiB) |')
    print('|' + '---|' * 5)
    for b in best.values():
        print(f"| {b['label']} | {b['runs']} | {b['torch_max_allocated_mib']} | "
              f"{b['torch_max_reserved_mib']} | {b['driver_peak_mib']} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--label', default='run')
    ap.add_argument('--out', default='outputs/vram/vram_peaks.jsonl')
    ap.add_argument('--interval', type=float, default=0.5)
    ap.add_argument('--summarize', metavar='JSONL')
    ap.add_argument('cmd', nargs=argparse.REMAINDER)
    args = ap.parse_args()
    if args.summarize:
        summarize(args.summarize)
        return 0
    args.cmd = args.cmd[1:] if args.cmd[:1] == ['--'] else args.cmd
    if not args.cmd:
        ap.error('pass a runner command after -- (or use --summarize)')
    return run(args)


if __name__ == '__main__':
    sys.exit(main())
