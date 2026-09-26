#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""avh_holdout_gen.py — held-out generator comparison (no confidence path).

Selection by confidence/consensus is already refuted on dev, so this held-out
confirmation drops the view_logp/manual-prefix path entirely (that path triggers
an intermittent CUDA indexSelectLargeIndex assert on large-media items). We only
generate base + the three grounded candidates and log canonical Token-F1 (GT
offline only). Goal: confirm on disjoint ids that no generator beats base and the
candidate oracle headroom stays small.
"""
import argparse, json, sys, time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "OmniConfess"))
try:
    import transformers.utils.import_utils as _tiu
    _tiu.check_torch_load_is_safe = lambda: None
except Exception:
    pass

from methods.shared.model_utils import (            # noqa: E402
    load_model, build_inputs, make_messages,
    build_user_content_with_media_tokens, decode_outputs,
)
from methods.shared.prompts import get_system_prompt, get_user_prompt   # noqa: E402
from methods.shared.data_loader import load_dataset, load_media_for_item  # noqa: E402
from probe_cf_avhbench import retry                                     # noqa: E402
from scripts.avh_confess_select import (                               # noqa: E402
    token_f1, clean_text, CAND_PROMPTS, AVH_JSON, AVH_ROOT,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-start", type=int, default=120)
    ap.add_argument("--dev-n", type=int, default=100)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--max-new", type=int, default=80)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    avh_items = load_dataset(str(AVH_JSON), str(AVH_ROOT))
    by_uid = {}
    for it in avh_items:
        uid = it.get("uid", "")
        try:
            by_uid[str(int(uid.split("_")[-1]))] = it
        except (ValueError, IndexError):
            pass
    all_ids = sorted(by_uid.keys(), key=lambda x: int(x))
    dev_ids = all_ids[args.dev_start:args.dev_start + args.dev_n]
    my_ids = [d for k, d in enumerate(dev_ids) if k % args.nshard == args.shard]
    print(f"[ho] pool={len(all_ids)} block[{args.dev_start}:{args.dev_start+args.dev_n}] "
          f"shard {args.shard}/{args.nshard} -> {len(my_ids)}", flush=True)

    processor, model = load_model(str(Path("/dev/shm/Qwen2.5-Omni-7B")))
    sys_prompt = get_system_prompt("avhbench", "v1")

    done = set()
    if out_path.exists():
        for line in open(out_path):
            line = line.strip()
            if line:
                done.add(json.loads(line)["id"])
    fout = open(out_path, "a")

    def gen(prompt_text, pv, av, iv):
        uc = build_user_content_with_media_tokens(pv, av, iv, prompt_text)
        msgs = make_messages(sys_prompt, uc)
        inp = build_inputs(processor, model, msgs, pv, av, iv, "avhbench")
        out = model.generate(**inp, max_new_tokens=args.max_new, do_sample=False,
                             temperature=None, top_p=None, use_cache=True)
        if isinstance(out, (tuple, list)):
            out = out[0]
        texts = decode_outputs(processor, out, inp["input_ids"].shape[1])
        if isinstance(texts, (list, tuple)):
            return clean_text(texts[0] if texts else "")
        return clean_text(texts)

    t0 = time.time(); n_new = 0
    for k, rid in enumerate(my_ids):
        if rid in done:
            continue
        item = by_uid.get(rid)
        if item is None:
            continue
        gt = item.get("gt") or item.get("answer", "")
        question = item.get("question", "Describe what you see and hear in a single sentence.")
        try:
            pv, av, iv = retry(lambda: load_media_for_item(item, str(AVH_JSON), str(AVH_ROOT)),
                               tag="load_media")
            if pv is None or av is None:
                continue
            base_pred = gen(get_user_prompt(question, "avhbench", "v1"), pv, av, iv)
            torch.cuda.empty_cache()
            cands = {"base": base_pred}
            for name, pf in CAND_PROMPTS:
                try:
                    cands[name] = gen(pf(question, base_pred), pv, av, iv)
                except Exception as e:
                    print(f"[ho] id={rid} gen {name} err {type(e).__name__}: {e}", flush=True)
                    cands[name] = ""
                torch.cuda.empty_cache()
            rec = {"id": rid, "question": question, "gt": gt,
                   "cands": {nm: {"text": t, "f1": round(token_f1(t, gt), 4)}
                             for nm, t in cands.items()}}
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n"); fout.flush()
            n_new += 1
        except Exception as e:
            print(f"[ho ERROR] id={rid}: {type(e).__name__}: {e}", flush=True)
        if (k + 1) % 5 == 0:
            print(f"[ho {k+1}/{len(my_ids)}] new={n_new} {time.time()-t0:.0f}s", flush=True)
        torch.cuda.empty_cache()
    fout.close()
    print(f"[ho done] {out_path} (+{n_new})", flush=True)


if __name__ == "__main__":
    main()
