#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""avh_confess_select.py — AVHBench: quality-diverse candidates + confidence-select.

Motivation (all established on the CANONICAL token-F1 metric, GT offline only):
  * Base(645) F1=19.4, balanced P/R, right length -> a *wrong-content* problem.
  * Evidence-dependence (delta_video/delta_audio) does NOT track correctness
    (token gap -0.34, sample r=-0.10) -> dependence-based selection is refuted.
  * Token self-confidence DOES track correctness (token gap +0.22, sample r=+0.23).
  * Temperature / modality-isolation candidates are WORSE than base; only
    quality-diverse *grounded* re-description has shown oracle headroom.

This script tests the one surviving hypothesis: generate a few mechanistically
distinct *grounded* candidates, score each by confidence under identical full-A+V
conditioning, and let offline analysis compare selectors (oracle / confidence /
consensus / margin-gated commit). GT is used ONLY to log per-candidate F1 for
offline evaluation; it never enters generation, scoring, or any commit decision.

Candidates per sample:
  base       : existing base prediction (from avhbench_base_model_full.jsonl)
  redescribe : grounded one-sentence re-description (main seen + main heard)
  commit     : evidence-commit rewrite of the base draft (keep supported, fix wrong)
  salient    : single most-salient seen + single most-prominent heard

Confidence(cand) = mean content-token logp of cand tokens, teacher-forced under
the CANONICAL question prompt with full audio+video (same conditioning for all
candidates, so scores are comparable).

Usage (shard across cards 0-3):
  CUDA_VISIBLE_DEVICES=0 MMHALU_MODEL_ROOT=/tmp HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  PYTHONPATH=/dev/shm/site-packages LD_LIBRARY_PATH=/dev/shm/so_mirror \
  /home/tsq/.local/bin/python3.13 scripts/avh_confess_select.py \
    --dev-start 100 --dev-n 120 --shard 0 --nshard 4 \
    --out outputs/avh_confess_select/dev_shard0of4.jsonl
"""
import argparse, json, os, string, sys, time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "OmniConfess"))

# CVE patch mirror (same as sibling scripts)
try:
    import transformers.utils.import_utils as _tiu
    _tiu.check_torch_load_is_safe = lambda: None
except Exception:
    pass

from methods.shared.model_utils import (            # noqa: E402
    load_model, build_inputs, make_messages,
    build_user_content_with_media_tokens, prepare_for_thinker, decode_outputs,
)
from methods.shared.prompts import get_system_prompt, get_user_prompt   # noqa: E402
from methods.shared.data_loader import load_dataset, load_media_for_item  # noqa: E402
from probe_cf_avhbench import token_class, view_logp, retry             # noqa: E402

AVH_JSON = HERE / "OmniHalluBench/avhbench/annotations/avhbench.json"
AVH_ROOT = HERE / "OmniHalluBench/avhbench"
BASE_PREDS = HERE / "outputs/avhbench_base_model/avhbench_base_model_full.jsonl"

# ---- canonical token-F1 (verbatim footing of scripts/tokenf1_fullset.py) ----
STOP = set("a an the and or but if because as what which this that these those "
           "then just so than such both through about for is are was were of while "
           "during to in on at by with it there".split())


def _toks(s):
    s = str(s).lower().translate(str.maketrans("", "", string.punctuation))
    return [w for w in s.split() if w and w not in STOP]


def token_f1(pred, gt):
    p, g = _toks(pred), _toks(gt)
    if not p or not g:
        return 0.0
    c = Counter(p) & Counter(g)
    h = sum(c.values())
    if h == 0:
        return 0.0
    P, R = h / len(p), h / len(g)
    return 2 * P * R / (P + R)


# ---- grounded candidate prompts (mechanistically distinct, all A+V) ----
def prompt_redescribe(_q, _base):
    return ("Watch the video and listen to the audio. In one plain sentence, "
            "describe the main thing you see happening and the main sound you hear. "
            "Only mention what is clearly present. Answer with one sentence.")


def prompt_commit(_q, base):
    return ("Watch the video and listen to the audio. A draft description says: "
            f"\"{base}\". Rewrite it as one sentence, keeping only the details the "
            "video and audio actually support and correcting anything that is wrong "
            "or not present. Answer with one sentence.")


def prompt_salient(_q, _base):
    return ("Watch the video and listen to the audio. In one short sentence, state "
            "the single most important action or subject you see and the single most "
            "prominent sound you hear.")


CAND_PROMPTS = [("redescribe", prompt_redescribe),
                ("commit", prompt_commit),
                ("salient", prompt_salient)]


def clean_text(t):
    return str(t).replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-start", type=int, default=100)
    ap.add_argument("--dev-n", type=int, default=120)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--max-new", type=int, default=80)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # stored base predictions (only ids 0-80 exist; used for validation overlap)
    base_by_id = {}
    if BASE_PREDS.exists():
        for line in open(BASE_PREDS):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            base_by_id[str(r["id"])] = r   # dedup: last wins (dups are identical)

    # avhbench items keyed by numeric id — the full 645-item pool
    avh_items = load_dataset(str(AVH_JSON), str(AVH_ROOT))
    by_uid = {}
    for it in avh_items:
        uid = it.get("uid", "")
        try:
            by_uid[str(int(uid.split("_")[-1]))] = it
        except (ValueError, IndexError):
            pass

    # dev id block over the full pool, then shard
    all_ids = sorted(by_uid.keys(), key=lambda x: int(x))
    dev_ids = all_ids[args.dev_start:args.dev_start + args.dev_n]
    my_ids = [d for k, d in enumerate(dev_ids) if k % args.nshard == args.shard]
    print(f"[sel] pool={len(all_ids)} dev[{args.dev_start}:{args.dev_start+args.dev_n}] "
          f"shard {args.shard}/{args.nshard} -> {len(my_ids)} samples", flush=True)

    processor, model = load_model(str(Path("/dev/shm/Qwen2.5-Omni-7B")))
    thinker = model.thinker
    tok = processor.tokenizer
    sys_prompt = get_system_prompt("avhbench", "v1")

    done = set()
    if out_path.exists():
        for line in open(out_path):
            line = line.strip()
            if line:
                done.add(json.loads(line)["id"])
        print(f"[sel] resume: {len(done)} done", flush=True)
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

    def confidence(clean_inputs, text):
        """mean content-token logp of `text` under full-A+V canonical conditioning."""
        ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        if ids.numel() == 0:
            return None, 0
        ids = ids.to(model.device)
        lp = view_logp(thinker, clean_inputs, ids)
        pieces = [tok.decode([int(t)]) for t in ids]
        cls = [token_class(p, pieces[j - 1] if j else None) for j, p in enumerate(pieces)]
        content = [lp[j] for j in range(min(len(lp), len(cls))) if cls[j] == "content"]
        if not content:
            content = list(lp)
        return float(np.mean(content)), len(content)

    t0 = time.time()
    n_new = 0
    for k, rid in enumerate(my_ids):
        if rid in done:
            continue
        item = by_uid.get(rid)
        if item is None:
            print(f"[sel] id={rid} no item, skip", flush=True)
            continue
        gt = item.get("gt") or item.get("answer", "")
        question = item.get("question", "Describe what you see and hear in a single sentence.")
        stored_base = clean_text(base_by_id.get(rid, {}).get("pred", "")) if rid in base_by_id else ""
        try:
            pv, av, iv = retry(lambda: load_media_for_item(item, str(AVH_JSON), str(AVH_ROOT)),
                               tag="load_media")
            if pv is None or av is None:
                print(f"[sel] id={rid} media None, skip", flush=True)
                continue

            # base: generate with the canonical v1 user prompt (greedy, deterministic)
            base_pred = gen(get_user_prompt(question, "avhbench", "v1"), pv, av, iv)
            torch.cuda.empty_cache()

            # grounded candidates
            cands = {"base": base_pred}
            for name, pf in CAND_PROMPTS:
                try:
                    cands[name] = gen(pf(question, base_pred), pv, av, iv)
                except Exception as e:
                    print(f"[sel] id={rid} gen {name} err {type(e).__name__}: {e}", flush=True)
                    cands[name] = ""
                torch.cuda.empty_cache()

            # build ONE full-A+V clean-input under canonical question; score all cands
            up = get_user_prompt(question, "avhbench", "v1")
            prefix = "<|vision_bos|><|VIDEO|><|vision_eos|>\n<|audio_bos|><|AUDIO|><|audio_eos|>\n"
            clean = prepare_for_thinker(
                retry(lambda: build_inputs(processor, model,
                                           make_messages(sys_prompt, prefix + up),
                                           pv, av, None, "avhbench"), tag="build_view"),
                model)

            rec = {"id": rid, "question": question, "gt": gt,
                   "base_pred": base_pred, "stored_base": stored_base,
                   "stored_base_f1": round(token_f1(stored_base, gt), 4) if stored_base else None,
                   "cands": {}}
            for name, text in cands.items():
                conf, nctok = (None, 0)
                if text:
                    try:
                        conf, nctok = confidence(clean, text)
                    except Exception as e:
                        print(f"[sel] id={rid} conf {name} err {type(e).__name__}: {e}", flush=True)
                    torch.cuda.empty_cache()
                rec["cands"][name] = {
                    "text": text,
                    "conf": conf,
                    "n_content": nctok,
                    "f1": round(token_f1(text, gt), 4),   # OFFLINE eval only
                }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            n_new += 1
        except Exception as e:
            import traceback
            print(f"[sel ERROR] id={rid}: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
        if (k + 1) % 5 == 0:
            print(f"[sel {k+1}/{len(my_ids)}] new={n_new} {time.time()-t0:.0f}s "
                  f"mem={torch.cuda.max_memory_allocated()/2**30:.1f}G", flush=True)
        torch.cuda.empty_cache()

    fout.close()
    print(f"[sel done] {out_path} (+{n_new})", flush=True)


if __name__ == "__main__":
    main()
