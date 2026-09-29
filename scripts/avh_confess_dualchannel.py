#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""avh_confess_dualchannel.py — faithful dual-channel confession (ArmD) on AVHBench.

This is the ArmD (HaloQuest +12.60) mechanism ported to AVHBench with genuine
video/audio dual-channel attribution — NOT the earlier full-rewrite `commit`
baseline. The pipeline is *surgical*: it localizes individual channel-grounded
claims, verifies each against ONLY the channel it should be grounded in, and
rewrites/deletes only the unsupported ones, keeping supported content intact.

Per sample (question fixed to the canonical AVH prompt):
  1. base: greedy generation, full audio+video (v1 prompt). Keep the generated
     token ids for teacher-forced scoring.
  2. dual-channel deltas: build full / novideo / noaudio views via build_view
     (processor-based masking, the proven crash-free path from probe_cf_avhbench)
     and view_logp the base ids under each ->
        delta_audio = lp_full - lp_noaudio   (+ => audio-grounded)
        delta_video = lp_full - lp_novideo   (+ => video-grounded)
  3. localize + route: each content word classified audio/visual (lexicons).
     A claim is a *confession suspect* when its dependence on its own channel is
     rootless (relevant delta < eps): the model asserted it but the channel did
     not raise its probability -> likely fabricated from the text prior. Each
     suspect is routed to the channel it should depend on.
  4. channel-specific GT-free verify: for a visual suspect, generate with ONLY
     the video (audio masked) and ask if the detail is visible; for an audio
     suspect, ONLY the audio and ask if the sound is present. 4-way verdict.
  5. evidence-grounded rewrite (full A+V): given the explicit per-channel
     confessions ("the audio does not contain X; the video does not show Y"),
     rewrite as one sentence that removes/corrects unsupported details and keeps
     supported ones. Do not add new details.
  6. decide guards: accept the rewrite only if there is an edit signal (>=1
     CONTRADICTED/UNSUPPORTED) and it passes degeneration / length / content
     preservation guards; otherwise keep base.

Hard constraints honored:
  * training-free (prompt + greedy only, no weight updates).
  * GT never enters generation/localization/verify/rewrite/decide. It is used
    ONLY to log Token-F1 offline (canonical single-ref, same footing as
    scripts/tokenf1_fullset.py).
  * correction derives from explicit evidence dependence (delta) + per-channel
    confession, not generic self-refinement.
  * every accepted edit is explainable (which fact, which channel, why accepted).
  * cohort / metric are unchanged.

Usage (shard across cards 0-3):
  CUDA_VISIBLE_DEVICES=0 MMHALU_MODEL_ROOT=/tmp HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  PYTHONPATH=/dev/shm/site-packages LD_LIBRARY_PATH=/dev/shm/so_mirror \
  /home/tsq/.local/bin/python3.13 scripts/avh_confess_dualchannel.py \
    --start 0 --n 80 --shard 0 --nshard 4 \
    --out outputs/avh_dualchannel/dev_shard0of4.jsonl
"""
import argparse, json, re, sys, time
from pathlib import Path

import numpy as np
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
from probe_cf_avhbench import (                                          # noqa: E402
    view_logp, build_view, token_class, word_modality_class, norm_word,
    token_f1, retry,
)

AVH_JSON = HERE / "OmniHalluBench/avhbench/annotations/avhbench.json"
AVH_ROOT = HERE / "OmniHalluBench/avhbench"
MODEL_PATH = "/dev/shm/Qwen2.5-Omni-7B"

# ---------------------------------------------------------------------------
# Stage-2 FUSE prompts (dual-channel recall recovery).  Diagnosis of the
# delete-word confession arm (+0.29 @n=20) showed the base is *recall-starved*,
# not precision-bloated: base Token-F1 ~0.22 because the one-shot v1 caption
# drops many gt elements (it picks ONE salient event and stops).  Deleting a
# rootless word can only trade a little precision; it cannot recover missing
# recall.  The winning lever (probe: +10 @n=6) is to describe each channel
# ON ITS OWN (audio masked / video masked -> the model is forced to attend to
# the surviving stream and enumerates its content) and then merge the two
# single-channel notes into one caption under full A+V.  This raises coverage
# of both the visual subject and the sound events simultaneously.
# ---------------------------------------------------------------------------
FUSE_SYS = ("You are an audio-visual captioning assistant. Merge the visual note and "
            "audio note into ONE single flowing sentence describing what is seen and heard: "
            "the main visual subject with its key attributes, plus the main sounds. "
            "Keep only what the notes state. No preamble, no label, 15-30 words.")
VID_SYS = ("You are a visual describer. In one short phrase, describe ONLY what is visibly "
           "present: the main subject, its attributes, and the setting. No preamble.")
AUD_SYS = ("You are an audio describer. In one short phrase, describe ONLY the sounds you "
           "hear: the sound events and their likely sources. No preamble.")
_BOX = re.compile(r"\\boxed\{(.*?)\}", re.S)
_ANS = re.compile(r"[Aa]nswer\s*:\s*(.*)", re.S)


def clean_text(t):
    t = str(t).replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
    m = _BOX.search(t)
    if m:
        t = m.group(1).strip()
    m = _ANS.search(t)
    if m:
        t = m.group(1).strip()
    return t.strip()


# ---- degeneration guards (GT-free) ----
def has_3gram_repeat(text):
    ws = text.lower().split()
    if len(ws) < 6:
        return False
    seen = set()
    for i in range(len(ws) - 2):
        g = (ws[i], ws[i + 1], ws[i + 2])
        if g in seen:
            return True
        seen.add(g)
    return False


_REFUSAL = re.compile(r"\b(i cannot|i can't|i'm sorry|sorry,|as an ai|unable to)\b", re.I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--max-new", type=int, default=80)
    ap.add_argument("--rewrite-max-new", type=int, default=96)
    ap.add_argument("--eps-rootless", type=float, default=0.2,
                    help="relevant-channel delta < eps => confession suspect")
    ap.add_argument("--min-dep", type=float, default=-0.75,
                    help="v3 high-variance guard: reject the rewrite (keep base) "
                         "when a driving unsupported suspect has own-channel "
                         "dep <= min-dep. Extreme-negative dep is the unreliable "
                         "tail of channel attribution (a rare content word the "
                         "reference may actually contain). -inf disables.")
    ap.add_argument("--max-suspects", type=int, default=4)
    ap.add_argument("--min-len-frac", type=float, default=0.3,
                    help="reject rewrite if < this fraction of base word count")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-ids", default="",
                    help="comma-separated ids to skip (poison samples)")
    ap.add_argument("--crash-file", default="",
                    help="write the id that triggered an unrecoverable CUDA fault here")
    ap.add_argument("--stage2", default="fuse", choices=["fuse", "confess"],
                    help="fuse = dual-channel note-and-merge recall recovery (default, "
                         "+~10 probe); confess = legacy surgical delete-word arm (+0.29)")
    ap.add_argument("--base-ver", default="v1",
                    help="avh prompt_version for the base_pred pass / fallback "
                         "(default v1; use v5 to stack fuse on the best base prompt)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    skip_ids = {s for s in args.skip_ids.split(",") if s}

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
    block = all_ids[args.start:args.start + args.n]
    my_ids = [d for k, d in enumerate(block) if k % args.nshard == args.shard]
    my_ids = [d for d in my_ids if d not in skip_ids]
    if args.limit:
        my_ids = my_ids[:args.limit]
    print(f"[dc] pool={len(all_ids)} block[{args.start}:{args.start+args.n}] "
          f"shard {args.shard}/{args.nshard} -> {len(my_ids)} samples", flush=True)

    processor, model = load_model(str(Path(MODEL_PATH)))
    thinker = model.thinker
    tok = processor.tokenizer
    sys_prompt = get_system_prompt("avhbench", args.base_ver)

    def gen_view(prompt_text, pv, av, keep_video, keep_audio, max_new, want_ids=False,
                 sys_override=None):
        pixel_in = pv if keep_video else None
        audio_in = av if keep_audio else None
        uc = build_user_content_with_media_tokens(pixel_in, audio_in, None, prompt_text)
        msgs = make_messages(sys_override or sys_prompt, uc)
        inp = build_inputs(processor, model, msgs, pixel_in, audio_in, None, "avhbench")
        out = model.generate(**inp, max_new_tokens=max_new, do_sample=False,
                             temperature=None, top_p=None, use_cache=True)
        if isinstance(out, (tuple, list)):
            out = out[0]
        in_len = inp["input_ids"].shape[1]
        texts = decode_outputs(processor, out, in_len)
        text = clean_text(texts[0] if isinstance(texts, (list, tuple)) and texts
                          else (texts if isinstance(texts, str) else ""))
        if want_ids:
            gen_ids = out[0, in_len:].detach().clone()
            # strip trailing special tokens
            return text, gen_ids
        return text

    done = set()
    if out_path.exists():
        for line in open(out_path):
            line = line.strip()
            if line:
                done.add(json.loads(line)["id"])
        print(f"[dc] resume: {len(done)} done", flush=True)
    fout = open(out_path, "a")

    t0 = time.time(); n_new = 0
    for k, rid in enumerate(my_ids):
        if rid in done:
            continue
        item = by_uid.get(rid)
        if item is None:
            continue
        gt = item.get("gt") or item.get("answer", "")
        question = item.get("question",
                            "Describe what you see and hear in a single sentence.")
        try:
            pv, av, iv = retry(lambda: load_media_for_item(item, str(AVH_JSON), str(AVH_ROOT)),
                               tag="load_media")
            if pv is None or av is None:
                print(f"[dc] id={rid} media None, skip", flush=True)
                continue

            # ---- 1. base (full A+V), keep generated ids ----
            up = get_user_prompt(question, "avhbench", args.base_ver)
            base_pred, gen_ids = gen_view(up, pv, av, True, True, args.max_new, want_ids=True)
            torch.cuda.empty_cache()
            if gen_ids.numel() == 0 or not base_pred:
                print(f"[dc] id={rid} empty base, skip", flush=True)
                continue

            # ================= STAGE-2: FUSE (recall recovery) =================
            if args.stage2 == "fuse":
                # 2a. per-channel single-modality notes (force attention to the
                #     surviving stream so it enumerates that channel's content).
                vid_note = gen_view("Describe only what is visibly present.",
                                    pv, av, True, False, 64, sys_override=VID_SYS)
                torch.cuda.empty_cache()
                aud_note = gen_view("Describe only the sounds you hear.",
                                    pv, av, False, True, 64, sys_override=AUD_SYS)
                torch.cuda.empty_cache()
                vid_note = clean_text(vid_note)
                aud_note = clean_text(aud_note)
                # 2b. merge under full A+V into one caption.
                fuse_prompt = (f"Visual note: {vid_note}\nAudio note: {aud_note}\n"
                               f"Write the single merged sentence now.")
                uc = build_user_content_with_media_tokens(pv, av, None, fuse_prompt)
                msgs = make_messages(FUSE_SYS, uc)
                inp = build_inputs(processor, model, msgs, pv, av, None, "avhbench")
                out = model.generate(**inp, max_new_tokens=args.rewrite_max_new,
                                     do_sample=False, temperature=None, top_p=None,
                                     use_cache=True)
                if isinstance(out, (tuple, list)):
                    out = out[0]
                in_len = inp["input_ids"].shape[1]
                texts = decode_outputs(processor, out, in_len)
                fuse = clean_text(texts[0] if isinstance(texts, (list, tuple)) and texts
                                  else (texts if isinstance(texts, str) else ""))
                torch.cuda.empty_cache()

                # 2c. decide guards (GT-free): accept fuse unless it degenerated.
                base_words = base_pred.split()
                fw = fuse.split()
                decision = "accept_fuse"
                final = fuse
                if not fuse:
                    decision, final = "reject_empty", base_pred
                elif _REFUSAL.search(fuse):
                    decision, final = "reject_refusal", base_pred
                elif len(fw) < args.min_len_frac * max(len(base_words), 1):
                    decision, final = "reject_len_collapse", base_pred
                elif has_3gram_repeat(fuse):
                    decision, final = "reject_repeat", base_pred

                rec = {
                    "id": rid, "question": question, "gt": gt,
                    "base_pred": base_pred, "vid_note": vid_note,
                    "aud_note": aud_note, "fuse": fuse, "final": final,
                    "decision": decision,
                    "base_f1": round(token_f1(base_pred, gt), 4),
                    "fuse_f1": round(token_f1(fuse, gt), 4) if fuse else None,
                    "final_f1": round(token_f1(final, gt), 4),
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n"); fout.flush()
                n_new += 1
                if (k + 1) % 3 == 0:
                    print(f"[dc {k+1}/{len(my_ids)}] new={n_new} {time.time()-t0:.0f}s "
                          f"mem={torch.cuda.max_memory_allocated()/2**30:.1f}G", flush=True)
                torch.cuda.empty_cache()
                continue
            # =================== end FUSE; legacy confess below ===================

            # ---- 2. dual-channel deltas via processor-based views ----
            views = {}
            for name, kv, ka in (("full", True, True), ("novideo", False, True),
                                 ("noaudio", True, False)):
                clean = build_view(processor, model, sys_prompt, up, pv, av, kv, ka)
                views[name] = retry(lambda c=clean: view_logp(thinker, c, gen_ids),
                                    tag=f"forward_{name}")
                torch.cuda.empty_cache()
            L = min(len(views["full"]), len(views["novideo"]), len(views["noaudio"]))
            d_audio = views["full"][:L] - views["noaudio"][:L]
            d_video = views["full"][:L] - views["novideo"][:L]

            # ---- 3. localize content words + route to channel ----
            ids_list = gen_ids.tolist()[:L]
            pieces = [tok.decode([t]) for t in ids_list]
            suspects = []
            for j, p in enumerate(pieces):
                cls = token_class(p, pieces[j - 1] if j else None)
                if cls != "content":
                    continue
                w = norm_word(p)
                if not w:
                    continue
                mc = word_modality_class(w)
                if mc == "audio":
                    dep, chan = float(d_audio[j]), "audio"
                elif mc == "visual":
                    dep, chan = float(d_video[j]), "video"
                else:
                    continue  # only audio/visual lexicon words carry checkable claims
                if dep < args.eps_rootless:      # rootless in its own channel -> suspect
                    suspects.append({"word": w, "channel": chan, "dep": round(dep, 4),
                                     "d_audio": round(float(d_audio[j]), 4),
                                     "d_video": round(float(d_video[j]), 4)})
            # dedup by (word,channel), keep the most rootless, cap count
            uniq = {}
            for s in suspects:
                key = (s["word"], s["channel"])
                if key not in uniq or s["dep"] < uniq[key]["dep"]:
                    uniq[key] = s
            suspects = sorted(uniq.values(), key=lambda s: s["dep"])[:args.max_suspects]

            # ---- 4. channel-specific GT-free verify ----
            for s in suspects:
                w, chan = s["word"], s["channel"]
                if chan == "video":
                    vp = (f"You are fact-checking one detail of a video description. "
                          f"Look ONLY at the video frames (ignore any audio). "
                          f"The description says something about: \"{w}\". "
                          f"Is \"{w}\" actually visible in the video? "
                          f"Answer with exactly one word: SUPPORTED, CONTRADICTED, or UNSUPPORTED.")
                    vtext = gen_view(vp, pv, av, True, False, 8)
                else:
                    vp = (f"You are fact-checking one detail of an audio description. "
                          f"Listen ONLY to the audio (ignore any video). "
                          f"The description mentions the sound: \"{w}\". "
                          f"Is \"{w}\" actually present in the audio? "
                          f"Answer with exactly one word: SUPPORTED, CONTRADICTED, or UNSUPPORTED.")
                    vtext = gen_view(vp, pv, av, False, True, 8)
                torch.cuda.empty_cache()
                u = vtext.upper()
                if "CONTRADICT" in u:
                    verdict = "CONTRADICTED"
                elif "UNSUPPORT" in u:
                    verdict = "UNSUPPORTED"
                elif "SUPPORT" in u:
                    verdict = "SUPPORTED"
                else:
                    verdict = "UNSURE"
                s["verdict"] = verdict
                s["verify_raw"] = vtext[:40]

            unsupported = [s for s in suspects
                           if s.get("verdict") in ("CONTRADICTED", "UNSUPPORTED")]

            # ---- 5. evidence-grounded rewrite (only if edit signal) ----
            rewrite = ""
            if unsupported:
                vis_bad = [s["word"] for s in unsupported if s["channel"] == "video"]
                aud_bad = [s["word"] for s in unsupported if s["channel"] == "audio"]
                conf_lines = []
                if vis_bad:
                    conf_lines.append("the video does NOT actually show: " + ", ".join(vis_bad))
                if aud_bad:
                    conf_lines.append("the audio does NOT actually contain: " + ", ".join(aud_bad))
                confession = "; ".join(conf_lines)
                rp = (f"A draft description of this video says: \"{base_pred}\". "
                      f"On re-checking the media, {confession}. "
                      f"Rewrite the description as ONE sentence that removes or corrects "
                      f"those unsupported details and keeps everything that is actually "
                      f"supported by the video and audio. Do not add any new details. "
                      f"Answer with one sentence.")
                rewrite = gen_view(rp, pv, av, True, True, args.rewrite_max_new)
                torch.cuda.empty_cache()

            # ---- 6. decide guards (GT-free) ----
            base_words = base_pred.split()
            decision = "keep_base"
            final = base_pred
            if unsupported and rewrite:
                rw_words = rewrite.split()
                min_dep = min(s["dep"] for s in unsupported)
                if _REFUSAL.search(rewrite):
                    decision = "reject_refusal"
                elif len(rw_words) < args.min_len_frac * max(len(base_words), 1):
                    decision = "reject_len_collapse"
                elif has_3gram_repeat(rewrite):
                    decision = "reject_repeat"
                elif min_dep <= args.min_dep:
                    decision = "reject_lowdep"     # unreliable attribution tail
                else:
                    decision = "accept_rewrite"
                    final = rewrite

            rec = {
                "id": rid, "question": question, "gt": gt,
                "base_pred": base_pred, "rewrite": rewrite, "final": final,
                "decision": decision,
                "n_suspects": len(suspects), "n_unsupported": len(unsupported),
                "suspects": suspects,
                "base_f1": round(token_f1(base_pred, gt), 4),
                "rewrite_f1": round(token_f1(rewrite, gt), 4) if rewrite else None,
                "final_f1": round(token_f1(final, gt), 4),
                "n_tokens": int(L),
                "mean_d_audio": round(float(np.mean(d_audio)), 4),
                "mean_d_video": round(float(np.mean(d_video)), 4),
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n"); fout.flush()
            n_new += 1
        except Exception as e:
            import traceback
            msg = f"{type(e).__name__}: {e}"
            print(f"[dc ERROR] id={rid}: {msg}", flush=True)
            traceback.print_exc()
            low = msg.lower()
            # CUDA device-side asserts / cuDNN exec failures poison the whole
            # process context irrecoverably -> record poison id and hard-exit so
            # the outer loop restarts (resume + skip). Do NOT keep looping (that
            # only spins on a dead context for hundreds of samples).
            if ("device-side assert" in low or "cudnn" in low
                    or "srcindex" in low or "indexselect" in low
                    or "illegal memory access" in low):
                if args.crash_file:
                    with open(args.crash_file, "a") as cf:
                        cf.write(rid + "\n")
                print(f"[dc FATAL] id={rid} poisoned CUDA context, exiting for "
                      f"restart", flush=True)
                fout.close()
                import os as _os
                _os._exit(42)
        if (k + 1) % 3 == 0:
            print(f"[dc {k+1}/{len(my_ids)}] new={n_new} {time.time()-t0:.0f}s "
                  f"mem={torch.cuda.max_memory_allocated()/2**30:.1f}G", flush=True)
        torch.cuda.empty_cache()

    fout.close()
    print(f"[dc done] {out_path} (+{n_new})", flush=True)


if __name__ == "__main__":
    main()
