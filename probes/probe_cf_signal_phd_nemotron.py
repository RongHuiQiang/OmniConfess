#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""probe_cf_signal_phd_nemotron.py -- CF signal probe: phd x Nemotron-3-Nano-Omni-30B.

Third test point of the "generational self-healing" triangle
(Qwen2.5 diseased / Qwen3 self-healed / Nemotron ?). The phd dominance disease
(misleading context dominates the image at the decision token; CF@Qwen2.5
+12.9 F1, probe AUC 0.767) self-healed on Qwen3 (probe AUC 0.550, operator ==
random-sign control p=0.77). Nemotron base phd is 83.14/74.98 -- as strong as
Qwen3 raw (83.40/77.24). Same question, different backbone family: does the
context-pull signal survive on the NVIDIA omni backbone?

Base trajectories (from the [Nemotron 补格] agent, greedy v1 prompt,
enable_thinking=False, per-token logp recorded):
  /ultrafast_share/tsq/tsq/nemotron/outputs/phd_nemotron/phd_nemotron_single_scored.jsonl
  (906 rows; output format "Hypothesis ... Judgment: \\boxed{Yes|No}",
  'box'+'ed'+'{'+'Yes'+'}' tokenization -- the Qwen3 'boxed'-single-token
  locator does NOT apply, see find_decision_index below).

Views at the decision token (teacher-forced on the base greedy prefix,
identical protocol to probe_cf_signal_qwen3.py; masking = the misleading
TEXT, media kept):
  full  : image + context + marker + question
  noctx : image + question only            -> delta_noctx (primary; the
          operator's disease channel: misleading context pull on Yes)
  noimg : context + marker + question      -> delta_noimg (secondary)

Model layer: transformers 5.6.2 trust_remote_code in the nemotron-hf venv
(same six-pitfall repair chain as base_model_nemotron_hf.py: summary_idxs
rebuild, constant buffers, video_pruning_rate=0). The omni wrapper only
exposes generate(); teacher forcing is done by appending the base prefix to
the processor-clean input_ids and calling generate(max_new_tokens=1,
output_scores=True) -- scores[0] is the decision-token distribution.

PRE-REGISTERED JUDGMENT (written before running; conjunction copied from the
cmm-Qwen3 pre-registration, cf. EXPERIMENT_DESIGN.md "CF cross-backbone"):
  - Sample: 100 rows, 25 x (gt_yes/no x base-correct/incorrect), seed 42,
    from scored rows with a locatable decision token.
  - Signals: margin_full, delta_noctx (full - noctx), delta_noimg,
    d_noctx = lp_yes(noctx) - lp_yes(full) (operator convention, masked-full).
  - ALIVE := AUC(-delta_noctx -> gt) > 0.65 AND Qwen2.5-type direction
    structure (all three clauses; the Qwen3 lesson is that AUC alone
    misleads -- polarity reversal can pass AUC while inverting the causal
    story):
      (i) FP mass direction: gt=no & base=yes rows have delta_noctx mean > 0
          (the misleading text is propping the WRONG yes; Qwen2.5 phd cell
          mean +0.92);
     (ii) FN mass direction: gt=yes & base=no rows have delta_noctx mean < 0
          (the misleading text suppresses the CORRECT yes; Qwen2.5 -1.14),
          i.e. masking moves both error masses toward gt (disease = context
          dominates image, not polarity-inverted modality suppression);
    (iii) flippable band exists: median |margin_full| on base-error rows <
          median on base-correct rows (errors sit nearer the decision
          boundary; Qwen2.5 error |m| 3-4 nats vs Qwen3 self-healed 7-9
          nats -- clause fails when errors are made confidently).
  - ALIVE -> full 906 CF@Nemotron-phd (lambda re-selected on this dev-100:
    max Youden s.t. |yes-rate - base| <= 5pp; random-sign control arm
    alongside; online/offline consistency check as on the Qwen3 arm).
  - AUC <= 0.65 OR any direction clause failing -> DEAD: negative result,
    recorded as the third point of the self-healing triangle (family-free
    conclusion: the disease is a weak-backbone property, not a Qwen
    property). Supplementary (reported, not gate-relevant): offline lambda
    grid on the probe records, per-cell means, teacher-forcing reproduction
    check (decision-token lp == base-recorded logp, median |diff| ~ 0).

Usage (one exclusive ~clean card; probe is single-card):
  CUDA_VISIBLE_DEVICES=0 /ultrafast_share/tsq/venvs/nemotron-hf/bin/python \
      probe_cf_signal_phd_nemotron.py --n_per_cell 25
"""
import os, sys, json, time, argparse, random, hashlib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'OmniConfess', 'methods'))
sys.path.insert(0, '/ultrafast_share/tsq/tsq/nemotron')
os.chdir(HERE)

import torch

from shared import load_dataset, get_system_prompt, get_user_prompt
from shared.data_loader import load_image, resolve_media_path
from base_model_nemotron_hf import (
    DEFAULT_MODEL, _repair_buffers, VIDEO_FPS, VIDEO_MAX_FRAMES,
    load_video_frames_native, load_audio_wave,
)

MARKER = ("In case there is an inconsistency between the context and the image "
          "content, you should follow the image.")
# Nemotron tokenizer (verified 2026-08-16): 'Yes'->16860 'No'->4753
# (capitalized, as used inside \boxed{...}); '{'->1123
YES_ID, NO_ID = 16860, 4753
OPEN_ID = 1123


def split_question(q):
    idx = q.find(MARKER)
    if idx < 0:
        return None, None, q
    return q[:idx].rstrip(), MARKER, q[idx + len(MARKER):].lstrip()


def find_decision_index(tokenizer, token_ids):
    """phd Nemotron tail: ... 'box' 'ed' '{' 'Yes'|'No' '}' '<|im_end|>'.
    ('boxed' splits into two tokens on this tokenizer -- NOT the Qwen3
    single-token shape.) Locate the '{' whose successor is Yes/No; bare
    first-token Yes/No kept defensively."""
    ids = list(token_ids)
    if ids and ids[0] in (YES_ID, NO_ID):
        return 0, ids[0]
    for i in range(len(ids) - 2, -1, -1):
        if ids[i] == OPEN_ID and ids[i + 1] in (YES_ID, NO_ID):
            return i + 1, ids[i + 1]
    return None, None


def build_clean(tokenizer, system_prompt, user_text, image):
    """Replicate base_model_nemotron_hf.py's input construction exactly."""
    messages = [{'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_text}]
    try:
        text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
            enable_thinking=False)
    except TypeError:
        text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False)
    kw = dict(text=[text], return_tensors='pt')
    if image is not None:
        kw['images'] = image
    return kw


@torch.inference_mode()
def decision_logp(processor, model, clean_kw, prefix_ids, device):
    """Append the base greedy prefix to the clean input and read the
    decision-token distribution (single generate step)."""
    inputs = processor(**clean_kw)
    inputs = {k: (v.to(device) if torch.is_tensor(v) else v)
              for k, v in inputs.items()}
    ids = torch.cat([inputs['input_ids'][0],
                     torch.tensor(prefix_ids, dtype=torch.long,
                                  device=inputs['input_ids'].device)],
                    dim=0).unsqueeze(0)
    kw = dict(input_ids=ids, attention_mask=torch.ones_like(ids))
    for k in ('pixel_values', 'pixel_values_videos', 'sound_clips',
              'sound_length'):
        if k in inputs and inputs[k] is not None:
            kw[k] = inputs[k]
    out = model.generate(**kw, max_new_tokens=1, do_sample=False,
                         temperature=None, top_p=None, top_k=None,
                         output_scores=True, return_dict_in_generate=True,
                         pad_token_id=processor.tokenizer.pad_token_id or 0)
    logits = out.scores[0][0].float()
    lp = torch.log_softmax(logits, dim=-1)
    return (lp[YES_ID].item(), lp[NO_ID].item(),
            int(logits.argmax().item()), int(inputs['input_ids'].shape[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scored', default='/ultrafast_share/tsq/tsq/nemotron/outputs/phd_nemotron/phd_nemotron_single_scored.jsonl')
    ap.add_argument('--data_path', default='./OmniHalluBench/phd/phd.json')
    ap.add_argument('--data_root', default='./OmniHalluBench/')
    ap.add_argument('--n_per_cell', type=int, default=25)
    ap.add_argument('--full', action='store_true', help='all rows (no sampling)')
    ap.add_argument('--out', default='outputs/cf_probe/phd_probe_nemotron.jsonl')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--model_path', default=os.environ.get(
        'NEMOTRON_MODEL_PATH', DEFAULT_MODEL))
    args = ap.parse_args()

    random.seed(args.seed)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    rows = [json.loads(l) for l in open(args.scored)]
    data = load_dataset(args.data_path, args.data_root)
    by_q = {}
    for it in data:
        by_q.setdefault(it['question'], []).append(it)

    cells = {(1, 1): [], (1, 0): [], (0, 1): [], (0, 0): []}
    for r in rows:
        if not r.get('token_ids'):
            continue
        gy = 1 if r['gt'] == 'yes' else 0
        bc = 1 if r['eval_metrics']['Is_Correct'] == 1 else 0
        cells[(gy, bc)].append(r)
    if args.full:
        sample = [((1 if r['gt'] == 'yes' else 0,
                    1 if r['eval_metrics']['Is_Correct'] == 1 else 0), r)
                  for k in cells for r in cells[k]]
    else:
        sample = []
        for k, v in sorted(cells.items()):
            random.shuffle(v)
            sample += [(k, r) for r in v[:args.n_per_cell]]
    random.shuffle(sample)
    if args.limit:
        sample = sample[:args.limit]
    print(f'[probe] sampled {len(sample)} '
          f'({ {k: min(len(v), args.n_per_cell) for k, v in cells.items()} })',
          flush=True)

    from transformers import AutoModel, AutoProcessor
    print(f'[model] loading {args.model_path} (bf16, ~55GB, one card)...',
          flush=True)
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(args.model_path,
                                              trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model_path, trust_remote_code=True,
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        attn_implementation='eager')
    device = os.environ.get('NEMO_DEVICE', 'cuda:0')
    model.to(device).eval()
    _repair_buffers(model, args.model_path, device)
    if getattr(model, 'video_pruning_rate', 0):
        model.video_pruning_rate = 0.0
    tokenizer = processor.tokenizer
    print(f'[model] loaded in {time.time()-t0:.0f}s', flush=True)

    system_prompt = get_system_prompt('phd', 'v1')
    json_dir = os.path.dirname(os.path.abspath(args.data_path))

    t0 = time.time()
    n_done = 0
    with open(args.out, 'w') as fout:
        for (gy, bc), r in sample:
            try:
                items = by_q.get(r['question'])
                if not items:
                    print(f"  [skip] question not found id={r['id']}", flush=True)
                    continue
                item = items[0]
                ip = resolve_media_path(item.get('image_path'), json_dir,
                                        args.data_root)
                image = load_image(ip)
                if image is None:
                    print(f"  [skip] no image id={r['id']}", flush=True)
                    continue
                prefix_len, dec_tok = find_decision_index(tokenizer,
                                                          r['token_ids'])
                if prefix_len is None:
                    print(f"  [skip] no decision token id={r['id']}", flush=True)
                    continue
                prefix = r['token_ids'][:prefix_len]
                ctx, marker, q_only = split_question(r['question'])
                assert marker

                rec = dict(id=r['id'],
                           q_hash=hashlib.md5(r['question'].encode()).hexdigest()[:12],
                           gt=r['gt'], pred=r['pred'], gt_yes=gy, base_correct=bc,
                           dec_index=prefix_len, dec_token=dec_tok,
                           bare=int(prefix_len == 0),
                           dec_logp_recorded=r['logp_list'][prefix_len])

                u_full = get_user_prompt(r['question'], 'phd', 'v1')
                ly, ln, am, il_f = decision_logp(
                    processor, model, build_clean(
                        tokenizer, system_prompt, '<image>\n' + u_full, image),
                    prefix, device)
                rec.update(lp_yes_full=round(ly, 4), lp_no_full=round(ln, 4),
                           full_argmax=am,
                           full_argmax_is_dec=int(am == dec_tok),
                           margin_full=round(ly - ln, 4),
                           input_len_full=il_f,
                           input_len_matches=int(
                               il_f == r.get('input_len', -1)))

                u_nc = get_user_prompt(q_only, 'phd', 'v1')
                ly3, ln3, am3, _ = decision_logp(
                    processor, model, build_clean(
                        tokenizer, system_prompt, '<image>\n' + u_nc, image),
                    prefix, device)
                rec.update(lp_yes_noctx=round(ly3, 4), lp_no_noctx=round(ln3, 4),
                           margin_noctx=round(ly3 - ln3, 4),
                           delta_noctx=round(ly - ly3, 4))       # full - masked

                u_txt = get_user_prompt(r['question'], 'phd', 'v1')
                ly2, ln2, am2, _ = decision_logp(
                    processor, model, build_clean(
                        tokenizer, system_prompt, u_txt, None),
                    prefix, device)
                rec.update(lp_yes_noimg=round(ly2, 4), lp_no_noimg=round(ln2, 4),
                           margin_noimg=round(ly2 - ln2, 4),
                           delta_noimg=round(ly - ly2, 4))       # full - masked

                # operator convention (masked - full): what the operator adds
                rec.update(d_noctx=round(ly3 - ly, 4),
                           tf_repro=round(
                               (ly if dec_tok == YES_ID else ln)
                               - r['logp_list'][prefix_len], 4))

                fout.write(json.dumps(rec, ensure_ascii=False) + '\n')
                fout.flush()
                n_done += 1
                if n_done % 10 == 0:
                    print(f'  [{n_done}/{len(sample)}] {time.time()-t0:.0f}s '
                          f'd_noctx={rec["delta_noctx"]:+.2f} '
                          f'd_noimg={rec["delta_noimg"]:+.2f} '
                          f'tf={rec["tf_repro"]:+.4f}', flush=True)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"  [ERROR] id={r['id']}: {e}", flush=True)
            # NB: no per-sample empty_cache() -- the warm-claim reservation
            # must stay held for the process lifetime (farm contention; the
            # first attempt lost its card exactly this way).

    print(f'[done] {n_done}/{len(sample)} samples in {time.time()-t0:.0f}s '
          f'-> {args.out}', flush=True)


if __name__ == '__main__':
    main()
