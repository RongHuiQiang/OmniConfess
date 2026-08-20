# -*- coding: utf-8 -*-
"""bon_grounding.py — Best-of-N assertion-grounding rerank (描述类在线算子).

Motivation (probe_bon_sc_rerank.py, outputs/bon_probe/ragtruth_bon_sc_probe.json):
six rounds of decoding-side EDITING failed on descriptive tasks; the surviving
lever is SELECTION. Offline probe on the recorded base_sc rollouts showed:

  dispersion      : median cross-rollout range of the grounding score 0.76-1.04
                    (only 0.5-1% of samples have range < 0.1) -> rankable
  rerank gain     : F1(argmax grounding) 38.75-39.71 vs random 35.62 /
                    SC majority vote 35.38 (paired boot CI vs random
                    [+2.87, +5.28] for g_content, n=435) -> gate ALIVE
  no logit war    : unlike CAD there is no per-step logit subtraction; the
                    contrast view only ever SCORES candidates post-hoc.

Operator: generate 1 greedy + 3 temperature-0.7 candidates, teacher-force each
under a correct view and a contrast view (ragtruth: WRONG passage borrowed from
another sample; haloquest: SWAPPED image), compute assertion-level
Delta = logP(correct) - logP(contrast) per token, aggregate to a rollout
grounding score, and OUTPUT the most grounded candidate.

Ranking score (probe winner, pre-fixed before this run):
  g_content = mean Delta over content/num tokens of the whole rollout
  fallback chain g_content -> g_traj_mean (rollouts with no content tokens).

A random-selection control arm is derived from the same rollouts and written to
<file_name>_random_sel.jsonl (pred = random candidate, same seed for both).

Usage (mirrors base_sc.py / base_model.py CLI):
    CUDA_VISIBLE_DEVICES=2 python bon_grounding.py --datasets ragtruth \
        --data_path ./OmniHalluBench/ragtruth/ragtruth.json \
        --output_dir outputs/ragtruth_bon_grounding --file_name ragtruth_bon_grounding
    CUDA_VISIBLE_DEVICES=2 python bon_grounding.py --datasets haloquest \
        --data_path ./OmniHalluBench/haloquest/haloquest.json \
        --output_dir outputs/haloquest_bon_grounding --file_name haloquest_bon_grounding
"""
import os, sys, time, argparse, random, json, string
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'OmniConfess', 'methods'))
from shared import (  # noqa: E402
    load_dataset, load_media_for_item, resolve_media_path,
    load_model, build_inputs, prepare_for_thinker,
    build_user_content_with_media_tokens, make_messages,
    get_system_prompt, get_user_prompt,
    parse_pred, make_result, save_result,
)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault("MMHALU_MODEL_ROOT", "/ultrafast_share/tsq/MODELS")

# ---------------------------------------------------------------------------
# Word classification + assertion segmentation — verbatim from probe_claim_cf.py
# ---------------------------------------------------------------------------
STOPWORDS = set([
    "a", "an", "the", "and", "or", "but", "if", "because", "as", "what",
    "which", "this", "that", "these", "those", "then", "just", "so", "than",
    "such", "both", "through", "about", "for", "is", "are", "was", "were",
    "of", "while", "during", "to", "in", "on", "at", "by", "with", "it", "there",
])
_FUNCTION_WORDS = set("""
i me my myself we us our ours ourselves you your yours yourself yourselves
he him his himself she her hers herself it its itself they them their theirs
themselves what which who whom this that these those am is are was were being
been have has had having do does did doing would should could ought will shall
may might must can cannot not no nor as if because until while of at by for
with about against between into through during before after above below to
from up down in out on off over under again further then once here there when
where why how all any both each few more most other some such only own same
s t don now d ll m o re ve y ain aren couldn didn doesn hadn hasn haven isn
ma mightn mustn needn shan shouldn wasn weren won wouldn also however
therefore thus furthermore moreover additionally another overall
""".split())
_FUNC = STOPWORDS | _FUNCTION_WORDS


def _is_word_initial(tok_str, prev_tok):
    if tok_str[:1].isspace():
        return True
    if prev_tok is None:
        return True
    return not (prev_tok[-1] if prev_tok else ' ').isalnum()


def token_class(tok_str, prev_tok=None):
    if not tok_str:
        return 'punct'
    s = tok_str.strip()
    if not s:
        return 'punct'
    if not _is_word_initial(tok_str, prev_tok):
        return 'cont'
    w = s.lower().strip(string.punctuation + ' ')
    if not w:
        return 'punct'
    if any(c.isdigit() for c in w):
        return 'num'
    if not w[0].isalpha():
        return 'punct'
    if w in _FUNC:
        return 'func'
    return 'content'


def _has_content(cls_list, lo, hi, k=1):
    return sum(1 for c in cls_list[lo:hi] if c in ('content', 'num')) >= k


def claim_spans(toks, cls):
    raw, start = [], 0
    for i, t in enumerate(toks):
        if t == '' or '\n' in t:
            if i > start:
                raw.append((start, i))
            start = i + 1
            continue
        s = t.strip()
        if s and s[-1] in '.!?':
            raw.append((start, i + 1))
            start = i + 1
    if start < len(toks):
        raw.append((start, len(toks)))

    out = []
    for lo, hi in raw:
        segs, s0 = [], lo
        for j in range(lo, hi):
            st = toks[j].strip()
            if st and st[-1] in ',;':
                if _has_content(cls, s0, j) and _has_content(cls, j + 1, hi):
                    segs.append((s0, j + 1))
                    s0 = j + 1
        segs.append((s0, hi))
        for a, b in segs:
            if _has_content(cls, a, b, k=2):
                out.append((a, b))
    return out


# ---------------------------------------------------------------------------
# Teacher-forced view logp — verbatim from probe_claim_cf.py
# ---------------------------------------------------------------------------
@torch.inference_mode()
def view_logp(thinker, clean, gen_ids):
    input_len = clean['input_ids'].shape[1]
    full = dict(clean)
    full['input_ids'] = torch.cat([clean['input_ids'], gen_ids.unsqueeze(0)], dim=1)
    full['attention_mask'] = torch.ones_like(full['input_ids'])
    out = thinker(**full, use_cache=False)
    logits = out.logits[0].float()
    lsm = torch.log_softmax(logits, dim=-1)
    pos = torch.arange(input_len - 1, full['input_ids'].shape[1] - 1, device=logits.device)
    lp = lsm[pos, gen_ids.to(logits.device)]
    return lp.cpu().numpy()


def build_view(processor, model, system_prompt, user_prompt, dataset_name, image=None):
    user_content = build_user_content_with_media_tokens(None, None, image, user_prompt)
    messages = make_messages(system_prompt, user_content)
    raw = build_inputs(processor, model, messages, None, None, image, dataset_name)
    return prepare_for_thinker(raw, model)


# ---------------------------------------------------------------------------
# Rollout grounding score — same aggregators as probe_bon_sc_rerank.py
# ---------------------------------------------------------------------------
def rollout_scores(delta, spans, cls):
    if not spans:
        return None
    claim_dmean = [float(np.sum(delta[lo:hi])) / (hi - lo) for lo, hi in spans]
    cw = [j for j in range(len(delta)) if cls[j] in ('content', 'num')]
    return {
        'g_mean_claim': float(np.mean(claim_dmean)),
        'g_worst_claim': float(np.min(claim_dmean)),
        'g_content': (float(np.sum(delta[cw])) / len(cw)) if cw else None,
        'g_traj_mean': float(np.sum(delta) / len(delta)),
    }


def rank_value(scores, key):
    """Ranking value with the pre-fixed fallback chain."""
    if scores is None:
        return -1e18
    v = scores.get(key)
    if v is None:
        v = scores.get('g_traj_mean')
    return -1e18 if v is None else v


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets',       type=str, required=True)
    parser.add_argument('--data_path',      type=str, required=True)
    parser.add_argument('--data_root',      type=str, default='./OmniHalluBench/')
    parser.add_argument('--output_dir',     type=str, default='./outputs')
    parser.add_argument('--file_name',      type=str, default='bon_grounding_out')
    parser.add_argument('--time_path',      type=str, default='./outputs/time')
    parser.add_argument('--seed',           type=int, default=42)
    parser.add_argument('--num_sampled',    type=int, default=3,
                        help='sampled candidates on top of the greedy one')
    parser.add_argument('--temperature',    type=float, default=0.7)
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--prompt_version', type=str, default='v1', choices=['v0','v1','v2','v3'])
    parser.add_argument('--rank_score',     type=str, default='g_content',
                        choices=['g_content', 'g_mean_claim', 'g_worst_claim', 'g_traj_mean'])
    parser.add_argument('--dry_run',        action='store_true')
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.time_path, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.file_name + '.jsonl')

    done_ids = set()
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line.strip())['id'])
                except Exception:
                    pass
        print(f'[resume] skipping {len(done_ids)} samples', flush=True)

    print('[model] loading...', flush=True)
    processor, model = load_model()
    tokenizer = processor.tokenizer
    thinker = model.thinker

    data = load_dataset(args.data_path, args.data_root)
    if args.dry_run:
        data = data[:3]
    ds = args.datasets
    system_prompt = get_system_prompt(ds, args.prompt_version)

    # contrast-view partner pools (same construction as probe_claim_cf.py)
    pool_passages = [it['passage'] for it in data if it.get('passage')]
    pool_images = [it for it in data if it.get('image_path') and os.path.exists(
        resolve_media_path(it['image_path'], os.path.dirname(args.data_path), args.data_root))]
    print(f'[pool] passages={len(pool_passages)} images={len(pool_images)}', flush=True)

    rng = random.Random(args.seed + 7)          # contrast-view draws
    rng_rand = random.Random(args.seed + 13)    # random-selection control arm
    all_input_tokens = all_output_tokens = 0
    start = time.time()

    for idx, item in enumerate(data):
        if item['id'] in done_ids:
            continue
        t0 = time.time()
        print(f'\n[{idx+1}/{len(data)}] id={item["id"]}', flush=True)

        candidate_texts, candidate_ids, candidate_preds = [], [], []
        scores_list, sel_err = [], None
        try:
            # ---- correct view ------------------------------------------------
            pixel_values, audio_values, image_values = load_media_for_item(
                item, args.data_path, args.data_root)
            image = image_values if ds.lower().startswith('halo') else None
            user_prompt = get_user_prompt(item['question'], ds, args.prompt_version,
                                          passage=item.get('passage'))
            clean = build_view(processor, model, system_prompt, user_prompt, ds, image)
            input_len = clean['input_ids'].shape[1]
            all_input_tokens += input_len * (1 + args.num_sampled)

            # ---- contrast view ----------------------------------------------
            if ds.lower() == 'ragtruth':
                partner_psg = None
                while partner_psg is None:
                    cand_p = rng.choice(pool_passages)
                    if cand_p != (item.get('passage') or ''):
                        partner_psg = cand_p
                up_ctrl = get_user_prompt(item['question'], ds, args.prompt_version,
                                          passage=partner_psg)
                ctrl = build_view(processor, model, system_prompt, up_ctrl, ds, None)
                partner_id = 'psg-borrow'
            else:
                partner = None
                while partner is None:
                    cand_it = rng.choice(pool_images)
                    if cand_it['id'] != item['id']:
                        partner = cand_it
                pimg = load_media_for_item(partner, args.data_path, args.data_root)[2]
                ctrl = build_view(processor, model, system_prompt, user_prompt, ds, pimg)
                partner_id = partner['id']

            # ---- generate: 1 greedy + N sampled ------------------------------
            _t_gen_start = time.perf_counter()
            gen_specs = [(False, 1.0)] + [(True, args.temperature)] * args.num_sampled
            for s, (do_sample, temp) in enumerate(gen_specs):
                gen_kwargs = dict(
                    max_new_tokens=args.max_new_tokens,
                    do_sample=do_sample,
                    pad_token_id=tokenizer.eos_token_id,
                    stop_strings=['<end_of_reasoning>'],
                    tokenizer=tokenizer,
                )
                if do_sample:
                    gen_kwargs.update(temperature=temp)
                with torch.inference_mode():
                    out = thinker.generate(**clean, **gen_kwargs)
                gen_ids = out[0][input_len:]
                gen_ids = gen_ids[gen_ids != tokenizer.pad_token_id]
                text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                candidate_texts.append(text)
                candidate_ids.append(gen_ids.tolist())
                candidate_preds.append(parse_pred(text, ds))
                all_output_tokens += len(gen_ids)
                torch.cuda.empty_cache()
            _t_gen = time.perf_counter() - _t_gen_start

            # ---- score every candidate under both views ----------------------
            _t_sel_start = time.perf_counter()
            for gen_ids_list in candidate_ids:
                gen_t = torch.tensor(gen_ids_list, dtype=torch.long, device=model.device)
                lp_full = view_logp(thinker, clean, gen_t)
                lp_ctrl = view_logp(thinker, ctrl, gen_t)
                delta = lp_full - lp_ctrl
                toks = [tokenizer.decode([t], skip_special_tokens=True) for t in gen_ids_list]
                cls = [token_class(t, toks[i - 1] if i else None) for i, t in enumerate(toks)]
                spans = claim_spans(toks, cls)
                scores_list.append(rollout_scores(delta, spans, cls))
            # selection: argmax grounding (ties -> earlier = greedy-first)
            sel = max(range(len(candidate_ids)),
                      key=lambda i: rank_value(scores_list[i], args.rank_score))
            rand_sel = rng_rand.randrange(len(candidate_ids))
            _t_sel = time.perf_counter() - _t_sel_start

            sel_err = None
        except Exception as e:
            sel_err = str(e)
            print(f'  [ERROR] {e}', flush=True)
            torch.cuda.empty_cache()
            if not candidate_texts:
                candidate_texts, candidate_preds = [''], ['']
                candidate_ids, scores_list = [[]], [None]
                sel = rand_sel = 0
            else:
                sel = rand_sel = 0
            _t_gen = _t_sel = 0.0

        elapsed = time.time() - t0
        final_pred = candidate_preds[sel]
        print(f'  sel={sel}/{len(candidate_texts)-1} pred[:80]={final_pred[:80]!r} '
              f'err={sel_err} time={elapsed:.1f}s', flush=True)

        result = make_result(item, candidate_texts[sel], ds, 'bon_grounding',
                             time_sec=elapsed,
                             cot=json.dumps(candidate_texts, ensure_ascii=False))
        result['pred'] = final_pred
        result['generation_time_seconds'] = round(_t_gen, 4)
        result['selection_time_seconds'] = round(_t_sel, 4)
        result['record_process'] = True
        result['trajectories'] = [
            {'rollout': i, 'sampled': bool(i > 0), 'token_ids': candidate_ids[i],
             'pred': candidate_preds[i], 'grounding_scores': scores_list[i]}
            for i in range(len(candidate_texts))]
        result['selected_rollout'] = int(sel)
        result['random_sel_rollout'] = int(rand_sel)
        result['rank_score'] = args.rank_score
        result['partner'] = partner_id if 'partner_id' in dir() else None
        result['error'] = sel_err
        save_result(result, output_path)
        torch.cuda.empty_cache()

    total_time = time.time() - start
    with open(os.path.join(args.time_path, args.file_name + '.txt'), 'w') as f:
        f.write(f'total_time: {total_time:.2f}s\n')
        f.write(f'greedy + num_sampled: {args.num_sampled} @ T={args.temperature}\n')
        f.write(f'total_input_tokens: {all_input_tokens}\n')
        f.write(f'total_output_tokens: {all_output_tokens}\n')

    # ---- random-selection control arm (same rollouts, pred = random pick) ----
    ctrl_path = os.path.join(args.output_dir, args.file_name + '_random_sel.jsonl')
    n_lines = 0
    with open(output_path) as fin, open(ctrl_path, 'w') as fout:
        for line in fin:
            r = json.loads(line)
            rs = r.get('random_sel_rollout', 0)
            trajs = r.get('trajectories') or []
            if rs < len(trajs):
                r['raw_output'] = trajs[rs].get('pred', '')
                r['pred'] = trajs[rs].get('pred', '')
            r['method'] = 'bon_grounding_random_sel'
            fout.write(json.dumps(r, ensure_ascii=False) + '\n')
            n_lines += 1
    print(f'\n[done] {total_time:.1f}s | {output_path} | control arm {n_lines} lines -> {ctrl_path}')


if __name__ == '__main__':
    main()
