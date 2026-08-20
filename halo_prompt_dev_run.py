# -*- coding: utf-8 -*-
"""halo_prompt_dev_run.py — PROMPT_PROGRAM dev runner for HaloQuest prompt
iteration (v3+): contract prompt version X on the frozen four-layer pipeline.

Frozen mechanism ( NOTHING in this file tunes anything ):
  * routing: v1_evade -> keep v1 pred | gate-flagged -> T3 echo | passthrough
    -> BoN(1 greedy + 3 temp-0.7) ranked by the published feature-combo score
    z(Delta) - 0.25 z(CE) + 2.0 z(VAS)  (BETA/GAMMA + dev-100 z-params from
    bon_stacked_feat_combo.py, untouched)
  * gate flags / thresholds, BoN weights, contrast partner images: all taken
    verbatim from the published run (outputs/haloquest_bon_stacked/
    haloquest_bon_stacked_feat_combo.jsonl)
Only the contract prompt (system+user, get_system_prompt/get_user_prompt
version) changes between runs.

Dev set: OmniHalluBench/haloquest/dev_quarter/dev_ids.json (91 items, fixed).

Non-passthrough dev items still get ONE greedy generation under the new prompt
(contract-layer diagnostic + trajectories for the next round); their four-layer
pred stays the frozen baseline pred by construction.

Usage:
    CUDA_VISIBLE_DEVICES=0 python halo_prompt_dev_run.py --prompt_version v3
"""
import os, sys, json, time, random, argparse

# env first: shared.model_utils resolves MODEL_PATH at import time
os.environ.setdefault("MMHALU_MODEL_ROOT", "/ultrafast_share/tsq/MODELS")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'OmniConfess', 'methods'))
sys.path.insert(0, HERE)
from shared import (  # noqa: E402
    load_dataset, load_media_for_item, resolve_media_path,
    load_model, build_user_content_with_media_tokens, make_messages,
    get_system_prompt, get_user_prompt, parse_pred,
)
from probe_bon_features import forward_with_feats, build_view, evidence_mask  # noqa: E402
from bon_grounding import token_class, claim_spans, view_logp, rollout_scores, set_seed  # noqa: E402

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

BASE = os.path.join(HERE, 'outputs', 'haloquest_bon_stacked',
                    'haloquest_bon_stacked_feat_combo.jsonl')
DEV_IDS = os.path.join(HERE, 'OmniHalluBench', 'haloquest', 'dev_quarter',
                       'dev_ids.json')
BETA, GAMMA = 0.25, 2.0     # frozen: published feat-combo weights


def dev_zpars():
    """dev(100) z-params of the calibrated scorer — verbatim from the shipped
    bon_stacked_feat_combo.py (frozen, NOT re-tuned here)."""
    from bon_stacked_feat_combo import dev_zpars as _dz
    return _dz()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prompt_version', type=str, default='v3')
    ap.add_argument('--num_sampled', type=int, default=3)
    ap.add_argument('--temperature', type=float, default=0.7)
    ap.add_argument('--max_new_tokens', type=int, default=512)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--ids', choices=['dev', 'full'], default='dev',
                    help='dev = fixed 91-item split | full = all 365')
    ap.add_argument('--out_dir', default=os.path.join(HERE, 'outputs',
                                                      'haloquest_prompt_iter'))
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    tag = f'haloquest_prompt_{args.prompt_version}_{args.ids}'
    out_path = os.path.join(args.out_dir, tag + '.jsonl')

    base = {json.loads(l)['id']: json.loads(l) for l in open(BASE)}
    if args.ids == 'dev':
        run_ids = json.load(open(DEV_IDS))
    else:
        run_ids = list(base.keys())
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            try:
                done.add(json.loads(line)['id'])
            except Exception:
                pass
        print(f'[resume] skipping {len(done)}', flush=True)

    data_path = os.path.join(HERE, 'OmniHalluBench', 'haloquest', 'haloquest.json')
    data_root = os.path.join(HERE, 'OmniHalluBench')
    data = load_dataset(data_path, data_root)
    by_id = {it['id']: it for it in data}

    zp = dev_zpars()
    print(f'[z-params frozen] delta {zp[0][0]:.3f}/{zp[0][1]:.3f} '
          f'CE {zp[1][0]:.3f}/{zp[1][1]:.3f} VAS {zp[2][0]:.4f}/{zp[2][1]:.4f}',
          flush=True)

    # model loading mirrors bon_stacked_feat_combo.py (the published feature
    # layer): eager attention is REQUIRED -- VAS reads attentions, which are
    # None under sdpa. shared.load_model() omits attn_implementation.
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
    from shared.paths import resolve_model
    model_path = resolve_model("Qwen/Qwen2.5-Omni-7B")
    processor = Qwen2_5OmniProcessor.from_pretrained(model_path, trust_remote_code=True)
    processor.tokenizer.padding_side = 'left'
    if not processor.tokenizer.pad_token:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map='auto',
        trust_remote_code=True, attn_implementation='eager').eval()
    model.thinker.config.vocab_size = 152064   # same patch as the published run
    tokenizer = processor.tokenizer
    thinker = model.thinker
    system_prompt = get_system_prompt('haloquest', args.prompt_version)
    rng_rand = random.Random(args.seed + 13)

    t0 = time.time()
    n = 0
    for i in run_ids:
        if i in done:
            continue
        rec = base[i]
        item = by_id[i]
        entry = {'id': i, 'question': rec['question'], 'gt': rec['gt'],
                 'group_key': rec.get('group_key'), 'dataset': 'haloquest',
                 'method': f'haloquest_prompt_{args.prompt_version}_dev',
                 'prompt_version': args.prompt_version,
                 'branch': rec['branch'], 'base_pred': rec['pred'],
                 'v2_pred': rec.get('v2_pred'), 'flagged': rec.get('flagged'),
                 'v1_evade': rec.get('v1_evade')}
        try:
            img = load_media_for_item(item, data_path, data_root)[2]
            user_prompt = get_user_prompt(item['question'], 'haloquest',
                                          args.prompt_version)
            clean = build_view(processor, model, system_prompt, user_prompt,
                               'haloquest', img)
            input_len = clean['input_ids'].shape[1]
            messages = make_messages(system_prompt,
                                     build_user_content_with_media_tokens(
                                         None, None, img, user_prompt))

            if rec['branch'] != 'passthrough_bon':
                # frozen branch: diagnostic greedy under the new prompt only
                with torch.inference_mode():
                    out = thinker.generate(**clean, max_new_tokens=args.max_new_tokens,
                                           do_sample=False,
                                           pad_token_id=tokenizer.eos_token_id)
                gid = out[0][input_len:]
                gid = gid[gid != tokenizer.pad_token_id]
                text = tokenizer.decode(gid, skip_special_tokens=True).strip()
                entry.update(pred=rec['pred'],      # frozen four-layer pred
                             contract_pred=parse_pred(text, 'haloquest'),
                             trajectories=None, selected_rollout=None)
            else:
                # BoN branch under the new contract prompt; contrast partner is
                # the SAME image as the published run (frozen).
                partner = by_id[rec['partner']]
                pimg = load_media_for_item(partner, data_path, data_root)[2]
                ctrl = build_view(processor, model, system_prompt, user_prompt,
                                  'haloquest', pimg)
                ev_mask, _ = evidence_mask(tokenizer, processor, clean,
                                           'haloquest', item, messages)
                cands, cand_ids, scores_list = [], [], []
                for s, (do_sample, temp) in enumerate(
                        [(False, 1.0)] + [(True, args.temperature)] * args.num_sampled):
                    gk = dict(max_new_tokens=args.max_new_tokens,
                              do_sample=do_sample,
                              pad_token_id=tokenizer.eos_token_id,
                              stop_strings=['<end_of_reasoning>'],
                              tokenizer=tokenizer)
                    if do_sample:
                        gk.update(temperature=temp)
                    with torch.inference_mode():
                        out = thinker.generate(**clean, **gk)
                    gen_ids = out[0][input_len:]
                    gen_ids = gen_ids[gen_ids != tokenizer.pad_token_id]
                    cands.append(tokenizer.decode(gen_ids, skip_special_tokens=True).strip())
                    cand_ids.append(gen_ids.tolist())
                    torch.cuda.empty_cache()
                for gl in cand_ids:
                    gt_t = torch.tensor(gl, dtype=torch.long, device=model.device)
                    delta = (view_logp(thinker, clean, gt_t)
                             - view_logp(thinker, ctrl, gt_t))
                    toks = [tokenizer.decode([t], skip_special_tokens=True) for t in gl]
                    cls = [token_class(t, toks[k - 1] if k else None)
                           for k, t in enumerate(toks)]
                    scores_list.append(rollout_scores(delta, claim_spans(toks, cls), cls))
                # feature-combo rerank (frozen weights + z-params)
                best, bv, cands_rec = None, -1e18, []
                for k, gl in enumerate(cand_ids):
                    gt_t = torch.tensor(gl, dtype=torch.long, device=model.device)
                    toks = [tokenizer.decode([t], skip_special_tokens=True) for t in gl]
                    cls = [token_class(t, toks[k - 1] if k else None)
                           for k, t in enumerate(toks)]
                    content_pos = torch.tensor([c in ('content', 'num') for c in cls],
                                               device=model.device)
                    f = forward_with_feats(thinker, clean, gt_t, ev_mask, content_pos)
                    gs = scores_list[k]
                    g = gs.get('g_traj_mean', 0.0)
                    ce = f['CE']['CE_top5mean']
                    v = f['VAS'][14]['vas_content']
                    if v is None:
                        v = f['VAS'][14]['vas_all']
                    z = lambda x, j: (x - zp[j][0]) / zp[j][1]
                    score = z(g, 0) - BETA * z(ce, 1) + GAMMA * z(v, 2)
                    cands_rec.append({'rollout': k, 'sampled': bool(k > 0),
                                      'pred': parse_pred(cands[k], 'haloquest'),
                                      'score': score, 'g_traj': g, 'CE': ce,
                                      'VAS': v, 'grounding_scores': gs})
                    if score > bv:
                        bv, best = score, k
                rand_sel = rng_rand.randrange(len(cands))
                entry.update(pred=cands_rec[best]['pred'],
                             trajectories=cands_rec,
                             selected_rollout=int(best),
                             random_sel_rollout=int(rand_sel),
                             partner=rec['partner'],
                             rank_score=f'feat_combo_beta{BETA}_gamma{GAMMA}_devz')
        except Exception as e:
            import traceback
            print(f'  [ERROR] {i}: {e}', flush=True)
            traceback.print_exc()
            torch.cuda.empty_cache()
            entry.update(pred=rec['pred'], branch=rec['branch'] + '_fallback',
                         error=str(e))

        with open(out_path, 'a') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        n += 1
        if n % 10 == 0:
            print(f'  [{n}/{len(run_ids)}] {time.time()-t0:.0f}s', flush=True)
        torch.cuda.empty_cache()

    # random-selection control arm (offline from the recorded trajectories)
    ctrl_path = out_path.replace('.jsonl', '_random_sel.jsonl')
    with open(out_path) as fin, open(ctrl_path, 'w') as fout:
        for line in fin:
            r = json.loads(line)
            if r.get('random_sel_rollout') is not None and r.get('trajectories'):
                r['pred'] = r['trajectories'][r['random_sel_rollout']]['pred']
            r['method'] += '_random_sel'
            fout.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'[done] {time.time()-t0:.0f}s -> {out_path}', flush=True)


if __name__ == '__main__':
    main()
