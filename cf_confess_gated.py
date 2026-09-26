#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""cf_confess_gated.py -- OP1: VAS+margin trigger gate in front of CF-Confess.

REFERENCE_PAPERS.md 3-OP1. The CF-Confess operator (cf_confess_phd.py /
cf_confess_cmm.py) pays a counterfactual view forward on EVERY sample
(registered cost: phd +97%, cmm +42% over base). This script puts a trigger
gate at the operator entrance:

  gate forward  = one eager-attention prefill of the FULL view (identical
                  input to base) that yields, in a single pass:
                    - the step-0 decision logits  -> margin = lp_yes - lp_no
                    - layer-14 attention (VAS_LAYERS calibration, 28-layer
                      thinker, L14 = middle; probe_bon_features.py) -> VAS =
                      attention mass of the decision position on evidence
                      tokens (phd: <|IMAGE|>; cmm: <|VIDEO|>+|<AUDIO|>)
  gate rule     = trigger CF iff
                    (step-0 argmax not in {Yes,No})          [CoT fallback]
                    OR (VAS < tau_vas AND |margin| < margin_band)
  not triggered -> answer directly from the gate forward's own argmax
                  (zero extra forwards; the gate forward IS the base forward)
  triggered     -> run the unchanged CF-Confess operator (full audit +
                  correction, identical code path to the ungated arm).

No new tuning: tau_vas / margin_band are selected (stage `thresh`) from the
dev-100 VAS/margin distribution with the single pre-registered criterion that
the gate must fire on every sample the ungated CF arm actually intervenes on
(intervened==1). Selection is fully recorded in the thresholds JSON.

Stages (resumable, one GPU each):
  calib   : per sample run base arm, ungated CF arm, gate forward; record
            per-arm timings, preds, cf internals, VAS, margin.
  thresh  : pick tau_vas / margin_band from the calib records (pure CPU).
  gated   : per sample gate forward; run CF only if triggered; record
            per-sample total time and final pred.
  report  : three-arm latency (mean/median/p99), trigger rate, per-sample
            decision identity gated-vs-CF and gated-vs-base (pure CPU).

Three arms share ONE model instance loaded with eager attention (VAS needs
attention weights), so timings are internally comparable across arms; absolute
values are therefore not directly comparable to the registered SDPA numbers
(ratios are the reported metric).

Usage:
    CUDA_VISIBLE_DEVICES=1 python cf_confess_gated.py --datasets phd --stage calib
    python cf_confess_gated.py --datasets phd --stage thresh
    CUDA_VISIBLE_DEVICES=1 python cf_confess_gated.py --datasets phd --stage gated
    python cf_confess_gated.py --datasets phd --stage report
"""
import os, sys, time, json, argparse, random

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'OmniConfess', 'methods'))

import numpy as np
import torch

from shared import (
    load_dataset, load_media_for_item, build_inputs,
    prepare_for_thinker, build_user_content_with_media_tokens, make_messages,
    get_system_prompt, get_user_prompt, parse_pred,
)
from cf_confess_phd import CFConfessProcessor as CFProcPhD, split_question
from cf_confess_cmm import CFConfessProcessor as CFProcCMM, build_view as build_view_cmm

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("MMHALU_MODEL_ROOT", "/ultrafast_share/tsq/MODELS")

YES_ID, NO_ID = 9454, 2753
VAS_LAYER = 14            # probe_bon_features.py VAS_LAYERS calibration, middle of 28
SAMPLE_SEED = 20260824    # same seed convention as make_1of9.py (date seed)
N_SAMPLE = 100
DEFAULT_LAM = {'phd': 2.0, 'cmm': 1.0}   # registered arms (RESULTS.md 29/31)

MEDIA_TOKENS = ('<|IMAGE|>', '<|VIDEO|>', '<|AUDIO|>')


def stratified_sample(data, n, seed):
    """Proportional largest-remainder allocation over group_key, seeded
    per-stratum draw. Deterministic; report = (seed, stratum composition)."""
    groups = {}
    for it in data:
        groups.setdefault(it['group_key'], []).append(it)
    total = len(data)
    quota = {k: int(round(n * len(v) / total)) for k, v in groups.items()}
    # largest remainder fix to hit exactly n
    order = sorted(groups, key=lambda k: -(n * len(groups[k]) / total - quota[k]))
    i = 0
    while sum(quota.values()) != n:
        quota[order[i % len(order)]] += 1 if sum(quota.values()) < n else -1
        i += 1
    picked = []
    for k, v in sorted(groups.items()):
        idx = list(range(len(v)))
        rng = random.Random(f'{seed}|{k}')
        picked += [v[j] for j in sorted(rng.sample(idx, quota[k]))]
    return picked, {k: (len(v), quota[k]) for k, v in sorted(groups.items())}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model_eager():
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
    model.thinker.config.vocab_size = 152064
    return processor, model


def build_full_view(processor, model, system_prompt, pv, av, iv, user_prompt, ds):
    uc = build_user_content_with_media_tokens(pv, av, iv, user_prompt)
    return prepare_for_thinker(
        build_inputs(processor, model, make_messages(system_prompt, uc),
                     pv, av, iv, ds), model)


@torch.inference_mode()
def gate_forward(thinker, clean, ev_token_ids):
    """Single eager prefill of the full view -> (argmax, margin, vas_l14, sec).

    This is the base forward: it produces the step-0 decision logits (the
    base answer for bare step-0 outputs) and, as a byproduct, the layer-14
    attention row needed for VAS. Zero extra forward when the gate does not
    fire."""
    t0 = time.time()
    out = thinker(**clean, use_cache=False, output_attentions=True)
    logits = out.logits[0, -1].float()
    lp = torch.log_softmax(logits, dim=-1)
    margin = (lp[YES_ID] - lp[NO_ID]).item()
    am = int(logits.argmax().item())
    att = out.attentions[VAS_LAYER][0].float().mean(0)      # heads averaged
    row = att[-1]                                           # decision position
    ids = clean['input_ids'][0]
    ev = torch.isin(ids, ev_token_ids)
    denom = row.sum().clamp(min=1e-9)
    vas = (row[ev].sum() / denom).item()
    del out, logits, att
    return am, margin, vas, time.time() - t0


def run_cf_arm(processor, model, tokenizer, system_prompt, pv, av, iv, item,
               ds, lam, prompt_version, clean_f=None):
    """Unchanged CF-Confess operator (imported processors). Returns
    (response, cf_record, elapsed). elapsed covers view build + generation,
    the same scope as the registered per-sample time. clean_f may be passed in
    to reuse an already-built full view (the gated arm has it from the gate)."""
    t0 = time.time()
    if ds == 'phd':
        q_full, q_only = split_question(item['question'])
        if clean_f is None:
            u_full = get_user_prompt(q_full, ds, prompt_version)
            clean_f = build_full_view(processor, model, system_prompt, pv, av,
                                      iv, u_full, ds)
        u_cf = get_user_prompt(q_only, ds, prompt_version)
        input_len = clean_f['input_ids'].shape[1]
        uc_cf = build_user_content_with_media_tokens(pv, av, iv, u_cf)
        clean_cf = prepare_for_thinker(
            build_inputs(processor, model, make_messages(system_prompt, uc_cf),
                         pv, av, iv, ds), model)
        proc = CFProcPhD(model, tokenizer, clean_cf, input_len, lam)
    else:
        u = get_user_prompt(item['question'], ds, prompt_version)
        if clean_f is None:
            clean_f = build_view_cmm(processor, model, system_prompt, pv, av, iv, u)
        input_len = clean_f['input_ids'].shape[1]
        clean_a = build_view_cmm(processor, model, system_prompt, pv, None, iv, u)
        clean_v = build_view_cmm(processor, model, system_prompt, None, av, iv, u)
        proc = CFProcCMM(model, tokenizer, clean_a, clean_v, input_len, lam)
    with torch.inference_mode():
        out = model.thinker.generate(
            **clean_f, max_new_tokens=300, do_sample=False,
            pad_token_id=tokenizer.eos_token_id, output_scores=True,
            return_dict_in_generate=True, logits_processor=[proc])
    gen = out.sequences[0][input_len:]
    resp = tokenizer.decode(gen, skip_special_tokens=True).strip()
    return resp, (proc.record or dict(dec_index=None, intervened=0)), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', required=True, choices=['phd', 'cmm'])
    ap.add_argument('--stage', required=True,
                    choices=['calib', 'thresh', 'gated', 'report'])
    ap.add_argument('--data_path', default=None)
    ap.add_argument('--data_root', default=os.path.join(HERE, 'OmniHalluBench'))
    ap.add_argument('--n_sample', type=int, default=N_SAMPLE)
    ap.add_argument('--sample_seed', type=int, default=SAMPLE_SEED)
    ap.add_argument('--lambda_cf', type=float, default=None)
    ap.add_argument('--prompt_version', default='v1')
    ap.add_argument('--out_dir', default=os.path.join(HERE, 'outputs', 'cf_gate'))
    args = ap.parse_args()
    ds = args.datasets
    lam = args.lambda_cf if args.lambda_cf is not None else DEFAULT_LAM[ds]
    os.makedirs(args.out_dir, exist_ok=True)
    data_path = args.data_path or os.path.join(
        HERE, 'OmniHalluBench', ds, ds + '.json')

    data = load_dataset(data_path, args.data_root)
    sample, strata = stratified_sample(data, args.n_sample, args.sample_seed)
    print(f'[sample] seed={args.sample_seed} n={len(sample)} strata={strata}',
          flush=True)

    tag = f'{ds}_n{args.n_sample}_seed{args.sample_seed}_lam{lam}'
    calib_path = os.path.join(args.out_dir, f'calib_{tag}.json')
    thresh_path = os.path.join(args.out_dir, f'thresh_{tag}.json')
    gated_path = os.path.join(args.out_dir, f'gated_{tag}.jsonl')

    if args.stage == 'calib':
        set_seed(42)
        processor, model = load_model_eager()
        tokenizer = processor.tokenizer
        thinker = model.thinker
        ev_ids = torch.tensor(
            [tokenizer.convert_tokens_to_ids(t) for t in MEDIA_TOKENS
             if tokenizer.convert_tokens_to_ids(t) is not None
             and tokenizer.convert_tokens_to_ids(t) >= 0],
            device=model.device)
        system_prompt = get_system_prompt(ds, args.prompt_version)
        rows = []
        for idx, item in enumerate(sample):
            t_m0 = time.time()
            pv, av, iv = load_media_for_item(item, data_path, args.data_root)
            t_media = time.time() - t_m0
            if ds == 'phd':
                q_full, _ = split_question(item['question'])
                u_full = get_user_prompt(q_full, ds, args.prompt_version)
            else:
                u_full = get_user_prompt(item['question'], ds, args.prompt_version)
            t0 = time.time()
            clean_f = build_full_view(processor, model, system_prompt,
                                      pv, av, iv, u_full, ds)
            t_view = time.time() - t0
            input_len = clean_f['input_ids'].shape[1]

            # arm 1: base (plain greedy, same view / generate settings)
            t0 = time.time()
            with torch.inference_mode():
                out = thinker.generate(
                    **clean_f, max_new_tokens=300, do_sample=False,
                    pad_token_id=tokenizer.eos_token_id)
            base_resp = tokenizer.decode(out[0][input_len:],
                                         skip_special_tokens=True).strip()
            t_base = t_view + (time.time() - t0)

            # arm 2: ungated CF-Confess (registered operator, unchanged)
            cf_resp, cf_rec, t_cf = run_cf_arm(
                processor, model, tokenizer, system_prompt, pv, av, iv, item,
                ds, lam, args.prompt_version)

            # gate forward (base forward + VAS byproduct)
            am, margin, vas, t_gate = gate_forward(thinker, clean_f, ev_ids)

            rows.append(dict(
                id=item['id'], gt=item['gt'], group=item['group_key'],
                base_pred=parse_pred(base_resp, ds), t_base=round(t_base, 4),
                t_base_view=round(t_view, 4), t_base_gen=round(t_base - t_view, 4),
                cf_pred=parse_pred(cf_resp, ds), t_cf=round(t_cf, 4),
                cf_intervened=cf_rec.get('intervened'),
                cf_margin_full=cf_rec.get('margin_full'),
                cf_dec_index=cf_rec.get('dec_index'),
                gate_am=am, gate_bare=int(am in (YES_ID, NO_ID)),
                gate_margin=round(margin, 4), gate_vas=round(vas, 6),
                t_gate=round(t_gate, 4), t_media=round(t_media, 4),
            ))
            print(f'[{idx+1}/{len(sample)}] id={item["id"]} interv={cf_rec.get("intervened")} '
                  f'base={t_base:.2f}s cf={t_cf:.2f}s gate={t_gate:.2f}s '
                  f'vas={vas:.4f} margin={margin:.2f} bare={am in (YES_ID, NO_ID)}',
                  flush=True)
            torch.cuda.empty_cache()
        json.dump(dict(dataset=ds, n=len(rows), sample_seed=args.sample_seed,
                       lambda_cf=lam, strata=strata, rows=rows),
                  open(calib_path, 'w'), ensure_ascii=False)
        print(f'[done] -> {calib_path}')

    elif args.stage == 'thresh':
        c = json.load(open(calib_path))
        rows = c['rows']
        bare = [r for r in rows if r['gate_bare']]
        inter = [r for r in rows if r['cf_intervened'] == 1]
        inter_bare = [r for r in inter if r['gate_bare']]
        vas_all = np.array([r['gate_vas'] for r in rows])
        # pre-registered criterion: gate must fire on every CF-intervened
        # bare sample -> thresholds = distribution quantile that just covers
        # max(VAS) and max(|margin|) over the intervened set (x1.05 slack).
        tau_vas = round(max(r['gate_vas'] for r in inter_bare) * 1.05 + 1e-4, 6) \
            if inter_bare else round(float(np.quantile(vas_all, 0.5)), 6)
        m_band = round(max(abs(r['gate_margin']) for r in inter_bare) * 1.05 + 1e-3, 4) \
            if inter_bare else 10.0
        # empirical quantile of the chosen thresholds in the dev distribution
        q_tau = float((vas_all < tau_vas).mean())
        def fires(r):
            return (not r['gate_bare']) or (r['gate_vas'] < tau_vas
                                            and abs(r['gate_margin']) < m_band)
        trig = [r for r in rows if fires(r)]
        missed = [r['id'] for r in inter if not fires(r)]
        out = dict(
            dataset=ds, n=len(rows), lambda_cf=lam,
            tau_vas=tau_vas, margin_band=m_band,
            selection=dict(
                criterion='gate fires on all CF-intervened samples (pre-registered)',
                n_intervened=len(inter), n_intervened_bare=len(inter_bare),
                max_vas_intervened=max((r['gate_vas'] for r in inter_bare), default=None),
                max_abs_margin_intervened=max((abs(r['gate_margin']) for r in inter_bare), default=None),
                rule='tau_vas = 1.05*max(VAS|intervened); band = 1.05*max(|margin||intervened); '
                     'no-threshold case (no bare intervened): median VAS / band=10',
                quantile_tau_in_dev=q_tau,
                vas_quantiles={q: round(float(np.quantile(vas_all, q)), 6)
                               for q in (0.25, 0.5, 0.75, 0.9)}),
            n_triggered=len(trig), trigger_rate=round(len(trig) / len(rows), 4),
            missed_intervened_ids=missed,
            strata=c['strata'], sample_seed=args.sample_seed,
        )
        json.dump(out, open(thresh_path, 'w'), ensure_ascii=False, indent=1)
        print(json.dumps(out, ensure_ascii=False, indent=1))
        assert not missed, 'threshold selection failed coverage: %s' % missed

    elif args.stage == 'gated':
        set_seed(42)
        th = json.load(open(thresh_path))
        tau, band = th['tau_vas'], th['margin_band']
        processor, model = load_model_eager()
        tokenizer = processor.tokenizer
        thinker = model.thinker
        ev_ids = torch.tensor(
            [tokenizer.convert_tokens_to_ids(t) for t in MEDIA_TOKENS
             if tokenizer.convert_tokens_to_ids(t) is not None
             and tokenizer.convert_tokens_to_ids(t) >= 0],
            device=model.device)
        system_prompt = get_system_prompt(ds, args.prompt_version)
        done = set()
        if os.path.exists(gated_path):
            for l in open(gated_path):
                try:
                    done.add(json.loads(l)['id'])
                except Exception:
                    pass
        with open(gated_path, 'a') as fout:
            for idx, item in enumerate(sample):
                if item['id'] in done:
                    continue
                pv, av, iv = load_media_for_item(item, data_path, args.data_root)
                if ds == 'phd':
                    q_full, _ = split_question(item['question'])
                    u_full = get_user_prompt(q_full, ds, args.prompt_version)
                else:
                    u_full = get_user_prompt(item['question'], ds, args.prompt_version)
                t0 = time.time()
                clean_f = build_full_view(processor, model, system_prompt,
                                          pv, av, iv, u_full, ds)
                t_view = time.time() - t0
                # arm 3: gated = gate forward + optional CF
                am, margin, vas, t_gate = gate_forward(thinker, clean_f, ev_ids)
                bare = am in (YES_ID, NO_ID)
                fired = (not bare) or (vas < tau and abs(margin) < band)
                if fired:
                    cf_resp, cf_rec, t_cf = run_cf_arm(
                        processor, model, tokenizer, system_prompt, pv, av, iv,
                        item, ds, lam, args.prompt_version, clean_f=clean_f)
                    resp = cf_resp
                    t_arm = t_view + t_gate + t_cf
                else:
                    resp = tokenizer.decode([am], skip_special_tokens=True).strip()
                    cf_rec = {}
                    t_arm = t_view + t_gate
                fout.write(json.dumps(dict(
                    id=item['id'], gt=item['gt'], group=item['group_key'],
                    gated_pred=parse_pred(resp, ds), fired=int(fired),
                    gate_vas=round(vas, 6), gate_margin=round(margin, 4),
                    gate_bare=int(bare), t_gated=round(t_arm, 4),
                    t_gate=round(t_gate, 4), t_view=round(t_view, 4),
                    cf_intervened=cf_rec.get('intervened'),
                ), ensure_ascii=False) + '\n')
                fout.flush()
                print(f'[{idx+1}/{len(sample)}] id={item["id"]} fired={int(fired)} '
                      f't={t_arm:.2f}s vas={vas:.4f} m={margin:.2f}', flush=True)
                torch.cuda.empty_cache()

    elif args.stage == 'report':
        c = json.load(open(calib_path))
        th = json.load(open(thresh_path))
        g = {json.loads(l)['id']: json.loads(l) for l in open(gated_path)}
        rows = c['rows']
        stats = {}
        for arm in ('t_base', 't_cf'):
            v = np.array([r[arm] for r in rows if r[arm] is not None])
            stats[arm] = dict(mean=round(float(v.mean()), 3),
                              median=round(float(np.median(v)), 3),
                              p99=round(float(np.quantile(v, 0.99)), 3))
        vg = np.array([g[r['id']]['t_gated'] for r in rows])
        stats['t_gated'] = dict(mean=round(float(vg.mean()), 3),
                                median=round(float(np.median(vg)), 3),
                                p99=round(float(np.quantile(vg, 0.99)), 3))
        n_fire = sum(g[r['id']]['fired'] for r in rows)
        mismatch_cf = [r['id'] for r in rows
                       if g[r['id']]['gated_pred'] != r['cf_pred']]
        mismatch_base = [r['id'] for r in rows
                         if g[r['id']]['gated_pred'] != r['base_pred']]
        rep = dict(
            dataset=ds, n=len(rows), lambda_cf=lam, sample_seed=args.sample_seed,
            strata=c['strata'], thresholds=th, latency=stats,
            overhead_vs_base=dict(
                cf=round(stats['t_cf']['mean'] / stats['t_base']['mean'] - 1, 4),
                gated=round(stats['t_gated']['mean'] / stats['t_base']['mean'] - 1, 4)),
            gate_trigger_rate=round(n_fire / len(rows), 4),
            decision_identity=dict(
                gated_vs_cf=round(1 - len(mismatch_cf) / len(rows), 4),
                gated_vs_base=round(1 - len(mismatch_base) / len(rows), 4),
                mismatch_cf_ids=mismatch_cf, mismatch_base_ids=mismatch_base),
        )
        outp = os.path.join(args.out_dir, f'report_{tag}.json')
        json.dump(rep, open(outp, 'w'), ensure_ascii=False, indent=1)
        print(json.dumps(rep, ensure_ascii=False, indent=1))
        print(f'[done] -> {outp}')


if __name__ == '__main__':
    main()
