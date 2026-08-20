#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""confess_pipeline.py -- one entry point for the six frozen cells.

Every experiment in this repository is finished; what was missing was a single
reproducible command per dataset. This file is that command. It routes by TASK
TYPE (not by dataset name) and reads every per-dataset knob from a declarative
``configs/<dataset>.yaml``:

    judgement        (phd / cmm / pubmedqa)  -> CF-Confess operator
                                                  (merge of cf_confess_phd.py,
                                                   cf_confess_cmm.py,
                                                   cf_confess_pubmedqa.py)
    descriptive-text (ragtruth)              -> output-contract layer
                                                  (base_model.py logic,
                                                   prompt_version v4)
    descriptive-vqa  (haloquest)             -> four-layer stack
                                                  (contract v2 x answerability
                                                   gate x BoN x feature-combo)
    perception-heavy (avhbench)              -> Vote-SC
                                                  (multi-arm generation +
                                                   vote rebuild, merge of
                                                   avhbench_vote_sc_rebuild.py)

DISCIPLINE -- algorithm code is merged, never rewritten.  Every numeric path
below is a verbatim transplant of the research-stage script named in the
section header, including dict key order (the JSONL rows are byte-compared
against the research outputs; only ``time_sec`` is volatile).  The research
scripts themselves stay in the repository untouched for provenance.

Offline stages that consume full-set artifacts (haloquest routing, vote
rebuild) never load the model and run without a GPU.

Usage:
    python confess_pipeline.py --dataset phd                        # full run
    python confess_pipeline.py --dataset phd --dry-run              # 3-item smoke
    python confess_pipeline.py --dataset cmm --prompt_version v1    # knob override
    python confess_pipeline.py --dataset haloquest --stage route    # offline only
    python confess_pipeline.py --dataset avhbench --stage vote
    python confess_pipeline.py --config configs/phd.yaml --limit 3

Stage selection (--stage, default per task type):
    judgement        cf
    descriptive-text contract
    descriptive-vqa  all | base | contract | route | bon | feat
                     (all = route -> bon -> feat over the artifacts named in the
                      config; base/contract explicitly regenerate layers 1-2;
                      route is offline, bon/feat need a GPU)
    perception-heavy all | arms | vote           (vote = offline)
"""
import os, sys, time, json, argparse, random

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'OmniConfess', 'methods'))
sys.path.insert(0, HERE)

from transformers import LogitsProcessor  # noqa: E402

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

YES_ID, NO_ID, MAYBE_ID = 9454, 2753, 21390
MARKER = ("In case there is an inconsistency between the context and the image "
          "content, you should follow the image.")

# Task type -> default stage. Routing is by task type only; which variant of an
# operator runs (CF views, BoN weights, vote arms, ...) comes from the config.
TASK_ROUTES = {
    'judgement': 'cf',
    'descriptive-text': 'contract',
    'descriptive-vqa': 'all',
    'perception-heavy': 'all',
}


def set_seed(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ===========================================================================
# Config
# ===========================================================================
def load_config(path):
    import yaml
    with open(path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    for key in ('task_type', 'dataset'):
        if key not in cfg:
            raise SystemExit(f'[config] {path}: missing required key "{key}"')
    return cfg


def apply_overrides(cfg, args):
    if args.prompt_version:
        cfg['prompt_version'] = args.prompt_version
    if args.lambda_cf is not None:
        cfg.setdefault('cf', {})['lambda_cf'] = args.lambda_cf
    if args.data_path:
        cfg['data_path'] = args.data_path
    if args.output_dir:
        cfg['output_dir'] = args.output_dir
    if args.file_name:
        cfg['file_name'] = args.file_name
    if args.time_path:
        cfg['time_path'] = args.time_path
    if args.seed is not None:
        cfg['seed'] = args.seed
    if args.control is not None:
        cfg.setdefault('cf', {})['control'] = args.control
    return cfg


def resolve(p):
    """Config paths are repo-root-relative; absolute paths pass through."""
    if p is None or os.path.isabs(p):
        return p
    return os.path.normpath(os.path.join(HERE, p))


# ===========================================================================
# Shared generation bookkeeping (transplanted from cf_confess_*.py / base_model.py)
# ===========================================================================
def token_logprobs_cf(gen_ids, scores):
    """Per-token logprob; NaN at the hard-forced decision step (cf operator)."""
    import torch
    out = []
    for tid, logits in zip(gen_ids.tolist(), scores):
        logits = logits.float()
        if logits.dim() > 1:
            logits = logits[0]
        if not torch.isfinite(logits[tid]):
            lp = float('nan')      # decision step under intervention
        else:
            lp = torch.log_softmax(logits, dim=-1)[tid]
        out.append(round(lp.item(), 4))
    return out


def token_logprobs_base(gen_ids, scores):
    """Per-token logprob, base_model.py convention (no forced token)."""
    import torch
    out = []
    for tid, logits in zip(gen_ids.tolist(), scores):
        logits = logits.float()
        if logits.dim() > 1:          # scores come back as [batch, vocab]
            logits = logits[0]
        lp = torch.log_softmax(logits, dim=-1)[tid]
        out.append(round(lp.item(), 4))
    return out


def load_done_ids(output_path):
    done_ids = set()
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line.strip())['id'])
                except Exception:
                    pass
        print(f'[resume] skipping {len(done_ids)} already-done samples', flush=True)
    return done_ids


def write_time_summary(time_path, file_name, t_start, n_out_tok, n_in_tok=None):
    os.makedirs(time_path, exist_ok=True)
    with open(os.path.join(time_path, file_name + '.txt'), 'w') as f:
        f.write(f'total_time: {time.time()-t_start:.2f}s\n')
        if n_in_tok is not None:
            f.write(f'total_input_tokens: {n_in_tok}\n')
        f.write(f'total_output_tokens: {n_out_tok}\n')


# ===========================================================================
# TASK TYPE 1: judgement -- the CF-Confess operator
# (merged from cf_confess_phd.py / cf_confess_cmm.py / cf_confess_pubmedqa.py;
#  variant selected by cfg['cf']['variant'], everything else identical)
# ===========================================================================
class CFConfessProcessor(LogitsProcessor):
    """Intervenes exactly once, at the decision token.

    phd      : d_cf     = log P(Yes | image+question view) - log P(Yes | full)
               margin'  = margin_full + lambda * d_cf ; force Yes if margin' > 0
    cmm      : d_cf_i   = log P(Yes | modality-i-masked view) - log P(Yes | full)
               delta    = w_audio*d_audio + w_video*d_video ; margin' as above
    pubmedqa : d_cf     = log P(Yes | no-passage view) - log P(Yes | full)
               yes'     = lp_yes_full + lambda * d_cf ; argmax over 3 classes

    Decision step := (no tokens generated yet and greedy argmax is a class
    token) or (phd/cmm: generated text ends with `\\boxed{`).
    """

    def __init__(self, model, tokenizer, cf_views, prompt_len, lam,
                 variant='phd', w_audio=1.0, w_video=1.0, control_rng=None):
        self.model = model
        self.tokenizer = tokenizer
        self.cf_views = cf_views      # {'cf': clean} | {'noaudio':..,'novideo':..}
        self.prompt_len = prompt_len
        self.lam = lam
        self.variant = variant
        self.w_audio = w_audio
        self.w_video = w_video
        self.control_rng = control_rng      # None => real Delta; else sign flip
        self.used = False
        self.record = None

    def _cf_forward(self, clean, prefix_ids):
        import torch
        ids = torch.cat([clean['input_ids'][0],
                         torch.tensor(prefix_ids, dtype=torch.long,
                                      device=clean['input_ids'].device)],
                        dim=0).unsqueeze(0)
        kw = dict(input_ids=ids, attention_mask=torch.ones_like(ids))
        for k in ('pixel_values_images', 'pixel_values', 'image_grid_thw',
                  'pixel_values_videos', 'video_grid_thw',
                  'video_second_per_grid', 'input_features',
                  'feature_attention_mask'):
            if k in clean:
                kw[k] = clean[k]
        logits = self.model.thinker(**kw).logits[0, -1].float()
        return torch.log_softmax(logits, dim=-1)

    def __call__(self, input_ids, scores):
        if self.used:
            return scores
        gen = input_ids[0, self.prompt_len:].tolist()
        logits = scores[0].float()
        am = int(logits.argmax().item())

        if self.variant == 'pubmedqa':
            if gen:                       # decision is only ever at step 0 here
                return scores
            at_decision = am in (YES_ID, NO_ID, MAYBE_ID)
        else:
            at_decision = False
            if len(gen) == 0:
                at_decision = am in (YES_ID, NO_ID)          # bare immediate answer
            else:
                tail = self.tokenizer.decode(gen[-4:])
                at_decision = tail.endswith('\\boxed{') or tail.endswith('\\boxed{{')

        if not at_decision:
            return scores

        lp = logits.log_softmax(dim=-1)

        if self.variant == 'pubmedqa':
            # --- cf_confess_pubmedqa.py: 3-class, Yes-logit shift only ------
            ly, ln, lm = lp[YES_ID].item(), lp[NO_ID].item(), lp[MAYBE_ID].item()
            lp_cf = self._cf_forward(self.cf_views['cf'], gen)
            ly_cf, ln_cf, lm_cf = (lp_cf[YES_ID].item(), lp_cf[NO_ID].item(),
                                   lp_cf[MAYBE_ID].item())
            d_cf = ly_cf - ly
            d_eff = d_cf
            if self.control_rng is not None:
                d_eff = d_cf * self.control_rng.choice([-1.0, 1.0])
            yes_adj = ly + self.lam * d_eff
            choices = {YES_ID: yes_adj, NO_ID: ln, MAYBE_ID: lm}
            forced = max(choices, key=choices.get)
            scores[0][:] = float('-inf')
            scores[0][forced] = 0.0
            self.used = True
            self.record = dict(
                dec_index=0,
                greedy_token=am,
                forced_token=forced,
                intervened=int(forced != am),
                lp_yes_full=round(ly, 4), lp_no_full=round(ln, 4),
                lp_maybe_full=round(lm, 4),
                lp_yes_cf=round(ly_cf, 4), lp_no_cf=round(ln_cf, 4),
                lp_maybe_cf=round(lm_cf, 4),
                d_cf=round(d_cf, 4), d_eff=round(d_eff, 4),
                yes_adj=round(yes_adj, 4),
            )
            return scores

        ly_full, ln_full = lp[YES_ID].item(), lp[NO_ID].item()
        margin_full = ly_full - ln_full

        if self.variant == 'cmm':
            # --- cf_confess_cmm.py: two masked-modality views ----------------
            d_audio = self._cf_forward(self.cf_views['noaudio'], gen)[YES_ID].item() - ly_full
            d_video = self._cf_forward(self.cf_views['novideo'], gen)[YES_ID].item() - ly_full
            delta = self.w_audio * d_audio + self.w_video * d_video
            delta_eff = delta
            if self.control_rng is not None:
                delta_eff = delta * self.control_rng.choice([-1.0, 1.0])
            margin_adj = margin_full + self.lam * delta_eff
            forced = YES_ID if margin_adj > 0 else NO_ID
            scores[0][:] = float('-inf')
            scores[0][forced] = 0.0
            self.used = True
            self.record = dict(
                dec_index=len(gen),
                greedy_token=am,
                forced_token=forced,
                intervened=int(forced != am),
                lp_yes_full=round(ly_full, 4), lp_no_full=round(ln_full, 4),
                margin_full=round(margin_full, 4),
                d_cf_audio=round(d_audio, 4), d_cf_video=round(d_video, 4),
                w_audio=self.w_audio, w_video=self.w_video,
                delta=round(delta, 4),
                delta_eff=round(delta_eff, 4),
                margin_adj=round(margin_adj, 4),
            )
            return scores

        # --- cf_confess_phd.py: context-removed view -------------------------
        lp_cf = self._cf_forward(self.cf_views['cf'], gen)
        ly_cf, ln_cf = lp_cf[YES_ID].item(), lp_cf[NO_ID].item()
        delta = ly_cf - ly_full
        delta_eff = delta
        if self.control_rng is not None:
            delta_eff = delta * self.control_rng.choice([-1.0, 1.0])
        margin_adj = margin_full + self.lam * delta_eff
        forced = YES_ID if margin_adj > 0 else NO_ID
        scores[0][:] = float('-inf')
        scores[0][forced] = 0.0          # hard-force the decision token
        self.used = True
        self.record = dict(
            dec_index=len(gen),
            greedy_token=am,
            forced_token=forced,
            intervened=int(forced != am),
            lp_yes_full=round(ly_full, 4), lp_no_full=round(ln_full, 4),
            margin_full=round(margin_full, 4),
            lp_yes_cf=round(ly_cf, 4), lp_no_cf=round(ln_cf, 4),
            delta_cf=round(delta, 4),
            delta_eff=round(delta_eff, 4),
            margin_adj=round(margin_adj, 4),
        )
        return scores


# NOTE: the three research scripts each define their own processor subclass;
# the merged class above dispatches on cfg['cf']['variant'] with the numeric
# path of each script kept line-for-line.


def split_question(q):
    """cf_confess_phd.py: full question vs question-only (marker stripped)."""
    idx = q.find(MARKER)
    if idx < 0:
        return q, q
    return q, q[idx + len(MARKER):].lstrip()     # (full, question-only)


def strip_passage_line(user_prompt):
    """cf_confess_pubmedqa.py: drop the leading 'Passage:' line."""
    lines = user_prompt.split('\n')
    if lines and lines[0].startswith('Passage:'):
        return '\n'.join(lines[1:])
    return user_prompt


def build_cf_views(cfg, processor, model, system_prompt, item, pv, av, iv):
    """Build the full view plus the counterfactual view(s), per variant."""
    from shared import (build_inputs, prepare_for_thinker,
                        build_user_content_with_media_tokens, make_messages,
                        get_user_prompt)
    ds = cfg['dataset']
    pver = cfg['prompt_version']
    variant = cfg['cf']['variant']

    def view(user_prompt, img, aud, vid):
        uc = build_user_content_with_media_tokens(img, aud, vid, user_prompt)
        return prepare_for_thinker(
            build_inputs(processor, model, make_messages(system_prompt, uc),
                         img, aud, vid, ds), model)

    if variant == 'phd':
        q_full, q_only = split_question(item['question'])
        u_full = get_user_prompt(q_full, ds, pver)
        clean_f = view(u_full, pv, av, iv)
        u_cf = get_user_prompt(q_only, ds, pver)
        clean_cf = view(u_cf, pv, av, iv)
        return clean_f, {'cf': clean_cf}
    if variant == 'cmm':
        u = get_user_prompt(item['question'], ds, pver)
        clean_f = view(u, pv, av, iv)
        clean_a = view(u, pv, None, iv)      # noaudio (video only)
        clean_v = view(u, None, av, iv)      # novideo (audio only)
        return clean_f, {'noaudio': clean_a, 'novideo': clean_v}
    if variant == 'pubmedqa':
        u_full = get_user_prompt(item['question'], ds, pver,
                                 passage=item.get('passage'))
        clean_f = view(u_full, None, None, None)
        u_cf = strip_passage_line(u_full)
        clean_c = view(u_cf, None, None, None)
        return clean_f, {'cf': clean_c}
    raise SystemExit(f'[cf] unknown variant {variant!r}')


def run_judgement(cfg, args):
    """Main loop of cf_confess_{phd,cmm,pubmedqa}.py, unified."""
    import numpy as np
    import torch
    from shared import (load_dataset, load_media_for_item, load_model,
                        get_system_prompt, make_result, save_result)

    ds = cfg['dataset']
    cf_cfg = cfg.get('cf', {})
    variant = cf_cfg.get('variant', ds)
    lam = float(cf_cfg.get('lambda_cf', 2.0))
    control = cf_cfg.get('control', 'none')
    seed = int(cfg.get('seed', 42))
    max_new_tokens = int(cfg.get('max_new_tokens', 300))
    record_process = bool(cfg.get('record_process', True))

    data_path = resolve(cfg['data_path'])
    data_root = resolve(cfg.get('data_root', os.path.join('OmniHalluBench', '')))
    output_dir = resolve(cfg.get('output_dir', f'./outputs/{ds}_cf_confess'))
    time_path = resolve(cfg.get('time_path', os.path.join(output_dir, 'time')))
    file_name = cfg.get('file_name', f'cf_confess_{ds}')

    set_seed(seed)
    control_rng = random.Random(seed + 7) if control == 'random_sign' else None
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(time_path, exist_ok=True)
    output_path = os.path.join(output_dir, file_name + '.jsonl')
    done_ids = load_done_ids(output_path)

    print('[model] loading Qwen2.5-Omni...', flush=True)
    processor, model = load_model()
    tokenizer = processor.tokenizer

    data = load_dataset(data_path, data_root)
    if args.limit:
        data = data[:args.limit]
        print(f'[limit] processing {len(data)} samples', flush=True)

    system_prompt = get_system_prompt(ds, cfg['prompt_version'])

    all_out_tok = 0
    t_start = time.time()

    for idx, item in enumerate(data):
        if item['id'] in done_ids:
            continue
        t0 = time.time()
        print(f'\n[{idx+1}/{len(data)}] id={item["id"]}', flush=True)

        rec_extra, gen_ids, logp_list, input_len = {}, None, None, None
        clean_f, cf_views = None, None
        try:
            pv, av, iv = load_media_for_item(item, data_path, data_root)
            clean_f, cf_views = build_cf_views(cfg, processor, model,
                                               system_prompt, item, pv, av, iv)
            input_len = clean_f['input_ids'].shape[1]

            cf_proc = CFConfessProcessor(
                model, tokenizer, cf_views, input_len, lam,
                variant=variant,
                w_audio=float(cf_cfg.get('w_audio', 1.0)),
                w_video=float(cf_cfg.get('w_video', 1.0)),
                control_rng=control_rng)

            with torch.inference_mode():
                out = model.thinker.generate(
                    **clean_f,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    output_scores=True,
                    return_dict_in_generate=True,
                    logits_processor=[cf_proc],
                )
            gen_ids = out.sequences[0][input_len:].cpu()
            if record_process:
                logp_list = token_logprobs_cf(gen_ids, out.scores)
            response = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            rec_extra = cf_proc.record or dict(dec_index=None, intervened=0)
            rec_extra.update(lambda_cf=lam, control=control)
            all_out_tok += len(gen_ids)
            if variant == 'pubmedqa':
                print(f'  response[:60]: {response[:60]!r} | '
                      f'dec={rec_extra.get("dec_index")} '
                      f'interv={rec_extra.get("intervened")} | '
                      f'{time.time()-t0:.1f}s', flush=True)
            else:
                print(f'  response[-60]: {response[-60:]!r} | '
                      f'dec={rec_extra.get("dec_index")} '
                      f'interv={rec_extra.get("intervened")} | '
                      f'{time.time()-t0:.1f}s', flush=True)
            del clean_f
            cf_views.clear()
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f'  [ERROR] {e}', flush=True)
            response = ''
            rec_extra = dict(dec_index=None, intervened=0, error=str(e)[:200],
                             lambda_cf=lam, control=control)

        torch.cuda.empty_cache()

        result = make_result(item, response, ds,
                             f'cf_confess_lam{lam}'
                             + ('_randctl' if control != 'none' else ''),
                             time_sec=time.time() - t0)
        if record_process:
            result['record_process'] = True
            result['cf'] = rec_extra
            if gen_ids is not None:
                cum = float(np.nansum(logp_list))
                result['token_ids'] = gen_ids.tolist()
                result['logp_list'] = logp_list
                result['cumulative_logprob'] = round(cum, 4)
                result['norm_logp'] = round(cum / (len(gen_ids) + 1e-8), 4)
                result['input_len'] = int(input_len)
                result['output_tokens'] = int(len(gen_ids))
        save_result(result, output_path)

    write_time_summary(time_path, file_name, t_start, all_out_tok)
    print(f'\n[done] output={output_path}', flush=True)


# ===========================================================================
# TASK TYPE 2: descriptive-text -- the output-contract layer
# (base_model.py logic; the "contract" is entirely the prompt version)
# ===========================================================================
def run_contract(cfg, args):
    import numpy as np
    import torch
    from shared import (load_dataset, load_media_for_item, load_model,
                        build_inputs, prepare_for_thinker,
                        build_user_content_with_media_tokens, make_messages,
                        get_system_prompt, get_user_prompt,
                        make_result, save_result)

    ds = cfg['dataset']
    seed = int(cfg.get('seed', 42))
    max_new_tokens = int(cfg.get('max_new_tokens', 512))
    record_process = bool(cfg.get('record_process', True))

    data_path = resolve(cfg['data_path'])
    data_root = resolve(cfg.get('data_root', os.path.join('OmniHalluBench', '')))
    output_dir = resolve(cfg.get('output_dir', './outputs'))
    time_path = resolve(cfg.get('time_path', os.path.join(output_dir, 'time')))
    file_name = cfg.get('file_name', f'{ds}_contract')

    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(time_path, exist_ok=True)
    output_path = os.path.join(output_dir, file_name + '.jsonl')
    done_ids = load_done_ids(output_path)

    print('[model] loading Qwen2.5-Omni...', flush=True)
    processor, model = load_model()
    tokenizer = processor.tokenizer

    data = load_dataset(data_path, data_root)
    if args.limit:
        data = data[:args.limit]
        print(f'[limit] processing {len(data)} samples', flush=True)

    system_prompt = get_system_prompt(ds, cfg['prompt_version'])

    all_input_tokens = 0
    all_output_tokens = 0
    start = time.time()

    for idx, item in enumerate(data):
        if item['id'] in done_ids:
            continue
        t0 = time.time()
        print(f'\n[{idx+1}/{len(data)}] id={item["id"]}', flush=True)

        pixel_values, audio_values, image_values = load_media_for_item(
            item, data_path, data_root)

        user_prompt = get_user_prompt(
            item['question'], ds, cfg['prompt_version'],
            passage=item.get('passage'))
        user_content = build_user_content_with_media_tokens(
            pixel_values, audio_values, image_values, user_prompt)
        messages = make_messages(system_prompt, user_content)

        try:
            raw_inputs = build_inputs(
                processor, model, messages,
                pixel_values, audio_values, image_values, ds)
            clean = prepare_for_thinker(raw_inputs, model)
            input_len = clean['input_ids'].shape[1]
            all_input_tokens += input_len

            gen_kwargs = dict(
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            if record_process:
                gen_kwargs.update(output_scores=True,
                                  return_dict_in_generate=True)
            with torch.inference_mode():
                out = model.thinker.generate(**clean, **gen_kwargs)

            if record_process:
                gen_ids = out.sequences[0][input_len:]
                logp_list = token_logprobs_base(gen_ids.cpu(), out.scores)
            else:
                gen_ids = out[0][input_len:]
            response = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            all_output_tokens += len(gen_ids)

            elapsed = time.time() - t0
            print(f'  response[:120]: {response[:120]!r}', flush=True)
            print(f'  tokens_out={len(gen_ids)} | time={elapsed:.1f}s', flush=True)

        except Exception as e:
            print(f'  [ERROR] {e}', flush=True)
            response = ''
            elapsed = time.time() - t0
            gen_ids = None

        torch.cuda.empty_cache()

        result = make_result(item, response, ds, 'base_model',
                             time_sec=elapsed)

        if record_process:
            result['record_process'] = True
            if gen_ids is not None:
                cum_lp = float(np.sum(logp_list))
                result['token_ids'] = gen_ids.tolist()
                result['logp_list'] = logp_list
                result['cumulative_logprob'] = round(cum_lp, 4)
                result['norm_logp'] = round(cum_lp / (len(gen_ids) + 1e-8), 4)
                result['input_len'] = int(input_len)
                result['output_tokens'] = int(len(gen_ids))
            else:
                result['token_ids'] = None
                result['logp_list'] = None
                result['cumulative_logprob'] = None
                result['norm_logp'] = None
                result['input_len'] = None
                result['output_tokens'] = 0

        save_result(result, output_path)

    write_time_summary(time_path, file_name, start, all_output_tokens,
                       all_input_tokens)
    print(f'\n[done] total={time.time()-start:.1f}s | output={output_path}',
          flush=True)


# ===========================================================================
# TASK TYPE 3: descriptive-vqa -- the haloquest four-layer stack
#   layer 1  base v1 greedy          (base_model.py, --stage contract)
#   layer 2  contract v2 greedy      (base_model.py, --stage contract)
#   layer 3  answerability gate      (answerability_probe.py + answerability_gate.py,
#                                     research scripts; flags consumed as an artifact)
#   layer 4  route + BoN + feat-combo (below; merged from
#                                     answerability_contract_gate.py /
#                                     bon_stacked_haloquest.py /
#                                     bon_stacked_feat_combo.py)
# ===========================================================================
def _sibling(path, tag):
    """Derive a control-arm output path, research-script convention.

    bon_stacked_haloquest.py writes <name>_full.jsonl -> <name>_random_sel.jsonl
    and bon_stacked_feat_combo.py <name>_feat_combo.jsonl -> <name>_random_sel;
    a plain .replace() silently no-ops (and then self-truncates) when the main
    output is named without the marker, so guard explicitly.
    """
    for marker in ('_full.jsonl', '_feat_combo.jsonl'):
        if path.endswith(marker):
            return path[:-len(marker)] + tag + '.jsonl'
    out = path[:-len('.jsonl')] + tag + '.jsonl' if path.endswith('.jsonl') \
        else path + tag + '.jsonl'
    if os.path.abspath(out) == os.path.abspath(path):
        raise SystemExit(f'[pipeline] control-arm path collides with main output '
                         f'{path!r}; name the main file ..._full.jsonl')
    return out


def _halo_route_policy(v1, v2, gate, meta, T3, name):
    def pred_A0(i):
        return gate[i]['pred']

    def pred_A1(i):
        m = meta[i]
        if not m['flagged'] and not m['v2_avoid']:
            return v2[i]['pred']
        return gate[i]['pred']

    def pred_A2(i):
        m = meta[i]
        if not m['flagged'] and not m['v1_evade']:
            return v2[i]['pred']
        return gate[i]['pred']

    def pred_A3(i):
        m = meta[i]
        if m['flagged'] and m['v2_avoid']:
            return v2[i]['pred']
        return pred_A2(i)

    def pred_A4(i):
        m = meta[i]
        if m['flagged'] and not m['v2_avoid']:
            return v2[i]['pred']
        return pred_A2(i)

    def pred_C1(i):
        m = meta[i]
        if m['v1_evade']:
            return v1[i]['pred']
        if m['flagged']:
            return T3(v1[i]['question'])
        return v2[i]['pred']

    fam = {'A0': pred_A0, 'A1': pred_A1, 'A2': pred_A2, 'A3': pred_A3,
           'A4': pred_A4, 'C1': pred_C1}
    if name not in fam:
        raise SystemExit(f'[haloquest] unknown routing policy {name!r} '
                         f'(A0|A1|A2|A3|A4|C1)')
    return fam[name]


def halo_stage_route(cfg, args):
    """answerability_contract_gate.py, frozen to the dev-selected policy.

    The source script selected the policy on the fixed 100-sample dev split;
    the dev grid read {A0 37.523, A1 38.306, A2 40.164, A3 39.402, A4 40.078,
    C1 40.164} and max() returns A2 (first maximum; A2 and C1 tie). The config
    pins that outcome so reproduction needs neither the dev split nor the
    probe rows. Byte-compared against
    outputs/answerability_contract/haloquest_answerability_contract_gate_full.jsonl.
    """
    import re
    from answerability_gate import TEMPLATES
    from probe_claim_cf import EVADERE

    h = cfg['haloquest']
    V1_PATH, V2_PATH = resolve(h['base_v1']), resolve(h['contract_v2'])
    GATE_PATH = resolve(h['gate'])
    combined_path = resolve(h['routed'])
    os.makedirs(os.path.dirname(combined_path), exist_ok=True)

    V2_AVOID = re.compile(r'there\s+is\s+no\b|not\s+visible|cannot|can\s*not', re.I)

    v1 = {r['id']: r for r in map(json.loads, open(V1_PATH))}
    v2 = {r['id']: r for r in map(json.loads, open(V2_PATH))}
    gate = {r['id']: r for r in map(json.loads, open(GATE_PATH))}

    ids = [r['id'] for r in v1.values()]
    meta = {}
    for i in ids:
        meta[i] = {
            'v1_evade': bool(EVADERE.search(v1[i]['pred'] or '')),
            'v2_avoid': bool(V2_AVOID.search(v2[i]['pred'] or '')),
            'flagged': bool(gate[i]['flagged']),
        }
    gt = {i: v1[i]['gt'] for i in ids}
    T3 = TEMPLATES['T3']

    policy = h.get('policy', 'A2')
    best_fn = _halo_route_policy(v1, v2, gate, meta, T3, policy)
    print(f'[route] policy {policy} over {len(ids)} items', flush=True)

    # random-flag control, same structure/seed as the source (seed 49)
    n_flag = sum(1 for i in ids if meta[i]['flagged'])
    probe_path = resolve(h.get('probe', ''))
    row_by_id = {}
    if probe_path and os.path.exists(probe_path):
        row_by_id = {r['id']: r for r in json.load(open(probe_path))['rows']
                     if r.get('ok')}
    rng = random.Random(49)
    eligible = [i for i in ids if not meta[i]['v1_evade'] and i in row_by_id]
    rand_flag = set(rng.sample(eligible, min(n_flag, len(eligible))))

    def rand_C1(i):
        m = meta[i]
        if m['v1_evade']:
            return v1[i]['pred']
        if i in rand_flag:
            return T3(v1[i]['question'])
        return v2[i]['pred']

    with open(combined_path, 'w') as f:
        for r in v1.values():
            i = r['id']
            f.write(json.dumps({'id': i, 'question': r['question'], 'gt': gt[i],
                                'group_key': r.get('group_key'), 'dataset': 'haloquest',
                                'method': 'answerability_contract_gate',
                                'base_pred': r['pred'], 'v2_pred': v2[i]['pred'],
                                'flagged': meta[i]['flagged'],
                                'pred': best_fn(i)}, ensure_ascii=False) + '\n')
    rand_path = _sibling(combined_path, '_randctrl')
    if rand_flag:
        with open(rand_path, 'w') as f:
            for r in v1.values():
                i = r['id']
                f.write(json.dumps({'id': i, 'question': r['question'], 'gt': gt[i],
                                    'group_key': r.get('group_key'), 'dataset': 'haloquest',
                                    'method': 'answerability_contract_gate_randctrl',
                                    'pred': rand_C1(i)}, ensure_ascii=False) + '\n')
        print(f'[route] -> {combined_path}\n        -> {rand_path}', flush=True)
    else:
        print(f'[route] -> {combined_path}\n'
              f'        (random-flag control skipped: probe artifact not found at '
              f'{probe_path})', flush=True)


def halo_stage_bon(cfg, args):
    """bon_stacked_haloquest.py main loop (BoN grafted on the pass-through branch)."""
    import torch
    from shared import (load_dataset, load_media_for_item, resolve_media_path,
                        load_model, get_system_prompt, get_user_prompt, parse_pred)
    from answerability_gate import EVADERE, TEMPLATES
    from bon_grounding import (token_class, claim_spans, view_logp, build_view,
                               rollout_scores, rank_value, set_seed)

    h = cfg['haloquest']
    bon = h['bon']
    COMBINED_PATH = resolve(h['routed'])
    out_dir = resolve(bon.get('out_dir', os.path.dirname(resolve(bon['out']))))
    out_main = resolve(bon['out'])
    seed = int(bon.get('seed', cfg.get('seed', 42)))
    num_sampled = int(bon.get('num_sampled', 3))
    temperature = float(bon.get('temperature', 0.7))
    max_new_tokens = int(bon.get('max_new_tokens', 512))
    rank_score = bon.get('rank_score', 'g_content')
    if rank_score not in ('g_content', 'g_mean_claim', 'g_worst_claim', 'g_traj_mean'):
        raise SystemExit(f'[haloquest] bad rank_score {rank_score!r}')

    set_seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    combined = [json.loads(l) for l in open(COMBINED_PATH)]
    done_ids = set()
    if os.path.exists(out_main):
        for line in open(out_main):
            try:
                done_ids.add(json.loads(line)['id'])
            except Exception:
                pass
        print(f'[resume] skipping {len(done_ids)}', flush=True)

    data_path = resolve(cfg['data_path'])
    data_root = resolve(cfg.get('data_root', os.path.join('OmniHalluBench', '')))
    data = load_dataset(data_path, data_root)
    by_id = {it['id']: it for it in data}
    pool_images = [it for it in data if it.get('image_path') and os.path.exists(
        resolve_media_path(it['image_path'], os.path.dirname(data_path), data_root))]

    processor, model = load_model()
    tokenizer = processor.tokenizer
    thinker = model.thinker
    system_prompt = get_system_prompt('haloquest', cfg['prompt_version'])

    rng = random.Random(seed + 7)
    rng_rand = random.Random(seed + 13)
    T3 = TEMPLATES['T3']
    t0 = time.time()
    n_done = 0
    limit = args.limit or 0
    for rec in combined:
        i = rec['id']
        if i in done_ids:
            continue
        if limit and n_done >= limit:
            break
        item = by_id[i]
        v1_evade = bool(EVADERE.search(rec['base_pred'] or ''))
        flagged = bool(rec['flagged'])

        entry = {'id': i, 'question': rec['question'], 'gt': rec['gt'],
                 'group_key': rec.get('group_key'), 'dataset': 'haloquest',
                 'method': 'bon_stacked', 'base_pred': rec['base_pred'],
                 'v2_pred': rec['v2_pred'], 'flagged': flagged,
                 'v1_evade': v1_evade, 'branch': None, 'rank_score': rank_score}

        if v1_evade:
            entry.update(branch='v1evade_keep', pred=rec['base_pred'],
                         trajectories=None, selected_rollout=None, random_sel_rollout=None)
        elif flagged:
            entry.update(branch='flagged_T3', pred=T3(rec['question']),
                         trajectories=None, selected_rollout=None, random_sel_rollout=None)
        else:
            entry['branch'] = 'passthrough_bon'
            cands, cand_ids, scores_list = [], [], []
            try:
                img = load_media_for_item(item, data_path, data_root)[2]
                user_prompt = get_user_prompt(item['question'], 'haloquest', cfg['prompt_version'])
                clean = build_view(processor, model, system_prompt, user_prompt,
                                   'haloquest', img)
                input_len = clean['input_ids'].shape[1]
                partner = None
                while partner is None:
                    cand_it = rng.choice(pool_images)
                    if cand_it['id'] != item['id']:
                        partner = cand_it
                pimg = load_media_for_item(partner, data_path, data_root)[2]
                ctrl = build_view(processor, model, system_prompt, user_prompt,
                                  'haloquest', pimg)
                for s, (do_sample, temp) in enumerate(
                        [(False, 1.0)] + [(True, temperature)] * num_sampled):
                    gk = dict(max_new_tokens=max_new_tokens,
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
                    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                    cands.append(text)
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
                sel = max(range(len(cands)),
                          key=lambda k: rank_value(scores_list[k], rank_score))
                rand_sel = rng_rand.randrange(len(cands))
                entry.update(pred=parse_pred(cands[sel], 'haloquest'),
                             trajectories=[{'rollout': k, 'sampled': bool(k > 0),
                                            'token_ids': cand_ids[k],
                                            'pred': parse_pred(cands[k], 'haloquest'),
                                            'grounding_scores': scores_list[k]}
                                           for k in range(len(cands))],
                             selected_rollout=int(sel), random_sel_rollout=int(rand_sel),
                             partner=partner['id'])
            except Exception as e:
                print(f'  [ERROR] {i}: {e}', flush=True)
                torch.cuda.empty_cache()
                entry.update(pred=rec['v2_pred'], trajectories=None,
                             selected_rollout=None, random_sel_rollout=None,
                             branch='passthrough_bon_fallback_v2', error=str(e))

        with open(out_main, 'a') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        n_done += 1
        if n_done % 20 == 0:
            print(f'  [{n_done}] {time.time()-t0:.0f}s', flush=True)
        torch.cuda.empty_cache()

    # ---- random-selection control arm (differs only on the BoN branch) -------
    ctrl_path = _sibling(out_main, '_random_sel')
    with open(out_main) as fin, open(ctrl_path, 'w') as fout:
        for line in fin:
            r = json.loads(line)
            if r.get('random_sel_rollout') is not None and r.get('trajectories'):
                rs = r['random_sel_rollout']
                r['pred'] = r['trajectories'][rs]['pred']
            r['method'] = 'bon_stacked_random_sel'
            fout.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'[done] {time.time()-t0:.0f}s -> {out_main} / {ctrl_path}', flush=True)


def halo_stage_feat(cfg, args):
    """bon_stacked_feat_combo.py (feature-combo ranker on the pass-through branch).

    score = z(Delta) - beta*z(CE) + gamma*z(VAS); weights AND z-params both
    fixed on the haloquest v1-pool dev split -- nothing is re-tuned here.
    """
    import numpy as np
    import torch
    from shared import (load_dataset, load_media_for_item, get_system_prompt,
                        get_user_prompt, build_user_content_with_media_tokens,
                        make_messages)
    from probe_bon_features import forward_with_feats, build_view, evidence_mask
    from bon_grounding import token_class

    h = cfg['haloquest']
    feat = h['feat']
    STACKED = resolve(h['bon']['out'])
    V1_FEAT = resolve(feat['v1_feat'])
    V1_POOL = resolve(feat['v1_pool'])
    BETA = float(feat.get('beta', 0.25))
    GAMMA = float(feat.get('gamma', 2.0))
    out_path = resolve(feat['out'])

    def dev_zpars():
        """Re-derive the dev(100) z-params of the calibrated scorer (v1 pool)."""
        rows = json.load(open(V1_FEAT))['rows']
        pool = {r['id']: {t['rollout']: t for t in r['trajectories']}
                for r in map(json.loads, open(V1_POOL))}

        def feats(c):
            v = c['VAS']['14']['vas_content']
            if v is None:
                v = c['VAS']['14']['vas_all']
            gs = pool[c['_id']][c['rollout']].get('grounding_scores')
            g = gs['g_traj_mean'] if gs else 0.0
            return (g, c['CE']['CE_top5mean'], v)

        for r in rows:
            for c in r['candidates']:
                c['_id'] = r['id']
        rng = random.Random(42)
        idx = list(range(len(rows)))
        rng.shuffle(idx)
        dev = [rows[i] for i in idx[:100]]
        zp = []
        for fi in range(3):
            vals = [feats(c)[fi] for r in dev for c in r['candidates']]
            zp.append((float(np.mean(vals)), float(np.std(vals)) + 1e-9))
        return zp

    zp = dev_zpars()
    print(f'[z-params] delta {zp[0][0]:.3f}/{zp[0][1]:.3f} '
          f'CE {zp[1][0]:.3f}/{zp[1][1]:.3f} VAS {zp[2][0]:.4f}/{zp[2][1]:.4f}', flush=True)

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
    tokenizer = processor.tokenizer
    system_prompt = get_system_prompt('haloquest', cfg['prompt_version'])

    data_path = resolve(cfg['data_path'])
    data_root = resolve(cfg.get('data_root', os.path.join('OmniHalluBench', '')))
    data = load_dataset(data_path, data_root)
    by_id = {it['id']: it for it in data}

    recs = [json.loads(l) for l in open(STACKED)]
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            try:
                done.add(json.loads(line)['id'])
            except Exception:
                pass
    outf = open(out_path, 'a')

    t0 = time.time()
    n = 0
    limit = args.limit or 0
    for rec in recs:
        if rec['id'] in done:
            continue
        item = by_id[rec['id']]
        if rec['branch'] != 'passthrough_bon':
            outf.write(json.dumps(rec, ensure_ascii=False) + '\n')
            outf.flush()
            continue
        if limit and n >= limit:
            break
        # feature extraction for the candidates under the generation prompt
        img = load_media_for_item(item, data_path, data_root)[2]
        user_prompt = get_user_prompt(item['question'], 'haloquest', cfg['prompt_version'])
        messages = make_messages(system_prompt,
                                 build_user_content_with_media_tokens(None, None, img, user_prompt))
        clean = build_view(processor, model, system_prompt, user_prompt, 'haloquest', img)
        ev_mask, _ = evidence_mask(tokenizer, processor, clean, 'haloquest', item, messages)
        best, bv = None, -1e18
        cands = []
        for tr in rec['trajectories']:
            gen_ids = torch.tensor(tr['token_ids'], dtype=torch.long, device=model.device)
            toks = [tokenizer.decode([t], skip_special_tokens=True) for t in tr['token_ids']]
            cls = [token_class(t, toks[k - 1] if k else None) for k, t in enumerate(toks)]
            content_pos = torch.tensor([c in ('content', 'num') for c in cls],
                                       device=model.device)
            f = forward_with_feats(model.thinker, clean, gen_ids, ev_mask, content_pos)
            gs = tr.get('grounding_scores') or {}
            g = gs.get('g_traj_mean', 0.0)
            ce = f['CE']['CE_top5mean']
            v = f['VAS'][14]['vas_content']   # in-memory dict: int layer keys
            if v is None:
                v = f['VAS'][14]['vas_all']
            z = lambda x, i: (x - zp[i][0]) / zp[i][1]
            score = z(g, 0) - BETA * z(ce, 1) + GAMMA * z(v, 2)
            cands.append({'rollout': tr['rollout'], 'score': score,
                          'g_traj': g, 'CE': ce, 'VAS': v})
            if score > bv:
                bv, best = score, tr
        rec = dict(rec)
        rec['method'] = 'bon_stacked_feat_combo'
        rec['pred'] = best['pred']
        rec['raw_output'] = best['pred']
        rec['rank_score'] = f'feat_combo_beta{BETA}_gamma{GAMMA}_devz'
        rec['selected_rollout_feat'] = best['rollout']
        rec['feat_combo_candidates'] = cands
        outf.write(json.dumps(rec, ensure_ascii=False) + '\n')
        outf.flush()
        n += 1
        if n % 20 == 0:
            print(f'  [{n}] {time.time()-t0:.0f}s', flush=True)
        torch.cuda.empty_cache()
    outf.close()

    # random-selection control arm (same rollouts, same recorded random picks)
    ctrl = _sibling(out_path, '_random_sel')
    with open(out_path) as fin, open(ctrl, 'w') as fout:
        for line in fin:
            r = json.loads(line)
            if r.get('random_sel_rollout') is not None and r.get('trajectories'):
                r['pred'] = r['trajectories'][r['random_sel_rollout']]['pred']
            r['method'] = 'bon_stacked_random_sel'
            fout.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'[done] -> {out_path}', flush=True)


def run_haloquest(cfg, args):
    stage = args.stage or 'all'
    if stage in ('base', 'contract'):
        # layers 1/2: plain contract-layer generation (base_model.py logic) at
        # prompt v1 / v2 -- EXPLICIT stages only; 'all' consumes the artifacts
        # named in the config. Written under outputs/pipeline/...; when
        # regenerating from scratch, point cfg['haloquest']['base_v1' /
        # 'contract_v2'] at the files produced here before running route.
        out_dir = resolve(cfg.get('output_dir', './outputs/pipeline/haloquest'))
        for st, pver, fname in (('base', 'v1', f'{cfg["dataset"]}_base_model'),
                                ('contract', cfg['prompt_version'],
                                 f'{cfg["dataset"]}_contract_{cfg["prompt_version"]}')):
            if stage != st:
                continue
            c = dict(cfg)
            c['prompt_version'] = pver
            c['output_dir'] = out_dir
            c['file_name'] = fname
            c['time_path'] = os.path.join(out_dir, 'time')
            run_contract(c, args)
    if stage in ('all', 'route'):
        halo_stage_route(cfg, args)
    if stage in ('all', 'bon'):
        halo_stage_bon(cfg, args)
    if stage in ('all', 'feat'):
        halo_stage_feat(cfg, args)


# ===========================================================================
# TASK TYPE 4: perception-heavy -- Vote-SC for avhbench
# (merge of avhbench_vote_sc_rebuild.py; voting logic carried verbatim from the
#  run that produced avhbench_vote_sc_full.jsonl = 21.56)
# ===========================================================================
VOTE_STOP = set('''a an the and or of to in on at as is are was were be been being it its this that these those with while by for from into near nearby
there here his her their our your my he she they we you i who which what when where how not no nor but so then than also very just only
one two single describe see hear seen heard video audio image picture clip scene shows show can cannot could would should may might'''.split())
VOTE_PUNCT = "!\"#$%&'()*+,-./:;<=>?@[]^_`{|}~\\"


def _normgt(s):
    import re
    return re.sub(r'\s+', ' ', str(s).strip().lower())


def _clean_boxed(s):
    import re
    m = re.search(r'\\boxed\{(.*)\}', str(s), re.S)
    return (m.group(1) if m else str(s)).strip()


def _cwords(s):
    s = _clean_boxed(s).lower().translate(str.maketrans('', '', VOTE_PUNCT))
    return [w for w in s.split() if w not in VOTE_STOP and len(w) > 1]


def _backfill(skeleton_sent, voted):
    """Phrasing backfill: skeleton arm's function words + voted content words."""
    import re
    out, seen = [], set()
    for tok in skeleton_sent.split():
        w = tok.lower().translate(str.maketrans('', '', VOTE_PUNCT))
        if not w:
            continue
        if w in VOTE_STOP or len(w) <= 1:
            out.append(tok)                       # function word / phrasing
        elif w in voted:
            out.append(tok)
            seen.add(w)
        # else: non-voted content word of the skeleton arm -> dropped
    missing = [w for w in voted if w not in seen]
    if not any(t.lower().translate(str.maketrans('', '', VOTE_PUNCT)) in voted
               for t in out):
        return ' '.join(voted) + '.'             # skeleton had no voted content
    s = ' '.join(out + missing)
    s = re.sub(r'\s+([,.!?;:])', r'\1', s)
    s = re.sub(r'\s+', ' ', s).strip()
    if s and s[0].islower():
        s = s[0].upper() + s[1:]
    if s and s[-1] not in '.!?':
        s += '.'
    return s


def vote_build(arms, keys, armlist, thr, out_path, method, backfill_path=None):
    """avhbench_vote_sc_rebuild.py build(): content-word vote + rebuild."""
    import collections
    import statistics
    from scipy import stats as st
    from answerability_gate import token_f1
    F = lambda a, b: token_f1(a, b) * 100

    recs, f1s, bf_f1s = [], [], []
    for gi, g in enumerate(keys):
        base = arms['224'][g]
        arm_pred, arm_f1, arm_nlp = {}, {}, {}
        for a in armlist:
            r = arms[a][g]
            arm_pred[a] = _clean_boxed(r['pred'])
            arm_f1[a] = round(F(r['pred'], r['gt']), 2)
            arm_nlp[a] = round(r.get('norm_logp') or 0.0, 4)
        cnt = collections.Counter()
        contrib = collections.defaultdict(list)
        for a in armlist:
            for w in set(_cwords(arm_pred[a])):
                cnt[w] += 1
                contrib[w].append(a)
        voted = [w for w, c in cnt.items() if c >= thr]
        order = []
        for a in armlist:
            for w in _cwords(arm_pred[a]):
                if w in voted and w not in order:
                    order.append(w)
        pred = ' '.join(order)
        f1 = F(pred, base['gt'])
        f1s.append(f1)
        skeleton = max(armlist, key=lambda a: arm_nlp[a])
        rec = dict(
            idx=gi, id=base['id'], dataset='avhbench', method=method,
            question=base['question'], gt=base['gt'], pred=pred,
            vote_threshold=thr, n_arms=len(armlist), arms=armlist,
            votes={w: cnt[w] for w in voted},
            word_contributors={w: contrib[w] for w in voted},
            dropped_words_below_thr={w: cnt[w] for w, c in cnt.items() if c < thr},
            skeleton_source_arm=skeleton,
            arm_preds=arm_pred, arm_f1=arm_f1, arm_norm_logp=arm_nlp,
            local_f1=round(f1, 2),
        )
        recs.append(rec)
        if backfill_path is not None:
            bf = _backfill(arm_pred[skeleton], order)
            rec['pred_backfill'] = bf
            rec['backfill_source_arm'] = skeleton
            bf_f1 = F(bf, base['gt'])
            rec['local_f1_backfill'] = round(bf_f1, 2)
            bf_f1s.append(bf_f1)
    with open(out_path, 'w', encoding='utf-8') as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    base_f1s = [r['arm_f1']['224'] for r in recs]
    w = st.wilcoxon(f1s, base_f1s)
    print(f"{method:>28} thr>={thr} -> {out_path.split('/')[-1]}")
    print("   n=%d  F1=%.2f  (base %.2f, delta %+.2f, p=%.2e, win/lose %d/%d)"
          % (len(recs), statistics.mean(f1s), statistics.mean(base_f1s),
             statistics.mean(f1s) - statistics.mean(base_f1s), w.pvalue,
             sum(1 for a, b in zip(f1s, base_f1s) if a > b),
             sum(1 for a, b in zip(f1s, base_f1s) if a < b)), flush=True)
    if backfill_path is not None:
        with open(backfill_path, 'w', encoding='utf-8') as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        wb = st.wilcoxon(bf_f1s, f1s)
        print("   backfill F1=%.2f (delta %+.2f vs plain vote, p=%.2e, win/lose %d/%d)"
              % (statistics.mean(bf_f1s),
                 statistics.mean(bf_f1s) - statistics.mean(f1s), wb.pvalue,
                 sum(1 for a, b in zip(bf_f1s, f1s) if a > b),
                 sum(1 for a, b in zip(bf_f1s, f1s) if a < b)), flush=True)
        return statistics.mean(f1s), statistics.mean(bf_f1s)
    return statistics.mean(f1s), None


def vote_load_arms(vote_cfg):
    arms = {}
    for k, arm in vote_cfg['arms'].items():
        k = str(k)                    # yaml renders bare 224 as an int
        p = resolve(arm['path'])
        recs = [json.loads(l) for l in open(p, encoding='utf-8')]
        d = {}
        for r in recs:
            d.setdefault(_normgt(r['gt']), r)      # first occurrence wins
        arms[k] = d
        print(f'[arm {k}] {len(recs)} lines -> {len(d)} gt-unique', flush=True)
    keys = sorted(set.intersection(*[set(a) for a in arms.values()]))
    print(f'[align] gt-key intersection: {len(keys)} samples', flush=True)
    return arms, keys


def avh_stage_arms(cfg, args):
    """Generate the same-backbone input-perturbation arms (224/384/448/f16).

    Each arm is a plain greedy contract-layer run under
    OMNI_FRAME_SIZE / OMNI_NUM_FRAMES, written straight to the arm's configured
    path. The cross-generation arm (q3, Qwen3-Omni-30B-A3B) is a different
    backbone and stays with its research script
    (OmniConfess/methods/base_model_qwen3_omni.py, separate venv) -- the
    config only points at its output file.
    """
    for k, arm in cfg['vote_sc']['arms'].items():
        if arm.get('kind', 'base') != 'base':
            print(f'[arm {k}] kind={arm["kind"]}: not generated here '
                  f'(artifact: {resolve(arm["path"])})', flush=True)
            continue
        if os.path.exists(resolve(arm['path'])):
            print(f'[arm {k}] exists, skipping (resume-safe)', flush=True)
            continue
        saved_env = {}
        for var in ('OMNI_FRAME_SIZE', 'OMNI_NUM_FRAMES'):
            if var in arm:
                saved_env[var] = os.environ.get(var)
                os.environ[var] = str(arm[var])
        arm_cfg = dict(cfg)
        arm_cfg['output_dir'] = os.path.dirname(resolve(arm['path']))
        arm_cfg['file_name'] = os.path.splitext(os.path.basename(resolve(arm['path'])))[0]
        arm_cfg['time_path'] = os.path.join(arm_cfg['output_dir'], 'time')
        arm_cfg['prompt_version'] = arm.get('prompt_version', cfg['prompt_version'])
        print(f'[arm {k}] generating -> {resolve(arm["path"])}', flush=True)
        run_contract(arm_cfg, args)
        for var, old in saved_env.items():
            if old is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = old


def avh_stage_vote(cfg, args):
    v = cfg['vote_sc']
    arms, keys = vote_load_arms(v)
    armlist = [str(k) for k in v['arms'].keys()]   # config order = vote order
    thr = int(v.get('threshold', 2))
    out_dir = resolve(v.get('out_dir', './outputs/avhbench_vote_sc'))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, v.get('out', 'avhbench_vote_sc_full.jsonl'))
    if v.get('backfill'):
        bf_path = out_path.replace('.jsonl', '_backfill.jsonl')
        vote_build(arms, keys, armlist, thr, out_path, v.get('method', 'vote_sc'),
                   backfill_path=bf_path)
    else:
        vote_build(arms, keys, armlist, thr, out_path, v.get('method', 'vote_sc'))
    print(f'[done] -> {out_path}', flush=True)


def run_vote_sc(cfg, args):
    stage = args.stage or 'all'
    if stage in ('all', 'arms'):
        avh_stage_arms(cfg, args)
    if stage in ('all', 'vote'):
        avh_stage_vote(cfg, args)


# ===========================================================================
# CLI
# ===========================================================================
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', type=str, default=None,
                    help='phd | cmm | pubmedqa | ragtruth | haloquest | avhbench')
    ap.add_argument('--config', type=str, default=None,
                    help='default: configs/<dataset>.yaml')
    ap.add_argument('--prompt_version', type=str, default=None,
                    help='override cfg prompt_version')
    ap.add_argument('--lambda_cf', type=float, default=None,
                    help='override cfg cf.lambda_cf (judgement only)')
    ap.add_argument('--control', type=str, default=None,
                    choices=['none', 'random_sign'],
                    help='override cf control arm')
    ap.add_argument('--data_path', type=str, default=None)
    ap.add_argument('--output_dir', type=str, default=None)
    ap.add_argument('--file_name', type=str, default=None)
    ap.add_argument('--time_path', type=str, default=None)
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--stage', type=str, default=None,
                    help='cf | contract | all | base | route | bon | feat | arms | vote')
    ap.add_argument('--limit', type=int, default=0,
                    help='process only the first N items (generation stages)')
    ap.add_argument('--dry-run', action='store_true',
                    help='alias for --limit 3 (smoke test)')
    args = ap.parse_args()

    if not args.config:
        if not args.dataset:
            ap.error('either --dataset or --config is required')
        args.config = os.path.join(HERE, 'configs', f'{args.dataset}.yaml')
    args.config = resolve(args.config)
    if not os.path.exists(args.config):
        raise SystemExit(f'[config] not found: {args.config}')

    cfg = apply_overrides(load_config(args.config), args)
    if args.dry_run:
        args.limit = max(args.limit, 3)
    if not args.dataset:
        args.dataset = cfg['dataset']

    task = cfg['task_type']
    if task not in TASK_ROUTES:
        raise SystemExit(f'[pipeline] unknown task_type {task!r}; expected one of '
                         f'{sorted(TASK_ROUTES)}')
    if not args.stage:
        args.stage = TASK_ROUTES[task]
    print(f'[pipeline] dataset={cfg["dataset"]} task_type={task} stage={args.stage} '
          f'config={os.path.relpath(args.config, HERE)} '
          f'prompt_version={cfg.get("prompt_version")}', flush=True)
    if task == 'judgement':
        run_judgement(cfg, args)
    elif task == 'descriptive-text':
        run_contract(cfg, args)
    elif task == 'descriptive-vqa':
        run_haloquest(cfg, args)
    elif task == 'perception-heavy':
        run_vote_sc(cfg, args)


if __name__ == '__main__':
    main()
