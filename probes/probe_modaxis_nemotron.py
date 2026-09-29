#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""probe_modaxis_nemotron.py -- modality-axis third backbone (Nemotron-3-Nano-
Omni-30B): per-token counterfactual deltas along the teacher-forced base
greedy trajectory, replicating the Qwen2.5 modality-axis calibers.

Families (one exclusive card, nemotron-hf venv, resumable):
  phd  (n<=150): full=<image>+own-q ; noq=<image>+NEUTRAL-q ; imgswap=<partner
        image>+own-q.   delta_v = full - imgswap (image-swap caliber, same as
        Qwen probe_cot_cf_phd_text.py), delta_t = full - noq.
  halo (n<=150): full=<image>+own-q ; noq=<image>+NEUTRAL-q ; noimg=q only
        (image masked -- halo v caliber is MASK, same as Qwen haloquest).
        delta_v = full - noimg, delta_t = full - noq.
  avh  (n=100): full=<video>+<audio>+q ; novideo / noaudio masked views
        (probe_cf_avhbench.py bigN caliber). delta_v = full - novideo,
        delta_a = full - noaudio.

NEUTRAL_Q = "(no question provided)" (placeholder scheme, primary in
analyze_textcf_robust.py).

Base trajectories (greedy, enable_thinking=False, token_ids + logp_list):
  phd  /ultrafast_share/tsq/tsq/nemotron/outputs/phd_nemotron/phd_nemotron_single_scored.jsonl
  halo /ultrafast_share/tsq/tsq/nemotron/outputs/haloquest_nemotron/haloquest_nemotron_single_scored.jsonl
  avh  /ultrafast_share/tsq/tsq/nemotron/outputs/avhbench_nemotron/avhbench_nemotron_single_scored.jsonl

Model loading / media plumbing copied verbatim from probe_cf_nemotron_blindspot.py
(yesterday's blindspot probe env, /ultrafast_share/tsq/venvs/nemotron-hf/bin/python).

Usage (card 6):
  CUDA_VISIBLE_DEVICES=6 PYTHONDONTWRITEBYTECODE=1 \
    /ultrafast_share/tsq/venvs/nemotron-hf/bin/python \
    probe_modaxis_nemotron.py --family phd --n 150
"""
import os, sys, json, time, argparse, random, hashlib, string

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'OmniConfess', 'methods'))
sys.path.insert(0, '/ultrafast_share/tsq/tsq/nemotron')
os.chdir(HERE)

import numpy as np
import torch

from shared import (load_dataset, resolve_media_path, get_system_prompt,
                    get_user_prompt)
from shared.data_loader import load_image
from base_model_nemotron_hf import (
    DEFAULT_MODEL, _repair_buffers, load_video_frames_native, load_audio_wave,
)

NEUTRAL_Q = "(no question provided)"

SCORED = {
    'phd': '/ultrafast_share/tsq/tsq/nemotron/outputs/phd_nemotron/phd_nemotron_single_scored.jsonl',
    'halo': '/ultrafast_share/tsq/tsq/nemotron/outputs/haloquest_nemotron/haloquest_nemotron_single_scored.jsonl',
    'avh': '/ultrafast_share/tsq/tsq/nemotron/outputs/avhbench_nemotron/avhbench_nemotron_single_scored.jsonl',
}
DATA = {
    'phd': './OmniHalluBench/phd/annotations/phd.json',
    'halo': './OmniHalluBench/haloquest/annotations/haloquest.json',
    'avh': './OmniHalluBench/avhbench/annotations/avhbench.json',
}

# ---------------- token classification (copied from probe_cf_ragtruth.py) ---
STOPWORDS = set([
    "a", "an", "the", "and", "or", "but", "if", "because", "as", "what",
    "which", "this", "that", "these", "those", "then", "just", "so", "than",
    "such", "both", "through", "about", "for", "is", "are", "was", "were",
    "of", "while", "during", "to", "in", "on", "at", "by", "with", "it",
    "there",
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
therefore thus furthermore moreover additionally another overall appears
appearing seem seems seemed likely probably possibly perhaps
""".split())
_FUNC = STOPWORDS | _FUNCTION_WORDS


def _is_word_initial(tok_str, prev_tok):
    if tok_str[:1].isspace():
        return True
    if prev_tok is None:
        return True
    return not (prev_tok[-1] if prev_tok else ' ').isalnum()


def token_class(tok_str, prev_tok=None):
    if tok_str.startswith('<|') or tok_str.startswith('<'):
        return 'punct'
    s = tok_str.strip()
    if not s:
        return 'punct'
    if not _is_word_initial(tok_str, prev_tok):
        return 'cont'
    w = s.lower().strip(string.punctuation + ' ')
    if not w:
        return 'punct'
    if any(c.isdigit() for w_ in [w] for c in w_):
        pass
    if not w[0].isalpha():
        return 'punct'
    if w in _FUNC:
        return 'func'
    return 'content'


def token_f1(pred, gt):
    from collections import Counter
    def toks(s):
        s = str(s).lower().translate(str.maketrans("", "", string.punctuation))
        return [w for w in s.split() if w not in STOPWORDS and w.strip()]
    p_t, g_t = toks(pred), toks(gt)
    if not p_t or not g_t:
        return 0.0
    common = Counter(p_t) & Counter(g_t)
    hit = sum(common.values())
    p, r = hit / len(p_t), hit / len(g_t)
    return (2 * p * r) / (p + r) if (p + r) > 0 else 0.0


def build_clean(tokenizer, system_prompt, user_text, image=None,
                videos=None, audio=None):
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
    if videos is not None:
        kw['videos'] = videos
    if audio is not None:
        kw['audio'] = audio
    return kw


def install_tf_shim(model):
    """Teacher-forcing shim: the omni wrapper only exposes generate(); its
    media-embedding prep (image/video/sound scatter) happens inside generate
    just before it delegates to language_model.generate(...). We intercept
    that delegation and run a single forward pass instead, so generate()
    becomes a full-sequence teacher-forced logit pass."""
    lm = model.language_model

    def _tf_generate(self, input_ids=None, inputs_embeds=None,
                     attention_mask=None, generation_config=None,
                     output_hidden_states=None, use_cache=False, **kw):
        assert inputs_embeds is not None, 'wrapper must pass inputs_embeds'
        with torch.inference_mode():
            out = self.forward(inputs_embeds=inputs_embeds,
                               attention_mask=attention_mask,
                               use_cache=False)
        return out.logits          # [B, N-1?, V]; wrapper returns it verbatim

    lm.generate = _tf_generate.__get__(lm, type(lm))


@torch.inference_mode()
def view_logp(processor, model, clean_kw, gen_ids, device):
    """Teacher-forced forward: logp of gen_ids under the view prompt.

    Implemented via model.generate + the tf shim above, so the wrapper's own
    media scatter (video/sound embedding placement) is reused verbatim."""
    inputs = processor(**clean_kw)
    inputs = {k: (v.to(device) if torch.is_tensor(v) else v)
              for k, v in inputs.items()}
    for k in ('num_patches', 'num_tokens', 'imgs_sizes'):
        inputs.pop(k, None)
    ids = torch.cat([inputs['input_ids'][0], gen_ids.to(
        inputs['input_ids'].device)], dim=0).unsqueeze(0)
    feed = ids[:, :-1]                       # logits[p] predicts token p+1
    kw = dict(input_ids=feed, attention_mask=torch.ones_like(feed),
              max_new_tokens=1)
    for k in ('pixel_values', 'pixel_values_videos', 'sound_clips',
              'sound_length'):
        if k in inputs and inputs[k] is not None:
            kw[k] = inputs[k]
    logits = model.generate(**kw)            # shim -> [B, N, V] float
    logits = logits[0].float()
    lsm = torch.log_softmax(logits, dim=-1)
    input_len = inputs['input_ids'].shape[1]
    pos = torch.arange(input_len - 1, ids.shape[1] - 1, device=logits.device)
    lp = lsm[pos, ids[0, input_len:].to(logits.device)]
    del logits, lsm
    return lp.cpu().numpy(), int(input_len)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--family', required=True, choices=['phd', 'halo', 'avh'])
    ap.add_argument('--n', type=int, default=150)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--min_tok', type=int, default=3)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--out', default='')
    ap.add_argument('--model_path', default=os.environ.get(
        'NEMOTRON_MODEL_PATH', DEFAULT_MODEL))
    args = ap.parse_args()

    fam = args.family
    out_path = args.out or (
        f'outputs/cf_probe/modaxis_nemotron/{fam}_modaxis_nemotron.jsonl')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    rows = [json.loads(l) for l in open(SCORED[fam])]
    rows = [r for r in rows
            if r.get('token_ids') and len(r['token_ids']) >= args.min_tok]
    data = load_dataset(DATA[fam], './OmniHalluBench/')
    json_dir = os.path.dirname(os.path.abspath(DATA[fam]))

    if fam == 'phd':
        by_q = {}
        for it in data:
            by_q.setdefault(it['question'], []).append(it)
        for r in rows[:50]:
            assert r['question'] in by_q
    elif fam == 'halo':
        by_qg = {(it['question'], it['gt']): it for it in data}
        assert all((r['question'], r['gt']) in by_qg for r in rows[:50])
    else:  # avh
        by_qg = {(it['question'], it['gt']): it for it in data}
        for r in rows:
            r['_f1'] = token_f1(r['pred'], r['gt'])
        assert all((r['question'], r['gt']) in by_qg for r in rows[:50])

    rng = random.Random(args.seed)
    if fam == 'avh':
        # stratified over F1 tiers (probe_cf_avhbench.py caliber)
        ordered = sorted(rows, key=lambda r: r['_f1'])
        k = len(ordered) // 3
        tiers = [ordered[:k], ordered[k:2 * k], ordered[2 * k:]]
        for t in tiers:
            rng.shuffle(t)
        sample = []
        while len(sample) < args.n:
            for t in tiers:
                if t and len(sample) < args.n:
                    sample.append(t.pop())
    else:
        pool = list(rows)
        rng.shuffle(pool)
        sample = pool[:args.n]
    if args.limit:
        sample = sample[:args.limit]
    print(f'[{fam}] {len(sample)}/{len(rows)} sampled (seed {args.seed})',
          flush=True)

    from transformers import AutoModel, AutoProcessor
    print(f'[model] loading {args.model_path} (bf16, one exclusive card)...',
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
    install_tf_shim(model)
    tokenizer = processor.tokenizer
    print(f'[model] loaded in {time.time()-t0:.0f}s', flush=True)

    system_prompt = get_system_prompt(fam, 'v1')

    done = set()
    if os.path.exists(out_path):
        for l in open(out_path):
            try:
                done.add(json.loads(l)['key'])
            except Exception:
                pass
        print(f'[resume] {len(done)} already done', flush=True)
    mode = 'a' if done else 'w'

    # phd partner-image pool: all items with a readable image (paths only)
    phd_pool = []
    if fam == 'phd':
        for it in data:
            ip = it.get('image_path') or (it.get('media') or {}).get('image')
            rp = resolve_media_path(ip, json_dir, './OmniHalluBench/') if ip else None
            if rp and os.path.exists(rp):
                phd_pool.append((it['question'], rp))
        print(f'[phd] partner pool: {len(phd_pool)} items with image', flush=True)

    def img_of(item):
        ip = item.get('image_path') or (item.get('media') or {}).get('image')
        rp = resolve_media_path(ip, json_dir, './OmniHalluBench/') if ip else None
        if not rp or not os.path.exists(rp):
            return None
        return load_image(rp)

    t0 = time.time()
    n_done = 0
    with open(out_path, mode) as fout:
        for r in sample:
            key = f"{r['id']}|{hashlib.md5(r['question'].encode()).hexdigest()[:8]}"
            if key in done:
                continue
            t1 = time.time()
            try:
                gen_ids = torch.tensor(r['token_ids'], dtype=torch.long)
                up = get_user_prompt(r['question'], fam, 'v1')
                up_noq = get_user_prompt(NEUTRAL_Q, fam, 'v1')
                assert up != up_noq
                n_tok = len(r['token_ids'])
                toks = [tokenizer.decode([t], skip_special_tokens=False)
                        for t in r['token_ids']]
                cls = [token_class(t, toks[i - 1] if i else None)
                       for i, t in enumerate(toks)]

                rec = dict(family=fam, key=key, id=r['id'],
                           q_hash=hashlib.md5(r['question'].encode()).hexdigest()[:12],
                           question=r['question'][:160], gt=r['gt'],
                           pred=r['pred'][:200], n_tok=n_tok,
                           tokens=toks, classes=cls)

                if fam in ('phd', 'halo'):
                    item = (by_q[r['question']][0] if fam == 'phd'
                            else by_qg[(r['question'], r['gt'])])
                    image = img_of(item)
                    if image is None:
                        print(f"  [skip] no image id={r['id']}", flush=True)
                        continue

                    lp_full, il = view_logp(processor, model, build_clean(
                        tokenizer, system_prompt, '<image>\n' + up,
                        image=image), gen_ids, device)
                    lp_noq, _ = view_logp(processor, model, build_clean(
                        tokenizer, system_prompt, '<image>\n' + up_noq,
                        image=image), gen_ids, device)

                    if fam == 'phd':
                        # image-swap v caliber: deterministic partner by q-hash
                        h = int(hashlib.md5(r['question'].encode())
                                .hexdigest(), 16)
                        for _ in range(20):
                            pq, prp = phd_pool[(h + _) % len(phd_pool)]
                            if pq != r['question']:
                                pim = load_image(prp)
                                if pim is not None:
                                    break
                        else:
                            pim = None
                        if pim is None:
                            print(f"  [skip] no partner id={r['id']}", flush=True)
                            continue
                        lp_v, _ = view_logp(processor, model, build_clean(
                            tokenizer, system_prompt, '<image>\n' + up,
                            image=pim), gen_ids, device)
                    else:  # halo: v = mask image
                        lp_v, _ = view_logp(processor, model, build_clean(
                            tokenizer, system_prompt, up), gen_ids, device)

                    rec.update(input_len=il,
                               delta_v=(lp_full - lp_v).round(4).tolist(),
                               delta_t=(lp_full - lp_noq).round(4).tolist(),
                               lp_full=lp_full.round(4).tolist())
                else:  # avh
                    item = by_qg[(r['question'], r['gt'])]
                    vp = resolve_media_path(item.get('video_path'),
                                            json_dir, './OmniHalluBench/')
                    ap_ = resolve_media_path(item.get('audio_path'),
                                             json_dir, './OmniHalluBench/')
                    videos = load_video_frames_native(vp)
                    audio = load_audio_wave(ap_)
                    if videos is None or audio is None:
                        print(f"  [skip] media missing id={r['id']}", flush=True)
                        continue

                    lp_full, il = view_logp(processor, model, build_clean(
                        tokenizer, system_prompt,
                        '<video>\n<so_embedding>\n' + up,
                        videos=videos, audio=audio), gen_ids, device)
                    lp_nv, _ = view_logp(processor, model, build_clean(
                        tokenizer, system_prompt, '<so_embedding>\n' + up,
                        audio=audio), gen_ids, device)
                    lp_na, _ = view_logp(processor, model, build_clean(
                        tokenizer, system_prompt, '<video>\n' + up,
                        videos=videos), gen_ids, device)
                    rec.update(input_len=il, f1=round(r['_f1'], 4),
                               delta_v=(lp_full - lp_nv).round(4).tolist(),
                               delta_a=(lp_full - lp_na).round(4).tolist(),
                               lp_full=lp_full.round(4).tolist())

                base_lp = np.asarray(r['logp_list'], dtype=np.float64)
                rec['repro_err'] = round(float(
                    np.max(np.abs(np.asarray(rec['lp_full']) - base_lp))), 4) \
                    if len(base_lp) == n_tok else None
                rec['sec'] = round(time.time() - t1, 2)
                fout.write(json.dumps(rec, ensure_ascii=False) + '\n')
                fout.flush()
                n_done += 1
                if n_done % 10 == 0:
                    print(f'  [{n_done}/{len(sample)}] {time.time()-t0:.0f}s '
                          f'repro={rec["repro_err"]} {rec["sec"]}s', flush=True)
                torch.cuda.empty_cache()
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f'  [ERROR] id={r["id"]}: {e}', flush=True)
                torch.cuda.empty_cache()

    print(f'[done] {n_done}/{len(sample)} in {time.time()-t0:.0f}s -> {out_path}',
          flush=True)


if __name__ == '__main__':
    main()
