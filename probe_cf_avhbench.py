# -*- coding: utf-8 -*-
"""probe_cf_avhbench.py — Step-1 signal probe for CF-Confess (counterfactual
modality-grounding) on AVHBench, with THREE pre-registered gates.

Background (EXPERIMENT_DESIGN.md, RAGTruth section): the text-only CF probe
died because Delta measured "word specificity / question echo", not
"parametric prior vs evidence". AVHBench is audio-visual captioning — masking
a modality token is REAL physical ablation (not deleting a text span), so the
signal may behave differently. Per the pre-registered lessons, NO operator is
built unless the gates below pass.

Delta definition (teacher-forced on the base_model greedy trajectory):

    Delta_audio(tok) = log P(tok | full view)  - log P(tok | video-only view)
    Delta_video(tok) = log P(tok | full view)  - log P(tok | audio-only view)

Positive Delta_audio => the audio stream raises this word's probability
(audio-grounded). Views are built exactly like base_model.py
(build_user_content_with_media_tokens), with the masked modality's
placeholder line AND processor media removed.

PRE-REGISTERED GATES (fixed before running; see EXPERIMENT_DESIGN.md):

  Gate 1 — word-initial piece determination. Qwen BPE merges the leading
      space into the token (" key"), so only pieces that BEGIN a word are
      eligible content positions. Verification: report the fraction of
      continuation pieces and their median |Delta|, which must be far below
      the word-initial median (the mechanical near-zero of continuations is
      exactly the bug the first ragtruth probe had).
      PASS := median|Delta(cont)| < 0.25 x median|Delta(word-initial)|.

  Gate 2 — length-confound control. Sample-level Spearman(rootless_rate,
      Token-F1) must remain < 0 with p < 0.05 AFTER controlling for pred
      length (log words; partial Spearman on ranks) for at least one eps on
      the grid. Rootless word := max(Delta_audio, Delta_video) < eps
      (grounded in NEITHER modality).

  Gate 3 — wrong-word vs right-word discriminability, PER MODALITY (the
      avhbench-specific structure). A word "hits" gt if its normalized form
      appears in the gt token set. The theory predicts:
        (3a) audio-class words:   Delta_audio(hit) > Delta_audio(miss)
        (3b) visual-class words:  Delta_video(hit) > Delta_video(miss)
      each tested one-sided (Mann-Whitney, rank AUC), cells with n>=10 per
      side. PASS := at least one matched pair significant (AUC>0.5,
      p<0.05) in the predicted direction AND no matched pair significantly
      reversed. Crossed pairs (audio words x Delta_video etc.) are controls.

Usage (sharded over GPUs):
    MMHALU_MODEL_ROOT=/ultrafast_share/tsq/MODELS \
    CUDA_VISIBLE_DEVICES=0 /ultrafast_share/tsq/venvs/omniconfess/bin/python \
        probe_cf_avhbench.py --shard_idx 0 --n_shards 4
    # aggregate once all shards are done:
    python probe_cf_avhbench.py --aggregate
"""
import os, sys, json, argparse, random, string, time, glob
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'OmniConfess', 'methods'))
from shared import (  # noqa: E402
    load_dataset, load_media_for_item, load_model, build_inputs,
    prepare_for_thinker, build_user_content_with_media_tokens, make_messages,
    get_system_prompt, get_user_prompt,
)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("MMHALU_MODEL_ROOT", "/ultrafast_share/tsq/MODELS")

# ---------------------------------------------------------------------------
# Token classification — copied from probe_cf_ragtruth.py (word-initial fix).
# ---------------------------------------------------------------------------
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
    """'num' | 'content' | 'func' | 'punct' | 'cont' (subword continuation)."""
    if tok_str.startswith('<|'):          # special tokens (<|im_end|> etc.)
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


# ---------------------------------------------------------------------------
# Word-class lexicons (audio sound/source words vs visual attribute words).
# Coverage is reported — words in neither bucket are 'other'.
# ---------------------------------------------------------------------------
_AUDIO_STEMS = """
sound sounds noise noisy loud louder quiet quietly silence silent music
musical song songs melody tune rhythm beat beats audio audible hear hears
heard hearing voice voices speech speaking speaks spoke talk talks talking
narration narrates narrating scream screams screaming shout shouts shouting
yell yells yelling whisper whispers whispering humming hums hum buzz buzzing
buzzer drone drones droning roar roaring rumble rumbling engine engines
motor motors revving idling horn honk honks honking honked beep beeps beeping
alarm sirens siren ringing ring rings chime chimes chiming bell bells
applause clap claps clapping clapped cheering cheers cheer guitar guitars
piano drum drums drumming violin flutes flute trumpet saxophone keyboard
cymbals tambourine instrumental acoustic chirp chirps chirping chirped tweet
tweets tweeting squawk squawking squawks bark barks barking barked meow
purring purrs moo neigh quack quacking croak croaking hoot hooting howl
howling growl growling hiss hissing sizzle sizzling crackle crackling pop
pops popping whoosh swoosh swish swishing thud thuds thump thumping bang
bangs banging crash crashes crashing screech screeching squeak squeaks
squeaking squeal squealing grunt grunts snort snorts bleat clang clangs
clanging jingle jingles jingling rattle rattles rattling whistle whistles
whistling gasp gasps gasping sigh sighs sighing breath breathes breathing
snore snoring wind winds breeze rustle rustles rustling rustled splash
splashes splashing splashed trickle trickles trickling gurgle gurgling waves
rain rains raining thunder thundering storm footsteps footstep tapping taps
tapped click clicks clicking knock knocks knocking hammered hammering hammer
drilling drill sawing whirring whir vibration vibrating pitched tones
background screaming yelled shouted spoken
""".split()

_VISUAL_STEMS = """
red blue green yellow purple pink brown black white gray grey golden silver
dark bright colorful colored man men woman women person people boy boys girl
girls child children baby guy guys lady crowd audience player players worker
workers shirt jackets jacket dress dresses suit suits pants jeans shorts hat
hats cap helmet helmets shoes boots sweater hoodie uniform tie scarf gloves
glasses sunglasses mask hair beard face hands hand arm arms leg legs table
tables chair chairs car cars truck trucks bus buses motorcycle bicycles
bicycle bike plane airplane helicopter boat ship train road street building
buildings house room kitchen bathroom bedroom wall walls floor ceiling door
doors window windows desk screen phone camera bottle cup glass plate food
pizza cake ball dog cat dogs cats kitten puppy animal animals tree trees
grass rock rocks water river mountain sky cloud clouds forest field flowers
flower leaf leaves background podium microphone stage television laptop book
books paper box boxes striped suitcases
""".split()


def _inflect(stems):
    out = set()
    for s in stems:
        out.update({s, s + 's', s + 'es', s + 'ing', s + 'd', s + 'ed'})
        if s.endswith('e'):
            out.add(s[:-1] + 'ing')            # wave -> waving
        if s.endswith('y'):
            out.add(s[:-1] + 'ies')            # city -> cities
    return {w for w in out if w}


AUDIO_WORDS = _inflect(_AUDIO_STEMS)
VISUAL_WORDS = _inflect(_VISUAL_STEMS)
_overlap = AUDIO_WORDS & VISUAL_WORDS
AUDIO_WORDS -= _overlap
VISUAL_WORDS -= _overlap


def word_modality_class(w):
    if w in AUDIO_WORDS:
        return 'audio'
    if w in VISUAL_WORDS:
        return 'visual'
    return 'other'


def norm_word(tok_str):
    return tok_str.strip().lower().strip(string.punctuation + ' ')


# ---------------------------------------------------------------------------
# Token-F1 (identical footing to scripts/tokenf1_fullset.py).
# ---------------------------------------------------------------------------
def _f1_tokens(s):
    s = str(s).lower().translate(str.maketrans("", "", string.punctuation))
    return [w for w in s.split() if w not in STOPWORDS and w.strip()]


def token_f1(pred, gt):
    p_t, g_t = _f1_tokens(pred), _f1_tokens(gt)
    if not p_t or not g_t:
        return 0.0
    common = Counter(p_t) & Counter(g_t)
    hit = sum(common.values())
    p = hit / len(p_t)
    r = hit / len(g_t)
    return (2 * p * r) / (p + r) if (p + r) > 0 else 0.0


def stratified_sample(recs, n, seed=42):
    rng = random.Random(seed)
    ordered = sorted(recs, key=lambda r: r['_f1'])
    k = len(ordered) // 3
    tiers = [ordered[:k], ordered[k:2 * k], ordered[2 * k:]]
    for t in tiers:
        rng.shuffle(t)
    picked = []
    while len(picked) < n:
        for t in tiers:
            if t and len(picked) < n:
                picked.append(t.pop())
    return picked


# ---------------------------------------------------------------------------
# Model forward — teacher-forced logp of the base greedy tokens under a view.
# ---------------------------------------------------------------------------
_MEDIA_KEYS = (
    'pixel_values_images', 'pixel_values', 'image_grid_thw',
    'pixel_values_videos', 'video_grid_thw', 'video_second_per_grid',
    'input_features', 'feature_attention_mask',
)


@torch.inference_mode()
def view_logp(thinker, clean, gen_ids):
    """Teacher-forced forward of [view prompt ; base greedy tokens].

    All media tensors of the view must be forwarded too (mirrors
    probe_cf_signal_phd.py) — otherwise the video/audio encoders never run
    and the placeholder tokens carry no media information.
    """
    input_len = clean['input_ids'].shape[1]
    full_ids = torch.cat([clean['input_ids'], gen_ids.unsqueeze(0)], dim=1)
    am = torch.ones_like(full_ids)
    kw = dict(input_ids=full_ids, attention_mask=am, use_cache=False)
    for k in _MEDIA_KEYS:
        if k in clean:
            kw[k] = clean[k]
    out = thinker(**kw)
    logits = out.logits[0].float()
    lsm = torch.log_softmax(logits, dim=-1)
    pos = torch.arange(input_len - 1, full_ids.shape[1] - 1, device=logits.device)
    lp = lsm[pos, gen_ids.to(logits.device)]
    del out, logits, lsm
    return lp.cpu().numpy()


def retry(fn, n=3, wait=5, tag=''):
    for i in range(n):
        try:
            return fn()
        except Exception as e:
            print(f'  [retry {i+1}/{n}] {tag}: {type(e).__name__}: {e}',
                  flush=True)
            time.sleep(wait * (i + 1))
    raise RuntimeError(f'retries exhausted: {tag}')


def build_view(processor, model, system_prompt, user_prompt,
               pixel_values, audio_values, keep_video, keep_audio):
    """Build one (full / video-only / audio-only) view.

    Masking = remove the modality's placeholder line from user content AND
    do not pass the media tensor to the processor (mirrors base_model.py).
    """
    prefix = ''
    if keep_video and pixel_values is not None:
        prefix += "<|vision_bos|><|VIDEO|><|vision_eos|>\n"
    if keep_audio and audio_values is not None:
        prefix += "<|audio_bos|><|AUDIO|><|audio_eos|>\n"
    messages = make_messages(system_prompt, prefix + user_prompt)

    pixel_in = pixel_values if keep_video else None
    audio_in = audio_values if keep_audio else None
    raw = retry(lambda: build_inputs(processor, model, messages,
                                     pixel_in, audio_in, None, 'avhbench'),
                tag='build_inputs')
    return prepare_for_thinker(raw, model)


# ---------------------------------------------------------------------------
# Per-shard run
# ---------------------------------------------------------------------------
def run_shard(args):
    recs = [json.loads(l) for l in open(args.scored)]
    recs = [r for r in recs if r.get('token_ids')]
    for r in recs:
        r['_f1'] = token_f1(r['pred'], r['gt'])
    sample = stratified_sample(recs, args.n_samples, args.seed)
    if args.limit:
        sample = sample[:args.limit]
    # shard split AFTER stratification so each shard spans the F1 range
    sample = [s for i, s in enumerate(sample)
              if i % args.n_shards == args.shard_idx]
    print(f'[probe shard {args.shard_idx}/{args.n_shards}] {len(sample)} samples '
          f'(F1 mean {np.mean([s["_f1"] for s in sample]):.3f})', flush=True)

    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'OmniHalluBench', 'avhbench', 'avhbench.json')
    data_root = os.path.dirname(data_path)
    data = load_dataset(data_path, data_root)
    # id is shard-local in the base output -> join on (question, gt), unique
    by_qg = {(d['question'], d['gt']): d for d in data}

    processor, model = load_model()
    tokenizer = processor.tokenizer
    thinker = model.thinker
    system_prompt = get_system_prompt('avhbench', 'v1')

    out_path = args.out_pattern.format(idx=args.shard_idx, n=args.n_shards)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fout = open(out_path, 'w')

    t_start = time.time()
    for i, rec in enumerate(sample):
        item = by_qg.get((rec['question'], rec['gt']))
        if item is None:
            print(f'  [skip] no dataset match', flush=True)
            continue
        try:
            pixel_values, audio_values, _ = retry(
                lambda: load_media_for_item(item, data_path, data_root),
                tag='load_media')
            if pixel_values is None or audio_values is None:
                print(f'  [skip] media missing id={rec["id"]}', flush=True)
                continue

            gen_ids = torch.tensor(rec['token_ids'], dtype=torch.long,
                                   device=model.device)
            user_prompt = get_user_prompt(item['question'], 'avhbench', 'v1')

            views = {}
            for name, kv, ka in (('full', True, True),
                                 ('novideo', False, True),
                                 ('noaudio', True, False)):
                clean = build_view(processor, model, system_prompt, user_prompt,
                                   pixel_values, audio_values, kv, ka)
                views[name] = retry(
                    lambda c=clean: view_logp(thinker, c, gen_ids),
                    tag=f'forward_{name}')
                torch.cuda.empty_cache()

            d_audio = views['full'] - views['noaudio']   # + => audio-grounded
            d_video = views['full'] - views['novideo']   # + => video-grounded

            toks = [tokenizer.decode([t]) for t in rec['token_ids']]
            cls = [token_class(t, toks[j - 1] if j else None)
                   for j, t in enumerate(toks)]

            base_lp = np.array(rec['logp_list'], dtype=np.float64)
            repro_err = (float(np.max(np.abs(views['full'] - base_lp)))
                         if len(base_lp) == len(views['full']) else float('nan'))

            fout.write(json.dumps({
                'id': rec['id'], 'gt': rec['gt'], 'pred': rec['pred'],
                'f1': rec['_f1'],
                'pred_words': len(rec['pred'].split()),
                'tokens': toks, 'classes': cls,
                'lp_full': views['full'].round(4).tolist(),
                'lp_noaudio': views['noaudio'].round(4).tolist(),
                'lp_novideo': views['novideo'].round(4).tolist(),
                'delta_audio': d_audio.round(4).tolist(),
                'delta_video': d_video.round(4).tolist(),
                'repro_err': round(repro_err, 4),
            }, ensure_ascii=False) + '\n')
            fout.flush()
        except Exception as e:
            print(f'  [ERROR] id={rec["id"]}: {type(e).__name__}: {e}',
                  flush=True)
        if (i + 1) % 5 == 0:
            print(f'  [{i+1}/{len(sample)}] {time.time()-t_start:.0f}s '
                  f'mem={torch.cuda.max_memory_allocated()/2**30:.1f}G',
                  flush=True)
        torch.cuda.empty_cache()

    fout.close()
    print(f'[done shard {args.shard_idx}] -> {out_path} '
          f'({time.time()-t_start:.0f}s)', flush=True)


# ---------------------------------------------------------------------------
# Aggregate + the three gates
# ---------------------------------------------------------------------------
def partial_spearman(x, y, z):
    """Spearman corr of x,y after linearly regressing out z (on ranks)."""
    from scipy.stats import spearmanr, rankdata
    x, y, z = map(np.asarray, (x, y, z))
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)

    def resid(a):
        A = np.stack([rz, np.ones_like(rz)], 1)
        beta, *_ = np.linalg.lstsq(A, a, rcond=None)
        return a - A @ beta

    rho, p = spearmanr(resid(rx), resid(ry))
    return float(rho), float(p)


def auc_score(pos, neg):
    """Rank AUC (pos > neg) + one-sided MWU p."""
    from scipy.stats import mannwhitneyu
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float('nan'), float('nan')
    u, p = mannwhitneyu(pos, neg, alternative='greater')
    return float(u / (len(pos) * len(neg))), float(p)


def iter_words(s):
    out = []
    for t, c, da, dv in zip(s['tokens'], s['classes'],
                            s['delta_audio'], s['delta_video']):
        if c != 'content':
            continue
        w = norm_word(t)
        if w:
            out.append((w, word_modality_class(w), da, dv))
    return out


def aggregate(args):
    from scipy.stats import spearmanr

    files = sorted(glob.glob(args.shard_glob))
    assert files, f'no shard files match {args.shard_glob}'
    per_sample = []
    for f in files:
        for line in open(f):
            line = line.strip()
            if line:
                per_sample.append(json.loads(line))
    print(f'[aggregate] {len(per_sample)} samples from {len(files)} files',
          flush=True)

    # ---------- word-level records ----------
    word_rows = []
    cont_d, init_d = [], []
    n_content = n_audio = n_visual = n_other = 0
    for s in per_sample:
        gt_tok = set(_f1_tokens(s['gt']))
        for t, c, da, dv in zip(s['tokens'], s['classes'],
                                s['delta_audio'], s['delta_video']):
            if c == 'cont':
                cont_d.extend([da, dv])
            elif c in ('content', 'num'):
                init_d.extend([da, dv])
            if c != 'content':
                continue
            w = norm_word(t)
            if not w:
                continue
            n_content += 1
            mc = word_modality_class(w)
            n_audio += mc == 'audio'
            n_visual += mc == 'visual'
            n_other += mc == 'other'
            word_rows.append({'w': w, 'mc': mc, 'hit': w in gt_tok,
                              'da': float(da), 'dv': float(dv)})

    # ---------- Gate 1: word-initial determination ----------
    g1 = {
        'n_pieces_total': int(sum(len(s['tokens']) for s in per_sample)),
        'n_content_positions': n_content,
        'n_continuation_pieces': len(cont_d) // 2,
        'median_abs_delta_continuation': float(np.median(np.abs(cont_d)))
        if cont_d else None,
        'median_abs_delta_word_initial': float(np.median(np.abs(init_d)))
        if init_d else None,
        'note': 'continuation pieces excluded everywhere below (the fix)',
    }

    # ---------- Gate 2: rootless rate vs F1, length-controlled ----------
    eps_grid = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0]
    g2 = {}
    for eps in eps_grid:
        rates, f1s, lens = [], [], []
        for s in per_sample:
            cw = iter_words(s)
            if not cw:
                continue
            rootless = [max(da, dv) < eps for (_, _, da, dv) in cw]
            rates.append(float(np.mean(rootless)))
            f1s.append(s['f1'])
            lens.append(np.log(max(s['pred_words'], 1)))
        rho_raw, p_raw = spearmanr(rates, f1s)
        rho_pc, p_pc = partial_spearman(rates, f1s, lens)
        g2[str(eps)] = {
            'rootless_rate_mean': round(float(np.mean(rates)), 4),
            'spearman_raw': round(float(rho_raw), 4),
            'p_raw': round(float(p_raw), 4),
            'spearman_len_controlled': round(rho_pc, 4),
            'p_len_controlled': round(p_pc, 4),
            'pearson_rate_loglen': round(float(
                np.corrcoef(rates, lens)[0, 1]), 4),
        }

    # ---------- Gate 3: per-modality hit vs miss discriminability ----------
    def rows(mc=None, hit=None):
        sel = word_rows
        if mc is not None:
            sel = [r for r in sel if r['mc'] == mc]
        if hit is not None:
            sel = [r for r in sel if r['hit'] == hit]
        return sel

    g3 = {}
    for mc, dkey, gname in (('audio', 'da', 'audio_x_delta_audio'),
                            ('visual', 'dv', 'visual_x_delta_video'),
                            ('audio', 'dv', 'audio_x_delta_video'),
                            ('visual', 'da', 'visual_x_delta_audio'),
                            ('other', 'da', 'other_x_delta_audio'),
                            ('other', 'dv', 'other_x_delta_video')):
        pos = [r[dkey] for r in rows(mc=mc, hit=True)]
        neg = [r[dkey] for r in rows(mc=mc, hit=False)]
        auc, p = auc_score(pos, neg)
        g3[gname] = {
            'n_hit': len(pos), 'n_miss': len(neg),
            'mean_hit': round(float(np.mean(pos)), 4) if pos else None,
            'mean_miss': round(float(np.mean(neg)), 4) if neg else None,
            'median_hit': round(float(np.median(pos)), 4) if pos else None,
            'median_miss': round(float(np.median(neg)), 4) if neg else None,
            'auc': round(auc, 4), 'mwu_one_sided_p': round(p, 6),
        }
    for mc in ('audio', 'visual'):
        for eps in (0.2,):
            h = [max(r['da'], r['dv']) < eps for r in rows(mc=mc, hit=True)]
            m = [max(r['da'], r['dv']) < eps for r in rows(mc=mc, hit=False)]
            g3[f'{mc}_rootless_rate_eps{eps}'] = {
                'hit': round(float(np.mean(h)), 4) if h else None,
                'miss': round(float(np.mean(m)), 4) if m else None,
            }

    coverage = {
        'content_words': n_content,
        'audio_class': n_audio, 'visual_class': n_visual, 'other': n_other,
        'classified_coverage': round((n_audio + n_visual) / max(n_content, 1), 4),
    }
    cnt_rooted, cnt_rootless = Counter(), Counter()
    for r in word_rows:
        (cnt_rooted if max(r['da'], r['dv']) >= 0.2 else cnt_rootless)[r['w']] += 1

    repro = [s['repro_err'] for s in per_sample]
    report = {
        'n_samples': len(per_sample),
        'repro_err_max': float(np.nanmax(repro)),
        'repro_err_median': float(np.nanmedian(repro)),
        'gate1_word_initial': g1,
        'gate2_length_controlled': g2,
        'gate3_modality_discriminability': g3,
        'word_class_coverage': coverage,
        'top_rooted_words': cnt_rooted.most_common(25),
        'top_rootless_words': cnt_rootless.most_common(25),
    }

    # ---------- pre-registered verdicts ----------
    g1_pass = (g1['median_abs_delta_continuation'] is not None and
               g1['median_abs_delta_word_initial'] is not None and
               g1['median_abs_delta_continuation'] <
               0.25 * max(g1['median_abs_delta_word_initial'], 1e-9))
    g2_pass = any(v['spearman_len_controlled'] < 0 and
                  v['p_len_controlled'] < 0.05 for v in g2.values())
    matched = [g3['audio_x_delta_audio'], g3['visual_x_delta_video']]
    testable = [v for v in matched if v['n_hit'] >= 10 and v['n_miss'] >= 10]
    g3_pass = bool(testable) and \
        any(v['auc'] > 0.5 and v['mwu_one_sided_p'] < 0.05 for v in testable) \
        and not any(v['auc'] < 0.5 and v['mwu_one_sided_p'] < 0.05
                    for v in testable)
    report['verdicts'] = {
        'gate1_word_initial': 'PASS' if g1_pass else 'FAIL',
        'gate2_length_controlled': 'PASS' if g2_pass else 'FAIL',
        'gate3_modality_discriminability': 'PASS' if g3_pass else 'FAIL',
        'all_gates': 'PASS' if (g1_pass and g2_pass and g3_pass) else 'FAIL',
    }

    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, 'w') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(args.report.replace('.json', '_words.jsonl'), 'w') as f:
        for r in word_rows:
            f.write(json.dumps(r) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[done] -> {args.report}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scored', default='outputs/avhbench_base_model/'
                                        'avhbench_base_model_full.jsonl')
    ap.add_argument('--n_samples', type=int, default=120)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--shard_idx', type=int, default=0)
    ap.add_argument('--n_shards', type=int, default=1)
    ap.add_argument('--out_pattern',
                    default='outputs/cf_probe/avh_probe_shard{idx}of{n}.jsonl')
    ap.add_argument('--aggregate', action='store_true')
    ap.add_argument('--shard_glob',
                    default='outputs/cf_probe/avh_probe_shard*of*.jsonl')
    ap.add_argument('--report', default='outputs/cf_probe/avh_probe_report.json')
    args = ap.parse_args()

    if args.aggregate:
        aggregate(args)
    else:
        run_shard(args)


if __name__ == '__main__':
    main()
