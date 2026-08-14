# -*- coding: utf-8 -*-
"""
base_opera_minicpm.py -- MiniCPM-o-2.6 OPERA baseline.

What OPERA is
-------------
Huang et al., CVPR 2024, "OPERA: Alleviating Hallucination in Multi-Modal Large
Language Models via Over-Trust Penalty and Retrospection-Allocation".

It introduces neither an extra branch nor an extra model, adding two mechanisms
inside beam search instead:

1. Over-trust penalty
   Hallucination often accompanies a knowledge-aggregation pattern in which several
   consecutive newly generated tokens all concentrate their attention on the same
   earlier position, appearing as a vertical stripe in the attention map. At that
   point the model has stopped looking at the image and is simply continuing its
   own text. The remedy takes the self-attention among the most recent k generated
   tokens (last layer, averaged over heads), forms a k-by-k lower-triangular local
   matrix, takes the product along each column and then the maximum, giving sigma,
   which is subtracted from the beam score.

2. Retrospection-allocation
   If the same position repeatedly becomes the maximal column of that stripe, the
   decoder is stuck: it rolls back to that position, bans the token chosen there,
   and re-expands.

Hyperparameters and their provenance
------------------------------------
No OPERA reference implementation was available to compare against while writing
this, so the defaults below trace to the paper and its official demo configuration
rather than to code:

  threshold           = 15   how many hits on the same column trigger a rollback
  num_attn_candidates = 5    the local window width k
  scale_factor        = 10   calibrated empirically here, NOT the paper's 50

Why scale_factor had to be recalibrated
---------------------------------------
The paper's scale_factor of 50 goes with its own definition of sigma, and copying
it directly breaks:
  - multiplying every element of the local matrix by scale_factor means that at
    k = 5 the column product is amplified by about 50^5 = 3e8. The measured sigma
    then sits stably between 15 and 50 while the log-probability differences are
    only about +-3, so beam search ends up minimising the penalty rather than
    maximising likelihood, and the output collapses to minimal sentences such as
    "There.";
  - more fundamentally, the columns of a lower-triangular matrix contain different
    numbers of valid terms (k-1 down to 1), so their products are not commensurate.
    The last column, which holds only the diagonal (self-attention around 0.15,
    far above the roughly 0.01 of cross-token attention), wins unconditionally and
    the penalty loses all discriminative power.
This implementation therefore excludes the diagonal (strict lower triangle) and
takes a geometric mean along each column. Under that definition sigma lands around
0.01 to 0.03 normally and around 0.08 when a stripe appears, so scale_factor = 10
yields penalties of 0.1 to 0.2 and around 0.8, the same order as the
log-probability differences: enough to correct the beam ordering without destroying
fluency. Changing the definition of sigma necessarily requires recalibrating, since
the two go together; with no reference implementation to compare against, empirical
calibration was the only option, and it is recorded here rather than presented as
the paper's number. Passing --scale_factor 0 degenerates OPERA to plain beam
search, which separates the effect of the penalty from the correctness of the beam
search implementation itself.

Known differences from the official implementation
--------------------------------------------------
  - The official code maintains the history per beam; here a rollback is taken
    against the current best beam's sequence, and all beams re-expand from that
    state afterwards.
  - The local attention is the last layer averaged over heads. The paper also uses
    the last layer but does not spell out how it aggregates over heads.
  - To avoid repeated rollbacks on pathological samples, --max_rollbacks (default
    8) caps them; the official code has no such limit.

Why model.chat() cannot be used
-------------------------------
OPERA needs the attention weights at every step, needs to rewrite the beam scores,
and needs to truncate the KV cache to roll back. chat() offers none of the three,
since it only forwards keys already present in the generation config. This script
therefore uses shared/minicpm_decode to reproduce chat()'s preprocessing and builds
its own beam search.

Note that this is the only baseline using beam search; the others are greedy, which
should be kept in mind when comparing.
"""
import os, sys, time, argparse, json, random, math
import torch
import numpy as np
import librosa
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
from shared.data_loader import load_dataset, resolve_media_path
from shared.eval_utils import make_result, save_result
from shared.prompts import get_system_prompt, get_user_prompt
from shared.minicpm_decode import (
    patch_smart_apply, build_minicpm_inputs, get_multimodal_embeds,
    get_terminators, opera_beam_generate,
)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

MODEL_PATH = "openbmb/MiniCPM-o-2_6"

DEFAULT_NUM_BEAMS = 5
DEFAULT_SCALE_FACTOR = 10.0
DEFAULT_THRESHOLD = 15
DEFAULT_NUM_ATTN_CANDIDATES = 5
DEFAULT_PENALTY_WEIGHTS = 1.0


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _patch_smart_apply():
    """Fix transformers 4.52+ incompatibility with MiniCPM-o custom modules."""
    patch_smart_apply()


def load_minicpm_model():
    from transformers import AutoModel, AutoTokenizer
    _patch_smart_apply()
    print(f'[model] loading MiniCPM-o-2.6 from {MODEL_PATH}...', flush=True)
    model = AutoModel.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        attn_implementation='sdpa',
        torch_dtype=torch.bfloat16,
        init_vision=True,
        init_audio=True,
        init_tts=False,
    )
    model = model.eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print(f'[model] loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB', flush=True)
    return model, tokenizer


def build_omni_content(video_path, audio_path):
    """Build per-second <unit> chunks for MiniCPM-o omni mode (video+audio)."""
    import cv2
    audio_np, sr = librosa.load(audio_path, sr=16000, mono=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    num_units = math.ceil(duration)

    contents = []
    for i in range(num_units):
        frame_idx = min(int((i + 1) * fps), total_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            image = Image.new('RGB', (224, 224))
        audio_chunk = audio_np[sr * i:sr * (i + 1)]
        if len(audio_chunk) == 0:
            audio_chunk = np.zeros(sr, dtype=np.float32)
        contents.extend(["<unit>", image, audio_chunk])
    cap.release()
    print(f'  [omni] {num_units} units from {duration:.1f}s video', flush=True)
    return contents


def build_content_for_item(item, data_path, data_root, dataset_name, prompt_version):
    """Build MiniCPM-o message content list. Returns (content_list, is_omni)."""
    json_dir = os.path.dirname(data_path)

    if item.get('video_path') and item.get('audio_path'):
        vp = resolve_media_path(item['video_path'], json_dir, data_root)
        ap = resolve_media_path(item['audio_path'], json_dir, data_root)
        if os.path.exists(vp) and os.path.exists(ap):
            content = build_omni_content(vp, ap)
            user_prompt = get_user_prompt(
                item['question'], dataset_name, prompt_version,
                passage=item.get('passage'))
            content.append(user_prompt)
            return content, True

    content = []
    if item.get('image_path'):
        ip = resolve_media_path(item['image_path'], json_dir, data_root)
        if os.path.exists(ip):
            content.append(Image.open(ip).convert('RGB'))

    user_prompt = get_user_prompt(
        item['question'], dataset_name, prompt_version,
        passage=item.get('passage'))
    content.append(user_prompt)

    return content, False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--data_root', type=str, default='./OmniHalluBench/')
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--file_name', type=str, default='minicpm_opera_out')
    parser.add_argument('--time_path', type=str, default='./results/time')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--max_samples', type=int, default=0,
                        help='0 = all samples')
    parser.add_argument('--prompt_version', type=str, default='v0')
    parser.add_argument('--num_beams', type=int, default=DEFAULT_NUM_BEAMS,
                        help='beam width (default 5, per the OPERA paper)')
    parser.add_argument('--scale_factor', type=float, default=DEFAULT_SCALE_FACTOR,
                        help='attention scaling before the column-wise product (default 50)')
    parser.add_argument('--threshold', type=int, default=DEFAULT_THRESHOLD,
                        help='how many times a column must win before rollback fires '
                             '(default 15)')
    parser.add_argument('--num_attn_candidates', type=int,
                        default=DEFAULT_NUM_ATTN_CANDIDATES,
                        help='local attention window width k (default 5)')
    parser.add_argument('--penalty_weights', type=float,
                        default=DEFAULT_PENALTY_WEIGHTS,
                        help='over-trust penalty weight (default 1.0)')
    parser.add_argument('--max_rollbacks', type=int, default=8,
                        help='safety cap on retrospection rollbacks per sample '
                             '(not present in the paper; guards pathological samples)')
    args = parser.parse_args()

    print(f"[opera] num_beams={args.num_beams}, scale_factor={args.scale_factor}, "
          f"threshold={args.threshold}, k={args.num_attn_candidates}, "
          f"penalty_weights={args.penalty_weights}", flush=True)

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.time_path, exist_ok=True)

    output_path = os.path.join(args.output_dir, args.file_name + '.jsonl')

    # Resume
    done_ids = set()
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line.strip())['id'])
                except Exception:
                    pass
        if done_ids:
            print(f'[resume] skipping {len(done_ids)} already-done samples', flush=True)

    model, tokenizer = load_minicpm_model()
    terminators = get_terminators(model, tokenizer)

    data = load_dataset(args.data_path, args.data_root)
    if args.max_samples > 0:
        data = data[:args.max_samples]

    system_prompt = get_system_prompt(args.datasets, args.prompt_version)
    start = time.time()
    n_rollback, n_pen, n_steps = 0, 0, 0

    for idx, item in enumerate(data):
        if item['id'] in done_ids:
            continue

        t0 = time.time()
        print(f'\n[{idx+1}/{len(data)}] id={item["id"]}', flush=True)

        try:
            content, is_omni = build_content_for_item(
                item, args.data_path, args.data_root,
                args.datasets, args.prompt_version)

            msgs = [{'role': 'user', 'content': content}]
            if is_omni:
                msgs.insert(0, model.get_sys_prompt(mode='omni', language='en'))
            elif system_prompt:
                msgs.insert(0, {'role': 'system', 'content': [system_prompt]})

            # Same preprocessing as chat(); only the decoding differs.
            inputs = build_minicpm_inputs(
                model, model.processor, msgs,
                omni_input=is_omni, max_slice_nums=1, use_image_id=False)
            embeds = get_multimodal_embeds(model, inputs)

            response, _, stats = opera_beam_generate(
                model, tokenizer, embeds,
                attention_mask=inputs.get('attention_mask'),
                num_beams=args.num_beams,
                scale_factor=args.scale_factor,
                threshold=args.threshold,
                num_attn_candidates=args.num_attn_candidates,
                penalty_weights=args.penalty_weights,
                max_new_tokens=args.max_new_tokens,
                max_rollbacks=args.max_rollbacks,
                terminators=terminators,
            )

            n_rollback += stats['rollbacks']
            n_pen += stats['penalised_steps']
            n_steps += stats['total_steps']

            elapsed = time.time() - t0
            print(f'  rollbacks={stats["rollbacks"]} '
                  f'penalised={stats["penalised_steps"]}/{stats["total_steps"]}',
                  flush=True)
            print(f'  response[:120]: {response[:120]!r}', flush=True)
            print(f'  time={elapsed:.1f}s', flush=True)

        except Exception as e:
            print(f'  [ERROR] {e}', flush=True)
            response = ''
            elapsed = time.time() - t0

        torch.cuda.empty_cache()

        result = make_result(item, response, args.datasets, 'minicpm_opera',
                             time_sec=elapsed)
        save_result(result, output_path)

    total_time = time.time() - start
    n_done = len(data) - len(done_ids)
    print(f'\n[done] {n_done} samples in {total_time/60:.1f} min '
          f'({n_rollback} rollbacks, {n_pen}/{n_steps} steps penalised). '
          f'Results: {output_path}', flush=True)

    summary = os.path.join(args.time_path, args.file_name + '.txt')
    with open(summary, 'w') as f:
        f.write(f'total_time: {total_time:.2f}s\n')
        f.write(f'samples: {n_done}\n')
        f.write(f'rollbacks: {n_rollback}\n')
        f.write(f'penalised_steps: {n_pen}\n')
        f.write(f'total_steps: {n_steps}\n')
        f.write(f'num_beams: {args.num_beams}\n')
        f.write(f'scale_factor: {args.scale_factor}\n')
        f.write(f'threshold: {args.threshold}\n')
        f.write(f'num_attn_candidates: {args.num_attn_candidates}\n')
        f.write(f'penalty_weights: {args.penalty_weights}\n')
        f.write(f'max_rollbacks: {args.max_rollbacks}\n')
        f.write('decode: beam search with over-trust penalty + '
                'retrospection-allocation rollback (bypasses model.chat)\n')


if __name__ == '__main__':
    main()
