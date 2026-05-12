# -*- coding: utf-8 -*-
"""
base_pd_minicpm.py — MiniCPM-o-2.6 prompt-based decoding (simplified PD).

Since MiniCPM only exposes model.chat(), we adapt PD as follows:
  1. Generate num_rollout candidate responses via sampling
  2. TF-IDF + KMeans clustering to find the largest cluster
  3. Pick the representative closest to the centroid

Usage:
    CUDA_VISIBLE_DEVICES=1 python base_pd_minicpm.py \
        --datasets avhbench --data_path ./OmniHalluBench/avhbench/avh_converted.json \
        --output_dir ./results --file_name minicpm_pd_avh_v0 --max_samples 2
"""
import os, sys, time, argparse, json, random, math
import torch
import numpy as np
import librosa
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from shared.data_loader import load_dataset, resolve_media_path
from shared.eval_utils import make_result, save_result
from shared.prompts import get_system_prompt, get_user_prompt

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

MODEL_PATH = "./models/MiniCPM-o-2.6"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _patch_smart_apply():
    """Fix transformers 4.52+ incompatibility with MiniCPM-o custom modules."""
    import torch
    def patched_smart_apply(self, fn):
        for module in self.children():
            if hasattr(module, "_init_weights"):
                if hasattr(module, "_initialize_weights"):
                    module.smart_apply(module._initialize_weights)
                else:
                    module.smart_apply(fn)
            else:
                module.smart_apply(fn)
        fn(self)
        return self
    torch.nn.Module.smart_apply = patched_smart_apply


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
    # Load audio
    audio_np, sr = librosa.load(audio_path, sr=16000, mono=True)
    # Get video duration and fps
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
    """Build MiniCPM-o message content list.

    For video+audio (avhbench): uses omni mode with per-second <unit> chunks.
    For image-only / text-only: uses standard content list.
    Returns (content_list, is_omni).
    """
    json_dir = os.path.dirname(data_path)

    # --- Omni mode: video + audio (avhbench) ---
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

    # --- Standard mode: image / text ---
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


def select_representative(candidates, cluster_num, seed):
    """TF-IDF + KMeans clustering, return the representative from the largest cluster.

    Falls back to the first non-empty candidate if clustering fails.
    """
    # Filter non-empty candidates
    valid = [(i, c) for i, c in enumerate(candidates) if c.strip()]
    if not valid:
        return candidates[0] if candidates else '', candidates

    if len(valid) == 1:
        return valid[0][1], candidates

    v_indices = [v[0] for v in valid]
    v_texts = [v[1] for v in valid]

    try:
        k_val = min(cluster_num, len(v_texts))
        vect = TfidfVectorizer(min_df=1)
        X = vect.fit_transform(v_texts)
        km = KMeans(n_clusters=k_val, n_init='auto', random_state=seed)
        km.fit(X)
        labels = km.labels_

        # Find the largest cluster
        counts = np.bincount(labels, minlength=k_val)
        best_cluster = int(np.argmax(counts))
        print(f'    cluster sizes: {counts.tolist()}, picking cluster {best_cluster}', flush=True)

        # Among members of the largest cluster, pick the one closest to centroid
        cluster_members = [j for j, l in enumerate(labels) if l == best_cluster]
        centroid = km.cluster_centers_[best_cluster]
        dists = []
        for j in cluster_members:
            vec = X[j].toarray().flatten()
            dists.append(np.linalg.norm(vec - centroid))
        best_j = cluster_members[int(np.argmin(dists))]
        return v_texts[best_j], candidates

    except Exception as e:
        print(f'    cluster failed: {e}, using first candidate', flush=True)
        return v_texts[0], candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--data_root', type=str, default='./OmniHalluBench/')
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--file_name', type=str, default='minicpm_pd_out')
    parser.add_argument('--time_path', type=str, default='./results/time')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--max_samples', type=int, default=0,
                        help='0 = all samples')
    parser.add_argument('--prompt_version', type=str, default='v0')
    parser.add_argument('--num_rollout', type=int, default=4,
                        help='Number of candidate responses to generate')
    parser.add_argument('--temperature', type=float, default=0.6,
                        help='Sampling temperature for candidate generation')
    parser.add_argument('--cluster_num', type=int, default=2,
                        help='Number of KMeans clusters')
    args = parser.parse_args()

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

    data = load_dataset(args.data_path, args.data_root)
    if args.max_samples > 0:
        data = data[:args.max_samples]

    system_prompt = get_system_prompt(args.datasets, args.prompt_version)
    start = time.time()

    for idx, item in enumerate(data):
        if item['id'] in done_ids:
            continue

        t0 = time.time()
        print(f'\n[{idx+1}/{len(data)}] id={item["id"]}', flush=True)

        final_response = ''
        cot = ''

        try:
            content, is_omni = build_content_for_item(
                item, args.data_path, args.data_root,
                args.datasets, args.prompt_version)

            # ── Phase 1: Generate num_rollout candidates with sampling ──
            candidates = []
            for r in range(args.num_rollout):
                msgs = [{'role': 'user', 'content': content}]

                if is_omni:
                    omni_sys = model.get_sys_prompt(mode='omni', language='en')
                    msgs.insert(0, omni_sys)
                    resp = model.chat(
                        msgs=msgs,
                        tokenizer=tokenizer,
                        sampling=True,
                        temperature=args.temperature,
                        max_new_tokens=args.max_new_tokens,
                        omni_input=True,
                        use_tts_template=False,
                        generate_audio=False,
                        max_slice_nums=1,
                        use_image_id=False,
                        return_dict=False,
                    )
                else:
                    if system_prompt:
                        msgs.insert(0, {'role': 'system', 'content': [system_prompt]})
                    resp = model.chat(
                        msgs=msgs,
                        tokenizer=tokenizer,
                        sampling=True,
                        temperature=args.temperature,
                        max_new_tokens=args.max_new_tokens,
                    )

                candidates.append(resp)
                print(f'    rollout {r+1}/{args.num_rollout}: {resp[:80]!r}', flush=True)
                torch.cuda.empty_cache()

            # ── Phase 2: TF-IDF clustering to select representative ──
            final_response, _ = select_representative(
                candidates, args.cluster_num, args.seed)

            # Store all candidates in cot for traceability
            cot_parts = []
            for i, c in enumerate(candidates):
                cot_parts.append(f'[candidate {i+1}] {c}')
            cot_parts.append(f'[selected] {final_response}')
            cot = '\n---\n'.join(cot_parts)

            elapsed = time.time() - t0
            print(f'  response[:120]: {final_response[:120]!r}', flush=True)
            print(f'  time={elapsed:.1f}s', flush=True)

        except Exception as e:
            print(f'  [ERROR] {e}', flush=True)
            final_response = ''
            cot = ''
            elapsed = time.time() - t0

        torch.cuda.empty_cache()

        result = make_result(item, final_response, args.datasets, 'minicpm_pd',
                             time_sec=elapsed, cot=cot)
        save_result(result, output_path)

    total_time = time.time() - start
    n_done = len(data) - len(done_ids)
    print(f'\n[done] {n_done} samples in {total_time/60:.1f} min. Results: {output_path}', flush=True)

    summary = os.path.join(args.time_path, args.file_name + '.txt')
    with open(summary, 'w') as f:
        f.write(f'total_time: {total_time:.2f}s\n')
        f.write(f'samples: {n_done}\n')
        f.write(f'num_rollout: {args.num_rollout}\n')
        f.write(f'temperature: {args.temperature}\n')
        f.write(f'cluster_num: {args.cluster_num}\n')


if __name__ == '__main__':
    main()
