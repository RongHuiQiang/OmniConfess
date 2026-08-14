# -*- coding: utf-8 -*-
"""
Unified dataset loader for all Baselines methods.
Handles CMM, PhD, HaloQuest, AVH, HaluEval, RAGTruth, PubMedQA.
Media loading patterns verified from GUARD/base_hook_best_0314.py.
"""
import os
import json
import numpy as np
import cv2
import librosa
from PIL import Image


# ================= Media Loading =================

def load_video_frames(video_path, num_frames=8):
    """Load video as numpy array of RGB frames (224x224), verified pattern from GUARD."""
    if not video_path or not os.path.exists(video_path):
        return None
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return None
    indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (224, 224))
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.array(frames) if frames else None


def load_audio_data(audio_path, sr=24000):
    """Load audio as numpy array at 24kHz, verified pattern from GUARD."""
    if not audio_path or not os.path.exists(audio_path):
        return None
    try:
        audio, _ = librosa.load(audio_path, sr=sr)
        return audio if (audio is not None and len(audio) > 0) else None
    except Exception:
        return None


def load_image(image_path):
    """Load image as PIL RGB Image, verified pattern from GUARD."""
    if not image_path or not os.path.exists(image_path):
        return None
    try:
        return Image.open(image_path).convert('RGB')
    except Exception:
        return None


# ================= Path Resolution =================

def resolve_media_path(rel_path, json_dir, data_root):
    """
    3-step fallback path resolution, verified from GUARD/base_hook_best_0314.py lines 774-787.
    Also handles COCO val2014/train2014 swap.
    """
    if not rel_path:
        return ""
    clean_rel = rel_path.lstrip('./')

    path1 = os.path.abspath(os.path.join(json_dir, clean_rel))
    if os.path.exists(path1):
        return path1

    path2 = os.path.abspath(os.path.join(data_root, clean_rel))
    if os.path.exists(path2):
        return path2

    path3 = os.path.abspath(os.path.join(data_root, "PhD", clean_rel))
    if os.path.exists(path3):
        return path3

    # val2014 / train2014 COCO swap (GUARD lines 813-816)
    for candidate in [path1, path2]:
        swapped = None
        if 'val2014' in candidate:
            swapped = candidate.replace('val2014', 'train2014')
        elif 'train2014' in candidate:
            swapped = candidate.replace('train2014', 'val2014')
        if swapped and os.path.exists(swapped):
            return swapped

    return path1  # best-effort fallback


# ================= Dataset Loading =================

def load_json_file(data_path):
    """Load a JSON or JSONL file, return list of dicts."""
    items = []
    with open(data_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    if content.startswith('['):
        items = json.loads(content)
    else:
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return items


def load_dataset(data_path, data_root):
    """
    Load a dataset JSON file and normalize each item to a unified schema:
        {
            'id'         : str,
            'question'   : str,
            'gt'         : str,
            'group_key'  : str,   # sub_category / task label
            'video_path' : str or None,  # raw relative path from JSON
            'audio_path' : str or None,
            'image_path' : str or None,
            'passage'    : str or None,  # text-only datasets
            '_raw'       : dict,          # original item for debug
        }
    Resolved absolute media paths are NOT stored here — call resolve_media_path() at inference time.
    """
    items = load_json_file(data_path)
    normalized = []
    for i, item in enumerate(items):
        entry = {
            'id':          str(item.get('id', item.get('video_id', i))),
            'question':    item.get('question', ''),
            'gt':          str(item.get('answer', item.get('gt', item.get('ground_truth', item.get('label', ''))))),
            'group_key':   item.get('sub_category', item.get('task', item.get('category', 'Default'))),
            'video_path':  item.get('video_path', item.get('video', None)),
            'audio_path':  item.get('audio_path', item.get('audio', None)),
            'image_path':  item.get('image_path', item.get('image', None)),
            'passage':     item.get('passage', item.get('context', None)),
            '_raw':        item,
        }
        normalized.append(entry)
    return normalized


def load_media_for_item(item, data_path, data_root):
    """
    Resolve and load all media for a normalized item dict.
    Returns (pixel_values, audio_values, image_values).
    """
    json_dir = os.path.dirname(data_path)

    pixel_values = None
    audio_values = None
    image_values = None

    if item.get('video_path'):
        vp = resolve_media_path(item['video_path'], json_dir, data_root)
        pixel_values = load_video_frames(vp)
        if pixel_values is None:
            print(f"[MEDIA WARN] video not found: {vp}", flush=True)

    if item.get('audio_path'):
        ap = resolve_media_path(item['audio_path'], json_dir, data_root)
        audio_values = load_audio_data(ap)
        if audio_values is None:
            print(f"[MEDIA WARN] audio not found: {ap}", flush=True)

    if item.get('image_path'):
        ip = resolve_media_path(item['image_path'], json_dir, data_root)
        image_values = load_image(ip)
        if image_values is None:
            print(f"[MEDIA WARN] image not found: {ip}", flush=True)

    return pixel_values, audio_values, image_values
