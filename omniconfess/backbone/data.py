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

def load_video_frames(video_path, num_frames=None):
    """Load video as numpy array of RGB frames, verified pattern from GUARD.

    Resolution and frame count are controlled by environment variables so that
    A/B ladders can rerun the pipeline without touching call sites:
        OMNI_FRAME_SIZE  square resize target, default 224 (legacy behaviour)
        OMNI_NUM_FRAMES  number of sampled frames, default 8 (legacy behaviour)
    An explicit ``num_frames`` argument still wins over the env var.
    """
    if num_frames is None:
        num_frames = int(os.environ.get("OMNI_NUM_FRAMES", "8"))
    frame_size = int(os.environ.get("OMNI_FRAME_SIZE", "224"))
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
        frame = cv2.resize(frame, (frame_size, frame_size))
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.array(frames) if frames else None


def _load_audio_pyav(audio_path, sr=24000):
    """Decode audio from a media container (e.g. .mp4) with PyAV, resampled to
    mono float32 at ``sr``. Fallback for when libsndfile/audioread cannot open
    the container (this environment lost its ffmpeg audioread backend, so
    librosa.load fails on avhbench's mp4 files that carry the audio track)."""
    try:
        import av
        import numpy as _np
        container = av.open(audio_path)
        if not any(s.type == 'audio' for s in container.streams):
            container.close()
            return None
        resampler = av.audio.resampler.AudioResampler(
            format='flt', layout='mono', rate=sr)
        chunks = []
        for frame in container.decode(audio=0):
            for rf in resampler.resample(frame):
                chunks.append(rf.to_ndarray().reshape(-1))
        container.close()
        if not chunks:
            return None
        a = _np.concatenate(chunks).astype('float32')
        return a if len(a) > 0 else None
    except Exception:
        return None


def load_audio_data(audio_path, sr=24000):
    """Load audio as numpy array at 24kHz, verified pattern from GUARD.

    Tries librosa (soundfile -> audioread) first; if that fails (e.g. no
    ffmpeg/audioread backend for an .mp4 container) falls back to PyAV, which
    decodes the mp4's audio track in-process."""
    if not audio_path or not os.path.exists(audio_path):
        return None
    try:
        audio, _ = librosa.load(audio_path, sr=sr)
        if audio is not None and len(audio) > 0:
            return audio
    except Exception:
        pass
    return _load_audio_pyav(audio_path, sr=sr)


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


# Which facet acts as the grouping key for each subset. The unified schema keeps
# each subset's own taxonomy under ``facets`` with explicit names instead of the
# positional category/sub_category pair, because for some subsets the second field
# was never a child of the first: HaloQuest's two fields are orthogonal axes
# (image source x question type), and CMM's ``granularity`` was a function of its
# ``sub_category``. Grouping only needs a one-to-one label, so picking one facet
# per subset reproduces the original grouping exactly -- the label text is shorter
# ("icc" rather than "PhD-icc"), but no group is merged or split.
GROUP_FACET = {
    'cmm':       'hallucination_type',
    'phd':       'mode',
    'avhbench':  'task',
    'haloquest': 'question_type',
}


def _alias_legacy(item):
    """Map a unified-schema record onto the legacy field names.

    The annotation JSONs ship the unified schema (``uid`` / ``media`` / ``facets``
    / ``label``); a lot of call sites still read ``sub_category``, ``video_path``
    and friends. Rather than duplicating both spellings on disk, the legacy names
    are derived here at load time. Records already in the legacy layout pass
    through untouched, so either layout loads.
    """
    if 'media' not in item and 'facets' not in item:
        return item                                   # legacy record, nothing to do

    media = item.get('media') or {}
    facets = item.get('facets') or {}
    ds = item.get('dataset', '')
    group = facets.get(GROUP_FACET.get(ds, ''))
    if group is None:                                 # subsets with no taxonomy
        group = ds or 'Default'

    out = dict(item)
    out.update({
        'sub_category': group,
        'category':     ds,
        'video_path':   media.get('video'),
        'audio_path':   media.get('audio'),
        'image_path':   media.get('image'),
    })
    # Only surface an ``id`` when the subset genuinely had one. CMM, AVHBench and
    # PhD never did, and load_dataset falls back to the row number for them --
    # substituting ``uid`` here would silently renumber every result file. Note
    # that the ids that do exist are pre-filter indices into the original release
    # (HaloQuest has 365 rows numbered up to 608), so they are provenance, not a
    # join key: cross-file alignment keys on (question, ground truth).
    orig_id = (item.get('source') or {}).get('orig_id')
    if 'id' not in out and orig_id is not None:
        out['id'] = orig_id
    return out


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
    Resolved absolute media paths are NOT stored here - call resolve_media_path() at inference time.

    Both the unified and the legacy annotation layouts are accepted; see
    ``_alias_legacy``. The unified extras (``task_type``, ``answer_space``,
    ``label``, ``facets``, ``uid``) are carried through when present, so a scorer
    can pick its mode from the data instead of matching on the file name.
    """
    items = load_json_file(data_path)
    normalized = []
    for i, raw in enumerate(items):
        item = _alias_legacy(raw)
        entry = {
            'id':          str(item.get('id', item.get('video_id', i))),
            'question':    item.get('question', ''),
            'gt':          str(item.get('answer', item.get('gt', item.get('ground_truth', item.get('label', ''))))),
            'group_key':   item.get('sub_category', item.get('task', item.get('category', 'Default'))),
            'video_path':  item.get('video_path', item.get('video', None)),
            'audio_path':  item.get('audio_path', item.get('audio', None)),
            'image_path':  item.get('image_path', item.get('image', None)),
            'passage':     item.get('passage', item.get('context', None)),
            '_raw':        raw,
        }
        # Legacy spelling kept alongside group_key: base_vcd and base_dola read
        # ``sub_category`` off this entry, which never existed here, so their
        # result files recorded an empty group. Supplying it fixes that without
        # touching either file.
        entry['sub_category'] = entry['group_key']
        for k in ('uid', 'task_type', 'answer_space', 'label', 'facets'):
            if k in item:
                entry[k] = item[k]
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
