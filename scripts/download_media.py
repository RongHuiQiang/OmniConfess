#!/usr/bin/env python3
"""download_media.py — download media files for OmniHalluBench datasets.

Each dataset's media files are fetched from the original source repositories.
Downloaded files are placed under OmniHalluBench/<ds>/media/ to match the
paths in the annotation files.

Usage
-----
# Download all datasets
python scripts/download_media.py --dataset all

# Download a single dataset
python scripts/download_media.py --dataset haloquest

# Download to a custom root
python scripts/download_media.py --dataset phd --data_root /data/OmniHalluBench
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "OmniHalluBench"

# ---------------------------------------------------------------------------
# Per-dataset download instructions
# ---------------------------------------------------------------------------
DATASETS = {
    "phd": {
        "description": "PHD — Passage-level Hallucination Detection (EMNLP 2023)",
        "paper":  "https://arxiv.org/abs/2310.06498",
        "source": "https://github.com/maybenotime/PHD",
        "media_type": "images",
        "instructions": """
PHD does not provide a bulk media download script. Please:
  1. Visit https://github.com/maybenotime/PHD and follow the data download
     instructions in that repository.
  2. Place the downloaded images under:
       OmniHalluBench/phd/media/
  3. Image paths in annotations/phd.json use the prefix media/images/.
""",
    },
    "cmm": {
        "description": "CMM — The Curse of Multi-Modalities (NeurIPS 2025)",
        "paper":  "https://arxiv.org/abs/2410.12787",
        "source": "https://github.com/DAMO-NLP-SG/CMM",
        "media_type": "video + audio",
        "instructions": """
CMM video/audio files are hosted by the original authors. Please:
  1. Visit https://github.com/DAMO-NLP-SG/CMM or
     https://huggingface.co/datasets/DAMO-NLP-SG/CMM
     and follow the download instructions.
  2. Place the downloaded files under:
       OmniHalluBench/cmm/media/
""",
    },
    "pubmedqa": {
        "description": "PubMedQA — Biomedical QA (EMNLP 2019)",
        "paper":  "https://arxiv.org/abs/1909.06146",
        "source": "https://github.com/pubmedqa/pubmedqa",
        "media_type": "text only",
        "instructions": """
PubMedQA is text-only — no media files are needed.
Annotations are already included in OmniHalluBench/pubmedqa/annotations/.
""",
    },
    "ragtruth": {
        "description": "RAGTruth — RAG Hallucination Benchmark (ACL 2024)",
        "paper":  "https://arxiv.org/abs/2401.00396",
        "source": "https://github.com/ParticleMedia/RAGTruth",
        "media_type": "text only",
        "instructions": """
RAGTruth is text-only — no media files are needed.
Annotations are already included in OmniHalluBench/ragtruth/annotations/.
""",
    },
    "haloquest": {
        "description": "HaloQuest — Visual Hallucination Benchmark (ECCV 2024)",
        "paper":  "https://arxiv.org/abs/2407.15680",
        "source": "https://github.com/google/haloquest",
        "license": "CC BY 4.0",
        "media_type": "images",
        "instructions": """
HaloQuest images are available from the original Google Research repository.
Please follow the download instructions at:
  https://github.com/google/haloquest

Place the downloaded images under:
  OmniHalluBench/haloquest/media/images/

Note: 3 images from Flickr are missing from the original release:
  media/images/3_2718_4502888288_e4860b29c3_o.jpg  (2 samples)
  media/images/4_3256_2789033013_c2f79a2bec_o.jpg  (1 sample)
These 3 samples will be skipped during evaluation.
""",
    },
    "avhbench": {
        "description": "AVHBench — Audio-Visual Hallucination Benchmark (ICLR 2025)",
        "paper":  "https://arxiv.org/abs/2410.18325",
        "source": "https://github.com/kaist-ami/AVHBench",
        "media_type": "video + audio",
        "instructions": """
AVHBench video/audio files are hosted by the original authors. Please:
  1. Visit https://github.com/kaist-ami/AVHBench and follow the download
     instructions in that repository.
  2. Place the downloaded files under:
       OmniHalluBench/avhbench/media/
""",
    },
}


def print_instructions(ds_name, data_root):
    cfg = DATASETS[ds_name]
    print(f"\n{'='*60}")
    print(f"  {cfg['description']}")
    print(f"  Paper:  {cfg['paper']}")
    print(f"  Source: {cfg['source']}")
    if "license" in cfg:
        print(f"  License: {cfg['license']}")
    print(f"  Target: {data_root}/{ds_name}/media/")
    print(cfg["instructions"])


def check_annotations(ds_name, data_root):
    ann_path = Path(data_root) / ds_name / "annotations" / f"{ds_name}.json"
    if not ann_path.exists():
        print(f"[warn] annotations not found: {ann_path}")
        return 0
    with open(ann_path) as f:
        data = json.load(f)
    print(f"[ok] {ds_name}: {len(data)} samples in annotations")
    return len(data)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="all",
                    choices=list(DATASETS) + ["all"],
                    help="Dataset to download (default: all).")
    ap.add_argument("--data_root", default=str(DATA_ROOT),
                    help="Root directory for OmniHalluBench data.")
    args = ap.parse_args()

    datasets = list(DATASETS) if args.dataset == "all" else [args.dataset]

    print(f"\nOmniHalluBench media downloader")
    print(f"Data root: {args.data_root}")
    print(f"Datasets:  {', '.join(datasets)}")
    print(
        "\nNote: Media files must be downloaded from the original sources due to\n"
        "licensing constraints. This script prints the instructions for each dataset.\n"
    )

    for ds in datasets:
        check_annotations(ds, args.data_root)
        print_instructions(ds, args.data_root)


if __name__ == "__main__":
    main()
