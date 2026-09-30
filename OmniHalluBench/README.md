# OmniHalluBench

OmniHalluBench is a unified evaluation benchmark covering six multimodal hallucination datasets. Annotations are provided in this repository. Media files (images, videos, audio) must be downloaded separately from the original sources due to licensing constraints.

## Structure

```
OmniHalluBench/
├── phd/annotations/phd.json           # 906 samples — image + text
├── cmm/annotations/cmm.json           # 721 samples — video + audio
├── pubmedqa/annotations/pubmedqa.json # 468 samples — text only
├── ragtruth/annotations/ragtruth.json # 435 samples — text only
├── haloquest/annotations/haloquest.json # 365 samples — image
└── avhbench/annotations/avhbench.json # 645 samples — video + audio
```

Each annotation file contains the full sample set with media paths normalized to `media/` prefix. Evaluation scripts expect:

```
--data_path OmniHalluBench/<ds>/annotations/<ds>.json
--data_root OmniHalluBench/<ds>/
```

## Downloading Media Files

Run the provided download script to fetch media files for each dataset:

```bash
python scripts/download_media.py --dataset all
# or for a single dataset:
python scripts/download_media.py --dataset haloquest
```

Media files will be placed under `OmniHalluBench/<ds>/media/` to match the paths in the annotation files.

## Original Sources

| Dataset | Paper | Source | License |
|---------|-------|--------|---------|
| PHD | [EMNLP 2023](https://arxiv.org/abs/2310.06498) | [GitHub](https://github.com/maybenotime/PHD) | Not specified |
| CMM | [NeurIPS 2025](https://arxiv.org/abs/2410.12787) | [GitHub](https://github.com/DAMO-NLP-SG/CMM) | See repo |
| PubMedQA | [EMNLP 2019](https://arxiv.org/abs/1909.06146) | [GitHub](https://github.com/pubmedqa/pubmedqa) | Public domain (PubMed) |
| RAGTruth | [ACL 2024](https://arxiv.org/abs/2401.00396) | [GitHub](https://github.com/ParticleMedia/RAGTruth) | Not specified |
| HaloQuest | [ECCV 2024](https://arxiv.org/abs/2407.15680) | [GitHub](https://github.com/google/haloquest) | CC BY 4.0 |
| AVHBench | [ICLR 2025](https://arxiv.org/abs/2410.18325) | [GitHub](https://github.com/kaist-ami/AVHBench) | See repo |
