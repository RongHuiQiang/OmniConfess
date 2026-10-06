# OmniConfess

[![arXiv](https://img.shields.io/badge/arXiv-2610.02999-b31b1b.svg)](https://arxiv.org/abs/2610.02999)

Official resources of **"OmniConfess: Eliciting Token Confessions to Mitigate Omni-Modal Hallucination"**. Huiqiang Rong, Haoran Luo, Hui Feng, Zhonghong Ou, Kaiwen Xue, Guoxin Zhang, Yifan Zhu. **arXiv Preprint 2026** [[paper](https://arxiv.org/abs/2610.02999)].

## Overview

<div align="center">
  <img src="./figs/figure4.png" width="90%"/>
</div>

## Environment Setup

```bash
source scripts/env.sh
```

For an open-source install:

```bash
pip install -r requirements.txt
```

Model weights are loaded from `MMHALU_MODEL_ROOT` (default: Qwen2.5-Omni-7B).

## Quick Start

```bash
# PHD
$PY run_confess.py phd --output_dir outputs/omniconfess_phd

# CMM
$PY run_confess.py cmm --output_dir outputs/omniconfess_cmm

# PubMedQA
$PY run_confess.py pubmedqa --output_dir outputs/omniconfess_pubmedqa

# RAGTruth
$PY run_confess.py ragtruth --output_dir outputs/omniconfess_ragtruth

# HaloQuest (shardable across GPUs)
CUDA_VISIBLE_DEVICES=4,5,6,7 \
$PY run_confess.py haloquest \
    --halo_shard 0 --halo_nshards 4 \
    --output_dir outputs/omniconfess_haloquest

# AVHBench
$PY run_confess.py avhbench \
    --arm_paths a0.jsonl,a1.jsonl,a2.jsonl,a3.jsonl \
    --output_dir outputs/omniconfess_avhbench
```

`--dry_run` limits each job to the first few samples for a smoke test.

> For evaluation, please refer to the [evaluation](./evaluation/README.md) folder.

## BibTeX

If you find this work helpful for your research, please cite:

```bibtex
@misc{rong2026omniconfess,
  title         = {OmniConfess: Eliciting Token Confessions to Mitigate Omni-Modal Hallucination},
  author        = {Huiqiang Rong and Haoran Luo and Hui Feng and Zhonghong Ou and Kaiwen Xue and Guoxin Zhang and Yifan Zhu},
  year          = {2026},
  eprint        = {2610.02999},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2610.02999}
}
```
