#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_confess.py -- top-level entry for the OmniConfess three-stage runner.

Usage:
    python run_confess.py <dataset> [pipeline args ...]

    <dataset> = phd | cmm | pubmedqa | ragtruth | haloquest | avhbench

Examples:
    python run_confess.py phd
    python run_confess.py haloquest --halo_shard 0 --halo_nshards 8
    python run_confess.py pubmedqa --dry_run

Model weights are resolved through ``MMHALU_MODEL_ROOT`` (see
``omniconfess/backbone/paths.py``). Because the resolved path is computed when
the package is imported, this variable is set here BEFORE importing omniconfess.
Point ``MMHALU_MODEL_ROOT_OMNI`` at a directory holding a local
``Qwen2.5-Omni-7B`` checkpoint to run fully offline; otherwise transformers
falls back to the Hugging Face Hub.
"""
import os
import sys

# Must precede the omniconfess import so paths.resolve_model() sees it.
os.environ.setdefault('MMHALU_MODEL_ROOT',
                      os.environ.get('MMHALU_MODEL_ROOT_OMNI', '/dev/shm'))

from omniconfess.cli import main

if __name__ == '__main__':
    raise SystemExit(main())
