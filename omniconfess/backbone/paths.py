"""paths.py -- repository and model-weight location resolution.

Everything in the ``omniconfess`` package resolves paths through this module so
that a fresh clone runs without editing source. Two anchors:

* ``REPO_ROOT`` is derived from this file's location, so dataset and result
  paths never depend on the working directory.
* ``MODEL_ROOT`` comes from the ``MMHALU_MODEL_ROOT`` environment variable.
  Model weights are not distributed with this repository; set the variable to
  a directory holding local checkpoints, or leave it unset to let
  ``transformers`` download from the Hugging Face Hub.

Example::

    export MMHALU_MODEL_ROOT=/srv/hf_models
    # a checkpoint at /srv/hf_models/Qwen2.5-Omni-7B is then picked up
    # automatically in place of the hub id "Qwen/Qwen2.5-Omni-7B"
"""

import os

# omniconfess/backbone/paths.py -> repo root is three levels up.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET_DIR = os.path.join(REPO_ROOT, 'OmniHalluBench')
RESULTS_DIR = os.path.join(REPO_ROOT, 'results_shipped')

MODEL_ROOT = os.environ.get('MMHALU_MODEL_ROOT')


def resolve_model(hf_id):
    """Map a Hugging Face repo id to a local checkpoint when one is available.

    Looks for ``$MMHALU_MODEL_ROOT/<name>`` where ``<name>`` is tried both as
    the full repo id and as the bare model name, plus a lowercased variant with
    dots and dashes normalised to underscores (the layout many local mirrors
    use). Falls back to ``hf_id`` unchanged, which makes ``transformers``
    download from the hub.
    """
    if not MODEL_ROOT:
        return hf_id
    name = hf_id.split('/')[-1]
    candidates = [
        hf_id,
        name,
        name.lower().replace('-', '_').replace('.', '_'),
        name.lower().replace('-', '_'),
    ]
    for c in candidates:
        p = os.path.join(MODEL_ROOT, c)
        if os.path.isdir(p):
            return p
    return hf_id


def repo_path(*parts):
    """Join a path relative to the repository root."""
    return os.path.join(REPO_ROOT, *parts)
