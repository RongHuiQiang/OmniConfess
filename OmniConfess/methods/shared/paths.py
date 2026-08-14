"""Shim so that ``methods/`` scripts can reach the top-level paths module."""

import os
import sys

_OMNICONFESS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _OMNICONFESS not in sys.path:
    sys.path.insert(0, _OMNICONFESS)

from paths import (  # noqa: E402,F401
    REPO_ROOT, DATASET_DIR, RESULTS_DIR, MODEL_ROOT, resolve_model, repo_path,
)
