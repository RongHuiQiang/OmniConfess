"""omniconfess.datasets -- per-dataset instantiations of the confession spine.

Each module implements only the dataset-specific hooks (the counterfactual
channels ablated in Stage 2 and the Stage-3 decision math); the shared spine
lives in :mod:`omniconfess.spine` / :mod:`omniconfess.spine_freeform`.
"""
from .haloquest import run_haloquest_pipeline

__all__ = ["run_haloquest_pipeline"]
