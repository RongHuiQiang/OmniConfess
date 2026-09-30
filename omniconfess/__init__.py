"""omniconfess -- counterfactual-confession inference for multimodal hallucination.

A single importable package implementing the three-stage confession spine:

    Stage 1  Generate   greedy decode to the decision point.
    Stage 2  Confess     teacher-force the counterfactual / ablated view(s) on the
                        same prefix (once per sample, unconditionally).
    Stage 3  Decide      map the full-view signal + confession signal to the
                        committed answer (argmax_d E_d^R).

Six datasets are instantiated on this spine (phd / cmm / pubmedqa /
ragtruth / haloquest / avhbench) via :mod:`omniconfess.pipeline`.

Typical use is the top-level ``run_confess.py`` entry or::

    from omniconfess import main
    main(["haloquest", "--dry_run"])
"""

from .cli import main

__version__ = "1.0.0"
__url__ = "https://github.com/tsq/omniconfess"

__all__ = ["main", "__version__"]
