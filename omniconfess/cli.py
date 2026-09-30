# -*- coding: utf-8 -*-
"""cli.py -- friendly command-line front-end for the confession pipeline.

The heavy lifting lives in :func:`omniconfess.pipeline.main`, which uses a
``--datasets <name>`` flag plus a large family of per-dataset knobs. This module
wraps it in the ergonomic form the README advertises::

    python run_confess.py <dataset> [extra pipeline args ...]

    <dataset> = phd | cmm | pubmedqa | ragtruth | haloquest | avhbench

It does two things and nothing else (so the underlying decision logic is never
touched):

  1. Turns the leading positional ``<dataset>`` into ``--datasets <dataset>``.
  2. Fills in the conventional path defaults (data_path / output_dir /
     file_name) that ``run_omniconfess.sh`` used to pass, unless the caller
     already supplied them. Judgement datasets read
     ``OmniHalluBench/<ds>/annotations/<ds>.json``; haloquest / avhbench manage
     their own inputs and only receive an output dir.

Everything after the dataset token is forwarded verbatim to the pipeline parser.
"""
import os
import sys

from . import pipeline

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VALID = ('phd', 'cmm', 'pubmedqa', 'ragtruth', 'haloquest', 'avhbench')
# Datasets that read a flat annotations JSON via --data_path.
_JUDGE = ('phd', 'cmm', 'pubmedqa', 'ragtruth')


def _default_file_name(ds):
    if ds == 'avhbench':
        return 'avhbench_vote'
    return f'{ds}_confess'


def _build_argv(ds, rest):
    """Assemble the argv the pipeline parser expects from the friendly form."""
    argv = ['--datasets', ds]

    def has(flag):
        return any(a == flag or a.startswith(flag + '=') for a in rest)

    # Judgement datasets need an input annotations file.
    if ds in _JUDGE and not has('--data_path'):
        argv += ['--data_path',
                 os.path.join(REPO_ROOT, 'OmniHalluBench', ds,
                              'annotations', f'{ds}.json')]
    if not has('--output_dir'):
        argv += ['--output_dir',
                 os.path.join(REPO_ROOT, 'outputs', f'omniconfess_{ds}')]
    if not has('--file_name'):
        argv += ['--file_name', _default_file_name(ds)]

    return argv + list(rest)


def main(argv=None):
    """Entry point. ``argv`` defaults to ``sys.argv[1:]``."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ('-h', '--help'):
        prog = os.path.basename(sys.argv[0]) or 'run_confess.py'
        print(f'usage: {prog} <{"|".join(VALID)}> [pipeline args ...]')
        print('\nRuns the three-stage confession pipeline on one dataset.')
        print('Extra arguments are forwarded to omniconfess.pipeline; see')
        print(f'  {prog} <dataset> --help  (via --datasets) for the full list.')
        return 0 if argv else 2

    ds = argv[0].lower()
    if ds not in VALID:
        sys.stderr.write(
            f'error: unknown dataset {argv[0]!r}. '
            f'choose from: {" ".join(VALID)}\n')
        return 2

    # Hand the reconstructed argv to the pipeline parser, which reads sys.argv.
    sys.argv = [sys.argv[0]] + _build_argv(ds, argv[1:])
    pipeline.main()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
