# GUIDE_ROOT.md

## Part 1: Conceptual Explanation

This repository is a notebook-to-project conversion. The notebook reference materials live under `docs/reference/`, and the executable workflow is split into ordered Python step scripts under `src/ftx_crash/steps/`. The root folder keeps the project thin: `pyproject.toml` defines the Python 3.13 environment and registers the `ftx-crash` CLI, `scripts/` holds thin wrappers and one-off helpers, `notebooks/` holds the new thin execution notebook, `docs/` records both the section split and a direct notebook copy-out, and `data/` plus `outputs/` hold local inputs and generated artifacts. Because this notebook depended on local crypto source files, `scripts/convert_raw_to_parquet.py` records the one-off raw-to-parquet conversion process, and `data/processed/` stores the parquet files used by the scripts.

The execution model intentionally mirrors notebook semantics. Each step script is executed in order inside one shared namespace, so variables, functions, and imported modules persist across sections just as they did in the original notebook. This keeps the code close to the source notebook while moving the reusable logic out of the new notebook wrapper.

## Part 2: Code Reference

- `pyproject.toml`: Python 3.13 package metadata, dependencies for `uv`, and the `ftx-crash` console script entrypoint.
- `scripts/run_pipeline.py`: Thin wrapper around `ftx_crash.cli.main`.
- `scripts/convert_raw_to_parquet.py`: One-off helper that rebuilds the filtered parquet inputs when the original source files are restored locally under `data/raw/`.
- `src/ftx_crash/config.py`: Defines project paths and builds the shared execution context.
- `src/ftx_crash/pipeline.py`: Runs each generated step script in notebook order.
- `src/ftx_crash/cli.py`: Command-line entrypoint for the pipeline.
- `src/ftx_crash/steps/`: Contains the notebook-derived Python scripts, one file per major notebook section.
- `notebooks/demo.ipynb`: Thin notebook that only calls the backend pipeline.
- `docs/reference/ftx-crash.ipynb`: Unchanged copy of the original source notebook.
- `docs/reference/notebook_reference.md`: Markdown copy-out of the source notebook in notebook order.
- `docs/reference/notebook_split.md`: Maps notebook sections to generated script files.
- `data/processed/`: Parquet copies used by the converted pipeline.

## Part 3: Short Journal

- 2026-04-16: Split the original notebook into ordered step scripts while preserving the raw notebook under `docs/reference/`.
- 2026-04-16: Added a Markdown notebook copy-out so the project keeps a readable version of the original notebook text locally.
- 2026-04-16: Replaced the earlier pickle-based processed data layer with an explicit raw-to-parquet conversion step.
- 2026-05-20: Aligned the repo layout with the standard project structure: package CLI under `src/ftx_crash/cli.py`, thin script wrapper under `scripts/`, and placeholder folders for `data/raw/`, `docs/user/`, `tests/`, and `logs/`.
