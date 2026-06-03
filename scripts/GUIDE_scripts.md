# GUIDE_scripts.md

## Part 1: Conceptual Explanation

This folder holds thin wrappers and one-off maintenance scripts. The main pipeline command lives in the package as `ftx_crash.cli`, registered in `pyproject.toml` as the `ftx-crash` console script. `run_pipeline.py` is a thin wrapper for callers who prefer an explicit script path. `convert_raw_to_parquet.py` documents how the local runtime dataset was rebuilt from the original source files restored under `data/raw/`.

## Part 2: Code Reference

- `run_pipeline.py`: Thin wrapper that calls `ftx_crash.cli.main`.
- `convert_raw_to_parquet.py`: Reads the original crypto files when they are restored locally, applies the same date-window filtering used by the project, and writes the runtime parquet files consumed by `src/ftx_crash/steps/`.

## Part 3: Short Journal

- 2026-04-16: Replaced the earlier pickle-based processed data layer with an explicit raw-to-parquet conversion script.
- 2026-05-20: Moved the root pipeline CLI into the package and kept a thin script wrapper here.
