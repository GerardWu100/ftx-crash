# Project Overview

## File Tree

```text
ftx-crash/
├── GUIDE_OVERVIEW.md
├── GUIDE_ROOT.md
├── README.md
├── pyproject.toml
├── docs/
│   ├── user/
│   └── reference/
│       ├── GUIDE_reference.md
│       ├── ftx-crash.ipynb
│       ├── notebook_reference.md
│       └── notebook_split.md
├── notebooks/
│   ├── GUIDE_notebooks.md
│   └── demo.ipynb
├── scripts/
│   ├── GUIDE_scripts.md
│   ├── run_pipeline.py
│   └── convert_raw_to_parquet.py
├── src/
│   ├── GUIDE_src.md
│   └── ftx_crash/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── pipeline.py
│       └── steps/
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── outputs/
│   ├── figures/
│   ├── tables/
│   ├── reports/
│   └── runs/
├── tests/
└── logs/
```

## Purpose

Preserves the event-study notebook as sequential scripts and packages the required crypto inputs as local parquet assets.

## Flow

1. `uv run ftx-crash`, `scripts/run_pipeline.py`, or `notebooks/demo.ipynb` calls the package pipeline.
2. `scripts/convert_raw_to_parquet.py` documents the one-off conversion used to rebuild the runtime parquet files from the original source files under `data/raw/`.
3. The pipeline builds a shared execution context with project paths and optional smoke-test overrides.
4. The step scripts in `src/ftx_crash/steps/` execute in notebook order.
5. Outputs are written under `outputs/`, while `docs/reference/` holds the original notebook, the notebook copy-out, and the split map, and the runtime parquet data lives under `data/processed/`.

## Main Assumptions

- The generated step scripts should stay close to the notebook code instead of being deeply refactored.
- Notebook state is preserved through one shared execution context.
- The bundled runtime data in `data/processed/` is local to this project copy and does not mutate the original `one-time-projects` files.
