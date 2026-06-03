# Comparative Anatomy of Volatility Spillovers: Terra/LUNA vs FTX

Preserves the event-study notebook as sequential scripts and packages the required crypto inputs as local parquet assets.

The original notebook is preserved unchanged in `docs/reference/ftx-crash.ipynb`. The new execution notebook in `notebooks/demo.ipynb` only calls the Python backend under `src/`.

## Layout

```text
ftx-crash/
├── README.md
├── pyproject.toml
├── src/
│   ├── GUIDE_src.md
│   └── ftx_crash/          # importable package (config, pipeline, cli, steps)
├── scripts/
│   ├── run_pipeline.py     # thin CLI wrapper
│   └── convert_raw_to_parquet.py
├── notebooks/
│   └── demo.ipynb
├── data/
│   ├── raw/                # restore original source files here for rebuild
│   ├── interim/
│   └── processed/          # runtime parquet inputs
├── outputs/
│   ├── figures/
│   ├── tables/
│   ├── reports/
│   └── runs/
├── docs/
│   ├── user/
│   └── reference/          # original notebook, copy-out, split map
├── tests/
├── logs/
```

See `docs/reference/notebook_split.md` for the section-to-script map.

## Run

```bash
uv sync
uv run ftx-crash
```

Equivalent wrappers:

```bash
uv run python scripts/run_pipeline.py
```

If you ever need to rebuild the parquet files, restore the original crypto source files under `data/raw/` and rerun:

```bash
uv run python scripts/convert_raw_to_parquet.py
```
