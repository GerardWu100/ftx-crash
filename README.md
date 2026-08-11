# Comparative Anatomy of Volatility Spillovers: Terra/LUNA vs FTX

An event study of BTC futures basis around two crypto crashes, converted from a
Jupyter notebook into an ordered, runnable pipeline.

## What it does

Compares how the BTC futures basis (the gap between futures and spot price)
behaved during two crises: the Terra/LUNA collapse (April-June 2022) and the
FTX collapse (October-December 2022). It uses BTC 1-minute spot data and the
BITO ETF as a futures proxy, plus Binance BTC/USDT perpetual and quarterly
futures data.

For each crisis it computes the abnormal basis (the futures basis minus its
pre-crisis average), tests whether the two crises differ using a t-test,
Mann-Whitney U test, and Cohen's d, and estimates the half-life of the
dislocation. Beyond the core event study it adds several extensions built on
the [RiskLabAI](https://github.com/RiskLabAI/RiskLabAI.py) library: a
real-time crisis classifier, CUSUM-filter event-based sampling, fractional
differentiation to make the basis series stationary, a causal-inference
backdoor adjustment, and synthetic-data robustness checks.

The original notebook is preserved unchanged at
`docs/reference/ftx-crash.ipynb`. This repo splits it into 19 ordered step
scripts under `src/ftx_crash/steps/` that run in one shared namespace, so the
code stays close to the notebook instead of being refactored into new
abstractions. See `docs/reference/notebook_split.md` for the section-to-script
map.

## Requirements

- Python >=3.13
- No external services or API keys. The pipeline reads local parquet files
  already bundled under `data/processed/`.
- Rebuilding those parquet files from scratch requires the original raw
  CSV/TXT source files restored under `data/raw/` (not included in the repo).

## Setup

```bash
uv sync
```

## Usage

```bash
uv run ftx-crash                                 # run the full pipeline
uv run python scripts/run_pipeline.py             # equivalent thin wrapper
uv run python scripts/convert_raw_to_parquet.py    # rebuild data/processed/ from data/raw/
```

`ftx-crash` accepts `--smoke` (reserved for smaller verification settings,
currently a no-op) and `--print-keys` (prints the final execution context
keys after the run).

## Layout

```text
ftx-crash/
├── src/ftx_crash/          # config, pipeline runner, CLI, and the 19 step scripts
├── scripts/                # run_pipeline.py wrapper, convert_raw_to_parquet.py
├── notebooks/demo.ipynb    # thin notebook that only calls the package pipeline
├── data/
│   ├── raw/                # restore original source files here to rebuild
│   └── processed/          # runtime parquet inputs used by the pipeline
├── outputs/                # figures/, tables/, reports/, runs/
├── docs/reference/         # original notebook, its markdown copy-out, split map
└── tests/
```

## Output

Running the pipeline writes:

- `outputs/figures/` - plots (BTC price trajectory, crisis comparison panels,
  feature importance, statistical significance).
- `outputs/tables/` - CSV tables (descriptive statistics, statistical test
  results, feature importance, event timeline).

## License

All rights reserved. See [LICENSE](LICENSE).
