"""Notebook section: import libraries and setup."""

# Import libraries and setup
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import RiskLabAI.utils.publication_plots as pub_plots
from scipy import stats

# Output paths shared by every downstream step (created once in config.build_execution_context).
FIG_DIR = str(FIGURES_DIR)
TAB_DIR = str(TABLES_DIR)

# Publication defaults used before RiskLabAI theme is applied.
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman"],
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

SAVE_PLOTS = True
PLOT_THEME = "light"
PLOT_QUALITY = 300
PROJECT_PATH = str(PROJECT_ROOT)


def save_table(df: pd.DataFrame, filename: str, index: bool = True) -> None:
    """Write a DataFrame to ``outputs/tables/<filename>.csv``."""
    path = os.path.join(TAB_DIR, f"{filename}.csv")
    df.to_csv(path, index=index)
    print(f"Saved table: {path}")


def save_paper_table(df: pd.DataFrame, filename: str) -> None:
    """Alias kept for notebook-derived steps that call ``save_paper_table``."""
    save_table(df, filename)


def save_paper_fig(filename: str) -> None:
    """Save the active matplotlib figure to ``outputs/figures/<filename>.png``."""
    path = os.path.join(FIG_DIR, f"{filename}.png")
    plt.savefig(path, dpi=PLOT_QUALITY, bbox_inches="tight", facecolor="white")
    print(f"Saved figure: {path}")


# RiskLabAI publication theme (overrides rcParams above for saved figures).
pub_plots.setup_publication_style(
    theme=PLOT_THEME,
    quality=PLOT_QUALITY,
    save_plots=SAVE_PLOTS,
    save_dir=FIG_DIR,
)

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

print("Libraries loaded and Output Directories setup successfully.")
