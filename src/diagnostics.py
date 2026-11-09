"""Normality diagnostics for the mixed models (Methods: Q-Q check; Discussion: limitations).

Prints skewness, excess kurtosis and Shapiro-Wilk for the raw and log durations,
the residuals of Models I3 and D1 and the observation-level random intercepts of I3,
and saves results/model_diagnostics.png (histogram, normal Q-Q plot, residuals vs fitted).

Usage: python src/diagnostics.py [--data XLSX] [--sheet apple] [--outdir results]
"""
import argparse
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import analysis_durations as da
import analysis_intervals as ia

warnings.simplefilter("ignore")


def describe(x, name):
    x = np.asarray(x)
    sw = stats.shapiro(x)
    print(f"{name:28s} n={len(x):5d}  skew={stats.skew(x):+.2f}  "
          f"excess kurtosis={stats.kurtosis(x):+.2f}  Shapiro W={sw.statistic:.3f} p={sw.pvalue:.2g}")


def main():
    os.chdir(Path(__file__).resolve().parents[1])   # paths relative to the repository folder
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, default=Path("data/apple_flower_visits.xlsx"))
    ap.add_argument("--sheet", default="apple")
    ap.add_argument("--outdir", type=Path, default=Path("results"))
    a = ap.parse_args()
    a.outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_excel(a.data, sheet_name=a.sheet)

    d = ia.prepare(df, "honeybee")
    cov = "Temp_z + solar_z + wind_z + hour_z + rank_z + C(sunshade) + log_prev_dur"
    i3 = ia.fit(f"log_dur ~ log_int_within + log_int_between + {cov}", d)
    dp = da.prepare_pairs(df, "honeybee")
    d1 = smf.mixedlm("log_dur ~ log_prev", dp, groups=dp["observation"]).fit(reml=True)

    print("Raw response")
    describe(d["duration"], "duration (s)")
    describe(d["log_dur"], "log(duration)")
    print("\nResiduals")
    describe(i3.resid, "I3 residuals")
    describe(d1.resid, "D1 residuals")
    print("\nRandom effects (BLUPs), I3")
    re_obs = pd.Series({k: v.iloc[0] for k, v in i3.random_effects.items()}).sort_values()
    describe(re_obs, "observation intercepts")
    print("lowest observation intercepts:\n" + re_obs.head(3).round(2).to_string())

    fig, ax = plt.subplots(1, 3, figsize=(13, 4), facecolor="#fcfcfb")
    for x in ax:
        x.set_facecolor("#fcfcfb")
        for s in ("top", "right"):
            x.spines[s].set_visible(False)
    ax[0].hist(d["log_dur"], bins=40, color="#2a78d6", alpha=0.8, edgecolor="#fcfcfb")
    ax[0].set(title="log(duration), honeybee", xlabel="log(duration, s)", ylabel="records")
    stats.probplot(i3.resid, dist="norm", plot=ax[1])
    ax[1].get_lines()[0].set(markerfacecolor="#2a78d6", markeredgecolor="none", markersize=3, alpha=0.6)
    ax[1].get_lines()[1].set(color="#52514e")
    ax[1].set(title="Normal Q–Q, residuals of Model I3")
    ax[2].scatter(i3.fittedvalues, i3.resid, s=6, color="#2a78d6", alpha=0.4)
    ax[2].axhline(0, color="#52514e", lw=1, ls="--")
    ax[2].set(title="Residuals vs fitted, Model I3", xlabel="fitted log(duration)", ylabel="residual")
    fig.tight_layout()
    fig.savefig(a.outdir / "model_diagnostics.png", dpi=150)
    print(f"-> {a.outdir / 'model_diagnostics.png'}")


if __name__ == "__main__":
    main()
