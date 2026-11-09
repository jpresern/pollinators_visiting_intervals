"""Does the previous interval affect the current duration within a visit bout?

Models I1-I4 (manuscript Tables 4-5) and Figure 1 (right panel: Model I3).
A "visit" in the code is a visit bout (visit_ID).

Usage: python src/analysis_intervals.py [--data XLSX] [--sheet apple] [--subject honeybee]
"""

__author__ = "jpresern"

import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from analysis_durations import (MIN_PAIRS, INK, MUTED, GRID, SURF, BLUE, check_positive,
                                format_vc, parse_args, pct_doubling, pct_unit, style)

COVARS = ["Temp", "solar", "wind", "hour", "rank"]
COV = "Temp_z + solar_z + wind_z + hour_z + rank_z + C(sunshade) + log_prev_dur"
QUARTILES = ["Q1 shortest", "Q2", "Q3", "Q4 longest"]

# model formulas; all with random intercepts for observation / flower / bout
I_FORMULA = {"I1": "log_dur ~ log_int",
             "I2": f"log_dur ~ log_int + {COV}",
             "I3": f"log_dur ~ log_int_within + log_int_between + {COV}",
             "I4": f"log_dur ~ C(int_bin) + {COV}"}


def prepare(df, subject):
    """Records of one taxon that have a previous record in the same bout. No records are
    excluded other than first records of a bout and, in the models, missing covariates."""
    d = df[df["subject"] == subject].copy()
    d = d.sort_values(["visit_ID", "timestamp"])
    d["rank"] = d.groupby("visit_ID").cumcount() + 1          # position in bout
    d["hour"] = d["date_time"].dt.hour + d["date_time"].dt.minute / 60
    d = d.dropna(subset=["previous interval", "duration_n-1"])
    check_positive(d, ["duration", "previous interval", "duration_n-1"])
    d["log_dur"] = np.log(d["duration"])
    d["log_int"] = np.log(d["previous interval"])
    d["log_prev_dur"] = np.log(d["duration_n-1"])
    # within-/between-bout split of the interval
    d["log_int_between"] = d.groupby("visit_ID")["log_int"].transform("mean")
    d["log_int_within"] = d["log_int"] - d["log_int_between"]
    d["flower_ID"] = d["observation"] + "_" + d["observed"]
    d = d.dropna(subset=COVARS + ["sunshade"])
    for c in COVARS:
        d[c + "_z"] = (d[c] - d[c].mean()) / d[c].std()
    return d


def fit(formula, d):
    """LMM with random intercepts for observation, flower within observation and bout within
    flower. re_formula="1" is needed: with vc_formula alone statsmodels drops the observation intercept."""
    vc = {"flower": "0 + C(flower_ID)", "visit": "0 + C(visit_ID)"}
    return smf.mixedlm(formula, d, groups=d["observation"], re_formula="1",
                       vc_formula=vc).fit(reml=True)


def effect_row(m, term, label):
    b, p = m.params[term], m.pvalues[term]
    lo, hi = m.conf_int().loc[term]
    print(f"  {label:44s} {b:7.3f} [{lo:6.3f}, {hi:6.3f}]  p = {p:.3g}"
          f"   2x longer interval -> {pct_doubling(b):+.1f}% duration")


def analyse(df, subject="honeybee", plot_path=None):
    """Prints the results; returns the data and Models I1-I4."""
    warnings.simplefilter("ignore")
    d = prepare(df, subject)
    print(f"\n=== {subject}: {len(d)} records from {d['visit_ID'].nunique()} visits, "
          f"{d['observation'].nunique()} observations ===")
    if len(d) < MIN_PAIRS:
        print(f"Too few records (< {MIN_PAIRS}) for a meaningful analysis.")
        return None

    r = stats.spearmanr(d["previous interval"], d["duration"])
    print(f"Spearman rho (no adjustment) = {r.statistic:.3f}, p = {r.pvalue:.3g}")
    print("Correlation of log(previous interval) with covariates (Spearman):")
    print("  " + ", ".join(f"{c} {stats.spearmanr(d['log_int'], d[c]).statistic:+.2f}"
                           for c in COVARS + ["log_prev_dur"]))

    d["int_bin"] = pd.qcut(d["previous interval"], 4, labels=QUARTILES)   # for I4
    m = {name: fit(formula, d) for name, formula in I_FORMULA.items()}

    print("\nEffect of log(previous interval) on log(duration)  [slope, 95% CI]")
    effect_row(m["I1"], "log_int", "unadjusted")
    effect_row(m["I2"], "log_int", "adjusted for all covariates")
    effect_row(m["I3"], "log_int_within", "adjusted, within visit (same bee)")
    effect_row(m["I3"], "log_int_between", "adjusted, between visits")

    print("\nInterval quartiles (adjusted, relative to Q1):")
    edges = pd.qcut(d["previous interval"], 4).cat.categories
    for i, lab in enumerate(QUARTILES):
        rng = f"{edges[i].left:7.1f} – {edges[i].right:7.1f} s"
        if i == 0:
            print(f"  {lab:12s} {rng}  reference")
            continue
        t = f"C(int_bin)[T.{lab}]"
        (lo, hi), p = m["I4"].conf_int().loc[t], m["I4"].pvalues[t]
        print(f"  {lab:12s} {rng}  {pct_unit(m['I4'].params[t]):+6.1f}% "
              f"[{pct_unit(lo):+.1f}, {pct_unit(hi):+.1f}]  p = {p:.3g}")

    print("\nRandom-effect variances:")
    for name, model in m.items():
        print(f"  {name}: {format_vc(model)}")

    print("\nFull adjusted model (m2) – fixed effects:")
    print(m["I3"].summary().tables[1].round(3).to_string())

    if plot_path:
        plot(d, m["I3"], subject, plot_path)
    return {"data": d, **m}


# Figure 1 terms: (term, label, interval term?)
FIGURE_TERMS = [("log_int_within", "Previous interval ×2, within bout", True),
                ("log_int_between", "Previous interval ×2, between bouts", True),
                ("C(sunshade)[T.sun]", "Sun vs shade", False),
                ("log_prev_dur", "Previous duration ×2", False),
                ("Temp_z", "Temperature +1 SD", False),
                ("solar_z", "Solar radiation +1 SD", False),
                ("wind_z", "Wind speed +1 SD", False),
                ("hour_z", "Time of day +1 SD", False),
                ("rank_z", "Position in bout +1 SD", False)]


def plot(d, m, subject, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 5.4), facecolor=SURF,
                                 gridspec_kw={"width_ratios": [1.1, 1]})
    for ax in (a1, a2):
        style(ax)

    # left: scatter + medians per interval bin
    rng = np.random.default_rng(0)
    a1.grid(color=GRID, lw=0.8, zorder=0)
    a1.scatter(d["previous interval"], d["duration"] + rng.uniform(-0.0125, 0.0125, len(d)),
               s=14, color=BLUE, alpha=0.35, edgecolor=SURF, lw=0.4, zorder=3)
    bins = np.geomspace(d["previous interval"].min(), d["previous interval"].max(), 11)
    g = d.groupby(pd.cut(d["previous interval"], bins), observed=True)
    mid = np.sqrt(bins[:-1] * bins[1:])[: len(g)]
    med = g["duration"].median().to_numpy()
    keep = g.size().to_numpy() >= 10
    a1.plot(mid[keep], med[keep], color=INK, lw=2, marker="o", ms=6, zorder=4)
    a1.text(mid[keep][-1], med[keep][-1] * 1.25, "median\nper bin", color=INK,
            fontsize=9, ha="right", va="bottom")
    a1.set(xscale="log", yscale="log")
    a1.set_xlabel("Previous interval (s)", color=INK)
    a1.set_ylabel("Current duration (s)", color=INK)
    a1.set_title(f"Duration vs previous interval, n = {len(d)}", color=INK, loc="left", fontsize=11)

    # right: effects in Model I3 as % change; blue = interval terms, dark grey filled =
    # other terms with p < 0.05, light grey open = not significant
    DARK, LIGHT = "#52514e", "#c3c2b7"
    ci = m.conf_int()
    ys = np.arange(len(FIGURE_TERMS))[::-1]
    for y, (t, lab, focus) in zip(ys, FIGURE_TERMS):
        f = pct_doubling if "log_" in t else pct_unit
        b, lo, hi = f(m.params[t]), f(ci.loc[t, 0]), f(ci.loc[t, 1])
        sig = m.pvalues[t] < 0.05
        col = BLUE if focus else (DARK if sig else LIGHT)
        a2.plot([lo, hi], [y, y], color=col, lw=2, zorder=3)
        a2.plot(b, y, "o", ms=8, color=col if sig else SURF, markeredgecolor=col, mew=1.5, zorder=4)
        if sig:   # value label: right of the CI, or above the dot for negative effects
            if b < 0:
                a2.text(b, y + 0.28, f"{b:+.1f}%", color=DARK, fontsize=9, ha="center", va="bottom")
            else:
                a2.text(hi + 1, y, f"{b:+.1f}%", color=INK if focus else DARK, fontsize=9, va="center")
    a2.axvline(0, color=MUTED, lw=1, ls="--", zorder=2)
    a2.set_yticks(ys, [lab for _, lab, _ in FIGURE_TERMS], color=INK)
    for tick, (_, _, focus) in zip(a2.get_yticklabels(), FIGURE_TERMS):
        tick.set_fontweight("bold" if focus else "normal")
    a2.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    a2.set_xlabel("Change in current duration (%), 95% CI", color=INK)
    a2.set_title("Adjusted model I3\nblue: interval terms · filled: p < 0.05 · open: n.s.",
                 color=INK, loc="left", fontsize=10)

    fig.suptitle(f"{subject.capitalize()}: does the previous interval affect the current duration?",
                 color=INK, x=0.01, ha="left", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"-> {path}")


if __name__ == "__main__":
    args, df = parse_args(__doc__, subject=True)
    analyse(df, args.subject, args.outdir / f"{args.subject.replace(' ', '_')}_interval_analysis.png")
