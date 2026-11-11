"""Does the previous duration affect the current duration within a visit bout?

Models D1-D3, Spearman correlation and the within-bout permutation test
(manuscript Table 3). A "visit" in the code is a visit bout (visit_ID).
Also holds the helpers shared by the other scripts.

Usage: python src/analysis_durations.py [--data XLSX] [--sheet apple] [--subject honeybee]
"""
__author__ = "jpresern"

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")


MIN_PAIRS = 30

# model formulas (random effects: D1, D2 observation; D3 observation / bout)
D_FORMULA = {"D1": "log_dur ~ log_prev",
             "D2": "log_dur ~ log_prev + Temp_z + solar_z + wind_z + C(sunshade)",
             "D3": "log_dur ~ log_prev"}

# plot colours
INK, MUTED, GRID, SURF, BLUE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#2a78d6"


# ---------- shared helpers ----------

def pct_doubling(b):
    """log-log slope -> % change in duration per doubling of the predictor"""
    return 100 * (2 ** b - 1)


def pct_unit(b):
    """coefficient of an untransformed predictor -> % change per unit (1 SD, sun vs shade)"""
    return 100 * (np.exp(b) - 1)


def check_positive(d, cols):
    """Stop (rather than silently drop records) if a value cannot be logged."""
    for c in cols:
        bad = d[d[c] <= 0]
        if len(bad):
            raise ValueError(f"{len(bad)} record(s) with {c!r} <= 0 cannot be log-transformed:\n"
                             + bad[["observation", "visit_ID", "timestamp", c]].to_string())


def variance_components(m):
    """Random-effect variances of a fitted MixedLM: observation, nested components, residual."""
    out = {}
    if m.cov_re.size:
        out["observation"] = float(m.cov_re.iloc[0, 0])
    if m.model.exog_vc is not None:
        out.update({n: float(v) for n, v in zip(m.model.exog_vc.names, m.vcomp)})
    out["residual"] = float(m.scale)
    return out


def format_vc(m):
    return ", ".join(f"{k} {v:.3f}" for k, v in variance_components(m).items())


def style(ax):
    """House style of all panels: light background, no top/right frame, muted ticks."""
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED)


def parse_args(doc, subject=False):
    """Common command line: --data, --sheet, --outdir (and --subject). Paths are relative to
    the repository folder. Returns the arguments and the data sheet."""
    os.chdir(Path(__file__).resolve().parents[1])
    ap = argparse.ArgumentParser(description=doc.splitlines()[0])
    ap.add_argument("--data", type=Path, default=Path("data/apple_flower_visits.xlsx"))
    ap.add_argument("--sheet", default="apple")
    ap.add_argument("--outdir", type=Path, default=Path("results"))
    if subject:
        ap.add_argument("--subject", default="honeybee")
    a = ap.parse_args()
    a.outdir.mkdir(parents=True, exist_ok=True)
    return a, pd.read_excel(a.data, sheet_name=a.sheet)


# ---------- previous duration ----------

def prepare_pairs(df, subject):
    """Record pairs (previous, current) within a bout. Only the first record of a bout,
    which has no previous record, lacks the predictor; nothing else is excluded."""
    d = df[df["subject"] == subject].copy()
    d = d.dropna(subset=["duration", "duration_n-1"])
    check_positive(d, ["duration", "duration_n-1"])
    d["log_dur"] = np.log(d["duration"])
    d["log_prev"] = np.log(d["duration_n-1"])
    return d


def fit_d_models(d):
    """Models D1-D3. D2 uses the records with weather data, z-standardised."""
    dw = d.dropna(subset=["Temp", "solar", "wind"]).copy()
    for c in ("Temp", "solar", "wind"):
        dw[c + "_z"] = (dw[c] - dw[c].mean()) / dw[c].std()
    return {
        "D1": smf.mixedlm(D_FORMULA["D1"], d, groups=d["observation"]).fit(reml=True),
        "D2": smf.mixedlm(D_FORMULA["D2"], dw, groups=dw["observation"]).fit(reml=True),
        "D3": smf.mixedlm(D_FORMULA["D3"], d, groups=d["observation"], re_formula="1",
                          vc_formula={"visit": "0 + C(visit_ID)"}).fit(reml=True),
    }


def within_bout_permutation(records, n_perm=2000, seed=1):
    """Spearman rho of (previous, current) duration vs a null where durations are shuffled
    within each bout. Bout means are kept, so only the ORDER inside a bout is tested."""
    rng = np.random.default_rng(seed)
    bouts = [g["duration"].to_numpy() for _, g in records.groupby("visit_ID") if len(g) >= 2]

    def rho(seqs):
        prev = np.concatenate([s[:-1] for s in seqs])
        cur = np.concatenate([s[1:] for s in seqs])
        return stats.spearmanr(prev, cur).statistic

    obs = rho(bouts)
    null = np.array([rho([rng.permutation(s) for s in bouts]) for _ in range(n_perm)])
    p = (np.sum(np.abs(null - null.mean()) >= abs(obs - null.mean())) + 1) / (n_perm + 1)
    return obs, null, p


def analyse(df, subject="honeybee", plot_path=None):
    """Prints the results; returns the data, the three models and the two tests."""
    d = prepare_pairs(df, subject)
    print(f"\n=== {subject}: {len(d)} pairs from {d['visit_ID'].nunique()} visits in "
          f"{d['observation'].nunique()} observations ===")
    if len(d) < MIN_PAIRS:
        print(f"Too few pairs (< {MIN_PAIRS}) for a meaningful analysis.")
        return None

    spearman = stats.spearmanr(d["duration_n-1"], d["duration"])
    print(f"1) Spearman rho = {spearman.statistic:.3f}, p = {spearman.pvalue:.3g}  (ignores clustering)")

    models = fit_d_models(d)
    labels = {"D1": "random intercept: observation", "D2": "+ Temp, solar, wind, sun/shade",
              "D3": "random intercepts: observation / visit"}
    for name, m in models.items():
        b, p = m.params["log_prev"], m.pvalues["log_prev"]
        lo, hi = m.conf_int().loc["log_prev"]
        print(f"2) LMM {labels[name]:40s} slope = {b:.3f} [{lo:.3f}, {hi:.3f}], p = {p:.3g}"
              f"   -> 2x longer previous = {pct_doubling(b):+.1f}% current")
        print(f"   variances: {format_vc(m)}")

    records = df[(df["subject"] == subject) & df["visit_ID"].isin(d["visit_ID"])]
    perm_rho, null, perm_p = within_bout_permutation(records.sort_values(["visit_ID", "timestamp"]))
    print(f"3) Within-visit order test: rho = {perm_rho:.3f} vs shuffled {null.mean():.3f} "
          f"(95% {np.percentile(null, 2.5):.3f}..{np.percentile(null, 97.5):.3f}), p = {perm_p:.3g}")

    if plot_path:
        plot(d, models["D1"].params["log_prev"], perm_rho, null, perm_p, subject, plot_path)
    return {"data": d, **models, "spearman": spearman,
            "perm_rho": perm_rho, "perm_null_mean": null.mean(), "perm_p": perm_p}


def plot(d, slope, perm_rho, null, perm_p, subject, path):

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5.2), facecolor=SURF,
                                 gridspec_kw={"width_ratios": [1.15, 1]})
    for ax in (a1, a2):
        style(ax)
        ax.grid(color=GRID, lw=0.8, zorder=0)

    # left: scatter (jittered within one 0.025 s step) + D1 fit
    rng = np.random.default_rng(0)
    jx = d["duration_n-1"] + rng.uniform(-0.0125, 0.0125, len(d))
    jy = d["duration"] + rng.uniform(-0.0125, 0.0125, len(d))
    a1.scatter(jx, jy, s=16, color=BLUE, alpha=0.45, edgecolor=SURF, lw=0.5, zorder=3)
    xs = np.geomspace(d["duration_n-1"].min(), d["duration_n-1"].max(), 50)
    a = np.mean(d["log_dur"] - slope * d["log_prev"])
    a1.plot(xs, np.exp(a + slope * np.log(xs)), color=INK, lw=2, zorder=4)
    a1.text(xs[-1], np.exp(a + slope * np.log(xs[-1])) * 1.15, f"model fit\nslope {slope:.2f}",
            color=INK, fontsize=9, ha="right", va="bottom")
    a1.set(xscale="log", yscale="log")
    a1.set_xlabel("Previous duration (s)", color=INK)
    a1.set_ylabel("Current duration (s)", color=INK)
    a1.set_title(f"Current vs previous duration, n = {len(d)}", color=INK, loc="left", fontsize=11)

    # right: permutation null
    a2.hist(null, bins=40, color="#a9c8ef", edgecolor=SURF, lw=0.8, zorder=3)
    a2.axvline(perm_rho, color=INK, lw=2, zorder=4)
    a2.text(perm_rho, a2.get_ylim()[1] * 0.95, f"  observed ρ = {perm_rho:.3f}",
            color=INK, fontsize=9, va="top",
            ha="left" if perm_rho < np.percentile(null, 80) else "right")
    a2.set_xlabel("Spearman ρ (previous vs current)", color=INK)
    a2.set_ylabel("Number of shuffles", color=INK)
    a2.set_title(f"Order within visits shuffled 2000×, p = {perm_p:.3f}",
                 color=INK, loc="left", fontsize=11)

    fig.suptitle(f"{subject.capitalize()}: does the previous duration affect the current one?",
                 color=INK, x=0.01, ha="left", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"-> {path}")


if __name__ == "__main__":
    args, df = parse_args(__doc__, subject=True)
    analyse(df, args.subject, args.outdir / f"{args.subject.replace(' ', '_')}_duration_analysis.png")
