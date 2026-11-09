"""Does the previous duration affect the current duration within a visit bout?

Models D1-D3, Spearman correlation and the within-bout permutation test
(manuscript Table 3). A "visit" in the code is a visit bout (visit_ID).

Usage: python src/analysis_durations.py [--data XLSX] [--sheet apple] [--subject honeybee]
"""
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

MIN_PAIRS = 30


def check_positive(d, cols):
    """Stop (rather than silently drop records) if a value cannot be logged."""
    for c in cols:
        bad = d[d[c] <= 0]
        if len(bad):
            raise ValueError(f"{len(bad)} record(s) with {c!r} <= 0 cannot be log-transformed:\n"
                             + bad[["observation", "visit_ID", "timestamp", c]].to_string())


def variance_components(m):
    """Random-effect variances of a fitted MixedLM: observation (group
    intercept), any nested components (flower, visit) and the residual."""
    out = {}
    if m.cov_re.size:
        out["observation"] = float(m.cov_re.iloc[0, 0])
    if m.model.exog_vc is not None:
        out.update({n: float(v) for n, v in zip(m.model.exog_vc.names, m.vcomp)})
    out["residual"] = float(m.scale)
    return out


def format_vc(m):
    return ", ".join(f"{k} {v:.3f}" for k, v in variance_components(m).items())


def prepare_pairs(df, subject):
    """Record pairs (previous, current) within a visit for one pollinator.

    Only records without a previous record in the same visit (first record of
    a visit) lack the predictor; no other records are excluded.
    """
    d = df[df["subject"] == subject].copy()
    d = d.dropna(subset=["duration", "duration_n-1"])
    check_positive(d, ["duration", "duration_n-1"])
    d["log_dur"] = np.log(d["duration"])
    d["log_prev"] = np.log(d["duration_n-1"])
    return d


def within_visit_permutation(d, n_perm=2000, seed=1):
    """Spearman rho of (prev, current) vs a null where durations are shuffled
    within each visit. Keeps visit-level differences (one bee being slower
    than another), so only the ORDER inside a visit is tested."""
    rng = np.random.default_rng(seed)
    full = d.attrs["visit_records"]          # all records of the visits, in time order
    groups = [g["duration"].to_numpy() for _, g in full.groupby("visit_ID") if len(g) >= 2]

    def rho(seqs):
        prev = np.concatenate([s[:-1] for s in seqs])
        cur = np.concatenate([s[1:] for s in seqs])
        return stats.spearmanr(prev, cur).statistic

    obs = rho(groups)
    null = np.array([rho([rng.permutation(s) for s in groups]) for _ in range(n_perm)])
    p = (np.sum(np.abs(null - null.mean()) >= abs(obs - null.mean())) + 1) / (n_perm + 1)
    return obs, null, p


def analyse(df, subject="honeybee", plot_path=None):
    d = prepare_pairs(df, subject)
    n, n_vis, n_obs = len(d), d["visit_ID"].nunique(), d["observation"].nunique()
    print(f"\n=== {subject}: {n} pairs from {n_vis} visits in {n_obs} observations "
          "===")
    if n < MIN_PAIRS:
        print(f"Too few pairs (< {MIN_PAIRS}) for a meaningful analysis.")
        return None

    res = {"subject": subject, "n_pairs": n, "n_visits": n_vis, "n_observations": n_obs}

    # 1. Simple association
    r = stats.spearmanr(d["duration_n-1"], d["duration"])
    res["spearman_rho"], res["spearman_p"] = r.statistic, r.pvalue
    print(f"1) Spearman rho = {r.statistic:.3f}, p = {r.pvalue:.3g}  (ignores clustering)")

    # 2. Mixed models on log durations; slope = elasticity
    #    (1% longer previous duration -> slope % longer current duration)
    warnings.simplefilter("ignore")
    m1 = smf.mixedlm("log_dur ~ log_prev", d, groups=d["observation"]).fit(reml=True)
    dd = d.dropna(subset=["Temp", "solar", "wind"]).copy()
    for c in ("Temp", "solar", "wind"):
        dd[c + "_z"] = (dd[c] - dd[c].mean()) / dd[c].std()
    m2 = smf.mixedlm("log_dur ~ log_prev + Temp_z + solar_z + wind_z + C(sunshade)",
                     dd, groups=dd["observation"]).fit(reml=True)
    # random intercept per visit nested in observation: compares records of the same bee
    m3 = smf.mixedlm("log_dur ~ log_prev", d, groups=d["observation"], re_formula="1",
                     vc_formula={"visit": "0 + C(visit_ID)"}).fit(reml=True)
    for name, m, label in [("m1", m1, "random intercept: observation"),
                           ("m2", m2, "+ Temp, solar, wind, sun/shade"),
                           ("m3", m3, "random intercepts: observation / visit")]:
        b, se, p = m.params["log_prev"], m.bse["log_prev"], m.pvalues["log_prev"]
        lo, hi = m.conf_int().loc["log_prev"]
        res[f"{name}_slope"], res[f"{name}_ci"], res[f"{name}_p"] = b, (lo, hi), p
        print(f"2) LMM {label:40s} slope = {b:.3f} [{lo:.3f}, {hi:.3f}], p = {p:.3g}"
              f"   -> 2x longer previous = {100 * (2 ** b - 1):+.1f}% current")
        print(f"   variances: {format_vc(m)}")
    res["m1_model"], res["m2_model"], res["m3_model"] = m1, m2, m3

    # 3. Within-visit permutation test
    full = df[df["subject"] == subject]
    full = full[full["visit_ID"].isin(d["visit_ID"])].sort_values(["visit_ID", "timestamp"])
    d.attrs["visit_records"] = full
    obs_rho, null, p = within_visit_permutation(d)
    res["perm_rho"], res["perm_null_mean"], res["perm_p"] = obs_rho, null.mean(), p
    print(f"3) Within-visit order test: rho = {obs_rho:.3f} vs shuffled {null.mean():.3f} "
          f"(95% {np.percentile(null, 2.5):.3f}..{np.percentile(null, 97.5):.3f}), p = {p:.3g}")

    if plot_path:
        plot(d, res, null, plot_path)
    return res


def plot(d, res, null, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, MUTED, GRID, SURF, BLUE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#2a78d6"

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5.2), facecolor=SURF,
                                 gridspec_kw={"width_ratios": [1.15, 1]})
    for ax in (a1, a2):
        ax.set_facecolor(SURF)
        ax.grid(color=GRID, lw=0.8, zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=MUTED)

    # left: scatter (jittered within one 0.025 s step) + LMM fit
    rng = np.random.default_rng(0)
    jx = d["duration_n-1"] + rng.uniform(-0.0125, 0.0125, len(d))
    jy = d["duration"] + rng.uniform(-0.0125, 0.0125, len(d))
    a1.scatter(jx, jy, s=16, color=BLUE, alpha=0.45, edgecolor=SURF, lw=0.5, zorder=3)
    b = res["m1_slope"]
    xs = np.geomspace(d["duration_n-1"].min(), d["duration_n-1"].max(), 50)
    a = np.mean(d["log_dur"] - b * d["log_prev"])
    a1.plot(xs, np.exp(a + b * np.log(xs)), color=INK, lw=2, zorder=4)
    a1.text(xs[-1], np.exp(a + b * np.log(xs[-1])) * 1.15, f"model fit\nslope {b:.2f}",
            color=INK, fontsize=9, ha="right", va="bottom")
    a1.set(xscale="log", yscale="log")
    a1.set_xlabel("Previous duration (s)", color=INK)
    a1.set_ylabel("Current duration (s)", color=INK)
    a1.set_title(f"Current vs previous duration, n = {res['n_pairs']}",
                 color=INK, loc="left", fontsize=11)

    # right: permutation null
    a2.hist(null, bins=40, color="#a9c8ef", edgecolor=SURF, lw=0.8, zorder=3)
    a2.axvline(res["perm_rho"], color=INK, lw=2, zorder=4)
    a2.text(res["perm_rho"], a2.get_ylim()[1] * 0.95, f"  observed ρ = {res['perm_rho']:.3f}",
            color=INK, fontsize=9, va="top",
            ha="left" if res["perm_rho"] < np.percentile(null, 80) else "right")
    a2.set_xlabel("Spearman ρ (previous vs current)", color=INK)
    a2.set_ylabel("Number of shuffles", color=INK)
    a2.set_title(f"Order within visits shuffled 2000×, p = {res['perm_p']:.3f}",
                 color=INK, loc="left", fontsize=11)

    fig.suptitle(f"{res['subject'].capitalize()}: does the previous duration affect the current one?",
                 color=INK, x=0.01, ha="left", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"-> {path}")


def main():
    import argparse
    import os
    from pathlib import Path
    os.chdir(Path(__file__).resolve().parents[1])   # paths relative to the repository folder
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=Path("data/apple_flower_visits.xlsx"))
    ap.add_argument("--sheet", default="apple")
    ap.add_argument("--subject", default="honeybee")
    ap.add_argument("--outdir", default=Path("results"))
    a = ap.parse_args()
    df = pd.read_excel(a.data, sheet_name=a.sheet)
    Path(a.outdir).mkdir(parents=True, exist_ok=True)
    analyse(df, a.subject, Path(a.outdir) / f"{a.subject.replace(' ', '_')}_duration_analysis.png")


if __name__ == "__main__":
    main()
