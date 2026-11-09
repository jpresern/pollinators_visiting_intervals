"""Reproduce all tables and figures of the manuscript in one run.

Reads data/apple_flower_visits.xlsx (sheet "apple") and writes to results/:
    table2_pollinators.csv           Table 2  records and visit bouts per taxon
    table3_previous_duration.csv     Table 3  Models D1-D3, Spearman, permutation test
    table4_previous_interval.csv     Table 4  Models I1, I2, I4
    table5_variance_components.csv   Table 5  random-effect variances, all models
    figure1_model_I3_effects.csv     Figure 1 effects in Model I3
    honeybee_interval_analysis.png   Figure 1 (right panel) + duration vs interval
    honeybee_duration_analysis.png   previous vs current duration, permutation null
    results_summary.txt              full printed output

Usage: python src/run_all.py [--data XLSX] [--sheet apple] [--outdir results]
"""
import argparse
import os
import contextlib
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import analysis_durations as da
import analysis_intervals as ia
from analysis_durations import variance_components

warnings.simplefilter("ignore")

L = lambda b: 100 * (2 ** b - 1)          # log-log slope  -> % change per doubling
E = lambda b: 100 * (np.exp(b) - 1)       # other predictor -> % change per 1 SD / sun vs shade


def effect(m, term, f):
    c = m.conf_int().loc[term]
    return {"beta": m.params[term], "beta_lo": c[0], "beta_hi": c[1],
            "pct": f(m.params[term]), "pct_lo": f(c[0]), "pct_hi": f(c[1]),
            "p": m.pvalues[term]}


def run(data, sheet, out):
    df = pd.read_excel(data, sheet_name=sheet)

    print(f"=== DATASET: {data.name}, sheet '{sheet}' ===")
    print("records", len(df), "| observations", df.observation.nunique(),
          "| dates", sorted(df.date_time.dt.strftime("%Y-%m-%d").unique()))
    t2 = (df.groupby(["subject", "visit_ID"]).size().groupby("subject")
          .agg(bouts="count", median_records_per_bout="median", max_records_per_bout="max"))
    t2.insert(0, "records", df.subject.value_counts())
    t2 = t2.sort_values("records", ascending=False)
    t2.to_csv(out / "table2_pollinators.csv")
    print(t2.to_string())
    h = df[df.subject == "honeybee"]
    print(f"honeybee share {100 * len(h) / len(df):.1f}% | duration median {h.duration.median():.2f} s "
          f"IQR {h.duration.quantile(.25):.2f}-{h.duration.quantile(.75):.2f} | previous interval median "
          f"{h['previous interval'].median():.0f} s IQR {h['previous interval'].quantile(.25):.0f}-"
          f"{h['previous interval'].quantile(.75):.0f} | sun/shade {h.sunshade.value_counts().to_dict()}")
    for c in ["Temp", "solar", "wind"]:
        print(f"{c}: min {df[c].min():.1f}, median {df[c].median():.1f}, max {df[c].max():.1f}")

    print("\n=== PREVIOUS DURATION (Models D1-D3) ===")
    r = da.analyse(df, "honeybee", out / "honeybee_duration_analysis.png")
    rows = [{"model": "Spearman", "rho": r["spearman_rho"], "p": r["spearman_p"]}]
    for k in ["m1", "m2", "m3"]:
        rows.append({"model": f"D{k[1]}", "n": int(r[k + "_model"].nobs),
                     **effect(r[k + "_model"], "log_prev", L)})
    rows.append({"model": "permutation", "rho": r["perm_rho"],
                 "rho_null_mean": r["perm_null_mean"], "p": r["perm_p"]})
    for t in ["C(sunshade)[T.sun]", "Temp_z", "solar_z", "wind_z"]:
        rows.append({"model": f"D2 {t}", **effect(r["m2_model"], t, E)})
    pd.DataFrame(rows).to_csv(out / "table3_previous_duration.csv", index=False)

    print("\n=== PREVIOUS INTERVAL (Models I1-I4) ===")
    res = ia.analyse(df, "honeybee", out / "honeybee_interval_analysis.png")
    rows = [{"model": "I1", "term": "log_int", **effect(res["m0"], "log_int", L)},
            {"model": "I2", "term": "log_int", **effect(res["m1"], "log_int", L)}]
    for q in ["Q2", "Q3", "Q4 longest"]:
        rows.append({"model": "I4", "term": f"{q} vs Q1", **effect(res["m3"], f"C(int_bin)[T.{q}]", E)})
    for row in rows:
        row["n"] = int(res["m0"].nobs)
    pd.DataFrame(rows).to_csv(out / "table4_previous_interval.csv", index=False)

    terms = [("log_int_within", L), ("log_int_between", L), ("log_prev_dur", L),
             ("C(sunshade)[T.sun]", E), ("Temp_z", E), ("solar_z", E), ("wind_z", E),
             ("hour_z", E), ("rank_z", E)]
    pd.DataFrame([{"term": t, **effect(res["m2"], t, f)} for t, f in terms]) \
        .to_csv(out / "figure1_model_I3_effects.csv", index=False)

    models = {"D1": r["m1_model"], "D2": r["m2_model"], "D3": r["m3_model"],
              "I1": res["m0"], "I2": res["m1"], "I3": res["m2"], "I4": res["m3"]}
    vc = pd.DataFrame({k: variance_components(m) for k, m in models.items()}).T
    vc = vc.rename(columns={"visit": "bout"})[["observation", "flower", "bout", "residual"]]
    vc["n"] = [int(m.nobs) for m in models.values()]
    vc.to_csv(out / "table5_variance_components.csv", index_label="model")

    print("\n=== MANUSCRIPT SUMMARY (% change, 95% CI, p) ===")
    for k in ["m1", "m2", "m3"]:
        e = effect(r[k + "_model"], "log_prev", L)
        print(f"D{k[1]} previous duration x2: {e['pct']:+.1f}% [{e['pct_lo']:+.1f}, {e['pct_hi']:+.1f}], p = {e['p']:.3g}")
    for row in rows:
        print(f"{row['model']} {row['term']}: {row['pct']:+.1f}% [{row['pct_lo']:+.1f}, {row['pct_hi']:+.1f}], p = {row['p']:.3g}")
    for t, f in terms:
        e = effect(res["m2"], t, f)
        print(f"I3 {t}: {e['pct']:+.1f}% [{e['pct_lo']:+.1f}, {e['pct_hi']:+.1f}], p = {e['p']:.3g}")
    print("\nVariance components (log scale):")
    print(vc.round(3).to_string())


def main():
    os.chdir(Path(__file__).resolve().parents[1])   # paths relative to the repository folder
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, default=Path("data/apple_flower_visits.xlsx"))
    ap.add_argument("--sheet", default="apple")
    ap.add_argument("--outdir", type=Path, default=Path("results"))
    a = ap.parse_args()
    a.outdir.mkdir(parents=True, exist_ok=True)

    class Tee:
        def __init__(self, *s): self.s = s
        def write(self, x): [f.write(x) for f in self.s]
        def flush(self): [f.flush() for f in self.s]

    with open(a.outdir / "results_summary.txt", "w") as log, \
            contextlib.redirect_stdout(Tee(sys.stdout, log)):
        print("Python", sys.version.split()[0], "| pandas", pd.__version__, "| numpy", np.__version__,
              "| statsmodels", __import__("statsmodels").__version__,
              "| scipy", __import__("scipy").__version__, "\n")
        run(a.data, a.sheet, a.outdir)


if __name__ == "__main__":
    main()
