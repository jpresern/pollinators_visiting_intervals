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
import contextlib
import sys
import warnings

import numpy as np
import pandas as pd

import analysis_durations as da
import analysis_intervals as ia
from analysis_durations import pct_doubling, pct_unit, variance_components

warnings.simplefilter("ignore")


def effect(m, term):
    """Coefficient, 95% Wald CI and p, also as % change in duration."""
    f = pct_doubling if term.startswith("log_") else pct_unit
    b, (lo, hi) = m.params[term], m.conf_int().loc[term]
    return {"beta": b, "beta_lo": lo, "beta_hi": hi,
            "pct": f(b), "pct_lo": f(lo), "pct_hi": f(hi), "p": m.pvalues[term]}


def line(label, e):
    return f"{label}: {e['pct']:+.1f}% [{e['pct_lo']:+.1f}, {e['pct_hi']:+.1f}], p = {e['p']:.3g}"


def describe_data(df, out):
    """Dataset numbers and Table 2."""
    print("records", len(df), "| observations", df.observation.nunique(),
          "| dates", sorted(df.date_time.dt.strftime("%Y-%m-%d").unique()))
    t2 = (df.groupby(["subject", "visit_ID"]).size().groupby("subject")
          .agg(bouts="count", median_records_per_bout="median", max_records_per_bout="max"))
    t2.insert(0, "records", df.subject.value_counts())
    t2 = t2.sort_values("records", ascending=False)
    t2.to_csv(out / "table2_pollinators.csv")
    print(t2.to_string())
    h = df[df.subject == "honeybee"]
    iv = h["previous interval"]
    print(f"honeybee share {100 * len(h) / len(df):.1f}% | duration median {h.duration.median():.2f} s "
          f"IQR {h.duration.quantile(.25):.2f}-{h.duration.quantile(.75):.2f} | previous interval median "
          f"{iv.median():.0f} s IQR {iv.quantile(.25):.0f}-{iv.quantile(.75):.0f} | "
          f"sun/shade {h.sunshade.value_counts().to_dict()}")
    for c in ["Temp", "solar", "wind"]:
        print(f"{c}: min {df[c].min():.1f}, median {df[c].median():.1f}, max {df[c].max():.1f}")


def run(df, data_name, sheet, out):
    print(f"=== DATASET: {data_name}, sheet '{sheet}' ===")
    describe_data(df, out)

    print("\n=== PREVIOUS DURATION (Models D1-D3) ===")
    r = da.analyse(df, "honeybee", out / "honeybee_duration_analysis.png")
    t3 = [{"model": "Spearman", "rho": r["spearman"].statistic, "p": r["spearman"].pvalue}]
    t3 += [{"model": k, "n": int(r[k].nobs), **effect(r[k], "log_prev")} for k in ["D1", "D2", "D3"]]
    t3 += [{"model": "permutation", "rho": r["perm_rho"], "rho_null_mean": r["perm_null_mean"], "p": r["perm_p"]}]
    t3 += [{"model": f"D2 {t}", **effect(r["D2"], t)} for t in ["C(sunshade)[T.sun]", "Temp_z", "solar_z", "wind_z"]]
    pd.DataFrame(t3).to_csv(out / "table3_previous_duration.csv", index=False)

    print("\n=== PREVIOUS INTERVAL (Models I1-I4) ===")
    m = ia.analyse(df, "honeybee", out / "honeybee_interval_analysis.png")
    n = int(m["I1"].nobs)
    t4 = [{"model": "I1", "term": "log_int", **effect(m["I1"], "log_int"), "n": n},
          {"model": "I2", "term": "log_int", **effect(m["I2"], "log_int"), "n": n}]
    t4 += [{"model": "I4", "term": f"{q} vs Q1", **effect(m["I4"], f"C(int_bin)[T.{q}]"), "n": n}
           for q in ia.QUARTILES[1:]]
    pd.DataFrame(t4).to_csv(out / "table4_previous_interval.csv", index=False)

    i3_terms = ["log_int_within", "log_int_between", "log_prev_dur", "C(sunshade)[T.sun]",
                "Temp_z", "solar_z", "wind_z", "hour_z", "rank_z"]
    fig1 = [{"term": t, **effect(m["I3"], t)} for t in i3_terms]
    pd.DataFrame(fig1).to_csv(out / "figure1_model_I3_effects.csv", index=False)

    models = {k: r[k] for k in ["D1", "D2", "D3"]} | {k: m[k] for k in ["I1", "I2", "I3", "I4"]}
    vc = pd.DataFrame({k: variance_components(x) for k, x in models.items()}).T
    vc = vc.rename(columns={"visit": "bout"})[["observation", "flower", "bout", "residual"]]
    vc["n"] = [int(x.nobs) for x in models.values()]
    vc.to_csv(out / "table5_variance_components.csv", index_label="model")

    print("\n=== MANUSCRIPT SUMMARY (% change, 95% CI, p) ===")
    for k in ["D1", "D2", "D3"]:
        print(line(f"{k} previous duration x2", effect(r[k], "log_prev")))
    for row in t4:
        print(line(f"{row['model']} {row['term']}", row))
    for row in fig1:
        print(line(f"I3 {row['term']}", row))
    print("\nVariance components (log scale):")
    print(vc.round(3).to_string())


class Tee:
    """Write printed output to the screen and to results_summary.txt."""
    def __init__(self, *streams): self.streams = streams
    def write(self, x): [s.write(x) for s in self.streams]
    def flush(self): [s.flush() for s in self.streams]


if __name__ == "__main__":
    args, df = da.parse_args(__doc__)
    with open(args.outdir / "results_summary.txt", "w") as log, \
            contextlib.redirect_stdout(Tee(sys.stdout, log)):
        print("Python", sys.version.split()[0], "| pandas", pd.__version__, "| numpy", np.__version__,
              "| statsmodels", __import__("statsmodels").__version__,
              "| scipy", __import__("scipy").__version__, "\n")
        run(df, args.data.name, args.sheet, args.outdir)
