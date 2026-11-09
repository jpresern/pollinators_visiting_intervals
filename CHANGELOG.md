# Changelog

All notable changes to this repository are listed here. Versions follow semantic versioning.

## [1.0.1] - 2026-11-09

### Changed
- Analysis scripts simplified: shared command-line handling, plot styling and percentage
  conversions in one place (`analysis_durations.py`); model formulas defined once under their
  manuscript names (D1-D3, I1-I4); models returned under those names.
- All outputs in `results/` and all printed output are byte-for-byte identical to 1.0.0.

## [1.0.0] - 2026-11-09

First release, supporting the manuscript submission.

- `data/apple_flower_visits.xlsx`: scored honeybee and wild-pollinator visits to apple flowers
  (1,597 records, 2017-2018) with weather and visit-bout variables.
- `analysis_durations.py`: Models D1-D3, Spearman correlation and within-bout permutation test.
- `analysis_intervals.py`: Models I1-I4 with random intercepts for observation, flower and bout; Figure 1.
- `run_all.py`: writes Tables 2-5 and the Figure 1 effects to `results/`.
- `diagnostics.py`: normality checks of residuals and random effects.
- PyCharm run configurations for all four scripts (`.idea/runConfigurations/`).
- Licences: MIT for the code (`LICENSE`), CC BY 4.0 for the data (`data/LICENSE.md`).
- Results verified to match the manuscript with pandas 3.1.1, numpy 2.5.4, scipy 1.18.1 and
  statsmodels 0.15.0, and with the versions reported in the manuscript
  (pandas 2.3.3, numpy 2.4.2, scipy 1.17.0, statsmodels 0.14.6).
