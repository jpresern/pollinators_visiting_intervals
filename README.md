# Honeybee visit duration on apple flowers: data and code

Data and analysis code for the manuscript *Pollination of apple orchards: effect of previous pollinator visit*
(Prešern J., Bevk D.; [journal, year, DOI to be added]).

Flower visits by honeybees and wild pollinators were video-recorded in an apple orchard
(Agricultural Institute of Slovenia, Lukovica; 2017 and 2018) and scored in BORIS.
The code tests whether the duration of a honeybee visit depends on the duration of the
preceding visit and on the time since the preceding visit (inter-visit interval),
using linear mixed models.

## Contents

```
data/
  apple_flower_visits.xlsx          scored flower visits with weather and visit-bout variables
  LICENSE.md                        data licence (CC BY 4.0)
src/
  analysis_durations.py             previous duration -> current duration (Models D1-D3, Table 3)
  analysis_intervals.py             previous interval -> current duration (Models I1-I4, Tables 4-5, Figure 1)
  run_all.py                        reproduces all tables and Figure 1 in one run
  diagnostics.py                    normality checks of residuals and random effects (Q-Q plots)
results/                            output of run_all.py and diagnostics.py (regenerated on every run)
requirements.txt                    Python packages, versions used to verify the results
CHANGELOG.md                        version history
LICENSE                             code licence (MIT)
```

## Reproducing the results

Tested with Python 3.14.2.

```bash
pip install -r requirements.txt
python src/run_all.py
python src/diagnostics.py
```

The scripts can be started from any folder; all paths are relative to the repository folder.
Each script also accepts `--data`, `--sheet` and `--outdir` (see `--help`).

In PyCharm, open the repository folder and pick one of the shared run configurations
(`.idea/runConfigurations/`): *run_all*, *diagnostics*, *analysis_durations* or *analysis_intervals*.
They use the project interpreter, which needs the packages in `requirements.txt`.

| Manuscript item | Produced by | Output file |
|---|---|---|
| Table 2, records and bouts per taxon | `run_all.py` | `results/table2_pollinators.csv` |
| Table 3, previous duration (D1-D3, Spearman, permutation test) | `run_all.py` | `results/table3_previous_duration.csv` |
| Table 4, previous interval (I1, I2, I4) | `run_all.py` | `results/table4_previous_interval.csv` |
| Table 5, variance components | `run_all.py` | `results/table5_variance_components.csv` |
| Figure 1, effects in Model I3 | `run_all.py` | `results/figure1_model_I3_effects.csv`, `results/honeybee_interval_analysis.png` |
| Q-Q check (Methods), limitations (Discussion) | `diagnostics.py` | `results/model_diagnostics.png` |
| All numbers quoted in the text | `run_all.py` | `results/results_summary.txt` |

In the `.csv` files, `beta` is the model coefficient with its 95% Wald interval (`beta_lo`, `beta_hi`),
and `pct` is the corresponding percentage change in visit duration (`pct_lo`, `pct_hi`):
per doubling for log-transformed predictors, 100 × (2^β − 1); per 1 SD or sun vs shade for the others, 100 × (e^β − 1).

## Data

`data/apple_flower_visits.xlsx` (sheet `apple`) has one row per scored visit (record).

The file is the output of the preprocessing pipeline: BORIS project → paired arrival/departure
events → visit duration → weather interpolated to the visit time → visit bouts and neighbour variables.
Records shorter than one video frame (0.04 s at 25 frames s⁻¹) were excluded at the last step
(five records); each excluded record ended its bout. No other records are excluded.

| Column | Unit | Description |
|---|---|---|
| `timestamp` | s | time of the departure (STOP) event from the start of the video |
| `subject` | | visitor taxon: honeybee, bumblebee type 1-4, solitary bee, hoverfly, fly, bee-fly |
| `observed`, `flower` | | filmed flower within the observation (flower1 ... flower6); identical columns |
| `observation` | | recording session (continuous video acquisition) |
| `date_time`, `date`, `time` | local time | real time of the departure event |
| `sunshade` | | flower in direct sun (`sun`) or shade (`shade`) at the time of the visit |
| `flag` | | always `STOP`: each row is a completed visit |
| `duration` | s | time the visitor spent on the flower |
| `Temp` | °C | air temperature, interpolated from the 30-min weather records |
| `solar` | W m⁻² | global solar radiation, interpolated |
| `wind` | m s⁻¹ | wind speed, interpolated |
| `visit_ID` | | visit bout: successive visits of the same taxon to the same flower, `<observation>_<flower>_<n>` |
| `duration_n-1`, `duration_n+1` | s | duration of the preceding and following visit in the same bout |
| `previous interval` | s | start of this visit minus end of the preceding visit in the same bout |
| `next interval` | s | start of the following visit minus end of this visit in the same bout |
| `pollinated_plant` | | always `apple` |

Neighbour variables are empty (NaN) for the first and last visit of a bout. Nine records lack a
solar radiation value and eight a wind value; they are omitted from models using these covariates.

A visit bout may include more than one individual; bees were not marked.

## Licence

- **Code** (`src/`): MIT licence, see [`LICENSE`](LICENSE).
- **Data** (`data/`): Creative Commons Attribution 4.0 International (CC BY 4.0), see [`data/LICENSE.md`](data/LICENSE.md).

## Citation

If you use these data or code, please cite the article above. [Zenodo DOI to be added]
