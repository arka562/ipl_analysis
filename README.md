# IPL Analytics Platform

An end-to-end cricket analytics pipeline built on ball-by-ball IPL data (Cricsheet, 2008–present). Covers data parsing, advanced metric engineering, three ML models, and a multi-page Streamlit dashboard.

**Live app:** https://iplanalysis-ja3jmqhz5kra7wfydke9xw.streamlit.app/

---

## What it does

The dashboard has 7 pages:

| Page | What it shows |
|---|---|
| Season Overview | Run rate, average innings score, and wickets trends across seasons |
| Team Performance | Win %, toss impact, chasing vs defending trends, phase-wise team strength |
| Player Analysis | Player impact scores, strike rate vs runs, phase-wise top batters, death-overs specialists |
| Venue Insights | Par scores, chasing win % by venue, scoring patterns by ground |
| Win Probability | Over-by-over win probability curve for any match in the dataset |
| Score Predictor | Predicts final innings score from the first 10 overs of input stats |
| Tournament Simulator | Monte Carlo simulation of championship and playoff probabilities |

## How it's built

**Data pipeline**
- Source: Cricsheet ball-by-ball JSON (`json_files/`), parsed into structured CSVs (`parsers/`)
- Covers 294K+ deliveries across 1,239+ matches (2008–present)
- Processed output feeds a `reports/` directory split into `tables/`, `advanced_metrics/`, and `modeling/`

**Models** (trained via `model1.py`, `model2.py`, `model3.py` / `win_probablity_model.py`, `tournament_simulator.py`)
- **Win Probability** — `RandomForestClassifier` predicting batting-team win probability at each over checkpoint. Trained on engineered features: current score, wickets lost, run rate, target, required run rate, chase indicator. Evaluated on accuracy, ROC-AUC, and log loss (exact numbers are in `reports/modeling/win_probability_metrics.json`, generated at training time — check there rather than this README for current values).
- **Score Predictor** — regression model predicting final innings score from the first 10 overs (runs, wickets, boundaries, dot balls, run rate). Reports MAE, R², and RMSE at runtime via `reports/modeling/score_model3_metrics.json`.
- **Tournament Simulator** — Monte Carlo simulation (1,000 runs) estimating per-team championship and playoff probabilities for an upcoming season.

**Advanced metrics** (`advanced_metrices.py`)
- Player impact score (role-aware, eligibility-filtered)
- Death-overs batting/bowling index
- Venue par score and team phase-strength scoring

**Frontend**
- Streamlit, dark theme, Plotly charts (bar, line, scatter, gauge)
- Custom CSS for metric cards, sidebar, and typography (Bebas Neue / DM Sans)

## Tech stack

`Python` · `Streamlit` · `pandas` · `NumPy` · `scikit-learn` · `Plotly` · `joblib`

## Project structure

```
ipl_analysis/
├── app.py                      # Streamlit dashboard (7 pages)
├── advanced_metrices.py        # Player/venue/team advanced metrics
├── insights.py
├── model1.py / model2.py / model3.py
├── win_probablity_model.py     # Win probability RandomForest training
├── win_probablity_curve.py
├── tournament_simulator.py     # Monte Carlo tournament simulation
├── warehouse_and_reports.py
├── parsers/                    # Cricsheet JSON → structured CSV
├── json_files/                 # Raw Cricsheet match data
└── requirements.txt
```

## Running it locally

```bash
git clone https://github.com/arka562/ipl_analysis.git
cd ipl_analysis
pip install -r requirements.txt
streamlit run app.py
```

Note: the dashboard expects pre-built CSVs/models under `reports/` and `models/`. If these aren't present, run the relevant pipeline script first (e.g. `python win_probablity_model.py`, `python tournament_simulator.py`) to generate them — the Tournament Simulator page will tell you explicitly if its output file is missing.

## Data source

[Cricsheet](https://cricsheet.org/) — ball-by-ball IPL data, 2008 to present.

## Status

Actively developed as a personal analytics project. Not production-hardened — no tests, no CI, and directory paths currently have fallback logic to handle inconsistent local vs. deployed folder structures (see `BASE` resolution in `app.py`). Treat model metrics as a snapshot of the latest training run, not guaranteed constants.
