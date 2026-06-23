# Score Prediction Model — v2

## Task

Predict final innings score using match context and first 10 overs.

## Metrics

- Rows used       : 2,449
- Train rows      : 1,959
- Test rows       : 490
- MAE             : 15.51 runs
- RMSE            : 21.36 runs
- R²              : 0.618
- MAE (1st inn)   : 18.6 runs
- MAE (2nd inn)   : 12.37 runs
- R² (1st inn)    : 0.505
- R² (2nd inn)    : 0.68

## Key Improvements Over v1

- Extended feature window from 6 to 10 overs
- Added venue as a categorical feature
- Added target score, runs needed, and required run rate for 2nd innings
- Added sixes count and six percentage
- Added balls remaining
- Switched from Random Forest to Gradient Boosting

## Features

- `season`
- `innings`
- `batting_team`
- `venue`
- `runs_so_far`
- `wickets_so_far`
- `balls_so_far`
- `boundaries`
- `sixes`
- `dot_balls`
- `run_rate_so_far`
- `wickets_remaining`
- `boundary_pct`
- `dot_ball_pct`
- `six_pct`
- `is_first_innings`
- `balls_remaining`
- `target_score`
- `runs_needed`
- `required_run_rate`

## Interpretation

This model estimates where an innings is likely to finish after 10 overs. Second-innings predictions are stronger because the model knows the target. Useful for score projection, par-score analysis, and live match storytelling.
