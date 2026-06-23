# Score Prediction Model — v3 (Time Decay)

## Task

Predict final innings score using match context and first 10 overs. Recent seasons are given exponentially more weight (decay factor = 0.15).

## Metrics

- Rows used       : 2,449
- Train rows      : 1,959
- Test rows       : 490
- MAE             : 15.59 runs
- RMSE            : 21.3 runs
- R²              : 0.62
- MAE (1st inn)   : 18.84 runs
- MAE (2nd inn)   : 12.28 runs
- R² (1st inn)    : 0.501
- R² (2nd inn)    : 0.69

## Key Improvements Over v2

- Added exponential time decay to sample weights
- Decay factor of 0.15 heavily prioritizes data from the last 3-5 years
- Better reflects modern scoring trends

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

This model estimates where an innings is likely to finish after 10 overs. By weighting recent data higher, it avoids under-predicting modern IPL scores where 200+ run totals are significantly more common than in 2008-2015.
