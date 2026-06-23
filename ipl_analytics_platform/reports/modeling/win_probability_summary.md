# Win Probability Model

## Task

Predict whether the batting team will win from over-by-over match checkpoints.

## Metrics

- Rows used  : 47,966
- Train rows : 38,372
- Test rows  : 9,594
- Accuracy   : 0.74
- ROC AUC    : 0.834
- Log loss   : 0.496

## Features

- `season`
- `innings`
- `batting_team`
- `over_number`
- `current_score`
- `wickets_lost`
- `balls_bowled`
- `overs_remaining`
- `current_run_rate`
- `last_over_runs`
- `last_over_wickets`
- `target`
- `runs_required`
- `required_run_rate`
- `is_chase`

## Interpretation

This model powers live-style win probability curves. It shows how a team's chances evolved over each over based on score, wickets, innings context, target pressure, and recent scoring.
