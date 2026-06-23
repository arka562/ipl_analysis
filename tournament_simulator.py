import argparse
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm

# Current 10 IPL Franchises and their primary home venues
TEAMS_AND_VENUES = {
    "Chennai Super Kings": "MA Chidambaram Stadium, Chepauk, Chennai",
    "Mumbai Indians": "Wankhede Stadium, Mumbai",
    "Royal Challengers Bengaluru": "M Chinnaswamy Stadium, Bengaluru",
    "Kolkata Knight Riders": "Eden Gardens, Kolkata",
    "Rajasthan Royals": "Sawai Mansingh Stadium, Jaipur",
    "Sunrisers Hyderabad": "Rajiv Gandhi International Stadium, Uppal, Hyderabad",
    "Delhi Capitals": "Arun Jaitley Stadium, Delhi",
    "Punjab Kings": "Punjab Cricket Association IS Bindra Stadium, Mohali",
    "Gujarat Titans": "Narendra Modi Stadium, Ahmedabad",
    "Lucknow Super Giants": "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow",
}

# Standardize historical team names to their modern equivalents
TEAM_MAPPINGS = {
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Kings XI Punjab": "Punjab Kings",
    "Delhi Daredevils": "Delhi Capitals",
    "Deccan Chargers": "Sunrisers Hyderabad",       # Approximate franchise lineage
    "Gujarat Lions": "Gujarat Titans",              # Approximate region mapping for data volume
    "Rising Pune Supergiant": "Lucknow Super Giants", # Approximate replacement
    "Rising Pune Supergiants": "Lucknow Super Giants",
    "Pune Warriors": "Lucknow Super Giants",
}

def load_and_prep_data(processed_dir: Path):
    matches = pd.read_csv(processed_dir / "matches.csv")
    
    # Apply standardizations
    matches["team_1"] = matches["team_1"].replace(TEAM_MAPPINGS)
    matches["team_2"] = matches["team_2"].replace(TEAM_MAPPINGS)
    matches["winner"] = matches["winner"].replace(TEAM_MAPPINGS)
    
    # We only care about matches with a definitive winner
    matches = matches.dropna(subset=["winner"]).copy()
    
    # To make the model symmetrical, we add each match twice (Team 1 perspective, then Team 2 perspective)
    rows = []
    for _, row in matches.iterrows():
        t1, t2, winner, venue = row["team_1"], row["team_2"], row["winner"], row["venue"]
        if pd.isna(t1) or pd.isna(t2) or pd.isna(venue):
            continue
            
        rows.append({
            "team_A": t1, "team_B": t2, "venue": venue,
            "team_A_won": 1 if winner == t1 else 0
        })
        rows.append({
            "team_A": t2, "team_B": t1, "venue": venue,
            "team_A_won": 1 if winner == t2 else 0
        })
        
    return pd.DataFrame(rows)


def build_prematch_model(df: pd.DataFrame):
    """Trains a Random Forest classifier to predict win probability before the toss."""
    print("Training Pre-Match Win Probability Model...")
    
    X = df[["team_A", "team_B", "venue"]]
    y = df["team_A_won"]
    
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]),
                ["team_A", "team_B", "venue"],
            ),
        ]
    )
    
    model = Pipeline([
        ("preprocess", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1))
    ])
    
    model.fit(X, y)
    print("Model Training Complete.")
    return model


def generate_schedule():
    """Generates a double round-robin schedule for the 10 teams."""
    teams = list(TEAMS_AND_VENUES.keys())
    schedule = []
    
    for i in range(len(teams)):
        for j in range(len(teams)):
            if i != j:
                home_team = teams[i]
                away_team = teams[j]
                venue = TEAMS_AND_VENUES[home_team]
                schedule.append((home_team, away_team, venue))
                
    return schedule


def simulate_match(model, team_A, team_B, venue):
    """Simulates a single match using the model's predicted probabilities."""
    df = pd.DataFrame([{"team_A": team_A, "team_B": team_B, "venue": venue}])
    prob_A_wins = model.predict_proba(df)[0][1]
    
    # Roll a random number between 0 and 1
    if random.random() < prob_A_wins:
        return team_A
    return team_B


def run_monte_carlo(model, num_simulations=1000):
    schedule = generate_schedule()
    teams = list(TEAMS_AND_VENUES.keys())
    
    # Track stats across all simulations
    results = {team: {
        "total_group_wins": 0,
        "playoff_appearances": 0,
        "championships": 0,
    } for team in teams}
    
    neutral_venue = "Narendra Modi Stadium, Ahmedabad"  # Final & playoffs venue
    
    print(f"\nSimulating {num_simulations} seasons...")
    for _ in tqdm(range(num_simulations)):
        points = defaultdict(int)
        
        # 1. GROUP STAGE
        for home_team, away_team, venue in schedule:
            winner = simulate_match(model, home_team, away_team, venue)
            points[winner] += 2
            
        # Add random float tiebreaker (simulating NRR)
        standings = [(team, points[team] + random.random()) for team in teams]
        standings.sort(key=lambda x: x[1], reverse=True)
        
        # Log group stage wins (removing the tiebreaker float and dividing by 2)
        for rank, (team, score) in enumerate(standings):
            wins = int(score) // 2
            results[team]["total_group_wins"] += wins
            if rank < 4:
                results[team]["playoff_appearances"] += 1
                
        # 2. PLAYOFFS
        rank_1, rank_2, rank_3, rank_4 = [s[0] for s in standings[:4]]
        
        # Qualifier 1: Rank 1 vs Rank 2
        q1_winner = simulate_match(model, rank_1, rank_2, neutral_venue)
        q1_loser = rank_2 if q1_winner == rank_1 else rank_1
        
        # Eliminator: Rank 3 vs Rank 4
        elim_winner = simulate_match(model, rank_3, rank_4, neutral_venue)
        
        # Qualifier 2: Q1 Loser vs Eliminator Winner
        q2_winner = simulate_match(model, q1_loser, elim_winner, neutral_venue)
        
        # Final: Q1 Winner vs Q2 Winner
        champion = simulate_match(model, q1_winner, q2_winner, neutral_venue)
        
        results[champion]["championships"] += 1
        
    # Aggregate
    final_df = []
    for team in teams:
        final_df.append({
            "Team": team,
            "Avg Group Wins": round(results[team]["total_group_wins"] / num_simulations, 1),
            "Playoff Probability (%)": round(results[team]["playoff_appearances"] / num_simulations * 100, 1),
            "Championship Probability (%)": round(results[team]["championships"] / num_simulations * 100, 1),
        })
        
    return pd.DataFrame(final_df).sort_values("Championship Probability (%)", ascending=False)


def main():
    parser = argparse.ArgumentParser(description="Monte Carlo IPL Tournament Simulator")
    
    default_processed = Path("ipl_analytics_platform/data/processed")
    if Path("parsers/ipl_analytics_platform/data/processed").exists():
        default_processed = Path("parsers/ipl_analytics_platform/data/processed")
        
    parser.add_argument("--processed-dir", default=str(default_processed))
    parser.add_argument("--simulations", type=int, default=1000)
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    
    print("Loading historical data...")
    df = load_and_prep_data(processed_dir)
    
    model = build_prematch_model(df)
    
    results_df = run_monte_carlo(model, num_simulations=args.simulations)
    
    print("\n" + "="*60)
    print("🏆 TOURNAMENT SIMULATION RESULTS 🏆")
    print("="*60)
    print(results_df.to_string(index=False))
    print("="*60)
    
    # Save to disk
    reports_dir = Path("ipl_analytics_platform/reports/modeling")
    reports_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(reports_dir / "tournament_simulation_results.csv", index=False)
    print(f"\nSaved full results to {reports_dir / 'tournament_simulation_results.csv'}")


if __name__ == "__main__":
    main()
