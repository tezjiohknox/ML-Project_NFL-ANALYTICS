""""
NFL Win/Loss Prediction (2010-2019)
Group Members: TezJioh Knox, Nate Bamikole

WHAT THIS PROGRAM DOES (plain English):
  1. Load NFL game stats from 10 seasons of CSV files (2010-2019)
  2. For each team, calculate their AVERAGE stats from all PREVIOUS games
     (so we never use the current game's stats to predict the current game)
  3. Create "difference" features: home team average MINUS away team average
     (this shows who has the advantage going into the game)
  4. Train 4 machine learning models to predict who wins:
       - Logistic Regression on offensive stats only        (Q1)
       - XGBoost on offensive stats only                   (Q1)
       - Logistic Regression on all stats (+ turnovers)    (Q2)
       - XGBoost on all stats (+ turnovers)                (Q2)
  5. Compare how well each model does on data it has never seen before
  6. Make charts to visualize the results
"""

# =============================================================================
# IMPORTS — bringing in tools we need
# =============================================================================

import os                          # helps build file paths that work on any computer
import warnings                    # lets us silence annoying warning messages
import numpy as np                 # math tools for arrays of numbers
import pandas as pd                # spreadsheet-style data tables (DataFrames)
import matplotlib.pyplot as plt    # drawing charts
import matplotlib.gridspec as gridspec  # lets us arrange multiple charts in a grid
import seaborn as sns              # prettier charts (built on top of matplotlib)

# scikit-learn: the main machine learning library
from sklearn.linear_model import LogisticRegression  # our first ML model
from sklearn.preprocessing import StandardScaler     # scales feature values to the same range
from sklearn.metrics import (
    accuracy_score,        # what % of predictions were correct
    classification_report, # precision, recall, and F1 score for each class
    confusion_matrix,      # table showing correct vs incorrect predictions
)

from xgboost import XGBClassifier  # our second ML model (gradient boosted trees)

warnings.filterwarnings("ignore")  # hide unimportant warnings so output stays clean


# =============================================================================
# SETTINGS — values we can easily change in one place
# =============================================================================

# Path to the folder containing the CSV files (relative to this script's location)
DATA_DIR = os.path.join(os.path.dirname(__file__), "NFL_Game_Stats")

# List of years we want to load: [2010, 2011, 2012, ..., 2019]
ALL_YEARS = list(range(2010, 2020))

# Seasons BEFORE this year go into training; this year and after go into testing
# Training: 2010-2016  |  Testing: 2017-2019
TRAIN_CUTOFF = 2017

# A fixed random seed so results are the same every time we run the code
RANDOM_STATE = 42

# The 7 stats we track for each team in each game
# "def_" columns are the OPPONENT'S yards — which is how many yards THIS team ALLOWED
TEAM_STAT_COLS = [
    "rush_att",       # rushing attempts
    "rush_yards",     # rushing yards gained
    "pass_yards",     # passing yards gained
    "turnovers",      # turnovers committed
    "score",          # points scored
    "def_rush_yards", # rushing yards this team ALLOWED (opponent's rush yards)
    "def_pass_yards", # passing yards this team ALLOWED (opponent's pass yards)
]

# Q1 uses only offensive features (no turnovers, no defense)
Q1_FEATURES = [
    "diff_rush_yards",      # home team avg rush yards MINUS away team avg rush yards
    "diff_pass_yards",      # home team avg pass yards MINUS away team avg pass yards
    "diff_rush_att",        # home team avg rush attempts MINUS away team avg rush attempts
    "home_avg_rush_yards",  # home team's own rolling average rushing yards
    "home_avg_pass_yards",  # home team's own rolling average passing yards
    "away_avg_rush_yards",  # away team's own rolling average rushing yards
    "away_avg_pass_yards",  # away team's own rolling average passing yards
]

# Q2 uses everything in Q1 PLUS turnovers and defensive yards allowed
Q2_FEATURES = Q1_FEATURES + [
    "diff_turnovers",           # home avg turnovers minus away avg turnovers
    "diff_def_rush_yards",      # home avg rush yards allowed minus away avg
    "diff_def_pass_yards",      # home avg pass yards allowed minus away avg
    "home_avg_turnovers",       # home team's average turnovers per game
    "home_avg_def_rush_yards",  # home team's average rush yards allowed per game
    "home_avg_def_pass_yards",  # home team's average pass yards allowed per game
    "away_avg_turnovers",       # away team's average turnovers per game
    "away_avg_def_rush_yards",  # away team's average rush yards allowed per game
    "away_avg_def_pass_yards",  # away team's average pass yards allowed per game
]


# =============================================================================
# STEP 1: LOAD THE DATA
# =============================================================================

def load_raw(years):
    """
    Read all the CSV files (one per year) and combine them into one big table.
    Also remove tied games and create our target column (home_win = 1 or 0).
    """

    # Start with an empty list — we'll add one DataFrame per season
    all_seasons = []

    for year in years:
        # Build the file path, e.g.: "NFL_Game_Stats/game_stats_2019.csv"
        file_path = os.path.join(DATA_DIR, f"game_stats_{year}.csv")

        # Read the CSV into a table (DataFrame)
        one_season = pd.read_csv(file_path)

        # Add a "Season" column so we know which year each row came from
        one_season["Season"] = year

        # Add this season's data to our running list
        all_seasons.append(one_season)

    # Stack all 10 seasons into one big table
    # ignore_index=True re-numbers the rows from 0 to N so there are no duplicates
    data = pd.concat(all_seasons, ignore_index=True)

    # Remove tied games — we can only predict a winner if there is one
    data = data[data["H-Score"] != data["A-Score"]].copy()

    # Create our target variable:
    #   1 = home team won (their score was higher)
    #   0 = away team won
    # .astype(int) converts True/False to 1/0
    data["home_win"] = (data["H-Score"] > data["A-Score"]).astype(int)

    return data


# =============================================================================
# STEP 2: RESHAPE THE DATA — one row per TEAM per game (instead of one per game)
# =============================================================================

def build_team_log(data):
    """
    The raw data has ONE row per game.
    To calculate each team's rolling average over time, we need ONE row per TEAM per game.
    So every game becomes TWO rows: one for the home team, one for the away team.

    We also set up "def_rush_yards" and "def_pass_yards" as the OPPONENT's offensive yards.
    For example, if Team A is at home and Team B (away) rushed for 120 yards,
    then Team A's "def_rush_yards" for that game = 120 (they ALLOWED 120 rush yards).
    """

    # --- Home team's perspective for every game ---
    home_rows = pd.DataFrame({
        "Season":         data["Season"],
        "Week":           data["Week"],
        "Team":           data["HomeTeam"],
        "rush_att":       data["H-RushAtt"],
        "rush_yards":     data["H-RushYards"],
        "pass_yards":     data["H-PassYards"],
        "turnovers":      data["H-Turnover"],
        "score":          data["H-Score"],
        "def_rush_yards": data["A-RushYards"],  # away team rushed this many yards AGAINST the home team
        "def_pass_yards": data["A-PassYards"],  # away team passed this many yards AGAINST the home team
    })

    # --- Away team's perspective for every game ---
    away_rows = pd.DataFrame({
        "Season":         data["Season"],
        "Week":           data["Week"],
        "Team":           data["AwayTeam"],
        "rush_att":       data["A-RushAtt"],
        "rush_yards":     data["A-RushYards"],
        "pass_yards":     data["A-PassYards"],
        "turnovers":      data["A-Turnover"],
        "score":          data["A-Score"],
        "def_rush_yards": data["H-RushYards"],  # home team rushed this many yards AGAINST the away team
        "def_pass_yards": data["H-PassYards"],  # home team passed this many yards AGAINST the away team
    })

    # Stack the home and away rows into one long table
    team_log = pd.concat([home_rows, away_rows], ignore_index=True)

    # Sort by Team, then Season, then Week
    # This is CRITICAL — rolling averages must be computed in time order
    team_log = team_log.sort_values(["Team", "Season", "Week"])
    team_log = team_log.reset_index(drop=True)  # re-number rows cleanly

    return team_log


# =============================================================================
# STEP 3: CALCULATE ROLLING AVERAGES — only using PAST games (no leakage!)
# =============================================================================

def calculate_prior_game_average(stat_values):
    """
    Given a list of a team's stat values in order [game1, game2, game3, ...],
    return the RUNNING AVERAGE leading up to each game — NOT including that game.

    Example for rush_yards = [100, 150, 80, 200]:
      Before game 1: no history yet         → NaN
      Before game 2: avg(100)               → 100.0
      Before game 3: avg(100, 150)          → 125.0
      Before game 4: avg(100, 150, 80)      → 110.0

    How it works:
      .expanding().mean() = cumulative average INCLUDING the current game
      .shift(1)           = slide every value down by 1 row,
                            so the current row now holds the PREVIOUS row's value
    The result: each row has the average of ALL games BEFORE it — never the current one.
    """
    running_avg_including_current = stat_values.expanding().mean()
    running_avg_before_current    = running_avg_including_current.shift(1)
    return running_avg_before_current


def add_rolling_averages(team_log):
    """
    For every stat column, calculate each team's rolling average going INTO each game.
    A new column called "avg_STAT" is added for each stat.

    This is what prevents data leakage — the model never sees the current game's
    real stats, only what the team averaged in all games BEFORE this one.
    """

    for stat in TEAM_STAT_COLS:
        # Group the table by team name so we process each team separately
        # Then apply our rolling average function within each team's games
        team_log["avg_" + stat] = (
            team_log
            .groupby("Team")[stat]
            .transform(calculate_prior_game_average)
        )

    return team_log


# =============================================================================
# STEP 4: BUILD GAME-LEVEL FEATURES
#         — attach both teams' rolling averages to each game row
# =============================================================================

def build_game_features(data, team_log):
    """
    For each game, we need to know:
      - The home team's rolling averages going into that game
      - The away team's rolling averages going into that game
      - The DIFFERENCE between them (the "matchup advantage")

    We do this by joining (merging) the rolling averages back onto the original
    game table — once for the home team and once for the away team.
    """

    # Build a list of the rolling average column names we created in Step 3
    # e.g.: ["avg_rush_att", "avg_rush_yards", "avg_pass_yards", ...]
    avg_column_names = ["avg_" + stat for stat in TEAM_STAT_COLS]

    # Grab just the columns we need for the lookup: season, week, team, and averages
    lookup_table = team_log[["Season", "Week", "Team"] + avg_column_names]

    # --- Prepare the HOME team lookup ---
    # Rename "Team" to "HomeTeam" and rename "avg_X" to "home_avg_X"
    # so the column names don't clash when we join
    home_rename_map = {"Team": "HomeTeam"}
    for stat in TEAM_STAT_COLS:
        home_rename_map["avg_" + stat] = "home_avg_" + stat

    home_lookup = lookup_table.rename(columns=home_rename_map)

    # --- Prepare the AWAY team lookup ---
    # Same idea but rename "avg_X" to "away_avg_X"
    away_rename_map = {"Team": "AwayTeam"}
    for stat in TEAM_STAT_COLS:
        away_rename_map["avg_" + stat] = "away_avg_" + stat

    away_lookup = lookup_table.rename(columns=away_rename_map)

    # --- Join the home team's rolling averages onto each game row ---
    # Match on Season + Week + HomeTeam so we get the right team's averages
    games = data.merge(home_lookup, on=["Season", "Week", "HomeTeam"], how="left")

    # --- Join the away team's rolling averages onto each game row ---
    games = games.merge(away_lookup, on=["Season", "Week", "AwayTeam"], how="left")

    # --- Create DIFFERENTIAL features: home average minus away average ---
    # A positive diff_pass_yards means the home team historically passes for more yards
    # A negative value means the away team has the passing advantage
    for stat in TEAM_STAT_COLS:
        games["diff_" + stat] = games["home_avg_" + stat] - games["away_avg_" + stat]

    # --- Drop games where we don't have rolling averages for both teams ---
    # This happens for the very first game a team ever plays (no prior history)
    all_feature_columns = (
        ["home_avg_" + stat for stat in TEAM_STAT_COLS] +
        ["away_avg_" + stat for stat in TEAM_STAT_COLS] +
        ["diff_"     + stat for stat in TEAM_STAT_COLS]
    )
    games = games.dropna(subset=all_feature_columns)  # remove rows with missing values
    games = games.reset_index(drop=True)               # re-number rows cleanly

    return games


# =============================================================================
# STEP 5: EVALUATE A MODEL — measure how well it predicted on a dataset
# =============================================================================

def evaluate_model(label, model, features, true_labels, scaler=None, show_full_report=True):
    """
    Test a trained model on a set of games and print the results.

    Parameters:
      label           - a name to print (e.g., "LR Q1 [TEST]")
      model           - the trained ML model
      features        - the input columns (X)
      true_labels     - the real answers (y) — 1 = home win, 0 = away win
      scaler          - if provided, scale the features before predicting
                        (Logistic Regression needs this; XGBoost does not)
      show_full_report - if False, only print the headline accuracy number
                         (used for training-set checks to keep output short)

    Returns:
      accuracy    - percentage of correct predictions (0.0 to 1.0)
      predictions - list of predicted labels (0 or 1)
    """

    # Scale the features if a scaler was provided
    # IMPORTANT: we only call .transform() here (not .fit_transform())
    # because the scaler was already fitted on training data
    if scaler is not None:
        features_ready = scaler.transform(features)
    else:
        features_ready = features

    # Ask the model to make predictions: 0 (away win) or 1 (home win)
    predictions = model.predict(features_ready)

    # Calculate accuracy: how many predictions were correct
    accuracy = accuracy_score(true_labels, predictions)

    # Print results
    print(f"\n{'─' * 55}")
    print(f"  {label}")
    print(f"{'─' * 55}")
    print(f"  Accuracy : {accuracy:.4f}")

    if show_full_report:
        # Print a detailed breakdown: precision, recall, and F1 for each class
        print(classification_report(true_labels, predictions,
                                    target_names=["Away Win", "Home Win"],
                                    zero_division=0))

    return accuracy, predictions


# =============================================================================
# CHART HELPER FUNCTIONS — each one draws one specific chart
# =============================================================================

def chart_class_balance(games, ax):
    """Bar chart showing how many home wins vs away wins are in the dataset."""

    # Count how many 0s (away wins) and 1s (home wins) are in the target column
    counts = games["home_win"].value_counts().sort_index()

    # Draw the bars
    bars = ax.bar(["Away Win", "Home Win"], counts.values,
                  color=["#e74c3c", "#2ecc71"], edgecolor="black")

    # Add the count number on top of each bar
    for bar, count in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 3,
                str(count),
                ha="center", va="bottom", fontweight="bold")

    ax.set_title("Class Balance (2010-2019)", fontsize=12)
    ax.set_ylabel("Number of Games")


def chart_diff_distributions(games, ax):
    """
    Histogram showing how differential features are distributed
    for games the home team won vs lost.
    If the distributions are separated, the feature has predictive power.
    """

    diff_cols = ["diff_rush_yards", "diff_pass_yards", "diff_turnovers"]
    colors    = ["#3498db", "#e67e22", "#9b59b6"]

    for col, color in zip(diff_cols, colors):
        # Grab the diff values for games the home team WON
        home_wins = games.loc[games["home_win"] == 1, col]
        # Grab the diff values for games the home team LOST
        home_losses = games.loc[games["home_win"] == 0, col]

        # Draw a filled histogram for wins and an outline histogram for losses
        ax.hist(home_wins,   bins=25, alpha=0.5, color=color, label=f"{col} (win)")
        ax.hist(home_losses, bins=25, alpha=0.3, color=color,
                histtype="step", linewidth=1.5, label=f"{col} (loss)")

    # Draw a vertical line at 0 (the breakeven point — neither team has an advantage)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)

    ax.set_title("Differential Feature Distributions", fontsize=12)
    ax.set_xlabel("Home Avg − Away Avg")
    ax.legend(fontsize=7, ncol=2)


def chart_correlation_heatmap(games, feature_cols, ax):
    """
    Heatmap showing how strongly each pair of features is related to each other
    and to the target (home_win). Values close to 1 or -1 = strong relationship.
    """

    # Include the target column so we can see which features correlate with winning
    cols_to_show = feature_cols + ["home_win"]

    # Calculate the correlation matrix (every feature vs every other feature)
    correlation_matrix = games[cols_to_show].corr()

    # mask = hide the upper triangle (it's a mirror of the lower triangle)
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

    sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt=".2f",
                cmap="coolwarm", center=0, ax=ax,
                linewidths=0.4, annot_kws={"size": 7})

    ax.set_title("Q2 Feature Correlation Matrix", fontsize=12)
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.tick_params(axis="y", labelsize=7)


def chart_confusion_matrix(cm, title, ax):
    """
    Draw a confusion matrix: a 2x2 table showing
      - True Negatives  (predicted away win,  actual away win)  ← correct
      - False Positives (predicted home win,  actual away win)  ← wrong
      - False Negatives (predicted away win,  actual home win)  ← wrong
      - True Positives  (predicted home win,  actual home win)  ← correct
    """
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Away Win", "Home Win"],
                yticklabels=["Away Win", "Home Win"])
    ax.set_title(title, fontsize=10)
    ax.set_ylabel("Actual",    fontsize=9)
    ax.set_xlabel("Predicted", fontsize=9)


def chart_model_comparison(summary, ax):
    """
    Bar chart comparing Accuracy across all 4 models.
    """

    # Pull the name and accuracy out of the summary list
    model_names = [row[0] for row in summary]
    accuracies  = [row[1] for row in summary]

    # x = position on the x-axis for each model (0, 1, 2, 3)
    x = np.arange(len(model_names))

    # Draw accuracy bars
    accuracy_bars = ax.bar(x, accuracies, color="#2980b9", edgecolor="black")

    # Add the value as a label on top of each bar
    for bar in accuracy_bars:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=8, rotation=10)
    ax.set_ylim(0, 1.1)

    # Draw a horizontal line at 0.5 to show the random-guessing baseline
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)

    ax.set_title("Model Comparison: Accuracy", fontsize=12)


def chart_xgb_importance(model, feature_names, ax):
    """
    Horizontal bar chart showing which features XGBoost relied on most.
    Longer bar = more important feature.
    """

    importances = model.feature_importances_

    # Sort the features from least to most important so the chart reads nicely
    sorted_indices = np.argsort(importances)
    sorted_names   = np.array(feature_names)[sorted_indices]
    sorted_values  = importances[sorted_indices]

    ax.barh(sorted_names, sorted_values, color="#9b59b6", edgecolor="black")
    ax.set_xlabel("Importance Score")
    ax.tick_params(axis="y", labelsize=8)


# =============================================================================
# MAIN — runs all the steps in order
# =============================================================================

def main():
    print("=" * 65)
    print("  NFL Win/Loss Prediction — 2010 to 2019")
    print("=" * 65)

    # ------------------------------------------------------------------
    # STEP 1-4: Load data and build leakage-free features
    # ------------------------------------------------------------------
    print("\nLoading data...")
    raw_data = load_raw(ALL_YEARS)

    print("Reshaping into team-level log...")
    team_log = build_team_log(raw_data)

    print("Calculating rolling averages (prior games only)...")
    team_log = add_rolling_averages(team_log)

    print("Building game-level features and difference columns...")
    games = build_game_features(raw_data, team_log)

    # Show some basic stats about the final dataset
    home_win_rate = games["home_win"].mean()
    print(f"\nFinal dataset: {len(games)} games")
    print(f"  Home team wins: {home_win_rate:.1%}")
    print(f"  Away team wins: {1 - home_win_rate:.1%}")
    print(f"  Games removed (no prior history available): {len(raw_data) - len(games)}")

    # ------------------------------------------------------------------
    # TRAIN / TEST SPLIT — older seasons train the model, recent seasons test it
    # We never shuffle randomly because the ORDER of time matters here.
    # ------------------------------------------------------------------
    train_games = games[games["Season"] <  TRAIN_CUTOFF]  # 2010 - 2016
    test_games  = games[games["Season"] >= TRAIN_CUTOFF]  # 2017 - 2019

    print(f"\nTraining on {len(train_games)} games (2010-2016)")
    print(f"Testing  on {len(test_games)} games (2017-2019)")

    # Separate the target column (what we want to predict) from the feature columns
    y_train = train_games["home_win"]  # the answers for training
    y_test  = test_games["home_win"]   # the answers for testing (model never sees these during training)

    # ------------------------------------------------------------------
    # Q1a: LOGISTIC REGRESSION — offensive stats only
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("  Q1a — Logistic Regression (Offensive Stats Only)")
    print("=" * 65)
    print("  Logistic Regression gives us coefficients we can read like a formula.")
    print("  It shows exactly which offensive stats matter most for predicting wins.")

    # StandardScaler brings all features to the same scale (mean=0, std=1)
    # This is required for Logistic Regression to work well
    scaler_q1 = StandardScaler()

    # fit_transform = LEARN the mean and std from training data, then scale it
    # We ONLY call fit on training data — never on test data
    train_features_q1_scaled = scaler_q1.fit_transform(train_games[Q1_FEATURES])

    # Create and train the Logistic Regression model
    lr_q1 = LogisticRegression(
        max_iter=2000,          # allow up to 2000 steps to find the best coefficients
        random_state=RANDOM_STATE  # fixed seed so results are reproducible
    )
    lr_q1.fit(train_features_q1_scaled, y_train)

    # Evaluate on TRAINING data (to check for overfitting later)
    acc_lr_q1_train, _ = evaluate_model(
        "LR Q1 [TRAIN]", lr_q1,
        train_games[Q1_FEATURES], y_train,
        scaler=scaler_q1, show_full_report=False  # keep output short for training eval
    )

    # Evaluate on TEST data (the real performance score)
    acc_lr_q1_test, pred_lr_q1 = evaluate_model(
        "LR Q1 [TEST]", lr_q1,
        test_games[Q1_FEATURES], y_test,
        scaler=scaler_q1, show_full_report=True
    )

    # Print the coefficients so we can see which features matter most
    # lr_q1.coef_[0] is a list of one coefficient per feature
    # Positive coefficient = this feature being higher makes a home win MORE likely
    coeff_table = pd.DataFrame({
        "Feature":     Q1_FEATURES,
        "Coefficient": lr_q1.coef_[0],
    }).sort_values("Coefficient", ascending=False)
    print("\n  LR Q1 Coefficients (larger positive = stronger predictor of home win):")
    print(coeff_table.to_string(index=False))

    # ------------------------------------------------------------------
    # Q1b: XGBOOST — offensive stats only (same features as Q1a)
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("  Q1b — XGBoost (Offensive Stats Only)")
    print("=" * 65)
    print("  XGBoost builds 300 decision trees that each learn from the previous ones.")
    print("  Using the same Q1 features, we can compare it directly against LR.")

    # XGBoost does NOT need scaling — decision trees work fine with raw values
    xgb_q1 = XGBClassifier(
        n_estimators=300,       # build 300 trees total
        max_depth=4,            # each tree can have at most 4 levels of decisions
        learning_rate=0.05,     # each new tree corrects only 5% of the previous error (slow and steady)
        subsample=0.8,          # each tree randomly uses 80% of the training rows
        colsample_bytree=0.8,   # each tree randomly uses 80% of the features
        min_child_weight=3,     # a leaf node must cover at least 3 games (prevents tiny overfit splits)
        gamma=0.1,              # a split must improve the score by at least 0.1 to be made
        eval_metric="logloss",  # the error metric used internally during training
        random_state=RANDOM_STATE,
        verbosity=0,            # don't print anything during training
    )
    xgb_q1.fit(train_games[Q1_FEATURES], y_train)

    # Evaluate on training data
    acc_xgb_q1_train, _ = evaluate_model(
        "XGB Q1 [TRAIN]", xgb_q1,
        train_games[Q1_FEATURES], y_train,
        show_full_report=False
    )

    # Evaluate on test data
    acc_xgb_q1_test, pred_xgb_q1 = evaluate_model(
        "XGB Q1 [TEST]", xgb_q1,
        test_games[Q1_FEATURES], y_test,
        show_full_report=True
    )

    # ------------------------------------------------------------------
    # Q2a: LOGISTIC REGRESSION — all features (offensive + turnovers + defense)
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("  Q2a — Logistic Regression (All Features)")
    print("=" * 65)
    print("  Same LR setup as Q1, but with 9 extra features added.")
    print("  Comparing Q2a accuracy vs Q1a accuracy shows how much turnovers help.")

    scaler_q2 = StandardScaler()
    train_features_q2_scaled = scaler_q2.fit_transform(train_games[Q2_FEATURES])

    lr_q2 = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
    lr_q2.fit(train_features_q2_scaled, y_train)

    acc_lr_q2_train, _ = evaluate_model(
        "LR Q2 [TRAIN]", lr_q2,
        train_games[Q2_FEATURES], y_train,
        scaler=scaler_q2, show_full_report=False
    )
    acc_lr_q2_test, pred_lr_q2 = evaluate_model(
        "LR Q2 [TEST]", lr_q2,
        test_games[Q2_FEATURES], y_test,
        scaler=scaler_q2, show_full_report=True
    )

    # ------------------------------------------------------------------
    # Q2b: XGBOOST — all features
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("  Q2b — XGBoost (All Features)")
    print("=" * 65)
    print("  XGBoost on the full feature set. Can it find non-linear turnover")
    print("  interactions that Logistic Regression cannot capture?")

    xgb_q2 = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        verbosity=0,
    )
    xgb_q2.fit(train_games[Q2_FEATURES], y_train)

    acc_xgb_q2_train, _ = evaluate_model(
        "XGB Q2 [TRAIN]", xgb_q2,
        train_games[Q2_FEATURES], y_train,
        show_full_report=False
    )
    acc_xgb_q2_test, pred_xgb_q2 = evaluate_model(
        "XGB Q2 [TEST]", xgb_q2,
        test_games[Q2_FEATURES], y_test,
        show_full_report=True
    )

    # ------------------------------------------------------------------
    # SUMMARY — compare all 4 models side by side
    # ------------------------------------------------------------------

    # This list is used by the bar chart (test scores only)
    summary_for_chart = [
        ("LR  Q1 (off. only)", acc_lr_q1_test),
        ("XGB Q1 (off. only)", acc_xgb_q1_test),
        ("LR  Q2 (all feats)", acc_lr_q2_test),
        ("XGB Q2 (all feats)", acc_xgb_q2_test),
    ]

    # This table shows train vs test for overfitting analysis
    train_vs_test = [
        ("LR  Q1",  acc_lr_q1_train,  acc_lr_q1_test),
        ("XGB Q1",  acc_xgb_q1_train, acc_xgb_q1_test),
        ("LR  Q2",  acc_lr_q2_train,  acc_lr_q2_test),
        ("XGB Q2",  acc_xgb_q2_train, acc_xgb_q2_test),
    ]

    print("\n" + "=" * 60)
    print("  TRAIN vs TEST SUMMARY")
    print("  Gap = Train Accuracy - Test Accuracy")
    print("  A large positive gap means the model memorized training data (overfitting)")
    print("=" * 60)
    print(f"  {'Model':<10} {'Train Acc':>10} {'Test Acc':>10} {'Gap':>8}")
    print(f"  {'─'*10} {'─'*10} {'─'*10} {'─'*8}")

    for name, tr_acc, te_acc in train_vs_test:
        gap = tr_acc - te_acc
        overfit_warning = " ← overfit!" if gap > 0.08 else ""
        print(f"  {name:<10} {tr_acc:>10.4f} {te_acc:>10.4f} {gap:>+8.4f}{overfit_warning}")

    # Calculate useful comparisons between models
    xgb_vs_lr_q1    = acc_xgb_q1_test - acc_lr_q1_test   # does XGBoost beat LR on Q1?
    lr_turnover_lift = acc_lr_q2_test  - acc_lr_q1_test   # how much do turnovers help LR?
    xgb_turnover_lift= acc_xgb_q2_test - acc_xgb_q1_test  # how much do turnovers help XGBoost?
    xgb_vs_lr_q2    = acc_xgb_q2_test - acc_lr_q2_test   # does XGBoost beat LR on Q2?

    print(f"\n  Does XGBoost beat LR on Q1 (same features)?  {xgb_vs_lr_q1:+.4f}")
    print(f"  How much do turnovers help LR?               {lr_turnover_lift:+.4f}")
    print(f"  How much do turnovers help XGBoost?          {xgb_turnover_lift:+.4f}")
    print(f"  Does XGBoost beat LR on Q2 (same features)?  {xgb_vs_lr_q2:+.4f}")

    # Determine which model won each question
    best_model_q1 = "XGBoost" if acc_xgb_q1_test >= acc_lr_q1_test else "Logistic Regression"
    best_model_q2 = "XGBoost" if acc_xgb_q2_test >= acc_lr_q2_test else "Logistic Regression"

    print("\n  Conclusions:")
    print(f"  Q1: {best_model_q1} was better on offensive-only features "
          f"({max(acc_lr_q1_test, acc_xgb_q1_test):.1%} accuracy).")
    print(f"  Q2: {best_model_q2} was better on all features "
          f"({max(acc_lr_q2_test, acc_xgb_q2_test):.1%} accuracy).")

    if lr_turnover_lift > 0.01 or xgb_turnover_lift > 0.01:
        print(f"  Adding turnovers + defense improved both models "
              f"(LR: {lr_turnover_lift:+.1%}, XGBoost: {xgb_turnover_lift:+.1%}).")
    else:
        print("  Turnovers and defense provided only a small improvement.")

    # ------------------------------------------------------------------
    # VISUALIZATIONS — create a 3x4 grid of charts and save to results.png
    # ------------------------------------------------------------------
    #
    # Grid layout (3 rows x 4 columns):
    #   Row 0: [class balance] [diff distributions] [correlation heatmap (2 cols wide)]
    #   Row 1: [LR Q1 CM]     [XGB Q1 CM]          [LR Q2 CM]     [XGB Q2 CM]
    #   Row 2: [model comparison (2 cols wide)] [XGB Q1 importance] [XGB Q2 importance]
    #

    fig = plt.figure(figsize=(22, 17))
    fig.suptitle("NFL Win/Loss Prediction — LR & XGBoost on Q1 and Q2 (2010-2019)",
                 fontsize=15, fontweight="bold", y=0.99)

    # Create the 3x4 grid
    grid = gridspec.GridSpec(3, 4, figure=fig, hspace=0.55, wspace=0.4)

    # Row 0: exploratory charts
    ax_balance      = fig.add_subplot(grid[0, 0])       # 1 column wide
    ax_diff_dist    = fig.add_subplot(grid[0, 1])       # 1 column wide
    ax_correlation  = fig.add_subplot(grid[0, 2:4])     # 2 columns wide

    # Row 1: confusion matrices (one per model)
    ax_cm_lr_q1  = fig.add_subplot(grid[1, 0])
    ax_cm_xgb_q1 = fig.add_subplot(grid[1, 1])
    ax_cm_lr_q2  = fig.add_subplot(grid[1, 2])
    ax_cm_xgb_q2 = fig.add_subplot(grid[1, 3])

    # Row 2: model comparison + XGBoost feature importances
    ax_comparison    = fig.add_subplot(grid[2, 0:2])   # 2 columns wide
    ax_importance_q1 = fig.add_subplot(grid[2, 2])     # 1 column wide
    ax_importance_q2 = fig.add_subplot(grid[2, 3])     # 1 column wide

    # Draw each chart
    chart_class_balance(games, ax_balance)
    chart_diff_distributions(games, ax_diff_dist)
    chart_correlation_heatmap(games, Q2_FEATURES[:9], ax_correlation)

    chart_confusion_matrix(confusion_matrix(y_test, pred_lr_q1),
                           "LR — Offensive Only (Q1)",  ax_cm_lr_q1)
    chart_confusion_matrix(confusion_matrix(y_test, pred_xgb_q1),
                           "XGB — Offensive Only (Q1)", ax_cm_xgb_q1)
    chart_confusion_matrix(confusion_matrix(y_test, pred_lr_q2),
                           "LR — All Features (Q2)",    ax_cm_lr_q2)
    chart_confusion_matrix(confusion_matrix(y_test, pred_xgb_q2),
                           "XGB — All Features (Q2)",   ax_cm_xgb_q2)

    chart_model_comparison(summary_for_chart, ax_comparison)

    chart_xgb_importance(xgb_q1, Q1_FEATURES, ax_importance_q1)
    ax_importance_q1.set_title("XGB Feature Importance\n(Q1 — Offensive Only)", fontsize=10)

    chart_xgb_importance(xgb_q2, Q2_FEATURES, ax_importance_q2)
    ax_importance_q2.set_title("XGB Feature Importance\n(Q2 — All Features)", fontsize=10)

    # Save the full chart grid as a PNG file
    output_path = os.path.join(os.path.dirname(__file__), "results.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n  Charts saved to: {output_path}")
    plt.show()


# =============================================================================
# ENTRY POINT — this runs main() when you execute the script directly
# =============================================================================
if __name__ == "__main__":
    main()