"""
NFL Win/Loss Prediction (2010-2019) — Revised
Group Members: TezJioh Knox, Nate Bamikole

Addresses professor feedback:
  1. Both Logistic Regression AND XGBoost run on BOTH research questions
  2. No data leakage — each prediction uses only rolling averages from PRIOR games
  3. Expanded to all 10 available seasons (2010-2019)
  4. Differential features (home rolling avg - away rolling avg per stat)
  5. Chronological train/test split (train 2010-2016, test 2017-2019)

Model matrix:
  Q1 (offensive stats only): Logistic Regression | XGBoost
  Q2 (all features + turnovers + defense): Logistic Regression | XGBoost
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve,
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR     = os.path.join(os.path.dirname(__file__), "NFL_Game_Stats")
ALL_YEARS    = list(range(2010, 2020))
TRAIN_CUTOFF = 2017          # train: 2010-2016 | test: 2017-2019
RANDOM_STATE = 42

# Raw stat columns tracked per team per game
TEAM_STAT_COLS = [
    "rush_att", "rush_yards", "pass_yards",
    "turnovers", "score",
    "def_rush_yards",   # opponent's rush yards = yards this team allowed
    "def_pass_yards",   # opponent's pass yards = yards this team allowed
]

# Feature subsets for each research question
Q1_FEATURES = [
    "diff_rush_yards", "diff_pass_yards", "diff_rush_att",
    "home_avg_rush_yards", "home_avg_pass_yards",
    "away_avg_rush_yards", "away_avg_pass_yards",
]
Q2_FEATURES = Q1_FEATURES + [
    "diff_turnovers", "diff_def_rush_yards", "diff_def_pass_yards",
    "home_avg_turnovers", "home_avg_def_rush_yards", "home_avg_def_pass_yards",
    "away_avg_turnovers", "away_avg_def_rush_yards", "away_avg_def_pass_yards",
]


# ── 1. Load raw data ──────────────────────────────────────────────────────────

def load_raw(years):
    frames = []
    for yr in years:
        df = pd.read_csv(os.path.join(DATA_DIR, f"game_stats_{yr}.csv"))
        df["Season"] = yr
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    data = data[data["H-Score"] != data["A-Score"]].copy()   # drop ties
    data["home_win"] = (data["H-Score"] > data["A-Score"]).astype(int)
    return data


# ── 2. Build per-team game log ────────────────────────────────────────────────

def build_team_log(data):
    """
    Convert game-level data to a long-format team-level log.
    Each row = one team's performance in one game.
    'def_rush_yards' and 'def_pass_yards' are the OPPONENT's offensive yards,
    representing how many yards this team ALLOWED on defense.
    """
    home = pd.DataFrame({
        "Season":        data["Season"],
        "Week":          data["Week"],
        "Team":          data["HomeTeam"],
        "rush_att":      data["H-RushAtt"],
        "rush_yards":    data["H-RushYards"],
        "pass_yards":    data["H-PassYards"],
        "turnovers":     data["H-Turnover"],
        "score":         data["H-Score"],
        "def_rush_yards": data["A-RushYards"],
        "def_pass_yards": data["A-PassYards"],
    })
    away = pd.DataFrame({
        "Season":        data["Season"],
        "Week":          data["Week"],
        "Team":          data["AwayTeam"],
        "rush_att":      data["A-RushAtt"],
        "rush_yards":    data["A-RushYards"],
        "pass_yards":    data["A-PassYards"],
        "turnovers":     data["A-Turnover"],
        "score":         data["A-Score"],
        "def_rush_yards": data["H-RushYards"],
        "def_pass_yards": data["H-PassYards"],
    })
    log = pd.concat([home, away], ignore_index=True)
    return log.sort_values(["Team", "Season", "Week"]).reset_index(drop=True)


# ── 3. Compute expanding rolling averages (no leakage) ────────────────────────

def add_rolling_averages(log):
    """
    For each team, compute the cumulative average of each stat up to (but NOT
    including) the current game by using expanding().mean().shift(1).

    This guarantees that when predicting game N, only stats from games 1..N-1
    are used — eliminating any data leakage from same-game statistics.
    """
    for col in TEAM_STAT_COLS:
        log[f"avg_{col}"] = (
            log.groupby("Team")[col]
               .transform(lambda x: x.expanding().mean().shift(1))
        )
    return log


# ── 4. Join rolling averages onto game-level rows & build diff features ───────

def build_game_features(data, log):
    """
    For each game, look up both the home and away team's pre-game rolling
    averages, then compute differential features (home avg - away avg).
    Games with any NaN feature (team has no prior game history) are dropped.
    """
    avg_cols = [f"avg_{c}" for c in TEAM_STAT_COLS]
    lookup   = log[["Season", "Week", "Team"] + avg_cols]

    home_lkp = lookup.rename(columns={
        "Team": "HomeTeam",
        **{f"avg_{c}": f"home_avg_{c}" for c in TEAM_STAT_COLS},
    })
    away_lkp = lookup.rename(columns={
        "Team": "AwayTeam",
        **{f"avg_{c}": f"away_avg_{c}" for c in TEAM_STAT_COLS},
    })

    games = data.merge(home_lkp, on=["Season", "Week", "HomeTeam"], how="left")
    games = games.merge(away_lkp, on=["Season", "Week", "AwayTeam"], how="left")

    for c in TEAM_STAT_COLS:
        games[f"diff_{c}"] = games[f"home_avg_{c}"] - games[f"away_avg_{c}"]

    all_feature_cols = (
        [f"home_avg_{c}" for c in TEAM_STAT_COLS] +
        [f"away_avg_{c}" for c in TEAM_STAT_COLS] +
        [f"diff_{c}"     for c in TEAM_STAT_COLS]
    )
    return games.dropna(subset=all_feature_cols).reset_index(drop=True)


# ── 5. Evaluation helper ──────────────────────────────────────────────────────

def evaluate(name, model, X_test, y_test, scaler=None):
    X_in   = scaler.transform(X_test) if scaler else X_test
    y_pred = model.predict(X_in)
    y_prob = (model.predict_proba(X_in)[:, 1]
              if hasattr(model, "predict_proba") else None)

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob) if y_prob is not None else float("nan")

    print(f"\n{'─'*55}")
    print(f"  {name}")
    print(f"{'─'*55}")
    print(f"  Accuracy : {acc:.4f}   AUC-ROC : {auc:.4f}")
    print(classification_report(y_test, y_pred,
                                target_names=["Away Win", "Home Win"],
                                zero_division=0))
    return acc, auc, y_pred, y_prob


# ── 6. Plotting helpers ───────────────────────────────────────────────────────

def plot_class_balance(data, ax):
    counts = data["home_win"].value_counts().sort_index()
    bars = ax.bar(["Away Win", "Home Win"], counts.values,
                  color=["#e74c3c", "#2ecc71"], edgecolor="black")
    for bar, v in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 3, str(v),
                ha="center", va="bottom", fontweight="bold")
    ax.set_title("Class Balance (2010-2019)", fontsize=12)
    ax.set_ylabel("Number of Games")


def plot_diff_feature_distributions(games, ax):
    diff_cols = ["diff_rush_yards", "diff_pass_yards", "diff_turnovers"]
    colors    = ["#3498db", "#e67e22", "#9b59b6"]
    for col, color in zip(diff_cols, colors):
        wins = games.loc[games["home_win"] == 1, col]
        loss = games.loc[games["home_win"] == 0, col]
        ax.hist(wins, bins=25, alpha=0.5, color=color, label=f"{col} (win)")
        ax.hist(loss, bins=25, alpha=0.3, color=color, histtype="step",
                linewidth=1.5, label=f"{col} (loss)")
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_title("Differential Feature Distributions", fontsize=12)
    ax.set_xlabel("Home Avg − Away Avg")
    ax.legend(fontsize=7, ncol=2)


def plot_correlation(games, feature_cols, ax):
    cols = feature_cols + ["home_win"]
    corr = games[cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, ax=ax, linewidths=0.4, annot_kws={"size": 7})
    ax.set_title("Q2 Feature Correlation Matrix", fontsize=12)
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.tick_params(axis="y", labelsize=7)


def plot_confusion(cm, title, ax):
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Away Win", "Home Win"],
                yticklabels=["Away Win", "Home Win"])
    ax.set_title(title, fontsize=10)
    ax.set_ylabel("Actual", fontsize=9)
    ax.set_xlabel("Predicted", fontsize=9)


def plot_roc(results, ax):
    colors = ["#3498db", "#2ecc71", "#e67e22", "#9b59b6"]
    for (name, y_t, y_p), color in zip(results, colors):
        if y_p is None:
            continue
        fpr, tpr, _ = roc_curve(y_t, y_p)
        auc = roc_auc_score(y_t, y_p)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models", fontsize=12)
    ax.legend(loc="lower right", fontsize=8)


def plot_model_comparison(summary, ax):
    names = [r[0] for r in summary]
    accs  = [r[1] for r in summary]
    aucs  = [r[2] for r in summary]
    x     = np.arange(len(names))
    w     = 0.35
    b1 = ax.bar(x - w/2, accs, w, label="Accuracy", color="#2980b9", edgecolor="black")
    b2 = ax.bar(x + w/2, aucs, w, label="AUC-ROC",  color="#e74c3c", edgecolor="black")
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8, rotation=10)
    ax.set_ylim(0, 1.1)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_title("Model Comparison: Accuracy & AUC-ROC", fontsize=12)
    ax.legend(fontsize=9)


def plot_xgb_importance(model, feature_names, ax):
    imps = model.feature_importances_
    idx  = np.argsort(imps)
    ax.barh(np.array(feature_names)[idx], imps[idx],
            color="#9b59b6", edgecolor="black")
    ax.set_title("XGBoost Feature Importances (Q2)", fontsize=12)
    ax.set_xlabel("Importance Score")
    ax.tick_params(axis="y", labelsize=8)


# ── 7. Main pipeline ──────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  NFL Win/Loss Prediction — Revised (2010-2019)")
    print("=" * 65)

    # ── Load and build leakage-free features ─────────────────────────────────
    raw    = load_raw(ALL_YEARS)
    log    = build_team_log(raw)
    log    = add_rolling_averages(log)
    games  = build_game_features(raw, log)

    home_rate = games["home_win"].mean()
    print(f"\nDataset after leakage-free preprocessing: {len(games)} games")
    print(f"  Home wins: {home_rate:.1%}  |  Away wins: {1-home_rate:.1%}")
    print(f"  Games dropped (no prior history): {len(raw) - len(games)}")

    # ── Temporal train / test split ───────────────────────────────────────────
    train = games[games["Season"] <  TRAIN_CUTOFF]
    test  = games[games["Season"] >= TRAIN_CUTOFF]
    print(f"\nTrain: {len(train)} games (2010-2016)  |  Test: {len(test)} games (2017-2019)")

    y_train, y_test = train["home_win"], test["home_win"]

    # ── Q1a: Logistic Regression — offensive stats only ──────────────────────
    print("\n" + "=" * 65)
    print("  QUESTION 1a — Offensive Stats Only (Logistic Regression)")
    print("=" * 65)
    print("  LR gives interpretable coefficients showing which offensive")
    print("  rolling averages drive win probability.")

    scaler_q1 = StandardScaler()
    Xtr_q1    = scaler_q1.fit_transform(train[Q1_FEATURES])
    lr_q1     = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
    lr_q1.fit(Xtr_q1, y_train)

    acc_lr_q1, auc_lr_q1, pred_lr_q1, prob_lr_q1 = evaluate(
        "LR — Offensive Only (Q1)", lr_q1, test[Q1_FEATURES], y_test, scaler_q1
    )

    coef_df = pd.DataFrame({
        "Feature":     Q1_FEATURES,
        "Coefficient": lr_q1.coef_[0],
    }).sort_values("Coefficient", ascending=False)
    print("\n  Logistic Regression Coefficients (Q1):")
    print(coef_df.to_string(index=False))

    # ── Q1b: XGBoost — offensive stats only ──────────────────────────────────
    print("\n" + "=" * 65)
    print("  QUESTION 1b — Offensive Stats Only (XGBoost)")
    print("=" * 65)
    print("  XGBoost on the same Q1 feature set reveals whether non-linear")
    print("  interactions between offensive stats improve predictions.")

    xgb_q1 = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=3, gamma=0.1,
        eval_metric="logloss", random_state=RANDOM_STATE, verbosity=0,
    )
    xgb_q1.fit(train[Q1_FEATURES], y_train)

    acc_xgb_q1, auc_xgb_q1, pred_xgb_q1, prob_xgb_q1 = evaluate(
        "XGBoost — Offensive Only (Q1)", xgb_q1, test[Q1_FEATURES], y_test
    )

    # ── Q2a: Logistic Regression — all features ───────────────────────────────
    print("\n" + "=" * 65)
    print("  QUESTION 2a — All Features (Logistic Regression)")
    print("=" * 65)
    print("  Extending Q1 LR with turnovers + defensive stats isolates")
    print("  exactly how much those features improve a linear model.")

    scaler_q2 = StandardScaler()
    Xtr_q2    = scaler_q2.fit_transform(train[Q2_FEATURES])
    lr_q2     = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
    lr_q2.fit(Xtr_q2, y_train)

    acc_lr_q2, auc_lr_q2, pred_lr_q2, prob_lr_q2 = evaluate(
        "LR — All Features (Q2)", lr_q2, test[Q2_FEATURES], y_test, scaler_q2
    )

    # ── Q2b: XGBoost — all features ───────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  QUESTION 2b — All Features (XGBoost)")
    print("=" * 65)
    print("  XGBoost on the full Q2 feature set captures non-linear turnover")
    print("  interactions and defensive matchup effects.")

    xgb_q2 = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=3, gamma=0.1,
        eval_metric="logloss", random_state=RANDOM_STATE, verbosity=0,
    )
    xgb_q2.fit(train[Q2_FEATURES], y_train)

    acc_xgb_q2, auc_xgb_q2, pred_xgb_q2, prob_xgb_q2 = evaluate(
        "XGBoost — All Features (Q2)", xgb_q2, test[Q2_FEATURES], y_test
    )

    # ── Summary & conclusions ─────────────────────────────────────────────────
    summary = [
        ("LR  Q1 (off. only)",  acc_lr_q1,  auc_lr_q1),
        ("XGB Q1 (off. only)",  acc_xgb_q1, auc_xgb_q1),
        ("LR  Q2 (all feats)",  acc_lr_q2,  auc_lr_q2),
        ("XGB Q2 (all feats)",  acc_xgb_q2, auc_xgb_q2),
    ]

    print("\n" + "=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    print(f"  {'Model':<26} {'Accuracy':>10} {'AUC-ROC':>10}")
    print(f"  {'─'*26} {'─'*10} {'─'*10}")
    for name, acc, auc in summary:
        print(f"  {name:<26} {acc:>10.4f} {auc:>10.4f}")

    # Q1 comparison: does XGBoost beat LR on offensive stats alone?
    q1_xgb_lift = acc_xgb_q1 - acc_lr_q1
    # Q2 comparison: does adding turnovers/defense help each algorithm?
    lr_to_lift  = acc_lr_q2  - acc_lr_q1
    xgb_to_lift = acc_xgb_q2 - acc_xgb_q1
    # Q2 comparison: does XGBoost beat LR on the full feature set?
    q2_xgb_lift = acc_xgb_q2 - acc_lr_q2

    print(f"\n  Q1 XGBoost vs LR (same features) : {q1_xgb_lift:+.4f}")
    print(f"  Turnover/defense lift — LR       : {lr_to_lift:+.4f}")
    print(f"  Turnover/defense lift — XGBoost  : {xgb_to_lift:+.4f}")
    print(f"  Q2 XGBoost vs LR (same features) : {q2_xgb_lift:+.4f}")

    print("\n  Conclusions:")
    best_q1 = "XGBoost" if acc_xgb_q1 >= acc_lr_q1 else "Logistic Regression"
    best_q2 = "XGBoost" if acc_xgb_q2 >= acc_lr_q2 else "Logistic Regression"
    print(f"  Q1: {best_q1} performs better on offensive-only features "
          f"({max(acc_lr_q1, acc_xgb_q1):.1%} accuracy).")
    print(f"  Q2: {best_q2} performs better on the full feature set "
          f"({max(acc_lr_q2, acc_xgb_q2):.1%} accuracy).")
    if lr_to_lift > 0.01 or xgb_to_lift > 0.01:
        print(f"  Adding turnovers + defense improves both models "
              f"(LR: {lr_to_lift:+.1%}, XGB: {xgb_to_lift:+.1%}).")
    else:
        print("  Turnover/defensive features provide only modest improvement.")

    # ── Visualizations ────────────────────────────────────────────────────────
    # Layout: 4 columns, 4 rows
    #   Row 0: [class balance][diff distributions][correlation x2]
    #   Row 1: [LR Q1 CM][XGB Q1 CM][LR Q2 CM][XGB Q2 CM]
    #   Row 2: [ROC curves x3][model comparison]
    #   Row 3: [XGB Q1 importance x2][XGB Q2 importance x2]

    fig = plt.figure(figsize=(22, 22))
    fig.suptitle("NFL Win/Loss Prediction — LR & XGBoost on Q1 and Q2 (2010-2019)",
                 fontsize=15, fontweight="bold", y=0.99)
    gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.55, wspace=0.4)

    ax_bal   = fig.add_subplot(gs[0, 0])
    ax_dist  = fig.add_subplot(gs[0, 1])
    ax_corr  = fig.add_subplot(gs[0, 2:4])

    ax_cm_lr_q1  = fig.add_subplot(gs[1, 0])
    ax_cm_xgb_q1 = fig.add_subplot(gs[1, 1])
    ax_cm_lr_q2  = fig.add_subplot(gs[1, 2])
    ax_cm_xgb_q2 = fig.add_subplot(gs[1, 3])

    ax_roc  = fig.add_subplot(gs[2, 0:3])
    ax_comp = fig.add_subplot(gs[2, 3])

    ax_imp_q1 = fig.add_subplot(gs[3, 0:2])
    ax_imp_q2 = fig.add_subplot(gs[3, 2:4])

    plot_class_balance(games, ax_bal)
    plot_diff_feature_distributions(games, ax_dist)
    plot_correlation(games, Q2_FEATURES[:9], ax_corr)

    plot_confusion(confusion_matrix(y_test, pred_lr_q1),
                   "LR — Offensive Only (Q1)",  ax_cm_lr_q1)
    plot_confusion(confusion_matrix(y_test, pred_xgb_q1),
                   "XGB — Offensive Only (Q1)", ax_cm_xgb_q1)
    plot_confusion(confusion_matrix(y_test, pred_lr_q2),
                   "LR — All Features (Q2)",    ax_cm_lr_q2)
    plot_confusion(confusion_matrix(y_test, pred_xgb_q2),
                   "XGB — All Features (Q2)",   ax_cm_xgb_q2)

    roc_data = [
        ("LR  Q1 (off. only)",  y_test, prob_lr_q1),
        ("XGB Q1 (off. only)",  y_test, prob_xgb_q1),
        ("LR  Q2 (all feats)",  y_test, prob_lr_q2),
        ("XGB Q2 (all feats)",  y_test, prob_xgb_q2),
    ]
    plot_roc(roc_data, ax_roc)
    plot_model_comparison(summary, ax_comp)

    plot_xgb_importance(xgb_q1, Q1_FEATURES, ax_imp_q1)
    ax_imp_q1.set_title("XGBoost Feature Importances — Q1 (Offensive Only)", fontsize=11)

    plot_xgb_importance(xgb_q2, Q2_FEATURES, ax_imp_q2)
    ax_imp_q2.set_title("XGBoost Feature Importances — Q2 (All Features)", fontsize=11)

    out_path = os.path.join(os.path.dirname(__file__), "results.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n  Saved visualizations → {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
