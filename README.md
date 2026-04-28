# NFL Win/Loss Prediction — Project Proposal (Revised)

## Group Members
TezJioh Knox, Nate Bamikole

---

## Dataset
**Source:** NFL Game Stats 2010–2019  
**Kaggle:** https://www.kaggle.com/datasets/davidsasser/nfl-game-stats-20102019

10 CSV files (one per season). After removing ties, the dataset contains approximately
1,530 games. Each file has 15 columns:

| Column | Description |
|---|---|
| Week | Week number within the season |
| HomeTeam / AwayTeam | Team abbreviations |
| Total | Vegas over/under line |
| H-RushAtt / A-RushAtt | Rushing attempts |
| H-RushYards / A-RushYards | Rushing yards |
| H-PassYards / A-PassYards | Passing yards |
| H-Turnover / A-Turnover | Turnovers committed |
| H-Score / A-Score | Final points scored |
| Result | Over/under outcome (1 = over, -1 = under) |

**Target Variable:** `home_win` — derived as 1 if H-Score > A-Score, else 0. Ties are
excluded because a binary classifier requires a definitive outcome.

---

## Why 10 Years Instead of 3

Using the full 2010–2019 window rather than only 3 seasons provides several benefits:

- **More training examples (~3× more games)** — reduces overfitting and gives models
  a better chance to learn stable patterns.
- **Better rolling-average stability** — cumulative team averages become reliable
  faster when there is more historical data to draw from.
- **Broader distribution of team styles and rule environments** — the dataset covers
  multiple rule-change eras (notably the increase in pass-heavy offenses), making the
  model more robust to variation.
- **Stronger test-set validation** — holding out three full seasons (2017–2019) as
  unseen test data gives a more trustworthy estimate of real-world performance.

---

## Research Questions

### Q1 — Can we predict win/loss using only prior offensive statistics?

**Algorithm: Logistic Regression**

Logistic Regression is the right choice for Q1 because the goal is to establish a
*baseline* and understand *which offensive features carry signal*. The model's
coefficients are directly interpretable: a large positive coefficient for
`diff_pass_yards` means that when the home team historically outgains opponents
through the air, they are more likely to win. This transparency directly answers
whether offensive history has predictive power.

### Q2 — Does adding turnovers (and defensive stats) meaningfully improve prediction?

**Algorithms: Logistic Regression (extended) + XGBoost**

To isolate the turnover effect fairly, we first extend the Q1 Logistic Regression
with turnover and defensive features so the comparison is apples-to-apples (same
algorithm, different feature set). We then train an XGBoost model on the same
extended feature set.

**Why XGBoost for Q2?**  
Turnovers likely have non-linear, context-dependent effects — one turnover in a
close game may be decisive while the same turnover in a blowout is not. Gradient
boosted trees can capture these interaction effects naturally, whereas Logistic
Regression assumes a linear relationship between each feature and the log-odds of
winning. Comparing LR and XGBoost on the same features tells us whether that
non-linearity is large enough to matter in practice.

**Why different algorithms for each question?**  
Q1 prioritizes interpretability to confirm whether offensive stats have predictive
power at all. Q2 adds predictive capacity — we need a model that can express the
complex, conditional role of turnovers. Using LR for Q1 and LR+XGBoost for Q2 lets
us build understanding incrementally rather than jumping straight to a black-box model.

---

## Methodology

### Step 1 — Build a Per-Team Game Log (Leakage-Free)

The raw data is game-centric (one row = one game). The first preprocessing step
converts it to a *team-centric* long format (one row = one team's performance in one
game). Each entry records the team's own offensive stats **and** the opponent's
offensive stats (which become the team's defensive "yards allowed" metrics).

### Step 2 — Compute Expanding Rolling Averages

For each team, we compute the cumulative average of every stat across all games
played **to date**, then shift by one position. Concretely, the feature used to
predict Game N for a team is the average of Games 1 through N-1. This **eliminates
data leakage** — the model never sees the stats from the game it is trying to predict.

Games where either team has no prior history (Week 1 of a team's first recorded
season) are dropped, since there is no prior evidence to average.

### Step 3 — Compute Differential Features

For each game, we subtract the away team's rolling averages from the home team's
rolling averages, producing *relative strength* features. See the Feature Engineering
section for the full list.

### Step 4 — Temporal Train / Test Split

We preserve strict chronological order:

| Split | Seasons | Approximate Games |
|---|---|---|
| **Train** | 2010–2016 | ~1,050 (70%) |
| **Test** | 2017–2019 | ~480 (30%) |

No future data is ever used to train. This mirrors real-world deployment: a model
trained on past seasons is used to predict upcoming games. Cross-validation within
the training set, if used, should also respect time ordering (walk-forward CV).

---

## Feature Engineering

### Why Differential Features?

Raw rolling averages tell us a team's historical strength in isolation. But a game
is won by the *better team on the day*. A team averaging 300 passing yards/game is
only dominant if their opponent allows fewer than that. **Differential features
capture the relative matchup advantage**, which is closer to the question the model
is actually answering.

### Feature Inventory

| Feature | Formula | Interpretation |
|---|---|---|
| `diff_rush_yards` | home_avg − away_avg | Home rushing advantage |
| `diff_pass_yards` | home_avg − away_avg | Home passing advantage |
| `diff_rush_att` | home_avg − away_avg | Rushing volume tendency |
| `diff_turnovers` | home_avg − away_avg | Turnover tendency (negative = home team safer) |
| `diff_def_rush_yards` | home_avg_allowed − away_avg_allowed | Defensive run-stop edge |
| `diff_def_pass_yards` | home_avg_allowed − away_avg_allowed | Defensive coverage edge |
| `home_avg_rush_yards` | Home team rolling avg | Absolute home rushing strength |
| `home_avg_pass_yards` | Home team rolling avg | Absolute home passing strength |
| `home_avg_turnovers` | Home team rolling avg | Home turnover risk |
| `away_avg_*` | Away team rolling avg | Mirrors for away team |

### Feature Sets Per Question

- **Q1 (offensive only, no turnovers):**  
  `diff_rush_yards`, `diff_pass_yards`, `diff_rush_att`,  
  `home_avg_rush_yards`, `home_avg_pass_yards`,  
  `away_avg_rush_yards`, `away_avg_pass_yards`

- **Q2 (offensive + turnovers + defensive):**  
  All Q1 features plus `diff_turnovers`, `diff_def_rush_yards`, `diff_def_pass_yards`,  
  `home_avg_turnovers`, `away_avg_turnovers`

---

## Remaining Weaknesses and Risks

1. **Home-field advantage is implicit, not modeled.** The label is always
   `home_win`, so the model does learn some home/away asymmetry through patterns in
   the data, but a team's historical home-vs-away record is not explicitly provided
   as a feature.

2. **Injuries and roster changes are not captured.** Rolling averages reflect a
   team's average performance over months or years, not the quality of the specific
   roster available on game day.

3. **Strength of schedule is ignored.** A team posting high passing yard averages
   against weak defenses will look stronger than they actually are. Opponent-adjusted
   stats would improve accuracy but require more complex feature engineering.

4. **Early-season noise.** In Weeks 1–3, rolling averages are based on very few
   games (or only the prior season) and may be unreliable. A minimum-games threshold
   or prior-season weighting could reduce noise.

5. **Decade-long distribution shift.** The NFL became significantly more
   pass-heavy between 2010 and 2019. Older stats may be systematically different from
   modern ones, potentially misleading the model.

6. **Mild class imbalance (~56% home wins).** We monitor precision and recall for
   both classes separately to ensure the model is not simply predicting the majority
   class.
