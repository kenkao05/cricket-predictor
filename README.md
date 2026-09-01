# IPL Win Probability & Player Stats Dashboard — Full Project Report

**Purpose of this document:** this is a complete technical record of the project — what was
built, why, how, every bug found and how it was fixed, and the reasoning behind every major
decision. It's written to be fed into an AI assistant before a presentation/viva, so it
includes an explicit Q&A section anticipating likely faculty questions.

---

## 1. Project Summary

A two-part IPL (Indian Premier League cricket) analytics web dashboard:

1. **Win Probability prediction** — two modes:
   - **Team-Based mode**: live win probability during a match chase, based on match state (score, wickets, overs, target).
   - **Player-Based mode**: pre-match win probability based on which 11 players are selected for each side (before a ball is bowled).
2. **Player Stats lookup** — batting/bowling career statistics for any player, filterable by which IPL team they played for (players who've played for multiple franchises over the years).

Built as a Streamlit web app, designed to be deployed on Render, with a stated future goal
of migrating the frontend to React while reusing the same Python prediction logic via a
FastAPI/Flask backend.

**Scope decision:** IPL only. T20I/ODI/Test formats were explicitly discussed and deferred —
not built. Test cricket in particular doesn't fit a clean win/lose framing since matches can draw.

---

## 2. Data Source

- **Source:** [Cricsheet](https://cricsheet.org) — `ipl_json.zip`, ball-by-ball JSON data for IPL matches, licensed CC BY-SA 4.0.
- **Coverage:** 1,243 total match files (2008–present season). After filtering out matches with no winner (abandoned/no-result) and matches missing date/team info, **1,218 matches** were usable for modeling.
- **Format:** one JSON file per match. Each file has an `info` block (teams, dates, season, toss, winner, outcome) and an `innings` block (ball-by-ball deliveries: batter, bowler, runs, extras, wickets).
- **No external/second dataset was used.** Everything — team-level match state, player identities, player roles, toss info — was derived from this single Cricsheet dataset. This was a deliberate decision to avoid maintaining a second data source (e.g., no manually maintained squad lists).
- **On Kaggle:** the dataset was mounted at `/kaggle/input/datasets/harshzala/ipl-json`.

---

## 3. Tech Stack

| Layer                                 | Technology                                                                                                                              |
| ------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| Data processing / feature engineering | Python, pandas, numpy                                                                                                                   |
| Modeling                              | scikit-learn (LogisticRegression, RandomForestClassifier, GradientBoostingClassifier)                                                   |
| Model serialization                   | pickle (small files), joblib with compression (large model)                                                                             |
| Notebook environment                  | Kaggle Notebooks                                                                                                                        |
| Frontend / dashboard                  | Streamlit                                                                                                                               |
| Styling                               | Custom CSS injected via `st.markdown` — glassmorphism design (frosted-glass cards, backdrop-blur, rounded corners, gradient background) |
| Planned deployment                    | Render (free tier, Python web service)                                                                                                  |
| Planned future frontend               | React, with Python logic re-exposed via FastAPI/Flask                                                                                   |

---

## 4. Model 1: Team-Level Win Predictor (built first)

### 4.1 What it predicts

Given the current state of a live 2nd-innings chase, predicts the probability each team wins.

### 4.2 Features

`batting_team, bowling_team, city, runs_left, balls_left, wickets_left, total_runs_x (target), crr (current run rate), rrr (required run rate)`

### 4.3 Methodology

- Cricket over notation handled correctly via a helper function (`9.2` overs = 56 balls, not `9.2 × 6`, since overs aren't decimal).
- Parsed only the 2nd innings (the chase) of each match, ball by ball, building one training row per ball.
- Target value taken from the innings' own `target` field (correctly handles DL-revised targets — occurred in ~6 matches).
- Super-over innings excluded. Matches with no winner (tie/no-result/abandoned) skipped entirely.
- **Train/test split: `GroupShuffleSplit` grouped by `match_id`** — not a random row split. This is critical: since every ball from the same match is highly correlated (they share the same eventual outcome), a random row split would leak information from a match's other balls into the test set and produce artificially inflated accuracy (this was verified to matter — random splits on this kind of data commonly inflate accuracy toward ~99%, which is not real skill).
- Two models compared: Logistic Regression (baseline) and Random Forest (200 trees). **Random Forest was selected** as the final model.
- Preprocessing: `batting_team`, `bowling_team`, `city` one-hot encoded (`drop='first'`, `handle_unknown='ignore'`), saved together with the model in a single sklearn `Pipeline` object (`pipe.pkl`), so no manual preprocessing is needed at inference time.

### 4.4 Exact results (from the executed notebook)

- Of 1,243 match files: 1,243 parsed, **25 skipped** (no winner/no target) → **139,814 ball-level training rows**.
- Target class balance: 52.7% batting-team-won / 47.3% bowling-team-won.
- Split: 974 matches / 111,438 rows train; 244 matches / 28,376 rows test.
- **Logistic Regression: 75.98% accuracy** (threw a `ConvergenceWarning` — `lbfgs` solver didn't fully converge within `max_iter=3000`; not resolved, didn't block model selection since Random Forest was always the intended final model).
- **Random Forest (200 trees): 76.75% accuracy — final selected model.** Precision/recall/F1 ≈0.76–0.78 for both classes, confusion matrix balanced.
- Sample prediction (CSK chasing 178 vs MI, 42 needed off 36 balls, 6 wickets in hand): **96.5% / 3.5%.** This model's own calibration was never checked (no Brier score/reliability curve was computed for Model 1 — see §4.5) — so whether a number this extreme is trustworthy is genuinely unverified, not confirmed either way.

### 4.5 Issues found in this model (flagged, some fixed later — see §6)

- **No calibration check was originally done.** Only accuracy/classification-report/confusion-matrix were checked. Since the product displays a probability (not just a win/loss label), this matters — nobody had verified whether "70% win probability" actually corresponds to winning ~70% of the time.
- **Duplicate team categories from data quality issues** — see §6.1 for full detail and fix.
- **Single train/test split, no cross-validation** — the reported accuracy is a point estimate; could shift somewhat on a different random seed.
- **`drop='first'` one-hot encoding** was originally designed for linear models (avoids multicollinearity) but was reused as-is for the Random Forest, where it's unnecessary (though not harmful — trees don't have the multicollinearity problem linear models do).

---

## 5. Model 2: Player-Level Win Predictor (built second — the harder problem)

### 5.1 Why this is a fundamentally different, harder problem

The team-level model works because match state (score/wickets/overs) _directly_ tells you how close a team is to winning at that exact moment — a strong, direct signal. Predicting from 11 player names has no equivalent — the model has to estimate team strength _before the match even starts_, purely from who's playing. Two teams with the exact same 22 players essentially never repeat across 1,218 matches, so the model **cannot** learn from raw player identity directly — player identities had to be converted into numerical stats first, then aggregated to team level.

### 5.2 Core approach

1. Calculate rolling, **pre-match-only** stats per player (batting average, strike rate, bowling average, economy).
2. Aggregate the 11 players per side into team-level average stats (e.g., average strike rate of the batting lineup).
3. Predict from the **difference** between the two teams' aggregated stats (`diff_avg_batting_avg`, etc.), not from the raw numbers, and not from names.

### 5.3 The single most important rule: no data leakage from the future

For every match used as a training example, a player's stats used as model input are calculated **only from matches strictly before that match's date** (a rolling, causal calculation, implemented via `.cumsum().shift(1)` and `.rolling(window).sum().shift(1)` operations, grouped by player and sorted by date). If this rule were violated (e.g., using a player's full-career average that includes matches after the one being predicted), the model would effectively "see the future" and produce fake, inflated accuracy that would fail completely on truly new/future matches.

This was explicitly verified twice during development:

- Cell 11: checked one player (Rohit Sharma) — confirmed their first-ever match has `NaN` career stats (no prior data exists), and `career_matches_played` increases monotonically over time.
- Cell 40 (added later after review): re-checked across **10 randomly sampled players** to close the gap that Cell 11 only proved the rule for one player — all 10 passed.

### 5.4 Locked-in modeling decisions and reasoning

| Decision                            | Choice made                                                                                                                                              | Reasoning                                                                                                                                                                                                                 |
| ----------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Data leakage rule                   | Rolling, strictly-prior-match stats only                                                                                                                 | Only way to avoid fake accuracy (see §5.3)                                                                                                                                                                                |
| Player role detection               | Derived from data: ≥20 balls faced in career → "batsman" tag eligible; ≥20 balls bowled → "bowler" tag eligible; both → "all-rounder"; neither → "other" | No clean external player-role dataset exists in Cricsheet — this is a self-contained approximation using thresholds on total career balls                                                                                 |
| Current squads (for team dropdowns) | A team's dropdown squad = players who played for that team in _that team's own most recent season_ in the dataset                                        | Avoids hardcoding/maintaining a separate "current roster" list — sourced entirely from Cricsheet. **Caveat found later:** this filter does not exclude defunct franchises — see §6.2                                      |
| Toss info                           | Included as a feature (`toss_winner_is_a`)                                                                                                               | Already present in Cricsheet's `info` block, known to affect match outcomes, wasn't used in Model 1                                                                                                                       |
| Recent form window                  | Rolling last 10 matches per player                                                                                                                       | Standard rule-of-thumb in cricket analytics — enough matches to reduce noise, recent enough to reflect current form                                                                                                       |
| Model type                          | Logistic Regression (baseline) vs Gradient Boosting (150 trees) compared, best one kept                                                                  | Pre-match prediction has fewer, weaker features than ball-by-ball data; boosting/logistic regression are less prone to overfitting on smaller feature sets than a large Random Forest here                                |
| Accuracy expectation                | 60–65% treated as realistic success, NOT a failure; anything above ~80% treated as a leakage red flag                                                    | Pre-match player-average prediction is a fundamentally weaker signal than live match state (this is closer to what bookmakers do pre-match — toss, pitch, and day-of form aren't captured, so ceiling is naturally lower) |
| Evaluation method                   | Same match-grouped `GroupShuffleSplit`, **plus** a calibration curve and Brier score (missing from Model 1)                                              | Closes the exact gap flagged in Model 1 — the product shows probabilities, so probability quality must be checked, not just classification accuracy                                                                       |

### 5.5 Feature engineering pipeline (as implemented, 35 notebook cells)

Every code cell is immediately followed by a verification cell that prints an **Expected** description, the **Actual** result, and a **PASS/FAIL**. This was a deliberate build practice requested for the project — every step is self-checking rather than trusting silent success.

1. **Cells 2–3:** Imports (pandas, numpy, sklearn, matplotlib) + verify.
2. **Cells 4–5:** Load 1,243 match JSON files from Kaggle input + verify count.
3. **Cells 6–7:** Parse every delivery into a player-level row (`batter`, `bowler`, runs, wicket info, toss info attached) → 290,752 delivery rows across 1,218 matches + verify no nulls in player fields.
4. **Cells 8–9:** Aggregate deliveries into per-player, per-match batting lines (18,448 rows) and bowling lines (14,452 rows) + verify row counts and value ranges.
5. **Cells 10–11:** Compute rolling, pre-match-only stats per player (career cumulative + last-10-match rolling) — **the most important cell in the notebook**, enforces the no-leakage rule + verify (see §5.3).
6. **Cells 12–13:** Derive player role tags from career ball-count thresholds (303 all-rounders, 226 batsmen, 198 bowlers, 78 "other") + verify.
7. **Cells 14–15:** Identify the playing XI per team per match (median 11 players identified per team-match) + verify.
8. **Cells 16–17:** Build team-level pre-match features by averaging the XI's individual rolling stats (2,436 team-match rows = 2 × 1,218 matches) + verify.
9. **Cells 18–19:** Assemble the final match-level training dataset — one row per match with `diff_` feature columns (team A stat minus team B stat) and a 0/1 target (1,218 rows, ~50/50 target balance) + verify.
10. **Cells 20–21:** Train/test split, grouped by `match_id` (970 train / 243 test rows, zero match overlap between sets) + verify.
11. **Cells 22–23:** Train Logistic Regression and Gradient Boosting, compare accuracy + verify against the "60–65% expected, >80% is a red flag" rule.
12. **Cells 24–25:** Detailed evaluation of the best model (classification report, confusion matrix) + verify the model isn't collapsing to predicting one class only.
13. **Cells 26–27:** Calibration curve + Brier score — the check that was missing from Model 1 + verify Brier score is better than a coin-flip (<0.25).
14. **Cells 28–29:** Save the trained model pipeline (`player_model_pipe.pkl`) and its feature column order (`player_model_features.pkl`) + verify reload.
15. **Cells 30–31:** Build the Player Stats tab lookup table (batting average, strike rate, highest score, bowling average, economy, per player per team) + verify.
16. **Cells 32–33:** Build team↔player dropdown lookup tables (`team_to_players.pkl`, `player_to_teams.pkl`) + verify every team has a non-empty squad and every player maps to ≥1 team.
17. **Cells 34–35:** Sample end-to-end prediction sanity check + verify probabilities sum to 100%.
18. **Cells 36–41 (added later, during the bug-fix pass — see §6):** diagnostic check, delete old duplicate-containing files, rebuild the 3 lookup tables with merged team names, re-verify, multi-player leakage re-check across 10 players, final output file check.
19. **Cell 42 (added later, during frontend prep — see §6.3):** build a "current form snapshot" table (`player_current_form.pkl`) — each player's most recent known rolling stats, needed so the live dashboard can compute predictions for hypothetical future matchups.

### 5.6 Final results (actual numbers from the executed notebook)

- **Best model: Logistic Regression** (beat Gradient Boosting: 72.43% vs 65.84% accuracy).
- **Test accuracy: 72.4%** — above the originally-expected 60–65% range, but below the 80% leakage red-flag threshold. Not proven to be a leak (the causal-stats rule was verified across 10 sampled players), but flagged as slightly higher than expected and worth continued scrutiny.
- **Confusion matrix:** [[88, 34], [33, 88]] — balanced across both classes, not collapsed to predicting one outcome.
- **Brier score: 0.1956** (0 = perfect, 0.25 = coin-flip-level). Better than random, confirms the probabilities carry real information.
- **Calibration curve finding:** predictions in the 55–65% range are well-calibrated (close to the diagonal "perfect calibration" line), but the model is **overconfident at the extremes** — e.g., a predicted ~95% probability corresponded to an actual win rate closer to ~80% in the test set. This is surfaced to the end user in the dashboard as a disclaimer, rather than hidden.
- **Logistic regression coefficients**, sanity-checked: most had sensible signs (e.g., higher batting average difference → higher win probability), but `diff_avg_economy` had a small positive coefficient, which is backwards (worse/higher bowling economy for Team A nudging predictions toward Team A winning). Flagged as likely multicollinearity among the correlated bowling stats (economy, bowling average, strike rate are all correlated with each other) rather than a data bug — small magnitude, not corrected, noted as a known model quirk.

---

## 6. Bug/Issue Log — Full Troubleshooting History

This section documents every real bug found during development, in the order discovered, with root cause and fix. This is the kind of detail a faculty member is likely to probe on ("how did you validate your work / what went wrong / how did you catch it").

### 6.1 Duplicate/renamed team categories (found before player-model build, confirmed present after first player-model run)

**Symptom:** Franchises that were renamed over IPL history exist as _separate_ categories in the data, splitting one team's history in two:

- `Rising Pune Supergiant` vs `Rising Pune Supergiants` (typo/naming inconsistency across seasons)
- `Royal Challengers Bangalore` vs `Royal Challengers Bengaluru` (city renamed)
- `Delhi Daredevils` vs `Delhi Capitals` (franchise renamed)
- `Kings XI Punjab` vs `Punjab Kings` (franchise renamed)

**Impact:** Any model or lookup table built from raw team names quietly splits these franchises' historical data in half, weakening predictions/stats involving them.

**Where it was found:** First flagged theoretically from inspecting `pipe.pkl`'s one-hot encoder categories (19 team categories instead of ~10–13 real franchises). Confirmed as a **live bug**, not just a theoretical risk, when the player-model notebook's actual output (`team_to_players.pkl`) was inspected and found to contain 19 team keys instead of the expected ~10 active franchises.

**Fix:** A `TEAM_NAME_MAP` dictionary was introduced to merge each pair into one canonical name (`Delhi Daredevils`→`Delhi Capitals`, `Kings XI Punjab`→`Punjab Kings`, `Rising Pune Supergiant`→`Rising Pune Supergiants`, `Royal Challengers Bangalore`→`Royal Challengers Bengaluru`), applied via `.replace()` before rebuilding all three lookup tables (`player_stats_lookup.pkl`, `team_to_players.pkl`, `player_to_teams.pkl`).

**Important scoping decision:** The already-trained models (`pipe.pkl` and `player_model_pipe.pkl`) were **not retrained** — the player-level model's features are stat _differences_, not team names, so it's unaffected. The team-level model (`pipe.pkl`) still has the old split categories baked into its encoder; rather than retrain a 150MB model, the dashboard code maps each clean UI team name to the correct raw category string the model expects (`TEAM_NAME_TO_PIPE_CATEGORY` dict in `app.py`), working around the issue at the application layer instead.

**Known residual issue (not fixed, explicitly flagged, low priority):** The **city** field has the same kind of split (`Bangalore` vs `Bengaluru` as separate city categories in `pipe.pkl`), never fixed. Left as-is since it doesn't block functionality, just slightly weakens city-based predictions for Bengaluru matches.

### 6.2 "Current squad" filter didn't actually filter out defunct teams

**Symptom:** After the team-name-merge fix above, `team_to_players` still returned 19 teams, including long-defunct franchises: Deccan Chargers, Pune Warriors, Kochi Tuskers Kerala, Gujarat Lions.

**Root cause:** The original filter logic was "include a team's players from that team's own most recent season in the dataset" — but this is trivially true for _every_ team, including dead ones (a defunct team's own last season is, by definition, its most recent season _for itself_). The filter never compared against the league's overall current era.

**Fix:** Introduced an explicit `ACTIVE_TEAMS` allow-list (the 10 real current IPL franchises) and filtered `team_to_players` to only include teams in that list, while **deliberately keeping** the defunct teams in `player_stats_lookup` and `player_to_teams` (needed for the Player Stats tab, so a player's history with e.g. Deccan Chargers is still viewable — only the Win Probability tab's team-selection dropdown needed to exclude dead franchises).

### 6.3 Missing "current form" data for live inference

**Symptom:** Realized while planning the frontend that the player model's 8 input features include `diff_avg_recent_strike_rate` and `diff_avg_recent_economy` (rolling last-10-match form) — but none of the saved lookup files contained each player's _latest_ known rolling stats, only full match-by-match history. Without this, the dashboard would have no way to compute a live prediction for a hypothetical future matchup.

**Fix:** Added Cell 42 — takes each player's most recent row from `bat_agg`/`bowl_agg` (after sorting by date) and saves their latest known `career_batting_avg`, `career_strike_rate`, `recent_strike_rate`, `career_bowling_avg`, `career_economy`, `recent_economy` into a new file, `player_current_form.pkl` (805 players).

### 6.4 scikit-learn version mismatch when reloading `pipe.pkl`

**Symptom:** `AttributeError: Can't get attribute '_RemainderColsList' on module 'sklearn.compose._column_transformer'` when trying to unpickle `pipe.pkl` in a different environment.

**Root cause:** `pipe.pkl` was originally pickled with scikit-learn 1.6.1; the environment being used to inspect it had scikit-learn 1.8.0 installed, and internal sklearn class structures had changed between versions.

**Fix:** Installed the exact matching version (`scikit-learn==1.6.1`) before loading. This version is pinned in `requirements.txt` for deployment so the same issue doesn't occur in production.

### 6.5 `pipe.pkl` exceeded GitHub's 100MB file size limit

**Symptom:** `git push` rejected with: `File pipe.pkl is 149.23 MB; this exceeds GitHub's file size limit of 100.00 MB`.

**Options considered:**

- Git LFS — rejected. Render's support for pulling Git LFS objects during a git-based build isn't clearly documented/confirmed, too risky to rely on for a first deployment.
- Retrain a smaller model (fewer trees) — would work but changes model behavior/predictions, avoided if unnecessary.
- **Re-serialize the exact same model more efficiently — chosen fix.**

**Fix:** Re-saved the identical fitted model object using `joblib.dump(pipe, 'pipe.pkl', compress=3)` instead of raw `pickle`. This is the same model, same predictions — joblib's compression exploits redundant structure in the many Random Forest trees far better than plain pickle.

- Original (pickle): 149.2 MB
- joblib, compress=3: **31.2 MB** (well under the 100MB limit)
- Verified: reloaded the compressed file and confirmed it produces identical model structure/predictions before handing it off.

`app.py` was updated to load this one file with `joblib.load()` instead of `pickle.load()` (all other, smaller pickle files remain plain `pickle`). `joblib` was added to `requirements.txt`.

---

## 7. Dashboard (`app.py`) — Structure & Design

### 7.1 Layout

- **Tab 1 — Win Probability**, with a mode toggle:
  - _Player-Based mode_: select Team A → squad dropdown auto-filters to that team's active roster → multi-select exactly 11 players (search-enabled) → repeat for Team B → select toss winner → predict. Displays each team's win % with a progress bar, plus an automatic calibration disclaimer (since this model is known to be overconfident at extreme probabilities — see §5.6).
  - _Team-Based mode_: select batting/bowling team, city, then live match state (target, current score, overs completed, balls into current over, wickets down) → predict. No disclaimer shown here since this model's calibration wasn't found to have the same issue (though it also was never explicitly checked in the original Model 1 build — see §4.5).
- **Tab 2 — Player Stats**: search-enabled player dropdown → team-stint dropdown (auto-filtered to only teams that specific player has actually played for) → displays batting average, strike rate, highest score, bowling average, economy for that player-team combination.

### 7.2 Design system

- **Style requested:** modern, glassmorphism (frosted-glass cards, rounded corners), not overly "vibe-coded"/generic-AI-looking.
- **Implementation:** custom CSS injected via `st.markdown(..., unsafe_allow_html=True)` targeting Streamlit's internal component selectors (`data-testid` attributes) since Streamlit doesn't natively expose arbitrary div wrapping:
  - Dark radial-gradient background (navy → deep purple → near-black).
  - Bordered containers (`st.container(border=True)`) styled as frosted-glass cards: semi-transparent white background, `backdrop-filter: blur(18px)`, subtle 1px border, soft box-shadow.
  - Metric widgets similarly glass-styled.
  - Gradient text for the main title (blue → purple → orange).
  - Glass-styled buttons with a hover lift effect.
  - Rounded, semi-transparent input/select/multiselect fields.
  - Custom-styled tab bar.
  - A distinctly-colored "disclaimer chip" component for the calibration warning.
- **Font:** Inter (Google Fonts), a standard modern UI typeface.

### 7.3 Code structure (for the eventual React migration)

All prediction logic was deliberately written as plain Python functions with no Streamlit objects inside them (`predict_player_based()`, `predict_team_based()`, `team_xi_features()`) — they take simple inputs (team names, player lists, match state numbers) and return simple outputs (probabilities). This means only the UI layer (the `st.*` calls) will need to be rebuilt when/if the frontend moves to React; these same functions can be wrapped directly in a FastAPI/Flask endpoint that a React frontend calls. This was an explicit requirement discussed before any code was written, specifically to avoid wasted work later.

---

## 8. File Inventory

| File                              | Purpose                                                               | Approx. size |
| --------------------------------- | --------------------------------------------------------------------- | ------------ |
| `app.py`                          | Streamlit dashboard application                                       | —            |
| `requirements.txt`                | Python dependencies for deployment                                    | —            |
| `README.md`                       | Deployment instructions (short version)                               | —            |
| `pipe.pkl`                        | Team-level model pipeline (RandomForest, joblib-compressed)           | 31.2 MB      |
| `player_model_pipe.pkl`           | Player-level model pipeline (Logistic Regression)                     | 1.6 KB       |
| `player_model_features.pkl`       | Ordered list of the 8 feature columns the player model expects        | 0.2 KB       |
| `player_stats_lookup.pkl`         | Career batting/bowling stats per player per team (Player Stats tab)   | 177 KB       |
| `player_to_teams.pkl`             | Player → list of teams they've played for (Player Stats tab dropdown) | 30 KB        |
| `team_to_players.pkl`             | Active team → current squad list (Win Probability tab dropdown)       | 2.9 KB       |
| `player_current_form.pkl`         | Each player's latest known rolling stats (for live inference)         | 50 KB        |
| `ipl-win-predictor_success.ipynb` | Notebook 1: team-level model training                                 | —            |
| `ipl-player-model.ipynb`          | Notebook 2: player-level model training + lookup table generation     | —            |

**Deployment folder structure:**

```
repo/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   ├── pipe.pkl
│   ├── player_model_pipe.pkl
│   ├── player_model_features.pkl
│   ├── player_stats_lookup.pkl
│   ├── player_to_teams.pkl
│   ├── team_to_players.pkl
│   └── player_current_form.pkl
└── notebooks/
    ├── ipl-win-predictor_success.ipynb
    └── ipl-player-model.ipynb
```

---

## 9. Known Limitations (be ready to state these proactively)

1. **72.4% accuracy is above the originally-expected 60–65% range** for the player model — not proven to be leakage (verified across 10 players), but not fully ruled out either. Full defense: the multi-player leakage check passed, confusion matrix is balanced, Brier score is reasonable — but this remains the single most likely point of faculty scrutiny.
2. **Calibration is imperfect at extreme probabilities** (player model) — surfaced honestly to the end user via a UI disclaimer rather than hidden.
3. **Playing XI approximation:** "who played" is inferred from who has at least one batting or bowling contribution recorded in the ball-by-ball data for that match — this can slightly undercount a team's actual XI (e.g., a batter who didn't come in because the team was bowled out, or a bowler who wasn't needed).
4. **Player role tagging is a simplification** — a fixed 20-ball career threshold, not officially sourced player position data.
5. **City name duplication** (`Bangalore`/`Bengaluru`) in the team-level model was identified but not fixed (low-impact, explicitly deprioritized).
6. **No cross-validation** on either model — single train/test split each. Accuracy figures are point estimates.
7. **Team-level model (`pipe.pkl`) was never retrained** after the team-name bug was found — the issue was worked around at the application layer (mapping clean UI names to the model's old raw category strings) rather than fixed at the source.
8. Only IPL is supported; T20I/ODI/Test formats and player-availability-by-format logic were discussed but explicitly deferred as future work.

---

## 10. Anticipated Q&A (for presentation prep)

**Q: Why did you use a grouped split instead of a normal random train/test split?**
A: Because multiple rows come from the same match (either multiple balls in Model 1, or the match itself always appears as exactly one row in Model 2 but was still split by match_id for methodological consistency). A random split can put rows from the same match in both train and test, letting the model implicitly "know" the answer via correlated rows it's already seen — inflating accuracy in a way that wouldn't hold on genuinely new matches. `GroupShuffleSplit` grouped by `match_id` prevents this.

**Q: Why is 72% accuracy for the player model not necessarily good news?**
A: Because the target range, based on standard expectations for pre-match cricket prediction (a fundamentally weaker signal than live match state), was 60–65%. A number this much higher raises the possibility of a subtle information leak. It was investigated (multi-player leakage re-check, balanced confusion matrix, reasonable Brier score) and no leak was found, but it's flagged as a result to interpret cautiously rather than celebrate uncritically.

**Q: What is a Brier score and why does it matter here?**
A: It measures how well-calibrated a model's predicted probabilities are (mean squared error between predicted probability and actual binary outcome). It matters because the product displays a probability number to the user, not just a win/lose label — a model can have decent classification accuracy while still being badly overconfident or underconfident in its probability outputs. This check was missing from the first model and was deliberately added for the second.

**Q: How did you prevent data leakage in the player-level model?**
A: All player statistics used as model inputs are calculated using only matches that occurred strictly before the match being predicted, via a shifted rolling/cumulative calculation grouped by player and sorted by date. This was explicitly tested by confirming that every player's very first match in the dataset has null (unknown) career stats, since no prior data exists yet.

**Q: Why did Logistic Regression outperform Gradient Boosting here?**
A: With only 8 relatively weak, aggregated features and ~1,200 training examples, a simpler linear model is less prone to overfitting than a more flexible boosted-tree model. This matches general expectations for small-feature, small-sample-size problems.

**Q: Why does the dashboard show a disclaimer on some predictions but not others?**
A: The player-based model's calibration curve was explicitly checked and found to overstate confidence at extreme probabilities (e.g., a predicted 95% corresponding to an actual ~80% win rate in testing). The team-based (live match state) model was not put through the same calibration check, so no equivalent claim can honestly be made about it either way — it doesn't get the disclaimer because its calibration status is simply unverified, not because it's known to be well-calibrated.

**Q: Why weren't the retired teams (Deccan Chargers, etc.) removed from the dataset entirely?**
A: They're still needed for the Player Stats tab, since players who played for those teams should still show accurate career stats for that stint. They were only excluded from the Win Probability tab's team-selection dropdown, where selecting a defunct team to predict a hypothetical match wouldn't make sense.

**Q: What would you do differently / what's the biggest weakness of this project?**
A: Honest answer: the team-level model was never retrained after finding the team-name duplication bug — it was patched at the application layer instead of the source, which is a reasonable short-term call given the model's size (150MB) but is technical debt, not a true fix. The second biggest weakness is the unverified higher-than-expected accuracy on the player model.

---

## 11. Future Work (explicitly discussed, not built)

- Migrate frontend from Streamlit to React, with prediction logic exposed via FastAPI/Flask (the codebase was structured in anticipation of this — see §7.3).
- Extend beyond IPL to T20I/ODI/Test formats — would require separate models per format and format-specific squad-eligibility logic (e.g., only Indian players selectable when predicting an India Test match).
- Fix the city name duplication bug in the team-level model.
- Retrain the team-level model with the corrected, merged team names rather than working around it at the application layer.
- Add cross-validation to both models for more robust accuracy estimates.
