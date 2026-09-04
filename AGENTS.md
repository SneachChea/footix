# footix

Football analytics and prediction framework.

## Package identity

- PyPI package: **pyfootix**; import package: **footix**
- Python: **>=3.12, <3.15**
- Version is defined in `pyproject.toml`.

## Commands

```sh
uv sync --all-groups
uv run pre-commit run --all-files       # Ruff format/lint + mypy
uv run pytest -v --cov=footix
uv run pytest tests/test_foo.py -v --cov=footix
uv run sphinx-build -b html -W docs/source docs/build/html
uv run sphinx-build -b linkcheck docs/source docs/build/linkcheck
uv build
```

The CI order is `lint → test`: the lint job runs the complete pre-commit
configuration (Ruff and mypy), then tests run on Python 3.12, 3.13 and 3.14
on Ubuntu, plus Python 3.12 on macOS. The documentation build is covered by
`tests/test_docs.py`.

## Architecture

- **`footix/models/`** — prediction and score models:
  - `basic_poisson.py`: `PoissonModel`, SciPy maximum-likelihood model returning
    a `GoalMatrix`.
  - `bayesian.py`: `BayesianModel`, PyMC hierarchical goal model with posterior
    predictive matrices, 1X2/market samples, optional shots/SOT/corners inputs,
    and MCMC diagnostics.
  - `elo.py` and `team_elo.py`: chronological `EloDavidson` ratings and `EloTeam`
    state, returning `ProbaResult` for 1X2.
  - `score_matrix.py`: `GoalMatrix`, joint home/away score probabilities,
    1X2, totals, Asian handicap, double-chance, BTTS and probable-score helpers.
  - `calibration.py`: causal `OutcomeCalibrator` for out-of-sample 1X2
    temperature-and-bias calibration.
  - `utils.py`: Poisson vectors, goal expectations and bookmaker-implied Poisson
    rates.
- **`footix/evaluation/`** — strict chronological walk-forward evaluation:
  `ModelSpec`, `BacktestConfig`, `BacktestResult`, `run_backtest`, and model
  spec helpers for Poisson, Elo, Bayesian and uniform baselines. Windows are
  Friday-to-Friday, training uses only matches before the cutoff, and betting
  supports 1X2/O-U 2.5 plus flat, fractional-Kelly and portfolio staking.
- **`footix/data_io/`** — data providers and export:
  - `footballdata.py`: historical CSVs from football-data.co.uk.
  - `understat.py`: Understat results, xG, forecasts and shot data.
  - `football_data_org.py`: cached upcoming fixtures from football-data.org.
  - `apifootball_com.py`: cached upcoming Ligue 2 fixtures from APIFootball.com.
  - `base_scrapper.py`, `utils_scrapper.py`: scraper base behavior, competition
    mappings, column normalization and stable `match_id` creation.
  - `data_reader.py`: `EloDataReader` and `MatchupResult`.
  - `prediction_export.py`: prediction JSON records, score matrices,
    distributions, confidence and per-match errors.
- **`footix/strategy/`** — betting domain and allocation:
  `Bet`/`OddsInput`, edge-floor and posterior-aware selection, candidate
  diagnostics, classic/fractional/Bayesian/shrinkage Kelly, flat staking, and
  SciPy/PyTorch portfolio optimizers (`PortfolioScenarios`).
- **`footix/metrics/`** — `rps`, `log_loss`, `brier_score`, `accuracy`,
  `incertity`, `zscore`, posterior confidence metrics, and standings/form
  calculations.
- **`footix/implied_odds/`** — multiplicative, power and Shin bookmaker-margin
  normalization.
- **`footix/utils/`** — `TeamNameResolver` (static YAML + rapidfuzz + optional
  persistence), `verify_required_column`, and shared probability named tuples.
- **`footix/vizu/`** — Matplotlib visualization of a `GoalMatrix`.

The package's top-level exports are the core bet, odds, threshold and
probability types. Use the subpackage APIs for models, data providers,
evaluation and strategies.

## Data and experiments

- **`data/`** contains football-data.co.uk historical CSVs, ranking inputs and
  `data/team_name_mappings/{ligue_1,ligue_2,bundesliga_1}.yaml`.
- Provider competition keys are defined in
  `footix.data_io.utils_scrapper.MAPPING_COMPETITIONS`; use keys such as
  `FRA Ligue 1`, `FRA Ligue 2`, `ENG Premier League` and
  `DEU Bundesliga 1`, not provider-specific slugs.
- Historical outputs use snake_case columns including `date`, `home_team`,
  `away_team`, `fthg`, `ftag`, `ftr` and stable `match_id`. Upcoming fixture
  providers add timezone-aware `kickoff`, `status`, `gameweek` and
  `source_fixture_id`.
- API credentials for football-data.org and APIFootball.com must come from
  environment/configuration; never hard-code them.
- **`experimentation/`** is a frozen auto-research sandbox. Read
  `experimentation/program.md` first. Its Phase 2 market-anchored V1 is a
  causal falsification replay; `ma_policy.py`, `ma_calibrate.py`, the generated
  artifacts and guardrails are frozen. Do not edit or regenerate artifacts or
  alter results to make a strategy pass.

## Documentation

Sphinx sources live in `docs/source/`: installation and data-source guides,
Elo/Poisson/walk-forward tutorials, the prediction-export tutorial, the
`GoalMatrix` cookbook, and autodoc API pages for all `footix` subpackages.
Public code follows Google-style docstrings.

## Graphify

`graphify-out/` is generated and ignored by Git. Read
`graphify-out/GRAPH_REPORT.md` before exploring unfamiliar code; it maps the
cross-module relationships and current hubs.

Current refresh: **1,314 nodes, 2,066 edges, 95 communities**, built from
commit `0b0100f9`.

A git post-commit hook (`graphify hook install`) re-runs the AST-only rebuild
after each code commit — no manual step needed for code changes. No API key
is configured (AST-only): Python and markdown files are covered; images and
semantic edges are not re-extracted.

```sh
graphify update .       # refresh code + markdown structure, no LLM/API cost
```
