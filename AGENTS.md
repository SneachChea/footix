# footix

Football analytics and prediction framework.

## Package identity

- PyPI name: **pyfootix**; import package: **footix** (flat layout via `[tool.hatch.build.targets.wheel] packages = ["footix"]`)
- Python: `>=3.10, <3.15`

## Commands

```sh
uv sync --all-groups         # install everything (runtime + dev)
uv run pre-commit run --all-files  # lint + typecheck
uv run pytest -v --cov=footix    # full test suite
uv run pytest tests/test_foo.py -v --cov=footix  # single test file
uv build                      # build sdist + wheel
uv lock                       # regenerate lockfile
```

## CI pipeline order

`lint → type → test` — each job gates the next.

- **lint**: `uv run pre-commit run --all-files` (ruff-format + ruff --fix)
- **type**: `uv run pre-commit run mypy --all-files` (needs `types-requests` + `types-PyYAML` as additional_dependencies in `.pre-commit-config.yaml`)
- **test**: `uv run pytest -v --cov=footix` on py3.10, 3.11, 3.12, 3.13, 3.14

## Known test failures

`test_bayesian_calibration.py` and `test_bayesian_use_stats.py` fail with `ImportError: cannot import name 'concat' from 'arviz'`. This is a `pymc`/`arviz` version incompatibility — `pymc 5.x`'s `backends.arviz` imports `concat` which was removed in `arviz 1.x`. Skip these when verifying.

## Architecture

- **models/**: `PoissonModel` (scipy optimize), `BayesianModel` (pymc MCMC), `EloDavidson` (rating system), `GoalMatrix` (score probability matrix)
- **data_io/**: `ScrapFootballData` (football-data.co.uk CSV), `ScrapUnderstat` (understat.com), `prediction_export.py` (JSON export)
- **strategy/**: `Bet` dataclass (core domain — `edge_mean` auto-computed as `prob_mean * odds - 1`), Kelly criterion variants, portfolio optimization (torch/scipy)
- **metrics/**: `rps`, `incertity`, `zscore`, confidence curve from MCMC samples
- **implied_odds/**: multiplicative, power, and Shin normalization methods
- **utils/**: `TeamNameResolver` (static map + fuzzy + YAML persistence), `verify_required_column` decorator, `ProbaResult`/`SampleProbaResult` named tuples
- **vizu/**: `plot_goal_matrix` (matplotlib)

## Data

- `data/` — historical match CSV files keyed by `"{competition}_{season}.csv"`
- `data/team_name_mappings/` — YAML files mapping calendar names → canonical names
- `export/` and `scripts/export/` — JSON prediction exports by gameweek

## Graphify

A knowledge graph of the codebase lives in `graphify-out/` (710 nodes, 1044 edges, 48 communities). Before exploring unfamiliar code, read `graphify-out/GRAPH_REPORT.md` for community hubs and cross-module connections — it maps how models, data loaders, strategies, and metrics wire together.

## Docs

- Sphinx, `docs/source/conf.py`, Google-style docstrings (napoleon extension)
- ReadTheDocs: `.readthedocs.yaml` installs via pip (not uv) using `docs/requirements.txt`
- Build locally: `uv run sphinx-build -b html -W docs/source docs/build/html`
- `.streamlit/config.toml` exists (app runner config)
