# Notebooks

The notebooks in this folder are the runnable companions to the Sphinx
tutorials (`docs/source/tutorials/`). They are executed manually before a
release — never in CI — and their saved outputs are what readers see when
they open them on GitHub.

## Prerequisites

```sh
uv sync --all-groups
```

## Running a notebook

```sh
uv run jupyter execute notebooks/elo.ipynb
uv run jupyter execute notebooks/poisson.ipynb
```

Run them from anywhere in the repository — the kernel runs with the notebook
directory as its working directory. The notebooks store downloaded data under
`notebooks/data/` (git-ignored); the first run therefore needs network access
to football-data.co.uk, later runs reuse the cached CSV.

## Status

| Notebook              | Tutorial                    | Notes                             |
| --------------------- | --------------------------- | --------------------------------- |
| `elo.ipynb`           | `docs/source/tutorials/elo.rst`     | Deterministic (Elo fitting).      |
| `poisson.ipynb`       | `docs/source/tutorials/poisson.rst` | Deterministic (scipy optimize).   |
| `bayesian.ipynb`      | deferred                     | MCMC: slow, not yet reproducible. |

Do not commit notebook outputs containing credentials, live API responses,
or absolute local paths.
