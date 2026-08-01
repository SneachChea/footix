<div align="center">
    <img src="img/logo_footix.png" alt="Footix Logo" width="200">
</div>

# 🐓 Footix: Smart Sports Analysis & Prediction Toolkit

[Features](#-features) • [Installation](#-installation) • [Quick Start](#-quick-start)

## 🎮 Overview

Footix is your intelligent companion for sports analysis and prediction. Leveraging advanced machine learning algorithms and comprehensive data analysis, it helps you make data-driven decisions in sports betting and analysis.

Supports Python 3.12 through 3.14. The package is published as `pyfootix` and imported as `footix`.

## ✨ Features

- 📊 **Advanced Data Analysis**
  - Import data from multiple sports databases
  - Clean and preprocess sports statistics
  - Comprehensive historical data analysis

- 🤖 **Smart Prediction Engine**
  - Machine learning-powered outcome prediction

- 💰 **Strategic Betting Tools**
  - Risk assessment algorithms
  - Bankroll management system
  - Multiple betting strategy templates

## 🚀 Installation

### CPU (default)

```bash
# uv (recommended) — CPU torch resolved automatically
uv add pyfootix

# pip — pass the PyTorch CPU index to avoid CUDA packages
pip install pyfootix --extra-index-url https://download.pytorch.org/whl/cpu
```

### GPU (Linux with CUDA)

```bash
# uv — override torch to use PyPI (includes CUDA on Linux)
uv add pyfootix --no-sources

# pip — standard PyPI install includes CUDA on Linux
pip install pyfootix
```

## 🎯 Quick Start

Build a score matrix straight from expected goals and read off the 1X2 probabilities:

```python
from footix.models.score_matrix import GoalMatrix
from footix.models.utils import poisson_proba

gm = GoalMatrix(
    home_goals_probs=poisson_proba(lambda_param=1.5, k=20),
    away_goals_probs=poisson_proba(lambda_param=1.2, k=20),
)
probas = gm.return_probas()
print(f"Home: {probas.proba_home:.2f}, Draw: {probas.proba_draw:.2f}, Away: {probas.proba_away:.2f}")
```

Want to train a model on real match data? Follow the [Elo](docs/source/tutorials/elo.rst) or
[Poisson](docs/source/tutorials/poisson.rst) tutorial.

## 📤 Exporting Predictions

You can export Bayesian predictions to JSON using:

- Core Python utilities for script/automation workflows

See the full tutorial in [docs/source/prediction_export_tutorial.rst](docs/source/prediction_export_tutorial.rst).

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.
