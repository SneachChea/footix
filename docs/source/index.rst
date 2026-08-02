Welcome to **footix**'s documentation!
======================================

Footix is a Python package for football analytics and prediction. It ships
statistical models for match scores, tools for odds analysis, and betting
strategies, all on top of a small set of data providers.

Features
--------

* **Data import**
    * Historical results from football-data.co.uk and understat.com
    * Upcoming fixtures from football-data.org and APIFootball.com
    * Team name normalization between sources

* **Prediction models**
    * `EloDavidson` rating system (fast, chronological)
    * `PoissonModel` (maximum-likelihood score model)
    * `BayesianModel` (PyMC MCMC, with calibration and posterior samples)

* **Score and betting tools**
    * `GoalMatrix`: joint score distribution and market probabilities
    * Implied odds normalization (multiplicative, power, Shin)
    * Bet selection and staking strategies (flat, Kelly, portfolio)

Quick start
-----------

No data download required — build a score matrix straight from expected
goals and read off the 1X2 probabilities:

.. testcode::

   from footix.models.score_matrix import GoalMatrix
   from footix.models.utils import poisson_proba

   gm = GoalMatrix(
       home_goals_probs=poisson_proba(lambda_param=1.5, k=20),
       away_goals_probs=poisson_proba(lambda_param=1.2, k=20),
   )
   probas = gm.return_probas()
   print(f"Home: {probas.proba_home:.2f}, Draw: {probas.proba_draw:.2f}, Away: {probas.proba_away:.2f}")

.. testoutput::

   Home: 0.44, Draw: 0.25, Away: 0.30

Want to train a model on real match data? Start with the
:doc:`Elo tutorial <tutorials/elo>` or the :doc:`Poisson tutorial <tutorials/poisson>`.
For chronological model comparisons, see the
:doc:`walk-forward evaluation tutorial <tutorials/model_evaluation>`.

Which model should I use?
-------------------------

.. list-table::
   :header-rows: 1

   * - Model
     - Required columns
     - Output
     - Runtime
     - Best for
   * - :class:`~footix.models.elo.EloDavidson`
     - ``date``, ``home_team``, ``away_team``, ``fthg``, ``ftag``, ``ftr``
     - :class:`~footix.utils.typing.ProbaResult` (1X2)
     - seconds
     - Fast rating-style probabilities, chronological updates
   * - :class:`~footix.models.basic_poisson.PoissonModel`
     - ``home_team``, ``away_team``, ``fthg``, ``ftag``, ``ftr``
     - :class:`~footix.models.score_matrix.GoalMatrix`
     - seconds
     - Score distributions and market probabilities
   * - :class:`~footix.models.bayesian.BayesianModel`
     - ``home_team``, ``away_team``, ``fthg``, ``ftag``
     - ``GoalMatrix`` + posterior samples
     - minutes (MCMC)
     - Calibrated probabilities with uncertainty

The data contract for every provider is described in
:doc:`Data sources and data contracts <guides/data_sources>`.

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   installation
   guides/data_sources

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/elo
   tutorials/poisson
   tutorials/model_evaluation
   prediction_export_tutorial

.. toctree::
   :maxdepth: 2
   :caption: Cookbook

   cookbook/goal_matrix

.. toctree::
   :maxdepth: 4
   :caption: API Reference

   api/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
