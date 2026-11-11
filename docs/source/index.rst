.. image:: _static/logo_footix.png
   :alt: Footix, a football-loving robot rooster
   :width: 132px
   :align: center
   :class: footix-hero-logo

footix
======

.. rst-class:: footix-tagline

**Football analytics and prediction for Python.**

.. rst-class:: footix-intro

**footix** brings historical match data, score models and betting tools
together in one focused package. Import results, fit an Elo or Poisson model,
read market probabilities from a ``GoalMatrix``, and turn an edge into a
staking plan.

Start here
----------

.. grid:: 1 2 3 3
   :gutter: 3

   .. grid-item-card:: Installation
      :link: installation
      :link-type: doc

      Install with ``uv`` or ``pip``, including the CPU-only PyTorch default.

   .. grid-item-card:: Guides
      :link: guides/index
      :link-type: doc

      Understand the data providers, competition keys and data contracts.

   .. grid-item-card:: Tutorials
      :link: tutorials/index
      :link-type: doc

      Train an Elo or Poisson model on Ligue 1 data, step by step.

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Cookbook
      :link: cookbook/index
      :link-type: doc

      Work through score matrices and market probabilities.

   .. grid-item-card:: API reference
      :link: api/index
      :link-type: doc

      Browse every public class, function and module.

Quick start
-----------

No data download required. Build a score matrix straight from expected goals
and read off the 1X2 probabilities:

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

The same matrix powers totals, Asian handicaps, double chance and BTTS; see
the :doc:`GoalMatrix cookbook <cookbook/goal_matrix>` for the full tour.

What's inside
-------------

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Data import

      Historical results from football-data.co.uk and understat, upcoming
      fixtures from football-data.org and APIFootball.com, plus team-name
      normalization between sources.

   .. grid-item-card:: Prediction models

      ``EloDavidson`` for fast chronological ratings, ``PoissonModel`` for
      maximum-likelihood score distributions, and ``BayesianModel`` for PyMC
      posteriors with uncertainty.

   .. grid-item-card:: Score and betting tools

      ``GoalMatrix`` for joint score probabilities, bookmaker-margin
      normalization (multiplicative, power, Shin), and flat, Kelly or
      portfolio staking strategies.

   .. grid-item-card:: Analysis and visualization

      RPS, log loss, Brier score, calibration metrics, standings helpers and
      Matplotlib views of any score matrix.

.. _model-comparison:

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
     - Posterior probabilities with uncertainty; calibrate 1X2 out-of-sample with ``OutcomeCalibrator``

The data contract for every provider is described in
:doc:`Data sources and data contracts <guides/data_sources>`.

.. toctree::
   :hidden:
   :caption: Getting started

   installation

.. toctree::
   :hidden:
   :caption: Guides

   guides/index

.. toctree::
   :hidden:
   :caption: Tutorials

   tutorials/index

.. toctree::
   :hidden:
   :caption: Cookbook

   cookbook/index

.. toctree::
   :hidden:
   :caption: API Reference

   api/index

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
