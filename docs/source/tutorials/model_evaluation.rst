Walk-forward model evaluation
=============================

The evaluation API measures models without using future results. It uses a
weekly expanding window:

* the cutoff is Friday at 00:00;
* training contains only matches with ``kickoff < cutoff``;
* the target contains matches in ``[cutoff, cutoff + 7 days)``;
* the input dataframe must contain one competition and one season.

The runnable companion is available here:
:download:`model_performance.ipynb <../../../notebooks/model_performance.ipynb>`.

This deliberately uses real kickoff dates rather than official gameweek
numbers. Postponed matches therefore remain in the time window in which they
were actually played. The final window is flagged in ``result.windows`` so a
partial season can be excluded when interpreting final aggregates.

Running the evaluator
---------------------

The notebook ``notebooks/model_performance.ipynb`` is the interactive entry
point. The package API is also usable from a script:

.. code-block:: python

   from footix.evaluation import BacktestConfig, ModelSpec, run_backtest

   config = BacktestConfig(markets=("1X2", "O/U2.5"))
   result = run_backtest(matches, [my_model_spec], config)
   result.predictions.to_csv("predictions.csv", index=False)

``ModelSpec.factory`` must return a fresh model. This is important for an
expanding window: a model fitted on one cutoff must not be reused at the next
cutoff. A future model can be added without inheriting from a base class:

.. code-block:: python

   spec = ModelSpec(
       name="my-model",
       factory=lambda: MyModel(...),
       markets=("1X2",),
   )

The evaluator records warm-up and fit failures in ``result.windows`` instead
of silently dropping them. ``result.predictions`` contains one row per model,
match and market with RPS, log-loss, Brier score and accuracy. The individual
betting selections are kept in ``result.bets``.

Markets and models
------------------

``uniform_spec()`` provides a data-free 1X2 baseline. It is useful for
separating the first technically usable model window from a model that is
actually better than a trivial forecast.

``PoissonModel`` and ``BayesianModel`` expose 1X2 plus Under/Over 2.5. The
``EloDavidson`` model exposes 1X2 only because it does not produce a score
distribution. Other X.5 lines are not evaluated in the first version because
the historical football-data files do not contain matching odds for them.

Betting simulation
------------------

The evaluator assumes the opening B365 columns are available at the cutoff.
Missing or invalid odds remove a match from the betting table, not from the
prediction metrics. At most one selection is kept per match.

Poisson and Elo keep the candidate with the highest positive point edge and
stake quarter Kelly with a configurable total bankroll cap. Bayesian
selection is posterior-aware (``select_bets_posterior``): a bet is kept only
if its 10% pessimistic edge bound is positive, and within a match the
selection with the largest robust Kelly fraction wins, provided it is the
best edge of the match in more than 60% of the posterior draws. Stakes then
maximise the expected log-growth of the bankroll over joint P&L scenarios
(``optimise_portfolio_torch``, 10,000 scenarios by default): discrete
payoffs, asymmetry and the dependence induced by the common posterior
uncertainty are all kept, and outcomes are simulated independently between
matches conditional on the posterior draw.

The ``flat`` staking mode applies the same robust posterior selection but
stakes a fixed fraction of the bankroll per bet (``flat_fraction``, 1% by
default). Use it to evaluate the quality of the selection alone with fixed
stakes before interpreting portfolio results. ``select_bets_diagnostics``
returns per-candidate statistics (``q_edge``, ``robust_kelly``, ``rho``,
``rejection_reason``) for tuning ``select_alpha``, ``select_delta`` and
``select_rho_min`` on a walk-forward grid instead of the final season.

Interpretation
--------------

The first usable window answers "can this model fit and predict?" It does not
answer whether the model is statistically useful. Use the metric curves and
their comparison with a baseline before declaring a data threshold. Results
from different seasons must be run separately so no training window crosses a
season boundary.
