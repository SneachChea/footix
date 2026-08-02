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
prediction metrics. Only one selection is kept per match: the candidate with
the highest positive edge among the enabled markets.

Poisson and Elo use quarter Kelly with a configurable total bankroll cap.
Bayesian uses the posterior-aware ``optimise_portfolio_torch`` path. The
initial defaults are a 1,000-unit bankroll, 30% weekly exposure and a 5%
risk parameter. The portfolio currently assumes independence between
different matches by selecting at most one bet per match.

Interpretation
--------------

The first usable window answers "can this model fit and predict?" It does not
answer whether the model is statistically useful. Use the metric curves and
their comparison with a baseline before declaring a data threshold. Results
from different seasons must be run separately so no training window crosses a
season boundary.
