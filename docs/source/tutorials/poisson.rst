Poisson score model tutorial
============================

This tutorial trains a :class:`~footix.models.basic_poisson.PoissonModel` on
the French Ligue 1 season 2024-2025, evaluates it on a held-out matchday,
and explores the score distribution of one match with
:class:`~footix.models.score_matrix.GoalMatrix`.

.. note::

   The runnable companion notebook is available here:
   :download:`poisson.ipynb <../../../notebooks/poisson.ipynb>`.

Load and split the data
-----------------------

.. code-block:: python

   import numpy as np
   import pandas as pd

   from footix.data_io.footballdata import ScrapFootballData

   dataset = ScrapFootballData(
       competition="FRA Ligue 1",
       season="2024-2025",
       path="./data",
       force_reload=False,
   ).get_fixtures()

   dataset["date"] = pd.to_datetime(dataset["date"], dayfirst=True)
   dataset = dataset.sort_values("date").reset_index(drop=True)

   train_dataset = dataset.iloc[:-27]       # first 279 matches
   test_dataset = dataset.iloc[-27:-18]     # matchday 32: 9 matches
   # dataset.iloc[-18:] is kept as a reserve, unused here

Fit the model
-------------

.. code-block:: python

   from footix.models.basic_poisson import PoissonModel

   model = PoissonModel(n_teams=18, n_goals=20)
   model.fit(X_train=train_dataset)

``n_goals`` is the number of scorelines considered per team (0 to
``n_goals - 1``). Probabilities for scores above ``n_goals - 1`` are
truncated and the matrix is renormalized.

Evaluate on the held-out matchday
---------------------------------

Poisson predicts a :class:`~footix.models.score_matrix.GoalMatrix`; reduce
it to 1X2 probabilities with ``return_probas()`` before scoring:

.. code-block:: python

   from footix.metrics import incertity, rps, zscore

   def _outcome_idx(result: str) -> int:
       return {"H": 0, "D": 1, "A": 2}[result]

   rps_list, incertity_list, zscore_list = [], [], []
   for _, row in test_dataset.iterrows():
       probas = model.predict(home_team=row["home_team"], away_team=row["away_team"]).return_probas()
       outcome_idx = _outcome_idx(row["ftr"])
       rps_list.append(rps(probas=probas, outcome_idx=outcome_idx))
       incertity_list.append(incertity(probas=probas, outcome_idx=outcome_idx))
       zscore_list.append(zscore(probas=probas, rps_observed=rps_list[-1], seed=42).z_score)

   print(f"Incertity: {np.mean(incertity_list):.3f} +/- {np.std(incertity_list):.3f}")
   print(f"RPS:       {np.mean(rps_list):.3f} +/- {np.std(rps_list):.3f}")
   print(f"Z-score:   {np.mean(zscore_list):.3f} +/- {np.std(zscore_list):.3f}")

The metric semantics are described in the
:doc:`Elo tutorial <elo>`; the same caveat applies — one matchday is a
smoke test, not a backtest.

Explore a score distribution
----------------------------

.. code-block:: python

   gm = model.predict(home_team="St Etienne", away_team="Monaco")
   probas = gm.return_probas()
   print(f"Home: {probas.proba_home:.2f}, Draw: {probas.proba_draw:.2f}, Away: {probas.proba_away:.2f}")
   print("Most probable score:", gm.get_probable_score())
   gm.visualize()

``visualize()`` renders the joint score matrix; ``get_probable_score()``
returns the most likely ``(home_goals, away_goals)`` pair.

Compare with the actual result
------------------------------

.. code-block:: python

   score = test_dataset.loc[
       test_dataset["home_team"] == "St Etienne", ["fthg", "ftag"]
   ].iloc[0]
   print(f"Actual result: {int(score['fthg'])}-{int(score['ftag'])}")

Where to go next
----------------

* :doc:`../cookbook/goal_matrix` — all market probabilities you can derive
  from a ``GoalMatrix``.
* :doc:`elo` — a rating-based alternative to the Poisson model.
