Elo ratings tutorial
====================

This tutorial trains an :class:`~footix.models.elo.EloDavidson` model on the
French Ligue 1 season 2024-2025, compares the resulting ratings with the
league standings, evaluates the model on a held-out matchday, and predicts
one match.

.. note::

   The runnable companion notebook is available here:
   :download:`elo.ipynb <../../../notebooks/elo.ipynb>`.

Load the data
-------------

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

The scraper downloads the season once and caches it as a CSV in
``./data``. The first run needs network access; later runs reuse the file.
The returned frame contains one row per match with snake_case columns
(see :doc:`../guides/data_sources`).

Sort chronologically and split
------------------------------

Elo is a chronological model: the order of matches matters. Sort by date
and hold out the last matchday (9 matches for an 18-team league):

.. code-block:: python

   dataset["date"] = pd.to_datetime(dataset["date"], dayfirst=True)
   dataset = dataset.sort_values("date").reset_index(drop=True)

   train_dataset = dataset.iloc[:-9]
   test_dataset = dataset.iloc[-9:]

Fit the model
-------------

.. code-block:: python

   from footix.models.elo import EloDavidson

   model = EloDavidson(
       n_teams=18,
       k0=75,
       lambd=0.1,
       sigma=400,
       agnostic_probs=(0.45, 0.25, 0.30),
   )
   model.fit(X_train=train_dataset)

The parameters control the rating update: ``k0`` is the base K-factor,
``lambd`` scales the K-factor with the goal difference
(``K = k0 * (1 + gamma)**lambd``), ``sigma`` fixes the rating scale, and
``agnostic_probs`` are the league-average home/draw/away probabilities used
to center the model.

The ranking
-----------

Printing the model renders the rating table, strongest first:

.. code-block:: python

   print(model)

Compare with the actual standings to get a first sanity check:

.. code-block:: python

   from footix.metrics import standings

   actual_standings = standings.compute_standings(train_dataset)
   elo_ratings = pd.DataFrame(
       {"team": team, "elo": team_elo.rank}
       for team, team_elo in model.championnat.items()
   ).sort_values("elo", ascending=False).reset_index(drop=True)
   elo_ratings["elo_rank"] = elo_ratings.index + 1

   comparison = pd.merge(
       actual_standings[["team", "position"]],
       elo_ratings[["team", "elo", "elo_rank"]],
       on="team",
   )
   comparison["rank_diff"] = comparison["position"] - comparison["elo_rank"]
   comparison.sort_values("position")

Evaluate on the held-out matchday
---------------------------------

For each test match, predict the 1X2 probabilities and compare them with the
realized outcome using the metrics from :mod:`footix.metrics.metrics_function`:

* :func:`~footix.metrics.metrics_function.rps` — ranked probability score; lower is better,
  0 is a perfect forecast.
* :func:`~footix.metrics.metrics_function.incertity` — the surprise of the realized outcome
  (normalized log-loss); 1 is maximally uncertain, 0 is a confident hit.
* :func:`~footix.metrics.metrics_function.zscore` — compares the observed RPS with a Monte
  Carlo distribution of RPS simulated from the forecast. ``|z| >> 2``
  signals miscalibration. Seed it for reproducibility and read the
  ``.z_score`` field of the returned :class:`~footix.utils.typing.RPSResult`.

.. code-block:: python

   from footix.metrics import incertity, rps, zscore

   def _outcome_idx(result: str) -> int:
       return {"H": 0, "D": 1, "A": 2}[result]

   rps_list, incertity_list, zscore_list = [], [], []
   for _, row in test_dataset.iterrows():
       probas = model.predict(home_team=row["home_team"], away_team=row["away_team"])
       outcome_idx = _outcome_idx(row["ftr"])
       rps_list.append(rps(probas=probas, outcome_idx=outcome_idx))
       incertity_list.append(incertity(probas=probas, outcome_idx=outcome_idx))
       zscore_list.append(zscore(probas=probas, rps_observed=rps_list[-1], seed=42).z_score)

   print(f"Incertity: {np.mean(incertity_list):.3f} +/- {np.std(incertity_list):.3f}")
   print(f"RPS:       {np.mean(rps_list):.3f} +/- {np.std(rps_list):.3f}")
   print(f"Z-score:   {np.mean(zscore_list):.3f} +/- {np.std(zscore_list):.3f}")

.. warning::

   A single held-out matchday is a smoke test, not a backtest. Use many
   matchdays and report the distribution of the metrics, not just the mean,
   before drawing conclusions about calibration.

Predict a match
---------------

.. code-block:: python

   probability = model.predict(home_team="St Etienne", away_team="Toulouse")
   print(f"Home: {probability.proba_home:.2f}")
   print(f"Draw: {probability.proba_draw:.2f}")
   print(f"Away: {probability.proba_away:.2f}")

Where to go next
----------------

* :doc:`../tutorials/poisson` — score distributions with the Poisson model.
* :doc:`../cookbook/goal_matrix` — turn a score distribution into market
  probabilities.
