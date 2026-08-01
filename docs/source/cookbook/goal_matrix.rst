GoalMatrix cookbook
===================

:class:`~footix.models.score_matrix.GoalMatrix` wraps a joint distribution
over scores ``(home_goals, away_goals)``. It is the output of
:class:`~footix.models.basic_poisson.PoissonModel.predict` and the building
block for every market probability in footix.

Build a matrix from two marginal goal distributions (here Poisson PMFs with
expected goals 1.5 and 1.2, truncated at 20 scorelines per team):

.. testcode::

   from footix.models.score_matrix import GoalMatrix
   from footix.models.utils import poisson_proba

   gm = GoalMatrix(
       home_goals_probs=poisson_proba(lambda_param=1.5, k=20),
       away_goals_probs=poisson_proba(lambda_param=1.2, k=20),
   )

The input vectors are validated (finite, non-negative, same length,
positive mass) and normalized to sum to 1.

1X2 probabilities
-----------------

.. testcode::

   probas = gm.return_probas()
   print(f"Home: {probas.proba_home:.4f} Draw: {probas.proba_draw:.4f} Away: {probas.proba_away:.4f}")

.. testoutput::

   Home: 0.4415 Draw: 0.2548 Away: 0.3037

Most probable score
-------------------

.. testcode::

   print(gm.get_probable_score())

.. testoutput::

   (1, 1)

Over / Under goals
------------------

.. testcode::

   print(f"Under 1.5: {gm.less_15_goals():.4f}  Over 1.5: {gm.more_15_goals():.4f}")
   print(f"Under 2.5: {gm.less_25_goals():.4f}  Over 2.5: {gm.more_25_goals():.4f}")

.. testoutput::

   Under 1.5: 0.2487  Over 1.5: 0.7513
   Under 2.5: 0.4936  Over 2.5: 0.5064

Both teams to score
-------------------

.. testcode::

   print(f"BTTS: {gm.probability_both_teams_scores():.4f}")

.. testoutput::

   BTTS: 0.5429

Double chance
-------------

.. testcode::

   one_x, x_two, one_two = gm.double_chance()
   print(f"1X: {one_x:.4f}  X2: {x_two:.4f}  12: {one_two:.4f}")

.. testoutput::

   1X: 0.6963  X2: 0.5585  12: 0.7452

Asian handicap
--------------

Handicaps are applied to the home team's goal count. A half-line handicap
(e.g. -0.5) removes the draw outcome:

.. testcode::

   ah = gm.asian_handicap_results(handicap=-0.5)
   print(f"AH -0.5 Home: {ah.proba_home:.4f} Draw: {ah.proba_draw:.4f} Away: {ah.proba_away:.4f}")

.. testoutput::

   AH -0.5 Home: 0.4415 Draw: 0.0000 Away: 0.5585

A zero handicap reproduces the 1X2 probabilities:

.. testcode::

   ah0 = gm.asian_handicap_results(handicap=0.0)
   print(f"AH 0 Home: {ah0.proba_home:.4f} Draw: {ah0.proba_draw:.4f} Away: {ah0.proba_away:.4f}")

.. testoutput::

   AH 0 Home: 0.4415 Draw: 0.2548 Away: 0.3037

Visualize the matrix
--------------------

.. plot::

   from footix.models.score_matrix import GoalMatrix
   from footix.models.utils import poisson_proba

   gm = GoalMatrix(
       home_goals_probs=poisson_proba(lambda_param=1.5, k=20),
       away_goals_probs=poisson_proba(lambda_param=1.2, k=20),
   )
   gm.visualize()

Caveats
-------

* Scorelines above ``n_goals - 1`` are not represented: the matrix is
  truncated, so market probabilities are renormalized over the supported
  range.
* ``less_15_goals`` needs at least 2 scorelines per team,
  ``less_25_goals`` at least 3 — ``GoalMatrix`` raises otherwise.
* An optional ``correlation_matrix`` (non-negative, same shape) can
  reweight scorelines element-wise before renormalization.
