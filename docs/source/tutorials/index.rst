Tutorials
=========

End-to-end walkthroughs on French Ligue 1 data. Each tutorial has a runnable
companion notebook you can download and follow at your own pace.

.. grid:: 1 1 2 3
   :gutter: 3

   .. grid-item-card:: Elo ratings
      :link: elo
      :link-type: doc

      Train ``EloDavidson`` on a full season, read the ratings, evaluate a
      held-out matchday and predict one match.

   .. grid-item-card:: Poisson score model
      :link: poisson
      :link-type: doc

      Fit ``PoissonModel``, score its 1X2 forecasts and explore the score
      distribution of a single fixture.

   .. grid-item-card:: Exporting predictions
      :link: ../prediction_export_tutorial
      :link-type: doc

      Turn model output into prediction JSON records for scripts and
      automation.

.. toctree::
   :hidden:

   elo
   poisson
   ../prediction_export_tutorial
