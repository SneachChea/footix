Welcome to **footix**'s documentation!
=============

Footix is your intelligent companion for sports analysis and prediction. Leveraging advanced machine learning algorithms 
and comprehensive data analysis, it helps you make data-driven decisions in sports betting and analysis.

Features
--------

* **Advanced Data Analysis**
    * Import data from multiple sports databases
    * Clean and preprocess sports statistics
    * Comprehensive historical data analysis

* **Smart Prediction Engine**
    * Machine learning-powered outcome prediction

* **Strategic Betting Tools**
    * Risk assessment algorithms
    * Bankroll management system
    * Multiple betting strategy templates

Quick Start
----------

.. code-block:: python

    from footix.models.bayesian import Bayesian
    from footix.data_io.footballdata import ScrapFootballData

    # Load match data (example: Ligue 1 fixtures)
    dataset = ScrapFootballData(
        competition="FRA Ligue 1", 
        season="2024-2025", 
        path="./data", 
        force_reload=True
    ).get_fixtures()

    # Initialize and fit the Bayesian model
    model = Bayesian(n_teams=18, n_goals=20)
    model.fit(X_train=dataset)

    # Predict probabilities for a specific match
    probas = model.predict(home_team="Marseille", away_team="Lyon").return_probas()
    print(f"Home: {probas[0]:.2f}, Draw: {probas[1]:.2f}, Away: {probas[2]:.2f}")

.. toctree::cumentation master file, created by
   sphinx-quickstart on Thu Jun  5 00:06:47 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


.. toctree::
   :maxdepth: 2
   :caption: How to start
   :hidden:
   
   installation

.. toctree::
   :maxdepth: 4
   :caption: API informations
   :hidden:

   api/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
