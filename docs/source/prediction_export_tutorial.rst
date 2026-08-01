Prediction Export Tutorial
==========================

This tutorial explains how to export ML predictions to a prediction JSON format
from core Python code (scriptable workflow).

The export pipeline is shared and implemented in ``footix.data_io.prediction_export``.


What the exporter produces
--------------------------

Each exported match contains:

- ``id``
- ``league``
- ``kickoff``
- ``home_team`` / ``away_team``
- ``prediction`` with:

  - ``home_win_prob``
  - ``draw_prob``
  - ``away_win_prob``
  - ``confidence_score``

- ``home_distribution`` / ``draw_distribution`` / ``away_distribution``
- ``distribution_bins``
- ``score_matrix`` (5x5)

Notes:

- 1X2 probabilities are normalized to sum to approximately 1.
- Distribution arrays are normalized and exported with length 20.
- ``distribution_bins`` is exported with length 21.
- ``score_matrix`` is forced to 5x5 (truncate/pad).
- Records are sorted by ``kickoff``.


Core-code export (recommended for automation)
---------------------------------------------

Use this when you want CLI/script/batch export without UI.

High-level API
~~~~~~~~~~~~~~

Use:

- ``export_prediction_records_from_model(...)`` to predict + export in one call
- ``build_prediction_records_from_predictions(...)`` if you already have predictions

Example: predict + export in one call
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import json
    from pathlib import Path

    from footix.data_io.prediction_export import export_prediction_records_from_model
    from footix.models.bayesian import BayesianModel

    # model already fitted earlier
    model = BayesianModel(n_goals=20, n_teams=18, calibrate=True, use_stats=True)

    # Load a fixture payload from your own JSON file.
    payload = json.loads(Path("path/to/fixture_payload.json").read_text())
    fixtures = payload.get("fixtures", [])

    payload_metadata = {
        "league": payload.get("league"),
        "league_url": payload.get("league_url"),
        "scraped_at_paris": payload.get("scraped_at_paris"),
        "next_matchday_window_paris": payload.get("next_matchday_window_paris"),
    }

    records, errors = export_prediction_records_from_model(
        model=model,
        fixtures=fixtures,
        payload_metadata=payload_metadata,
    )

    Path("prediction_export.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if errors:
        Path("prediction_export_errors.json").write_text(
            json.dumps(errors, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


Example: export from existing predictions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you already have per-match ``GoalMatrix`` and posterior samples:

.. code-block:: python

    from footix.data_io.prediction_export import build_prediction_records_from_predictions

    records, errors = build_prediction_records_from_predictions(
        fixtures=fixtures,
        goal_matrices=goal_matrices_by_match_id,
        samples=samples_by_match_id,
        payload_metadata=payload_metadata,
    )


Confidence score behavior
-------------------------

``confidence_score`` is computed from posterior 1X2 samples using
``confidence_1x2_from_samples_array(...)`` and uses
``ConfidenceComponents.confidence``.

Current project decision:

- ``confidence_score`` is exported as a float between 0 and 1, rounded to 2 decimal places.


Validation and rejected matches
-------------------------------

When a match cannot be corrected to a valid schema (for example missing predictions), it is excluded from the final records.
Technical errors are returned separately in ``errors``.

Recommended practice:

1. Persist the successful records JSON.
2. Persist ``errors`` as a separate file for diagnostics.


Troubleshooting
---------------

Many exported errors
~~~~~~~~~~~~~~~~~~~~

Check team-name normalization consistency between fixture payload and trained model teams.

Wrong or missing kickoff
~~~~~~~~~~~~~~~~~~~~~~~~

Provide kickoff fields in fixtures (``kickoff_paris``, ``kickoff``, ``datetime``, etc.).
If missing, fallback logic applies.
