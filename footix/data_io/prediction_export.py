"""Prediction export utilities for model predictions.

This module transforms model outputs into a normalized JSON-compatible structure
for prediction record consumers.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np
from scipy.stats import gaussian_kde

from footix.metrics import confidence_1x2_from_samples, confidence_curve
from footix.models.score_matrix import GoalMatrix
from footix.utils.typing import SampleProbaResult


def _format_distribution_bin(value: float) -> str:
    rounded = float(np.round(value, decimals=2))
    if np.isclose(rounded * 10.0, np.round(rounded * 10.0)):
        return f"{rounded:.1f}"
    return f"{rounded:.2f}"


DEFAULT_DISTRIBUTION_BINS: tuple[str, ...] = tuple(
    _format_distribution_bin(value) for value in np.linspace(0.0, 1.0, 21)
)
DEFAULT_DISTRIBUTION_EDGES = np.linspace(0.0, 1.0, len(DEFAULT_DISTRIBUTION_BINS))
DEFAULT_DISTRIBUTION_COUNT = len(DEFAULT_DISTRIBUTION_BINS) - 1
DEFAULT_FALLBACK_TIME = "20:00:00"
DEFAULT_TZ = timezone(timedelta(hours=1))


class PredictionExportModel(Protocol):
    """Protocol for models that can generate prediction-export inputs."""

    def predict(self, home_team: str, away_team: str, **kwargs: Any) -> GoalMatrix: ...

    def get_samples(self, home_team: str, away_team: str, **kwargs: Any) -> SampleProbaResult: ...


def _round_float(value: float, decimals: int = 4) -> float:
    return float(np.round(float(value), decimals=decimals))


def _format_confidence_score(value: float) -> float:
    bounded = float(np.clip(value, 0.0, 1.0))
    return float(np.round(bounded, decimals=2))


def _normalize_prob_vector(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    total = float(clipped.sum())
    if not np.isfinite(total) or total <= 0.0:
        return np.array([1.0 / clipped.size] * clipped.size, dtype=np.float64)
    return clipped / total


def _slugify(text: str, compact: bool = False) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    lowercase = ascii_text.lower().replace("'", " ")
    if compact:
        compact_text = re.sub(r"[^a-z0-9]+", "", lowercase)
        return compact_text or "unknown"
    dashed = re.sub(r"[^a-z0-9]+", "-", lowercase).strip("-")
    return re.sub(r"-{2,}", "-", dashed) or "unknown"


def _resolve_mapping_value(
    values: Mapping[str, Any] | None,
    candidates: tuple[str, ...],
) -> Any | None:
    if values is None:
        return None
    for candidate in candidates:
        if candidate in values:
            candidate_value = values[candidate]
            if candidate_value is None:
                continue
            if isinstance(candidate_value, str) and candidate_value.strip() == "":
                continue
            return candidate_value
    return None


def _league_from_payload(
    fixture: Mapping[str, Any], payload_metadata: Mapping[str, Any] | None
) -> str:
    from_fields = _resolve_mapping_value(
        fixture,
        ("league", "competition", "championship"),
    )
    if from_fields is None:
        from_fields = _resolve_mapping_value(
            payload_metadata,
            ("league", "competition", "championship"),
        )
    if isinstance(from_fields, str) and from_fields.strip():
        return from_fields.strip()

    league_url = _resolve_mapping_value(payload_metadata, ("league_url",))
    if isinstance(league_url, str):
        lowered = league_url.lower()
        if "ligue-1" in lowered:
            return "Ligue 1"
        if "ligue-2" in lowered:
            return "Ligue 2"
        if "premier-league" in lowered:
            return "Premier League"
        if "bundesliga" in lowered:
            return "Bundesliga"

    return "Unknown League"


def _parse_kickoff_iso(
    fixture: Mapping[str, Any], payload_metadata: Mapping[str, Any] | None
) -> str:
    kickoff_raw = _resolve_mapping_value(
        fixture,
        ("kickoff", "match_time", "datetime", "date", "kickoff_paris"),
    )
    if kickoff_raw is None:
        kickoff_raw = _resolve_mapping_value(payload_metadata, ("scraped_at_paris",))
    if kickoff_raw is None:
        now_dt = datetime.now(DEFAULT_TZ)
        return now_dt.isoformat(timespec="seconds")

    kickoff_str = str(kickoff_raw).strip()
    if len(kickoff_str) == 10 and re.match(r"^\d{4}-\d{2}-\d{2}$", kickoff_str):
        kickoff_str = f"{kickoff_str}T{DEFAULT_FALLBACK_TIME}"

    try:
        kickoff_dt = datetime.fromisoformat(kickoff_str.replace("Z", "+00:00"))
    except ValueError:
        for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y"):
            try:
                parsed_date = datetime.strptime(kickoff_str, fmt)
                kickoff_dt = datetime(
                    parsed_date.year,
                    parsed_date.month,
                    parsed_date.day,
                    20,
                    0,
                    0,
                )
                break
            except ValueError:
                continue
        else:
            fallback = datetime.now(DEFAULT_TZ)
            return fallback.isoformat(timespec="seconds")

    if kickoff_dt.tzinfo is None:
        kickoff_dt = kickoff_dt.replace(tzinfo=DEFAULT_TZ)
    return kickoff_dt.isoformat(timespec="seconds")


def _kde_distribution_from_samples(values: np.ndarray) -> list[float]:
    """Build probability masses from KDE-smoothed posterior samples."""
    cleaned = np.asarray(values, dtype=np.float64)
    cleaned = cleaned[np.isfinite(cleaned)]
    if cleaned.size == 0:
        base = np.zeros(DEFAULT_DISTRIBUTION_COUNT, dtype=np.float64)
        base[0] = 1.0
        return [_round_float(x) for x in base]

    clipped = np.clip(cleaned, 0.0, 1.0)
    if clipped.size <= 1 or np.allclose(clipped, clipped[0]):
        base = np.zeros(DEFAULT_DISTRIBUTION_COUNT, dtype=np.float64)
        idx = int(
            np.clip(
                np.floor(clipped.mean() * DEFAULT_DISTRIBUTION_COUNT),
                0,
                DEFAULT_DISTRIBUTION_COUNT - 1,
            )
        )
        base[idx] = 1.0
        return [_round_float(x) for x in base]

    try:
        kde = gaussian_kde(clipped)
        grid = np.linspace(0.0, 1.0, 200)
        density = np.clip(kde(grid), 0.0, None)

        edges = DEFAULT_DISTRIBUTION_EDGES
        masses = np.zeros(DEFAULT_DISTRIBUTION_COUNT, dtype=np.float64)
        for i in range(DEFAULT_DISTRIBUTION_COUNT):
            if i < DEFAULT_DISTRIBUTION_COUNT - 1:
                mask = (grid >= edges[i]) & (grid < edges[i + 1])
            else:
                mask = (grid >= edges[i]) & (grid <= edges[i + 1])

            x_slice = grid[mask]
            y_slice = density[mask]
            if x_slice.size >= 2:
                masses[i] = float(np.trapezoid(y_slice, x=x_slice))
            else:
                center = 0.5 * (edges[i] + edges[i + 1])
                masses[i] = max(float(kde(center)), 0.0) * 0.1

        norm_masses = _normalize_prob_vector(masses)
        return [_round_float(x) for x in norm_masses]
    except Exception:
        hist, _ = np.histogram(clipped, bins=DEFAULT_DISTRIBUTION_EDGES, density=False)
        norm_hist = _normalize_prob_vector(hist.astype(np.float64))
        return [_round_float(x) for x in norm_hist]


def _score_matrix_5x5(matrix_array: np.ndarray) -> list[list[float]]:
    source = np.asarray(matrix_array, dtype=np.float64)
    if source.ndim != 2:
        source = np.zeros((5, 5), dtype=np.float64)

    rows = min(5, source.shape[0])
    cols = min(5, source.shape[1])
    output = np.zeros((5, 5), dtype=np.float64)
    output[:rows, :cols] = source[:rows, :cols]
    output = np.clip(output, 0.0, 1.0)
    return [[_round_float(x) for x in row] for row in output]


def _sanitize_distribution(values: list[float]) -> list[float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size > DEFAULT_DISTRIBUTION_COUNT:
        arr = arr[:DEFAULT_DISTRIBUTION_COUNT]
    if arr.size < DEFAULT_DISTRIBUTION_COUNT:
        arr = np.pad(
            arr,
            (0, DEFAULT_DISTRIBUTION_COUNT - arr.size),
            mode="constant",
            constant_values=0.0,
        )
    arr = _normalize_prob_vector(arr)
    return [_round_float(x) for x in arr]


def _sanitize_prediction(
    home_win: float, draw: float, away_win: float
) -> tuple[float, float, float]:
    normalized = _normalize_prob_vector(np.array([home_win, draw, away_win], dtype=np.float64))
    return _round_float(normalized[0]), _round_float(normalized[1]), _round_float(normalized[2])


def _has_invalid_values(obj: Any) -> bool:
    if obj is None:
        return True
    if isinstance(obj, float):
        return not np.isfinite(obj)
    if isinstance(obj, str):
        return obj.strip() == ""
    if isinstance(obj, list):
        return any(_has_invalid_values(item) for item in obj)
    if isinstance(obj, dict):
        return any(_has_invalid_values(v) for v in obj.values())
    return False


def _validate_record_shape(record: Mapping[str, Any]) -> tuple[bool, str | None]:
    required_fields = (
        "id",
        "league",
        "kickoff",
        "home_team",
        "away_team",
        "prediction",
        "home_distribution",
        "draw_distribution",
        "away_distribution",
        "distribution_bins",
        "score_matrix",
    )
    for field in required_fields:
        if field not in record:
            return False, f"missing field '{field}'"
    prediction = record["prediction"]
    if not isinstance(prediction, dict):
        return False, "prediction must be an object"
    total = (
        float(prediction.get("home_win_prob", 0.0))
        + float(prediction.get("draw_prob", 0.0))
        + float(prediction.get("away_win_prob", 0.0))
    )
    if abs(total - 1.0) > 0.01:
        return False, "prediction probabilities do not sum to 1"
    if len(record["distribution_bins"]) != len(DEFAULT_DISTRIBUTION_BINS):
        return False, f"distribution_bins must have length {len(DEFAULT_DISTRIBUTION_BINS)}"
    expected_distribution_length = len(record["distribution_bins"]) - 1
    if len(record["home_distribution"]) != expected_distribution_length:
        return False, f"home_distribution must have length {expected_distribution_length}"
    if len(record["draw_distribution"]) != expected_distribution_length:
        return False, f"draw_distribution must have length {expected_distribution_length}"
    if len(record["away_distribution"]) != expected_distribution_length:
        return False, f"away_distribution must have length {expected_distribution_length}"
    score_matrix = record["score_matrix"]
    if len(score_matrix) != 5 or any(len(row) != 5 for row in score_matrix):
        return False, "score_matrix must be 5x5"
    if _has_invalid_values(record):
        return False, "record contains null, NaN, or empty values"
    return True, None


def _build_match_identifier(league: str, home_team: str, away_team: str, kickoff_iso: str) -> str:
    kickoff_date = kickoff_iso.split("T", maxsplit=1)[0]
    league_slug = _slugify(league, compact=True)
    home_slug = _slugify(home_team, compact=False)
    away_slug = _slugify(away_team, compact=False)
    return f"{league_slug}-{home_slug}-{away_slug}-{kickoff_date}"


def build_prediction_records_from_predictions(
    fixtures: Sequence[Mapping[str, Any]],
    goal_matrices: Mapping[str, GoalMatrix],
    samples: Mapping[str, SampleProbaResult],
    payload_metadata: Mapping[str, Any] | None = None,
    team_normalizer: Callable[[str], str] | None = None,
    confidence_gamma: float | None = 0.7,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Build prediction records from existing prediction artifacts.

    Args:
        fixtures: Raw fixtures payload from odds JSON.
        goal_matrices: Mapping from match key to score matrix predictions.
        samples: Mapping from match key to posterior probability samples.
        payload_metadata: Optional metadata extracted from odds payload.
        team_normalizer: Optional callable for team-name normalization.

    Returns:
        Tuple of valid records and technical error reports.
    """
    records: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for fixture in fixtures:
        raw_home = _resolve_mapping_value(fixture, ("home_team", "team_home", "home"))
        raw_away = _resolve_mapping_value(fixture, ("away_team", "team_away", "away"))
        if raw_home is None or raw_away is None:
            errors.append({"match": "unknown", "error": "missing home/away"})
            continue

        home_team = str(raw_home).replace("Cotes", "").strip()
        away_team = str(raw_away).strip()
        if team_normalizer is not None:
            home_team = team_normalizer(home_team)
            away_team = team_normalizer(away_team)

        match_key = f"{home_team} - {away_team}"
        goal_matrix = goal_matrices.get(match_key)
        sample_result = samples.get(match_key)

        if goal_matrix is None:
            errors.append({"match": match_key, "error": "missing goal matrix"})
            continue
        if sample_result is None:
            errors.append({"match": match_key, "error": "missing posterior samples"})
            continue

        league = _league_from_payload(fixture, payload_metadata)
        kickoff_iso = _parse_kickoff_iso(fixture, payload_metadata)
        proba_result = goal_matrix.return_probas()
        p_home, p_draw, p_away = _sanitize_prediction(
            proba_result.proba_home,
            proba_result.proba_draw,
            proba_result.proba_away,
        )

        confidence_components = confidence_1x2_from_samples(sample_result)
        raw_confidence = float(np.clip(confidence_components.confidence, 0.0, 100.0)) / 100.0
        confidence_score = (
            _format_confidence_score(
                confidence_curve(raw_confidence * 100.0, gamma=confidence_gamma) / 100.0
            )
            if confidence_gamma is not None
            else _format_confidence_score(raw_confidence)
        )

        home_distribution = _sanitize_distribution(
            _kde_distribution_from_samples(np.asarray(sample_result.proba_home, dtype=np.float64))
        )
        draw_distribution = _sanitize_distribution(
            _kde_distribution_from_samples(np.asarray(sample_result.proba_draw, dtype=np.float64))
        )
        away_distribution = _sanitize_distribution(
            _kde_distribution_from_samples(np.asarray(sample_result.proba_away, dtype=np.float64))
        )

        gameweek_value = _resolve_mapping_value(fixture, ("gameweek", "game_week"))
        parsed_gameweek: int | None = None
        if gameweek_value is not None:
            try:
                parsed_gameweek = int(gameweek_value)
            except (TypeError, ValueError):
                parsed_gameweek = None

        record: dict[str, object] = {
            "id": _build_match_identifier(league, home_team, away_team, kickoff_iso),
            "league": league,
            "kickoff": kickoff_iso,
            "home_team": home_team,
            "away_team": away_team,
            "prediction": {
                "home_win_prob": p_home,
                "draw_prob": p_draw,
                "away_win_prob": p_away,
                "confidence_score": confidence_score,
            },
            "home_distribution": home_distribution,
            "draw_distribution": draw_distribution,
            "away_distribution": away_distribution,
            "distribution_bins": list(DEFAULT_DISTRIBUTION_BINS),
            "score_matrix": _score_matrix_5x5(goal_matrix.matrix_array),
        }
        if parsed_gameweek is not None:
            record["gameweek"] = parsed_gameweek

        is_valid, validation_error = _validate_record_shape(record)
        if not is_valid:
            errors.append({"match": match_key, "error": validation_error or "invalid record"})
            continue
        records.append(record)

    records.sort(key=lambda match: match["kickoff"])
    return records, errors


def export_prediction_records_from_model(
    model: PredictionExportModel,
    fixtures: Sequence[Mapping[str, Any]],
    payload_metadata: Mapping[str, Any] | None = None,
    team_normalizer: Callable[[str], str] | None = None,
    predict_kwargs: Mapping[str, Any] | None = None,
    sample_kwargs: Mapping[str, Any] | None = None,
    confidence_gamma: float | None = 0.7,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Compute predictions from a model and export prediction records.

    Args:
        model: Predictive model supporting predict/get_samples.
        fixtures: Raw fixtures payload from odds JSON.
        payload_metadata: Optional metadata extracted from odds payload.
        team_normalizer: Optional callable for team-name normalization.
        predict_kwargs: Optional extra kwargs forwarded to predict.
        sample_kwargs: Optional extra kwargs forwarded to get_samples.

    Returns:
        Tuple of valid records and technical error reports.
    """
    predict_kwargs = dict(predict_kwargs or {})
    sample_kwargs = dict(sample_kwargs or {})

    goal_matrices: dict[str, GoalMatrix] = {}
    sample_map: dict[str, SampleProbaResult] = {}
    errors: list[dict[str, str]] = []

    for fixture in fixtures:
        raw_home = _resolve_mapping_value(fixture, ("home_team", "team_home", "home"))
        raw_away = _resolve_mapping_value(fixture, ("away_team", "team_away", "away"))
        if raw_home is None or raw_away is None:
            errors.append({"match": "unknown", "error": "missing home/away"})
            continue

        home_team = str(raw_home).replace("Cotes", "").strip()
        away_team = str(raw_away).strip()
        if team_normalizer is not None:
            home_team = team_normalizer(home_team)
            away_team = team_normalizer(away_team)
        match_key = f"{home_team} - {away_team}"

        try:
            goal_matrices[match_key] = model.predict(home_team, away_team, **predict_kwargs)
            sample_map[match_key] = model.get_samples(home_team, away_team, **sample_kwargs)
        except Exception as exc:  # pragma: no cover
            errors.append({"match": match_key, "error": str(exc)})

    records, build_errors = build_prediction_records_from_predictions(
        fixtures=fixtures,
        goal_matrices=goal_matrices,
        samples=sample_map,
        payload_metadata=payload_metadata,
        team_normalizer=team_normalizer,
        confidence_gamma=confidence_gamma,
    )
    return records, errors + build_errors


__all__ = [
    "build_prediction_records_from_predictions",
    "export_prediction_records_from_model",
]
