"""Tests for the football-data.org live data provider."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest
import requests

from footix.data_io.football_data_org import FootballDataOrgError, ScrapFootballDataOrg

_SAMPLE_RESPONSE = {
    "competition": {"id": 2015, "name": "Ligue 1", "code": "FL1"},
    "filters": {"season": "2026"},
    "matches": [
        {
            "id": 500001,
            "utcDate": "2026-08-16T19:00:00Z",
            "status": "SCHEDULED",
            "matchday": 2,
            "homeTeam": {"id": 524, "name": "Paris Saint-Germain FC"},
            "awayTeam": {"id": 516, "name": "Olympique de Marseille"},
        },
        {
            "id": 500002,
            "utcDate": "2026-08-17T17:00:00Z",
            "status": "SCHEDULED",
            "matchday": 2,
            "homeTeam": {"id": 523, "name": "Olympique Lyonnais"},
            "awayTeam": {"id": 521, "name": "LOSC Lille"},
        },
    ],
}

_TWO_MD_RESPONSE = {
    **_SAMPLE_RESPONSE,
    "matches": [
        {
            "id": 500003,
            "utcDate": "2026-08-15T15:00:00Z",
            "status": "SCHEDULED",
            "matchday": 1,
            "homeTeam": {"id": 510, "name": "Stade Rennais FC"},
            "awayTeam": {"id": 511, "name": "FC Nantes"},
        },
    ]
    + _SAMPLE_RESPONSE["matches"],
}

_TIMED_RESPONSE = {
    "competition": {"id": 2015, "name": "Ligue 1", "code": "FL1"},
    "filters": {"season": "2026"},
    "matches": [
        {
            "id": 500001,
            "utcDate": "2026-08-16T19:00:00Z",
            "status": "TIMED",
            "matchday": 2,
            "homeTeam": {"id": 524, "name": "Paris Saint-Germain FC"},
            "awayTeam": {"id": 516, "name": "Olympique de Marseille"},
        }
    ],
}

_MIXED_OFFSET_RESPONSE = {
    "competition": {"id": 2015, "name": "Ligue 1", "code": "FL1"},
    "filters": {"season": "2026"},
    "matches": [
        {
            "id": 500004,
            "utcDate": "2026-03-28T20:00:00Z",
            "status": "TIMED",
            "matchday": 27,
            "homeTeam": {"id": 10, "name": "Team A"},
            "awayTeam": {"id": 11, "name": "Team B"},
        },
        {
            "id": 500005,
            "utcDate": "2026-03-29T20:00:00Z",
            "status": "TIMED",
            "matchday": 27,
            "homeTeam": {"id": 12, "name": "Team C"},
            "awayTeam": {"id": 13, "name": "Team D"},
        },
    ],
}


def _response(payload: object, status_code: int = 200) -> requests.Response:
    response = requests.Response()
    response.status_code = status_code
    response._content = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    response.encoding = "utf-8"
    return response


@pytest.fixture
def mock_get(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Replace ``requests.get`` with a call-recording mock."""
    calls: list[dict] = []

    def _mock(url: str, **kwargs: object) -> requests.Response:
        calls.append(
            {"url": url, "headers": kwargs.get("headers"), "params": kwargs.get("params")}
        )
        return _response(_SAMPLE_RESPONSE)

    monkeypatch.setattr(requests, "get", _mock)
    return calls


def _make_scraper(
    tmp_path: Path,
    mapping_teams: dict[str, str] | None = None,
    force_reload: bool = False,
    ttl: int = 21600,
) -> ScrapFootballDataOrg:
    return ScrapFootballDataOrg(
        competition="FRA Ligue 1",
        season="2026",
        api_token="test_token_xyz",
        path=str(tmp_path),
        mapping_teams=mapping_teams,
        force_reload=force_reload,
        ttl=ttl,
    )


# ---------------------------------------------------------------------------
# API request tests
# ---------------------------------------------------------------------------


def test_url_and_params(mock_get: list[dict], tmp_path: Path) -> None:
    _make_scraper(tmp_path).get_fixtures()
    call = mock_get[0]
    assert "/competitions/FL1/matches" in call["url"]
    assert call["params"]["status"] == "SCHEDULED,TIMED"
    assert call["params"]["season"] == "2026"
    assert call["params"]["limit"] == 200
    assert call["headers"]["X-Auth-Token"] == "test_token_xyz"


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def test_dataframe_columns(mock_get: list[dict], tmp_path: Path) -> None:
    df = _make_scraper(tmp_path).get_fixtures()
    expected = {
        "source_fixture_id",
        "competition",
        "league",
        "season",
        "gameweek",
        "status",
        "kickoff",
        "date",
        "home_team",
        "away_team",
        "match_id",
    }
    assert set(df.columns) == expected


def test_matchday_selection(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def _mock(url: str, **kwargs: object) -> requests.Response:
        return _response(_TWO_MD_RESPONSE)

    monkeypatch.setattr(requests, "get", _mock)
    df = _make_scraper(tmp_path).get_fixtures()
    assert len(df) == 1  # only matchday 1
    assert df["gameweek"].iloc[0] == 1


def test_gameweek_values(mock_get: list[dict], tmp_path: Path) -> None:
    df = _make_scraper(tmp_path).get_fixtures()
    assert list(df["gameweek"]) == [2, 2]


def test_timed_fixtures_included(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def _mock(url: str, **kwargs: object) -> requests.Response:
        return _response(_TIMED_RESPONSE)

    monkeypatch.setattr(requests, "get", _mock)
    df = _make_scraper(tmp_path).get_fixtures()
    assert list(df["status"]) == ["TIMED"]


def test_kickoff_ordered(mock_get: list[dict], tmp_path: Path) -> None:
    df = _make_scraper(tmp_path).get_fixtures()
    assert df["kickoff"].is_monotonic_increasing


def test_kickoff_paris_timezone(mock_get: list[dict], tmp_path: Path) -> None:
    """UTC 19:00 in August → Paris 21:00 (UTC+2)."""
    df = _make_scraper(tmp_path).get_fixtures()
    first = df["kickoff"].iloc[0]
    assert "+02:00" in str(first)
    assert "21:00:00" in str(first)


def test_mixed_daylight_saving_offsets(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def _mock(url: str, **kwargs: object) -> requests.Response:
        return _response(_MIXED_OFFSET_RESPONSE)

    monkeypatch.setattr(requests, "get", _mock)
    df = _make_scraper(tmp_path).get_fixtures()
    assert list(df["kickoff"].dt.strftime("%H:%M")) == ["21:00", "22:00"]


def test_match_id_present(mock_get: list[dict], tmp_path: Path) -> None:
    df = _make_scraper(tmp_path).get_fixtures()
    assert df["match_id"].str.match(r".+ - .+ - \d{4}-\d{2}-\d{2}").all()


def test_team_mapping_applied(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def _mock(url: str, **kwargs: object) -> requests.Response:
        return _response(_SAMPLE_RESPONSE)

    monkeypatch.setattr(requests, "get", _mock)
    mapping = {"Paris Saint-Germain FC": "PSG", "Olympique de Marseille": "Marseille"}
    df = _make_scraper(tmp_path, mapping_teams=mapping).get_fixtures()
    assert df["home_team"].iloc[0] == "PSG"
    assert df["away_team"].iloc[0] == "Marseille"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_http_error_raised(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def _mock(url: str, **kwargs: object) -> requests.Response:
        return _response({}, status_code=403)

    monkeypatch.setattr(requests, "get", _mock)
    with pytest.raises(FootballDataOrgError, match="Request failed"):
        _make_scraper(tmp_path).get_fixtures()


def test_invalid_json_raised(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def _mock(url: str, **kwargs: object) -> requests.Response:
        return _response(b"not json")

    monkeypatch.setattr(requests, "get", _mock)
    with pytest.raises(FootballDataOrgError, match="Invalid JSON"):
        _make_scraper(tmp_path).get_fixtures()


def test_empty_matches(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def _mock(url: str, **kwargs: object) -> requests.Response:
        return _response(
            {"competition": {"id": 2015, "name": "Ligue 1"}, "filters": {}, "matches": []}
        )

    monkeypatch.setattr(requests, "get", _mock)
    df = _make_scraper(tmp_path).get_fixtures()
    assert len(df) == 0
    assert "match_id" in df.columns


# ---------------------------------------------------------------------------
# Cache behaviour
# ---------------------------------------------------------------------------


def test_cache_hit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[int] = []

    def _mock(url: str, **kwargs: object) -> requests.Response:
        calls.append(1)
        return _response(_SAMPLE_RESPONSE)

    monkeypatch.setattr(requests, "get", _mock)
    scraper = _make_scraper(tmp_path)
    scraper.get_fixtures()
    scraper.get_fixtures()
    assert len(calls) == 1


def test_cache_bypass_on_force_reload(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[int] = []

    def _mock(url: str, **kwargs: object) -> requests.Response:
        calls.append(1)
        return _response(_SAMPLE_RESPONSE)

    monkeypatch.setattr(requests, "get", _mock)
    scraper = _make_scraper(tmp_path, force_reload=True)
    scraper.get_fixtures()
    scraper.get_fixtures()
    assert len(calls) == 2


def test_cache_expiry(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[int] = []

    def _mock(url: str, **kwargs: object) -> requests.Response:
        calls.append(1)
        return _response(_SAMPLE_RESPONSE)

    monkeypatch.setattr(requests, "get", _mock)
    scraper = _make_scraper(tmp_path, ttl=0)
    scraper.get_fixtures()
    scraper.get_fixtures()
    assert len(calls) == 2


def test_cache_invalid_file_skipped(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[int] = []

    def _mock(url: str, **kwargs: object) -> requests.Response:
        calls.append(1)
        return _response(_SAMPLE_RESPONSE)

    monkeypatch.setattr(requests, "get", _mock)
    scraper = _make_scraper(tmp_path)
    scraper.get_fixtures()

    cache_file = Path(scraper._cache_path())
    cache_file.write_text("{not json", encoding="utf-8")
    now = time.time()
    os.utime(str(cache_file), (now, now))

    scraper.get_fixtures()
    assert len(calls) == 2


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_competitions_registered() -> None:
    from footix.data_io.utils_scrapper import MAPPING_COMPETITIONS

    info = MAPPING_COMPETITIONS["FRA Ligue 1"]
    assert "football_data_org" in info
    assert info["football_data_org"]["code"] == "FL1"
