"""Tests for the APIFootball.com Ligue 2 provider."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
import requests

from footix.data_io.apifootball_com import (
    APIFootballComError,
    ScrapAPIFootballCom,
)

_PARIS_TZ = ZoneInfo("Europe/Paris")


def _event(
    *,
    match_id: str,
    match_date: str,
    match_time: str,
    match_round: str,
    status: str = "",
    home: str = "Paris FC",
    away: str = "Pau FC",
) -> dict[str, str]:
    return {
        "match_id": match_id,
        "league_id": "164",
        "league_name": "Ligue 2",
        "league_year": "2026/2027",
        "match_date": match_date,
        "match_status": status,
        "match_time": match_time,
        "match_round": match_round,
        "match_hometeam_name": home,
        "match_awayteam_name": away,
    }


def _make_scraper(
    tmp_path: Path,
    mapping_teams: dict[str, str] | None = None,
    force_reload: bool = False,
    ttl: int = 21600,
) -> ScrapAPIFootballCom:
    return ScrapAPIFootballCom(
        competition="FRA Ligue 2",
        api_key="test_api_key",
        path=str(tmp_path),
        mapping_teams=mapping_teams,
        force_reload=force_reload,
        ttl=ttl,
    )


def _response(payload: object, status_code: int = 200) -> requests.Response:
    response = requests.Response()
    response.status_code = status_code
    response._content = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    response.encoding = "utf-8"
    return response


def _future_dates() -> tuple[str, str]:
    start = datetime.now(_PARIS_TZ).date() + timedelta(days=2)
    return start.isoformat(), (start + timedelta(days=1)).isoformat()


def test_provider_is_ligue_2_only(tmp_path: Path) -> None:
    assert ScrapAPIFootballCom.competitions() == ["FRA Ligue 2"]
    with pytest.raises(ValueError, match="FRA Ligue 1"):
        ScrapAPIFootballCom(
            competition="FRA Ligue 1",
            api_key="test_api_key",
            path=str(tmp_path),
        )


def test_request_parameters_and_league_id(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []
    first_date, _ = _future_dates()

    def mock_get(url: str, **kwargs: object) -> requests.Response:
        calls.append({"url": url, **kwargs})
        return _response(
            [_event(match_id="1", match_date=first_date, match_time="20:00", match_round="1")]
        )

    monkeypatch.setattr(requests, "get", mock_get)
    _make_scraper(tmp_path).get_fixtures()

    assert calls[0]["url"] == "https://apiv3.apifootball.com/"
    params = calls[0]["params"]
    assert isinstance(params, dict)
    assert params["action"] == "get_events"
    assert params["APIkey"] == "test_api_key"
    assert params["league_id"] == "164"
    assert params["timezone"] == "Europe/Paris"
    assert params["from"] <= params["to"]


def test_next_matchday_is_selected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    first_date, second_date = _future_dates()

    def mock_get(url: str, **kwargs: object) -> requests.Response:
        return _response(
            [
                _event(
                    match_id="2",
                    match_date=second_date,
                    match_time="20:00",
                    match_round="Round 2",
                ),
                _event(
                    match_id="1",
                    match_date=first_date,
                    match_time="19:00",
                    match_round="Round 1",
                ),
            ]
        )

    monkeypatch.setattr(requests, "get", mock_get)
    df = _make_scraper(tmp_path).get_fixtures()
    assert list(df["source_fixture_id"]) == ["1"]
    assert list(df["gameweek"]) == [1]


def test_terminal_and_live_statuses_are_excluded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    first_date, _ = _future_dates()

    def mock_get(url: str, **kwargs: object) -> requests.Response:
        return _response(
            [
                _event(
                    match_id="finished",
                    match_date=first_date,
                    match_time="18:00",
                    match_round="1",
                    status="Finished",
                ),
                _event(
                    match_id="live",
                    match_date=first_date,
                    match_time="19:00",
                    match_round="1",
                    status="23'",
                ),
                _event(
                    match_id="upcoming",
                    match_date=first_date,
                    match_time="20:00",
                    match_round="1",
                ),
            ]
        )

    monkeypatch.setattr(requests, "get", mock_get)
    df = _make_scraper(tmp_path).get_fixtures()
    assert list(df["source_fixture_id"]) == ["upcoming"]


def test_tbd_kickoff_is_excluded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    first_date, _ = _future_dates()

    def mock_get(url: str, **kwargs: object) -> requests.Response:
        return _response(
            [_event(match_id="tbd", match_date=first_date, match_time="TBD", match_round="1")]
        )

    monkeypatch.setattr(requests, "get", mock_get)
    assert _make_scraper(tmp_path).get_fixtures().empty


def test_timezone_and_match_id(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def mock_get(url: str, **kwargs: object) -> requests.Response:
        return _response(
            [
                _event(
                    match_id="1",
                    match_date="2027-03-28",
                    match_time="20:00",
                    match_round="1",
                )
            ]
        )

    monkeypatch.setattr(requests, "get", mock_get)
    df = _make_scraper(tmp_path).get_fixtures()
    assert "+02:00" in str(df["kickoff"].iloc[0])
    assert df["match_id"].iloc[0].endswith("2027-03-28")


def test_team_mapping(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    first_date, _ = _future_dates()

    def mock_get(url: str, **kwargs: object) -> requests.Response:
        return _response(
            [
                _event(
                    match_id="1",
                    match_date=first_date,
                    match_time="20:00",
                    match_round="1",
                )
            ]
        )

    monkeypatch.setattr(requests, "get", mock_get)
    df = _make_scraper(
        tmp_path,
        mapping_teams={"Paris FC": "Paris FC", "Pau FC": "Pau FC"},
    ).get_fixtures()
    assert df["home_team"].iloc[0] == "Paris FC"
    assert df["away_team"].iloc[0] == "Pau FC"


def test_http_error_does_not_expose_api_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def mock_get(url: str, **kwargs: object) -> requests.Response:
        return _response({"error": 403}, status_code=403)

    monkeypatch.setattr(requests, "get", mock_get)
    with pytest.raises(APIFootballComError) as error:
        _make_scraper(tmp_path).get_fixtures()
    assert "test_api_key" not in str(error.value)


def test_request_error_does_not_expose_api_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def mock_get(url: str, **kwargs: object) -> requests.Response:
        raise requests.RequestException("APIkey=test_api_key")

    monkeypatch.setattr(requests, "get", mock_get)
    with pytest.raises(APIFootballComError) as error:
        _make_scraper(tmp_path).get_fixtures()
    assert "test_api_key" not in str(error.value)


def test_error_payload_is_not_cached(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = 0

    def mock_get(url: str, **kwargs: object) -> requests.Response:
        nonlocal calls
        calls += 1
        return _response({"error": 404, "message": "Authentication failed"})

    monkeypatch.setattr(requests, "get", mock_get)
    scraper = _make_scraper(tmp_path)
    with pytest.raises(APIFootballComError):
        scraper.get_fixtures()
    with pytest.raises(APIFootballComError):
        scraper.get_fixtures()
    assert calls == 2


def test_invalid_json_is_rejected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def mock_get(url: str, **kwargs: object) -> requests.Response:
        return _response(b"not json")

    monkeypatch.setattr(requests, "get", mock_get)
    with pytest.raises(APIFootballComError, match="Invalid JSON"):
        _make_scraper(tmp_path).get_fixtures()


def test_empty_payload_returns_schema(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(requests, "get", lambda url, **kwargs: _response([]))
    df = _make_scraper(tmp_path).get_fixtures()
    assert df.empty
    assert "match_id" in df.columns


def test_cache_hit_and_force_reload(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    first_date, _ = _future_dates()
    calls = 0

    def mock_get(url: str, **kwargs: object) -> requests.Response:
        nonlocal calls
        calls += 1
        return _response(
            [_event(match_id="1", match_date=first_date, match_time="20:00", match_round="1")]
        )

    monkeypatch.setattr(requests, "get", mock_get)
    scraper = _make_scraper(tmp_path)
    scraper.get_fixtures()
    scraper.get_fixtures()
    assert calls == 1

    _make_scraper(tmp_path, force_reload=True).get_fixtures()
    assert calls == 2


def test_cache_expiry(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    first_date, _ = _future_dates()
    calls = 0

    def mock_get(url: str, **kwargs: object) -> requests.Response:
        nonlocal calls
        calls += 1
        return _response(
            [_event(match_id="1", match_date=first_date, match_time="20:00", match_round="1")]
        )

    monkeypatch.setattr(requests, "get", mock_get)
    scraper = _make_scraper(tmp_path, ttl=0)
    scraper.get_fixtures()
    scraper.get_fixtures()
    assert calls == 2


def test_registry_is_ligue_2_only() -> None:
    from footix.data_io.utils_scrapper import MAPPING_COMPETITIONS

    assert MAPPING_COMPETITIONS["FRA Ligue 2"]["apifootball_com"]["league_id"] == "164"
