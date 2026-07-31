"""APIFootball.com provider for French Ligue 2 fixtures."""

from __future__ import annotations

import json
import re
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from footix.data_io.base_scrapper import Scraper
from footix.data_io.utils_scrapper import MAPPING_COMPETITIONS, add_match_id

_PARIS_TZ = ZoneInfo("Europe/Paris")
_LOOKAHEAD_DAYS = 15
_TERMINAL_STATUSES = {
    "after et",
    "after pen.",
    "awarded",
    "cancelled",
    "finished",
    "postponed",
}
_LIVE_STATUSES = {"half time", "live", "paused"}
_MINUTE_STATUS = re.compile(r"^\d+(?:\+\d+)?'$", re.ASCII)
_EMPTY_COLUMNS = [
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
]


class APIFootballComError(RuntimeError):
    """Raised when APIFootball.com returns an error or invalid payload."""


class ScrapAPIFootballCom(Scraper):
    """Fetch the next French Ligue 2 matchday from APIFootball.com.

    Args:
        competition: Must be ``"FRA Ligue 2"`` for this provider.
        api_key: APIFootball.com API key.
        path: Directory for cache storage.
        mapping_teams: Optional mapping to training team names.
        force_reload: Bypass disk cache and re-fetch.
        ttl: Cache TTL in seconds (default 6 hours).
    """

    base_url: str = "https://apiv3.apifootball.com/"
    scraper_name: str | None = "apifootball_com"

    def __init__(
        self,
        competition: str,
        api_key: str,
        path: str,
        mapping_teams: dict[str, str] | None = None,
        force_reload: bool = False,
        ttl: int = 21600,
    ) -> None:
        self._check_competitions(competition_name=competition)
        if not api_key.strip():
            raise ValueError("api_key is required")
        super().__init__(path=path, mapping_teams=mapping_teams)
        self.competition = competition
        self._api_key = api_key
        self.force_reload = force_reload
        self.ttl = ttl
        assert self.scraper_name is not None  # set by class attribute
        info = MAPPING_COMPETITIONS[competition][self.scraper_name]
        self._league_id: str = info["league_id"]  # type: ignore[index]

    @staticmethod
    def _date_window() -> tuple[date, date]:
        start = datetime.now(_PARIS_TZ).date()
        return start, start + timedelta(days=_LOOKAHEAD_DAYS)

    def _cache_path(self, start: date, end: date) -> Path:
        return self.path / (f".apifootball_com_{self._league_id}_{start:%Y%m%d}_{end:%Y%m%d}.json")

    def _api_request(self, start: date, end: date) -> list[dict[str, Any]]:
        params = {
            "action": "get_events",
            "APIkey": self._api_key,
            "league_id": self._league_id,
            "from": start.isoformat(),
            "to": end.isoformat(),
            "timezone": "Europe/Paris",
        }
        try:
            response = requests.get(self.base_url, params=params, timeout=15)
        except requests.RequestException as exc:
            # APIFootball.com requires the key in the URL; never include the
            # exception text because it may contain that URL.
            raise APIFootballComError(f"Request failed ({type(exc).__name__})") from exc

        if not 200 <= response.status_code < 300:
            raise APIFootballComError(f"Request failed with HTTP {response.status_code}")

        try:
            payload = response.json()
        except ValueError as exc:
            raise APIFootballComError("Invalid JSON response") from exc

        if isinstance(payload, dict):
            raise APIFootballComError("APIFootball.com returned an error payload")
        if not isinstance(payload, list) or not all(isinstance(event, dict) for event in payload):
            raise APIFootballComError("Invalid APIFootball.com events payload")
        return payload

    def _load_cache(self, cache_path: Path) -> list[dict[str, Any]] | None:
        if not cache_path.exists() or self.force_reload:
            return None
        if time.time() - cache_path.stat().st_mtime > self.ttl:
            return None
        try:
            with cache_path.open(encoding="utf-8") as file:
                payload = json.load(file)
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(payload, list) or not all(isinstance(event, dict) for event in payload):
            return None
        return payload

    @staticmethod
    def _save_cache(cache_path: Path, payload: list[dict[str, Any]]) -> None:
        temporary_path = cache_path.with_suffix(".tmp")
        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False)
        temporary_path.replace(cache_path)

    @staticmethod
    def _is_upcoming(status: object) -> bool:
        normalized = str(status or "").strip().lower()
        if normalized in _TERMINAL_STATUSES or normalized in _LIVE_STATUSES:
            return False
        return _MINUTE_STATUS.fullmatch(normalized) is None

    @staticmethod
    def _parse_kickoff(event: dict[str, Any]) -> datetime | None:
        match_date = str(event.get("match_date") or "").strip()
        match_time = str(event.get("match_time") or "").strip()
        if not match_date or not match_time or match_time.upper() == "TBD":
            return None
        try:
            return datetime.strptime(f"{match_date} {match_time}", "%Y-%m-%d %H:%M").replace(
                tzinfo=_PARIS_TZ
            )
        except ValueError:
            return None

    @staticmethod
    def _parse_gameweek(value: object) -> int | None:
        match = re.search(r"\d+", str(value or ""))
        return int(match.group()) if match else None

    def get_fixtures(self) -> pd.DataFrame:
        """Return the first upcoming Ligue 2 matchday in the lookahead window."""
        start, end = self._date_window()
        cache_path = self._cache_path(start, end)
        events = self._load_cache(cache_path)
        if events is None:
            events = self._api_request(start, end)
            self._save_cache(cache_path, events)

        rows: list[dict[str, Any]] = []
        now = datetime.now(_PARIS_TZ)
        for event in events:
            if not self._is_upcoming(event.get("match_status")):
                continue
            kickoff = self._parse_kickoff(event)
            gameweek = self._parse_gameweek(event.get("match_round"))
            source_fixture_id = event.get("match_id")
            home_team = event.get("match_hometeam_name")
            away_team = event.get("match_awayteam_name")
            if (
                kickoff is None
                or kickoff < now
                or gameweek is None
                or not source_fixture_id
                or not home_team
                or not away_team
            ):
                continue
            rows.append(
                {
                    "source_fixture_id": source_fixture_id,
                    "competition": self.competition,
                    "league": event.get("league_name") or "Ligue 2",
                    "season": event.get("league_year") or "",
                    "gameweek": gameweek,
                    "status": event.get("match_status") or "",
                    "kickoff": kickoff.isoformat(timespec="seconds"),
                    "home_team": home_team,
                    "away_team": away_team,
                }
            )

        if not rows:
            return pd.DataFrame(columns=_EMPTY_COLUMNS)

        df = pd.DataFrame(rows)
        df["kickoff"] = pd.to_datetime(df["kickoff"], utc=True).dt.tz_convert(_PARIS_TZ)
        df = df.sort_values("kickoff").reset_index(drop=True)
        first_gameweek = df["gameweek"].iloc[0]
        df = df[df["gameweek"] == first_gameweek].copy()
        df["date"] = df["kickoff"].dt.strftime("%d/%m/%Y")
        df = self.replace_name_team(df, columns=["home_team", "away_team"])
        return add_match_id(df)
