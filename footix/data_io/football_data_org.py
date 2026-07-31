"""football-data.org live data provider (football-data.org).

Provides access to the football-data.org v4 REST API for live/pre-match data.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from footix.data_io.base_scrapper import Scraper
from footix.data_io.utils_scrapper import MAPPING_COMPETITIONS, add_match_id

_PARIS_TZ = ZoneInfo("Europe/Paris")
# ponytail: limit generously above one full Ligue-1 matchday (~10 fixtures).
_FIXTURE_LIMIT = 200
_UPCOMING_STATUSES = "SCHEDULED,TIMED"


class FootballDataOrgError(RuntimeError):
    """Raised when the API returns an error or an invalid response."""


class ScrapFootballDataOrg(Scraper):
    """Fetch upcoming fixtures from football-data.org.

    Args:
        competition: Footix competition key (e.g. ``"FRA Ligue 1"``).
        season: Season as 4-digit year string (e.g. ``"2026"``
            for the 2026‑2027 season).
        api_token: football-data.org API token.
        path: Directory for cache storage.
        mapping_teams: Optional dict to rename teams before
            ``add_match_id``.
        force_reload: Bypass disk cache and re-fetch.
        ttl: Cache TTL in seconds (default 6 hours).
    """

    base_url: str = "https://api.football-data.org/v4"
    scraper_name: str | None = "football_data_org"

    def __init__(
        self,
        competition: str,
        season: str,
        api_token: str,
        path: str,
        mapping_teams: dict[str, str] | None = None,
        force_reload: bool = False,
        ttl: int = 21600,
    ) -> None:
        self._check_competitions(competition_name=competition)
        super().__init__(path=path, mapping_teams=mapping_teams)
        self.competition = competition
        self.season = season
        self._api_token = api_token
        self.force_reload = force_reload
        self.ttl = ttl
        assert self.scraper_name is not None  # set by class attribute
        info = MAPPING_COMPETITIONS[competition][self.scraper_name]
        self._code: str = info["code"]  # type: ignore[index]

    # ------------------------------------------------------------------
    # HTTP / cache
    # ------------------------------------------------------------------

    def _api_request(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}{endpoint}"
        headers = {"X-Auth-Token": self._api_token}
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=15)
            resp.raise_for_status()
            data: dict[str, Any] = resp.json()
        except json.JSONDecodeError as exc:
            raise FootballDataOrgError(f"Invalid JSON: {exc}") from exc
        except requests.RequestException as exc:
            raise FootballDataOrgError(f"Request failed: {exc}") from exc
        return data

    def _cache_path(self) -> str:
        name = f".football_data_org_{self.competition}_{self.season}_scheduled_timed.json"
        return str(self.path / name)

    def _load_cache(self) -> dict[str, Any] | None:
        cpath = Path(self._cache_path())
        if not cpath.exists() or self.force_reload:
            return None
        if time.time() - cpath.stat().st_mtime > self.ttl:
            return None
        try:
            with open(cpath, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return None

    def _save_cache(self, data: dict[str, Any]) -> None:
        cpath = Path(self._cache_path())
        tmp = cpath.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        tmp.replace(cpath)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def get_fixtures(self) -> pd.DataFrame:
        """Return the next matchday's scheduled fixtures as a DataFrame.

        Fetches from the API (or disk cache), finds the earliest
        matchday with SCHEDULED or TIMED fixtures, normalises team names, and
        adds a stable ``match_id``.
        """
        cached = self._load_cache()
        if cached is not None:
            data = cached
        else:
            data = self._api_request(
                f"/competitions/{self._code}/matches",
                {
                    "status": _UPCOMING_STATUSES,
                    "season": self.season,
                    "limit": _FIXTURE_LIMIT,
                },
            )
            self._save_cache(data)

        matches_raw: list[dict[str, Any]] = data.get("matches") or []
        if not matches_raw:
            return pd.DataFrame(
                columns=[
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
            )

        rows = []
        for m in matches_raw:
            utc_dt = datetime.fromisoformat(m["utcDate"].replace("Z", "+00:00"))
            kickoff_dt = utc_dt.astimezone(_PARIS_TZ)
            rows.append(
                {
                    "source_fixture_id": m["id"],
                    "competition": self.competition,
                    "league": data["competition"]["name"],
                    "season": str(self.season),
                    "gameweek": m["matchday"],
                    "status": m["status"],
                    "kickoff": kickoff_dt.isoformat(timespec="seconds"),
                    "home_team": m["homeTeam"]["name"],
                    "away_team": m["awayTeam"]["name"],
                }
            )

        df = pd.DataFrame(rows)
        df["kickoff"] = pd.to_datetime(df["kickoff"], utc=True).dt.tz_convert(_PARIS_TZ)
        df = df.sort_values("kickoff").reset_index(drop=True)

        first_md = df["gameweek"].iloc[0]
        df = df[df["gameweek"] == first_md].copy()

        df["date"] = df["kickoff"].dt.strftime("%d/%m/%Y")
        df = self.replace_name_team(df, columns=["home_team", "away_team"])
        df = add_match_id(df)
        return df
