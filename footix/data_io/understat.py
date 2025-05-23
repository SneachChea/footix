import json
import re
from datetime import datetime
from functools import lru_cache
from typing import Any
from urllib.parse import urljoin

import numpy as np
import pandas as pd
from lxml import html

import footix.data_io.utils_scrapper as utils_scrapper
from footix.data_io.base_scrapper import Scraper


class ShotDataNotFound(RuntimeError):
    """Raised when the expected shotsData <script> block is not present."""


class FixtureDataNotFound(RuntimeError):
    """Raised when the fixture data are not present."""


class ScrapUnderstat(Scraper):
    """
    Scraper for downloading and processing football match data from understat.com.
    This class function is heavily inspired/copied from its counterpart from penalty blog:
    https://github.com/martineastwood/penaltyblog

    This class retrieves, parses, and processes football match data for a given competition
    and season from Understat. It extracts fixture details, expected goals (xG), forecasts,
    and normalizes team names. The data is returned as a processed pandas DataFrame.

    Args:
        competition (str): The competition code (e.g., 'EPL' for Premier League).
        season (str): The season string (e.g., '2020/2021', '2020-2021', or '2021').
        path (str): Directory path for any required file operations.
        force_reload (bool, optional): If True, forces re-download or reprocessing of data.
        mapping_teams (dict[str, str] | None, optional): Optional mapping for team name
        normalization.

    Attributes:
        base_url (str): Base URL for understat.com.
        scraper_name (str): Name identifier for the scraper.
        season (str): Processed season string.
        force_reload (bool): Whether to force data reload.
        slug (str): Slug for the competition used in URL construction.

    Methods:
        sanitize_columns(df): Converts DataFrame columns to snake_case.
        get_fixtures() -> pd.DataFrame: Downloads, parses, and returns processed match data.
        _process_season(season: str) -> str: Processes the season string for URL usage.
    """

    base_url: str = "https://understat.com/"
    scraper_name = "understat"

    def __init__(
        self,
        competition: str,
        season: str,
        path: str,
        force_reload: bool = False,
        mapping_teams: dict[str, str] | None = None,
    ):
        self._check_competitions(competition_name=competition)
        super().__init__(path=path, mapping_teams=mapping_teams)
        self.season = self._process_season(season)
        self.force_reload = force_reload
        self.slug = utils_scrapper.MAPPING_COMPETITIONS[competition]["understat"]["slug"]
        self.competition = competition

    @staticmethod
    def sanitize_columns(df: pd.DataFrame):
        df.columns = [utils_scrapper.to_snake_case(x) for x in df.columns]

    @lru_cache(maxsize=256)
    def get_fixtures(self):
        implied_url = urljoin(self.base_url, f"league/{self.slug}/{self.season}")
        content = self.get(implied_url)
        tree = html.fromstring(content)
        events = None
        for s in tree.cssselect("script"):
            if "datesData" in s.text:
                script = s.text
                script = " ".join(script.split())
                script = str(script.encode(), "unicode-escape")
                script = re.match(r"var datesData = JSON\.parse\('(?P<json>.*?)'\)", script)
                if script is not None:
                    script = script.group("json")
                events = json.loads(script)
                break

        if events is None:
            raise FixtureDataNotFound

        fixtures = list()
        for e in events:
            if not e["isResult"]:
                continue

            tmp: dict[str, Any] = dict()
            tmp["understat_id"] = str(e["id"])
            tmp["datetime"] = e["datetime"]
            tmp["home_team"] = e["h"]["title"]
            tmp["away_team"] = e["a"]["title"]
            tmp["fthg"] = int(e["goals"]["h"])
            tmp["ftag"] = int(e["goals"]["a"])
            tmp["fthxg"] = float(e["xG"]["h"])
            tmp["ftaxg"] = float(e["xG"]["a"])
            tmp["forecast_w"] = float(e["forecast"]["w"])
            tmp["forecast_d"] = float(e["forecast"]["d"])
            tmp["forecast_l"] = float(e["forecast"]["l"])
            fixtures.append(tmp)

        df = (
            pd.DataFrame(fixtures)
            .pipe(self.replace_name_team, columns=["home_team", "away_team"])
            .sort_index()
        )

        def _get_date(date: str) -> str:
            dt = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
            return dt.strftime("%d/%m/%Y")

        def get_ftr(row) -> str:
            fthg = row["fthg"]
            ftag = row["ftag"]
            if fthg > ftag:
                return "H"
            if fthg == ftag:
                return "D"
            return "A"

        self.sanitize_columns(df)
        df["ftr"] = df.apply(get_ftr, axis=1)
        df["date"] = df["datetime"].apply(_get_date)
        df = utils_scrapper.add_mathc_id(df)
        return df

    def _process_season(self, season: str) -> str:
        clean_season = season.replace(" ", "-").replace("/", "-").split("-")
        return clean_season[0]

    @lru_cache(maxsize=256)
    def get_shots(self, understat_id: str) -> pd.DataFrame:
        url = urljoin(self.base_url, f"match/{understat_id}")
        content = self.get(url)
        tree = html.fromstring(content)
        events = None

        for s in tree.cssselect("script"):
            if "shotsData" in s.text:
                script = s.text
                script = " ".join(script.split())
                script = str(script.encode(), "unicode-escape")
                script = re.match(r"var shotsData = JSON\.parse\('(?P<json>.*?)'\)", script)
                if script is not None:
                    script = script.group("json")
                events = json.loads(script)
                break

        if events is None:
            raise ShotDataNotFound

        shots = list()
        shots.extend(events["h"])
        shots.extend(events["a"])

        col_renames = {
            "h_team": "home_team",
            "a_team": "away_team",
            "h_goals": "goals_home",
            "a_goals": "goals_away",
            "match_id": "understat_id",
        }

        df = (
            pd.DataFrame(shots)
            .rename(columns=col_renames)
            .assign(season=self.season)
            .assign(competition=self.competition)
            .assign(date=lambda x: pd.to_datetime(x["date"]).dt.strftime("%d-%m-%Y"))
            .pipe(self.replace_name_team, columns=["home_team", "away_team"])
            .sort_index()
        )
        df["h_a"] = np.where(df["h_a"] == "h", df["home_team"], df["away_team"])
        df = utils_scrapper.add_mathc_id(df)
        self.sanitize_columns(df)
        return df
