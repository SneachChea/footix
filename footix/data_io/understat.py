
import pathlib

import pandas as pd
import requests
import io
import re
import json
from typing import Any
from lxml import html
import footix.data_io.utils_scrapper as utils_scrapper
from footix.data_io.base_scrapper import Scraper

class ScrapUnderstat(Scraper):
    base_url: str = "https://understat.com/"
    scraper_name = "understat"
    def __init__(self, competition: str, season: str, path: str, force_reload: bool = False, mapping_teams: dict[str, str] | None = None):
        self._check_competitions(competition_name=competition)
        super().__init__(path=path, mapping_teams=mapping_teams)
        self.season = self._process_season(season)
        self.force_reload = force_reload
        self.slug = utils_scrapper.MAPPING_COMPETITIONS[competition]["understat"]["slug"]


    @staticmethod
    def sanitize_columns(df: pd.DataFrame):
        df.columns = [utils_scrapper.to_snake_case(x) for x in df.columns]

    def get_fixtures(self):
        implied_url = (
            self.base_url
            + "league/"
            + self.slug
            + "/"
            + self.season
        )

        content = self.get(implied_url)
        tree = html.fromstring(content)
        events = None
        for s in tree.cssselect("script"):
            if "datesData" in s.text:
                script = s.text
                script = " ".join(script.split())
                script = str(script.encode(), "unicode-escape")
                script = re.match(
                    r"var datesData = JSON\.parse\('(?P<json>.*?)'\)", script
                )
                if script is not None:
                    script = script.group("json")
                events = json.loads(script)
                break

        if events is None:
            raise ValueError("Error: no data found")

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
            pd.DataFrame(fixtures).pipe(self.replace_name_team, columns=["home_team", "away_team"])
            .sort_index()
        )
        self.sanitize_columns(df)
        return df




    def _process_season(self, season: str)->str:
        clean_season = season.replace(" ", "-").replace("/", "-").split("-")
        return clean_season[0]
