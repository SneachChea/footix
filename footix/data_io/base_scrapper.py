import pathlib

import pandas as pd
import requests

from footix.data_io.utils_scrapper import MAPPING_COMPETITIONS


class Scraper:
    base_url: str = ""
    scraper_name: str | None = None

    def __init__(self, path: str, mapping_teams: dict[str, str] | None) -> None:
        self.mapping_teams = mapping_teams
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        }
        self.path = self.manage_path(path=path)

    def _check_competitions(self, competition_name: str):
        list_comp = self.competitions()
        if competition_name not in list_comp:
            raise ValueError(f"{competition_name} not available for scraper {self.scraper_name}")

    @classmethod
    def competitions(cls) -> list[str]:
        if cls.scraper_name is None:
            raise AttributeError("Scraper name undefined")
        list_compet = []
        for compet, info in MAPPING_COMPETITIONS.items():
            if cls.scraper_name in info.keys():
                list_compet.append(compet)
        return list_compet

    def replace_name_team(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        tmp_df = df.copy()
        if self.mapping_teams is not None:
            for column in columns:
                tmp_df[column] = tmp_df[column].replace(self.mapping_teams)
        return tmp_df

    def get(self, url: str) -> str:
        return requests.get(url, headers=self.headers).content.decode("utf-8-sig")

    @staticmethod
    def manage_path(path: str) -> pathlib.Path:
        tmp_pth = pathlib.Path(path)
        if tmp_pth.is_file():
            raise ValueError("Path should be a directory")
        if tmp_pth.exists():
            return tmp_pth
        else:
            tmp_pth.mkdir(parents=True, exist_ok=True)
        return tmp_pth
