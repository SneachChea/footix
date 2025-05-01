import pathlib

import pandas as pd
import requests
import io

import footix.data_io.utils_scrapper as utils_scrapper
from footix.data_io.base_scrapper import Scraper

class ScrapFootballData(Scraper):
    base_url: str = "https://www.football-data.co.uk/mmz4281/"
    scraper_name = "footballdata"
    def __init__(
        self, competition: str, season: str, path: str, force_reload: bool = False, mapping_teams:dict[str, str] |None= None
    ) -> None:
        super().__init__(path=path, mapping_teams=mapping_teams)
        self._check_competitions(competition_name=competition)
        self.competition = competition
        slug = utils_scrapper.MAPPING_COMPETITIONS[self.competition]["footballdata"]["slug"]
        self.season = _process_season(season)
        self.path = self.manage_path(path)
        self.force_reload = force_reload
        self.infered_url = self.base_url + self.season + "/" + slug + ".csv"
        self.df = self.load()
        self.sanitize_columns()



    def download(self):
        response = self.get(self.infered_url)
        df = pd.read_csv(io.StringIO(response), encoding='utf-8').sort_index().pipe(self.replace_name_team, columns=["home_team", "away_team"])
        df.to_csv(self.path / (self.competition + "_" + self.season + ".csv"), index=False, encoding="utf-8")

    def load(self) -> pd.DataFrame:
        if self._check_if_file_exist() and not self.force_reload:
            df = pd.read_csv(self.path / (self.competition + "_" + self.season + ".csv"))
        else:
            self.download()
            df = pd.read_csv(self.path / (self.competition + "_" + self.season + ".csv"))
        return df

    def sanitize_columns(self):
        self.df.columns = [utils_scrapper.to_snake_case(x) for x in self.df.columns]

    def get_fixtures(self)->pd.DataFrame:
        return self.df

    def _check_if_file_exist(self) -> bool:
        name = self.competition + "_" + self.season + ".csv"
        expected_file = self.path / name
        if not expected_file.is_file():
            return False
        return True


def _process_season(season: str) -> str:
    """Process a season string to extract a standardized format.

    Args:
        season (str): A string representing a football season in the format
            'YYYY/YYYY' or 'YYYY-YYYY'. For example: '2020/2021' or '2020-2021'

    Raises:
        ValueError: if the season string cannot be split into exactly two years.

    Returns:
        str: the formatted season string.

    """
    if len(season) == 4 and season.isdigit():
        if int(season[-2:]) - int(season[:2]) != 1:
            raise ValueError("Years must be consecutive")
        return season
    # Remove any whitespace and split on common separators
    clean_season = season.replace(" ", "-").replace("/", "-").split("-")
    # Extract last 2 digits from each year
    year1 = clean_season[0][-2:]
    year2 = clean_season[-1][-2:]
    # Check if the years are consecutive
    if int(year2) - int(year1) != 1:
        raise ValueError("Years must be consecutive")
    return year1 + year2
