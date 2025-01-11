import pathlib

import pandas as pd
import requests

import footix.data_io.utils_scrapper as utils_scrapper


class ScrapFootballData:
    base_url: str = "https://www.football-data.co.uk/mmz4281/"

    def __init__(
        self, competition: str, season: str, path: str, force_reload: bool = False
    ) -> None:
        self.competition = utils_scrapper.process_string(competition)
        slug = utils_scrapper.MAPPING_COMPETITIONS[self.competition]
        self.season = _process_season(season)
        self.path = self.manage_path(path)
        self.force_reload = force_reload
        self.infered_url = self.base_url + self.season + "/" + slug + ".csv"
        self.df = self.load()

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

    def download(self):
        response = requests.get(self.infered_url)
        with open(self.path / (self.competition + "_" + self.season + ".csv"), "wb") as file:
            file.write(response.content)

    def load(self) -> pd.DataFrame:
        if self._check_if_file_exist() and not self.force_reload:
            df = pd.read_csv(self.path / (self.competition + "_" + self.season + ".csv"))
        else:
            self.download()
            df = pd.read_csv(self.path / (self.competition + "_" + self.season + ".csv"))
        return df

    def get_data(self):
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
