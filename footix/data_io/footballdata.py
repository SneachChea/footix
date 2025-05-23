import io

import pandas as pd

import footix.data_io.utils_scrapper as utils_scrapper
from footix.data_io.base_scrapper import Scraper


class ScrapFootballData(Scraper):
    """Scraper for downloading and processing football match data from football-data.co.uk.

    This class handles the retrieval, local storage, and preprocessing of football match data
    for a given competition and season. It supports automatic downloading, file management,
    column sanitization, and team name mapping.

    Args:
        competition (str): The competition code (e.g., 'E0' for Premier League).
        season (str): The season string (e.g., '2020/2021', '2020-2021', or '2021').
        path (str): Directory path to store the downloaded CSV files.
        force_reload (bool, optional): If True, forces re-download of data even if file exists.
        mapping_teams (dict[str, str] | None, optional): Optional mapping for team name
        normalization.

    Attributes:
        base_url (str): Base URL for football-data.co.uk.
        scraper_name (str): Name identifier for the scraper.
        competition (str): Competition code.
        season (str): Processed season string.
        path (Path): Path object for data storage.
        force_reload (bool): Whether to force data reload.
        infered_url (str): Constructed URL for the CSV file.
        df (pd.DataFrame): Loaded and processed match data.

    Methods:
        download(): Downloads and saves the competition data as a CSV file.
        load() -> pd.DataFrame: Loads the data from file or downloads if not present.
        sanitize_columns(): Converts DataFrame columns to snake_case.
        get_fixtures() -> pd.DataFrame: Returns the processed match data.

    """

    base_url: str = "https://www.football-data.co.uk/mmz4281/"
    scraper_name = "footballdata"

    def __init__(
        self,
        competition: str,
        season: str,
        path: str,
        force_reload: bool = False,
        mapping_teams: dict[str, str] | None = None,
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
        self.df = utils_scrapper.add_mathc_id(self.df)

    def download(self):
        response = self.get(self.infered_url)
        df = (
            pd.read_csv(io.StringIO(response), encoding="utf-8")
            .sort_index()
            .pipe(self.replace_name_team, columns=["home_team", "away_team"])
        )
        df.to_csv(
            self.path / (self.competition + "_" + self.season + ".csv"),
            index=False,
            encoding="utf-8",
        )

    def load(self) -> pd.DataFrame:
        if self._check_if_file_exist() and not self.force_reload:
            df = pd.read_csv(self.path / (self.competition + "_" + self.season + ".csv"))
        else:
            self.download()
            df = pd.read_csv(self.path / (self.competition + "_" + self.season + ".csv"))
        return df

    def sanitize_columns(self):
        self.df.columns = [utils_scrapper.to_snake_case(x) for x in self.df.columns]

    def get_fixtures(self) -> pd.DataFrame:
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
