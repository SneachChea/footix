"""Module to compute league standings from match results."""

from typing import List

import pandas as pd


def compute_standings(
    matches: pd.DataFrame,
    points_win: int = 3,
    points_draw: int = 1,
    tiebreakers: List[str] | None = None,
) -> pd.DataFrame:
    """Compute league standings table from a DataFrame of match results.

    The input DataFrame should have at least the following columns:
    'home_team', 'away_team', 'fthg' (Full Time Home Goals), 'ftag' (Full Time Away Goals).
    Rows with missing score values (NaN) are treated as unplayed and ignored.

    Args:
        matches: DataFrame containing match results.
        points_win: Points awarded for a win. Defaults to 3.
        points_draw: Points awarded for a draw. Defaults to 1.
        tiebreakers: Ordered list of criteria to break ties.
            Supported: 'points', 'goal_difference', 'goals_for'.
            Defaults to ['points', 'goal_difference', 'goals_for'].

    Returns:
        pd.DataFrame: Sorted standings table with columns:
            'team', 'played', 'wins', 'draws', 'losses', 'gf', 'ga', 'gd', 'points', 'position'
    """
    if tiebreakers is None:
        tiebreakers = ["points", "goal_difference", "goals_for"]

    # Filter out unplayed matches
    played_matches = matches.dropna(subset=["fthg", "ftag"]).copy()

    # Normalize column names if needed (though we expect snake_case)
    # Mapping for internal consistency
    teams = pd.concat([played_matches["home_team"], played_matches["away_team"]]).unique()
    table_data = []

    for team in teams:
        # Home games
        home_games = played_matches[played_matches["home_team"] == team]
        # Away games
        away_games = played_matches[played_matches["away_team"] == team]

        # Stats calculation
        played = len(home_games) + len(away_games)

        home_wins = (home_games["fthg"] > home_games["ftag"]).sum()
        away_wins = (away_games["ftag"] > away_games["fthg"]).sum()
        wins = home_wins + away_wins

        home_draws = (home_games["fthg"] == home_games["ftag"]).sum()
        away_draws = (away_games["ftag"] == away_games["fthg"]).sum()
        draws = home_draws + away_draws

        losses = played - wins - draws

        gf = home_games["fthg"].sum() + away_games["ftag"].sum()
        ga = home_games["ftag"].sum() + away_games["fthg"].sum()
        gd = gf - ga

        points = (wins * points_win) + (draws * points_draw)

        table_data.append(
            {
                "team": team,
                "played": int(played),
                "wins": int(wins),
                "draws": int(draws),
                "losses": int(losses),
                "gf": int(gf),
                "ga": int(ga),
                "gd": int(gd),
                "points": int(points),
            }
        )

    standings = pd.DataFrame(table_data)

    # Sorting based on tiebreakers
    sort_cols = []
    ascending_flags = []

    mapping = {"points": "points", "goal_difference": "gd", "goals_for": "gf"}

    for tb in tiebreakers:
        if tb in mapping:
            sort_cols.append(mapping[tb])
            ascending_flags.append(False)  # All these are "higher is better"

    # Secondary sort by team name for stable rank in case of perfect ties
    sort_cols.append("team")
    ascending_flags.append(True)

    standings = standings.sort_values(by=sort_cols, ascending=ascending_flags).reset_index(
        drop=True
    )
    standings.index += 1
    standings.insert(0, "position", standings.index)

    return standings


def get_team_form(matches: pd.DataFrame, team: str, last_n: int = 5) -> List[str]:
    """Get the recent form of a team.

    Args:
        matches: DataFrame containing match results.
        team: Team name.
        last_n: Number of matches to retrieve. Defaults to 5.

    Returns:
        List[str]: List of results ('W', 'D', 'L') from oldest to newest.
    """
    # Filter matches where team played
    team_matches = matches[(matches["home_team"] == team) | (matches["away_team"] == team)].copy()

    # Drop unplayed matches
    team_matches = team_matches.dropna(subset=["fthg", "ftag"])

    # Sort by date if possible
    if "date" in team_matches.columns:
        try:
            team_matches["date"] = pd.to_datetime(team_matches["date"], dayfirst=True)
            team_matches = team_matches.sort_values(by="date", ascending=True)
        except (ValueError, TypeError):
            pass

    # Get last N matches
    recent_matches = team_matches.tail(last_n)
    form = []

    for _, row in recent_matches.iterrows():
        is_home = row["home_team"] == team
        fthg = row["fthg"]
        ftag = row["ftag"]

        if fthg == ftag:
            form.append("D")
        elif (is_home and fthg > ftag) or (not is_home and ftag > fthg):
            form.append("W")
        else:
            form.append("L")

    return form
