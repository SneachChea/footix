import pandas as pd

from footix.metrics.standings import compute_standings, get_team_form


def test_basic_standings():
    """Test a simple case with a few matches."""
    data = [
        {"home_team": "Team A", "away_team": "Team B", "fthg": 2, "ftag": 0},  # A wins
        {"home_team": "Team B", "away_team": "Team C", "fthg": 1, "ftag": 1},  # Draw
        {"home_team": "Team C", "away_team": "Team A", "fthg": 0, "ftag": 1},  # A wins
    ]
    df = pd.DataFrame(data)
    standings = compute_standings(df)

    # Team A: 2 wins, 0 losses, 0 draws -> 6 pts. GF: 3, GA: 0, GD: 3
    # Team B: 0 wins, 1 loss, 1 draw -> 1 pt. GF: 1, GA: 3, GD: -2
    # Team C: 0 wins, 1 loss, 1 draw -> 1 pt. GF: 1, GA: 2, GD: -1

    # Ranking expected: A (6), C (1, GD -1), B (1, GD -2)

    assert standings.iloc[0]["team"] == "Team A"
    assert standings.iloc[0]["points"] == 6
    assert standings.iloc[1]["team"] == "Team C"
    assert standings.iloc[1]["gd"] == -1
    assert standings.iloc[2]["team"] == "Team B"
    assert standings.iloc[2]["gd"] == -2


def test_tiebreakers_gf():
    """Test tiebreaker switching to goals for (GF)."""
    data = [
        {"home_team": "A", "away_team": "B", "fthg": 4, "ftag": 4},  # Both 1pt, GD 0, GF 4
        {"home_team": "C", "away_team": "D", "fthg": 0, "ftag": 0},  # Both 1pt, GD 0, GF 0
    ]
    df = pd.DataFrame(data)
    standings = compute_standings(df)

    assert standings.iloc[0]["team"] == "A"
    assert standings.iloc[1]["team"] == "B"
    assert standings.iloc[2]["team"] == "C"
    assert standings.iloc[3]["team"] == "D"


def test_unplayed_matches():
    """Test that matches with NaN scores are ignored."""
    data = [
        {"home_team": "A", "away_team": "B", "fthg": 1, "ftag": 0},
        {"home_team": "B", "away_team": "A", "fthg": None, "ftag": None},
    ]
    df = pd.DataFrame(data)
    standings = compute_standings(df)

    assert standings[standings["team"] == "A"]["played"].values[0] == 1
    assert standings[standings["team"] == "B"]["played"].values[0] == 1


def test_stable_sort_alphabetical():
    """Test that tied teams use alphabetical order as fallback."""
    data = [
        {"home_team": "Zebra", "away_team": "Apple", "fthg": 1, "ftag": 1},
    ]
    df = pd.DataFrame(data)
    standings = compute_standings(df)

    # Pts 1, GD 0, GF 1 for both. Apple should be first alphabetically.
    assert standings.iloc[0]["team"] == "Apple"
    assert standings.iloc[1]["team"] == "Zebra"


def test_get_team_form_basic():
    data = [
        {"home_team": "A", "away_team": "B", "fthg": 2, "ftag": 0, "date": "01/01/2023"},
        {"home_team": "A", "away_team": "C", "fthg": 1, "ftag": 1, "date": "05/01/2023"},
        {"home_team": "B", "away_team": "A", "fthg": 2, "ftag": 1, "date": "10/01/2023"},
    ]
    df = pd.DataFrame(data)
    form = get_team_form(df, team="A")
    assert form == ["W", "D", "L"]


def test_get_team_form_last_n():
    data = [
        {"home_team": "A", "away_team": "B", "fthg": 1, "ftag": 0, "date": "01/01/2023"},
        {"home_team": "A", "away_team": "C", "fthg": 2, "ftag": 0, "date": "05/01/2023"},
        {"home_team": "A", "away_team": "D", "fthg": 0, "ftag": 0, "date": "10/01/2023"},
    ]
    df = pd.DataFrame(data)
    form = get_team_form(df, team="A", last_n=2)
    assert form == ["W", "D"]


def test_get_team_form_unplayed_matches():
    data = [
        {"home_team": "A", "away_team": "B", "fthg": 1, "ftag": 0, "date": "01/01/2023"},
        {"home_team": "A", "away_team": "C", "fthg": None, "ftag": None, "date": "05/01/2023"},
    ]
    df = pd.DataFrame(data)
    form = get_team_form(df, team="A")
    assert form == ["W"]
