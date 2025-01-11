# Mapping of the different competitions to their respective slugs
MAPPING_COMPETITIONS: dict[str, str] = {"ligue1": "F1", "ligue2": "F2", "premierleague": "E0", "championship": "E1",
                        "bundesliga": "D1", "bundesliga2": "D2","seriea": "I1", "serieb": "I2", "laliga": "SP1", "laliga2": "SP2"}

def check_competition_exists(competition: str) -> bool:
    """Check if the competition exists in the MAPPING_COMPETITIONS dictionary.

    Args:
        competition (str): The name of the competition to check.

    Returns:
        bool: True if the competition exists, False otherwise.

    """
    return competition in MAPPING_COMPETITIONS


def process_string(input_string):
    lower_string = input_string.lower()
    no_space_string = lower_string.replace(" ", "")
    return no_space_string