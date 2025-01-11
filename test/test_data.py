from footix.data_io.data_scrapper import _process_season


def test_process_season():
    assert _process_season("2020/2021") == "2021"
    assert _process_season("2020-2021") == "2021"
    assert _process_season("2024/2025") == "2425"
    assert _process_season("2024-2025") == "2425"
    assert _process_season("2021") == "2021"  # Already in correct format

    try:
        _process_season("2020/2022")
    except ValueError as e:
        assert str(e) == "Years must be consecutive"

    try:
        _process_season("2020-2022")
    except ValueError as e:
        assert str(e) == "Years must be consecutive"
