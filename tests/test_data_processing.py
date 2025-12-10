"""Basic tests for data_processing utilities.

These will be expanded as the pipeline is implemented.
"""

from pathlib import Path

from src import data_processing


def test_load_raw_data_raises_on_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.csv"
    try:
        data_processing.load_raw_data(missing)
    except FileNotFoundError:
        assert True
    else:
        assert False, "Expected FileNotFoundError for missing file"
