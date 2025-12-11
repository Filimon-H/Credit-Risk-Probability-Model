"""Unit tests for data processing and feature engineering utilities.

Task 5: At least two unit tests for helper functions.
"""

from pathlib import Path

import pandas as pd
import pytest

from src import data_processing
from src import feature_engineering
from src import target_engineering


# ---------------------------------------------------------------------------
# Tests for data_processing.py
# ---------------------------------------------------------------------------

def test_load_raw_data_raises_on_missing_file(tmp_path: Path) -> None:
    """Test that load_raw_data raises FileNotFoundError for missing file."""
    missing = tmp_path / "missing.csv"
    with pytest.raises(FileNotFoundError):
        data_processing.load_raw_data(missing)


def test_run_eda_returns_expected_keys() -> None:
    """Test that run_eda returns a dict with all expected keys."""
    # Create a small synthetic DataFrame
    df = pd.DataFrame({
        "CustomerId": ["C1", "C2", "C3"],
        "Amount": [100.0, -50.0, 200.0],
        "Value": [100, 50, 200],
        "Category": ["A", "B", "A"],
    })

    result = data_processing.run_eda(df)

    expected_keys = [
        "overview",
        "summary_numeric",
        "summary_categorical",
        "missing_table",
        "corr_numeric",
        "outlier_info",
    ]

    for key in expected_keys:
        assert key in result, f"Missing key: {key}"

    # Check overview structure
    assert "n_rows" in result["overview"]
    assert "n_cols" in result["overview"]
    assert result["overview"]["n_rows"] == 3
    assert result["overview"]["n_cols"] == 4


# ---------------------------------------------------------------------------
# Tests for feature_engineering.py
# ---------------------------------------------------------------------------

def test_add_time_features_adds_expected_columns() -> None:
    """Test that add_time_features adds the expected time columns."""
    df = pd.DataFrame({
        "TransactionStartTime": [
            "2023-01-15T10:30:00Z",
            "2023-02-20T14:45:00Z",
        ],
        "Amount": [100, 200],
    })

    result = feature_engineering.add_time_features(df)

    expected_cols = [
        "txn_datetime",
        "txn_hour",
        "txn_day",
        "txn_month",
        "txn_year",
        "txn_dayofweek",
        "is_weekend",
    ]

    for col in expected_cols:
        assert col in result.columns, f"Missing column: {col}"

    # Check values
    assert result["txn_hour"].iloc[0] == 10
    assert result["txn_month"].iloc[1] == 2


def test_build_customer_feature_table_returns_expected_columns() -> None:
    """Test that build_customer_feature_table returns expected columns."""
    # Create synthetic transaction data
    df = pd.DataFrame({
        "TransactionId": ["T1", "T2", "T3", "T4"],
        "CustomerId": ["C1", "C1", "C2", "C2"],
        "Amount": [100.0, -50.0, 200.0, 300.0],
        "Value": [100, 50, 200, 300],
        "TransactionStartTime": [
            "2023-01-15T10:30:00Z",
            "2023-01-16T11:00:00Z",
            "2023-01-17T12:00:00Z",
            "2023-01-18T13:00:00Z",
        ],
        "ProductCategory": ["airtime", "financial_services", "airtime", "airtime"],
        "ChannelId": ["ChannelId_3", "ChannelId_2", "ChannelId_3", "ChannelId_3"],
        "ProviderId": ["ProviderId_4", "ProviderId_6", "ProviderId_4", "ProviderId_4"],
    })

    result = feature_engineering.build_customer_feature_table(df, top_k=2)

    # Should have 2 customers
    assert len(result) == 2

    # Should have CustomerId column
    assert "CustomerId" in result.columns

    # Should have numeric aggregation columns
    assert "txn_count" in result.columns
    assert "total_amount" in result.columns
    assert "avg_amount" in result.columns


# ---------------------------------------------------------------------------
# Tests for target_engineering.py
# ---------------------------------------------------------------------------

def test_compute_rfm_returns_expected_columns() -> None:
    """Test that compute_rfm returns expected RFM columns."""
    df = pd.DataFrame({
        "CustomerId": ["C1", "C1", "C2", "C2", "C2"],
        "txn_datetime": pd.to_datetime([
            "2023-01-10",
            "2023-01-15",
            "2023-01-05",
            "2023-01-08",
            "2023-01-12",
        ], utc=True),
        "Value": [100, 200, 50, 75, 125],
    })

    snapshot = pd.Timestamp("2023-01-20", tz="UTC")
    result = target_engineering.compute_rfm(df, snapshot_date=snapshot)

    assert "CustomerId" in result.columns
    assert "recency_days" in result.columns
    assert "frequency" in result.columns
    assert "monetary" in result.columns

    # Check values for C1: last txn on 2023-01-15, recency = 5 days
    c1_row = result[result["CustomerId"] == "C1"].iloc[0]
    assert c1_row["recency_days"] == 5
    assert c1_row["frequency"] == 2
    assert c1_row["monetary"] == 300  # 100 + 200


def test_assign_high_risk_creates_binary_label() -> None:
    """Test that assign_high_risk creates a binary is_high_risk column."""
    df_rfm = pd.DataFrame({
        "CustomerId": ["C1", "C2", "C3", "C4", "C5", "C6"],
        "recency_days": [5, 10, 50, 60, 3, 7],
        "frequency": [20, 15, 2, 1, 25, 18],
        "monetary": [5000, 3000, 100, 50, 8000, 4000],
        "rfm_cluster": [0, 0, 1, 1, 0, 0],
    })

    result, high_risk_cluster, _ = target_engineering.assign_high_risk(df_rfm)

    assert "is_high_risk" in result.columns
    assert set(result["is_high_risk"].unique()).issubset({0, 1})
    assert result["is_high_risk"].sum() > 0  # At least one high-risk
