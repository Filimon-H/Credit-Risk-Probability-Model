"""Data loading and feature engineering utilities.

Task 2: EDA helpers.
Task 3: Feature engineering (to be added).
"""

from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def get_project_root() -> Path:
    """Return the project root directory (parent of src/)."""
    return Path(__file__).resolve().parents[1]


def get_raw_data_path() -> Path:
    """Return the default path to the raw transactions CSV."""
    return get_project_root() / "data" / "raw" / "data.csv"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_raw_data(path: Union[str, Path]) -> pd.DataFrame:
    """Load raw transactions data from a CSV file.

    Parameters
    ----------
    path : str or Path
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
    """
    path = Path(path)
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# EDA helper
# ---------------------------------------------------------------------------

def run_eda(df: pd.DataFrame, top_n_categories: int = 10) -> Dict[str, Any]:
    """Run exploratory data analysis and return all results in a dictionary.

    Parameters
    ----------
    df : pd.DataFrame
        The raw transactions dataframe.
    top_n_categories : int
        Number of top categories to include in categorical summaries.

    Returns
    -------
    dict with keys:
        overview : dict
            n_rows, n_cols, columns, dtypes.
        summary_numeric : pd.DataFrame
            Descriptive statistics for numeric columns.
        summary_categorical : dict[str, pd.Series]
            Value counts (top N) for each categorical column.
        missing_table : pd.DataFrame
            Columns: column, n_missing, pct_missing.
        corr_numeric : pd.DataFrame
            Correlation matrix for numeric columns.
        outlier_info : pd.DataFrame
            For each numeric column: Q1, Q3, IQR, lower_bound, upper_bound,
            n_outliers, pct_outliers.
    """

    # --- Identify column types ---
    numeric_cols: List[str] = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols: List[str] = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    # --- Overview ---
    overview: Dict[str, Any] = {
        "n_rows": df.shape[0],
        "n_cols": df.shape[1],
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
    }

    # --- Summary statistics (numeric) ---
    summary_numeric: pd.DataFrame = df[numeric_cols].describe().T if numeric_cols else pd.DataFrame()

    # --- Summary statistics (categorical) ---
    summary_categorical: Dict[str, pd.Series] = {}
    for col in categorical_cols:
        summary_categorical[col] = df[col].value_counts().head(top_n_categories)

    # --- Missing values ---
    n_missing = df.isnull().sum()
    pct_missing = (n_missing / len(df)) * 100
    missing_table = pd.DataFrame({
        "column": df.columns,
        "n_missing": n_missing.values,
        "pct_missing": pct_missing.values,
    }).sort_values("pct_missing", ascending=False).reset_index(drop=True)

    # --- Correlation matrix (numeric) ---
    corr_numeric: pd.DataFrame = df[numeric_cols].corr() if numeric_cols else pd.DataFrame()

    # --- Outlier detection (IQR method) ---
    outlier_records = []
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        n_outliers = len(outliers)
        pct_outliers = (n_outliers / len(df)) * 100
        outlier_records.append({
            "column": col,
            "Q1": q1,
            "Q3": q3,
            "IQR": iqr,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "n_outliers": n_outliers,
            "pct_outliers": pct_outliers,
        })
    outlier_info = pd.DataFrame(outlier_records)

    return {
        "overview": overview,
        "summary_numeric": summary_numeric,
        "summary_categorical": summary_categorical,
        "missing_table": missing_table,
        "corr_numeric": corr_numeric,
        "outlier_info": outlier_info,
    }
