"""Feature engineering utilities for credit risk modeling.

Task 3: Transform raw transaction data into customer-level features.
Uses sklearn.pipeline.Pipeline for reproducible transformations.
"""

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ---------------------------------------------------------------------------
# Step 1: Time feature extraction (transaction level)
# ---------------------------------------------------------------------------

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Parse TransactionStartTime and extract time-based features.

    Parameters
    ----------
    df : pd.DataFrame
        Raw transactions dataframe with 'TransactionStartTime' column.

    Returns
    -------
    pd.DataFrame
        Copy of df with added columns:
        - txn_datetime: parsed datetime
        - txn_hour: hour of day (0-23)
        - txn_day: day of month (1-31)
        - txn_month: month (1-12)
        - txn_year: year
        - txn_dayofweek: day of week (0=Monday, 6=Sunday)
        - is_weekend: 1 if Saturday/Sunday, else 0
    """
    df = df.copy()
    df["txn_datetime"] = pd.to_datetime(df["TransactionStartTime"], utc=True)
    df["txn_hour"] = df["txn_datetime"].dt.hour
    df["txn_day"] = df["txn_datetime"].dt.day
    df["txn_month"] = df["txn_datetime"].dt.month
    df["txn_year"] = df["txn_datetime"].dt.year
    df["txn_dayofweek"] = df["txn_datetime"].dt.dayofweek
    df["is_weekend"] = (df["txn_dayofweek"] >= 5).astype(int)
    return df


# ---------------------------------------------------------------------------
# Step 2: Aggregate numeric features per customer
# ---------------------------------------------------------------------------

def aggregate_numeric_by_customer(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate numeric transaction features per CustomerId.

    Parameters
    ----------
    df : pd.DataFrame
        Transactions dataframe with time features already added.

    Returns
    -------
    pd.DataFrame
        Customer-level aggregations indexed by CustomerId.
    """
    agg_funcs = {
        "TransactionId": "count",  # txn_count
        "Amount": ["sum", "mean", "std", "min", "max"],
        "Value": ["sum", "mean"],
        "txn_hour": ["mean", "std"],
        "is_weekend": "mean",  # proportion of weekend transactions
    }

    df_agg = df.groupby("CustomerId").agg(agg_funcs)

    # Flatten column names
    df_agg.columns = [
        "_".join(col).strip() if isinstance(col, tuple) else col
        for col in df_agg.columns
    ]

    # Rename for clarity
    df_agg = df_agg.rename(columns={
        "TransactionId_count": "txn_count",
        "Amount_sum": "total_amount",
        "Amount_mean": "avg_amount",
        "Amount_std": "std_amount",
        "Amount_min": "min_amount",
        "Amount_max": "max_amount",
        "Value_sum": "total_value",
        "Value_mean": "avg_value",
        "txn_hour_mean": "avg_txn_hour",
        "txn_hour_std": "std_txn_hour",
        "is_weekend_mean": "weekend_txn_ratio",
    })

    # Fill NaN in std columns (customers with 1 transaction)
    df_agg["std_amount"] = df_agg["std_amount"].fillna(0)
    df_agg["std_txn_hour"] = df_agg["std_txn_hour"].fillna(0)

    # Add derived features
    # Net amount (credits are negative, debits positive)
    df_agg["net_amount"] = df_agg["total_amount"]

    # Count of credits vs debits
    credits = df[df["Amount"] < 0].groupby("CustomerId").size()
    debits = df[df["Amount"] > 0].groupby("CustomerId").size()
    df_agg["n_credits"] = credits.reindex(df_agg.index, fill_value=0)
    df_agg["n_debits"] = debits.reindex(df_agg.index, fill_value=0)

    return df_agg.reset_index()


# ---------------------------------------------------------------------------
# Step 3: Aggregate categorical features per customer
# ---------------------------------------------------------------------------

def aggregate_categorical_by_customer(
    df: pd.DataFrame,
    top_k: int = 3
) -> pd.DataFrame:
    """Aggregate categorical features per CustomerId as proportions.

    For each categorical column, compute the proportion of transactions
    in the top K categories.

    Parameters
    ----------
    df : pd.DataFrame
        Transactions dataframe.
    top_k : int
        Number of top categories to track per column.

    Returns
    -------
    pd.DataFrame
        Customer-level categorical proportions indexed by CustomerId.
    """
    cat_cols = ["ProductCategory", "ChannelId", "ProviderId"]
    result_dfs = []

    for col in cat_cols:
        # Get top K categories overall
        top_cats = df[col].value_counts().head(top_k).index.tolist()

        # For each customer, compute proportion in each top category
        for cat in top_cats:
            col_name = f"{col}_{cat}_ratio".replace(" ", "_").lower()
            cat_flag = (df[col] == cat).astype(int)
            cat_ratio = df.assign(**{col_name: cat_flag}).groupby("CustomerId")[col_name].mean()
            result_dfs.append(cat_ratio.rename(col_name))

    df_cat = pd.concat(result_dfs, axis=1).reset_index()
    return df_cat


# ---------------------------------------------------------------------------
# Step 4: Build complete customer feature table
# ---------------------------------------------------------------------------

def build_customer_feature_table(df_raw: pd.DataFrame, top_k: int = 3) -> pd.DataFrame:
    """Build the full customer-level feature table from raw transactions.

    Parameters
    ----------
    df_raw : pd.DataFrame
        Raw transactions dataframe.
    top_k : int
        Number of top categories to track for categorical proportions.

    Returns
    -------
    pd.DataFrame
        Customer-level feature table with numeric and categorical aggregations.
    """
    # Step 1: Add time features
    df_txn = add_time_features(df_raw)

    # Step 2: Aggregate numeric features
    df_numeric = aggregate_numeric_by_customer(df_txn)

    # Step 3: Aggregate categorical features
    df_categorical = aggregate_categorical_by_customer(df_txn, top_k=top_k)

    # Step 4: Merge
    df_features = df_numeric.merge(df_categorical, on="CustomerId", how="left")

    return df_features


# ---------------------------------------------------------------------------
# Step 5: Build sklearn preprocessing pipeline
# ---------------------------------------------------------------------------

def get_feature_lists(df_features: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Identify numeric and categorical feature columns.

    Parameters
    ----------
    df_features : pd.DataFrame
        Customer-level feature table.

    Returns
    -------
    Tuple[List[str], List[str]]
        (numeric_features, categorical_features)
    """
    exclude_cols = ["CustomerId"]
    numeric_features = df_features.select_dtypes(include=[np.number]).columns.tolist()
    numeric_features = [c for c in numeric_features if c not in exclude_cols]

    # For now, all our aggregated features are numeric (ratios)
    # If we had actual categorical columns, we'd identify them here
    categorical_features = []

    return numeric_features, categorical_features


def build_preprocessing_pipeline(
    numeric_features: List[str],
    categorical_features: List[str]
) -> Pipeline:
    """Build a sklearn preprocessing pipeline.

    Parameters
    ----------
    numeric_features : List[str]
        List of numeric column names.
    categorical_features : List[str]
        List of categorical column names.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Preprocessing pipeline with imputation, scaling, and encoding.
    """
    # Numeric transformer: impute missing + standardize
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # Categorical transformer: impute missing + one-hot encode
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    # Combine with ColumnTransformer
    transformers = []
    if numeric_features:
        transformers.append(("num", numeric_transformer, numeric_features))
    if categorical_features:
        transformers.append(("cat", categorical_transformer, categorical_features))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )

    # Final pipeline
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
    ])

    return pipeline


def get_transformed_feature_names(
    pipeline: Pipeline,
    numeric_features: List[str],
    categorical_features: List[str]
) -> List[str]:
    """Get feature names after transformation.

    Parameters
    ----------
    pipeline : Pipeline
        Fitted preprocessing pipeline.
    numeric_features : List[str]
        Original numeric feature names.
    categorical_features : List[str]
        Original categorical feature names.

    Returns
    -------
    List[str]
        Transformed feature names.
    """
    feature_names = []

    # Numeric features keep their names
    feature_names.extend(numeric_features)

    # Categorical features get expanded by OneHotEncoder
    if categorical_features:
        preprocessor = pipeline.named_steps["preprocessor"]
        cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        cat_feature_names = cat_encoder.get_feature_names_out(categorical_features)
        feature_names.extend(cat_feature_names.tolist())

    return feature_names


# ---------------------------------------------------------------------------
# Step 6: WoE / IV (placeholder - requires target variable)
# ---------------------------------------------------------------------------

def compute_woe_iv(
    df: pd.DataFrame,
    feature_col: str,
    target_col: str,
    bins: int = 10
) -> pd.DataFrame:
    """Compute Weight of Evidence and Information Value for a feature.

    Note: This is a simplified implementation. For production, consider
    using libraries like `xverse` or `scorecardpy`.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with feature and target columns.
    feature_col : str
        Name of the feature column.
    target_col : str
        Name of the binary target column (0/1).
    bins : int
        Number of bins for continuous features.

    Returns
    -------
    pd.DataFrame
        DataFrame with WoE and IV per bin.
    """
    df = df.copy()

    # Bin continuous features
    if df[feature_col].dtype in [np.float64, np.int64]:
        df["bin"] = pd.qcut(df[feature_col], q=bins, duplicates="drop")
    else:
        df["bin"] = df[feature_col]

    # Compute counts
    grouped = df.groupby("bin")[target_col].agg(["sum", "count"])
    grouped.columns = ["bad", "total"]
    grouped["good"] = grouped["total"] - grouped["bad"]

    # Compute distributions
    total_bad = grouped["bad"].sum()
    total_good = grouped["good"].sum()

    # Avoid division by zero
    grouped["dist_bad"] = grouped["bad"] / max(total_bad, 1)
    grouped["dist_good"] = grouped["good"] / max(total_good, 1)

    # Replace zeros with small value to avoid log(0)
    grouped["dist_bad"] = grouped["dist_bad"].replace(0, 0.0001)
    grouped["dist_good"] = grouped["dist_good"].replace(0, 0.0001)

    # WoE = ln(dist_good / dist_bad)
    grouped["woe"] = np.log(grouped["dist_good"] / grouped["dist_bad"])

    # IV = (dist_good - dist_bad) * WoE
    grouped["iv"] = (grouped["dist_good"] - grouped["dist_bad"]) * grouped["woe"]

    # Total IV
    total_iv = grouped["iv"].sum()
    grouped["total_iv"] = total_iv

    return grouped.reset_index()
