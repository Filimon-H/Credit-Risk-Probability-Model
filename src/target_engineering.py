"""Target engineering for credit risk modeling.

Task 4: Create proxy target variable via RFM metrics and K-Means clustering.
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Step 1: Compute RFM metrics
# ---------------------------------------------------------------------------

def compute_rfm(
    df_txn: pd.DataFrame,
    snapshot_date: Optional[pd.Timestamp] = None
) -> pd.DataFrame:
    """Calculate Recency, Frequency, and Monetary metrics per customer.

    Parameters
    ----------
    df_txn : pd.DataFrame
        Transaction-level data with columns:
        - CustomerId
        - txn_datetime (datetime, from add_time_features)
        - Value (absolute transaction value)
    snapshot_date : pd.Timestamp, optional
        Reference date for recency calculation.
        If None, uses max(txn_datetime) + 1 day.

    Returns
    -------
    pd.DataFrame
        Customer-level RFM metrics with columns:
        - CustomerId
        - recency_days: days since last transaction
        - frequency: number of transactions
        - monetary: sum of Value
    """
    if snapshot_date is None:
        snapshot_date = df_txn["txn_datetime"].max() + pd.Timedelta(days=1)

    # Last transaction datetime per customer
    last_txn = df_txn.groupby("CustomerId")["txn_datetime"].max()

    # Recency: days since last transaction
    recency_days = (snapshot_date - last_txn).dt.days

    # Frequency: number of transactions per customer
    frequency = df_txn.groupby("CustomerId").size()

    # Monetary: sum of Value per customer
    monetary = df_txn.groupby("CustomerId")["Value"].sum()

    # Build RFM DataFrame
    df_rfm = pd.DataFrame({
        "CustomerId": last_txn.index,
        "recency_days": recency_days.values,
        "frequency": frequency.reindex(last_txn.index).values,
        "monetary": monetary.reindex(last_txn.index).values,
    }).reset_index(drop=True)

    return df_rfm


# ---------------------------------------------------------------------------
# Step 2: Cluster customers using K-Means
# ---------------------------------------------------------------------------

def cluster_rfm(
    df_rfm: pd.DataFrame,
    n_clusters: int = 3,
    random_state: int = 42
) -> Tuple[pd.DataFrame, StandardScaler, KMeans]:
    """Segment customers into clusters based on RFM profiles.

    Parameters
    ----------
    df_rfm : pd.DataFrame
        RFM metrics with columns: CustomerId, recency_days, frequency, monetary.
    n_clusters : int
        Number of clusters (default 3).
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    Tuple[pd.DataFrame, StandardScaler, KMeans]
        - df_rfm with added 'rfm_cluster' column
        - fitted StandardScaler
        - fitted KMeans model
    """
    df_rfm = df_rfm.copy()

    # Features to cluster on
    rfm_features = ["recency_days", "frequency", "monetary"]
    X = df_rfm[rfm_features].values

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    kmeans.fit(X_scaled)

    # Assign cluster labels
    df_rfm["rfm_cluster"] = kmeans.labels_

    return df_rfm, scaler, kmeans


# ---------------------------------------------------------------------------
# Step 3: Analyze clusters and assign high-risk label
# ---------------------------------------------------------------------------

def analyze_clusters(df_rfm: pd.DataFrame) -> pd.DataFrame:
    """Compute mean RFM metrics per cluster for analysis.

    Parameters
    ----------
    df_rfm : pd.DataFrame
        RFM data with rfm_cluster column.

    Returns
    -------
    pd.DataFrame
        Cluster summary with mean recency, frequency, monetary, and count.
    """
    cluster_summary = df_rfm.groupby("rfm_cluster").agg({
        "recency_days": "mean",
        "frequency": "mean",
        "monetary": "mean",
        "CustomerId": "count"
    }).rename(columns={"CustomerId": "customer_count"})

    cluster_summary = cluster_summary.reset_index()
    return cluster_summary


def assign_high_risk(df_rfm: pd.DataFrame) -> pd.DataFrame:
    """Assign is_high_risk label based on cluster analysis.

    High-risk cluster is identified as the one with:
    - High recency (haven't transacted recently)
    - Low frequency
    - Low monetary value

    Parameters
    ----------
    df_rfm : pd.DataFrame
        RFM data with rfm_cluster column.

    Returns
    -------
    pd.DataFrame
        df_rfm with added 'is_high_risk' column (1 = high-risk, 0 = others).
    """
    df_rfm = df_rfm.copy()

    # Compute cluster summary
    cluster_summary = analyze_clusters(df_rfm)

    # Rank clusters:
    # - High recency = bad (rank ascending, higher recency gets higher rank)
    # - Low frequency = bad (rank descending, lower frequency gets higher rank)
    # - Low monetary = bad (rank descending, lower monetary gets higher rank)
    cluster_summary["recency_rank"] = cluster_summary["recency_days"].rank(ascending=True)
    cluster_summary["frequency_rank"] = cluster_summary["frequency"].rank(ascending=False)
    cluster_summary["monetary_rank"] = cluster_summary["monetary"].rank(ascending=False)

    # Risk score = sum of ranks (higher = worse)
    cluster_summary["risk_score"] = (
        cluster_summary["recency_rank"] +
        cluster_summary["frequency_rank"] +
        cluster_summary["monetary_rank"]
    )

    # High-risk cluster is the one with highest risk score
    high_risk_cluster = cluster_summary.loc[
        cluster_summary["risk_score"].idxmax(), "rfm_cluster"
    ]

    # Assign binary label
    df_rfm["is_high_risk"] = (df_rfm["rfm_cluster"] == high_risk_cluster).astype(int)

    return df_rfm, high_risk_cluster, cluster_summary


# ---------------------------------------------------------------------------
# Step 4: Build modeling dataset
# ---------------------------------------------------------------------------

def build_modeling_dataset(
    df_features: pd.DataFrame,
    df_rfm_with_target: pd.DataFrame
) -> pd.DataFrame:
    """Merge RFM target into customer features for model training.

    Parameters
    ----------
    df_features : pd.DataFrame
        Customer-level features from Task 3.
    df_rfm_with_target : pd.DataFrame
        RFM data with is_high_risk column.

    Returns
    -------
    pd.DataFrame
        Merged dataset ready for modeling.
    """
    # Columns to merge from RFM
    rfm_cols = [
        "CustomerId",
        "recency_days",
        "frequency",
        "monetary",
        "rfm_cluster",
        "is_high_risk"
    ]

    df_model = df_features.merge(
        df_rfm_with_target[rfm_cols],
        on="CustomerId",
        how="inner"
    )

    return df_model
