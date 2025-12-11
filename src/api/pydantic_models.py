"""Pydantic models for request/response schemas.

These models define the API contract for credit risk predictions.
"""

from typing import Optional

from pydantic import BaseModel, Field


class CustomerFeatures(BaseModel):
    """Request schema: customer-level features for prediction.

    These are the 26 engineered features from Task 3 + RFM from Task 4.
    """

    # Transaction aggregations
    txn_count: int = Field(..., description="Number of transactions")
    total_amount: float = Field(..., description="Sum of transaction amounts")
    avg_amount: float = Field(..., description="Average transaction amount")
    std_amount: float = Field(..., description="Std dev of transaction amounts")
    min_amount: float = Field(..., description="Minimum transaction amount")
    max_amount: float = Field(..., description="Maximum transaction amount")
    total_value: float = Field(..., description="Sum of transaction values")
    avg_value: float = Field(..., description="Average transaction value")

    # Time-based features
    avg_txn_hour: float = Field(..., description="Average transaction hour (0-23)")
    std_txn_hour: float = Field(..., description="Std dev of transaction hour")
    weekend_txn_ratio: float = Field(..., description="Ratio of weekend transactions")

    # Credit/debit features
    net_amount: float = Field(..., description="Net transaction amount")
    n_credits: int = Field(..., description="Number of credit transactions")
    n_debits: int = Field(..., description="Number of debit transactions")

    # Categorical proportions - ProductCategory
    productcategory_financial_services_ratio: float = Field(
        ..., description="Ratio of financial_services transactions"
    )
    productcategory_airtime_ratio: float = Field(
        ..., description="Ratio of airtime transactions"
    )
    productcategory_utility_bill_ratio: float = Field(
        ..., description="Ratio of utility_bill transactions"
    )

    # Categorical proportions - ChannelId
    channelid_channelid_3_ratio: float = Field(
        ..., description="Ratio of ChannelId_3 transactions"
    )
    channelid_channelid_2_ratio: float = Field(
        ..., description="Ratio of ChannelId_2 transactions"
    )
    channelid_channelid_5_ratio: float = Field(
        ..., description="Ratio of ChannelId_5 transactions"
    )

    # Categorical proportions - ProviderId
    providerid_providerid_4_ratio: float = Field(
        ..., description="Ratio of ProviderId_4 transactions"
    )
    providerid_providerid_6_ratio: float = Field(
        ..., description="Ratio of ProviderId_6 transactions"
    )
    providerid_providerid_5_ratio: float = Field(
        ..., description="Ratio of ProviderId_5 transactions"
    )

    # RFM features
    recency_days: int = Field(..., description="Days since last transaction")
    frequency: int = Field(..., description="Number of transactions (RFM)")
    monetary: float = Field(..., description="Total monetary value (RFM)")

    class Config:
        json_schema_extra = {
            "example": {
                "txn_count": 10,
                "total_amount": 5000.0,
                "avg_amount": 500.0,
                "std_amount": 200.0,
                "min_amount": -100.0,
                "max_amount": 1000.0,
                "total_value": 5500.0,
                "avg_value": 550.0,
                "avg_txn_hour": 14.5,
                "std_txn_hour": 3.2,
                "weekend_txn_ratio": 0.2,
                "net_amount": 4500.0,
                "n_credits": 3,
                "n_debits": 7,
                "productcategory_financial_services_ratio": 0.5,
                "productcategory_airtime_ratio": 0.3,
                "productcategory_utility_bill_ratio": 0.2,
                "channelid_channelid_3_ratio": 0.6,
                "channelid_channelid_2_ratio": 0.3,
                "channelid_channelid_5_ratio": 0.1,
                "providerid_providerid_4_ratio": 0.4,
                "providerid_providerid_6_ratio": 0.3,
                "providerid_providerid_5_ratio": 0.3,
                "recency_days": 15,
                "frequency": 10,
                "monetary": 5500.0,
            }
        }


class RiskPrediction(BaseModel):
    """Response schema: credit risk prediction result."""

    is_high_risk: int = Field(..., description="Predicted label (0=low risk, 1=high risk)")
    probability_high_risk: float = Field(
        ..., description="Probability of being high risk (0.0 to 1.0)"
    )
    model_name: Optional[str] = Field(None, description="Name of the model used")
    model_version: Optional[str] = Field(None, description="Version of the model")

    class Config:
        json_schema_extra = {
            "example": {
                "is_high_risk": 0,
                "probability_high_risk": 0.23,
                "model_name": "RandomForest",
                "model_version": "1.0.0",
            }
        }


class HealthResponse(BaseModel):
    """Response schema for health check endpoint."""

    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
