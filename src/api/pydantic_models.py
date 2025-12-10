"""Pydantic models for request/response schemas.

We will flesh these out when the feature set is finalized.
"""

from pydantic import BaseModel


class PredictionRequest(BaseModel):
    """Placeholder request schema."""
    example_feature: float


class PredictionResponse(BaseModel):
    """Placeholder response schema."""
    risk_probability: float
