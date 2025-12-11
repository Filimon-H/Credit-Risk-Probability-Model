"""FastAPI application for serving the credit risk model.

This API provides endpoints for:
- Health check
- Credit risk prediction
"""

import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).resolve().parents[1]
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.pydantic_models import CustomerFeatures, HealthResponse, RiskPrediction
from predict import is_model_loaded, load_model, predict_single

# Initialize FastAPI app
app = FastAPI(
    title="Credit Risk Probability API",
    description="API for predicting customer credit risk based on transaction behavior.",
    version="1.0.0",
)

# Add CORS middleware (for local development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Startup event: preload model
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    """Load the model on startup."""
    try:
        load_model()
        print("Model loaded successfully on startup.")
    except FileNotFoundError as e:
        print(f"Warning: {e}")
        print("The /predict endpoint will not work until the model is available.")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["Root"])
async def root() -> dict:
    """Root endpoint with API info."""
    return {
        "message": "Credit Risk Probability API",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """Health check endpoint.

    Returns the service status and whether the model is loaded.
    """
    return HealthResponse(
        status="ok",
        model_loaded=is_model_loaded(),
    )


@app.post("/predict", response_model=RiskPrediction, tags=["Prediction"])
async def predict(features: CustomerFeatures) -> RiskPrediction:
    """Predict credit risk for a customer.

    Takes customer-level features and returns a risk prediction.

    Parameters
    ----------
    features : CustomerFeatures
        The 26 engineered features for a customer.

    Returns
    -------
    RiskPrediction
        Prediction result with is_high_risk label and probability.
    """
    try:
        # Convert Pydantic model to dict
        features_dict = features.model_dump()

        # Make prediction
        result = predict_single(features_dict)

        return RiskPrediction(**result)

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Model not available: {str(e)}"
        )
    except KeyError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Missing feature: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
