"""FastAPI application for serving the credit risk model.

Implementation will follow in Task 6.
"""

from fastapi import FastAPI

app = FastAPI(title="Credit Risk Probability API")


@app.get("/health")
async def health_check() -> dict:
    """Simple health check endpoint."""
    return {"status": "ok"}
