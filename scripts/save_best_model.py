"""Save the best model from Task 5 to models/best_model.pkl.

Run this script after training to export the best model for API serving.
"""

import sys
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import joblib
import mlflow

from train import (
    load_modeling_data,
    split_data,
    train_all_models,
    get_best_model,
)


def main():
    print("Loading data...")
    df, X, y = load_modeling_data()

    print("Splitting data...")
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

    # Set MLflow tracking URI
    mlflow.set_tracking_uri("file:" + str(PROJECT_ROOT / "mlruns"))

    print("Training all models (this may take a few minutes)...")
    results = train_all_models(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        experiment_name="Credit_Risk_Models"
    )

    # Identify best model
    best_model_name = get_best_model(results, metric="roc_auc")
    best_model = results[best_model_name]["model"]
    best_metrics = results[best_model_name]["metrics"]

    print(f"\nBest model: {best_model_name}")
    print(f"Metrics: {best_metrics}")

    # Save to models/best_model.pkl
    output_path = PROJECT_ROOT / "models" / "best_model.pkl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_model, output_path)
    print(f"\nSaved best model to: {output_path}")

    # Also save model metadata
    metadata = {
        "model_name": best_model_name,
        "metrics": best_metrics,
    }
    metadata_path = PROJECT_ROOT / "models" / "model_metadata.json"
    import json
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved model metadata to: {metadata_path}")


if __name__ == "__main__":
    main()
