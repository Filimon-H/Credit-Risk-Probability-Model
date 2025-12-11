"""Model training and experiment tracking for credit risk modeling.

Task 5: Train models, tune hyperparameters, and track experiments with MLflow.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def get_project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parents[1]


def load_modeling_data(
    path: Path = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load the modeling dataset with features and target.

    Parameters
    ----------
    path : Path, optional
        Path to the CSV file. Defaults to data/processed/customer_features_with_target.csv.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series]
        (df_full, X, y) where X is features and y is target.
    """
    if path is None:
        path = get_project_root() / "data" / "processed" / "customer_features_with_target.csv"

    df = pd.read_csv(path)

    # Columns to exclude from features
    exclude_cols = ["CustomerId", "rfm_cluster", "is_high_risk"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    X = df[feature_cols]
    y = df["is_high_risk"]

    return df, X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into train and test sets.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target vector.
    test_size : float
        Proportion of data for testing.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        (X_train, X_test, y_train, y_test)
    """
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )


# ---------------------------------------------------------------------------
# Model Definitions
# ---------------------------------------------------------------------------

def get_model_configs() -> Dict[str, Dict[str, Any]]:
    """Return model configurations with pipelines and parameter grids.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Dictionary with model name as key and config dict containing:
        - pipeline: sklearn Pipeline
        - param_grid: dict for GridSearchCV
    """
    configs = {
        "LogisticRegression": {
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
            ]),
            "param_grid": {
                "classifier__C": [0.01, 0.1, 1.0, 10.0],
                "classifier__penalty": ["l2"],
                "classifier__class_weight": [None, "balanced"],
            },
        },
        "DecisionTree": {
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", DecisionTreeClassifier(random_state=42)),
            ]),
            "param_grid": {
                "classifier__max_depth": [3, 5, 10, None],
                "classifier__min_samples_split": [2, 5, 10],
                "classifier__class_weight": [None, "balanced"],
            },
        },
        "RandomForest": {
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", RandomForestClassifier(random_state=42, n_jobs=-1)),
            ]),
            "param_grid": {
                "classifier__n_estimators": [50, 100, 200],
                "classifier__max_depth": [5, 10, None],
                "classifier__min_samples_split": [2, 5],
                "classifier__class_weight": [None, "balanced"],
            },
        },
        "GradientBoosting": {
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", GradientBoostingClassifier(random_state=42)),
            ]),
            "param_grid": {
                "classifier__n_estimators": [50, 100, 200],
                "classifier__max_depth": [3, 5, 7],
                "classifier__learning_rate": [0.01, 0.1, 0.2],
            },
        },
    }
    return configs


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, float]:
    """Evaluate a trained model on test data.

    Parameters
    ----------
    model : Any
        Trained sklearn model or pipeline.
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        Test target.

    Returns
    -------
    Dict[str, float]
        Dictionary with accuracy, precision, recall, f1, roc_auc.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }
    return metrics


def plot_confusion_matrix(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str
) -> plt.Figure:
    """Plot confusion matrix for a model.

    Parameters
    ----------
    model : Any
        Trained model.
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        Test target.
    model_name : str
        Name for the plot title.

    Returns
    -------
    plt.Figure
        Matplotlib figure.
    """
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Low Risk", "High Risk"])
    disp.plot(ax=ax, cmap="Blues")
    ax.set_title(f"Confusion Matrix: {model_name}")
    plt.tight_layout()
    return fig


def plot_roc_curve(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str
) -> plt.Figure:
    """Plot ROC curve for a model.

    Parameters
    ----------
    model : Any
        Trained model.
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        Test target.
    model_name : str
        Name for the plot title.

    Returns
    -------
    plt.Figure
        Matplotlib figure.
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve: {model_name}")
    ax.legend(loc="lower right")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Training with MLflow
# ---------------------------------------------------------------------------

def train_and_log_model(
    model_name: str,
    pipeline: Pipeline,
    param_grid: Dict[str, List[Any]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cv: int = 5,
    scoring: str = "roc_auc"
) -> Tuple[Any, Dict[str, float], str]:
    """Train a model with GridSearchCV and log to MLflow.

    Parameters
    ----------
    model_name : str
        Name of the model.
    pipeline : Pipeline
        sklearn Pipeline with preprocessing and classifier.
    param_grid : Dict[str, List[Any]]
        Hyperparameter grid for GridSearchCV.
    X_train, y_train : pd.DataFrame, pd.Series
        Training data.
    X_test, y_test : pd.DataFrame, pd.Series
        Test data.
    cv : int
        Number of cross-validation folds.
    scoring : str
        Scoring metric for GridSearchCV.

    Returns
    -------
    Tuple[Any, Dict[str, float], str]
        (best_model, metrics_dict, run_id)
    """
    with mlflow.start_run(run_name=model_name) as run:
        run_id = run.info.run_id

        # Log model name
        mlflow.log_param("model_name", model_name)

        # GridSearchCV
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=0
        )
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_

        # Log best parameters
        for param, value in best_params.items():
            mlflow.log_param(param, value)

        mlflow.log_param("cv_folds", cv)
        mlflow.log_param("scoring", scoring)

        # Evaluate on test set
        metrics = evaluate_model(best_model, X_test, y_test)

        # Log metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        # Log confusion matrix plot
        cm_fig = plot_confusion_matrix(best_model, X_test, y_test, model_name)
        mlflow.log_figure(cm_fig, f"confusion_matrix_{model_name}.png")
        plt.close(cm_fig)

        # Log ROC curve plot
        roc_fig = plot_roc_curve(best_model, X_test, y_test, model_name)
        mlflow.log_figure(roc_fig, f"roc_curve_{model_name}.png")
        plt.close(roc_fig)

        # Log model
        mlflow.sklearn.log_model(best_model, artifact_path="model")

        print(f"[{model_name}] Run ID: {run_id}")
        print(f"  Best params: {best_params}")
        print(f"  Metrics: {metrics}")

    return best_model, metrics, run_id


def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    experiment_name: str = "Credit_Risk_Models"
) -> Dict[str, Dict[str, Any]]:
    """Train all configured models and log to MLflow.

    Parameters
    ----------
    X_train, y_train : pd.DataFrame, pd.Series
        Training data.
    X_test, y_test : pd.DataFrame, pd.Series
        Test data.
    experiment_name : str
        MLflow experiment name.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Dictionary with model name as key and dict containing:
        - model: trained model
        - metrics: evaluation metrics
        - run_id: MLflow run ID
    """
    # Set MLflow experiment
    mlflow.set_experiment(experiment_name)

    model_configs = get_model_configs()
    results = {}

    for model_name, config in model_configs.items():
        print(f"\n{'='*50}")
        print(f"Training: {model_name}")
        print(f"{'='*50}")

        model, metrics, run_id = train_and_log_model(
            model_name=model_name,
            pipeline=config["pipeline"],
            param_grid=config["param_grid"],
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )

        results[model_name] = {
            "model": model,
            "metrics": metrics,
            "run_id": run_id,
        }

    return results


def get_best_model(results: Dict[str, Dict[str, Any]], metric: str = "roc_auc") -> str:
    """Identify the best model based on a metric.

    Parameters
    ----------
    results : Dict[str, Dict[str, Any]]
        Results from train_all_models.
    metric : str
        Metric to compare (default: roc_auc).

    Returns
    -------
    str
        Name of the best model.
    """
    best_name = None
    best_value = -1

    for name, data in results.items():
        value = data["metrics"].get(metric, 0)
        if value > best_value:
            best_value = value
            best_name = name

    return best_name


def register_best_model(
    results: Dict[str, Dict[str, Any]],
    model_name: str,
    registry_name: str = "CreditRiskModel"
) -> str:
    """Register the best model in MLflow Model Registry.

    Parameters
    ----------
    results : Dict[str, Dict[str, Any]]
        Results from train_all_models.
    model_name : str
        Name of the model to register.
    registry_name : str
        Name for the registered model.

    Returns
    -------
    str
        Model version.
    """
    run_id = results[model_name]["run_id"]
    model_uri = f"runs:/{run_id}/model"

    result = mlflow.register_model(model_uri, registry_name)
    print(f"Registered model '{registry_name}' version {result.version}")

    return result.version
