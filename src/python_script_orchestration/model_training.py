import warnings
import os
import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV

# Suppress warnings
warnings.filterwarnings("ignore")


def train():
    """
    Train an ExtraTreesClassifier using GridSearchCV and log the parameters, metrics, and model to MLflow.
    """
    # Set up MLflow tracking environment
    #os.environ['MLFLOW_TRACKING_URI'] = "http://127.0.0.1:5000"  # Replace with your MLflow tracking URI
    mlflow.set_tracking_uri("http://localhost:5000")
    #os.environ['MLFLOW_TRACKING_USERNAME'] = "your-username"  # Replace with your username
    #os.environ['MLFLOW_TRACKING_PASSWORD'] = "your-password"  # Replace with your password

    # Define file paths
    X_train_path = "../../data/X_train.csv"
    y_train_path = "../../data/y_train.csv"
    model_path = "../../model/best_model.joblib"

    # Load training data
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path)

    # Initialize ExtraTreesClassifier
    model = ExtraTreesClassifier(random_state=42, class_weight="balanced")

    # Hyperparameter grid for GridSearchCV
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }

    # Perform Grid Search with cross-validation
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="accuracy",
        cv=3,
        verbose=1,
        n_jobs=-1
    )

    mlflow.set_experiment("pythonScriptOrchestrationExperiment")
    with mlflow.start_run():
        # Fit the model using GridSearchCV
        print("\nPerforming Grid Search...")
        grid_search.fit(X_train, y_train)

        # Get the best model and parameters
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_

        print("\nBest Hyperparameters:")
        print(best_params)

        # Log hyperparameters to MLflow
        for param_name, param_value in best_params.items():
            mlflow.log_param(param_name, param_value)

        # Train the best model on the entire training set
        best_model.fit(X_train, y_train)

        # Ensure the model directory exists
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        # Save the best model
        joblib.dump(best_model, model_path)

        print(f"\nBest model saved to: {model_path}")

        # Log the model to MLflow
        mlflow.sklearn.log_model(best_model, "model")

        # Log additional information
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_param("random_state", 42)

        # Log metrics on training data for completeness (Optional)
        train_accuracy = best_model.score(X_train, y_train)
        mlflow.log_metric("train_accuracy", train_accuracy)

        print(f"\nTraining accuracy logged to MLflow: {train_accuracy:.4f}")


if __name__ == "__main__":
    train()
