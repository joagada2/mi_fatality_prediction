import warnings
import os
import joblib
import mlflow
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
from mlflow import sklearn as mlflow_sklearn

# Suppress warnings
warnings.filterwarnings("ignore")

# Define BASE_DIR relative to the script's location
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "../../data")
MODEL_DIR = os.path.join(BASE_DIR, "../../model")
X_TEST_PATH = os.path.join(DATA_DIR, "../../data/X_test.csv")
Y_TEST_PATH = os.path.join(DATA_DIR, "../../data/y_test.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "../../model/best_model.joblib")


def calculate_sensitivity_specificity(conf_matrix):
    """
    Calculate sensitivity and specificity from a confusion matrix.
    """
    TN, FP, FN, TP = conf_matrix.ravel()
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0  # True Positive Rate
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0  # True Negative Rate
    return sensitivity, specificity


def evaluate():
    """
    Evaluate the model and log metrics to MLflow.
    """
    # Set up MLflow tracking environment
    mlflow.set_tracking_uri("http://localhost:5000")  # Replace with your MLflow tracking URI

    mlflow.set_experiment("fatalityPredictionModel")
    # Start an MLflow run
    with mlflow.start_run():
        # Load data and model
        X_test = pd.read_csv(X_TEST_PATH)
        y_test = pd.read_csv(Y_TEST_PATH).squeeze()  # Ensure target column is a Series
        model = joblib.load(MODEL_PATH)

        # Get predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]  # For ROC AUC

        # Confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        sensitivity, specificity = calculate_sensitivity_specificity(conf_matrix)

        # Standard metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        area_under_roc = roc_auc_score(y_test, y_prob)

        # Print metrics
        print(f"Confusion Matrix: \n{conf_matrix}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Sensitivity: {sensitivity:.4f}")
        print(f"Specificity: {specificity:.4f}")
        print(f"Area Under ROC: {area_under_roc:.4f}")

        # Log model and metrics to MLflow
        mlflow_sklearn.log_model(model, "model")
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("sensitivity", sensitivity)
        mlflow.log_metric("specificity", specificity)
        mlflow.log_metric("area_under_roc", area_under_roc)


if __name__ == "__main__":
    evaluate()
