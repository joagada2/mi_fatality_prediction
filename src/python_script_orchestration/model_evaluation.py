from prefect import flow, task, get_run_logger
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
import mlflow
import pandas as pd
import joblib
import os
import numpy as np

@task
def data_preprocessing(input_path="../../data/raw_data.csv", output_dir="../../data"):
    logger = get_run_logger()
    logger.info("Starting data preprocessing...")

    # Load dataset
    data = pd.read_csv(input_path)
    logger.info(f"Dataset loaded: {data.shape[0]} rows, {data.shape[1]} columns.")

    # Remove columns with more than 50% missing values
    missing_fraction = data.isnull().mean()
    cols_to_drop = missing_fraction[missing_fraction > 0.5].index
    data = data.drop(columns=cols_to_drop)
    logger.info(f"Removed {len(cols_to_drop)} columns with more than 50% missing values.")

    # Reduce to binary classification
    data['output'] = data['output'].apply(lambda x: 1 if x > 1 else x)

    # Separate features and target
    X = data.drop('output', axis=1)
    y = data['output']

    # Impute missing values
    imputer = SimpleImputer(strategy="mean")
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # Oversample the minority class using SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_imputed, y)

    # Perform train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
    )

    # Save datasets
    X_train.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)

    logger.info("Data preprocessing completed.")
    return X_train, X_test, y_train, y_test

@task
def model_training(X_train_path="../../data/X_train.csv", y_train_path="../../data/y_train.csv", model_path="../../model/best_model.joblib"):
    logger = get_run_logger()
    logger.info("Starting model training...")

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

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="accuracy",
        cv=3,
        verbose=1,
        n_jobs=-1
    )

    mlflow.set_experiment("prefectOrchestrationExperiment")
    with mlflow.start_run():
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_

        # Log hyperparameters to MLflow
        for param_name, param_value in best_params.items():
            mlflow.log_param(param_name, param_value)

        # Save the best model
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(best_model, model_path)

        # Log the model to MLflow
        mlflow.sklearn.log_model(best_model, "model")

        logger.info("Model training completed.")
        return model_path

@task
def model_evaluation(X_test_path="../../data/X_test.csv", y_test_path="../../data/y_test.csv", model_path="../../model/best_model.joblib"):
    logger = get_run_logger()
    logger.info("Starting model evaluation...")

    # Load data and model
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path)
    model = joblib.load(model_path)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    TN, FP, FN, TP = conf_matrix.ravel()
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

    # Metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "area_under_roc": roc_auc_score(y_test, y_prob)
    }

    # Log metrics to MLflow
    mlflow.set_experiment("prefectOrchestrationExperiment")
    with mlflow.start_run():
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

    logger.info("Model evaluation completed.")
    return metrics

@flow(name="Hourly ML Pipeline", schedule=IntervalSchedule(interval=timedelta(hours=1)))
def machine_learning_pipeline():
    data = data_preprocessing()
    model = model_training(data)
    metrics = model_evaluation(model)
    return metrics

# Execute the flow
if __name__ == "__main__":
    machine_learning_pipeline()
