from prefect import flow, task
from datetime import timedelta
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
import mlflow
import joblib
import os

# Data Preprocessing Task
@task
def data_preprocessing(output_dir="../../data"):
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    data = pd.read_csv("../../data/raw_data.csv")

    # Remove columns with more than 50% missing values
    missing_fraction = data.isnull().mean()
    cols_to_drop = missing_fraction[missing_fraction > 0.5].index
    data = data.drop(columns=cols_to_drop)

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
    X_train_path = os.path.join(output_dir, 'X_train.csv')
    X_test_path = os.path.join(output_dir, 'X_test.csv')
    y_train_path = os.path.join(output_dir, 'y_train.csv')
    y_test_path = os.path.join(output_dir, 'y_test.csv')

    X_train.to_csv(X_train_path, index=False)
    X_test.to_csv(X_test_path, index=False)
    y_train.to_csv(y_train_path, index=False)
    y_test.to_csv(y_test_path, index=False)

    return X_train_path, X_test_path, y_train_path, y_test_path

# Model Training Task
@task
def model_training(X_train_path, y_train_path, model_dir="../../model"):
    os.makedirs(model_dir, exist_ok=True)

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
        model_path = os.path.join(model_dir, "best_model.joblib")
        joblib.dump(best_model, model_path)

        # Log the model to MLflow
        mlflow.sklearn.log_model(best_model, "model")

    return model_path

# Model Evaluation Task
@task
def model_evaluation(X_test_path, y_test_path, model_path):
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

    return metrics

# Define your flow
@flow
def machine_learning_pipeline():
    X_train_path, X_test_path, y_train_path, y_test_path = data_preprocessing()
    model_path = model_training(X_train_path, y_train_path)
    metrics = model_evaluation(X_test_path, y_test_path, model_path)
    return metrics

if __name__ == "__main__":
    machine_learning_pipeline.serve(name="fatality_model_deployment", cron="* * * * *")
    #machine_learning_pipeline()

# Note that cron(="* * * * *") means the workflow should run continuously/every second.
# To configure this to run at specific interval, check this link https://www.warp.dev/terminus/how-to-run-cron-every-hour
# You can also schedule the job the run at preferred interval in prefect ui
# if you just want to run the script without deploying the code, uncomment the last line of code
# and comment the line before the last