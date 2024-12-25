from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import os
import subprocess

# Define the paths to your scripts and data directories
BASE_DIR = os.path.abspath(os.path.dirname(__file__))  # Adjust to the script's directory
DATA_DIR = os.path.join(BASE_DIR, "../../data")  # Adjust to your local directory structure
MODEL_DIR = os.path.join(BASE_DIR, "../../model")  # Adjust to your local directory structure
INPUT_DATASET_PATH = os.path.join(DATA_DIR, "raw_data.csv")  # Adjust to the actual dataset path
PREPROCESS_SCRIPT_PATH = os.path.join(BASE_DIR, "data_preprocessing.py")  # Adjust script location
TRAIN_SCRIPT_PATH = os.path.join(BASE_DIR, "model_training.py")
EVALUATE_SCRIPT_PATH = os.path.join(BASE_DIR, "model_evaluation.py")

# Default arguments for the Airflow DAG
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2024, 12, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
}

# Define the DAG
with DAG(
    "ml_workflow_with_mlflow",
    default_args=default_args,
    schedule_interval="@daily",  # Run daily
    catchup=False,
    description="A simple ML workflow orchestrated by Airflow with MLFlow tracking",
) as dag:

    def preprocess():
        cmd = [
            'python',
            PREPROCESS_SCRIPT_PATH,
            '--input_path', INPUT_DATASET_PATH,
            '--output_dir', DATA_DIR,
            '--target_column', 'output'
        ]
        result = subprocess.run(cmd, check=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        if result.stderr:
            print(result.stderr.decode())
        if result.stdout:
            print(result.stdout.decode())

    def train_model():
        """
        Task to train the model using the `train` function.
        """
        cmd = ["python", TRAIN_SCRIPT_PATH]
        result = subprocess.run(cmd, check=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        if result.stderr:
            print(result.stderr.decode())
        if result.stdout:
            print(result.stdout.decode())

    def evaluate_model():
        """
        Task to evaluate the model using the `evaluate` function.
        """
        cmd = ["python", EVALUATE_SCRIPT_PATH]
        result = subprocess.run(cmd, check=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        if result.stderr:
            print(result.stderr.decode())
        if result.stdout:
            print(result.stdout.decode())

    # Define the tasks
    preprocess_task = PythonOperator(
        task_id="process_data",
        python_callable=preprocess,
    )

    train_task = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
    )

    evaluate_task = PythonOperator(
        task_id="evaluate_model",
        python_callable=evaluate_model,
    )

    # Define task dependencies
    preprocess_task >> train_task >> evaluate_task
