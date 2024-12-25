import pandas as pd
from prefect import flow, task
import joblib
from sklearn.impute import SimpleImputer
import os
from datetime import datetime

# Load the trained model (Ensure your model is saved in the 'models' directory)
model_path = "../model/best_model.joblib"
model = joblib.load(model_path)


# data preprocessing task
@task
def process_data(input_path):
    """
    This function processes the data by loading it, handling missing values, and performing oversampling.
    """
    # Load dataset
    data = pd.read_csv(input_path)
    print(f"Dataset loaded from {input_path}: {data.shape[0]} rows, {data.shape[1]} columns.")

    # Remove columns with more than `missing_threshold` missing values
    missing_fraction = data.isnull().mean()
    cols_to_drop = missing_fraction[missing_fraction > 0.5].index
    data = data.drop(columns=cols_to_drop)
    print(f"Removed {len(cols_to_drop)} columns with more than 50% missing values.")

    # Impute missing values in remaining columns
    imputer = SimpleImputer(strategy="mean")  # Replace missing values with column mean
    data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
    print("Missing values imputed.")


    return data 

# prediction task
@task
def make_predictions(data):
    """
    This function will use the preprocessed data to make predictions using the trained model.
    The predictions and probabilities will be added to the original data.
    """
    # Make predictions and prediction probabilities
    predictions = model.predict(data)
    probabilities = model.predict_proba(data)

    # Add predictions and probabilities to the original DataFrame
    data['predicted_label'] = predictions
    data['predicted_prob_class_0'] = probabilities[:, 0]
    data['predicted_prob_class_1'] = probabilities[:, 1]

    return data

# task for proceesing multiple files and dynamic naminf
@task
def process_multiple_files(input_dir, output_dir):
    """
    This function processes multiple files from the input directory and saves results to the output directory.
    After processing, each file is deleted from the input folder.
    """
    # Get list of all CSV files in the input directory
    input_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]

    # Get the current date in YYYY-MM-DD format
    current_date = datetime.now().strftime('%Y-%m-%d')

    for input_file in input_files:
        input_path = os.path.join(input_dir, input_file)
        print(f"Processing file: {input_file}")

        # Step 1: Preprocess the raw data
        processed_data = process_data(input_path)

        # Step 2: Make predictions and save results
        result_data = make_predictions(processed_data)

        # Construct output filename with date, timestamp, and input filename
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        output_file = f"{os.path.splitext(input_file)[0]}_predictions_{timestamp}.csv"
        output_path = os.path.join(output_dir, output_file)

        # Save the updated DataFrame with predictions to a CSV
        result_data.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")

        # Delete the input file after processing to avoid reprocessing
        os.remove(input_path)
        print(f"Deleted processed file: {input_path}")


@flow
def batch_inferencing_pipeline():
    input_dir = "../data/input"
    output_dir = "../data/output"

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Process all files in the input directory
    process_multiple_files(input_dir, output_dir)



if __name__ == "__main__":
    batch_inferencing_pipeline.serve(name="fatality_model_batch_processing_deployment", cron="* * * * *")
    #machine_learning_pipeline()

# Note that cron(="* * * * *") means the workflow should run continuously/every second.
# To configure this to run at specific interval, check this link https://www.warp.dev/terminus/how-to-run-cron-every-hour
# You can also schedule the job the run at preferred interval in prefect ui
# if you just want to run the script without deploying the code, uncomment the last line of code
# and comment the line before the last

# To run this project, install prefect 3.1.8 in anaconder environment, install other dependencies, run prefect server start to start ui
# at localhost:4200, then cd to the folder where this script is and run python prefect_workflow.py. Project will be deployed to prefect
# server

