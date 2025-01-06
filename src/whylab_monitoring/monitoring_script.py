import os
import joblib
import whylogs as why
import pandas as pd
import warnings
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from datetime import datetime
import pytz
from whylogs.core import DatasetProfileView
from whylogs.api.logger.result_set import ViewResultSet
from prefect import flow, task
from datetime import timedelta

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure WhyLabs API
os.environ["WHYLABS_DEFAULT_ORG_ID"] = "org-jAKdPA" # ORG-ID is case sensitive
os.environ["WHYLABS_API_KEY"] = "UIn56UYSLA.BtGRwJIMbbzlHp0oYyGMGB9mauslM1MOHD9zmo9Lsl0nr1elwgs9W:org-jAKdPA"
os.environ["WHYLABS_DEFAULT_DATASET_ID"] = "model-3" # The selected project "mi_fatality_prediction (model-2)" is "model-2"

@task
def process_data(training_data_path):
    data = pd.read_csv(training_data_path)
    print(f"Dataset loaded: {data.shape[0]} rows, {data.shape[1]} columns.")

    # Remove columns with more than `missing_threshold` missing values
    missing_fraction = data.isnull().mean()
    cols_to_drop = missing_fraction[missing_fraction > 0.5].index
    data = data.drop(columns=cols_to_drop)
    print(f"Removed {len(cols_to_drop)} columns with more than 50% missing values.")

    # Reduce to binary classification
    data['output'] = data['output'].apply(lambda x: 1 if x > 1 else x)
    print(f"Reduced target column to binary classification.")

    # Separate features and target
    X = data.drop('output', axis=1)
    y = data['output']

    # Impute missing values in remaining columns
    imputer = SimpleImputer(strategy="mean")  # Replace missing values with column mean
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    print("Missing values imputed.")

    # Oversample the minority class using SMOTE
    smote = SMOTE(random_state=42)
    data, y_resampled = smote.fit_resample(X_imputed, y)
    print("Oversampled the minority class using SMOTE.")
    data['output'] = y_resampled
    return data

# Log and Save Training Data Profile
@task
def log_training_data(training_data_path):
    # Load training data
    training_data = process_data(training_data_path)

    # Generate a data profile
    reference_profile = why.log(pandas=training_data)

    # Save the training profile locally
    training_profile_path = "../../data/training_data_profile.bin"
    reference_profile.writer("local").write(training_profile_path)

    return reference_profile, training_profile_path

@task
def log_new_data(new_data_path):

    # loan and preprocess new data
    new_data = process_data(new_data_path)

    ground_truth = new_data[['output']]
    new_data = new_data.drop("output", axis = 1)
    
    # Generate a data profile
    target_profile = why.log(pandas=new_data)

    # Save the training profile locally
    new_data_profile_path = "../../data/new_data_profile.bin"
    target_profile.writer("local").write(new_data_profile_path)

    return target_profile, new_data_profile_path, new_data, ground_truth

# compare training and new data profile
@task
def compare_data(training_profile_path, new_data_profile_path):

    # Load reference and target profiles
    reference_profile = DatasetProfileView.read(training_profile_path)
    target_profile = DatasetProfileView.read(new_data_profile_path)

    #comparison_report = reference_profile.merge(target_profile)
    #print("Comparison Report:\n", comparison_report)
    # Upload reference profile to WhyLabs
    reference_profile.writer("whylabs").write()

    # Upload target profile to WhyLabs
    target_profile.writer("whylabs").write()

    return reference_profile, target_profile

# Predict and Log Model Outputs
@task
def log_predictions(new_data, model, ground_truth):
    # Predict labels and confidence scores
    prediction = model.predict(new_data)
    results = pd.DataFrame({
        "prediction": prediction,
    }, index=new_data.index)

    new_data = pd.concat([new_data, results, ground_truth], axis=1)

    results = why.log_classification_metrics(new_data,
                                     target_column = "output",
                                     prediction_column = "prediction"
    )
        
    # set dataset_timestamp using a datetime object (optional)
    dataset_timestamp = datetime.now(pytz.timezone("US/Eastern"))
    profile = results.profile()
    # write profile to whylabs
    results.writer("whylabs").write()
    profile.set_dataset_timestamp(dataset_timestamp)

    return new_data

@flow
def whylab_pipeline():
    training_data_path = "../../data/raw_data.csv"
    new_data_path = "../../data/new_data.csv"
    training_profile_path = "../../data/training_data_profile.bin"
    new_data_profile_path = "../../data/new_data_profile.bin"
    model_path = "../../model/best_model.joblib"

    # Load the actual model
    model = joblib.load(model_path)
    data = process_data(training_data_path)
    reference_profile, training_profile_pat = log_training_data(training_data_path)
    target_profile, new_data_profile_path, new_data, ground_truth = log_new_data(new_data_path)
    reference_profile, target_profile = compare_data(training_profile_path, new_data_profile_path)
    new_data = log_predictions(new_data, model, ground_truth)

if __name__ == "__main__":
    whylab_pipeline.serve(name="whylab_monitor_for_mi_fatality_prediction", cron="* * * * *")
    #whylab_monitor_for_mi_fatality_prediction()

# prefect 3.1.8 environment
# python 3.10
# run prefect server start to start prefect
# run script my mavigating to its folder and running python monitoring_script.py