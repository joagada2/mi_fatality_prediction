import warnings
import os
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Suppress warnings
warnings.filterwarnings("ignore")


def process_data():
    # Define the paths
    input_path = "../../data/raw_data.csv"
    output_dir = "../../data"

    # Load dataset
    data = pd.read_csv(input_path)
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
    X_resampled, y_resampled = smote.fit_resample(X_imputed, y)
    print("Oversampled the minority class using SMOTE.")

    # Perform train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
    )
    print(f"Train-test split completed: Train size={X_train.shape[0]}, Test size={X_test.shape[0]}")

    # Save datasets
    X_train.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)
    print("Preprocessed datasets saved.")

    # Return the data splits for potential use in further code
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    process_data()
