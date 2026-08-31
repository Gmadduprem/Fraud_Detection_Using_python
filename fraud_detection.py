import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

DATA_FILE = "Fraud.csv"
REQUIRED_COLUMNS = ["step", "type", "amount", "nameOrig", "oldbalanceOrg","newbalanceOrig", "nameDest", "oldbalanceDest","newbalanceDest", "isFraud", "isFlaggedFraud"]
FEATURE_COLUMNS = ["step", "type", "amount", "nameOrig", "oldbalanceOrg","newbalanceOrig", "nameDest", "oldbalanceDest", "newbalanceDest"]
TARGET_COLUMN = "isFraud"


def load_and_validate_data():
    """Load Fraud.csv and validate its structure."""
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            f"'{DATA_FILE}' was not found. Put the database CSV in the same "
            "folder as this script."
        )

    df = pd.read_csv(DATA_FILE)

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(
            "The following required columns are missing: "+ ", ".join(missing_columns))
    return df


def basic_analysis(df):
    """Print basic dataset information and create exploratory plots."""
    print("\n" + "=" * 60)
    print("DATASET INFORMATION")
    print("=" * 60)
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nMissing values:")
    print(df.isna().sum())

    print("\nDescriptive statistics:")
    print(df.describe(include="all").transpose())

    print("\nFirst five records:")
    print(df.head())

    # Transaction type distribution
    type_counts = df["type"].value_counts()

    plt.figure(figsize=(8, 6))
    plt.pie(type_counts.values,labels=type_counts.index,autopct="%1.1f%%",startangle=140)

    
    plt.title("Distribution of Transaction Types")
    plt.axis("equal")
    plt.tight_layout()
    plt.show()

    # Fraud distribution
    fraud_counts = df[TARGET_COLUMN].value_counts().sort_index()

    plt.figure(figsize=(7, 6))
    labels = [
        "Non-Fraudulent" if value == 0 else "Fraudulent"
        for value in fraud_counts.index
    ]
    plt.pie(fraud_counts.values,labels=labels,autopct="%1.1f%%",startangle=140)

    plt.title("Fraudulent vs. Non-Fraudulent Transactions")
    plt.axis("equal")
    plt.tight_layout()
    plt.show()


def build_preprocessor(X):
    """Build one consistent preprocessing pipeline for train and test data."""
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_features = X.select_dtypes(exclude=["object"]).columns.tolist()

    print("\nCategorical features:")
    print(categorical_features)

    print("\nNumerical features:")
    print(numerical_features)

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1
                ),
            ),
        ]
    )

    numerical_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")),]
    )

    preprocessor = ColumnTransformer(transformers=[("categorical", categorical_pipeline, categorical_features),("numerical", numerical_pipeline, numerical_features),])

    return preprocessor, categorical_features, numerical_features


def train_models(X_train, y_train, preprocessor):
    """Train the three models used in the original notebook."""
    models = {
        "LOGISTIC REGRESSION": Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]),
        "SUPPORT VECTOR MACHINE": Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("scaler", StandardScaler()),
            ("model", SVC(class_weight="balanced")),
        ]),
        "RANDOM FOREST CLASSIFIER": Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(
                n_estimators=200, random_state=2529, class_weight="balanced", n_jobs=-1
            )),
        ]),
    }
    trained_models = {}


def evaluate_models(models, X_test, y_test):
    """Evaluate all models and compare their accuracy."""
    results = []

    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    for name, model in models.items():
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        results.append({"Model": name,"Accuracy": accuracy})

        print(f"\n{name}")
        print(f"Accuracy: {accuracy:.4f}")

    results_df = pd.DataFrame(results)

    print("\nAccuracy comparison:")
    print(results_df.to_string(index=False))

    plt.figure(figsize=(9, 6))
    plt.bar(
        results_df["Model"],
        results_df["Accuracy"]
    )
    plt.ylim(0, 1.05)
    plt.ylabel("Accuracy")
    plt.xlabel("Models")
    plt.title("Accuracy Comparison of Models")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.show()

    return results_df


def evaluate_random_forest(rf_model, X_test, y_test):
    """Print confusion matrix and classification report for Random Forest."""
    y_pred = rf_model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    print("\n" + "=" * 60)
    print("RANDOM FOREST CLASSIFICATION RESULTS")
    print("=" * 60)

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(y_test,y_pred,target_names=["Non-Fraudulent", "Fraudulent"],zero_division=0))


def predict_sample(rf_model):
    """Predict the sample transaction used in the original notebook."""
    new_transaction = pd.DataFrame(
        [{
            "step": 0,
            "type": "TRANSFER",
            "amount": 0,
            "nameOrig": "C1",
            "oldbalanceOrg": 181.00,
            "newbalanceOrig": 181.0,
            "nameDest": "C2",
            "oldbalanceDest": 21182.0,
            "newbalanceDest": 0.0,
        }]
    )

    prediction = rf_model.predict(new_transaction)[0]

    print("\n" + "=" * 60)
    print("SAMPLE TRANSACTION PREDICTION")
    print("=" * 60)

    print("\nTransaction:")
    print(new_transaction.to_string(index=False))

    if prediction == 1:
        print("\nPrediction: FRAUDULENT TRANSACTION")
    else:
        print("\nPrediction: NON-FRAUDULENT TRANSACTION")


def main():
    print("=" * 60)
    print("FRAUD DETECTION IN FINANCIAL TRANSACTIONS")
    print("=" * 60)

    df = load_and_validate_data()

    # Remove completely empty rows only.
    df = df.dropna(how="all").copy()

    basic_analysis(df)

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].astype(int).copy()

    print("\nFeature matrix shape:", X.shape)
    print("Target shape:", y.shape)

    # Stratification keeps the fraud/non-fraud ratio in both sets.
    # If the dataset has too few samples of one class, fall back to a normal split.
    class_counts = y.value_counts()

    stratify_target = y if class_counts.min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.8,random_state=2529,stratify=stratify_target)

    print("\nTraining data shape:", X_train.shape)
    print("Testing data shape:", X_test.shape)
    print("Training target shape:", y_train.shape)
    print("Testing target shape:", y_test.shape)

    preprocessor, _, _ = build_preprocessor(X_train)

    models = train_models(X_train,y_train,preprocessor)

    evaluate_models(models,X_test,y_test)

    rf_model = models["RANDOM FOREST CLASSIFIER"]

    evaluate_random_forest(rf_model,X_test,y_test)

    predict_sample(rf_model)

    print("\nAnalysis completed successfully.")


if __name__ == "__main__":
    main()
