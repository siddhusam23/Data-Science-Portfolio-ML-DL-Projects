"""
Credit Card Transaction Fraud Detection
========================================
Binary classification of fraudulent vs. legitimate credit card
transactions using engineered temporal/demographic features, class
imbalance handling (SMOTE oversampling and random undersampling), and a
comparison of XGBoost, Logistic Regression, Decision Tree, Random
Forest, and Naive Bayes.

Dataset
-------
"Credit Card Transactions Fraud Detection Dataset" (simulated
transaction data generated with the Sparkov tool), available on Kaggle:
    https://www.kaggle.com/datasets/kartik2112/fraud-detection
Expects the `fraudTrain.csv` file from that dataset.

Usage
-----
    python credit_card_fraud_detection.py --data-path data/fraudTrain.csv
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42

AGE_BINS = [13, 19, 32, 42, 50, 62, float("inf")]
AGE_LABELS = ["Teenagers", "Young Adults", "Adults", "Middle-aged", "Seniors", "Retired"]

FEATURE_COLUMNS = [
    "amt", "gender", "lat", "long", "city_pop", "trans_hour",
    "trans_day_of_week", "age", "is_fraud", "age_category",
]

PREDICTOR_COLUMNS = [
    "amt", "city_pop", "trans_hour", "age", "gender_M", "lat", "long",
    "trans_day_of_week", "age_category_Young Adults", "age_category_Adults",
    "age_category_Middle-aged", "age_category_Seniors", "age_category_Retired",
]


@dataclass
class ModelResult:
    name: str
    accuracy: float
    confusion: np.ndarray
    report: str

    def as_row(self) -> dict:
        return {"Model": self.name, "Accuracy (%)": round(self.accuracy * 100, 2)}


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # The Kaggle CSV ships with an unnamed index column - drop it if present.
    if df.columns[0].startswith("Unnamed"):
        df = df.drop(columns=[df.columns[0]])
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive hour/day-of-week/age features and bucket age into categories."""
    df = df.copy()
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["trans_hour"] = df["trans_date_trans_time"].dt.hour
    df["trans_day_of_week"] = df["trans_date_trans_time"].dt.dayofweek + 1

    df["dob"] = pd.to_datetime(df["dob"])
    df["age"] = ((df["trans_date_trans_time"] - df["dob"]).dt.days / 365.25).astype(int)

    df["age_category"] = pd.cut(df["age"], bins=AGE_BINS, labels=AGE_LABELS, right=False)

    return df


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """One-hot encode categoricals and select the modelling predictors."""
    subset = pd.get_dummies(df[FEATURE_COLUMNS], drop_first=True)

    target_col = [c for c in subset.columns if c.startswith("is_fraud_")]
    if target_col:
        y = subset[target_col[0]]
    else:
        y = df["is_fraud"]

    predictors = [c for c in PREDICTOR_COLUMNS if c in subset.columns]
    X = subset[predictors]
    return X, y


def _evaluate(name, model, test_X, test_y) -> ModelResult:
    pred = model.predict(test_X)
    return ModelResult(
        name=name,
        accuracy=accuracy_score(test_y, pred),
        confusion=confusion_matrix(test_y, pred),
        report=classification_report(test_y, pred, digits=4),
    )


def run_all_models(X: pd.DataFrame, y: pd.Series, resample: str = "undersample") -> list[ModelResult]:
    """Train XGBoost, Logistic Regression, Decision Tree, Random Forest and
    Naive Bayes on class-imbalance-corrected training data.

    resample: 'smote' | 'undersample'
    """
    train_X, test_X, train_y, test_y = train_test_split(
        X, y, test_size=0.4, random_state=RANDOM_STATE
    )

    if resample == "smote":
        sampler = SMOTE(random_state=RANDOM_STATE)
    else:
        sampler = RandomUnderSampler(sampling_strategy="auto", random_state=RANDOM_STATE)
    train_X_res, train_y_res = sampler.fit_resample(train_X, train_y)

    results = []

    try:
        from xgboost import XGBClassifier
        xgb = XGBClassifier(random_state=RANDOM_STATE)
        xgb.fit(train_X_res, train_y_res)
        results.append(_evaluate("XGBoost", xgb, test_X, test_y))
    except ImportError:
        print("xgboost not installed - skipping XGBoost model (`pip install xgboost`).")

    logreg = LogisticRegression(random_state=RANDOM_STATE, max_iter=500)
    logreg.fit(train_X_res, train_y_res)
    results.append(_evaluate("Logistic Regression", logreg, test_X, test_y))

    dt = DecisionTreeClassifier(random_state=RANDOM_STATE)
    dt.fit(train_X_res, train_y_res)
    results.append(_evaluate("Decision Tree", dt, test_X, test_y))

    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    rf.fit(train_X_res, train_y_res)
    results.append(_evaluate("Random Forest", rf, test_X, test_y))

    nb = GaussianNB()
    nb.fit(train_X_res, train_y_res)
    results.append(_evaluate("Naive Bayes", nb, test_X, test_y))

    return results


def results_table(results: list[ModelResult]) -> pd.DataFrame:
    return pd.DataFrame([r.as_row() for r in results]).sort_values(
        "Accuracy (%)", ascending=False
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", required=True, help="Path to fraudTrain.csv")
    parser.add_argument(
        "--resample", choices=["smote", "undersample"], default="undersample",
        help="Class imbalance correction strategy applied to the training split",
    )
    parser.add_argument("--output-csv", default="results/credit_card_fraud_results.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading data from {args.data_path} ...")
    df = load_data(args.data_path)
    print(f"Raw shape: {df.shape}")

    df = engineer_features(df)
    X, y = build_feature_matrix(df)
    print(f"Feature matrix: {X.shape}, fraud rate: {100 * y.mean():.3f}%")

    print(f"Training models with '{args.resample}' resampling ...")
    results = run_all_models(X, y, resample=args.resample)

    table = results_table(results)
    print("\n" + table.to_string(index=False))

    for r in results:
        print(f"\n--- {r.name} ---\n{r.report}\nConfusion matrix:\n{r.confusion}")

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    table.to_csv(args.output_csv, index=False)
    print(f"\nSaved results to {args.output_csv}")


if __name__ == "__main__":
    main()
