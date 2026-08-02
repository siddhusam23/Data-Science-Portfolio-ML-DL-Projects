"""
Malayalam Palm Leaf Character Recognition using Gabor Features
=================================================================
Compares ten classifiers - Logistic Regression, SVM, Random Forest,
Decision Tree, KNN, MLP, Naive Bayes, XGBoost, CatBoost, and AdaBoost -
on a Gabor-feature representation of Malayalam palm-leaf characters,
both with default hyperparameters and after GridSearchCV tuning, then
uses LIME to explain the best model's predictions.

Reproduces the methodology from:
    Siddhesh T. S., P. Revanth Krishna Varma, P. Gowtham, Achyuta Siva
    Sai Kowshik, Ambati Sai Sindhur, Annem Gnaneswara Reddy, Remya
    Sivan, Peeta Basa Pati, "Malayalam Palm Leaf Character Recognition
    using Gabor Features".

Dataset
-------
Expects a CSV of pre-extracted Gabor features with one label column
(default column name: `label`). See `gabor_feature_extraction.py` to
build this file from a folder of character images, and `data/README.md`
for public image dataset sources.

Usage
-----
    python malayalam_char_recognition.py \
        --data-path data/malayalam_char_gabor.csv \
        --label-col label
"""

from __future__ import annotations

import argparse
import os
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

RANDOM_STATE = 42


@dataclass
class ModelReport:
    name: str
    test_accuracy: float
    train_accuracy: float
    test_precision: float
    train_precision: float
    test_recall: float
    train_recall: float
    test_f1: float
    train_f1: float

    def as_row(self) -> dict:
        return {
            "Classifier": self.name,
            "Accuracy (Test %)": round(self.test_accuracy * 100, 2),
            "Accuracy (Train %)": round(self.train_accuracy * 100, 2),
            "Precision (Test %)": round(self.test_precision * 100, 2),
            "Precision (Train %)": round(self.train_precision * 100, 2),
            "Recall (Test %)": round(self.test_recall * 100, 2),
            "Recall (Train %)": round(self.train_recall * 100, 2),
            "F1-score (Test %)": round(self.test_f1 * 100, 2),
            "F1-score (Train %)": round(self.train_f1 * 100, 2),
        }


# ---------------------------------------------------------------------
# Data loading and preprocessing
# ---------------------------------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(axis=0)
    return df


def preprocess(df: pd.DataFrame, label_col: str):
    """Min-max normalize the feature columns and label-encode the target."""
    X = df.drop(columns=[label_col])
    y_raw = df[label_col]

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)

    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    return X_scaled, y, encoder


# ---------------------------------------------------------------------
# Model zoo: default estimator + hyperparameter grid for tuning
# ---------------------------------------------------------------------

def get_model_zoo() -> dict:
    zoo = {
        "LR": (
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            {"C": [0.1, 1, 10], "penalty": ["l2"]},
        ),
        "SVM": (
            SVC(probability=True, random_state=RANDOM_STATE),
            {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"]},
        ),
        "RF": (
            RandomForestClassifier(random_state=RANDOM_STATE),
            {"n_estimators": [100, 200], "max_depth": [None, 10, 20]},
        ),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=RANDOM_STATE),
            {"max_depth": [None, 5, 10, 20], "min_samples_split": [2, 5, 10]},
        ),
        "KNN": (
            KNeighborsClassifier(),
            {"n_neighbors": [3, 5, 7, 9], "weights": ["uniform", "distance"]},
        ),
        "MLP": (
            MLPClassifier(max_iter=500, random_state=RANDOM_STATE),
            {"hidden_layer_sizes": [(64,), (100,), (64, 32)], "alpha": [0.0001, 0.001]},
        ),
        "Naive Bayes": (
            GaussianNB(),
            {"var_smoothing": [1e-9, 1e-8, 1e-7]},
        ),
        "XG-Boost": (
            None,  # populated below if xgboost is installed
            {"n_estimators": [100, 200], "max_depth": [3, 5, 7], "learning_rate": [0.05, 0.1]},
        ),
        "CAT-Boost": (
            None,  # populated below if catboost is installed
            {"depth": [4, 6, 8], "learning_rate": [0.05, 0.1]},
        ),
        "ADA-Boost": (
            AdaBoostClassifier(random_state=RANDOM_STATE),
            {"n_estimators": [50, 100, 200], "learning_rate": [0.5, 1.0]},
        ),
    }

    try:
        from xgboost import XGBClassifier
        zoo["XG-Boost"] = (
            XGBClassifier(random_state=RANDOM_STATE, eval_metric="mlogloss"),
            zoo["XG-Boost"][1],
        )
    except ImportError:
        del zoo["XG-Boost"]
        print("xgboost not installed - skipping XG-Boost (`pip install xgboost`).")

    try:
        from catboost import CatBoostClassifier
        zoo["CAT-Boost"] = (
            CatBoostClassifier(random_state=RANDOM_STATE, verbose=0),
            zoo["CAT-Boost"][1],
        )
    except ImportError:
        del zoo["CAT-Boost"]
        print("catboost not installed - skipping CAT-Boost (`pip install catboost`).")

    return zoo


# ---------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------

def _metrics(y_true, y_pred) -> tuple[float, float, float, float]:
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    return acc, prec, rec, f1


def evaluate_model(name, model, X_train, X_test, y_train, y_test) -> ModelReport:
    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_acc, train_prec, train_rec, train_f1 = _metrics(y_train, train_pred)
    test_acc, test_prec, test_rec, test_f1 = _metrics(y_test, test_pred)

    return ModelReport(
        name=name,
        test_accuracy=test_acc, train_accuracy=train_acc,
        test_precision=test_prec, train_precision=train_prec,
        test_recall=test_rec, train_recall=train_rec,
        test_f1=test_f1, train_f1=train_f1,
    )


def run_baseline(zoo, X_train, X_test, y_train, y_test) -> tuple[list[ModelReport], dict]:
    """Train every model in the zoo with its default hyperparameters."""
    reports = []
    fitted = {}
    for name, (model, _grid) in zoo.items():
        print(f"  training {name} (baseline) ...")
        report = evaluate_model(name, model, X_train, X_test, y_train, y_test)
        reports.append(report)
        fitted[name] = model
    return reports, fitted


def run_tuned(zoo, X_train, X_test, y_train, y_test, cv=3) -> tuple[list[ModelReport], dict]:
    """Re-fit every model after a GridSearchCV hyperparameter search."""
    reports = []
    fitted = {}
    for name, (model, grid) in zoo.items():
        print(f"  tuning {name} with GridSearchCV ...")
        search = GridSearchCV(model, grid, cv=cv, n_jobs=-1, scoring="accuracy")
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        report = evaluate_model(name, best_model, X_train, X_test, y_train, y_test)
        reports.append(report)
        fitted[name] = best_model
        print(f"    best params: {search.best_params_}")
    return reports, fitted


def results_table(reports: list[ModelReport]) -> pd.DataFrame:
    return pd.DataFrame([r.as_row() for r in reports]).sort_values(
        "Accuracy (Test %)", ascending=False
    )


# ---------------------------------------------------------------------
# LIME interpretability
# ---------------------------------------------------------------------

def explain_with_lime(model, X_train, X_test, class_names, feature_names,
                       instance_index=0, num_features=10, output_html=None):
    from lime.lime_tabular import LimeTabularExplainer

    explainer = LimeTabularExplainer(
        training_data=np.asarray(X_train),
        feature_names=feature_names,
        class_names=[str(c) for c in class_names],
        mode="classification",
    )

    instance = np.asarray(X_test)[instance_index]
    explanation = explainer.explain_instance(
        instance, model.predict_proba, num_features=num_features
    )

    print("\nLIME explanation for test instance", instance_index)
    for feature, weight in explanation.as_list():
        print(f"  {feature}: {weight:+.4f}")

    if output_html:
        os.makedirs(os.path.dirname(output_html) or ".", exist_ok=True)
        explanation.save_to_file(output_html)
        print(f"Saved LIME explanation to {output_html}")

    return explanation


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", required=True, help="Path to the Gabor-feature CSV")
    parser.add_argument("--label-col", default="label", help="Name of the label/target column")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--skip-tuning", action="store_true", help="Skip the GridSearchCV pass")
    parser.add_argument("--skip-lime", action="store_true", help="Skip the LIME explanation step")
    parser.add_argument("--cv", type=int, default=3, help="Cross-validation folds for GridSearchCV")
    parser.add_argument("--lime-instance", type=int, default=0, help="Test-set row index to explain")
    parser.add_argument("--results-dir", default="results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading data from {args.data_path} ...")
    df = load_data(args.data_path)
    print(f"Shape: {df.shape}, classes: {df[args.label_col].nunique()}")

    X, y, encoder = preprocess(df, args.label_col)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=RANDOM_STATE, stratify=y
    )

    zoo = get_model_zoo()

    print("\n=== Baseline (no hyperparameter tuning) ===")
    baseline_reports, baseline_fitted = run_baseline(zoo, X_train, X_test, y_train, y_test)
    baseline_table = results_table(baseline_reports)
    print("\n" + baseline_table.to_string(index=False))

    os.makedirs(args.results_dir, exist_ok=True)
    baseline_table.to_csv(os.path.join(args.results_dir, "results_before_tuning.csv"), index=False)

    best_name = baseline_table.iloc[0]["Classifier"]
    best_model = baseline_fitted[best_name]

    if not args.skip_tuning:
        print("\n=== After GridSearchCV hyperparameter tuning ===")
        tuned_reports, tuned_fitted = run_tuned(zoo, X_train, X_test, y_train, y_test, cv=args.cv)
        tuned_table = results_table(tuned_reports)
        print("\n" + tuned_table.to_string(index=False))
        tuned_table.to_csv(os.path.join(args.results_dir, "results_after_tuning.csv"), index=False)

        best_name = tuned_table.iloc[0]["Classifier"]
        best_model = tuned_fitted[best_name]

    print(f"\nBest classifier: {best_name}")

    if not args.skip_lime:
        print("\n=== LIME interpretability for the best classifier ===")
        try:
            explain_with_lime(
                best_model, X_train, X_test,
                class_names=encoder.classes_,
                feature_names=list(X.columns),
                instance_index=args.lime_instance,
                output_html=os.path.join(args.results_dir, "lime_explanation.html"),
            )
        except ImportError:
            print("lime not installed - skipping interpretability step (`pip install lime`).")


if __name__ == "__main__":
    main()
