"""
Weather Condition (Daily Summary) Classification
--------------------------------------------------
Trains and compares five classifiers (Logistic Regression, Random Forest,
Decision Tree, Multi-Layer Perceptron, Naive Bayes) to predict the
`Daily Summary` weather category from hourly sensor readings in the
"Weather in Szeged 2006-2016" dataset.

This script is the scikit-learn distillation of notebooks/weather_classification.ipynb,
which also contains equivalent PySpark MLlib pipelines for each model.

Usage:
    python src/train_models.py --data data/weatherHistory.csv
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier

NUMERICAL_COLUMNS = [
    "Temperature (C)", "Apparent Temperature (C)", "Humidity",
    "Wind Speed (km/h)", "Wind Bearing (degrees)", "Visibility (km)",
    "Loud Cover", "Pressure (millibars)",
]
CATEGORICAL_COLUMNS = ["Summary", "Precip Type", "Daily Summary"]
TARGET = "Daily Summary"
DROP_COLUMNS = ["Formatted Date"]


def load_and_clean(path: str) -> pd.DataFrame:
    """Load the raw CSV, drop nulls/duplicates, and report before/after counts."""
    df = pd.read_csv(path)

    before = df.shape[0]
    print(f"Rows before cleaning: {before}")
    print("Null values per column:")
    print(df.isnull().sum())

    df = df.dropna().drop_duplicates()
    print(f"Rows after dropping nulls/duplicates: {df.shape[0]} (removed {before - df.shape[0]})")

    return df


def prepare_features(df: pd.DataFrame):
    """Encode categorical columns, drop unneeded ones, and split X/y."""
    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])

    label_encoders = {}
    for col in CATEGORICAL_COLUMNS:
        le = LabelEncoder()
        df[f"{col}_encoded"] = le.fit_transform(df[col])
        label_encoders[col] = le

    feature_columns = NUMERICAL_COLUMNS + ["Summary_encoded", "Precip Type_encoded"]
    X = df[feature_columns]
    y = df[TARGET]  # keep original string labels for readable reports

    return X, y, label_encoders


def plot_heatmap(df: pd.DataFrame, out_dir: str) -> None:
    numeric_df = df[NUMERICAL_COLUMNS].copy()
    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Correlation Matrix — Numerical Weather Features")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "correlation_heatmap.png"), dpi=150)
    plt.close()


def evaluate(name: str, y_test, y_pred) -> dict:
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"[{name}] Accuracy={acc:.4f}  Precision={prec:.4f}  Recall={rec:.4f}  F1={f1:.4f}")
    return {"Model": name, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1}


def main():
    parser = argparse.ArgumentParser(description="Train weather condition classifiers")
    parser.add_argument("--data", type=str, default="data/weatherHistory.csv",
                         help="Path to the raw weatherHistory.csv file")
    parser.add_argument("--out", type=str, default="images", help="Directory to save plots/results")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    df = load_and_clean(args.data)
    plot_heatmap(df, args.out)

    X, y, _ = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    # Standard-scaled features for models sensitive to feature scale
    std_scaler = StandardScaler()
    X_train_std = std_scaler.fit_transform(X_train)
    X_test_std = std_scaler.transform(X_test)

    # Min-max scaled features (non-negative) for Naive Bayes
    mm_scaler = MinMaxScaler()
    X_train_mm = mm_scaler.fit_transform(X_train)
    X_test_mm = mm_scaler.transform(X_test)

    results = []

    # --- Logistic Regression ---
    lr = LogisticRegression(max_iter=200, random_state=args.random_state)
    lr.fit(X_train_std, y_train)
    results.append(evaluate("Logistic Regression", y_test, lr.predict(X_test_std)))

    # --- Random Forest ---
    rf = RandomForestClassifier(n_estimators=100, random_state=args.random_state)
    rf.fit(X_train_std, y_train)
    results.append(evaluate("Random Forest", y_test, rf.predict(X_test_std)))

    # --- Decision Tree ---
    dt = DecisionTreeClassifier(random_state=args.random_state)
    dt.fit(X_train_std, y_train)
    results.append(evaluate("Decision Tree", y_test, dt.predict(X_test_std)))

    # --- Multi-Layer Perceptron ---
    mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=200, random_state=args.random_state)
    mlp.fit(X_train_std, y_train)
    results.append(evaluate("MLP", y_test, mlp.predict(X_test_std)))

    # --- Naive Bayes (Multinomial, needs non-negative features) ---
    nb = MultinomialNB()
    nb.fit(X_train_mm, y_train)
    results.append(evaluate("Naive Bayes", y_test, nb.predict(X_test_mm)))

    results_df = pd.DataFrame(results).sort_values("F1", ascending=False)
    results_df.to_csv(os.path.join(args.out, "model_comparison.csv"), index=False)
    print("\nModel comparison (sorted by weighted F1):")
    print(results_df.to_string(index=False))

    plt.figure(figsize=(9, 5))
    plot_df = results_df.melt(id_vars="Model", value_vars=["Accuracy", "Precision", "Recall", "F1"],
                               var_name="Metric", value_name="Score")
    sns.barplot(data=plot_df, x="Model", y="Score", hue="Metric", palette="pastel")
    plt.title("Model Comparison — Weather Condition Classification")
    plt.xticks(rotation=20)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "model_comparison.png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
