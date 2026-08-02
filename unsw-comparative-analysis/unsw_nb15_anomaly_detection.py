"""
UNSW-NB15 Network Anomaly Detection
====================================
Compares Isolation Forest against Random Forest, Decision Tree, K-Means,
and Logistic Regression baselines for network intrusion / anomaly
detection on the UNSW-NB15 dataset.

Reproduces the methodology from:
    Siddhesh T.S., Shinu M. Rajagopal, Sreebha Bhaskaran,
    "Comparative Analysis of Machine Learning Algorithms for Anomaly
    Detection", 2024 IEEE 9th International Conference for Convergence
    in Technology (I2CT).

Dataset
-------
UNSW-NB15 (Moustafa & Slay, 2015), official source:
    https://research.unsw.edu.au/projects/unsw-nb15-dataset
Also mirrored on Kaggle, e.g.:
    https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15

Usage
-----
    python unsw_nb15_anomaly_detection.py --data-path data/UNSW-NB15_1.csv
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42

# Column header for the raw UNSW-NB15 CSV files (they ship without a header row).
HEADER = [
    "srcip", "sport", "dstip", "dsport", "proto", "state", "dur", "sbytes",
    "dbytes", "sttl", "dttl", "sloss", "dloss", "service", "Sload", "Dload",
    "Spkts", "Dpkts", "swin", "dwin", "stcpb", "dtcpb", "smeansz", "dmeansz",
    "trans_depth", "res_bdy_len", "Sjit", "Djit", "Stime", "Ltime",
    "Sintpkt", "Dintpkt", "tcprtt", "synack", "ackdat", "is_sm_ips_ports",
    "ct_state_ttl", "ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd",
    "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm",
    "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm", "attack_cat",
    "Label",
]

NUMERICAL_COLUMNS = [
    "dur", "sbytes", "dbytes", "Sload", "Dload", "Spkts", "Dpkts", "stcpb",
    "dtcpb", "smeansz", "dmeansz", "Sjit", "Djit", "Stime", "Ltime",
    "Sintpkt", "Dintpkt", "tcprtt", "synack", "ackdat",
]

CATEGORICAL_COLUMNS = [
    "srcip", "dstip", "sport", "dsport", "proto", "state", "sttl", "dttl",
    "sloss", "dloss", "service", "swin", "dwin", "trans_depth",
    "res_bdy_len", "is_sm_ips_ports", "ct_state_ttl", "ct_flw_http_mthd",
    "is_ftp_login", "ct_ftp_cmd", "ct_srv_src", "ct_srv_dst", "ct_dst_ltm",
    "ct_src_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
    "Label",
]

# Columns dropped after correlation analysis (kept the less redundant of
# each highly-correlated pair, see paper Section III-B).
DROP_AFTER_CORRELATION = [
    "synack", "dwin", "Spkts", "ackdat", "Dpkts", "ct_state_ttl",
    "ct_ftp_cmd", "Dintpkt", "dloss", "state", "swin", "res_bdy_len",
]


@dataclass
class ModelResult:
    name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    train_time: float
    test_time: float
    confusion: np.ndarray = field(repr=False)

    def as_row(self) -> dict:
        return {
            "Model": self.name,
            "Accuracy (%)": round(self.accuracy * 100, 2),
            "Precision (%)": round(self.precision * 100, 2),
            "Recall (%)": round(self.recall * 100, 2),
            "F1-score (%)": round(self.f1 * 100, 2),
            "Train time (s)": round(self.train_time, 3),
            "Test time (s)": round(self.test_time, 3),
        }


def load_data(path: str, nrows: int | None = None) -> pd.DataFrame:
    """Load a raw UNSW-NB15 CSV part file and attach the official header."""
    df = pd.read_csv(path, names=HEADER, low_memory=False, nrows=nrows)
    return df


def clean_and_encode(df: pd.DataFrame) -> pd.DataFrame:
    """Null handling, de-duplication, categorical encoding and scaling."""
    df = df.replace(to_replace="-", value=np.nan)
    df = df.drop(columns=["attack_cat"])
    df = df.dropna(axis=0).drop_duplicates()

    # sport / dsport occasionally contain hex strings -> coerce to numeric.
    df["sport"] = pd.to_numeric(df["sport"], errors="coerce")
    df["dsport"] = pd.to_numeric(df["dsport"], errors="coerce")
    df = df.dropna()

    label_encoder = preprocessing.LabelEncoder()
    for col in CATEGORICAL_COLUMNS:
        df[col] = label_encoder.fit_transform(df[col].astype(str))

    scaler = preprocessing.StandardScaler()
    for col in NUMERICAL_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[[col]] = scaler.fit_transform(df[[col]])
    df = df.dropna()

    df = df.drop(columns=[c for c in DROP_AFTER_CORRELATION if c in df.columns])
    return df


def get_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Everything except the final 'Label' column is a feature."""
    X = df.drop(columns=["Label"])
    y = df["Label"]
    return X, y


def _evaluate(name, y_test, y_pred, train_time, test_time) -> ModelResult:
    return ModelResult(
        name=name,
        accuracy=accuracy_score(y_test, y_pred),
        precision=precision_score(y_test, y_pred, average="weighted", zero_division=0),
        recall=recall_score(y_test, y_pred, average="weighted", zero_division=0),
        f1=f1_score(y_test, y_pred, average="weighted", zero_division=0),
        train_time=train_time,
        test_time=test_time,
        confusion=confusion_matrix(y_test, y_pred),
    )


def run_isolation_forest(X, y, contamination=0.05) -> ModelResult:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    model = IsolationForest(contamination=contamination, random_state=RANDOM_STATE)

    t0 = time.time()
    model.fit(X_train)
    train_time = time.time() - t0

    t0 = time.time()
    raw_pred = model.predict(X_test)
    test_time = time.time() - t0

    # IsolationForest returns -1 for anomalies, 1 for inliers -> map to {0,1}.
    y_pred = np.where(raw_pred == 1, 0, 1)
    return _evaluate("Isolation Forest", y_test, y_pred, train_time, test_time)


def run_kmeans(X, y, n_clusters=2) -> ModelResult:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    model = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)

    t0 = time.time()
    model.fit(X_train)
    train_time = time.time() - t0

    t0 = time.time()
    clusters = model.predict(X_test)
    test_time = time.time() - t0

    anomaly_cluster = np.argmin(np.bincount(clusters))
    y_pred = np.where(clusters == anomaly_cluster, 1, 0)
    return _evaluate("K-Means", y_test, y_pred, train_time, test_time)


def run_decision_tree(X, y, max_depth=None) -> ModelResult:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=RANDOM_STATE)

    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    t0 = time.time()
    y_pred = model.predict(X_test)
    test_time = time.time() - t0

    return _evaluate("Decision Tree", y_test, y_pred, train_time, test_time)


def run_random_forest(X, y, n_estimators=100, max_depth=None) -> ModelResult:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=RANDOM_STATE
    )

    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    t0 = time.time()
    y_pred = model.predict(X_test)
    test_time = time.time() - t0

    return _evaluate("Random Forest", y_test, y_pred, train_time, test_time)


def run_logistic_regression(X, y, max_iter=200) -> ModelResult:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    model = LogisticRegression(max_iter=max_iter, random_state=RANDOM_STATE)

    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    t0 = time.time()
    y_pred = model.predict(X_test)
    test_time = time.time() - t0

    return _evaluate("Logistic Regression", y_test, y_pred, train_time, test_time)


def run_all_models(X, y, contamination=0.05) -> list[ModelResult]:
    return [
        run_isolation_forest(X, y, contamination=contamination),
        run_kmeans(X, y),
        run_decision_tree(X, y, max_depth=3),
        run_random_forest(X, y),
        run_logistic_regression(X, y),
    ]


def results_table(results: list[ModelResult]) -> pd.DataFrame:
    return pd.DataFrame([r.as_row() for r in results]).sort_values(
        "Accuracy (%)", ascending=False
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path", required=True,
        help="Path to a raw UNSW-NB15 CSV file (e.g. UNSW-NB15_1.csv)",
    )
    parser.add_argument(
        "--nrows", type=int, default=None,
        help="Optionally limit the number of rows read (useful for a quick run)",
    )
    parser.add_argument(
        "--contamination", type=float, default=0.05,
        help="Expected proportion of anomalies for Isolation Forest",
    )
    parser.add_argument(
        "--output-csv", default="results/unsw_nb15_results.csv",
        help="Where to save the comparison table",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading data from {args.data_path} ...")
    df = load_data(args.data_path, nrows=args.nrows)
    print(f"Raw shape: {df.shape}")

    print("Cleaning, encoding and scaling ...")
    df = clean_and_encode(df)
    print(f"Processed shape: {df.shape}")

    X, y = get_features_and_target(df)

    print("Training and evaluating models ...")
    results = run_all_models(X, y, contamination=args.contamination)

    table = results_table(results)
    print("\n" + table.to_string(index=False))

    import os
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    table.to_csv(args.output_csv, index=False)
    print(f"\nSaved results to {args.output_csv}")


if __name__ == "__main__":
    main()
