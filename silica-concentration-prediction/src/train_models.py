"""
Prediction of Silica Concentration in Iron Ore Mining
--------------------------------------------------------
Trains and compares six regression models (Linear Regression, Random Forest,
K-Nearest Neighbors, Support Vector Regressor, Polynomial Regression, and
Decision Tree) to predict % Silica Concentrate from the froth flotation
process dataset.

Usage:
    python src/train_models.py --data data/MiningProcess_Flotation_Plant_Database.csv
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # allows running headless / saving plots without a display
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn import metrics

FEATURES = [
    "% Iron Feed", "% Silica Feed", "Starch Flow", "Amina Flow",
    "Ore Pulp Flow", "Ore Pulp pH", "Ore Pulp Density",
    "Flotation Column 01 Air Flow", "Flotation Column 02 Air Flow",
    "Flotation Column 03 Air Flow", "Flotation Column 04 Air Flow",
    "Flotation Column 05 Air Flow", "Flotation Column 06 Air Flow",
    "Flotation Column 07 Air Flow", "Flotation Column 01 Level",
    "Flotation Column 02 Level", "Flotation Column 03 Level",
    "Flotation Column 04 Level", "Flotation Column 05 Level",
    "Flotation Column 06 Level", "Flotation Column 07 Level",
    "% Iron Concentrate",
]
TARGET = "% Silica Concentrate"


def load_data(path: str) -> pd.DataFrame:
    """Load the mining process CSV. The source file uses ',' as a decimal separator."""
    df = pd.read_csv(path, decimal=",")
    return df


def plot_heatmap(df: pd.DataFrame, out_dir: str) -> None:
    plt.figure(figsize=(20, 16))
    sns.heatmap(df.drop(columns=["date"], errors="ignore").corr(), annot=False, cmap="rocket")
    plt.title("Correlation Heat Map of Dataset Features")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "correlation_heatmap.png"), dpi=150)
    plt.close()


def evaluate(name: str, y_test, pred, out_dir: str, color="steelblue") -> dict:
    mae = metrics.mean_absolute_error(y_test, pred)
    mse = metrics.mean_squared_error(y_test, pred)
    rmse = np.sqrt(mse)

    print(f"[{name}] MAE={mae:.3f}  MSE={mse:.3f}  RMSE={rmse:.3f}")

    plt.figure(figsize=(6, 5))
    sns.regplot(x=y_test, y=pred, scatter_kws={"color": color, "edgecolor": "white"},
                line_kws={"color": "black"})
    plt.xlabel(TARGET)
    plt.ylabel("Predicted")
    plt.title(f"Predicted vs Actual — {name}")
    plt.tight_layout()
    safe_name = name.lower().replace(" ", "_")
    plt.savefig(os.path.join(out_dir, f"{safe_name}_regplot.png"), dpi=150)
    plt.close()

    return {"Model": name, "MAE": mae, "MSE": mse, "RMSE": rmse}


def main():
    parser = argparse.ArgumentParser(description="Train silica concentration regression models")
    parser.add_argument("--data", type=str, default="data/MiningProcess_Flotation_Plant_Database.csv",
                         help="Path to the dataset CSV file")
    parser.add_argument("--out", type=str, default="images", help="Directory to save plots")
    parser.add_argument("--test-size", type=float, default=0.33)
    parser.add_argument("--random-state", type=int, default=101)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    df = load_data(args.data)
    plot_heatmap(df, args.out)

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=args.random_state),
        "KNN": KNeighborsRegressor(n_neighbors=5),
        "SVR": SVR(),
        "Polynomial Regression": make_pipeline(PolynomialFeatures(degree=2), LinearRegression()),
        "Decision Tree": DecisionTreeRegressor(random_state=args.random_state),
    }

    colors = {
        "Linear Regression": "red",
        "Random Forest": "orange",
        "KNN": "green",
        "SVR": "salmon",
        "Polynomial Regression": "mediumpurple",
        "Decision Tree": "royalblue",
    }

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        results.append(evaluate(name, y_test, pred, args.out, colors.get(name, "steelblue")))

    results_df = pd.DataFrame(results).sort_values("RMSE")
    results_df.to_csv(os.path.join(args.out, "model_comparison.csv"), index=False)
    print("\nModel comparison (sorted by RMSE):")
    print(results_df.to_string(index=False))

    plt.figure(figsize=(9, 5))
    sns.barplot(data=results_df, x="Model", y="RMSE", palette="pastel")
    plt.title("Root Mean Squared Error by Model")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "rmse_comparison.png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
