"""
Evaluation script — computes accuracy, precision, recall, F1-score, and a
confusion matrix on the held-out test set (Chapter 6 of the report).

Usage:
    python evaluate.py --model saved_models/custom_cnn_best.h5
"""

import argparse
import os

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
)

import config
from data_preprocessing import build_test_generator


def evaluate_model(model_path: str, test_dir: str = config.TEST_DIR,
                    img_size: tuple = None, save_plots: bool = True):
    model = tf.keras.models.load_model(model_path)

    if img_size is None:
        img_size = config.IMG_SIZE if "custom_cnn" in model_path else config.IMG_SIZE_TRANSFER

    test_gen = build_test_generator(test_dir=test_dir, img_size=img_size, batch_size=1)
    test_gen.reset()

    y_true = test_gen.classes
    preds = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(preds, axis=1)

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    print(f"\nAccuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}\n")

    print(classification_report(y_true, y_pred, target_names=config.CLASS_NAMES, zero_division=0))

    if save_plots:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(16, 14))
        sns.heatmap(cm, annot=False, cmap="Blues",
                    xticklabels=config.CLASS_NAMES, yticklabels=config.CLASS_NAMES)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix — {os.path.basename(model_path)}")
        plt.tight_layout()
        out_path = os.path.join(config.RESULTS_DIR, "confusion_matrix.png")
        plt.savefig(out_path, dpi=150)
        print(f"Confusion matrix plot saved to {out_path}")

    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained ASL model on the test set.")
    parser.add_argument("--model", required=True, help="Path to a trained .h5 model file.")
    parser.add_argument("--test-dir", default=config.TEST_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_model(args.model, args.test_dir)
