"""
Simple single-image prediction (no Grad-CAM) — useful for quick sanity checks.

Usage:
    python predict.py --image sample_data/example_a.jpg --model saved_models/custom_cnn_best.h5
"""

import argparse

import numpy as np
import tensorflow as tf

import config
from data_preprocessing import preprocess_single_image


def predict(image_path: str, model_path: str, img_size: tuple = config.IMG_SIZE, top_k: int = 3):
    model = tf.keras.models.load_model(model_path)
    img_array = preprocess_single_image(image_path, img_size=img_size)
    preds = model.predict(np.expand_dims(img_array, axis=0), verbose=0)[0]

    top_indices = np.argsort(preds)[::-1][:top_k]
    print(f"\nTop {top_k} predictions for {image_path}:")
    for idx in top_indices:
        print(f"  {config.CLASS_NAMES[idx]:<8s} {preds[idx]*100:6.2f}%")

    return config.CLASS_NAMES[top_indices[0]], float(preds[top_indices[0]])


def parse_args():
    parser = argparse.ArgumentParser(description="Predict the ASL class of a single image.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    predict(args.image, args.model, top_k=args.top_k)
