"""
Gabor Feature Extraction for Malayalam Character Images
==========================================================
Builds the tabular "Malayalam Char Gabor" style dataset used by
`malayalam_char_recognition.py` from a folder of labelled character
images.

Expected input layout (one sub-folder per character class, standard
`ImageFolder` convention):

    images/
        character_01/
            img_0001.png
            img_0002.png
            ...
        character_02/
            ...
        ...

For each image, a bank of Gabor filters (5 spatial frequencies x 5
orientations by default) is applied and the mean and variance of each
filtered response are used as features, giving 5 x 5 x 2 = 50 features
per image plus a `label` column - matching the 51-column layout
(50 features + 1 label) described in the source paper.

Usage
-----
    python gabor_feature_extraction.py \
        --images-dir data/images \
        --output-csv data/malayalam_char_gabor.csv
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from skimage import io, color, transform
from skimage.filters import gabor_kernel
from scipy import ndimage as ndi

IMAGE_SIZE = (64, 64)
FREQUENCIES = (0.05, 0.10, 0.15, 0.20, 0.25)
ORIENTATIONS = tuple(np.arange(0, np.pi, np.pi / 5))  # 5 orientations: 0, 36, 72, 108, 144 deg


def build_gabor_bank(frequencies=FREQUENCIES, orientations=ORIENTATIONS):
    """Pre-compute a bank of Gabor kernels for every (frequency, orientation) pair."""
    kernels = []
    for frequency in frequencies:
        for theta in orientations:
            kernel = np.real(gabor_kernel(frequency, theta=theta))
            kernels.append(kernel)
    return kernels


def extract_gabor_features(image: np.ndarray, kernels: list[np.ndarray]) -> np.ndarray:
    """Convolve a grayscale image with each kernel and return [mean, var] per kernel."""
    features = np.zeros(2 * len(kernels), dtype=np.float64)
    for i, kernel in enumerate(kernels):
        filtered = ndi.convolve(image, kernel, mode="wrap")
        features[2 * i] = filtered.mean()
        features[2 * i + 1] = filtered.var()
    return features


def load_grayscale(path: str, size=IMAGE_SIZE) -> np.ndarray:
    img = io.imread(path)
    if img.ndim == 3:
        img = color.rgb2gray(img)
    img = transform.resize(img, size, anti_aliasing=True)
    return img


def build_dataset(images_dir: str, size=IMAGE_SIZE) -> pd.DataFrame:
    kernels = build_gabor_bank()
    feature_names = [f"gabor_{i}" for i in range(2 * len(kernels))]

    rows = []
    labels = sorted(
        d for d in os.listdir(images_dir) if os.path.isdir(os.path.join(images_dir, d))
    )
    if not labels:
        raise ValueError(
            f"No class sub-folders found under {images_dir}. "
            "Expected one folder per character class."
        )

    for label in labels:
        class_dir = os.path.join(images_dir, label)
        image_files = [
            f for f in os.listdir(class_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
        ]
        print(f"[{label}] {len(image_files)} images")
        for fname in image_files:
            path = os.path.join(class_dir, fname)
            try:
                img = load_grayscale(path, size=size)
            except Exception as exc:  # noqa: BLE001
                print(f"  skipping {path}: {exc}")
                continue
            feats = extract_gabor_features(img, kernels)
            rows.append(list(feats) + [label])

    df = pd.DataFrame(rows, columns=feature_names + ["label"])
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images-dir", required=True,
        help="Root folder containing one sub-folder of images per character class",
    )
    parser.add_argument(
        "--output-csv", default="data/malayalam_char_gabor.csv",
        help="Where to write the extracted feature CSV",
    )
    parser.add_argument(
        "--image-size", type=int, nargs=2, default=list(IMAGE_SIZE),
        metavar=("HEIGHT", "WIDTH"),
        help="Size images are resized to before filtering",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = build_dataset(args.images_dir, size=tuple(args.image_size))
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"\nSaved {df.shape[0]} rows x {df.shape[1]} columns to {args.output_csv}")
    print(f"Classes: {df['label'].nunique()}")


if __name__ == "__main__":
    main()
