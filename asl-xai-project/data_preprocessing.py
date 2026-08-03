"""
Data loading and preprocessing pipeline for ASL Alphabet recognition.

Implements the preprocessing steps described in the project report:
  1. Resizing to a fixed input dimension
  2. Normalization (pixel values scaled to [0, 1])
  3. Data augmentation (flips, rotation, zoom, brightness)
  4. Label encoding (one-hot, handled automatically by Keras generators)
  5. Train / validation / test split
"""

import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import config


def build_data_generators(
    train_dir: str = config.TRAIN_DIR,
    img_size: tuple = config.IMG_SIZE,
    batch_size: int = config.BATCH_SIZE,
    val_split: float = config.VAL_SPLIT,
):
    """
    Creates training and validation generators from a directory of class
    subfolders (e.g. train_dir/A/, train_dir/B/, ..., train_dir/NOTHING/).

    Returns:
        train_gen, val_gen
    """
    datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.15,
        brightness_range=[0.8, 1.2],
        horizontal_flip=False,  # ASL signs are orientation-sensitive; keep this off by default
        validation_split=val_split,
    )

    train_gen = datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="training",
        seed=config.RANDOM_SEED,
        classes=config.CLASS_NAMES,
    )

    val_gen = datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation",
        seed=config.RANDOM_SEED,
        classes=config.CLASS_NAMES,
    )

    return train_gen, val_gen


def build_test_generator(
    test_dir: str = config.TEST_DIR,
    img_size: tuple = config.IMG_SIZE,
    batch_size: int = config.BATCH_SIZE,
):
    """
    Creates a generator for the held-out test set. No augmentation is applied,
    only rescaling, so evaluation reflects real-world performance.
    """
    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    test_gen = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
        classes=config.CLASS_NAMES,
    )
    return test_gen


def preprocess_single_image(image_path: str, img_size: tuple = config.IMG_SIZE):
    """
    Loads and preprocesses a single image for inference (used by predict.py
    and gradcam.py).
    """
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=img_size)
    array = tf.keras.preprocessing.image.img_to_array(img)
    array = array / 255.0
    return array


if __name__ == "__main__":
    print("This module provides data-loading utilities.")
    print(f"Expecting training data at: {config.TRAIN_DIR}")
    print(f"Classes ({config.NUM_CLASSES}): {config.CLASS_NAMES}")
