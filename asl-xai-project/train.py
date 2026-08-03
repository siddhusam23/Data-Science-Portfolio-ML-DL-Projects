"""
Training entry point.

Usage:
    python train.py --model custom_cnn --epochs 30
    python train.py --model resnet50 --epochs 20 --batch-size 64
"""

import argparse
import os

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

import config
from data_preprocessing import build_data_generators
from model import get_model, MODEL_BUILDERS


def parse_args():
    parser = argparse.ArgumentParser(description="Train an ASL recognition model.")
    parser.add_argument("--model", type=str, default=config.MODEL_NAME,
                         choices=list(MODEL_BUILDERS.keys()),
                         help="Which architecture to train.")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--train-dir", type=str, default=config.TRAIN_DIR)
    return parser.parse_args()


def main():
    args = parse_args()

    tf.random.set_seed(config.RANDOM_SEED)

    img_size = config.IMG_SIZE if args.model == "custom_cnn" else config.IMG_SIZE_TRANSFER

    print(f"Loading data from {args.train_dir} at resolution {img_size} ...")
    train_gen, val_gen = build_data_generators(
        train_dir=args.train_dir,
        img_size=img_size,
        batch_size=args.batch_size,
    )

    print(f"Building model: {args.model}")
    model = get_model(args.model)
    model.summary()

    checkpoint_path = os.path.join(config.MODEL_DIR, f"{args.model}_best.h5")
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=config.EARLY_STOPPING_PATIENCE,
                      restore_best_weights=True),
        ModelCheckpoint(checkpoint_path, monitor="val_accuracy",
                        save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    final_path = os.path.join(config.MODEL_DIR, f"{args.model}_final.h5")
    model.save(final_path)
    print(f"Training complete. Best model saved to {checkpoint_path}")
    print(f"Final model saved to {final_path}")

    return history


if __name__ == "__main__":
    main()
