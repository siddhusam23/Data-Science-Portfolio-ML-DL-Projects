"""
Central configuration for the ASL Recognition + Explainable AI project.
Edit these values to match your local dataset paths and training preferences.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "asl_alphabet_train")
TEST_DIR = os.path.join(DATA_DIR, "asl_alphabet_test")
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
GRADCAM_DIR = os.path.join(RESULTS_DIR, "gradcam_outputs")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(GRADCAM_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
# 26 letters (A-Z) + SPACE, DELETE, NOTHING = 29 classes
CLASS_NAMES = [chr(i) for i in range(65, 91)] + ["SPACE", "DELETE", "NOTHING"]
NUM_CLASSES = len(CLASS_NAMES)

IMG_SIZE = (200, 200)      # Custom CNN input size used in the report
IMG_SIZE_TRANSFER = (224, 224)  # Standard input size for ResNet/DenseNet/EfficientNet
CHANNELS = 3

TRAIN_SPLIT = 0.75
VAL_SPLIT = 0.15
TEST_SPLIT = 0.10

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 5

# Which backbone to train: "custom_cnn", "resnet50", "densenet121", "efficientnetb0"
MODEL_NAME = "custom_cnn"

# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------
# Name of the last convolutional layer to target for Grad-CAM per architecture.
# Leave as None to auto-detect the last Conv2D layer.
GRADCAM_LAYER_NAMES = {
    "custom_cnn": "conv2d_last",
    "resnet50": "conv5_block3_out",
    "densenet121": "conv5_block16_concat",
    "efficientnetb0": "top_conv",
}

RANDOM_SEED = 42
