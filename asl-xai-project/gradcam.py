"""
Grad-CAM (Gradient-weighted Class Activation Mapping) implementation.

This is the core Explainable AI (XAI) component of the project. It
highlights the regions of an input ASL gesture image that most influenced
the model's prediction, producing the heatmap visualizations described in
Chapter 4.2 / 6.2 of the project report.

Usage:
    python gradcam.py --image path/to/hand_sign.jpg --model saved_models/custom_cnn_best.h5
"""

import argparse
import os

import cv2
import numpy as np
import tensorflow as tf
import matplotlib.cm as cm

import config
from data_preprocessing import preprocess_single_image


def find_last_conv_layer(model: tf.keras.Model) -> str:
    """Auto-detects the name of the last Conv2D layer in the model, walking
    into nested (functional) sub-models such as a pretrained backbone."""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
        if hasattr(layer, "layers"):  # nested model, e.g. ResNet50 backbone
            for sub_layer in reversed(layer.layers):
                if isinstance(sub_layer, tf.keras.layers.Conv2D):
                    return sub_layer.name
    raise ValueError("Could not find a Conv2D layer in the given model.")


def make_gradcam_heatmap(img_array: np.ndarray, model: tf.keras.Model,
                          last_conv_layer_name: str = None, pred_index: int = None):
    """
    Computes the Grad-CAM heatmap for a single preprocessed image array.

    Args:
        img_array: preprocessed image, shape (H, W, C), values in [0, 1]
        model: trained keras model
        last_conv_layer_name: name of the target conv layer (auto-detected if None)
        pred_index: class index to explain (defaults to the predicted class)

    Returns:
        heatmap: 2D numpy array normalized to [0, 1]
        pred_index: the class index that was explained
        preds: the full prediction probability vector
    """
    if last_conv_layer_name is None:
        last_conv_layer_name = find_last_conv_layer(model)

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output],
    )

    inputs = np.expand_dims(img_array, axis=0)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(inputs)
        if pred_index is None:
            pred_index = int(tf.argmax(predictions[0]))
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), pred_index, predictions.numpy()[0]


def overlay_heatmap(image_path: str, heatmap: np.ndarray, alpha: float = 0.4):
    """Overlays a Grad-CAM heatmap on top of the original image and returns
    a BGR numpy array ready to save with cv2.imwrite."""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap_uint8]
    jet_heatmap = np.uint8(jet_heatmap * 255)

    superimposed = jet_heatmap * alpha + img
    superimposed = np.uint8(np.clip(superimposed, 0, 255))
    return cv2.cvtColor(superimposed, cv2.COLOR_RGB2BGR)


def explain_image(image_path: str, model_path: str, output_path: str = None,
                   img_size: tuple = config.IMG_SIZE):
    model = tf.keras.models.load_model(model_path)
    img_array = preprocess_single_image(image_path, img_size=img_size)

    layer_name = config.GRADCAM_LAYER_NAMES.get(config.MODEL_NAME)
    heatmap, pred_index, preds = make_gradcam_heatmap(img_array, model, layer_name)

    predicted_class = config.CLASS_NAMES[pred_index]
    confidence = float(preds[pred_index])
    print(f"Predicted class: {predicted_class}  (confidence={confidence:.4f})")

    result_img = overlay_heatmap(image_path, heatmap)

    if output_path is None:
        base = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(config.GRADCAM_DIR, f"{base}_gradcam.jpg")

    cv2.imwrite(output_path, result_img)
    print(f"Grad-CAM visualization saved to {output_path}")
    return output_path, predicted_class, confidence


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a Grad-CAM heatmap for one image.")
    parser.add_argument("--image", required=True, help="Path to the input ASL gesture image.")
    parser.add_argument("--model", required=True, help="Path to a trained .h5 model file.")
    parser.add_argument("--output", default=None, help="Where to save the overlay image.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    explain_image(args.image, args.model, args.output)
