"""
Real-time webcam ASL recognition with a live Grad-CAM overlay toggle.

Controls:
    q - quit
    g - toggle Grad-CAM heatmap overlay

Usage:
    python realtime_inference.py --model saved_models/custom_cnn_best.h5
"""

import argparse

import cv2
import numpy as np
import tensorflow as tf

import config
from gradcam import make_gradcam_heatmap, find_last_conv_layer

# Region Of Interest box where the user places their hand
ROI_TOP, ROI_BOTTOM, ROI_LEFT, ROI_RIGHT = 100, 400, 350, 650


def preprocess_frame(roi_frame, img_size):
    resized = cv2.resize(roi_frame, img_size)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype("float32") / 255.0
    return normalized


def overlay_heatmap_on_frame(frame, heatmap, alpha=0.4):
    heatmap_resized = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    return cv2.addWeighted(colored, alpha, frame, 1 - alpha, 0)


def run(model_path: str, img_size: tuple = config.IMG_SIZE, camera_index: int = 0):
    model = tf.keras.models.load_model(model_path)
    layer_name = config.GRADCAM_LAYER_NAMES.get(config.MODEL_NAME) or find_last_conv_layer(model)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Try a different --camera-index.")

    show_gradcam = False
    print("Press 'g' to toggle Grad-CAM overlay, 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        roi = frame[ROI_TOP:ROI_BOTTOM, ROI_LEFT:ROI_RIGHT]

        processed = preprocess_frame(roi, img_size)
        input_batch = np.expand_dims(processed, axis=0)
        preds = model.predict(input_batch, verbose=0)[0]
        pred_index = int(np.argmax(preds))
        label = config.CLASS_NAMES[pred_index]
        confidence = float(preds[pred_index])

        display_roi = roi.copy()
        if show_gradcam:
            heatmap, _, _ = make_gradcam_heatmap(processed, model, layer_name, pred_index)
            display_roi = overlay_heatmap_on_frame(display_roi, heatmap)

        frame[ROI_TOP:ROI_BOTTOM, ROI_LEFT:ROI_RIGHT] = display_roi
        cv2.rectangle(frame, (ROI_LEFT, ROI_TOP), (ROI_RIGHT, ROI_BOTTOM), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} ({confidence:.2f})", (ROI_LEFT, ROI_TOP - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("ASL Recognition (Explainable AI)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("g"):
            show_gradcam = not show_gradcam

    cap.release()
    cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time webcam ASL recognition.")
    parser.add_argument("--model", required=True, help="Path to a trained .h5 model file.")
    parser.add_argument("--camera-index", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.model, camera_index=args.camera_index)
