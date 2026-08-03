# Enhanced American Sign Language Recognition using Deep Learning and Explainable AI

A deep learning system for recognizing American Sign Language (ASL) hand gestures, paired with **Grad-CAM** based Explainable AI (XAI) so predictions can be visually understood and trusted. This project was built to help improve the reliability, transparency, and accessibility of ASL recognition technology for the Deaf and Hard of Hearing (DHH) community.

> Final year B.Tech project — Computer Science and Engineering, Amrita School of Computing, Bengaluru.

## Overview

Existing ASL recognition systems often behave as "black boxes" — accurate, but unable to explain *why* they classified a gesture a certain way. This project trains and compares several CNN-based architectures on the ASL alphabet, then applies **Grad-CAM** to generate heatmaps showing exactly which regions of the hand the model focused on for each prediction.

**Dataset:** 87,000 annotated images across 29 classes (A–Z, plus `SPACE`, `DELETE`, `NOTHING`).

**Models compared:**
| Model | Notes |
|---|---|
| Custom CNN | Best overall precision/recall/F1/accuracy (~0.99) and sharpest Grad-CAM focus |
| ResNet50 | Residual connections, deep feature extraction via transfer learning |
| DenseNet121 | Dense connectivity, strong feature reuse, crisp activation maps |
| EfficientNetB0 | Compound-scaled, efficient for lighter-weight deployment |
| HRNet | High-resolution representations, evaluated for comparison |

## Key Features

- 🖐️ Static ASL alphabet classification (29 classes)
- 🔥 Grad-CAM heatmap generation for every prediction (image or live webcam)
- 📊 Full evaluation suite — accuracy, precision, recall, F1-score, confusion matrix
- 🎥 Real-time webcam inference with a toggleable explainability overlay
- 🔁 Swappable backbones (`custom_cnn`, `resnet50`, `densenet121`, `efficientnetb0`)

## Project Structure

```
asl-xai-project/
├── config.py                 # Central configuration (paths, hyperparameters, classes)
├── data_preprocessing.py      # Resizing, normalization, augmentation, data generators
├── model.py                   # Custom CNN + transfer-learning model builders
├── train.py                   # Training loop with early stopping & checkpoints
├── evaluate.py                # Test-set evaluation: accuracy/precision/recall/F1 + confusion matrix
├── gradcam.py                 # Grad-CAM heatmap generation for a single image
├── realtime_inference.py      # Live webcam recognition + Grad-CAM overlay
├── requirements.txt
├── saved_models/               # Trained .h5 model weights land here
├── results/
│   └── gradcam_outputs/        # Saved Grad-CAM visualizations
└── sample_data/                 # Optional small sample images for a quick test
```

## Getting Started

### 1. Clone and install dependencies

```bash
git clone https://github.com/<your-username>/asl-xai-project.git
cd asl-xai-project
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Get the dataset

This project expects the [ASL Alphabet dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) (or an equivalent 29-class ASL dataset) arranged as:

```
data/
├── asl_alphabet_train/
│   ├── A/  B/  C/  ...  Z/  SPACE/  DELETE/  NOTHING/
└── asl_alphabet_test/
    ├── A/  B/  C/  ...
```

Download it and place it under `data/` (this folder is git-ignored due to size).

### 3. Train a model

```bash
python train.py --model custom_cnn --epochs 30 --batch-size 32
```

Swap `--model` for `resnet50`, `densenet121`, or `efficientnetb0` to train the transfer-learning variants.

### 4. Evaluate on the test set

```bash
python evaluate.py --model saved_models/custom_cnn_best.h5
```

Outputs accuracy, precision, recall, F1-score, a classification report, and a confusion matrix image saved to `results/`.

### 5. Generate a Grad-CAM explanation for one image

```bash
python gradcam.py --image sample_data/example_a.jpg --model saved_models/custom_cnn_best.h5
```

This saves a heatmap overlay to `results/gradcam_outputs/` showing which part of the hand drove the prediction.

### 6. Real-time webcam demo

```bash
python realtime_inference.py --model saved_models/custom_cnn_best.h5
```

Place your hand inside the green ROI box. Press `g` to toggle the live Grad-CAM heatmap, `q` to quit.

## Methodology Summary

1. **Preprocessing:** resize → normalize to [0,1] → augment (rotation, shift, zoom, brightness) → one-hot encode labels → train/val/test split.
2. **Model architecture:** stacked Conv2D + ReLU + MaxPooling blocks with batch normalization and dropout, ending in a softmax classifier (or a frozen pretrained backbone + classification head for the transfer-learning models).
3. **Training:** categorical cross-entropy loss, Adam optimizer, early stopping, and checkpointing on validation accuracy.
4. **Explainability:** Grad-CAM computes gradients of the predicted class with respect to the final convolutional layer's feature maps, producing a heatmap that highlights the most influential hand regions.
5. **Evaluation:** precision, recall, F1-score, and accuracy on a held-out test set, cross-checked qualitatively against Grad-CAM outputs.

## Results

The custom CNN achieved the strongest quantitative performance (precision/recall/F1/accuracy ≈ 0.99) and produced the sharpest, most localized Grad-CAM heatmaps — consistently focusing on finger position and hand shape rather than background clutter, which qualitatively confirms the model is learning the right features for the task.

## Future Work

- Extend from static letters to continuous, sentence-level sign language using RNNs/LSTMs/Transformers
- Real-time mobile/edge deployment via lightweight models (e.g., MobileNetV3) and quantization
- Multilingual sign language support (e.g., Indian Sign Language, Arabic Sign Language)

## License

This project is released under the [MIT License](LICENSE).

## Acknowledgements

Developed as a final-year B.Tech project at the Amrita School of Computing, Bengaluru, Amrita Vishwa Vidyapeetham, under faculty guidance in the Department of Computer Science and Engineering.
