# IMDB Reviews — Sentiment Analysis with LSTM

A binary sentiment classifier for movie reviews, trained on the [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) using an LSTM network built with TensorFlow/Keras.

## Overview

The model tokenizes raw review text, pads sequences to a fixed length, and feeds them through an embedding layer followed by an LSTM to predict whether a review is **positive** or **negative**.

**Pipeline:**
1. Load and clean the 50K-review dataset
2. Tokenize text (top 5,000 words) and pad sequences to length 200
3. Train an Embedding → LSTM → Dense(sigmoid) network
4. Evaluate on a held-out test split
5. Serve predictions on new review text

## Model Architecture

| Layer | Details |
|---|---|
| Embedding | vocab size 5,000, output dim 128, input length 200 |
| LSTM | 128 units, dropout 0.2, recurrent dropout 0.2 |
| Dense | 1 unit, sigmoid activation |

Compiled with Adam optimizer and binary cross-entropy loss.

## Repository Structure

```
imdb-sentiment-lstm/
├── data/
│   └── IMDB_Dataset.csv       # not committed — see Data section
├── model/                     # trained model + tokenizer (generated)
├── src/
│   ├── train.py                # training pipeline
│   └── predict.py               # inference on new reviews
├── requirements.txt
├── .gitignore
└── README.md
```

## Data

This project uses the [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) (50,000 reviews, evenly split positive/negative). Download it from Kaggle and place `IMDB_Dataset.csv` in the `data/` folder, or point `--data` at its location.

```bash
kaggle datasets download -d lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
unzip imdb-dataset-of-50k-movie-reviews.zip -d data/
```

## Setup

```bash
git clone https://github.com/<your-username>/imdb-sentiment-lstm.git
cd imdb-sentiment-lstm
pip install -r requirements.txt
```

## Usage

**Train the model:**
```bash
python src/train.py --data data/IMDB_Dataset.csv --epochs 5 --batch-size 64
```
This saves the trained model to `model/sentiment_lstm.h5` and the fitted tokenizer to `model/tokenizer.pkl`.

**Predict on a new review:**
```bash
python src/predict.py --review "This movie was fantastic. I loved it."
```

## Results

| Metric | Value |
|---|---|
| Test Accuracy | ~85–88% (5 epochs) |
| Test Loss | ~0.30–0.35 |

*(Exact numbers depend on the random split and training run.)*

## Tech Stack

- Python
- TensorFlow / Keras
- pandas, scikit-learn

## License

Released under the [MIT License](LICENSE).
