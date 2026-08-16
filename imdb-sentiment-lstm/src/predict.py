"""
Run sentiment predictions using a trained IMDB LSTM model.

Usage:
    python src/predict.py --review "This movie was fantastic. I loved it."
"""

import argparse
import pickle

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_SEQUENCE_LEN = 200


def load_artifacts(model_path: str, tokenizer_path: str):
    model = load_model(model_path)
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer


def predict_sentiment(review: str, model, tokenizer) -> str:
    sequence = tokenizer.texts_to_sequences([review])
    padded_sequence = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LEN)
    prediction = model.predict(padded_sequence)
    return "positive" if prediction[0][0] > 0.5 else "negative"


def main():
    parser = argparse.ArgumentParser(description="Predict sentiment of a movie review")
    parser.add_argument("--review", required=True, help="Review text to classify")
    parser.add_argument("--model", default="model/sentiment_lstm.h5")
    parser.add_argument("--tokenizer", default="model/tokenizer.pkl")
    args = parser.parse_args()

    model, tokenizer = load_artifacts(args.model, args.tokenizer)
    sentiment = predict_sentiment(args.review, model, tokenizer)
    print(f"Review: {args.review}")
    print(f"Predicted sentiment: {sentiment}")


if __name__ == "__main__":
    main()
