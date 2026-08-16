"""
Train an LSTM sentiment classifier on the IMDB 50K movie reviews dataset.

Usage:
    python src/train.py --data data/IMDB_Dataset.csv --epochs 5
"""

import argparse
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_VOCAB_SIZE = 5000
MAX_SEQUENCE_LEN = 200


def load_data(csv_path: str):
    data = pd.read_csv(csv_path)
    data.replace({"sentiment": {"positive": 1, "negative": 0}}, inplace=True)
    return data


def build_model():
    model = Sequential()
    model.add(Embedding(input_dim=MAX_VOCAB_SIZE, output_dim=128, input_length=MAX_SEQUENCE_LEN))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def main():
    parser = argparse.ArgumentParser(description="Train the IMDB LSTM sentiment model")
    parser.add_argument("--data", default="data/IMDB_Dataset.csv", help="Path to the IMDB dataset CSV")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--model-out", default="model/sentiment_lstm.h5")
    parser.add_argument("--tokenizer-out", default="model/tokenizer.pkl")
    args = parser.parse_args()

    print(f"Loading data from {args.data} ...")
    data = load_data(args.data)
    print(f"Dataset shape: {data.shape}")
    print(data["sentiment"].value_counts())

    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    print(f"Train: {train_data.shape} | Test: {test_data.shape}")

    tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
    tokenizer.fit_on_texts(train_data["review"])

    X_train = pad_sequences(tokenizer.texts_to_sequences(train_data["review"]), maxlen=MAX_SEQUENCE_LEN)
    X_test = pad_sequences(tokenizer.texts_to_sequences(test_data["review"]), maxlen=MAX_SEQUENCE_LEN)
    Y_train = train_data["sentiment"]
    Y_test = test_data["sentiment"]

    model = build_model()
    model.summary()

    model.fit(X_train, Y_train, epochs=args.epochs, batch_size=args.batch_size, validation_split=0.2)

    loss, accuracy = model.evaluate(X_test, Y_test)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    import os
    os.makedirs("model", exist_ok=True)
    model.save(args.model_out)
    with open(args.tokenizer_out, "wb") as f:
        pickle.dump(tokenizer, f)
    print(f"Saved model to {args.model_out} and tokenizer to {args.tokenizer_out}")


if __name__ == "__main__":
    main()
