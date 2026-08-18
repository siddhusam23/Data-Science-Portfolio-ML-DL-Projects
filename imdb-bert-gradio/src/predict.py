"""
Run sentiment predictions using a fine-tuned BERT model.

Usage:
    python src/predict.py --review "This movie was fantastic. I loved it."
"""

import argparse

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ID2LABEL = {0: "negative", 1: "positive"}
MAX_LENGTH = 256


def load_artifacts(model_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return tokenizer, model, device


def predict_sentiment(text: str, tokenizer, model, device):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]

    pred_id = torch.argmax(probs).item()
    return ID2LABEL[pred_id], probs[pred_id].item()


def main():
    parser = argparse.ArgumentParser(description="Predict sentiment of a movie review")
    parser.add_argument("--review", required=True, help="Review text to classify")
    parser.add_argument("--model-dir", default="./imdb_bert")
    args = parser.parse_args()

    tokenizer, model, device = load_artifacts(args.model_dir)
    label, confidence = predict_sentiment(args.review, tokenizer, model, device)

    print(f"Review: {args.review}")
    print(f"Predicted sentiment: {label} (confidence: {confidence * 100:.2f}%)")


if __name__ == "__main__":
    main()
