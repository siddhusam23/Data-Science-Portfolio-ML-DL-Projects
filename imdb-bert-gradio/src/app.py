"""
Gradio web UI for the fine-tuned BERT IMDB sentiment classifier.

Usage:
    python src/app.py
"""

import argparse

import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ID2LABEL = {0: "negative", 1: "positive"}
MAX_LENGTH = 256

EXAMPLES = [
    ["This movie was absolutely fantastic. Brilliant acting and a gripping story."],
    ["I hated this film. It was boring, too long, and badly written."],
    ["It was okay - some good moments, but overall forgettable."],
]


def load_artifacts(model_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    print(f"Model and tokenizer loaded on: {device}")
    return tokenizer, model, device


def build_classify_fn(tokenizer, model, device):
    def classify_review(review: str) -> str:
        if not review.strip():
            return "Please enter a review."

        inputs = tokenizer(
            review,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
        ).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]

        pred_id = torch.argmax(probs).item()
        label = ID2LABEL[pred_id]
        confidence = probs[pred_id].item()
        emoji = "✅" if label == "positive" else "❌"

        return f"{emoji} **{label.upper()}** (confidence: {confidence * 100:.2f}%)"

    return classify_review


def build_demo(model_dir: str) -> gr.Interface:
    tokenizer, model, device = load_artifacts(model_dir)
    classify_review = build_classify_fn(tokenizer, model, device)

    return gr.Interface(
        fn=classify_review,
        inputs=gr.Textbox(
            lines=5,
            placeholder="Type a movie review here...",
            label="Movie Review",
        ),
        outputs=gr.Markdown(label="Prediction"),
        title="IMDB Sentiment Classifier",
        description="Fine-tuned BERT predicting whether a movie review is positive or negative.",
        flagging_mode="never",
        examples=EXAMPLES,
    )


def main():
    parser = argparse.ArgumentParser(description="Launch the IMDB sentiment Gradio app")
    parser.add_argument("--model-dir", default="./imdb_bert")
    parser.add_argument("--share", action="store_true", help="Create a public shareable link")
    args = parser.parse_args()

    demo = build_demo(args.model_dir)
    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
