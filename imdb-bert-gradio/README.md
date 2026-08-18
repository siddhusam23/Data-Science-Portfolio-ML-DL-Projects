# IMDB Sentiment Classifier — BERT + Gradio

A fine-tuned BERT model for binary sentiment classification of movie reviews, with an interactive Gradio web UI for live predictions.

## Overview

This project fine-tunes `bert-base-uncased` on the [Stanford IMDB dataset](https://huggingface.co/datasets/stanfordnlp/imdb) (25K train / 25K test movie reviews) using Hugging Face `transformers` and `datasets`, then serves the model through a Gradio interface for real-time sentiment predictions.

**Pipeline:**
1. Load the IMDB dataset via Hugging Face `datasets`
2. Tokenize reviews with the BERT tokenizer (max length 256)
3. Fine-tune `bert-base-uncased` for sequence classification using the Hugging Face `Trainer`
4. Evaluate on accuracy and F1
5. Save the fine-tuned model and tokenizer
6. Serve predictions through a Gradio web app

## Repository Structure

```
imdb-bert-gradio/
├── src/
│   ├── train.py      # fine-tuning pipeline
│   ├── predict.py     # CLI sentiment prediction
│   └── app.py          # Gradio web UI
├── requirements.txt
├── .gitignore
└── README.md
```

*(The fine-tuned model is saved to `./imdb_bert/` at training time and is not committed — see `.gitignore`.)*

## Setup

```bash
git clone https://github.com/<your-username>/imdb-bert-gradio.git
cd imdb-bert-gradio
pip install -r requirements.txt
```

## Usage

**Fine-tune the model:**
```bash
python src/train.py --train-size 2000 --test-size 250 --epochs 2
```
This saves the fine-tuned model and tokenizer to `./imdb_bert/`.

**Predict from the command line:**
```bash
python src/predict.py --review "This movie was fantastic. I loved it."
```

**Launch the Gradio app:**
```bash
python src/app.py
```
Add `--share` to generate a temporary public link.

## Model Details

| | |
|---|---|
| Base model | `bert-base-uncased` |
| Task | Binary sequence classification (positive / negative) |
| Max sequence length | 256 tokens |
| Optimizer | AdamW, learning rate 2e-5 |
| Epochs | 2 |
| Batch size | 8 |

## Results

On a 2,000-example training subset (2 epochs):

| Epoch | Training Loss | Validation Loss | Accuracy | F1 |
|---|---|---|---|---|
| 1 | 0.394 | 0.304 | 0.870 | 0.857 |
| 2 | 0.357 | 0.409 | 0.895 | 0.889 |

*(Results scale up with more training data / epochs — the full 25K training set is available for a full run.)*

## Tech Stack

- Python
- PyTorch
- Hugging Face `transformers` and `datasets`
- scikit-learn (metrics)
- Gradio (web UI)

## License

Released under the [MIT License](LICENSE).
