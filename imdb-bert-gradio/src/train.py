"""
Fine-tune BERT (bert-base-uncased) for binary sentiment classification
on the IMDB movie reviews dataset.

Usage:
    python src/train.py --train-size 2000 --test-size 250 --epochs 2
"""

import argparse
import random

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

SEED = 42
MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 256


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
    }


def main():
    parser = argparse.ArgumentParser(description="Fine-tune BERT on IMDB sentiment")
    parser.add_argument("--train-size", type=int, default=2000)
    parser.add_argument("--test-size", type=int, default=250)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--output-dir", default="./bert_imdb_output")
    parser.add_argument("--save-dir", default="./imdb_bert")
    args = parser.parse_args()

    set_seed()

    print("Loading dataset (stanfordnlp/imdb) ...")
    dataset = load_dataset("stanfordnlp/imdb")

    small_train = dataset["train"].shuffle(seed=SEED).select(range(args.train_size))
    small_test = dataset["test"].shuffle(seed=SEED).select(range(args.test_size))

    split = small_train.train_test_split(test_size=0.1, seed=SEED)
    train_ds = split["train"]
    val_ds = split["test"]

    print(f"Train size: {len(train_ds)} | Val size: {len(val_ds)} | Test size: {len(small_test)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(batch["text"], max_length=MAX_LENGTH, truncation=True)

    train_tokenized = train_ds.map(tokenize, batched=True)
    val_tokenized = val_ds.map(tokenize, batched=True)
    test_tokenized = small_test.map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=SEED,
        logging_steps=50,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    final_results = trainer.evaluate(eval_dataset=test_tokenized)
    print("Final test results:")
    for k, v in final_results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")

    trainer.save_model(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)
    print(f"Saved model and tokenizer to: {args.save_dir}")


if __name__ == "__main__":
    main()
