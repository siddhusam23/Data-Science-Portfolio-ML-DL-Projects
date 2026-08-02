# Credit Card Transaction Fraud Detection

Binary classification of fraudulent vs. legitimate credit card
transactions using engineered temporal/demographic features, class
imbalance handling (SMOTE oversampling and random undersampling), and a
comparison of **XGBoost**, **Logistic Regression**, **Decision Tree**,
**Random Forest**, and **Naive Bayes**.

## Files

| File | Description |
|---|---|
| `credit_card_fraud_detection.py` | Clean, CLI-driven script — feature engineering + resampling + all five models |
| `credit_card_fraud_detection.ipynb` | Original exploratory notebook it was refactored from |
| `data/` | Put the dataset CSV here (not tracked in git) |

## Dataset

"Credit Card Transactions Fraud Detection Dataset" (simulated
transaction data generated with the Sparkov tool, Jan 2019–Dec 2020),
on Kaggle:

- https://www.kaggle.com/datasets/kartik2112/fraud-detection

Download `fraudTrain.csv` and save it as:

```
data/fraudTrain.csv
```

Via the Kaggle CLI:

```bash
pip install kaggle
kaggle datasets download -d kartik2112/fraud-detection -p data --unzip
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python credit_card_fraud_detection.py \
    --data-path data/fraudTrain.csv \
    --resample undersample   # or: smote
```

Optional flags:
- `--output-csv path.csv` — where to save the comparison table (default `results/credit_card_fraud_results.csv`)

## Methodology

From the raw transaction timestamp, derive `trans_hour`,
`trans_day_of_week`, and cardholder `age` (from `dob`); bucket age into
life-stage categories (Teenagers / Young Adults / Adults / Middle-aged
/ Seniors / Retired); one-hot encode categoricals; split the data, then
correct the training split's class imbalance (fraud is a small minority
of transactions) with either random undersampling or SMOTE before
training all five classifiers.

## Notes

- `xgboost` is optional — if it isn't installed the script skips that
  model and continues with the rest.
- Accuracy alone is a poor metric on this heavily imbalanced problem;
  the script prints full classification reports (precision/recall/F1
  per class) and confusion matrices for each model, not just accuracy.

## Citation

This code is derived from an internal comparative study of ML
techniques for fraud detection; if you use it in published work, cite
the accompanying UNSW-NB15/Bot-IoT papers by the same author (see the
`unsw-comparative-analysis` and `botnet-attack-detection` projects) or
your own report.

## License

MIT — see `LICENSE`.
