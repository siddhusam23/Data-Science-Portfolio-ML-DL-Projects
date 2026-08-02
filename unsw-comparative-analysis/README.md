# UNSW-NB15 Comparative Analysis — Network Anomaly Detection

Compares **Isolation Forest** against **Random Forest**, **Decision
Tree**, **K-Means**, and **Logistic Regression** for network intrusion
/ anomaly detection on the UNSW-NB15 dataset.

Reproduces the methodology from:

> Siddhesh T.S., Shinu M. Rajagopal, Sreebha Bhaskaran, "Comparative
> Analysis of Machine Learning Algorithms for Anomaly Detection", 2024
> IEEE 9th International Conference for Convergence in Technology
> (I2CT).

## Files

| File | Description |
|---|---|
| `unsw_nb15_anomaly_detection.py` | Clean, CLI-driven script — preprocessing + all five models + results table |
| `UNSW_NB15.ipynb` | Original exploratory notebook it was refactored from |
| `data/` | Put the dataset CSV here (not tracked in git) |

## Dataset

UNSW-NB15 (Moustafa & Slay, 2015):

- Official source: https://research.unsw.edu.au/projects/unsw-nb15-dataset
- Kaggle mirror: https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15

Download one of the four raw CSV parts (`UNSW-NB15_1.csv` ...
`UNSW-NB15_4.csv`, ~700K rows each, no header row — the script attaches
the official 49-column header for you) and save it as:

```
data/UNSW-NB15_1.csv
```

Via the Kaggle CLI:

```bash
pip install kaggle
kaggle datasets download -d mrwellsdavid/unsw-nb15 -p data --unzip
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python unsw_nb15_anomaly_detection.py \
    --data-path data/UNSW-NB15_1.csv \
    --contamination 0.05
```

Optional flags:
- `--nrows N` — read only the first N rows for a quick test run
- `--output-csv path.csv` — where to save the comparison table (default `results/unsw_nb15_results.csv`)

## Methodology

Null removal → categorical label-encoding → numeric standardization →
correlation-based feature pruning (of any pair with >75% correlation,
one column is dropped) → binary `Label` target (normal vs. attack) →
train/evaluate all five models, reporting accuracy, precision, recall,
F1-score, and training/testing time.

## Results (from the paper, Table I)

| Model | Accuracy | Precision | Recall | F1-score | Train (s) | Test (s) |
|---|---|---|---|---|---|---|
| Isolation Forest | 96.64% | 98.34% | 96.64% | 97.27% | 6.00 | 1.80 |
| K-Means | 50.40% | 95.69% | 50.40% | 65.23% | 5.04 | 0.026 |
| Decision Tree | 99.05% | 99.34% | 99.05% | 99.13% | 0.25 | 0.018 |
| Random Forest | 99.98% | 99.98% | 99.98% | 99.98% | 13.76 | 0.24 |
| Logistic Regression | 97.51% | 96.23% | 97.51% | 96.77% | 9.39 | 6.46 |

Your own numbers will vary with the exact data slice, random seeds,
and library versions.

## Citation

```bibtex
@inproceedings{siddhesh2024comparative,
  title={Comparative Analysis of Machine Learning Algorithms for Anomaly Detection},
  author={Siddhesh, T. S. and Rajagopal, Shinu M. and Bhaskaran, Sreebha},
  booktitle={2024 IEEE 9th International Conference for Convergence in Technology (I2CT)},
  year={2024},
  organization={IEEE}
}
```

## License

MIT — see `LICENSE`.
