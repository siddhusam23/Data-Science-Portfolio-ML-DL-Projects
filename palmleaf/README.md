# Malayalam Palm Leaf Character Recognition using Gabor Features

Recognizes Malayalam palm-leaf manuscript characters from Gabor-filter
features, comparing ten classifiers — Logistic Regression, SVM, Random
Forest, Decision Tree, KNN, MLP, Naive Bayes, XGBoost, CatBoost, and
AdaBoost — before and after `GridSearchCV` hyperparameter tuning, then
explains the best model's predictions with **LIME**
(Local Interpretable Model-agnostic Explanations).

Reproduces the methodology from:

> Siddhesh T. S., P. Revanth Krishna Varma, P. Gowtham, Achyuta Siva
> Sai Kowshik, Ambati Sai Sindhur, Annem Gnaneswara Reddy, Remya Sivan,
> Peeta Basa Pati, "Malayalam Palm Leaf Character Recognition using
> Gabor Features".

## Repository structure

```
.
├── src/
│   ├── gabor_feature_extraction.py    # Build the Gabor-feature CSV from character images
│   └── malayalam_char_recognition.py  # 10-classifier comparison + GridSearchCV + LIME
├── data/                              # Datasets go here (not tracked in git, see data/README.md)
├── results/                           # Output CSVs / LIME explanation HTML land here
├── requirements.txt
├── .gitignore
└── LICENSE
```

## Setup

```bash
git clone https://github.com/<your-username>/<this-repo>.git
cd <this-repo>
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 1. Get a dataset

The paper's exact "Malayalam Char Gabor" feature file isn't publicly
released. See [`data/README.md`](data/README.md) for open character-image
datasets (including one from the same institution, Amrita_MalCharDb,
and a dedicated palm-leaf manuscript dataset, HMPLMD) and instructions
for extracting Gabor features from them yourself — or point the
pipeline at your own CSV of numeric features + a label column.

## 2. Extract Gabor features (if starting from images)

```bash
python src/gabor_feature_extraction.py \
    --images-dir data/images \
    --output-csv data/malayalam_char_gabor.csv
```

Arrange images into one sub-folder per character class first (see
`data/README.md`). This produces 50 features per image (5 spatial
frequencies x 5 orientations x [mean, variance] of the filtered
response) plus a `label` column, matching the paper's 51-column,
48-class dataset layout.

## 3. Run the classifier comparison + LIME

```bash
python src/malayalam_char_recognition.py \
    --data-path data/malayalam_char_gabor.csv \
    --label-col label
```

Optional flags:
- `--skip-tuning` — only run the baseline (no `GridSearchCV`) pass, much faster
- `--skip-lime` — skip the LIME explanation step
- `--cv N` — number of cross-validation folds for `GridSearchCV` (default 3)
- `--lime-instance N` — which test-set row LIME explains (default 0)
- `--test-size F` — train/test split fraction (default 0.2)
- `--results-dir path` — where result CSVs / the LIME HTML report are written (default `results/`)

This prints and saves:
- `results/results_before_tuning.csv` — accuracy/precision/recall/F1 (train & test) for every model with default hyperparameters
- `results/results_after_tuning.csv` — the same, after `GridSearchCV`
- `results/lime_explanation.html` — an interactive LIME explanation for one test instance, using the best-performing model

## Methodology

Min-max normalization of the 50 Gabor features → label-encode the 48
character classes → stratified train/test split → train all ten
classifiers with default hyperparameters → re-train with
`GridSearchCV`-tuned hyperparameters → rank by test accuracy → apply
LIME to the best classifier to surface which Gabor features drive
individual predictions.

`XG-Boost` and `CAT-Boost` are optional — the script skips them
automatically if `xgboost` / `catboost` aren't installed.

## Results (from the paper)

**Without hyperparameter tuning** — Random Forest had the highest test
accuracy (86.75%) and F1-score (82.73%), followed by AdaBoost
(85.95% accuracy, 81.68% F1).

**With hyperparameter tuning** — Random Forest remained best
(86.69% test accuracy, 82.74% F1-score), followed by KNN and Decision
Tree (both 86.26% accuracy).

**LIME interpretability** — for the best model, one feature (referred
to as "feature 9" in the paper) contributed up to 49% probability
weight to individual predictions, with a second feature ("feature 44")
contributing up to 36%, indicating these Gabor responses are the most
influential for character discrimination.

Your own numbers will depend on which image dataset you extract
features from, the Gabor filter bank configuration, and random seeds.

## Citation

```bibtex
@inproceedings{siddhesh_palmleaf_gabor,
  title={Malayalam Palm Leaf Character Recognition using Gabor Features},
  author={Siddhesh, T. S. and Revanth Krishna Varma, P. and Gowtham, P. and
          Kowshik, Achyuta Siva Sai and Sindhur, Ambati Sai and
          Reddy, Annem Gnaneswara and Sivan, Remya and Pati, Peeta Basa},
  organization={Amrita School of Computing, Amrita Vishwa Vidyapeetham}
}
```

## License

MIT — see `LICENSE`.
