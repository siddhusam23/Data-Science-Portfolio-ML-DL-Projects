# UNSW-NB15 Network Intrusion Detection

## Overview
This notebook builds and compares several machine learning models to detect network intrusions using the **UNSW-NB15** dataset, a widely used benchmark dataset for network intrusion detection research containing both normal traffic and a variety of modern attack types.

## Dataset
- **Source file:** `UNSW-NB15_1.csv`
- **Format:** Raw CSV with no header row; column names are defined manually in the notebook (49 columns), including:
  - Flow identifiers: `srcip`, `sport`, `dstip`, `dsport`, `proto`, `state`
  - Traffic statistics: `dur`, `sbytes`, `dbytes`, `sttl`, `dttl`, `Sload`, `Dload`, `Spkts`, `Dpkts`, etc.
  - Connection/behavioral counters: `ct_state_ttl`, `ct_srv_src`, `ct_srv_dst`, `ct_dst_ltm`, etc.
  - Labels: `attack_cat` (attack category) and `Label` (binary: normal/attack)

## Workflow
1. **Data loading** — reads the raw CSV and applies the manual header.
2. **Data cleaning**
   - Replaces `-` placeholders with `NaN`
   - Drops the `attack_cat` column (multi-class label) for binary classification workflows
   - Drops rows with missing values and duplicate rows
3. **Exploratory analysis**
   - `nunique()`, `dtypes`, `describe()`, and `mode()` summaries
   - Boxplots for categorical/numerical attributes (`srcip`, `sport`, etc.)
   - Correlation heatmap of all features
4. **Preprocessing**
   - Label encoding of categorical columns
   - Feature scaling with `StandardScaler`
   - (Commented-out) feature selection experiments using `SelectKBest` (chi-squared) and `ExtraTreesClassifier` importance
   - (Commented-out) class balancing experiments using `TomekLinks` and `RandomUnderSampler`
   - Correlated-feature detection/removal helper function
5. **Modeling** — trains and evaluates multiple algorithms on the cleaned/reduced feature set:
   - Isolation Forest (with `GridSearchCV` tuning over `contamination`)
   - Decision Tree Classifier (with `GridSearchCV` tuning over `max_depth`)
   - Random Forest Classifier (with `GridSearchCV` tuning over `n_estimators`)
   - K-Means clustering (with `ParameterGrid` search and silhouette score)
   - One-Class SVM
   - Logistic Regression (with `ParameterGrid` search)
6. **Evaluation** — models are assessed with confusion matrices, accuracy, precision, recall, and F1-score.

## Requirements
```
pandas
numpy
scikit-learn
matplotlib
seaborn
imbalanced-learn (imblearn)
tensorflow (for optional Keras utilities)
```

## How to Run
1. Place `UNSW-NB15_1.csv` in the working directory referenced in the notebook (`/content/...` paths indicate it was originally run on Google Colab — update paths for local use).
2. Install the required packages (see above).
3. Run cells sequentially from top to bottom. Note some cells are wrapped in triple-quoted strings (commented out) and represent exploratory/alternative approaches — uncomment as needed.

## Notes
- Several cells are intentionally left as string-commented code blocks (feature selection, PCA, resampling experiments) — these are exploratory alternatives, not part of the main run path.
- The notebook was originally developed for Google Colab; file paths (`/content/...`) should be adjusted for local or other environments.
