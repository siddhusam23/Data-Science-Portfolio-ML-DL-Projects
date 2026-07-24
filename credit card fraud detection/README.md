# Credit Card Fraud Detection

## Overview
This notebook performs exploratory data analysis, feature engineering, and trains multiple machine learning models to detect **fraudulent credit card transactions**, along with model interpretability analysis using LIME, SHAP, and ELI5.

## Dataset
- **Source file:** `fraudTrain.csv` (loaded from Google Drive)
- Key raw columns include:
  - `trans_date_trans_time`, `dob` — used to derive time-based and demographic features
  - `category`, `gender`, `amt`, `city_pop`, `lat`, `long` — transactional/customer features
  - `is_fraud` — binary target label

## Workflow
1. **Setup** — mounts Google Drive, installs `seaborn` and `dmba`.
2. **Data cleaning & feature engineering**
   - Drops an unnamed index column
   - Converts `trans_date_trans_time` and `dob` to datetime
   - Derives `trans_hour`, `trans_day_of_week`, and customer `age`
   - Drops raw name/date columns no longer needed (`first`, `last`, `dob`, `trans_date_trans_time`)
   - Casts categorical columns (`category`, `gender`, etc.) to `category` dtype
   - Bins `age` into custom `age_category` groups (Teenagers, Young Adults, Adults, Middle-aged, Seniors, Retired)
3. **Exploratory analysis**
   - Class imbalance check (`% fraudulent transactions`)
   - Transaction amount distribution
   - Transactions over time (`year_month` trend)
   - Fraud rate by merchant `category`
4. **Feature preparation**
   - Selects relevant predictors (`amt`, `gender`, `lat`, `long`, `city_pop`, `trans_hour`, `trans_day_of_week`, `age`, `age_category`)
   - One-hot encodes categorical predictors
   - Applies **SMOTE** oversampling and **RandomUnderSampler** undersampling to address class imbalance
5. **Modeling** — trains and compares several classifiers (each typically run on both original/oversampled and undersampled data):
   - **XGBoost Classifier**
   - **Logistic Regression**
   - **Decision Tree Classifier**
   - **Random Forest Classifier**
   - **Gaussian Naive Bayes**
6. **Evaluation** — accuracy score, classification report, and confusion matrix (visualized with seaborn/`ConfusionMatrixDisplay`) for each model/sampling strategy combination.
7. **Model interpretability**
   - **LIME** for local explanation of individual predictions
   - **SHAP** for feature contribution analysis
   - **ELI5** with `PermutationImportance` for global feature importance

## Requirements
```
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn (imblearn)
xgboost
dmba
lime
shap
eli5
```

## How to Run
1. Place `fraudTrain.csv` in the working directory referenced in the notebook (update the Google Drive path if running locally instead of on Colab).
2. Install the required packages (see above).
3. Run cells sequentially from top to bottom — the interpretability sections (LIME/SHAP/ELI5) depend on a trained model from earlier cells, so preserve the execution order.

## Notes
- The notebook compares both **oversampling (SMOTE)** and **undersampling (RandomUnderSampler)** strategies to handle the severe class imbalance typical of fraud detection datasets.
- Interpretability tools (LIME, SHAP, ELI5) are included to explain individual predictions and global feature importance, which is valuable for understanding *why* a transaction was flagged as fraudulent.
