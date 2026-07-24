# IoT Botnet Traffic Classification

## Overview
This notebook trains multiple classifiers to detect and categorize **botnet attacks on IoT devices**, using a curated feature set derived from IoT network traffic captures.

## Dataset
- **Source file:** `features_having_most_influence_on_Botnet_IoT.csv`
- Loaded from Google Drive (`/content/drive/MyDrive/...`) — indicates the notebook was built for Google Colab.
- Key columns:
  - `pkSeqID`, `saddr`, `daddr`, `sport`, `dport`, `proto` — flow identifiers (some dropped before modeling)
  - `subcategory`, `attack` — auxiliary attack labels (dropped before modeling)
  - `category` — the target label used for classification

## Workflow
1. **Setup** — installs `category_encoders`, mounts Google Drive, loads the CSV.
2. **Data cleaning**
   - Drops identifier/auxiliary columns: `pkSeqID`, `subcategory`, `attack`, `dport`, `sport`
3. **Feature encoding**
   - `saddr` and `daddr` (source/destination IP addresses) encoded via `BinaryEncoder`
   - `proto` (protocol) one-hot encoded via `pd.get_dummies`
4. **Feature scaling** — `StandardScaler` applied to all features.
5. **Modeling** — trains and evaluates several classifiers on the `category` target:
   - **Multinomial Logistic Regression**
   - **Artificial Neural Network** (Keras `Sequential` model: 16 → 12 → 5 dense layers, ReLU/softmax activations, trained for 100 epochs)
   - **Decision Tree Classifier** (tuned via `GridSearchCV`)
   - **Random Forest Classifier** (tuned via `GridSearchCV` over `max_depth` and `n_estimators`)
6. **Evaluation** — accuracy score, classification report (precision/recall/F1), and confusion matrix (visualized with seaborn heatmaps) for each model.

## Requirements
```
pandas
numpy
category_encoders
matplotlib
seaborn
scikit-learn
tensorflow / keras
```

## How to Run
1. Place `features_having_most_influence_on_Botnet_IoT.csv` in the working directory (update the Google Drive path if running locally instead of on Colab).
2. Install the required packages (see above).
3. Run cells sequentially. The Google Drive mount cell can be skipped if the CSV is already available locally — just point `pd.read_csv` to the correct local path.

## Notes
- Multiple classifiers are compared on the same encoded/scaled feature set, making it easy to benchmark classical ML (Logistic Regression, Decision Tree, Random Forest) against a simple deep learning model (ANN).
- The neural network uses a 5-class softmax output, indicating `category` has 5 distinct classes.
