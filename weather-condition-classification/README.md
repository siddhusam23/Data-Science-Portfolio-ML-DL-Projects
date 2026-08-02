# Weather Condition Classification (Szeged, Hungary 2006–2016)

Predicting the daily weather condition (`Daily Summary`, e.g. *"Partly cloudy
throughout the day."*) from hourly sensor readings — temperature, humidity,
wind, pressure, and visibility — using and comparing five classification
models, implemented in both **scikit-learn** and **PySpark MLlib**.

Siddhesh T S

## Problem Statement

Weather forecasting from raw sensor telemetry is a classic multi-class
classification problem: given a snapshot of atmospheric conditions, predict
the categorical description of the day's weather. This project builds and
benchmarks several classifiers to see which best captures the relationship
between numeric weather measurements and the resulting weather summary.

## Dataset

[**Weather in Szeged 2006–2016**](https://www.kaggle.com/datasets/budincsevity/szeged-weather)
— hourly weather observations for Szeged, Hungary, spanning 2006–2016.
Columns include:

- `Formatted Date` — timestamp (dropped before modeling)
- `Summary`, `Precip Type` — short categorical weather tags
- `Temperature (C)`, `Apparent Temperature (C)`, `Humidity`
- `Wind Speed (km/h)`, `Wind Bearing (degrees)`
- `Visibility (km)`, `Loud Cover`, `Pressure (millibars)`
- `Daily Summary` — target label, a longer categorical description of the day

The CSV is not included in this repo (large file) — download it from Kaggle
as `weatherHistory.csv` and place it in `data/`.

## Approach

1. **Data cleaning** — drop rows with nulls, remove duplicates, drop the
   `Formatted Date` column.
2. **Encoding & scaling** — label-encode the categorical columns
   (`Summary`, `Precip Type`); standard-scale features for
   scale-sensitive models, min-max scale for Naive Bayes (which requires
   non-negative inputs).
3. **Modeling** — train and compare five classifiers predicting
   `Daily Summary`:
   - Logistic Regression
   - Random Forest
   - Decision Tree
   - Multi-Layer Perceptron (MLP)
   - Naive Bayes (Multinomial)
4. **Evaluation** — weighted Accuracy, Precision, Recall, and F1 (weighted
   averaging accounts for the multi-class, imbalanced label distribution).

The original notebook (`notebooks/weather_classification.ipynb`) implements
every model twice: once with **PySpark MLlib** (`Pipeline` + `StringIndexer`
+ `VectorAssembler` + `StandardScaler`/`MinMaxScaler`) and once with
**scikit-learn**, so you can compare a distributed-computing approach
against a single-machine one on the same data. `src/train_models.py`
distills the scikit-learn side into a single reusable script.

## Repository Structure

```
.
├── data/                                # place weatherHistory.csv here (not tracked)
├── notebooks/
│   └── weather_classification.ipynb     # original notebook: EDA + PySpark & sklearn models
├── src/
│   └── train_models.py                  # end-to-end script: cleaning, encoding, 5-model comparison
├── images/                               # generated plots + model_comparison.csv
├── requirements.txt
└── README.md
```

## Getting Started

```bash
# 1. Clone the repo
git clone https://github.com/siddhusam23/weather-condition-classification.git
cd weather-condition-classification

# 2. Install dependencies
pip install -r requirements.txt
# (pyspark is optional — only needed for the PySpark cells in the notebook)

# 3. Add the dataset
#    Download weatherHistory.csv from Kaggle and place it in data/

# 4. Run the training script
python src/train_models.py --data data/weatherHistory.csv
```

This prints Accuracy/Precision/Recall/F1 for all five models and saves a
correlation heatmap and comparison chart to `images/`.

Alternatively, open `notebooks/weather_classification.ipynb` to step through
the full exploratory analysis and both the PySpark and scikit-learn
implementations interactively. Note: the notebook was originally written for
Google Colab and mounts Google Drive to load the CSV — swap those cells for
a local `pd.read_csv('data/weatherHistory.csv')` if running elsewhere.

## Tech Stack

Python · pandas · scikit-learn · PySpark MLlib · matplotlib · seaborn

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE).
