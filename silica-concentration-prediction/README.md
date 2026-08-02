# Prediction of Silica Concentration in Iron Ore Mining

Predicting `% Silica Concentrate` in a froth flotation plant using regression
models, so that impurity levels in iron ore concentrate can be estimated
without waiting on slow lab analysis.

&#x20; Siddhesh T S
Amrita Vishwa Vidyapeetham, Bengaluru, India

## Problem Statement

Iron ore quality is largely determined by the amount of silica impurity left
after the froth flotation process. Traditional lab-based measurement of
silica content is accurate but slow — results can take over an hour to come
back, by which time process conditions have already moved on. This project
compares several regression models to see how well plant sensor readings
(pulp density, air flow, pH, feed composition, etc.) can predict silica
concentration in near real time.

## Dataset

[**Quality Prediction in a Mining Process**](https://www.kaggle.com/datasets/edumagalhaes/quality-prediction-in-a-mining-process)
(Eduardo Magalhães Oliveira, via Kaggle) — real sensor data from a Brazilian
iron ore flotation plant, collected every 20 seconds to hourly over a
6-month period (March–September 2017), with 24 columns including:

* `% Iron Feed`, `% Silica Feed` — incoming ore composition
* `Starch Flow`, `Amina Flow` — reagent dosing
* `Ore Pulp Flow`, `Ore Pulp pH`, `Ore Pulp Density`
* `Flotation Column 01–07 Air Flow` and `Level` — process controls
* `% Iron Concentrate`, `% Silica Concentrate` — output quality (targets)

The CSV is not included in this repo due to size — download it from Kaggle
and place it at `data/MiningProcess\_Flotation\_Plant\_Database.csv`.

## Approach

1. **EDA \& preprocessing** — inspect distributions, check for outliers and
noisy sensor readings, and visualize feature correlations with a heatmap.
2. **Modeling** — train and compare six regressors on the same train/test
split:

   * Linear Regression
   * Random Forest Regressor
   * K-Nearest Neighbors Regressor
   * Support Vector Regressor (SVR)
   * Polynomial Regression
   * Decision Tree Regressor
3. **Evaluation** — score each model with MAE, MSE, and RMSE, and visualize
predicted-vs-actual fit for each.

## Results

|Model|MAE|MSE|RMSE|
|-|-|-|-|
|**Random Forest**|**0.608**|**0.664**|**0.815**|
|Decision Tree|0.721|1.253|1.119|
|Polynomial Regression|0.748|0.940|0.969|
|SVR|0.804|1.215|1.102|
|KNN|0.794|1.073|1.036|
|Linear Regression|0.810|1.065|1.032|

**Random Forest Regressor** gave the lowest error across all three metrics.
Its ensemble of decision trees handles the non-linear relationships between
process variables and silica concentration better than the other models
tested. Full discussion, literature review, and figures are in
[`reports/Project\_Report.pdf`](reports/Project_Report.pdf).

## Repository Structure

```
.
├── data/                    # place MiningProcess\_Flotation\_Plant\_Database.csv here (not tracked)
├── notebooks/
│   └── silica\_prediction.ipynb   # original exploratory + modeling notebook
├── src/
│   └── train\_models.py           # end-to-end script: EDA plots + 6-model training/comparison
├── images/                  # generated plots (heatmap, regression plots, RMSE comparison)
├── reports/
│   └── Project\_Report.pdf        # full written project report
├── requirements.txt
└── README.md
```

## Getting Started

```bash
# 1. Clone the repo
git clone https://github.com/siddhusam23/silica-concentration-prediction.git
cd silica-concentration-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add the dataset
#    Download MiningProcess\_Flotation\_Plant\_Database.csv from Kaggle and
#    place it in the data/ folder.

# 4. Run the training script
python src/train\_models.py --data data/MiningProcess\_Flotation\_Plant\_Database.csv
```

This prints MAE/MSE/RMSE for all six models and saves comparison plots to
`images/`. Alternatively, open `notebooks/silica\_prediction.ipynb` to step
through the original exploratory analysis and Linear Regression / Random
Forest / KNN training interactively.

## Tech Stack

Python · pandas · NumPy · scikit-learn · matplotlib · seaborn

## References

Key related work is cited in full in the project report; notable sources
include studies on multi-target regression for mining quality prediction,
silicon content prediction in blast furnaces using GWO-SVR and ANN models,
and support vector regression approaches to hot metal silicon prediction.

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE).

