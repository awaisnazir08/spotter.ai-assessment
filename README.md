# Freight Rate Prediction

Predicting the posted rate of a freight load from its distance, equipment, weight, route and the market
conditions on the day it was posted.

## Setup

```bash
python -m pip install -r requirements.txt
```

## Run

`predict.py` trains the model and writes both prediction files:

```bash
python predict.py
python score.py --predictions validation_predictions.csv --december-predictions december-chart-inputs.csv
```

The scorer checks both files and creates `scorer_results/candidate_december.png`.

The full analysis is in `freight rate prediction model.ipynb`.

`predict.py` saves the trained model to `model/model.pkl`. That file is gitignored because the trees are
grown to full depth and it comes out around 1.3 GB so it is rebuilt by running the script instead.

## Approach

**Validation is two months in the future.** The training data covers January to October 2025 and the
validation data covers November and December, with no overlap. A random split would let the model see the
future and give a score that looks good but is not real, so every split in this project is made by date.

The models were compared on four rolling splits, each one training on the older months and testing on the
newer ones.

## Data quality issues found

1. **About 1.4 percent of the rates are wrong.** The log of rate per mile shows three separate distributions
   with empty gaps between them, which real prices would not do. Removing these rows from training brought
   the error down from 3.78 to 2.18 percent, which was the single biggest improvement in the project.
2. **Some weights are negative.** The absolute values match the positive weights, so the minus sign is a
   data entry mistake and we take the absolute value.
3. **Missing weight and market index values** are filled with the median.
4. **Eight cities appear in validation that are not in training**, so the model uses latitude and longitude
   instead of city names.

## Features

Rate per mile is predicted instead of the rate itself, because distance has a 0.91 correlation with the rate
and hides every other signal. The prediction is multiplied back by distance at the end.

The date is never used as a feature. The whole time signal sits in `market_index`, which is given for the
validation rows, so the model reads the market conditions of the day instead of extrapolating a trend.

## Model

Extra Trees, 200 trees. It had the lowest error on all four splits.

| Model | MAPE |
|---|---|
| Extra Trees | 2.75 |
| Gradient Boosting | 2.88 |
| Random Forest | 2.95 |
| Decision Tree | 3.24 |
| Linear Regression | 5.51 |

MAPE is used because the rates run from 57 dollars to 25,533 dollars, so a 150 dollar error means very different things on a
small load and a large one.
