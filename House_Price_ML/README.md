# Smart House Price Predictor

Hack-o-week Week 7 & 8 mini-project for 5th Semester.

## Algorithms
### Regression
- Linear Regression
- Polynomial Regression
- Ridge Regression
- Lasso Regression

### Classification
- Logistic Regression
- KNN

## Inputs
Area, bedrooms, bathrooms, house age, and distance from city center.

## Outputs
Regression models predict price in ₹ Lakhs. Classification models predict Affordable, Moderate, or Premium.

## Evaluation
Regression: MAE, RMSE, R². Classification: Accuracy.

## Run
```bash
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
python3 backend/app.py
```
Then open `frontend/index.html`.

The included CSV is a synthetic educational dataset, not real property data.

## Structure
```text
House_Price_ML/
├── backend/
├── frontend/
├── data/
├── requirements.txt
└── README.md
```
