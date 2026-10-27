# Loan Approval Model Evaluation

A Hack-o-week Week 9–10 project for 5th Semester CSE.

## Topics Covered

### Week 9 – Model Evaluation
- Train/test split
- 5-fold cross-validation
- Confusion matrix
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

### Week 10 – Feature Engineering & Scaling
- Missing-value handling using SimpleImputer
- Categorical encoding using OneHotEncoder
- Numerical feature scaling using StandardScaler
- Feature preprocessing using ColumnTransformer
- Loan approval prediction using trained models

## Models

1. Logistic Regression
2. Random Forest Classifier

## Project Structure

```text
Loan_Approval_Model_Evaluation/
├── backend/
│   ├── app.py
│   ├── preprocessing.py
│   ├── models.py
│   └── evaluation.py
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
├── data/
│   └── loan_data.csv
├── requirements.txt
├── README.md
└── .gitignore
```

## Run on Mac / Linux

Open Terminal:

```bash
cd Loan_Approval_Model_Evaluation
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cd backend
python3 app.py
```

Then open:

```text
frontend/index.html
```

in your browser.

The Flask API runs at:

```text
http://127.0.0.1:5000
```

## Evaluation

The dashboard displays Accuracy, Precision, Recall, F1 Score, ROC-AUC, 5-Fold Cross-Validation and the Confusion Matrix.

## Note

The included CSV is a small educational dataset designed for demonstrating the ML workflow. It is not intended for real-world financial lending decisions.
