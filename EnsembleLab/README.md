# EnsembleLab — Ensemble Learning, Bias–Variance & Regularization Analyzer

EnsembleLab is a beginner-friendly Hack-o-week Week 13–14 machine-learning dashboard for exploring ensemble learning, model evaluation, empirical bias–variance behavior, overfitting/underfitting, and L1/L2 regularization.

## Features
- CSV upload with target selection
- Automatic numeric imputation and categorical one-hot encoding
- Leakage-safe train/test preprocessing pipeline
- Decision Tree baseline
- Random Forest (Bagging)
- XGBoost and LightGBM (Boosting)
- Accuracy, precision, recall, F1, ROC-AUC
- Stratified cross-validation mean and standard deviation
- Generalization-gap diagnostics
- Training vs testing Plotly charts
- Empirical bias/variance heuristic
- L1/L2 regularization experiment
- Responsive dark analytics dashboard
- Reproducible random seeds and graceful API errors

## Technologies
Python, Flask, Flask-CORS, pandas, NumPy, scikit-learn, XGBoost, LightGBM, HTML, CSS, JavaScript and Plotly.js.

## Folder Structure
```text
EnsembleLab/
├── backend/
│   ├── app.py
│   ├── preprocessing.py
│   ├── models.py
│   ├── evaluation.py
│   └── regularization.py
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
├── data/
│   └── sample_data.csv
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation
Python 3.10+ is recommended because current XGBoost/LightGBM wheels vary by platform.

```bash
cd EnsembleLab
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
python3 backend/app.py
```
Open `http://127.0.0.1:5001` in your browser. The frontend is served by Flask, so no separate web server is required.

## API Endpoints
- `GET /` — dashboard
- `POST /api/upload` — inspect an uploaded CSV or return the bundled sample dataset
- `POST /api/train` — train all four models
- `POST /api/compare` — train and return comparison results
- `POST /api/bias-variance` — return empirical training/testing analysis
- `POST /api/regularization` — compare no/L1/L2 regularization

## Concepts
**Bagging:** models learn from bootstrap samples and their outputs are combined. Random Forest is the main example.

**Boosting:** learners are trained sequentially, with later learners focusing on earlier mistakes.

**XGBoost:** gradient boosting with useful controls including `reg_alpha` and `reg_lambda`.

**LightGBM:** an efficient gradient-boosting framework with `lambda_l1` and `lambda_l2`.

**Bias:** a model is too simple to capture the underlying pattern.

**Variance:** a model changes substantially with training data; a large train/test gap can be a warning sign.

**Overfitting:** training performance is high while testing performance is substantially lower.

**Underfitting:** both training and testing performance are low.

**L1 regularization:** penalizes the absolute magnitude of model parameters; in XGBoost it is controlled by `reg_alpha`.

**L2 regularization:** penalizes squared magnitude; in XGBoost it is controlled by `reg_lambda`.

## Evaluation Metrics
- Training and testing accuracy
- Precision
- Recall
- F1 score
- ROC-AUC
- Cross-validation mean and standard deviation
- Generalization gap = training accuracy − testing accuracy

The fit labels are intentionally heuristic: a large positive gap suggests possible overfitting, while low training and testing scores suggest possible underfitting. The dashboard does **not** claim this gap is an exact mathematical measurement of bias and variance.

## Expected Workflow
1. Load the bundled student-performance dataset or upload a CSV.
2. Select the binary target.
3. Configure model parameters.
4. Train the Decision Tree, Random Forest, XGBoost and LightGBM models.
5. Compare metrics and inspect the training/testing chart.
6. Review fit diagnostics and the empirical bias–variance section.
7. Run the L1/L2 regularization experiment.

## Sample Dataset
`data/sample_data.csv` contains 420 student-performance records with Study_Hours, Attendance, Assignment_Score, Midterm_Score, Lab_Score, Previous_GPA, Sleep_Hours, Participation, Previous_Backlogs and binary `Pass` (0 = Fail, 1 = Pass).

## Future Improvements
- Confusion-matrix and ROC curve tabs
- Feature-importance explanations
- Hyperparameter search
- Model export/import
- More ensemble algorithms
- Larger benchmark datasets

## Viva Questions and Answers
**What is ensemble learning?** Combining multiple models to improve robustness or predictive performance.

**What is bagging?** Training models on bootstrap samples and combining their predictions.

**What is boosting?** Sequentially adding learners that focus on errors made by earlier learners.

**Why Random Forest?** It reduces the variance of individual decision trees through randomized bagging.

**What is XGBoost?** A gradient-boosting library with efficient tree training and regularization.

**What is LightGBM?** A fast gradient-boosting framework designed for efficient tree learning.

**What is overfitting?** A model fits training data very well but generalizes poorly to unseen data.

**What is underfitting?** A model is too simple and performs poorly even on training data.

**What is L1 regularization?** A penalty based on absolute parameter magnitude.

**What is L2 regularization?** A penalty based on squared parameter magnitude.

**Why use a pipeline?** It keeps preprocessing inside each training fold and helps prevent test-data leakage.

**Why stratify?** It preserves class proportions between training and testing splits.

**Why ROC-AUC?** It summarizes how well predicted scores separate the two classes across thresholds.
