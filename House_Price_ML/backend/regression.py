from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

DATA = Path(__file__).resolve().parent.parent / "data" / "house_prices.csv"
FEATURES = ["area","bedrooms","bathrooms","age","distance"]

def train_regression_models():
    df = pd.read_csv(DATA)
    X, y = df[FEATURES], df["price_lakhs"]
    Xtr, Xte, ytr, yte = train_test_split(X,y,test_size=.2,random_state=42)
    models = {
        "Linear Regression": LinearRegression(),
        "Polynomial Regression": Pipeline([("poly",PolynomialFeatures(2,include_bias=False)),("model",LinearRegression())]),
        "Ridge Regression": Pipeline([("scale",StandardScaler()),("model",Ridge(alpha=1.0))]),
        "Lasso Regression": Pipeline([("scale",StandardScaler()),("model",Lasso(alpha=.05,max_iter=10000))])
    }
    metrics={}
    for name,m in models.items():
        m.fit(Xtr,ytr); p=m.predict(Xte)
        metrics[name]={"MAE":round(float(mean_absolute_error(yte,p)),3),
                       "RMSE":round(float(np.sqrt(mean_squared_error(yte,p))),3),
                       "R2":round(float(r2_score(yte,p)),3)}
    return models,metrics

def predict_regression(models, features):
    row=pd.DataFrame([features],columns=FEATURES)
    out={name:round(float(m.predict(row)[0]),2) for name,m in models.items()}
    out["Average Prediction"]=round(sum(out.values())/len(out),2)
    return out
