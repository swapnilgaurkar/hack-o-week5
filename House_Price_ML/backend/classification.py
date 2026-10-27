from pathlib import Path
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA = Path(__file__).resolve().parent.parent / "data" / "house_prices.csv"
FEATURES = ["area","bedrooms","bathrooms","age","distance"]

def train_classification_models():
    df=pd.read_csv(DATA)
    X,y=df[FEATURES],df["category"]
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
    models={
        "Logistic Regression":Pipeline([("scale",StandardScaler()),("model",LogisticRegression(max_iter=2000))]),
        "KNN":Pipeline([("scale",StandardScaler()),("model",KNeighborsClassifier(n_neighbors=5))])
    }
    metrics={}
    for name,m in models.items():
        m.fit(Xtr,ytr); p=m.predict(Xte)
        metrics[name]={"Accuracy":round(float(accuracy_score(yte,p)),3)}
    return models,metrics

def predict_classification(models,features):
    row=pd.DataFrame([features],columns=FEATURES)
    return {name:str(m.predict(row)[0]) for name,m in models.items()}
