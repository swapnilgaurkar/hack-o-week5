import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def get_dataset_info(df):
    return {"rows":int(len(df)),"features":int(max(0,len(df.columns)-1)),
            "columns":list(df.columns),"dtypes":{c:str(df[c].dtype) for c in df.columns},
            "missing":{c:int(df[c].isna().sum()) for c in df.columns},
            "preview":df.head(5).replace({np.nan:None}).to_dict(orient="records")}

def infer_target(df, target=None):
    if target and target in df.columns: return target
    for c in ["Pass","Target","target","Label","label","Outcome","outcome"]:
        if c in df.columns: return c
    return df.columns[-1]

def make_preprocessor(X):
    nums=X.select_dtypes(include=["number"]).columns.tolist()
    cats=[c for c in X.columns if c not in nums]
    num=Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler())])
    cat=Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num",num,nums),("cat",cat,cats)],remainder="drop")

def prepare_data(df,target=None):
    if df.empty or len(df)<8: raise ValueError("Dataset is too small. Provide at least 8 rows.")
    target=infer_target(df,target)
    if target not in df: raise ValueError("Selected target column was not found.")
    y=df[target]
    if y.nunique(dropna=True)!=2: raise ValueError("This project requires a binary classification target with exactly 2 classes.")
    if y.isna().any(): df=df.loc[~y.isna()].copy(); y=df[target]
    X=df.drop(columns=[target])
    y=pd.Series((y==y.dropna().unique()[1]).astype(int),index=y.index)
    if y.value_counts().min()<2: raise ValueError("Both target classes need at least 2 samples.")
    pre=make_preprocessor(X)
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
    return pre,Xtr,Xte,ytr,yte,target
