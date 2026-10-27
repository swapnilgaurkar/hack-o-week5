import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold,cross_val_score
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
from preprocessing import prepare_data

def _num(d,k,default):
    try:return int(float(d.get(k,default)))
    except:return default
def _float(d,k,default):
    try:return float(d.get(k,default))
    except:return default

def train_all_models(df,target,params):
    pre,Xtr,Xte,ytr,yte,target=prepare_data(df,target)
    p=params or {}
    rf=p.get("random_forest",{}); xg=p.get("xgboost",{}); lg=p.get("lightgbm",{}); dt=p.get("decision_tree",{})
    models={
      "Decision Tree":DecisionTreeClassifier(max_depth=_num(dt,"max_depth",5),random_state=42),
      "Random Forest":RandomForestClassifier(n_estimators=_num(rf,"n_estimators",120),max_depth=_num(rf,"max_depth",8),random_state=42,n_jobs=-1),
      "XGBoost":XGBClassifier(n_estimators=_num(xg,"n_estimators",120),max_depth=_num(xg,"max_depth",5),learning_rate=_float(xg,"learning_rate",.08),reg_alpha=_float(xg,"reg_alpha",0),reg_lambda=_float(xg,"reg_lambda",1),random_state=42,n_jobs=1,eval_metric="logloss"),
      "LightGBM":LGBMClassifier(n_estimators=_num(lg,"n_estimators",120),max_depth=_num(lg,"max_depth",6),learning_rate=_float(lg,"learning_rate",.08),lambda_l1=_float(lg,"lambda_l1",0),lambda_l2=_float(lg,"lambda_l2",1),random_state=42,verbosity=-1)
    }
    min_class=int(ytr.value_counts().min())
    folds=min(5,min_class)
    if folds<2: raise ValueError("Not enough samples per class for cross-validation.")
    cv=StratifiedKFold(n_splits=folds,shuffle=True,random_state=42)
    out={}
    for name,m in models.items():
        pipe=Pipeline([("preprocess",pre),("model",m)])
        pipe.fit(Xtr,ytr)
        predtr=pipe.predict(Xtr); pred=pipe.predict(Xte)
        proba=pipe.predict_proba(Xte)[:,1]
        cvscores=cross_val_score(pipe,Xtr,ytr,cv=cv,scoring="accuracy",n_jobs=1)
        train_acc=accuracy_score(ytr,predtr); test_acc=accuracy_score(yte,pred)
        out[name]={
          "train_accuracy":round(float(train_acc),4),"test_accuracy":round(float(test_acc),4),
          "precision":round(float(precision_score(yte,pred,zero_division=0)),4),
          "recall":round(float(recall_score(yte,pred,zero_division=0)),4),
          "f1":round(float(f1_score(yte,pred,zero_division=0)),4),
          "roc_auc":round(float(roc_auc_score(yte,proba)),4),
          "cv_mean":round(float(cvscores.mean()),4),"cv_std":round(float(cvscores.std()),4),
          "gap":round(float(train_acc-test_acc),4)
        }
    return out,{"target":target,"train_rows":len(Xtr),"test_rows":len(Xte),"classes":[0,1]}

def make_model(kind, alpha=0, lam=0):
    if kind=="L1":
        return XGBClassifier(n_estimators=120,max_depth=5,learning_rate=.08,reg_alpha=alpha,reg_lambda=1,random_state=42,n_jobs=1,eval_metric="logloss")
    if kind=="L2":
        return XGBClassifier(n_estimators=120,max_depth=5,learning_rate=.08,reg_alpha=0,reg_lambda=lam,random_state=42,n_jobs=1,eval_metric="logloss")
    return XGBClassifier(n_estimators=120,max_depth=5,learning_rate=.08,reg_alpha=0,reg_lambda=1,random_state=42,n_jobs=1,eval_metric="logloss")
