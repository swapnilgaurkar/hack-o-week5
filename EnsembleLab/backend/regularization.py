from models import make_model
from preprocessing import prepare_data
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score

def run_regularization_experiment(df,target,params):
    pre,Xtr,Xte,ytr,yte,target=prepare_data(df,target)
    strength=(params or {}).get("strength",{})
    alpha=float(strength.get("l1",1.0)); lam=float(strength.get("l2",2.0))
    specs=[("No Regularization",make_model("None")),("L1",make_model("L1",alpha=alpha)),("L2",make_model("L2",lam=lam))]
    out={}
    for name,m in specs:
        pipe=Pipeline([("preprocess",pre),("model",m)])
        pipe.fit(Xtr,ytr); pred=pipe.predict(Xte); proba=pipe.predict_proba(Xte)[:,1]
        tr=accuracy_score(ytr,pipe.predict(Xtr)); te=accuracy_score(yte,pred)
        out[name]={"train_accuracy":round(float(tr),4),"test_accuracy":round(float(te),4),
                   "f1":round(float(f1_score(yte,pred,zero_division=0)),4),
                   "roc_auc":round(float(roc_auc_score(yte,proba)),4),"gap":round(float(tr-te),4)}
    return out,{"target":target,"l1":alpha,"l2":lam}
