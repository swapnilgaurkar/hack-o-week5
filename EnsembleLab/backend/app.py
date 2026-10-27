from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os, traceback
import pandas as pd
from preprocessing import prepare_data, get_dataset_info
from models import train_all_models
from regularization import run_regularization_experiment
from evaluation import analyze_results, bias_variance_payload

app=Flask(__name__, static_folder="../frontend")
CORS(app)

BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA=os.path.join(BASE,"data","sample_data.csv")

@app.get("/")
def home():
    return send_from_directory(os.path.join(BASE,"frontend"),"index.html")

def load_df():
    if "file" in request.files:
        f=request.files["file"]
        if not f.filename.lower().endswith(".csv"): raise ValueError("Please upload a CSV file.")
        return pd.read_csv(f)
    return pd.read_csv(DEFAULT_DATA)

@app.post("/api/upload")
def upload():
    try:
        df=load_df()
        return jsonify({"success":True,"info":get_dataset_info(df)})
    except Exception as e:
        return jsonify({"success":False,"error":str(e)}),400

@app.post("/api/train")
def train():
    try:
        payload=request.get_json(silent=True) or {}
        df=load_df() if "file" not in request.files else load_df()
        target=payload.get("target")
        params=payload.get("params",{})
        results, meta=train_all_models(df,target,params)
        return jsonify({"success":True,"results":results,"meta":meta})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success":False,"error":str(e)}),400

@app.post("/api/compare")
def compare():
    try:
        payload=request.get_json(silent=True) or {}
        df=load_df()
        results,meta=train_all_models(df,payload.get("target"),payload.get("params",{}))
        return jsonify({"success":True,"results":results,"meta":meta})
    except Exception as e:
        return jsonify({"success":False,"error":str(e)}),400

@app.post("/api/bias-variance")
def bv():
    try:
        payload=request.get_json(silent=True) or {}
        df=load_df()
        results,meta=train_all_models(df,payload.get("target"),payload.get("params",{}))
        return jsonify({"success":True,"analysis":bias_variance_payload(results)})
    except Exception as e:
        return jsonify({"success":False,"error":str(e)}),400

@app.post("/api/regularization")
def regularization():
    try:
        payload=request.get_json(silent=True) or {}
        df=load_df()
        out,meta=run_regularization_experiment(df,payload.get("target"),payload.get("params",{}))
        return jsonify({"success":True,"results":out,"meta":meta})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success":False,"error":str(e)}),400

if __name__=="__main__":
    app.run(host="127.0.0.1",port=5001,debug=False)
