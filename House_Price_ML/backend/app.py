from flask import Flask, request, jsonify
from flask_cors import CORS
from regression import train_regression_models, predict_regression
from classification import train_classification_models, predict_classification

app = Flask(__name__)
CORS(app)

REG_MODELS, REG_METRICS = train_regression_models()
CLS_MODELS, CLS_METRICS = train_classification_models()

@app.get("/api/health")
def health():
    return jsonify({"status": "ok"})

@app.post("/api/predict")
def predict():
    try:
        d = request.get_json(force=True)
        features = {k: float(d[k]) for k in ["area","bedrooms","bathrooms","age","distance"]}
        return jsonify({
            "regression": predict_regression(REG_MODELS, features),
            "classification": predict_classification(CLS_MODELS, features),
            "metrics": {"regression": REG_METRICS, "classification": CLS_METRICS}
        })
    except (KeyError, TypeError, ValueError) as e:
        return jsonify({"error": str(e)}), 400

@app.get("/api/metrics")
def metrics():
    return jsonify({"regression": REG_METRICS, "classification": CLS_METRICS})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
