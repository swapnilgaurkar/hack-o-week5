from flask import Flask, jsonify, request
from flask_cors import CORS
from preprocessing import load_and_preprocess_data
from models import train_models
from evaluation import evaluate_model

app = Flask(__name__)
CORS(app)

DATA_PATH = "../data/loan_data.csv"

X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_data(DATA_PATH)
models = train_models(X_train, y_train)

@app.route("/api/health")
def health():
    return jsonify({"status": "Backend is running"})

@app.route("/api/evaluation")
def evaluation():
    results = {}
    for name, model in models.items():
        predictions = model.predict(X_test)
        results[name] = evaluate_model(model, X_test, y_test, predictions)
    return jsonify(results)

@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json()
    required = ["Gender", "Married", "Dependents", "Education", "Self_Employed",
                "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
                "Loan_Amount_Term", "Credit_History", "Property_Area"]

    missing = [field for field in required if field not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

    # Reuse preprocessing pipeline by constructing a one-row DataFrame.
    import pandas as pd
    from preprocessing import preprocess_new_data
    row = pd.DataFrame([data])
    X_new = preprocess_new_data(row)

    model = models["Random Forest"]
    prediction = int(model.predict(X_new)[0])
    probability = float(model.predict_proba(X_new)[0][1])

    return jsonify({
        "prediction": "Approved" if prediction == 1 else "Rejected",
        "probability": round(probability * 100, 2)
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
