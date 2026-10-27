from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
from io import StringIO
import uuid

from preprocessing import prepare_dataframe, dataset_summary
from pca import run_pca
from tsne import run_tsne

app = Flask(__name__)
CORS(app)

# Educational project: uploaded datasets are kept in memory while Flask runs.
DATASETS = {}


@app.get("/")
def home():
    return jsonify({
        "project": "DimViz",
        "message": "PCA & t-SNE Dimensionality Reduction Visualizer API",
        "status": "running"
    })


@app.post("/api/upload")
def upload_csv():
    try:
        if "file" not in request.files:
            return jsonify({"error": "Please choose a CSV file to upload."}), 400

        file = request.files["file"]
        if not file.filename:
            return jsonify({"error": "No file was selected."}), 400

        if not file.filename.lower().endswith(".csv"):
            return jsonify({"error": "Please upload a CSV file."}), 400

        raw = file.read()
        if not raw:
            return jsonify({"error": "The uploaded CSV is empty."}), 400

        try:
            df = pd.read_csv(StringIO(raw.decode("utf-8-sig")))
        except UnicodeDecodeError:
            df = pd.read_csv(StringIO(raw.decode("latin-1")))

        if df.empty:
            return jsonify({"error": "The CSV contains no data rows."}), 400

        prepared = prepare_dataframe(df)
        if prepared["numeric_columns"] == []:
            return jsonify({
                "error": "No numerical columns were found. PCA and t-SNE need numerical features."
            }), 400

        dataset_id = str(uuid.uuid4())
        DATASETS[dataset_id] = df

        summary = dataset_summary(df)
        summary["dataset_id"] = dataset_id
        summary["filename"] = file.filename
        summary["numeric_columns"] = prepared["numeric_columns"]
        summary["categorical_columns"] = prepared["categorical_columns"]
        summary["preview"] = df.head(8).fillna("").to_dict(orient="records")

        return jsonify(summary)

    except Exception as exc:
        return jsonify({"error": f"Could not read the CSV: {exc}"}), 500


def get_dataset():
    data = request.get_json(silent=True) or {}
    dataset_id = data.get("dataset_id")
    if not dataset_id or dataset_id not in DATASETS:
        return None, jsonify({"error": "Dataset not found. Please upload the CSV again."}), 400
    return DATASETS[dataset_id], None, None


@app.post("/api/pca")
def pca_endpoint():
    df, error, code = get_dataset()
    if error:
        return error, code

    try:
        data = request.get_json(silent=True) or {}
        class_column = data.get("class_column")
        result = run_pca(df, class_column=class_column)
        return jsonify(result)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"PCA failed: {exc}"}), 500


@app.post("/api/tsne")
def tsne_endpoint():
    df, error, code = get_dataset()
    if error:
        return error, code

    try:
        data = request.get_json(silent=True) or {}
        class_column = data.get("class_column")
        perplexity = data.get("perplexity", 30)
        result = run_tsne(df, class_column=class_column, perplexity=perplexity)
        return jsonify(result)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"t-SNE failed: {exc}"}), 500


@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": "The uploaded file is too large."}), 413


if __name__ == "__main__":
    app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
    app.run(debug=True, port=5000)
