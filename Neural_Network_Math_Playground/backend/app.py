from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from vectors import vector_operations
from matrices import matrix_operations
from gradients import derivative_example, gradient_example
from neural_network import backprop_demo

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND = os.path.join(ROOT, "frontend")

app = Flask(__name__, static_folder=FRONTEND, static_url_path="")
CORS(app)

@app.get("/")
def home():
    return send_from_directory(FRONTEND, "index.html")

@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "message": "Neural Network Math Playground is running!"})

@app.post("/api/vectors")
def vectors():
    data = request.get_json()
    return jsonify(vector_operations(data["a"], data["b"]))

@app.post("/api/matrices")
def matrices():
    data = request.get_json()
    return jsonify(matrix_operations(data["A"], data["B"], data["vector"]))

@app.post("/api/derivative")
def derivative():
    data = request.get_json()
    return jsonify(derivative_example(float(data["x"])))

@app.post("/api/gradient")
def gradient():
    data = request.get_json()
    return jsonify(gradient_example(float(data["x"]), float(data["y"])))

@app.post("/api/backprop")
def backprop():
    data = request.get_json()
    return jsonify(backprop_demo(
        float(data["x"]),
        float(data["w1"]),
        float(data["w2"]),
        float(data["target"]),
        float(data["learning_rate"])
    ))

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
