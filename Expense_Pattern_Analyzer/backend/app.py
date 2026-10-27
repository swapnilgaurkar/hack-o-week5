"""Flask application: REST API + static frontend. Run: python3 -m backend.app"""
import calendar
import math
import os
import sqlite3
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from .database import CATEGORIES, DEFAULT_DB_PATH, Database
from .model_deployment import DEFAULT_MODEL_PATH, ModelService, risk_label
from .perceptron import demo as perceptron_demo
from .synthetic_data import FEATURES

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


def _num(v, name, errors, min_=None, max_=None, positive=False):
    """Parse a finite number (rejects bools/NaN/inf)."""
    if isinstance(v, bool) or v is None or v == "":
        errors.append(f"{name} is required and must be a number")
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        errors.append(f"{name} must be a number")
        return None
    if not math.isfinite(x):
        errors.append(f"{name} must be a finite number")
        return None
    if positive and x <= 0:
        errors.append(f"{name} must be greater than 0")
    elif min_ is not None and x < min_:
        errors.append(f"{name} must be at least {min_}")
    elif max_ is not None and x > max_:
        errors.append(f"{name} must be at most {max_}")
    return x


def validate_expense(data):
    errors = []
    if not isinstance(data, dict):
        return None, ["Request body must be a JSON object"]
    amount = _num(data.get("amount"), "amount", errors, positive=True, max_=1e9)
    d = data.get("date")
    try:
        d = datetime.strptime(str(d), "%Y-%m-%d").date().isoformat()
    except ValueError:
        errors.append("date must be a valid date in YYYY-MM-DD format")
    cat = data.get("category")
    if cat not in CATEGORIES:
        errors.append("category is required and must be one of the listed categories")
    desc = data.get("description") or ""
    if not isinstance(desc, str) or len(desc.strip()) > 200:
        errors.append("description must be text of at most 200 characters")
    if errors:
        return None, errors
    return {"date": d, "amount": round(amount, 2), "category": cat, "description": desc.strip()}, []


def compute_dashboard(db, today=None, month=None):
    today = today or date.today()
    if month:
        y, m = (int(p) for p in month.split("-"))
    else:
        y, m = today.year, today.month
    prefix = f"{y:04d}-{m:02d}"
    py, pm = (y, m - 1) if m > 1 else (y - 1, 12)
    prev_prefix = f"{py:04d}-{pm:02d}"
    rows = db.list_expenses(sort="date", order="desc")
    cur = [r for r in rows if r["date"].startswith(prefix)]
    prev_total = round(sum(r["amount"] for r in rows if r["date"].startswith(prev_prefix)), 2)
    total = round(sum(r["amount"] for r in cur), 2)
    budget = db.get_budget()
    cats = defaultdict(float)
    daily = defaultdict(float)
    for r in cur:
        cats[r["category"]] += r["amount"]
        daily[r["date"]] += r["amount"]
    category_totals = [{"category": c, "total": round(t, 2),
                        "percent": round(100 * t / total, 1) if total else 0.0}
                       for c, t in sorted(cats.items(), key=lambda kv: -kv[1])]
    days_in_month = calendar.monthrange(y, m)[1]
    daily_series = [{"date": f"{prefix}-{d:02d}", "total": round(daily.get(f"{prefix}-{d:02d}", 0.0), 2)}
                    for d in range(1, days_in_month + 1)]
    weekly = defaultdict(float)
    for r in cur:
        dd = datetime.strptime(r["date"], "%Y-%m-%d").date()
        weekly[(dd - timedelta(days=dd.weekday())).isoformat()] += r["amount"]
    in_month = (today.year, today.month) == (y, m)
    day_now = today.day if in_month else days_in_month
    last7 = sum(v for k, v in daily.items()
                if (today - timedelta(days=6)).isoformat() <= k <= today.isoformat()) if in_month else total / max(day_now, 1) * 7
    return {
        "month": prefix, "total_spent": total, "budget": budget,
        "remaining": round(budget - total, 2),
        "utilization_percent": round(100 * total / budget, 1) if budget > 0 else None,
        "transaction_count": len(cur),
        "category_totals": category_totals,
        "top_category": category_totals[0] if category_totals else None,
        "daily": daily_series,
        "weekly": [{"week_start": k, "total": round(v, 2)} for k, v in sorted(weekly.items())],
        "recent": rows[:5],
        "predictor_defaults": {
            "monthly_budget": budget, "spent_so_far": total,
            "days_remaining": max(days_in_month - day_now, 0),
            "avg_daily_spending": round(last7 / 7, 2),
            "previous_period_spending": prev_total},
    }


def create_app(db_path=DEFAULT_DB_PATH, model_path=DEFAULT_MODEL_PATH):
    app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
    db = Database(db_path)
    svc = ModelService(model_path)
    app.config.update(DB=db, MODEL=svc)

    def err(msg, code=400, details=None):
        body = {"error": msg}
        if details:
            body["details"] = details
        return jsonify(body), code

    def body_json():
        data = request.get_json(silent=True)
        return data if isinstance(data, dict) else None

    @app.errorhandler(404)
    def _404(e):
        return err("Not found", 404)

    @app.errorhandler(405)
    def _405(e):
        return err("Method not allowed", 405)

    @app.errorhandler(sqlite3.Error)
    def _db(e):
        return err("Database error", 500)

    @app.errorhandler(500)
    def _500(e):
        return err("Internal server error", 500)

    @app.get("/")
    def index():
        return send_from_directory(app.static_folder, "index.html")

    @app.get("/api/categories")
    def categories():
        return jsonify({"categories": CATEGORIES})

    @app.get("/api/expenses")
    def list_expenses():
        sort = request.args.get("sort", "date")
        order = request.args.get("order", "desc")
        if sort not in ("date", "amount") or order not in ("asc", "desc"):
            return err("sort must be date|amount and order asc|desc")
        rows = db.list_expenses(request.args.get("q", "").strip(), request.args.get("category", ""), sort, order)
        return jsonify({"expenses": rows, "count": len(rows)})

    @app.post("/api/expenses")
    def create_expense():
        clean, errors = validate_expense(body_json())
        if errors:
            return err("Invalid expense", 400, errors)
        return jsonify(db.add_expense(clean)), 201

    @app.put("/api/expenses/<int:eid>")
    def update_expense(eid):
        clean, errors = validate_expense(body_json())
        if errors:
            return err("Invalid expense", 400, errors)
        row = db.update_expense(eid, clean)
        return jsonify(row) if row else err("Expense not found", 404)

    @app.delete("/api/expenses/<int:eid>")
    def delete_expense(eid):
        return jsonify({"deleted": eid}) if db.delete_expense(eid) else err("Expense not found", 404)

    @app.get("/api/dashboard")
    def dashboard():
        month = request.args.get("month")
        if month:
            try:
                datetime.strptime(month, "%Y-%m")
            except ValueError:
                return err("month must be YYYY-MM")
        return jsonify(compute_dashboard(db, month=month))

    @app.get("/api/budget")
    def get_budget():
        return jsonify({"monthly_budget": db.get_budget()})

    @app.post("/api/budget")
    def set_budget():
        data, errors = body_json(), []
        if data is None:
            return err("Request body must be a JSON object")
        v = _num(data.get("monthly_budget"), "monthly_budget", errors, positive=True, max_=1e9)
        if errors:
            return err("Invalid budget", 400, errors)
        db.set_budget(round(v, 2))
        return jsonify({"monthly_budget": db.get_budget()})

    @app.post("/api/train")
    def train():
        data = body_json()
        if data is None:
            data = {}
        errors = []
        opt = str(data.get("optimizer", "adam")).lower()
        act = str(data.get("activation", "relu")).lower()
        if opt not in ("sgd", "adam"):
            errors.append("optimizer must be 'sgd' or 'adam'")
        if act not in ("relu", "tanh"):
            errors.append("activation must be 'relu' or 'tanh'")
        lr = _num(data.get("learning_rate", 0.01), "learning_rate", errors, min_=1e-5, max_=1.0)
        ep = _num(data.get("epochs", 100), "epochs", errors, min_=1, max_=1000)
        if ep is not None and ep != int(ep):
            errors.append("epochs must be a whole number")
        if errors:
            return err("Invalid training settings", 400, errors)
        try:
            meta = svc.train(optimizer=opt, activation=act, learning_rate=lr, epochs=int(ep))
        except FloatingPointError as e:
            return err(str(e), 422)
        return jsonify({"message": "Training finished", "model": meta})

    @app.post("/api/model/save")
    def save_model():
        try:
            svc.save()
        except RuntimeError as e:
            return err(str(e), 409)
        except OSError:
            return err("Could not write the model file", 500)
        return jsonify({"message": "Model saved to models/budget_model.joblib", "status": svc.status()})

    @app.post("/api/model/load")
    def load_model():
        try:
            svc.load()
        except FileNotFoundError as e:
            return err(str(e), 404)
        except Exception:
            return err("Saved model could not be loaded (file may be corrupt or from another version)", 422)
        return jsonify({"message": "Model loaded", "status": svc.status()})

    @app.post("/api/predict")
    def predict():
        data, errors = body_json(), []
        if data is None:
            return err("Request body must be a JSON object")
        f = {}
        rules = {"monthly_budget": dict(positive=True, max_=1e9),
                 "spent_so_far": dict(min_=0, max_=1e9),
                 "days_remaining": dict(min_=0, max_=31),
                 "avg_daily_spending": dict(min_=0, max_=1e9),
                 "previous_period_spending": dict(min_=0, max_=1e9)}
        for name in FEATURES:
            f[name] = _num(data.get(name), name, errors, **rules[name])
        if errors:
            return err("Invalid prediction input", 400, errors)
        try:
            p = svc.predict(f)
        except RuntimeError as e:
            return err(str(e), 409)
        naive = f["spent_so_far"] + f["avg_daily_spending"] * f["days_remaining"]
        return jsonify({
            "probability": p, "risk": risk_label(p), "inputs": f,
            "naive_projection": round(naive, 2),
            "disclaimer": "Educational demo: the model is trained on synthetic data and is not financial advice."})

    @app.get("/api/status")
    def status():
        return jsonify({"app": "Moneta", "expenses": len(db.list_expenses()),
                        "model": svc.status(), "training_data": "synthetic",
                        "perceptron_demo": perceptron_demo()})

    return app


def main():
    port = int(os.environ.get("PORT", "5000"))
    create_app().run(host="127.0.0.1", port=port, debug=False)


if __name__ == "__main__":
    main()
