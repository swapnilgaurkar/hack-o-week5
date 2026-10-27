# Moneta — Expense Pattern Analyzer

A personal expense tracker with spending analytics and a **neural network built from scratch in NumPy** that estimates the chance of exceeding a monthly budget. Built for the Week 15 & 16 Hack-o-week assignment.

## Features
- Expense manager: add, edit, delete, search, filter by category, sort by date/amount (SQLite, survives restarts)
- Overview dashboard: month total, budget, remaining (negative when overspent), transaction count, utilization bar, charts, recent transactions
- Insights: category donut, weekly trend vs budget, top category, per-category totals and percentages
- AI budget predictor: probability of overrun + low/moderate/high label
- Model Lab: choose SGD/Adam, ReLU/Tanh, learning rate, epochs; train, save, load; loss curve; accuracy/precision/recall/F1 (only shown from real runs); plain-language explanations

> Screenshots: add your own after running the app (e.g. `docs/overview.png`, `docs/model-lab.png`). None are bundled because none could be generated in the build environment.

## Tech stack
Python 3.11/3.12 · Flask · NumPy (neural network) · SQLite · Joblib · HTML/CSS/vanilla JS · Chart.js (loaded at runtime). No TensorFlow, PyTorch, scikit-learn, React or API keys.

## Folder structure
```
Expense_Pattern_Analyzer/
├── backend/
│   ├── app.py               Flask app, validation, REST API, dashboard maths
│   ├── database.py          SQLite expenses + budget
│   ├── activations.py       sigmoid, ReLU, tanh + derivatives
│   ├── losses.py            binary cross-entropy, MSE (clipped, stable)
│   ├── optimizers.py        SGD and Adam from scratch
│   ├── perceptron.py        single-neuron demo
│   ├── neural_network.py    5-8-1 network: forward, backprop, training loop
│   ├── synthetic_data.py    reproducible synthetic dataset
│   ├── preprocessing.py     standardizer (fit on train only)
│   ├── evaluation.py        train/val/test split, metrics
│   └── model_deployment.py  training pipeline, Joblib save/load, ModelService
├── frontend/                index.html, style.css, script.js (+ optional vendor/)
├── data/                    SQLite DB created here at first run (git-ignored)
├── models/                  budget_model.joblib saved here (git-ignored)
├── tests/                   unittest suite (also works with pytest)
├── requirements.txt  run_mac.command  .gitignore  LICENSE  README.md
```

## Mac setup (Apple Silicon, no Homebrew/Docker needed)
Run all commands from the project root folder:
```bash
python3 --version          # should be 3.11 or 3.12
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 -m backend.app
```
Open **http://127.0.0.1:5000**. Or run `./run_mac.command` (creates the venv, installs only if needed, picks a free port, opens the browser). If macOS blocks it: right-click → Open, or `chmod +x run_mac.command`.

The server binds to `127.0.0.1` only (not reachable from other machines).

**Charts need Chart.js.** The page tries `frontend/vendor/chart.umd.min.js` first, then the jsDelivr CDN. If you are offline, download Chart.js 4 `chart.umd.min.js` once into `frontend/vendor/`. Without it everything else works; charts show text summaries instead.

## Using the app
1. **Overview** → set a monthly budget; totals and charts update.
2. **Expenses** → add/edit/delete; use search, category filter and sort.
3. **Insights** → category shares and weekly trend from your stored expenses.
4. **Model Lab** → pick settings, **Train model**, then **Save model**. Next time press **Load saved model** (no retraining).
5. **Predictor** → type five numbers or press **Use my tracked data**, then **Estimate risk**. Needs a trained or loaded model; otherwise you are told to train/load one.

Training is never run silently; expenses work without a model.

## The mathematics in plain words
- **Perceptron**: `output = activation(w·x + b)`.
- **Forward pass**: `z1 = X·W1 + b1`, `a1 = act(z1)` (ReLU or tanh), `z2 = a1·W2 + b2`, `p = sigmoid(z2)`.
- **Loss** (binary cross-entropy): `-mean(y·log p + (1-y)·log(1-p))`, with `p` clipped to avoid `log(0)`.
- **Backprop**: `dz2 = (p - y)/m`; `dW2 = a1ᵀ·dz2`; `dz1 = (dz2·W2ᵀ) ⊙ act'(z1)`; `dW1 = Xᵀ·dz1`; bias gradients are column sums.
- **SGD**: `w ← w − lr·grad`. **Adam**: bias-corrected running averages of the gradient (`m`) and squared gradient (`v`); `w ← w − lr·m̂/(√v̂+ε)`.
- Initialization: He (ReLU) / Xavier (tanh), zero biases, fixed seed.

## Data & model validity (important)
Training examples are **synthetic**, not your transactions. A scenario is labelled "overrun" when `spent + days_left × (0.75·recent daily average + 0.25·last month's daily rate) + noise > budget` (see `backend/synthetic_data.py`). Metrics show how well the network learned *that rule* on a held-out synthetic test split (70/15/15 train/validation/test; scaler fitted on train only). They say nothing about real financial behaviour; real use would need representative historical data, proper evaluation and calibration. The predictor is an educational demo, not financial advice.

## API
| Method | Path | Purpose |
|---|---|---|
| GET | `/api/expenses?q=&category=&sort=date\|amount&order=asc\|desc` | list |
| POST | `/api/expenses` | create `{date, amount, category, description?}` |
| PUT/DELETE | `/api/expenses/<id>` | update / delete |
| GET | `/api/categories` | category list |
| GET | `/api/dashboard[?month=YYYY-MM]` | totals, categories, daily/weekly series, recent, predictor defaults |
| GET/POST | `/api/budget` | `{monthly_budget}` |
| POST | `/api/train` | `{optimizer, activation, learning_rate, epochs}` |
| POST | `/api/model/save` · `/api/model/load` | persist / restore `models/budget_model.joblib` |
| POST | `/api/predict` | five features → `{probability, risk, ...}` |
| GET | `/api/status` | app + model status, metrics, loss history |

Errors return JSON `{"error": "...", "details": [...]}` with 400/404/409/422/500.

## Testing
```bash
python3 -m unittest discover -s tests -t . -v
```
Tests use temporary databases/model files and never touch `data/` or `models/`.

## Common problems
- **Port 5000 busy / 403 from "AirTunes"**: macOS AirPlay Receiver uses it. Run `PORT=5001 python3 -m backend.app` (or use `run_mac.command`).
- **`ModuleNotFoundError: backend`**: run from the project root with `python3 -m backend.app`.
- **`No module named flask`**: activate the venv (`source .venv/bin/activate`) and reinstall requirements.
- **Charts empty**: see the Chart.js note above.
- **Predictor says no model**: Model Lab → Train (or Load).

## Privacy & security
All data stays on your machine in `data/moneta.db`. Joblib files use pickle, which can run code when loaded: the app only loads its own `models/budget_model.joblib`, has no upload endpoint, and you should never place model files from other people there.

## GitHub submission
```bash
git init && git add . && git commit -m "Moneta Expense Pattern Analyzer"
git branch -M main
git remote add origin https://github.com/<you>/expense-pattern-analyzer.git
git push -u origin main
```

## Viva preparation
- **Perceptron?** One neuron: weighted sum of inputs plus bias, passed through a threshold/activation.
- **Why activation functions?** Without non-linearity, stacked layers collapse into one linear map. ReLU/tanh give hidden layers power; sigmoid turns the output into a probability.
- **Forward propagation?** Inputs → hidden (`XW1+b1`, activation) → output (`a1W2+b2`, sigmoid).
- **Backpropagation?** The chain rule applied backwards to get the loss gradient for every weight and bias.
- **Why binary cross-entropy?** It suits yes/no probabilities and punishes confident wrong answers; with sigmoid the output gradient is simply `p − y`.
- **Gradient descent?** Repeatedly step weights opposite the gradient to lower the loss; the learning rate is the step size.
- **SGD vs Adam?** SGD uses one global step size. Adam adds momentum and per-weight adaptive steps, often converging faster with less tuning.
- **Overfitting?** Memorizing training data; visible when validation loss rises while training loss falls. Mitigate with fewer epochs/neurons, more data, validation monitoring.
- **Train/validation split?** Held-out data is unseen in training so metrics are honest; the scaler is fitted on training data only to avoid leakage.
- **Model serialization?** Saving weights, biases, architecture, scaler and metadata (Joblib) so predictions work without retraining.
- **Why NumPy?** To show exactly how the maths works instead of hiding it inside a framework.
