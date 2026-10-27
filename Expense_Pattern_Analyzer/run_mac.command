#!/bin/bash
# Double-click (or run ./run_mac.command) to start Moneta on macOS.
cd "$(dirname "$0")" || exit 1

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 not found. Install Python 3.11/3.12 from https://www.python.org/downloads/"
  read -r -p "Press Enter to close..."; exit 1
fi

if [ ! -d ".venv" ]; then
  echo "Creating virtual environment (.venv)..."
  python3 -m venv .venv || { echo "ERROR: could not create .venv"; read -r -p "Press Enter..."; exit 1; }
fi
source .venv/bin/activate

if ! python3 -c "import flask, numpy, joblib" >/dev/null 2>&1; then
  echo "Installing dependencies (needs internet once)..."
  python3 -m pip install --upgrade pip >/dev/null
  python3 -m pip install -r requirements.txt || { echo "ERROR: dependency install failed"; read -r -p "Press Enter..."; exit 1; }
fi

# macOS AirPlay Receiver can occupy port 5000; fall back to 5001.
PORT="${PORT:-5000}"
if lsof -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Port $PORT is busy, using $((PORT+1))."
  PORT=$((PORT+1))
fi
export PORT

# Optional: open the browser shortly after the server starts.
( sleep 2; open "http://127.0.0.1:$PORT" >/dev/null 2>&1 ) &

echo "Moneta running at http://127.0.0.1:$PORT  (press Ctrl+C to stop)"
python3 -m backend.app
status=$?
[ $status -ne 0 ] && { echo "Server exited with an error (code $status)."; read -r -p "Press Enter to close..."; }
