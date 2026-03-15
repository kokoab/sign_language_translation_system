#!/usr/bin/env bash
set -e

echo "============================================"
echo " SLT - Data Collection Setup (macOS/Linux)"
echo "============================================"
echo

# ── Check Python version ──────────────────────────────────
if ! command -v python3 &>/dev/null; then
    echo "[ERROR] python3 is not installed."
    echo "        Install Python 3.10+ from https://www.python.org/downloads/"
    exit 1
fi

PYVER=$(python3 --version)
echo "Found $PYVER"

# ── Create virtual environment ────────────────────────────
if [ ! -d "venv_collect" ]; then
    echo
    echo "Creating virtual environment..."
    python3 -m venv venv_collect
    echo "Virtual environment created."
else
    echo "Virtual environment already exists, skipping creation."
fi

# ── Activate and install dependencies ─────────────────────
echo
echo "Installing dependencies..."
source venv_collect/bin/activate
python -m pip install --upgrade pip
pip install -r requirements_collect.txt

echo
echo "============================================"
echo " Setup complete! Running data collection..."
echo "============================================"
echo

# ── Run the script ────────────────────────────────────────
python src/collect_data.py

echo
echo "Done."
