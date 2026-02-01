#!/bin/bash

# =============================================================================
# Start Backend Only
# =============================================================================

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Activate virtual environment if exists
if [ -d "backend/venv" ]; then
    source backend/venv/bin/activate
fi

# Set PYTHONPATH
export PYTHONPATH="$PROJECT_DIR"

# Start backend
cd backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
