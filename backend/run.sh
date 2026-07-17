#!/usr/bin/env bash
set -e
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
echo "Pull models first:  ollama pull mistral:7b && ollama pull nomic-embed-text"
uvicorn app.main:app --reload --port 8000
