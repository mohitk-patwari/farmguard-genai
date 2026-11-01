#!/usr/bin/env bash
# One-time setup. Run from the repo root:  bash scripts/setup.sh
set -euo pipefail

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
pip install -r requirements.txt || echo "Some optional packages failed; core still works."

[ -f .env ] || cp .env.example .env
python -m farmguard.cli make-sounds
python -m farmguard.cli make-demo
pytest -q

# Local VLM (free): https://ollama.com  then:
if command -v ollama >/dev/null 2>&1; then
  ollama pull qwen2.5vl:7b || true
else
  echo "Ollama not found: install it from https://ollama.com or use --provider gemini/anthropic"
fi

# YOLO-World weights download automatically on first use (yolov8s-worldv2.pt).
echo "Done. Try: python -m farmguard.cli run --source data/demo/demo.mp4 --mock-annotations data/demo/demo_annotations.json --zones data/demo/demo_zones.json --provider mock"
