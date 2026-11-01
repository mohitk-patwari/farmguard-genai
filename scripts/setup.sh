#!/usr/bin/env bash
# One-time setup. Run from the repo root:  bash scripts/setup.sh
set -euo pipefail

# Git Bash on Windows: venv layout is Scripts/, and python3 is often the Store shim, not the install.
case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*) PY=python;  VENV_BIN=Scripts ;;
  *)                    PY=python3; VENV_BIN=bin ;;
esac
"$PY" -m venv .venv
source ".venv/$VENV_BIN/activate"
python -m pip install --upgrade pip  # "python -m": pip.exe cannot replace itself on Windows
python -m pip install -e ".[dev]"
python -m pip install -r requirements.txt || echo "Some optional packages failed; core still works."

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
