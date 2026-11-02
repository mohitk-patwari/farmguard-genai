# FarmGuard: scare first, reason later

A two-speed generative-AI agent that protects crops from raiding animals using farm video.

- **Layer 1 (fast path, no GenAI):** YOLO / YOLO-World + ByteTrack find the animal; once it is in
  the field polygon in 4 of the last 7 frames, an alarm sound plays at once on its own thread.
- **Layer 2 (GenAI agent, once per event):** a vision-language model confirms species, count and
  behaviour (eating vs passing); a retreat check (tracker + VLM) decides whether the animal left;
  if not, an LLM picks the next sound from a safe allow-list using past outcomes; finally the LLM
  writes the farmer's alert in Hindi or Marathi.

## Quick start (no GPU, no API key)
```bash
bash scripts/setup.sh          # venv, deps, placeholder sounds, synthetic demo, tests
python -m farmguard.cli run --source data/demo/demo.mp4 \
  --mock-annotations data/demo/demo_annotations.json --zones data/demo/demo_zones.json --provider mock
streamlit run app/streamlit_app.py
python -m farmguard.cli simulate            # E4 baselines in the simulator
```

## Windows notes
Tested on Windows 11 with Windows PowerShell 5.1 and Python 3.12. Use Python 3.10-3.12.
- Setup: `powershell -ExecutionPolicy Bypass -File scripts\setup.ps1` (add a folder path to set up a
  worktree, e.g. `..\fg-vision`). Each worktree needs its own `.venv`.
- Skip activation and call `.\.venv\Scripts\python -m farmguard.cli ...` directly. PowerShell 5.1 has
  no `&&`, so chain commands with `;`.
- Alarm audio uses pygame. With no working audio device, `auto` falls back to `none` (log only).
- FarmGuard writes UTF-8. If Hindi/Marathi looks garbled when piped, run
  `[Console]::OutputEncoding = [Text.Encoding]::UTF8` first. If it shows boxes in the console,
  the console font lacks Devanagari (Windows Terminal handles it).
- Ctrl+C exits straight away with code 130, abandoning any event the agent is still handling.

## With real models
```bash
ollama pull qwen2.5vl:3b                     # free local VLM (7b on Colab: needs more than 6 GB VRAM)
python -m farmguard.cli draw-zone data/clips/my_farm.mp4
python -m farmguard.cli run --source data/clips/my_farm.mp4 --provider ollama --language hi
# cloud alternatives: --provider gemini | anthropic | openai_compat  (keys in .env)
```

## Experiments (see docs/PAPER_PLAN.md)
`farmguard eval e1|e2|e3|e5|e6` and `farmguard simulate --policies fixed,random,rule_chain,bandit,llm`.

## Docs
- `CLAUDE.md`: architecture, invariants, commands (read by Claude Code automatically)
- `docs/RESEARCH_NOTES.md`: what we learned from 333 repos and 11 papers, with file/line references
- `docs/PAPER_PLAN.md`: contributions, experiments, venues
- `docs/FOUR_TERMINAL_PROMPTS.md`: how to continue the build with four Claude Code sessions

The sounds in `sounds/` are synthetic placeholders; replace them with licensed recordings before
any real test.
