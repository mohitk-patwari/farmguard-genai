# CLAUDE.md: read this first

You are working on **FarmGuard**, a GenAI research project that will become a conference paper.
The base code was written from a review of 333 GitHub repos and 11 papers; the findings live in
`docs/RESEARCH_NOTES.md`. The paper plan and experiments are in `docs/PAPER_PLAN.md`. Read both
before changing behaviour. Do not "simplify" a design choice without checking why it was made.

## What the system does
Farm video, frame by frame. When a crop-raiding animal (wild boar, nilgai, monkey, elephant,
stray cattle) enters the field, **scare it first, tell the farmer last**.

```
Layer 1  FAST PATH  (farmguard/fast)   every frame, NO network, NO LLM
  video -> motion gate -> YOLO/YOLO-World + ByteTrack -> field polygon -> 4-of-7 vote -> alarm thread
                                                                             \-> IntrusionEvent queue
Layer 2  GENAI AGENT (farmguard/agent) once per event, never blocks Layer 1
  VLM check (species, count, eating vs passing) || retreat check (tracker + VLM) ->
  LLM policy picks next sound from an allow-list -> validator -> alarm -> retreat check ... ->
  LLM farmer alert in Hindi/Marathi (template fallback) -> SQLite outcome memory
  -> after each event: recommend the species' next FIRST sound (cached dict read by Layer 1)
```

## Non-negotiable invariants (the paper's claims depend on them)
1. **Nothing in `farmguard/fast/` may call a network, an LLM or block on the agent.** The alarm is
   queued to its own thread. `fast/timing.py` logs frame-to-alarm latency (experiment E1).
2. **The LLM can only choose, never invent.** Every decision uses an enum JSON schema
   (`prompts.policy_schema`) and then `agent/validators.validate()`. Never pass free text to the
   speaker. (AniREP repo let Gemini invent an ultrasonic frequency: we cite that as the failure.)
3. **Never put the farm's location in a VLM prompt** (`agent.include_location_in_prompt: false`).
   A location hint dropped Claude's accuracy to ~10% in AlexKapy's pipeline. Only the E2 ablation
   turns it on, deliberately.
4. **Every GenAI step has a non-GenAI fallback** (detector label, tracker evidence, bandit policy,
   template alert). The system must keep working with the network unplugged.
5. **Baselines stay honest.** `fixed`, `random`, `rule_chain` (AEGISFARM-style) and `bandit`
   (Thompson sampling) must use the same PolicyContext, validator and memory as `llm`.
6. **Simulation numbers are assumptions** (`config/sim.yaml`). Never present them as field data.
7. Humane deterrents only (sound, and in a deployment light/water). No shock, no harm. One reviewed
   repo maps animals and humans to shock currents; we explicitly reject that.

## Layout
```
config/settings.yaml   all runtime knobs     config/species.yaml  species, aliases, allowed sounds
config/sim.yaml        simulator assumptions config/zones.json    field polygon (farmguard draw-zone)
farmguard/fast/        video_source, motion, detector, zones, trigger, alarm, timing, framebuffer, pipeline
farmguard/agent/       llm/{base,providers}, prompts, clip_builder, vlm_check, retreat, policy,
                       validators, memory, notifier, orchestrator, farmer (nightly summary, Q&A)
farmguard/sim/         simulator.py (E4)     farmguard/eval/  metrics.py, experiments.py (E1-E6)
farmguard/system.py    wires both layers     farmguard/cli.py  `farmguard run|simulate|eval|...`
app/streamlit_app.py   classroom demo        tests/            pytest (mock detector + mock LLM)
```

## Commands
```bash
pip install -e ".[dev]" && pip install -r requirements.txt
python -m farmguard.cli make-sounds && python -m farmguard.cli make-demo   # never overwrites a WAV
python scripts/bake_world_vocab.py                           # once, online: the real detector needs it
pytest -q                                                    # must stay green
python -m farmguard.cli run --source data/demo/demo.mp4 \
   --mock-annotations data/demo/demo_annotations.json --zones data/demo/demo_zones.json --provider mock
python -m farmguard.cli run ... --mute                       # no audio, still logged (or FARMGUARD_MUTE=1)
python -m farmguard.cli run --source data/clips/x.mp4 --provider ollama     # real YOLO-World + Qwen2.5-VL
python -m farmguard.cli run --source data/clips/x.mp4 --no-agent           # E1 baseline
python -m farmguard.cli simulate --policies fixed,random,rule_chain,bandit,llm --provider ollama
python -m farmguard.cli eval e2 --provider ollama --labels data/labels/vlm_samples.csv
python -m farmguard.eval <exp>                               # preferred: adds preflight + results log
python -m farmguard.agent.farmer summary                     # or: ask "today" --language hi
streamlit run app/streamlit_app.py
```

## Conventions
- Python 3.10+, type hints, dataclasses, `logging` (no prints except CLI output and the farmer alert).
- Heavy imports (ultralytics, SDKs) are lazy so tests run without them.
- New behaviour needs a unit test with MockDetector / MockClient. Do not call real APIs in tests.
- Config over constants: add knobs to settings.yaml with a comment citing why.
- Keep provider code in `agent/llm/providers.py`; verify model ids and SDK signatures against the
  current docs before relying on them (they change).
- When you borrow an idea from a reviewed repo, cite it in a comment the way existing code does.

## Known TODOs (good first tasks)
- Verify Gemini / Anthropic / OpenAI-compat calls against live APIs; add retries with backoff.
- `eval`: helper to cut clips into labelled frames (ffmpeg) and a tracker-only baseline for E3.
- Fine-tune YOLOv8n on Roboflow boar/monkey + YOLO-World pseudo-labels (autodistill pattern).
- Diffusion generator for night / nilgai test frames (`data/synth/`), always reported separately.
- Nightly summary + farmer question answering over `events` (RAG over SQLite rows).
- Plots for the paper (`results/` -> figures) and a LaTeX table exporter.
