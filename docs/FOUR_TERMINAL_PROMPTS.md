# Four-terminal workflow for Claude Code

Four Claude Code sessions work in parallel, each in its **own git worktree and branch**, each owning
different folders, so they never edit the same file. Terminal 1 integrates. Every session reads
`CLAUDE.md` automatically; the research lives in `docs/RESEARCH_NOTES.md` and `docs/PAPER_PLAN.md`.

## 0. One-time setup (you, in a normal shell)
```bash
unzip farmguard.zip && cd farmguard
git init && git add -A && git commit -m "Base code from research phase"
git worktree add ../fg-vision  -b vision
git worktree add ../fg-agent   -b agent
git worktree add ../fg-exp     -b experiments
# Terminal 1 stays in ./farmguard (branch main). Open terminals 2-4 in ../fg-vision, ../fg-agent, ../fg-exp
# and run `claude` in each. Each worktree needs its own venv: `bash scripts/setup.sh`.
```

## Shared rules (already in CLAUDE.md, repeated in every prompt)
- Only edit the folders you own. Need a change elsewhere? Write it in `docs/HANDOFF.md` under the
  owner's heading and keep going.
- `pytest -q` must pass before every commit. Commit small, on your branch.
- Never break the invariants in CLAUDE.md (no LLM in farmguard/fast, enum + validator for actions,
  no location in prompts, every GenAI step has a fallback, honest baselines, sim = assumptions).

---

## Terminal 1: Lead integrator and QA (worktree `farmguard`, branch `main`)
```
You are the lead integrator for FarmGuard. Read CLAUDE.md, docs/RESEARCH_NOTES.md and
docs/PAPER_PLAN.md fully before doing anything.

You own: pyproject.toml, requirements.txt, scripts/, app/, farmguard/system.py, farmguard/cli.py,
tests/test_end_to_end.py, CLAUDE.md, docs/HANDOFF.md. Other folders are owned by other terminals:
farmguard/fast (vision), farmguard/agent (agent), farmguard/sim + farmguard/eval + paper/ (experiments).

Step 1: Set up the environment (bash scripts/setup.sh), run pytest -q, run the mock demo command in
CLAUDE.md and the Streamlit app. Fix only packaging / environment issues. Report what works.
Step 2: Install and smoke-test the real stack: ultralytics with yolov8s-worldv2.pt on
data/demo/demo.mp4 and on one real clip in data/clips; Ollama with qwen2.5vl:7b
(`--provider ollama`). Record versions in docs/HANDOFF.md under "Environment".
Step 3: Every ~hour: `git merge vision agent experiments` (resolve conflicts, never by deleting
another terminal's work), run the full test suite + mock demo, and write a status block in
docs/HANDOFF.md: what changed, what broke, who should fix it.
Step 4: Keep CLAUDE.md accurate when the design changes. Add a GitHub Actions workflow that runs
pytest with the mock backends.
Acceptance: main always installs from scratch, pytest is green, the mock demo and the Streamlit app
run, and the real-model demo runs on a laptop GPU or Colab.
```

## Terminal 2: Vision and fast path (worktree `fg-vision`, branch `vision`)
```
You own farmguard/fast/, config/settings.yaml (detector, zones, trigger, alarm, video sections),
config/species.yaml, sounds/, data/clips, tests/test_fast_*.py. Read CLAUDE.md and section 2 of
docs/RESEARCH_NOTES.md (gotogrub, athiru, subith2005, SentaFito53, pigeon, critter-cam rows) first.

Invariant you guard: nothing in farmguard/fast may call a network, an LLM, or wait on the agent.

Tasks, in order:
1. Verify UltralyticsDetector against the installed ultralytics version: YOLO-World set_classes,
   model.track(persist=True, tracker="bytetrack.yaml"), boxes.id handling when tracking fails.
   Add tests with a tiny fake results object.
2. Collect footage: download the Roboflow wild boar / monkey datasets (check licences) and any
   farm/CCTV clips we have; write scripts/prepare_data.py that builds a YOLO dataset split by clip.
3. Fine-tune yolov8n on boar/monkey/nilgai/cattle (+ YOLO-World pseudo-labels for unlabelled
   frames, autodistill pattern). Report mAP50 per class, day vs night. Save weights path in config.
4. Night robustness: add optional CLAHE preprocessing before detection and measure its effect.
5. Replace synthetic sounds with licensed real recordings (bee swarm, tiger roar, boar alarm call,
   dog bark...). Keep file names = sound ids. Document sources in sounds/SOURCES.md.
6. Make alarm latency measurement airtight for E1: verify t_read / t_alarm_started stamps in live
   mode, and add a load test (agent running a real VLM in parallel).
Write anything the agent or experiments terminals need into docs/HANDOFF.md.
```

## Terminal 3: GenAI agent (worktree `fg-agent`, branch `agent`)
```
You own farmguard/agent/ (llm/, prompts, clip_builder, vlm_check, retreat, policy, validators,
memory, notifier, orchestrator), config/settings.yaml (agent and notifier sections),
tests/test_agent_*.py. Read CLAUDE.md and sections 2-4 of docs/RESEARCH_NOTES.md first (AlexKapy,
seedling-agent, muye, PharmaWatch, Shoo, scarecrow rows and the "Lessons" list).

Invariants you guard: the LLM only chooses from an enum + validate(); no location in prompts;
every GenAI step has a non-GenAI fallback; the agent never blocks the fast path.

Tasks, in order:
1. Verify each provider in agent/llm/providers.py against current docs (Ollama /api/chat with a
   JSON-schema `format`; google-genai structured output; Anthropic forced tool use;
   OpenAI-compatible json_schema). Add retries with exponential backoff and per-call timeouts.
   Record real latency per call in the logs. Unit-test with mocked HTTP.
2. Few-shot: support 3-5 labelled example frames per species in the VLM prompt (BLIP went 7.61% ->
   91.0% with few-shot in Alencar et al.). Make it a config flag so E2 can ablate it.
3. Retreat check: tune the tracker/VLM fusion rules on the dev split of data/labels/retreat_samples.csv
   only (never the test split). Keep fuse() pure and unit-tested.
4. Policy: make LLMPolicy's context compact and consistent; log the full prompt/response per
   decision (JSONL) for the paper's qualitative examples. Measure invalid-action rate before the
   validator.
5. Farmer side: nightly summary and question answering over the SQLite events table (simple
   retrieval of rows -> LLM answer with citations to event ids). Hindi and Marathi.
6. Optional: natural-language farmer rules ("ignore my brown cow") checked by the LLM, scarecrow
   pattern; a CLIP animal/non-animal gate before the VLM (critter-cam pattern, rewrite: AGPL).
Write questions for other terminals into docs/HANDOFF.md.
```

## Terminal 4: Experiments and paper (worktree `fg-exp`, branch `experiments`)
```
You own farmguard/sim/, farmguard/eval/, config/sim.yaml, data/labels/, results/, paper/,
scripts/figures_*.py, tests/test_sim_*.py. Read CLAUDE.md and docs/PAPER_PLAN.md fully, then
docs/RESEARCH_NOTES.md.

Rules: simulation numbers are assumptions, never field data. Never tune anything on the test
split. Report negative results honestly (the bandit may beat the LLM in some settings).

Tasks, in order:
1. Labelling kit: scripts/extract_frames.py (ffmpeg / OpenCV) and a minimal labelling helper that
   writes data/labels/*.csv in the formats in data/labels/README.md. Dev/test split by clip.
2. E4 simulator: run all policies (llm via local Ollama to keep cost zero) with --repeats 5;
   add the sensitivity grid from PAPER_PLAN.md; learning-curve plot; paired tests. Check the
   simulator code for bugs and for any unfair advantage to one policy.
3. E1, E2, E3, E5, E6 runners: make each reproducible from one command, writing JSON + CSV to
   results/, then scripts that turn results into LaTeX tables and matplotlib figures.
4. paper/: LaTeX skeleton (IEEE conference template) following the outline in PAPER_PLAN.md;
   related-work section from RESEARCH_NOTES.md sections 1-3 with BibTeX entries (verify every
   reference before adding it).
5. Keep a results log in paper/RESULTS.md: date, command, git commit, numbers.
```

---

## Daily loop
1. Morning: Terminal 1 merges, runs tests, updates docs/HANDOFF.md.
2. Terminals 2-4 read HANDOFF.md, pull main into their branch (`git merge main`), continue.
3. Evening: each terminal commits, writes a 5-line status in HANDOFF.md.

## If you only want to run one terminal at a time
Use the same prompts one after another in a single session; start with Terminal 1 step 1-2, then
Terminal 3 task 1 (providers), Terminal 2 tasks 1-3 (detector), Terminal 4 tasks 1-3 (experiments).
