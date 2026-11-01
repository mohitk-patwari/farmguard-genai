# HANDOFF (shared notes between the four Claude Code terminals)

## Environment
(Terminal 1: record Python, CUDA, ultralytics, Ollama and model versions here.)

## Status log
- Base code from the research phase: pytest green (14 tests), mock demo runs end to end, simulator
  runs with fixed / random / rule_chain / bandit. Real providers and the real detector are written
  but NOT yet verified against live APIs / weights.
- Agent (branch `agent`, 2026-11-01): providers fixed, retries + per-call latency logs, few-shot flag,
  decision log + invalid-action rate, farmer summary and Q&A. All tested with mock clients and a local
  fake HTTP server (real `requests` and real SDKs, fake responses). **No provider has been called live:**
  Ollama is not installed here and no paid API was called. Fusion rules are NOT tuned (no labels yet).

## For Terminal 1 (integrator)
- Before relying on results, smoke-test each provider you have keys for:
  `python -m farmguard.agent.farmer ask "today" --provider gemini --language hi`. It reaches the LLM only
  if events exist; a `llm_call ... ok=1` line in the log (run `farmguard run`) is the proof.
- pyproject `genai` extra is unpinned. The code needs `anthropic>=2` (API version 2026-10-21) and a
  google-genai with `response_json_schema`. Tested with anthropic 2.3.0, google-genai 2.34.0.
- CLAUDE.md layout: add `agent/farmer.py`. Optional: `farmguard summary` / `farmguard ask` in cli.py, wrapping
  `farmguard.agent.farmer.main`. Works today as `python -m farmguard.agent.farmer summary|ask`.
- The nightly summary is not scheduled by the app: use cron / Task Scheduler at `notifier.nightly_summary_hour`.
- test_end_to_end.py writes mock decisions to data/policy_decisions.jsonl (gitignored). Override
  `agent.policy.decision_log` there if you want it clean; the summary groups by model, so `mock` stays separate.

## For Terminal 2 (vision)
- `tracker_evidence` counts a detection with `track_id=None` as present anywhere in the frame, even outside
  the zone, because `FramePacket` only records in-zone track ids. If tracking often fails, retreat reads
  "still present" for an animal that left the field. Could FramePacket carry in-zone detections too?

## For Terminal 3 (agent)
- Blocked: fusion tuning needs labels (see Terminal 4 below). Then run `retreat.fusion_errors(rows)` and
  edit the rules in `fuse()` for the worst cells.

## For Terminal 4 (experiments)
- **Needed for fusion tuning:** `data/labels/retreat_samples.csv` with a `split` column (dev/test by clip),
  and an E3 per-sample dump with `tracker_status, vlm_status, vlm_confidence`. `retreat.fusion_errors()` reads
  those columns plus `status`, uses dev rows only, and raises if `split` is missing.
- E2: few-shot is `cfg.raw["agent"]["fewshot"]["enabled"]`, read on each `check()`, so flip it per condition like
  the location flag. Frames go in `data/fewshot/<species key|none|other_animal>/` (none are collected yet), kept out
  of the test clips. About 21 extra images per call at 3 per category, so report latency too. Labels name species,
  which hurt behaviour prompts in Dussert 2025: report behaviour accuracy with and without.
- E2 methods: Anthropic runs at its default sampling (`temperature` is rejected on API version 2026-10-21), while
  other providers use `agent.temperature`. Say so when comparing models.
- E4: `LLMPolicy.source` is now `llm_fallback_bandit` when the LLM failed, so count those apart from `llm`. To
  measure invalid actions in the simulator, set `pol.decision_log = ...` on the policy in `run_all`, then call
  `policy.decision_log_summary(path)`. `invalid_action_rate` includes in-enum picks the validator changed.
  The first-sound prompt now says no animal is present, which changes LLM runs compared with older logs.
- Cost per event is not logged yet (`llm_call` has latency only). Tell Terminal 3 if the paper needs token counts.
