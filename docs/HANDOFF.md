# HANDOFF (shared notes between the four Claude Code terminals)

## Environment
(Terminal 1: record Python, CUDA, ultralytics, Ollama and model versions here.)

## Status log
- Base code from the research phase: pytest green (14 tests), mock demo runs end to end, simulator
  runs with fixed / random / rule_chain / bandit. Real providers and the real detector are written
  but NOT yet verified against live APIs / weights.
- Vision, 2026-11-01 (branch `vision`). Done: detector verified on ultralytics 8.4.197 (CPU only so
  far), no-network fix, E1 stamp fixes, CLAHE code, 7 of 9 real sounds, dataset scripts.
  NOT done: no detector has been fine-tuned and there are NO mAP or CLAHE numbers yet. Written but
  never run end to end: `scripts/train_detector.py`, `scripts/e1_load_test.py`,
  `prepare_data.py lila`, `pseudo-label`, `build`.
- Agent (branch `agent`, 2026-11-01): providers fixed, retries + per-call latency logs, few-shot flag,
  decision log + invalid-action rate, farmer summary and Q&A. All tested with mock clients and a local
  fake HTTP server (real `requests` and real SDKs, fake responses). **No provider has been called live:**
  Ollama is not installed here and no paid API was called. Fusion rules are NOT tuned (no labels yet).

## For Terminal 1 (integrator)
- Add to requirements.txt: `lap>=0.5.12`. Without it ultralytics runs `pip install` from inside the
  frame loop; the detector now refuses to start instead. `soundfile` is needed by
  `scripts/fetch_sounds.py` only.
- GPU torch on Windows: plain `pip install torch` gives 2.15.0+cpu. The newest CUDA build found is
  2.14.1+cu130 (https://download.pytorch.org/whl/cu130), so pin torch and torchvision to it.
  `--upgrade torch==2.15.0` from that index reports success and changes nothing.
- Setup must now run `python scripts/bake_world_vocab.py` once while online (needs
  `weights/yolov8s-worldv2.pt` and the CLIP package). `detector.model` points at the baked file, and
  the real detector raises FileNotFoundError until it exists. Mock runs are unaffected.
- Ollama is not installed here and nothing listens on port 11434. The real-VLM E1 arm needs it.
- Changed outside my folders: `farmguard/tools.py` (make_sounds skips existing files; setup scripts
  would otherwise overwrite the real recordings) and `.gitignore`.
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
- Alarms from the agent go through the same AlarmPlayer. Sounds are decoded once at startup now, so
  a WAV added while running is read from disk on its first play.
- Sound ids are unchanged, but see the caveats in `sounds/SOURCES.md` before writing prompt text
  about a sound: `boar_alarm` is not a verified alarm call and `tiger_roar` is a growl.
- Blocked: fusion tuning needs labels (see Terminal 4 below). Then run `retreat.fusion_errors(rows)` and
  edit the rules in `fuse()` for the worst cells.

## For Terminal 4 (experiments)
- `data/timing.csv` has new columns: `backend`, `ok`, `t_first_zone`, `first_zone_to_start_ms`. An
  old file with the old header is renamed to `timing.old<time>.csv`, not appended to.
- E1 analysis should drop rows with `backend=none` (no audio played) and report
  `first_zone_to_start_ms` next to `read_to_start_ms`. Only the first includes the 4-of-7 vote, and
  a slower frame rate under load shows up there and not in the second.
- `t_alarm_started` is when play() returned. Mixer and driver output delay is not measured, so do not
  describe it as sound leaving the speaker. Failed alarms have an empty `read_to_start_ms`.
- Any timing recorded before this branch is not comparable: the old stamp was taken before the sound
  was loaded and played, and a fixed 0.2 s wait could drop slow alarms from the log.
- Behaviour change that affects E5: frames skipped by the motion gate no longer vote. Before, one
  false positive on a still scene could collect all 4 votes by being carried forward. A still animal
  now needs 4 real detector runs (keepalive is every 10 frames), so its alarm comes later.
- `results/e1_load_standin.json`, if present, is a GPU-stress stand-in and not an E1 result.
- Paper text: 2 of 9 sounds are still synthetic (`leopard_growl`, `human_shout`), and none of the
  recordings has been tested on animals.
- Detector labels: nilgai boxes will be YOLO-World pseudo-labels, because no human-boxed nilgai source
  was found. Scores on those are model agreement, not accuracy. `pig` in Open Images is mostly domestic.
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
