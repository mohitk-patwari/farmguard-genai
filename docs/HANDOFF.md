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

## For Terminal 3 (agent)
- Alarms from the agent go through the same AlarmPlayer. Sounds are decoded once at startup now, so
  a WAV added while running is read from disk on its first play.
- Sound ids are unchanged, but see the caveats in `sounds/SOURCES.md` before writing prompt text
  about a sound: `boar_alarm` is not a verified alarm call and `tiger_roar` is a growl.

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
