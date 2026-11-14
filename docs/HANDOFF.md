# HANDOFF (shared notes between the four Claude Code terminals)

## Environment
Recorded 2026-11-02 on the lead laptop (Windows 11, 16 GB RAM, RTX 3050 6 GB, driver 581.86).
- Python 3.12.5. torch 2.14.1+cu130, torchvision 0.29.1+cu130 (CUDA 13.0), in `farmguard` and `fg-vision`
  venvs only. ultralytics 8.4.197, lap 0.5.13, clip 1.0, opencv-python 5.0.0.93, numpy 2.5.4,
  streamlit 1.66.1, pygame 2.6.1, anthropic 2.3.0, google-genai 2.34.0.
- Setup installs CPU torch. For the GPU, afterwards:
  `pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130`
- Ollama 0.40.13 (winget), models `qwen2.5vl:7b` (6.0 GB) and `qwen2.5vl:3b` (3.2 GB).
- Weights: `weights/yolov8s-worldv2-farmguard.pt`, baked with `scripts/bake_world_vocab.py`. It worked from
  an empty `weights/` folder, fetching the base weights and installing CLIP by itself.
- Run heavy jobs one at a time on this laptop. Parallel jobs ran it out of RAM.

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
  decision log + invalid-action rate, farmer summary and Q&A. Tested with mock clients and a local fake
  HTTP server. No paid API has been called. Fusion rules are NOT tuned (no labels yet).
- Agent, 2026-11-03: fixes for the smoke test below (images, `num_ctx`, alert grounding, 3b default, policy
  prompt). Unit-tested only. **The re-run did not happen:** it hit the 4 GB free-RAM limit 8 s in (started at
  4339 MB; T1's 3b run started near 7 GB). So the VLM success rate after these fixes is still unknown.
- Agent, 2026-11-11: constrained policy enum, VLM species override and possible-false-alarm note, no farmer
  escalation before attempt 2 (except elephant), `bandit_sw`, Ollama seed. Unit-tested only; no model was run.
- Integration, 2026-11-02 (branch `main`). Merged vision, agent, experiments; only HANDOFF.md conflicted.
  `LatencyLog.save()` (vision) emptied the rows `e1_run` reads (experiments), so E1 reported 0 alarms: fixed
  in `fast/timing.py`. Added `--mute` / `FARMGUARD_MUTE=1`, `lap`, and the bake step in setup.
- Real-model smoke test, 2026-11-02, `--mute`, so alarm timings exclude audio. These are smoke numbers, NOT
  E1 results or field data. YOLO-World finds no animal in `data/demo/demo.mp4` (cartoon blob): 0 events, agent
  never ran; detect median 3.4 ms, p95 58 ms. The table is from one 10 s real clip,
  `boar__Sus_scrofa-wild_boars-Bulgaria-2_VP8.webm`, whole frame as the field. Each run: 11 events.

  | | 7b | 3b | 3b, num_ctx 8192 |
  |---|---|---|---|
  | Finished | no, stopped at my 570 s limit; 11/11 alerts sent | yes, 248 s, 11/11 alerts | no, stopped at 490 s by RAM floor; 8/11 alerts |
  | Ollama placement | 43%/57% CPU/GPU | 100% GPU | 100% GPU |
  | Peak VRAM (idle 312 MB) | 5175 MB | 4581 MB | 4833 MB |
  | Free RAM, start -> lowest | 6789 -> 2181 MB | 6982 -> 4174 MB | 6974 -> 1505 MB |
  | VLM event check | 0/23 ok | 0/22 ok | 18/18 ok, median 12.6 s |
  | VLM retreat check (1 image), median | 7.8 s | 3.2 s | 3.2 s |
  | Policy call, median | 5.3 s | 3.2 s | not split |
  | Alert + first-sound calls, median | 7.7 s | 4.2 s | not split |
  | Alarm read -> start, median / max | 30.8 / 47.8 ms | 34.6 / 42.4 ms | 35.1 / 58.0 ms |
  | Alarm first-zone -> start, median / max | 267 / 835 ms | 271 / 824 ms | 267 / 827 ms |
  | Decisions out of enum | 0/21 | 0/22 | 0/13 |

  Detect on this clip (3b run, the only one that printed it): median 31.7 ms, p95 61.6 ms. No accuracy was
  measured. Use 3b on this laptop; 7b needs more VRAM than it has. Neither is ready for a real run until the
  items below are fixed.

## Smoke test 2 (Terminal 3)
2026-11-13, branch `agent` at 89607de plus this commit's fixes. Model `qwen2.5vl:3b` (Ollama, num_ctx 4096,
temperature 0.2, seed 42), YOLO-World `yolov8s-worldv2-farmguard.pt`, `--mute`, Hindi alerts, `constrain_enum: true`.
Same 10 s clip as above, whole frame as the field, fresh memory per run. **A smoke test: not E2/E3/E5 results, one
clip, 3 events per run, no ground truth beyond "the clip shows wild boars".** The fg-agent venv has CPU-only torch
(2.15.0+cpu), so detect and alarm times are CPU numbers, NOT comparable with the GPU numbers above.

First Run A (as committed in 737f0ea) failed every multi-image VLM call: Ollama charged ~1030 tokens per image even
at 448 px, so 4 images were 4402-4435 tokens (event check) and 4310-4312 (retreat check) against 4096: HTTP 400,
0/3 event checks. Fixed (3 images, see below) and Run A was rerun once. Run B is with the fix.

| | Run A (override on) | Run B (override off) |
|---|---|---|
| Events (T1: 11 before species cooldown) | 3 | 3 |
| VLM event check ok / calls | 3/3 | 3/3 |
| Prompt tokens, event check (3 images): median / max | 3378 / 3401 | 3378 / 3401 |
| Prompt tokens, retreat (1-3 images) / text calls: max | 3278 / 366 | 3278 / 366 |
| Detector -> VLM (conf) -> effective, per event | cattle -> boar (med) -> cattle; elephant -> boar (med) -> elephant; boar -> boar (med) -> boar | same three |
| Overridden / possible_false_alarm | 0 / 0 | 0 / 0 |
| Retreat after 1st alarm (tracker / VLM / fused) | still / still / still, all 3 | same |
| Retreat after 2nd sound (tracker / VLM / fused) | still / leaving / leaving, all 3 | same |
| Policy actions (in-event) | dog_bark, tiger_roar, gunshot | dog_bark, tiger_roar, gunshot |
| Invalid actions / escalations | 0/6 decisions / none | 0/6 / none |
| Alerts: LLM / template_ungrounded | 0 / 3 | 1 / 2 |
| Detect (CPU) median / max ms | 118.6 / 170.3 | 101.7 / 130.7 |
| Alarm read -> start median / max ms (no audio) | 120.1 / 123.0 | 101.1 / 120.9 |
| Alarm first-zone -> start median / max ms | 631.6 / 634.5 | 438.0 / 458.5 |
| VLM event check median / max s | 7.6 / 12.6 | 12.4 / 13.3 |
| VLM retreat median / max s | 6.8 / 13.2 | 6.3 / 11.5 |
| Policy call median / max s | 3.1 / 3.3 | 3.1 / 3.3 |
| Alert call median / max s | 6.0 / 6.1 | 6.0 / 6.6 |
| Peak VRAM (idle 629 MB) / lowest free RAM | 4651 MB / 6.03 GB | 4651 MB / 5.97 GB |
| Wall time | 98 s | 100 s |

Findings:
- The VLM named wild boar on all 3 events (the detector was right on 1), but only at medium confidence, so the
  `high` threshold kept the detector's wrong species twice in both runs. With override off or on, behaviour was the
  same on this clip: it neither helped nor hurt. A `medium` threshold would have fixed 2/3 here, but that's 3 events
  with no accuracy measurement: decide it from E2, not from this.
- The retreat check after the 2nd sound saw ONE frame: the sound was played at the end of the 10 s clip, so the
  window had only the last frame, and "leaving" (recorded as success in memory) came from that frame. A live camera
  doesn't have this; short clips do. Not fixed: needs a rule for truncated windows (E3/E5 on clips) - see T3 notes.
- Grounding check: 5 of 6 LLM alerts were rejected, each for a paraphrased sound name ("आदमी ने चिल्लाने की आवाज़" for
  "आदमी के चिल्लाने की आवाज़"; "आवाज़ मधुमक्खियों"; "उसकी आवाज़" for the boar sound). The one accepted (Run B,
  cattle) ends with a nonsense sentence ("यदि यह नहीं जा रहा है, तो फ़रमान की जाने लायक है" = "if it is not leaving, a
  decree is worth issuing"), which the check can't judge. Same nonsense in the first, overflowing Run A's accepted
  boar alert ("...सायरन गिरे" = "the sirens fell"; "फ़रमान करें" = "issue a decree"). 3b's Hindi is weak.
- Constrained enum: 0 invalid, 0 escalations; elephant's enum included `escalate_farmer` at attempt 1 as designed.

## Smoke test 3 (Terminal 3)
2026-11-14, branch `agent` at 197ca03 (shuffle, llm_bandit, truncated windows, elephant safety, stem grounding).
Same setup as smoke test 2 Run A: same clip, whole frame as field, `qwen2.5vl:3b`, num_ctx 4096, seed 42,
`constrain_enum: true`, `shuffle_options: true`, `override_species: true` (threshold high), `--mute`, Hindi,
CPU-only torch. **A smoke test, not E2/E3 results: one clip, 3 events, one run.** No bug found; nothing rerun.

| | Smoke test 2 (Run A, fixed) | Smoke test 3 |
|---|---|---|
| Events | 3 | 3 |
| VLM event check ok / calls; max prompt tokens | 3/3; 3401 | 3/3; 3401 |
| VLM retreat calls ok / calls; max prompt tokens | 6/6; 3278 | 3/3; 3278 (no call on truncated windows) |
| Detector -> VLM (conf) -> effective | cattle -> boar (med) -> cattle; elephant -> boar (med) -> elephant; boar -> boar (med) -> boar | same |
| Species at none / high / medium | not logged | cattle / cattle / **boar**; elephant / elephant / **boar**; boar / boar / boar |
| Elephant disagreement | not handled | flagged; no `escalate_farmer` offered in-event; alert names both |
| Retreat after 1st alarm (tracker / VLM / fused) | still / still / still, x3 | same |
| Retreat after 2nd sound | still / leaving / **leaving from 1 frame**, x3 (written to memory) | **truncated**, x3 (not in memory) |
| Final status | leaving x3 | truncated x3 |
| In-event actions (shown order -> chosen position) | dog_bark, tiger_roar, gunshot: all position 0, fixed order | dog_bark [drum, dog, siren] -> 1; drum_beat [siren, drum, tiger] -> 1; siren [siren, gunshot, tiger] -> 0 |
| first_position_rate (all 6 LLM answers, incl. next-visit first sound) | 3/6 (no shuffle) | 2/6 |
| Invalid actions / escalations | 0/6 / none | 0/6 / none |
| Alerts sent from LLM / template | 0 / 3 (all rejected for rewording) | 2 / 1 (1 rejected for a wrong outcome) |
| Detect (CPU) median / max ms | 118.6 / 170.3 | 100.8 / 132.8 |
| Alarm read -> start; first-zone -> start, median / max ms | 120.1 / 123.0; 631.6 / 634.5 | 100.9 / 104.4; 440.7 / 444.2 |
| VLM event; VLM retreat, median / max s | 7.6 / 12.6; 6.8 / 13.2 | 13.2 / 25.5; 7.9 / 13.8 |
| Policy; alert call, median / max s | 3.1 / 3.3; 6.0 / 6.1 | 3.2 / 3.4; 4.9 / 5.9 |
| Peak VRAM (idle) / lowest free RAM / wall | 4651 MB (629) / 6.03 GB / 98 s | 4697 MB (675) / 7.31 GB / 96 s |

Alerts (all three sent at 13:39):
- Cattle event, **LLM, accepted**: "आवारा पशु 13:39 की आदमी के चिल्लाने की आवाज़ और कुत्ते की आवाज़ सुनी. स्थिति साफ़ नहीं है.
  फ़र्मर, फ़िल्ड में जाकर देखें." = "Stray cattle at 13:39 heard a man shouting and a dog. The situation is unclear.
  Farmer, go to the field and check." Facts right, grammar broken.
- Elephant event, **LLM, accepted**, plus the fixed safety sentence: "हाथी आयी 13:39 में मधुमक्खियों की आवाज़ और ढोल सुना।
  स्थिति साफ़ नहीं है। जानवर पक्का नहीं है (डिटेक्टर: हाथी, कैमरा जाँच: जंगली सूअर)।" = "Elephant came at 13:39, heard bees and
  the drum. The situation is unclear. Species not certain (detector: elephant, camera check: wild boar)."
- Boar event, **template** (`template_ungrounded`): "13:39 बजे खेत में जंगली सूअर आया। बजाई गई आवाज़ें: सूअर की आवाज़, सायरन। जानवर
  स्थिति साफ़ नहीं है। कृपया खेत देखें।" = "At 13:39 a wild boar came into the field. Sounds played: boar sound, siren. Animal:
  status unclear. Please check the field." The LLM wrote "जंगली सूअर आया था। उसने सूअर की आवाज़ और सायरन सुना। यह नहीं गया।
  फ़रमान देखें।" ("...It did not go. Look at the decree."): rejected, missing outcome. Correct: it claimed the boar stayed,
  but the window was truncated, so the outcome is unknown.

Findings: the VLM again said wild boar at medium on all 3 events, so at `medium` the effective species would have been
right 3/3 (2 overrides) on this clip; still no accuracy measurement, so the default stays `high` until E2. All three
second-sound windows ended with the clip and were truncated instead of being recorded as successes. Shuffle moved
the in-event choices off position 0 (1 of 3 first vs 3 of 3), but 3 answers measure nothing about bias.

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
From Terminal 4:
- One command per experiment is `python -m farmguard.eval <exp>` (owned by experiments). The older
  `farmguard eval` / `farmguard simulate` still work, but skip preflight and the results log.
  `simulate` writes `results/e4_<mode>.json` now, not `e4_simulation.json`. Worth pointing CLAUDE.md there.
- `scripts/figures_all.py` needs matplotlib, which is only in the `app` extra. Consider adding it to `dev`.
- Corrections for docs/RESEARCH_NOTES.md section 3, from reading the sources on 2026-11-01:
  - arXiv:2510.08955 (denoised diffusion, F1 0.64 vs 0.49) now shows as **withdrawn**. Not cited.
  - Li et al. (Frontiers 2026) only reports 87% -> 93%: the result is Shi et al. 2025, cited inside it.
    Its title is "Challenges and strategies for harnessing large language models in plant protection".
  - Conservation Evidence 81 -> 69 min is for noise deterrents AND tree nets together, not noise alone.
  - The ZHAW source is a press release. "Damage reduced" refers to preventive methods in general; it
    gives no figure for the acoustic device alone.
  - Dussert: the preprint says 97.45%, the authors' 2025 summary says 96%. The paper uses 96%.
  - The durian paper's test-set accuracy is 90/85/70%; the mAP values are from training.

## For Terminal 2 (vision)
From the smoke test (Terminal 1):
- All 11 events on the boar camera-trap clip were mislabelled: 9 `stray_cattle`, 2 `elephant`, none
  `wild_boar`. One 10 s clip giving 11 events (one per new track) also means repeated alarms and agent runs.
- `tracker_evidence` counts a detection with `track_id=None` as present anywhere in the frame, even outside
  the zone, because `FramePacket` only records in-zone track ids. If tracking often fails, retreat reads
  "still present" for an animal that left the field. Could FramePacket carry in-zone detections too?

Answered by Terminal 2 (commit a0eb2e0): `FramePacket.in_zone` now holds the in-zone detections, and
`tracker_evidence` reads only that. `in_zone_track_ids` is a read-only property now, so assign `in_zone`.
`trigger.species_cooldown_s` (60) stops new track ids of a species from starting events. It works per
species label, so a clip whose boar flips between cattle and elephant still gives one event per wrong
label until the detector is fine-tuned. Not re-run on the smoke-test clip yet. ByteTrack is not tuned
yet: `scripts/tune_tracker.py` is written but not run, and `detector.tracker` is still the default.

## For Terminal 3 (agent)
Status 2026-11-03: VLM overflow, alert checks and the policy prompt below are fixed, not yet re-measured live.
Validator changes, from T1's decision log: all were in-event decisions (attempt 2+), so the real rate there is
9/11 (7b) and 10/11 (3b); first-sound calls are never changed. 7b: 9 on cooldown, always its favourite sound.
3b: 9 re-picked the sound just tried, 1 on cooldown. In both, 7 of the changes became a forced escalation
because every sound was tried or on cooldown: `_last_played` is shared across events, and this clip has 11
events in 10 s. Not changed: the enum still offers tried and cooldown sounds, so the rate stays measurable.
From the smoke test (Terminal 1), first calls ever made to a live provider:
- **The VLM event check always fails on Ollama.** `request (4311 tokens) exceeds the available context size
  (4096 tokens)`; 6 images need about 6500 tokens. `OllamaClient` sends no `num_ctx`. The agent falls back to
  the detector label, so nothing crashes and the failure is easy to miss. It worked with 8192, set in a
  temporary Modelfile, but free RAM then fell to 1.5 GB. Cause not investigated. Try fewer or smaller images
  as well as a config knob for `num_ctx`.
- LLM farmer alerts in Hindi are not usable yet on 7b or 3b: garbled, wrong sound names, repeated phrases, and
  some Marathi mixed in. One 3b example: `हाथी आया था। यहाँ बीमारी की चिंता करते हैं।` ("An elephant came. Here
  we worry about disease."). They pass as `llm`, so the template fallback never fires. Facts should be
  checked before E6 and before any farmer sees them.
- Text calls sometimes hit the 120 s timeout (1 on 7b, 2 on 3b at 8192). Cause not investigated.
- The validator changed 9/21 (7b) and 10/22 (3b) decisions. Worth checking why.
- One `llama-server` crash on the first 7b call, recovered on retry: `exit status 0xc0000409 ... CUDA error:
  shared object initialization failed`. Not seen on 3b.
- Alarms from the agent go through the same AlarmPlayer. Sounds are decoded once at startup now, so
  a WAV added while running is read from disk on its first play.
- Sound ids are unchanged, but see the caveats in `sounds/SOURCES.md` before writing prompt text
  about a sound: `boar_alarm` is not a verified alarm call and `tiger_roar` is a growl.
- Blocked: fusion tuning needs labels (see Terminal 4 below). Then run `retreat.fusion_errors(rows)` and
  edit the rules in `fuse()` for the worst cells.
From the E4 audit (Terminal 4). None of these is changed in my code; the simulator works around the first:
- `orchestrator.handle` calls `policy.decide` when `attempt_no >= max_attempts`, so the LLM is called and
  its answer discarded, because the validator forces escalation anyway. Skip the call on the last attempt.
- `Policy.decide` reports `source="llm"` even when the bandit fallback made the decision. E4 detects this
  by matching "LLM failed" in the reason, which is fragile. Please expose it as a field or set
  `source="fallback"` (events.py already lists that value).
- `policy_prompt` opens with "the previous deterrent did not make it leave", which is false for the
  first-sound recommendation (`tried_this_event=[]`). This affects the live agent and E4.
- `policy_prompt` tells the LLM that animals habituate, which is the simulator's own model. E4 must
  state this: the LLM is told the structure of the world and the bandit is not.
- `BanditPolicy` is stationary, while the simulated world is not (habituation). A discounted or
  sliding-window bandit would be a stronger, fairer baseline for the LLM. Worth adding before E4 is final.
- `OllamaClient` sends no `seed`, so LLM runs are not exactly repeatable. E4 keeps each finished run.
- One E4 repeat on 3b (Terminal 4, commit f9ce1d4, 364 decisions, simulated): 0 failed calls, 0 out of enum, but the
  validator replaced an already-tried sound in 79 decisions: 16/123 at attempt 2, 63/87 at attempt 3. The prompt
  already warns against this, and `validate` then plays `candidates[0]`, so those picks are not the LLM's. The bandit
  never proposes a tried sound. Building the enum from `allowed - tried_this_event` would remove them. It changes the
  decision code, so E4's LLM runs would have to be redone: decide before the full run. Log:
  `results/e4_llm_runs/19084a00c2752883/llm_r0.decisions.jsonl` on the lead laptop (not committed).

## For Terminal 4 (experiments)
- Species cooldown (`trigger.species_cooldown_s`, 60 s) means fewer events and alarms per clip, so E1 and E5
  need more clips for the same sample size. Set it to 0 only to measure fragmentation, and say so.
  Numbers from before commit a0eb2e0 are not comparable. `_tracker_status` now sets `p.in_zone`.
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
- From 2026-11-03: `build_event_images` defaults to 4 images at 448 px, so E2's burst condition is smaller than
  before, and few-shot needs a larger `agent.ollama.num_ctx`. Default model is 3b, while `paper/main.tex` still
  names `ollama-qwen2.5vl-7b` tables. The LLM alert prompt and policy prompt changed: older E4 and E6 LLM runs are
  not comparable. The policy prompt no longer states habituation or eating/leaving advice. The bandit is still
  stationary. Alerts that fail the fact check have source `template_ungrounded`, which E6 counts as
  `template_fallback`. Failed VLM calls: `vlm` in `farmguard run` output, `vlm_failed` and
  `attempts[].vlm_error` in events.jsonl.
- E2: few-shot is `cfg.raw["agent"]["fewshot"]["enabled"]`, read on each `check()`, so flip it per condition like
  the location flag. Frames go in `data/fewshot/<species key|none|other_animal>/` (none are collected yet), kept out
  of the test clips. About 21 extra images per call at 3 per category, so report latency too. Labels name species,
  which hurt behaviour prompts in Dussert 2025: report behaviour accuracy with and without.
- E2 methods: Anthropic runs at its default sampling (`temperature` is rejected on API version 2026-10-21), while
  other providers use `agent.temperature`. Say so when comparing models.
- E4: `LLMPolicy.source` is now `llm_fallback_bandit` when the LLM failed, so count those apart from `llm`. To
  measure invalid actions in the simulator, set `pol.decision_log = ...` on the policy in `run_all`, then call
  `policy.decision_log_summary(path)`. `invalid_action_rate` includes in-enum picks the validator changed.
  (The first-sound prompt no longer says no animal is present: that line was removed on 2026-11-03.)
- Cost per event is not logged yet (`llm_call` has latency only). Tell Terminal 3 if the paper needs token counts.
- **From 2026-11-11, ready for E4:** E4 LLM runs from before this commit are not comparable (enum, escalation
  rule and seed changed; the decision-code hash in `_llm_run_path` changes too, so cached runs are not reused).
  - `agent.policy.constrain_enum: true` (default): the enum holds only legal actions (allowed, not tried this
    event, not on cooldown, plus `escalate_farmer` when allowed). `false` gives the old enum, to reproduce your
    3b probe. Decision records log `constrain_enum`, `enum`, `seed` and `model`. With `true`, a tried sound that a
    schema-ignoring client still returns counts as `out_of_enum` and goes to the bandit fallback, so report
    `invalid_action_rate` for both modes, not only `true` (the schema makes it near 0 by construction).
  - `run_all` now calls `build_policy(..., cfg=cfg)` so the flag reaches the simulator, and `_llm_run_path` adds
    `constrain_enum` and the client's seed to the cache key, so a `false` run never reuses a `true` run's file.
  - `bandit_sw`: Thompson sampling over the newest `agent.policy.bandit_window` (20) outcomes of the species x
    time bucket, all sounds together. Use it by name. The window comes from `Memory.sound_stats(window=)`, and the
    simulator calls that with the default 20: pass `window=` there if you vary it. I added `bandit_sw` to the
    adaptive-first-sound list in `run_policy` (one-word edit in your file).
  - Escalation: the validator turns `escalate_farmer` before attempt 2 into the first legal sound (counted as
    invalid) unless the species is in `agent.policy.immediate_escalation_species` (`[elephant]`) or no legal sound
    is left. This applies to every policy, so early escalations fall for all of them.
  - `tracker_evidence(pkts, t0, gone_frames)` lost its species argument: any target animal in the zone counts.
    I updated the call in `_tracker_status` (its `species` parameter is now unused).
- From 2026-11-11, E5 and live runs: `agent.vlm.override_species: true` and `override_min_confidence: high`.
  events.jsonl and `farmguard run` results log `detector_species, vlm_species, vlm_confidence, effective_species,
  overridden, possible_false_alarm`. A VLM "none" at high confidence with no animal in the retreat check stops the
  sounds, sends a fixed low-priority note (source `false_alarm_note`) and records no outcome in memory.
- From 2026-11-13: `agent.clip.max_images` is 3 (crop + full frame + 1 neighbour), and the retreat check sends the
  same number of frames (it sent 4 before). 4 images overflowed num_ctx 4096 on Ollama (smoke test 2), so E2's burst
  condition is smaller again. `farmguard run` now honours `agent.policy.constrain_enum` (it was always true there).
  Retreat windows cut short by the end of a clip still get a verdict from the last frame: count those apart in E3/E5.
- **From 2026-11-14 (T3), E4 decision code changed:** new cache keys, so no earlier LLM run is reused. Keep 773ad66's
  constrained-enum LLM runs as the **"ordered options" ablation** (3b took the first listed sound in 75% of 4276 calls).
  - `agent.policy.shuffle_options: true` (default): the legal sounds are shuffled in the enum and in the prompt's
    per-sound history on every call, seeded by (`agent.seed`, event id, attempt). The simulator now gives each event an
    id `r<run>e<i>`, so the order is reproducible. `false` = 773ad66's order (its own cache key).
  - Decision records log `shown_order` and `chosen_position` (0 = first shown). `summarize()` has
    `first_position_rate` (in-enum LLM answers that took the first option; `None` for non-LLM policies), and so does
    `decision_log_summary()`. With one legal sound the first position is forced, so compare against 1/len(enum).
  - New policy `llm_bandit`: the llm prompt plus `posterior_mean` = Beta(1+succ, 1+fail) mean per legal sound, same
    enum, shuffle, validator and bandit fallback. Run it by name like `llm`; it caches under its own name.
  - E3/E5 on clips: a retreat window with fewer than `agent.retreat.min_frames` (10) frames because the clip ended has
    status `truncated` (events.jsonl `attempts[].status`, `agent.truncated_windows` in `farmguard run` output). It
    writes nothing to memory. **Exclude truncated windows from E3/E5 or report them apart.**
  - E2: events.jsonl has `effective_by_threshold` = the species used at override off (`none`), `high` and `medium`,
    so the threshold can be studied without rerunning. `species_uncertain` marks detector/VLM disagreement involving
    elephant (no immediate escalation on the detector alone; the alert names both).
