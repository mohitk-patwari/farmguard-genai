# Research notes (the knowledge behind the code)

Compiled Oct 2026 from: 333 GitHub repositories (cloned and read, not just READMEs) and 11 papers /
studies. Line numbers refer to the default branch at that time; re-check before quoting in the paper.

## 1. Landscape in one paragraph
About 100 of the 333 repos are on-topic (farm intrusion, deterrent, alert); 21 of those are empty.
The typical repo is a single loop: YOLOv8 detects, one fixed sound or buzzer plays, a Twilio SMS
or email goes out, behind a fixed cooldown. Published systems are the same with better hardware.
**No repo or paper verifies with vision that the animal left after the deterrent, adapts the next
deterrent from outcomes, or uses an LLM to make that decision.** That is FarmGuard's contribution.

Feature counts over the 79 on-topic repos that contain code (regex scan, vendored libs excluded):

| Feature | Repos | Feature | Repos |
|---|---|---|---|
| Ultralytics YOLOv8/11/12 | 25 | Twilio SMS | 14 |
| Keras/CNN classifier | 19 | Email | 9 |
| Recorded sound playback | 23 | Firebase/app push | 8 |
| Buzzer | 14 | WhatsApp | 7 |
| Tracking (DeepSORT/ByteTrack) | 7 | Voice call | 4 |
| Escalation by distance/risk | 3 | Local-language alert | 2 (UI strings only) |
| Rotates deterrents | 1 | LLM/VLM in the loop | 2 |
| **Checks the animal left** | **0** | **Logs detection-to-alarm latency** | **0** |

## 2. Repos worth borrowing from (what, where, caveat)

| Repo | Borrowed into | Lines / idea | Caveat |
|---|---|---|---|
| gotogrub/OpenCV-Animal-Detector | fast/zones.py, fast/trigger.py, notifier | `src/core/zone_manager.py` 31-103 zone ENTRY/LOITER/EXIT; `alert_manager.py` 35-50 cooldown key zone:species; `detector.py` 16-27 COCO animal ids, 73-84 `.track(persist=True, tracker="bytetrack.yaml")` | sound + Telegram run in the frame loop (blocking) |
| cyriacjose1/athiru | fast/video_source.py, fast/detector.py | `detector/detect.py` 17 YOLO-World `set_classes(["elephant","tiger","wild boar","leopard"])`; `video/stream.py` 169, 183-184 capture/inference threads with `Queue(maxsize=1)` | conf 0.15; one alert per session |
| Avinash-builds/...Adaptive-Risk-Alert | config/species.yaml idea | `src/risk/risk_engine.py` 150-196 risk = a·S·psi(b)+b·(1-d)·v+g·log2(n+1)+d·T_night, sigmoid; `behavior/classifier.py` ROAMING/APPROACHING/AGGRESSIVE | `src/main.py` 294-295 hard-code distance=500, velocity=0 (formula never exercised); SMS inside frame loop |
| wadas-project/WADAS (49 stars) | actuator idea (future hardware) | `wadas/ai/pipeline.py` MegaDetector V5/V6 (OpenVINO) + DeepFaune; `domain/deterrent_actuator.py` 32-39 command enum; `actuator_server_app.py` 42,57 pull + ack | deterrent has only ON |
| subith2005-stack/wild-animal-intrusion-ai | fast/trigger.py | `app.py` 16-17, 126-131 alarm only if 4 of last 7 predictions agree | |
| SentaFito53/...coral-edgeTPU | fast/alarm.py | gunshot played in its own thread with cooldown | |
| adamp87/pigeon | fast/timing.py | `src/pigeon/pipeline.py` 36-130 per-stage timing class | |
| noelbase/Wildlife-Intrusion-Detection-System | hardware future work | `firmware/esp32_border_node.ino` 60, 102 ESP32 deep sleep woken by PIR | |
| DMWeiy/...projectile-deterrent | trigger idea | `Esp32Cam.py` 10-hit debounce, serial 'a'/'n' | typos; copy the idea only |
| Devashish-Modi/TETRA040 | species.yaml, policy context | `model2_decision_engine.py` 194-258 escalation by distance 10/5/2 m; `deterrent_priority` table (animal, weather, period) -> lvl1/2/3 devices; Hindi/Gujarati strings | static lookup, never learns |
| Anbuchellian/AEGISFARM | policy.RuleChainPolicy, memory.py | `response_engine.py` 12-38 approved strategies + fixed retry A->B->C->farmer; `memory.py` success rate per species/zone/strategy | `observation.py` 11 `verify_retreat()` returns "retreated" unless a test flag is set; `detection.py` 36,52 hard-code zone/direction. **Use as baseline, cite as stub** |
| device-kunkun/seedling-agent-orchestrator | agent/policy.py, validators.py, prompts.py | `app/decision_engine.py` 26-56 one tool, `method_name` enum = allow-list; 286-340 fallbacks; `validators.py` clamp args, cooldown, mutex | uses `eval` in validators (we don't) |
| NblScript/muye | agent design, llm/base.py | YOLO -> RAG (Chroma) -> LLM decision -> compliance check -> act -> re-inspect -> retry; `ai_decision.py` 571 repair call, 718 `_merge_with_fallback` | pest spraying domain; closed loop maps 1:1 |
| 962753401-cloud/PharmaWatch | two-stage cost design | local YOLO-World triggers, only flagged clips to Qwen-VL; README: >90% API cost cut | offline batch |
| tanushchauhan/Shoo | agent/vlm_check.py | `server/ai_pipeline.py` 24-35 JSON prompt (species, count, confidence, behaviour), strip ``` fences | VLM on every image, 300 s timeout: **too slow to scare anything**; HC-SR04 is a distance sensor, not a repellent |
| edycutjong/scarecrow | future: NL rules, known-entity memory | `src/core/rules.ts` plain-English rules judged by a local LLM with keyword fallback; `memory.ts` RAG of known pets | TypeScript |
| unclemattmakes/backyard-critter-cam | future CLIP gate | `clipfilter.py` 45-56, 95-115 animal vs non-animal prompt prototypes (mean of normalised text embeddings); MOG2 motion gate; behaviour time windows | AGPL-3.0: rewrite, don't copy |
| Igkerdouki/wildsafe-ml-service | eval ideas | `app/inference.py` 25-28 CLIP ViT-B/32 ONNX (Xenova) w/o torch; MegaDetector+CLIP eval scripts | ~500 ms/frame on M4 Pro |
| AlexKapy/camera-trap-ai-pipeline | prompts.py, clip_builder.py, eval/metrics.py | MegaDetector MDV5A 95% recall then Claude species ID. Location hint -> ~10% acc ("possum" 123x vs 3 true). Crop 65.0% -> +full frame 74.4% -> +burst 80.9% committed acc (80-image dev); full run Sonnet+burst 84.8% committed, 62/174 abstentions | research tool |
| google/cameratrapai (SpeciesNet) | baseline | MegaDetector + EfficientNetV2-M, 2000+ labels, `--country IND` geofence | |
| microsoft/Pytorch-Wildlife, MegaDetector | baseline | `pw_detection.MegaDetectorV6(version="MDV6-yolov10-e")` | animal/person/vehicle only |
| amathislab/wildclip | related work | CLIP fine-tuned on Serengeti captions; text-to-event retrieval (CVPR CV4Animals 2023) | |
| SanderGi/YADES | related work (edge) | animal/non-animal YOLOv8 classifier pruned+quantised to 0.6 MB at 99.5% test acc | |
| oahednoorforhad/No-Training...VLM | E2 baseline idea | GroundingDINO "animal" box -> LLaVA 1.5 4-bit one-word species, once every 5 s | slow |

**Do not adopt:** Rking20051806/SmartVision (maps humans and animals to shock currents up to
4000 uA); Abishekvit/AniREP (asks Gemini to invent an ultrasonic frequency).

## 3. Papers (numbers read from the papers / official summaries)
| Work | What | Numbers | Gap |
|---|---|---|---|
| Hybrid YOLOv5-SSD, durian plantations, arXiv:2511.00777 (2025) | elephant/boar/monkey, tiger-roar sound, Telegram | ~60 train / 10 test images per class; mAP 0.965/0.948/0.831; 20.8-24.3 s per inference | tiny data, one sound, slow |
| Fog-based intrusion detection, arXiv:2308.06614 (2023) | PIR+camera over LoRa, CNN on fog server, 18 simulated movements | 94.44% location prediction; MobileNet 1.64 s; **19.11-28.61 s** to farmer alert | no deterrent |
| Elemantra, arXiv:2310.15012 (2023) | YOLOv7-tiny IR + seismic, elephants | 537 IR images, AP50 0.675 TFLite vs 0.895 PyTorch; seismic recall 0.82 | **randomises bee sounds by rule** (speed change, pink noise, silences): not outcome-driven |
| CST Bhutan AI intrusion system | YOLOv8 on Pi 4, app/SMS/voice | 30,800 images, 7 species, 95.7% lab accuracy | lab only |
| ZHAW wild boar acoustic study | boar alarm calls at random intervals, 36 GPS-collared boars | damage reduced; "no single preventive measure" fully protects | habituation real |
| Conservation Evidence: random loud noise for primates | 2 studies | orangutan feeding 81 -> 69 min | weak evidence base |
| Dussert et al., Methods in Ecology and Evolution 2025 | zero-shot behaviour (eating/resting/moving): CLIP, SigLIP, PaliGemma, CogVLM | CogVLM 96%; >90% activity-pattern overlap; prompt wording matters; **species names in the prompt hurt** | camera-trap research only |
| Alencar, Cunha, dos Santos, JBCS 2026 | BLIP/CLIP/Gemini/GPT in camera-trap workflow | BLIP empty filter 7.61% zero-shot vs 91.0% few-shot; Gemini species 75.89% zero-shot; behaviour from sequences best (BLIP 75.57%) | offline |
| Stable Diffusion for augmentation, arXiv:2312.03996 | img2img, DreamBooth, ControlNet on COCO + weeds | gains on some classes only | |
| Denoised diffusion object-focused augmentation, arXiv:2510.08955 (2025) | SAM cut-outs + DDPM composites, drone cattle | F1 0.64 vs 0.49 Faster R-CNN baseline | |
| Li et al., Frontiers in Plant Science 2026 (LLMs in plant protection) | perspective | domain multi-agent 87% -> 93% | LLMs should advise within rules |
Not yet read in full (check before citing): Microsoft zero-shot camera-trap paper (site blocked
automated reading), UoM sound-based repel, SLIIT image-processing repel, Nepal macaque, stray-animal
tracking, IoT/AI HWC review (ScienceDirect S2772375525000620).

## 4. Lessons for the GenAI layer (each is implemented; keep them)
1. Keep GenAI out of the first alarm (papers take 19-28 s; Shoo's 7B VLM has a 300 s timeout).
2. Only triggered clips go to the VLM (PharmaWatch: >90% API cost cut). Report cost per event.
3. No location in prompts (AlexKapy ~10%); species names hurt behaviour prompts (Dussert).
4. Several frames, not one (AlexKapy 65.0 -> 80.9%; JBCS sequences best).
5. Fixed answer format + "unclassifiable" option; report raw accuracy AND accuracy-when-committed.
6. LLM chooses from an enum; code validates (seedling-agent).
7. Few-shot beats zero-shot for filtering (BLIP 7.61% -> 91.0%). TODO: few-shot example frames.
8. Habituation is real; current best practice is rule-based randomisation (Elemantra, ZHAW). Our
   step beyond: choose from outcomes (memory) + context (VLM behaviour), verified by a real
   retreat check.

## 5. Gaps -> claims -> experiments
| Gap | Claim | Experiment |
|---|---|---|
| Alarm speed never measured; GenAI put in the critical path | Two-speed design keeps frame-to-alarm latency unchanged with the agent on | E1 |
| VLM ability on farm footage unknown | VLM reads eating vs passing; multi-frame > single; location hint hurts | E2 |
| No outcome check anywhere | VLM+tracker fusion detects retreat better than tracker alone | E3 |
| Habituation handled by fixed rules at best | Outcome-driven policy (bandit / LLM) beats fixed, random and rule-chain in time-in-field | E4 (sim) |
| False alarms unaddressed | VLM gate reduces false alarms/hour at small recall cost | E5 |
| Alerts are generic English | LLM local-language alerts are understood better than template SMS | E6 |
