# Paper plan

## Working title
"Scare First, Reason Later: A Two-Speed Generative-AI Agent for Closed-Loop Crop-Raiding Animal
Deterrence from Farm Video"

## One-sentence contribution
A system where a fast detector signals the animal immediately and a VLM + LLM agent then verifies
whether it left, adapts the next deterrent from outcomes under a safety allow-list, and explains the
result to the farmer in their language, without slowing the alarm.

## Contributions (keep them falsifiable)
1. **Architecture.** A two-speed design that keeps GenAI off the alarm path (E1 shows latency
   unchanged with the agent on).
2. **Retreat verification.** The first vision-based check that the animal actually left
   (tracker + VLM fusion), evaluated against hand labels (E3). Prior "verification" (AEGISFARM) is a stub.
3. **Outcome-driven deterrence.** An LLM policy constrained to an enum allow-list + validator,
   compared with fixed, random (Elemantra-style), rule-chain (AEGISFARM-style) and Thompson-sampling
   bandit baselines on identical event streams (E4, simulated, with sensitivity analysis).
4. **Prompting evidence on farm footage.** Multi-frame vs single-frame, location-hint ablation,
   local vs cloud VLM (E2), extending camera-trap findings (Dussert 2025; Alencar 2026) to farms.
5. **Farmer communication.** Local-language LLM alerts vs template SMS (E6) + a released labelled
   clip set and code.

## Be honest about likely results
- The bandit may match or beat the LLM policy in pure success rate in the "random" simulator
  setting (no prior knowledge to exploit). That is still publishable: report it, then show where
  the LLM helps (uses VLM behaviour context, "informed" setting, explanations, cold start with
  zero history). Never tune the simulator to make the LLM win.
- The simulator is an assumption model. Say so in the abstract-level text, run a sensitivity
  analysis (habituation strength 0.1/0.35/0.6; effectiveness random vs informed; eating penalty),
  and list a field trial as future work.

## Experiments
| ID | Question | Setup | Metric / test |
|---|---|---|---|
| E1 | Does the agent delay the alarm? | Same clips, `run --no-agent` vs agent on (local Ollama, under load) | read->alarm-start ms: median, p95; Mann-Whitney U |
| E2 | Can a VLM read species + behaviour? | `eval e2`: modes crop / crop+full / crop+full+burst x location hint on/off; Qwen2.5-VL (local) vs Gemini/Claude | species & behaviour accuracy, macro-F1, abstention, accuracy-when-committed; McNemar between modes |
| E3 | Can we tell if it left? | `eval e3` on labelled post-sound windows; tracker-only vs VLM-only vs fused | left-vs-stayed accuracy, F1 |
| E4 | Does adaptation help? | `simulate --policies fixed,random,rule_chain,bandit,llm --repeats 5`, 60 nights; sensitivity grid | success rate (Wilson CI), mean time in field, escalation rate, learning curve; paired McNemar vs fixed |
| E5 | Fewer false alarms? | clips with labelled intrusion intervals; fast path alone vs + VLM gate | false alarms / hour, true events rejected |
| E6 | Better farmer alerts? | `eval e6` rating sheet, blind, 10+ raters (farmers if possible) | understood %, facts correct %, clarity 1-5; Wilcoxon |

Also report: VLM latency and cost per event (local vs cloud), number of LLM calls per event, and
invalid-action rate before the validator (target 0% after).

## Data
- Real: Roboflow Universe wild boar / monkey sets (check licences), LILA camera-trap sets, own
  phone/CCTV recordings at a farm (with permission). Split by clip. Two labellers + Cohen's kappa.
- Synthetic: Stable Diffusion img2img/ControlNet night and nilgai frames for robustness only;
  results on synthetic data are always reported separately from real data.

## Paper outline (8 pages typical)
1. Introduction: crop raiding, the "scare first" requirement, contributions.
2. Related work: intrusion systems (durian, fog, Elemantra, Bhutan, WADAS), deterrent habituation
   (Elemantra, ZHAW), VLMs for animals (Dussert, Alencar, WildCLIP, SpeciesNet/MegaDetector),
   LLM agents with constrained actions (seedling-agent, muye), and an open-source landscape table
   (our 333-repo review: Figure "features present in 79 repos").
3. System: two-speed architecture figure, fast path, agent, prompts (rules R1-R6), validator.
4. Experimental setup: data, labelling, models, simulator assumptions.
5. Results: E1-E6 tables/figures.
6. Discussion: limits (simulated deterrence, small data, synthetic placeholder sounds, ethics),
   deployment path (ESP32 speakers, solar).
7. Conclusion.

## Figures to produce (scripts go in farmguard/eval or scripts/)
1. Architecture (two layers + feedback loop). 2. Sequence of one event. 3. Feature-gap bar chart
from the repo review. 4. E1 latency box plot. 5. E2 accuracy by mode/hint. 6. E4 learning curves
and time-in-field per policy. 7. E5 false alarms/hour. 8. Example farmer alerts.

## Venue types to target (check current calls and deadlines yourself)
- Workshops: CV4Animals (CVPR), Climate Change AI workshops (NeurIPS / ICLR), AI for Agriculture
  workshops at AAAI / IJCAI.
- IEEE conferences (India-friendly): INDICON, ICCCNT, CONECCT and similar.
- Journals (longer version): Computers and Electronics in Agriculture, Smart Agricultural
  Technology, Ecological Informatics.

## Ethics statement points
Humane deterrents only; no harm devices; people in frames are never sent to cloud APIs; farmer
data stays local; synthetic sounds are placeholders and are not claimed to deter.
