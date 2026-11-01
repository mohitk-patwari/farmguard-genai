# HANDOFF (shared notes between the four Claude Code terminals)

## Environment
(Terminal 1: record Python, CUDA, ultralytics, Ollama and model versions here.)

## Status log
- Base code from the research phase: pytest green (14 tests), mock demo runs end to end, simulator
  runs with fixed / random / rule_chain / bandit. Real providers and the real detector are written
  but NOT yet verified against live APIs / weights.

## For Terminal 2 (vision)

## For Terminal 3 (agent)
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

## For Terminal 1 (integration)
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

## For Terminal 4 (experiments)
