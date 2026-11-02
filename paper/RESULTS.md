# Results log

Every non-mock `python -m farmguard.eval` run appends a row below (date, commit, model, command,
numbers). Full results are in `results/` (not committed). `+dirty` means uncommitted code changes:
re-run from a clean commit before quoting such a number.

E4 numbers come from a simulator: every parameter in `config/sim.yaml` is an assumption, never field
data. A row showing an `llm` fallback above 1% is the bandit's result, not the LLM's.

| Date | Commit | Model | Command | Numbers |
|---|---|---|---|---|
| 2026-11-01 | 997219f | - | `python -m farmguard.eval e4 --policies fixed,random,rule_chain,bandit --repeats 5` | fixed: 0.132 succ, 814s; random: 0.529 succ, 450s; rule_chain: 0.421 succ, 551s; bandit: 0.531 succ, 448s; |
| 2026-11-01 | 997219f | - | `python -m farmguard.eval e4 --policies fixed,random,rule_chain,bandit --repeats 5` | fixed: 0.241 succ, 715s; random: 0.583 succ, 401s; rule_chain: 0.528 succ, 451s; bandit: 0.594 succ, 390s; |
| 2026-11-01 | 997219f | - | `python -m farmguard.eval e4-grid --policies fixed,random,rule_chain,bandit --repeats 30` | 36 cells x 4 policies x 30 repeats: see e4_grid.csv |
| 2026-11-01 | fedc35a | - | `python -m farmguard.eval e4 --policies fixed,random,rule_chain,bandit --repeats 5` | SIMULATED, random: fixed: 0.132 succ, 814s; random: 0.529 succ, 450s; rule_chain: 0.421 succ, 551s; bandit: 0.531 succ, 448s; |
| 2026-11-01 | fedc35a | - | `python -m farmguard.eval e4 --policies fixed,random,rule_chain,bandit --repeats 5` | SIMULATED, informed: fixed: 0.241 succ, 715s; random: 0.583 succ, 401s; rule_chain: 0.528 succ, 451s; bandit: 0.594 succ, 390s; |
| 2026-11-01 | fedc35a | - | `python -m farmguard.eval e4-grid --policies fixed,random,rule_chain,bandit --repeats 30` | 36 cells x 4 policies x 30 repeats: see e4_grid.csv |
| 2026-11-03 | f9ce1d4 | qwen2.5vl:3b | `python -m farmguard.eval e4 --provider ollama --llm-model qwen2.5vl:3b --policies fixed,random,rule_chain,bandit,llm --mode random --repeats 1 --results-dir results/probe_3b` | SIMULATED, random: fixed: 0.117 succ, 829s; random: 0.500 succ, 477s; rule_chain: 0.429 succ, 544s; bandit: 0.571 succ, 411s; llm: 0.442 succ, 527s, fallback 0.0%, invalid 21.7%; TIMING PROBE, one world, not an E4 result: 1159 s wall, 364 decisions, median 3.09 s per call, all invalid = validator replaced an already-tried sound (79), out of enum 0 |
