# Results log

Every non-mock `python -m farmguard.eval` run appends a row below (date, commit, model, command,
numbers). Full results are in `results/` (not committed). `+dirty` means uncommitted code changes:
re-run from a clean commit before quoting such a number.

E4 numbers come from a simulator: every parameter in `config/sim.yaml` is an assumption, never field
data. A row showing an `llm` fallback above 1% is the bandit's result, not the LLM's.

| Date | Commit | Model | Command | Numbers |
|---|---|---|---|---|
