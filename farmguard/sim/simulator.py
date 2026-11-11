"""Deterrence simulator for experiment E4.

Recorded clips cannot react to a sound, so we simulate the animal's response while running the
SAME policy, validator and memory code as the real system. Every policy sees identical event
streams and identical outcome noise (common random numbers: one pre-drawn flee threshold per
event x attempt), which makes the comparison truly paired.

Model (all parameters in config/sim.yaml, all are assumptions to be varied in a sensitivity study):
  p_flee = p0[species][sound] * exp(-h * (own_exposure + cross * exposure_to_other_sounds))
           * (eating_penalty if eating) * (night_penalty if night) * group penalty
  exposures decay with time constant `recovery_hours` (habituation recovers).

Assumptions that shape the comparison (say these in the paper):
  - Retreat verification and behaviour are perfect here. Real errors are measured in E2/E3.
  - No modifier depends on which sound is played, so behaviour/time context cannot help any
    policy pick a better sound: this simulator cannot show a benefit from VLM behaviour context.
  - With cross = 0, habituation to one sound does not carry to others, which directly rewards any
    rotating policy over `fixed`. The sensitivity grid varies cross for that reason.
  - Every event that does not end in success ends in escalation, so escalation_rate is
    1 - success_rate by construction and time in field is dominated by farmer_arrival_s.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import itertools
import json
import math
import random
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from farmguard.agent.memory import Memory
from farmguard.agent.policy import LLMPolicy, Policy, PolicyContext, build_policy
from farmguard.agent.validators import ESCALATE
from farmguard.config import ROOT, Config


@dataclass
class SimEvent:
    night: int
    t_hours: float
    species: str
    eating: bool
    count: int
    bucket: str


@dataclass
class SimOutcome:
    policy: str
    night: int
    species: str
    sounds: list[str]
    success: bool
    attempts: int
    escalated: bool
    time_in_field_s: float
    run: int = 0
    event: int = 0
    early_escalation: bool = False   # gave up before max_attempts (only the LLM can do this)
    llm_calls: int = 0
    llm_fallbacks: int = 0           # decisions the bandit made because the LLM failed
    invalid_actions: int = 0         # proposals the validator or enum check had to reject


@dataclass
class World:
    p0: dict[str, dict[str, float]]
    exposures: dict[tuple[str, str], tuple[float, float]] = field(default_factory=dict)  # -> (E, t)

    def exposure(self, species: str, sound: str, t_h: float, tau: float) -> float:
        e, t_last = self.exposures.get((species, sound), (0.0, t_h))
        return e * math.exp(-(t_h - t_last) / tau)

    def expose(self, species: str, sound: str, t_h: float, tau: float) -> None:
        self.exposures[(species, sound)] = (self.exposure(species, sound, t_h, tau) + 1.0, t_h)


def load_sim(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def make_world(cfg: Config, sim: dict[str, Any], rng: random.Random) -> World:
    eff = sim["effectiveness"]
    p0: dict[str, dict[str, float]] = {}
    for sp, info in cfg.species.items():
        p0[sp] = {}
        for snd in info.allowed_sounds:
            if eff["mode"] == "informed":
                p0[sp][snd] = float(eff.get("prior", {}).get(sp, {}).get(snd, 0.25))
            else:
                p0[sp][snd] = rng.betavariate(eff["beta_a"], eff["beta_b"])
    return World(p0)


def make_events(sim: dict[str, Any], rng: random.Random) -> list[SimEvent]:
    mix = sim["species_mix"]
    names, weights = list(mix), list(mix.values())
    events: list[SimEvent] = []
    for night in range(sim["nights"]):
        n = _poisson(rng, sim["events_per_night"])
        for _ in range(n):
            hour = rng.choice([19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6])
            bucket = "night" if hour >= 21 or hour < 5 else ("evening" if hour >= 17 else "morning")
            # after-midnight hours belong to the same night, i.e. to the next calendar day
            clock = hour + 24 if hour < 12 else hour
            events.append(SimEvent(night, night * 24 + clock + rng.random(),
                                   rng.choices(names, weights)[0],
                                   rng.random() < sim["modifiers"]["p_eating"],
                                   1 + (rng.random() < 0.3) + (rng.random() < 0.1), bucket))
    events.sort(key=lambda e: e.t_hours)
    return events


def _poisson(rng: random.Random, lam: float) -> int:
    l, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= l:
            return k
        k += 1


def flee_probability(world: World, sim: dict[str, Any], ev: SimEvent, sound: str) -> float:
    m, hab = sim["modifiers"], sim["habituation"]
    tau = hab["recovery_hours"]
    p = world.p0[ev.species].get(sound, 0.0)
    exposure = world.exposure(ev.species, sound, ev.t_hours, tau)
    cross = hab.get("cross", 0.0)
    if cross:
        exposure += cross * sum(world.exposure(sp, s, ev.t_hours, tau)
                                for sp, s in world.exposures if sp == ev.species and s != sound)
    p *= math.exp(-hab["strength"] * exposure)
    if ev.eating:
        p *= m["eating_penalty"]
    if ev.bucket == "night":
        p *= m["night_penalty"]
    p *= max(0.0, 1 - m["group_penalty_per_extra"] * (ev.count - 1))
    return max(0.0, min(1.0, p))


def run_policy(cfg: Config, sim: dict[str, Any], policy: Policy, events: list[SimEvent],
               world_seed: int, outcome_seed: int, adaptive_first: bool = True,
               run: int = 0) -> list[SimOutcome]:
    """adaptive_first: the next event's FIRST alarm uses the policy's cached recommendation
    (computed off the critical path after each event). Fixed/rule_chain ignore it."""
    world = make_world(cfg, sim, random.Random(world_seed))
    memory = Memory(":memory:")
    first_cache: dict[str, str] = {}
    tm = sim["timing"]
    tau = sim["habituation"]["recovery_hours"]
    sigma = tm.get("leave_time_sigma", 0.4)
    mu = math.log(tm["leave_time_mean_s"]) - sigma ** 2 / 2  # so the lognormal's MEAN is the config value
    rng = random.Random(outcome_seed)
    # Common random numbers: drawn up front, so a policy's choices cannot shift the noise that
    # another policy sees. Attempt k of event i meets the same threshold under every policy.
    draws = [[(rng.random(), rng.lognormvariate(mu, sigma)) for _ in range(tm["max_attempts"])]
             for _ in events]
    out: list[SimOutcome] = []
    # same knobs as the live AgentWorker (_stats, _immediate), so E4 baselines see the same context
    window = cfg.get("agent.policy.bandit_window", 20)
    immediate = tuple(cfg.get("agent.policy.immediate_escalation_species", ["elephant"]) or ())

    def decide(ctx: PolicyContext, row: dict[str, int]):
        d = policy.decide(ctx)
        if isinstance(policy, LLMPolicy):
            row["llm_calls"] += 1
            row["llm_fallbacks"] += d.source.startswith("llm_fallback")
        row["invalid_actions"] += "bad action" in d.reason or bool(d.warnings)
        return d

    for i, ev in enumerate(events):
        info = cfg.species[ev.species]
        first = first_cache.get(ev.species, info.default_sound) if adaptive_first else info.default_sound
        sounds, success, escalated, early = [first], False, False, False
        elapsed = 0.0
        row = {"llm_calls": 0, "llm_fallbacks": 0, "invalid_actions": 0}
        while True:
            snd = sounds[-1]
            p = flee_probability(world, sim, ev, snd)
            world.expose(ev.species, snd, ev.t_hours, tau)
            u, leave_time = draws[i][len(sounds) - 1]
            flees = u < p
            memory.add_attempt(f"e{i}", len(sounds), ev.species, snd, policy.name, "",
                               "gone" if flees else "still_present", None, bucket=ev.bucket,
                               wall_time=ev.t_hours * 3600)
            if flees:
                success = True
                elapsed += min(tm["observe_seconds"], leave_time)
                break
            elapsed += tm["observe_seconds"]
            if len(sounds) >= tm["max_attempts"]:  # the validator would force this; skip the LLM call
                escalated = True
                elapsed += tm["farmer_arrival_s"]
                break
            ctx = PolicyContext(species=ev.species, species_label=ev.species.replace("_", " "),
                                allowed=info.allowed_sounds, default_sound=info.default_sound,
                                tried_this_event=list(sounds), attempt_no=len(sounds),
                                max_attempts=tm["max_attempts"], hour_bucket=ev.bucket,
                                behaviour="eating" if ev.eating else "moving_through", count=ev.count,
                                stats=memory.sound_stats(ev.species, ev.bucket, window=window), now=elapsed,
                                sound_cooldown_s=0, immediate_escalation_species=immediate)
            d = decide(ctx, row)
            if d.action == ESCALATE:
                escalated, early = True, True
                elapsed += tm["farmer_arrival_s"]
                break
            sounds.append(d.action)
        if adaptive_first and policy.name not in ("fixed", "rule_chain"):  # e.g. bandit_sw adapts too
            ctx0 = PolicyContext(species=ev.species, species_label=ev.species.replace("_", " "),
                                 allowed=info.allowed_sounds, default_sound=info.default_sound,
                                 tried_this_event=[], attempt_no=0, max_attempts=tm["max_attempts"],
                                 hour_bucket=ev.bucket, behaviour="unclear",
                                 stats=memory.sound_stats(ev.species, ev.bucket, window=window), sound_cooldown_s=0,
                                 immediate_escalation_species=immediate)
            rec = decide(ctx0, row).action
            if rec != ESCALATE:
                first_cache[ev.species] = rec
        out.append(SimOutcome(policy.name, ev.night, ev.species, sounds, success, len(sounds),
                              escalated, elapsed, run=run, event=i, early_escalation=early, **row))
    memory.conn.close()
    return out


def summarize(outcomes: list[SimOutcome]) -> dict[str, dict[str, float]]:
    by: dict[str, list[SimOutcome]] = {}
    for o in outcomes:
        by.setdefault(o.policy, []).append(o)
    res = {}
    for name, os_ in by.items():
        n = len(os_)
        mid = max(o.night for o in os_) / 2
        half = [o for o in os_ if o.night >= mid]
        runs: dict[int, list[bool]] = {}
        for o in os_:
            runs.setdefault(o.run, []).append(o.success)
        run_rates = [sum(v) / len(v) for v in runs.values()]
        calls = sum(o.llm_calls for o in os_)
        res[name] = {
            "events": n,
            "runs": len(runs),
            "success_rate": sum(o.success for o in os_) / n,
            "success_rate_run_sd": statistics.stdev(run_rates) if len(run_rates) > 1 else 0.0,
            "first_sound_success": sum(o.success and o.attempts == 1 for o in os_) / n,
            "mean_attempts": sum(o.attempts for o in os_) / n,
            "escalation_rate": sum(o.escalated for o in os_) / n,
            "early_escalation_rate": sum(o.early_escalation for o in os_) / n,
            "mean_time_in_field_s": sum(o.time_in_field_s for o in os_) / n,
            "late_success_rate": sum(o.success for o in half) / max(1, len(half)),
            "llm_calls_per_event": calls / n,
            "llm_fallback_rate": sum(o.llm_fallbacks for o in os_) / calls if calls else 0.0,
            "invalid_action_rate": sum(o.invalid_actions for o in os_) / calls if calls else 0.0,
        }
    return res


def learning_curve(outcomes: list[SimOutcome], bin_nights: int = 5) -> list[dict[str, Any]]:
    """Success rate per block of nights: mean and standard error across independent runs."""
    cells: dict[tuple[str, int, int], list[bool]] = {}
    for o in outcomes:
        cells.setdefault((o.policy, o.night // bin_nights, o.run), []).append(o.success)
    per_bin: dict[tuple[str, int], list[float]] = {}
    for (pol, b, _), v in sorted(cells.items()):
        per_bin.setdefault((pol, b), []).append(sum(v) / len(v))
    return [{"policy": pol, "night_start": b * bin_nights, "success_rate": statistics.fmean(v),
             "se": statistics.stdev(v) / math.sqrt(len(v)) if len(v) > 1 else 0.0, "runs": len(v)}
            for (pol, b), v in per_bin.items()]


def save_outcomes(outcomes: list[SimOutcome], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "event", "policy", "night", "species", "sounds", "success", "attempts",
                    "escalated", "early_escalation", "time_in_field_s", "llm_calls", "llm_fallbacks",
                    "invalid_actions"])
        for o in outcomes:
            w.writerow([o.run, o.event, o.policy, o.night, o.species, "|".join(o.sounds), int(o.success),
                        o.attempts, int(o.escalated), int(o.early_escalation),
                        round(o.time_in_field_s, 1), o.llm_calls, o.llm_fallbacks, o.invalid_actions])


# LLM runs take hours on a laptop, so each finished run is kept. The key covers everything that
# could change the answer: scenario, model and the decision code, so a stale file is never reused.
_DECISION_CODE = ["farmguard/sim/simulator.py", "farmguard/agent/policy.py", "farmguard/agent/prompts.py",
                  "farmguard/agent/validators.py", "farmguard/agent/memory.py", "config/species.yaml"]


def _llm_run_path(cache_dir: Path, sim: dict[str, Any], client: Any, run: int,
                  constrain_enum: bool = True, name: str = "llm") -> Path:
    h = hashlib.sha1(json.dumps({k: v for k, v in sim.items() if k != "sensitivity"},
                                sort_keys=True).encode())
    h.update(f"{client.name}:{client.model}:{client.temperature}:{getattr(client, 'seed', None)}".encode())
    h.update(f"constrain_enum={bool(constrain_enum)}".encode())
    for rel in _DECISION_CODE:
        h.update((ROOT / rel).read_bytes().replace(b"\r\n", b"\n"))
    return cache_dir / h.hexdigest()[:16] / f"{name}_r{run}.json"


def run_all(cfg: Config, sim: dict[str, Any], policies: list[str], llm_client=None,
            repeats: int = 1, cache_dir: str | Path | None = None) -> list[SimOutcome]:
    """Every repeat is an independent world + event stream; within it all policies are paired."""
    for name in policies:  # any name build_policy knows; a typo fails here, not hours into an LLM run
        build_policy(name, llm_client)
    outcomes: list[SimOutcome] = []
    for r in range(repeats):
        seed = sim["seed"] + 1000 * r
        events = make_events(sim, random.Random(seed))
        for name in policies:
            pol = build_policy(name, llm_client, seed=seed, cfg=cfg)
            cached = None
            if isinstance(pol, LLMPolicy) and cache_dir is not None:
                cached = _llm_run_path(Path(cache_dir), sim, llm_client, r, pol.constrain_enum, name)
                if cached.exists():
                    outcomes += [SimOutcome(**o) for o in json.loads(cached.read_text(encoding="utf-8"))]
                    continue
            if cached is not None:  # raw answers and per-call latency, for audit and invalid-action counts
                cached.parent.mkdir(parents=True, exist_ok=True)
                pol.decision_log = cached.with_suffix(".decisions.jsonl")
                pol.decision_log.unlink(missing_ok=True)  # left by an interrupted run
            res = run_policy(cfg, sim, pol, events, world_seed=seed + 1, outcome_seed=seed + 2, run=r)
            if cached is not None:
                cached.write_text(json.dumps([asdict(o) for o in res]), encoding="utf-8")
            outcomes += res
    return outcomes


def grid_cells(sim: dict[str, Any]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Full factorial from `sensitivity` in sim.yaml. Returns (factor levels, scenario) pairs."""
    grid = sim["sensitivity"]
    names = list(grid)
    cells = []
    for values in itertools.product(*(grid[n] for n in names)):
        levels = dict(zip(names, values))
        s = copy.deepcopy(sim)
        s["habituation"]["strength"] = levels["habituation_strength"]
        s["habituation"]["cross"] = levels["cross_habituation"]
        s["effectiveness"]["mode"] = levels["effectiveness"]
        s["modifiers"]["eating_penalty"] = levels["eating_penalty"]
        cells.append((levels, s))
    return cells
