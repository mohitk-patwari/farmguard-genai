"""Deterrence simulator for experiment E4.

Recorded clips cannot react to a sound, so we simulate the animal's response while running the
SAME policy, validator and memory code as the real system. Every policy sees identical event
streams (same seed), which makes the comparison paired.

Model (all parameters in config/sim.yaml, all are assumptions to be varied in a sensitivity study):
  p_flee = p0[species][sound] * exp(-h * exposures[species][sound])
           * (eating_penalty if eating) * (night_penalty if night) * group penalty
  exposures decay with time constant `recovery_hours` (habituation recovers).
"""
from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from farmguard.agent.memory import Memory
from farmguard.agent.policy import Policy, PolicyContext, build_policy
from farmguard.agent.validators import ESCALATE
from farmguard.config import Config


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
            events.append(SimEvent(night, night * 24 + hour + rng.random(),
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
    m = sim["modifiers"]
    p = world.p0[ev.species].get(sound, 0.0)
    p *= math.exp(-sim["habituation"]["strength"] *
                  world.exposure(ev.species, sound, ev.t_hours, sim["habituation"]["recovery_hours"]))
    if ev.eating:
        p *= m["eating_penalty"]
    if ev.bucket == "night":
        p *= m["night_penalty"]
    p *= max(0.0, 1 - m["group_penalty_per_extra"] * (ev.count - 1))
    return max(0.0, min(1.0, p))


def run_policy(cfg: Config, sim: dict[str, Any], policy: Policy, events: list[SimEvent],
               world_seed: int, outcome_seed: int, adaptive_first: bool = True) -> list[SimOutcome]:
    """adaptive_first: the next event's FIRST alarm uses the policy's cached recommendation
    (computed off the critical path after each event). Fixed/rule_chain ignore it."""
    world = make_world(cfg, sim, random.Random(world_seed))
    rng = random.Random(outcome_seed)
    memory = Memory(":memory:")
    first_cache: dict[str, str] = {}
    tm = sim["timing"]
    tau = sim["habituation"]["recovery_hours"]
    out: list[SimOutcome] = []
    for i, ev in enumerate(events):
        info = cfg.species[ev.species]
        first = first_cache.get(ev.species, info.default_sound) if adaptive_first else info.default_sound
        sounds, success, escalated = [first], False, False
        elapsed = 0.0
        while True:
            snd = sounds[-1]
            p = flee_probability(world, sim, ev, snd)
            world.expose(ev.species, snd, ev.t_hours, tau)
            flees = rng.random() < p
            memory.add_attempt(f"e{i}", len(sounds), ev.species, snd, policy.name, "",
                               "gone" if flees else "still_present", None, bucket=ev.bucket,
                               wall_time=ev.t_hours * 3600)
            if flees:
                success = True
                elapsed += min(tm["observe_seconds"], rng.lognormvariate(math.log(tm["leave_time_mean_s"]), 0.4))
                break
            elapsed += tm["observe_seconds"]
            ctx = PolicyContext(species=ev.species, species_label=ev.species.replace("_", " "),
                                allowed=info.allowed_sounds, default_sound=info.default_sound,
                                tried_this_event=list(sounds), attempt_no=len(sounds),
                                max_attempts=tm["max_attempts"], hour_bucket=ev.bucket,
                                behaviour="eating" if ev.eating else "moving_through", count=ev.count,
                                stats=memory.sound_stats(ev.species, ev.bucket), now=elapsed,
                                sound_cooldown_s=0)
            d = policy.decide(ctx)
            if d.action == ESCALATE:
                escalated = True
                elapsed += tm["farmer_arrival_s"]
                break
            sounds.append(d.action)
        out.append(SimOutcome(policy.name, ev.night, ev.species, sounds, success, len(sounds),
                              escalated, elapsed))
        if adaptive_first and policy.name in ("bandit", "llm", "random"):
            ctx0 = PolicyContext(species=ev.species, species_label=ev.species.replace("_", " "),
                                 allowed=info.allowed_sounds, default_sound=info.default_sound,
                                 tried_this_event=[], attempt_no=0, max_attempts=tm["max_attempts"],
                                 hour_bucket=ev.bucket, behaviour="unclear",
                                 stats=memory.sound_stats(ev.species, ev.bucket), sound_cooldown_s=0)
            rec = policy.decide(ctx0).action
            if rec != ESCALATE:
                first_cache[ev.species] = rec
    return out


def summarize(outcomes: list[SimOutcome]) -> dict[str, dict[str, float]]:
    by: dict[str, list[SimOutcome]] = {}
    for o in outcomes:
        by.setdefault(o.policy, []).append(o)
    res = {}
    for name, os_ in by.items():
        n = len(os_)
        half = [o for o in os_ if o.night >= max(o.night for o in os_) / 2]
        res[name] = {
            "events": n,
            "success_rate": sum(o.success for o in os_) / n,
            "first_sound_success": sum(o.success and o.attempts == 1 for o in os_) / n,
            "mean_attempts": sum(o.attempts for o in os_) / n,
            "escalation_rate": sum(o.escalated for o in os_) / n,
            "mean_time_in_field_s": sum(o.time_in_field_s for o in os_) / n,
            "late_success_rate": sum(o.success for o in half) / max(1, len(half)),
        }
    return res


def save_outcomes(outcomes: list[SimOutcome], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["policy", "night", "species", "sounds", "success", "attempts", "escalated",
                    "time_in_field_s"])
        for o in outcomes:
            w.writerow([o.policy, o.night, o.species, "|".join(o.sounds), int(o.success), o.attempts,
                        int(o.escalated), round(o.time_in_field_s, 1)])


def run_all(cfg: Config, sim: dict[str, Any], policies: list[str], llm_client=None,
            repeats: int = 1) -> list[SimOutcome]:
    outcomes: list[SimOutcome] = []
    for r in range(repeats):
        seed = sim["seed"] + 1000 * r
        events = make_events(sim, random.Random(seed))
        for name in policies:
            pol = build_policy(name, llm_client, seed=seed)
            outcomes += run_policy(cfg, sim, pol, events, world_seed=seed + 1, outcome_seed=seed + 2)
    return outcomes
