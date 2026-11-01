"""Deterrent policies. The LLM policy is the paper's contribution; the others are baselines.

  fixed       - always the species' default sound (what ~55 student repos do)
  random      - random allowed sound, never the one that just failed (Elemantra-style rule-based
                randomising; Ashil10 random sound per species)
  rule_chain  - fixed escalation chain A -> B -> C -> farmer (AEGISFARM `response_engine.py`)
  bandit      - Thompson sampling over past outcomes per species x time bucket (strong
                non-GenAI learner; the LLM must be compared against this, not only weak baselines)
  llm         - LLM with an enum-constrained action, given behaviour from the VLM, attempts so far
                and outcome memory; falls back to bandit on any failure.
All policies see the same PolicyContext and go through validators.validate().
"""
from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any

from farmguard.agent.llm.base import LLMClient
from farmguard.agent.prompts import SYSTEM_POLICY, policy_prompt, policy_schema
from farmguard.agent.validators import ESCALATE, ValidationContext, validate
from farmguard.events import PolicyDecision

log = logging.getLogger(__name__)


@dataclass
class PolicyContext:
    species: str
    species_label: str
    allowed: list[str]
    default_sound: str
    tried_this_event: list[str]
    attempt_no: int
    max_attempts: int
    hour_bucket: str
    behaviour: str = "unclear"
    count: int = 1
    stats: dict[str, dict[str, Any]] = field(default_factory=dict)
    last_played_at: dict[str, float] = field(default_factory=dict)
    now: float = 0.0
    sound_cooldown_s: float = 20.0

    def validation(self) -> ValidationContext:
        return ValidationContext(self.allowed, self.tried_this_event, self.attempt_no,
                                 self.max_attempts, self.last_played_at, self.now,
                                 self.sound_cooldown_s)


class Policy:
    name = "base"

    def propose(self, ctx: PolicyContext) -> tuple[str, str, float]:
        raise NotImplementedError

    def decide(self, ctx: PolicyContext) -> PolicyDecision:
        t0 = time.perf_counter()
        action, reason, conf = self.propose(ctx)
        final, warnings = validate(action, ctx.validation())
        if final != action:
            reason = f"{reason} [validator: {'; '.join(warnings)} -> {final}]"
        return PolicyDecision(final, reason, conf, self.name, warnings, time.perf_counter() - t0)


class FixedPolicy(Policy):
    name = "fixed"

    def propose(self, ctx):
        return ctx.default_sound, "always the default sound", 1.0

    def decide(self, ctx):  # fixed really repeats the same sound: bypass the no-repeat rule
        if ctx.attempt_no >= ctx.max_attempts:
            return PolicyDecision(ESCALATE, "max attempts", 1.0, self.name)
        return PolicyDecision(ctx.default_sound, "always the default sound", 1.0, self.name)


class RandomPolicy(Policy):
    name = "random"

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def propose(self, ctx):
        pool = [s for s in ctx.allowed if s not in ctx.tried_this_event] or ctx.allowed
        return self.rng.choice(pool), "random allowed sound", 0.5


class RuleChainPolicy(Policy):
    name = "rule_chain"

    def propose(self, ctx):
        chain = [ctx.default_sound] + [s for s in ctx.allowed if s != ctx.default_sound]
        for s in chain:
            if s not in ctx.tried_this_event:
                return s, "next sound in the fixed escalation chain", 0.8
        return ESCALATE, "chain exhausted", 1.0


class BanditPolicy(Policy):
    """Thompson sampling with Beta(1+successes, 1+failures) per sound."""
    name = "bandit"

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def propose(self, ctx):
        best, best_v = None, -1.0
        for s in ctx.allowed:
            if s in ctx.tried_this_event:
                continue
            st = ctx.stats.get(s, {})
            succ, trials = st.get("successes", 0), st.get("trials", 0)
            v = self.rng.betavariate(1 + succ, 1 + trials - succ)
            if v > best_v:
                best, best_v = s, v
        if best is None:
            return ESCALATE, "no untried sound left", 1.0
        st = ctx.stats.get(best, {})
        return best, f"Thompson sample {best_v:.2f} (succ {st.get('successes', 0)}/{st.get('trials', 0)})", best_v


class LLMPolicy(Policy):
    name = "llm"

    def __init__(self, client: LLMClient, fallback: Policy | None = None):
        self.client = client
        self.fallback = fallback or BanditPolicy()

    def context_json(self, ctx: PolicyContext) -> dict[str, Any]:
        return {
            "species": ctx.species_label, "animals": ctx.count,
            "behaviour_seen_by_camera": ctx.behaviour,
            "time_of_day": ctx.hour_bucket,
            "attempt_number": ctx.attempt_no + 1, "max_attempts": ctx.max_attempts,
            "sounds_already_tried_in_this_event_and_failed": ctx.tried_this_event,
            "allowed_actions": ctx.allowed + [ESCALATE],
            "past_results_for_this_species_and_time": {
                s: {"trials": v.get("trials", 0), "success_rate": v.get("success_rate"),
                    "recent": v.get("recent", [])}
                for s, v in ctx.stats.items() if s in ctx.allowed},
        }

    def propose(self, ctx):
        allowed = ctx.allowed + [ESCALATE]
        res = self.client.generate_json(policy_prompt(self.context_json(ctx)),
                                        policy_schema(allowed), None, SYSTEM_POLICY)
        if not res.data or res.data.get("action") not in allowed:
            a, r, c = self.fallback.propose(ctx)
            return a, f"LLM failed ({res.error or 'bad action'}); fallback {self.fallback.name}: {r}", c
        try:
            conf = float(res.data.get("confidence", 0.5))
        except (TypeError, ValueError):
            conf = 0.5
        return res.data["action"], str(res.data.get("reason", ""))[:400], conf


def build_policy(kind: str, client: LLMClient | None = None, seed: int | None = None) -> Policy:
    if kind == "fixed":
        return FixedPolicy()
    if kind == "random":
        return RandomPolicy(seed)
    if kind == "rule_chain":
        return RuleChainPolicy()
    if kind == "bandit":
        return BanditPolicy(seed)
    if kind == "llm":
        if client is None:
            raise ValueError("llm policy needs an LLM client")
        return LLMPolicy(client, BanditPolicy(seed))
    raise ValueError(f"unknown policy {kind}")
