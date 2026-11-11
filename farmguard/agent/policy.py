"""Deterrent policies. The LLM policy is the paper's contribution; the others are baselines.

  fixed       - always the species' default sound (what ~55 student repos do)
  random      - random allowed sound, never the one that just failed (Elemantra-style rule-based
                randomising; Ashil10 random sound per species)
  rule_chain  - fixed escalation chain A -> B -> C -> farmer (AEGISFARM `response_engine.py`)
  bandit      - Thompson sampling over past outcomes per species x time bucket (strong
                non-GenAI learner; the LLM must be compared against this, not only weak baselines)
  bandit_sw   - the same over only the last N outcomes of that species x bucket (sliding window),
                so it can follow habituation, which a stationary bandit can't (T4's E4 note)
  llm         - LLM with an enum-constrained action, given behaviour from the VLM, attempts so far
                and outcome memory; falls back to bandit on any failure.
All policies see the same PolicyContext and go through validators.validate().
"""
from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from farmguard.agent.llm.base import LLMClient
from farmguard.agent.prompts import SYSTEM_POLICY, policy_prompt, policy_schema
from farmguard.agent.validators import ESCALATE, ValidationContext, legal_actions, on_cooldown, validate
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
    event_id: str = ""
    immediate_escalation_species: tuple[str, ...] = ("elephant",)  # agent.policy.immediate_escalation_species

    def validation(self) -> ValidationContext:
        return ValidationContext(self.allowed, self.tried_this_event, self.attempt_no,
                                 self.max_attempts, self.last_played_at, self.now,
                                 self.sound_cooldown_s,
                                 self.species in self.immediate_escalation_species)


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

    def counts(self, st: dict[str, Any]) -> tuple[int, int]:
        return st.get("successes", 0), st.get("trials", 0)

    def propose(self, ctx):
        best, best_v, best_c = None, -1.0, (0, 0)
        for s in ctx.allowed:
            if s in ctx.tried_this_event:
                continue
            succ, trials = self.counts(ctx.stats.get(s, {}))
            v = self.rng.betavariate(1 + succ, 1 + trials - succ)
            if v > best_v:
                best, best_v, best_c = s, v, (succ, trials)
        if best is None:
            return ESCALATE, "no untried sound left", 1.0
        return best, f"Thompson sample {best_v:.2f} (succ {best_c[0]}/{best_c[1]})", best_v


class SlidingWindowBanditPolicy(BanditPolicy):
    """Thompson sampling over the window_* counts from Memory.sound_stats(window=N): the newest N
    outcomes of this species x time bucket, all sounds together. Older outcomes are forgotten."""
    name = "bandit_sw"

    def counts(self, st):
        return st.get("window_successes", 0), st.get("window_trials", 0)


class LLMPolicy(Policy):
    """One worker thread decides at a time, so propose() can leave its trace on self for decide()."""
    name = "llm"

    def __init__(self, client: LLMClient, fallback: Policy | None = None,
                 decision_log: str | Path | None = None, constrain_enum: bool = True):
        self.client = client
        self.fallback = fallback or BanditPolicy()
        # True: the schema enum holds only legal actions (validators.legal_actions), so a tried or
        # cooling-down sound can't be chosen. False: every allowed sound + escalate, as in T4's probe.
        self.constrain_enum = constrain_enum
        self.decision_log = Path(decision_log) if decision_log else None  # JSONL, one line per decision
        self._trace: dict[str, Any] = {}

    def enum(self, ctx: PolicyContext) -> list[str]:
        return legal_actions(ctx.validation()) if self.constrain_enum else ctx.allowed + [ESCALATE]

    def context_json(self, ctx: PolicyContext) -> dict[str, Any]:
        """Same keys every time, and every allowed sound always listed in the same order, so a
        sound with no trials reads as "no evidence" instead of being absent."""
        v = ctx.validation()
        return {
            "species": ctx.species_label, "animals": ctx.count, "behaviour": ctx.behaviour,
            "time_of_day": ctx.hour_bucket,
            "attempt": ctx.attempt_no + 1, "max_attempts": ctx.max_attempts,
            "tried_this_event": ctx.tried_this_event,
            "on_cooldown": [s for s in ctx.allowed if on_cooldown(s, v)],
            "allowed_actions": self.enum(ctx),
            "history": {s: {"trials": ctx.stats.get(s, {}).get("trials", 0),
                            "success_rate": ctx.stats.get(s, {}).get("success_rate"),
                            "recent": ctx.stats.get(s, {}).get("recent", [])}
                        for s in ctx.allowed},
        }

    def propose(self, ctx):
        allowed = self.enum(ctx)
        prompt = policy_prompt(self.context_json(ctx))
        res = self.client.generate_json(prompt, policy_schema(allowed), None, SYSTEM_POLICY)
        proposed = (res.data or {}).get("action")
        in_enum = proposed in allowed
        self._trace = {"provider": self.client.name, "model": res.model,
                       "seed": getattr(self.client, "seed", None),  # only Ollama sends one
                       "constrain_enum": self.constrain_enum, "enum": allowed,
                       "system": SYSTEM_POLICY, "prompt": prompt,
                       "raw_response": res.raw, "llm_error": res.error, "llm_latency_s": res.latency_s,
                       "proposed": proposed, "llm_failed": res.data is None,
                       "out_of_enum": res.data is not None and not in_enum}
        if not in_enum:
            a, r, c = self.fallback.propose(ctx)
            return a, f"LLM failed ({res.error or 'bad action'}); fallback {self.fallback.name}: {r}", c
        try:
            conf = min(1.0, max(0.0, float(res.data.get("confidence", 0.5))))
        except (TypeError, ValueError):
            conf = 0.5
        return proposed, str(res.data.get("reason", ""))[:400], conf

    def decide(self, ctx):
        if ctx.attempt_no >= ctx.max_attempts:  # the validator would force this: don't spend a call
            return PolicyDecision(ESCALATE, f"max_attempts={ctx.max_attempts} reached", 1.0, self.name)
        if self.constrain_enum and self.enum(ctx) == [ESCALATE]:  # nothing to choose: no call either
            return PolicyDecision(ESCALATE, "no legal sound left", 1.0, self.name)
        d = super().decide(ctx)
        t = self._trace
        used_fallback = t["llm_failed"] or t["out_of_enum"]
        if used_fallback:  # so results never credit the LLM with a decision the bandit made
            d.source = f"llm_fallback_{self.fallback.name}"
        self._log({"ts": time.time(), "event_id": ctx.event_id, "attempt": ctx.attempt_no + 1, **t,
                   "validator_changed": bool(d.warnings) and not used_fallback,
                   "final_action": d.action, "source": d.source, "reason": d.reason, "warnings": d.warnings})
        return d

    def _log(self, record: dict[str, Any]) -> None:
        if not self.decision_log:
            return
        try:  # losing a log line must never cost the decision
            self.decision_log.parent.mkdir(parents=True, exist_ok=True)
            with self.decision_log.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except OSError as exc:
            log.warning("Could not write decision log: %s", exc)


def decision_log_summary(path: str | Path) -> dict[str, dict[str, Any]]:
    """Per model, from the JSONL. invalid_action_rate is over the answers the LLM actually gave:
    outside the enum, or changed by validate() (already tried, on cooldown). Failed calls are counted
    apart. Compare across providers with care: those that enforce the schema can't go out of enum."""
    out: dict[str, dict[str, Any]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        s = out.setdefault(r["model"], {"decisions": 0, "llm_failed": 0, "out_of_enum": 0, "validator_changed": 0})
        s["decisions"] += 1
        for key in ("llm_failed", "out_of_enum", "validator_changed"):
            s[key] += bool(r[key])
    for s in out.values():
        answers = s["decisions"] - s["llm_failed"]
        s["invalid_action_rate"] = (s["out_of_enum"] + s["validator_changed"]) / answers if answers else None
    return out


def build_policy(kind: str, client: LLMClient | None = None, seed: int | None = None,
                 cfg: Any = None) -> Policy:
    """cfg (a Config) only supplies agent.policy.constrain_enum; without it the default (true) holds."""
    if kind == "fixed":
        return FixedPolicy()
    if kind == "random":
        return RandomPolicy(seed)
    if kind == "rule_chain":
        return RuleChainPolicy()
    if kind == "bandit":
        return BanditPolicy(seed)
    if kind == "bandit_sw":
        return SlidingWindowBanditPolicy(seed)
    if kind == "llm":
        if client is None:
            raise ValueError("llm policy needs an LLM client")
        constrain = cfg.get("agent.policy.constrain_enum", True) if cfg is not None else True
        return LLMPolicy(client, BanditPolicy(seed), constrain_enum=bool(constrain))
    raise ValueError(f"unknown policy {kind}")
