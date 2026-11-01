"""Code-level checks on every policy decision (seedling-agent `validators.py` pattern, without
its `eval`). The LLM proposes; these rules dispose. Nothing outside the allowed list can ever
reach the speaker."""
from __future__ import annotations

from dataclasses import dataclass, field

ESCALATE = "escalate_farmer"


@dataclass
class ValidationContext:
    allowed: list[str]
    tried_this_event: list[str]
    attempt_no: int                     # attempts already made in this event (incl. first alarm)
    max_attempts: int
    last_played_at: dict[str, float] = field(default_factory=dict)  # sound -> video time
    now: float = 0.0
    sound_cooldown_s: float = 20.0


def on_cooldown(sound: str, ctx: ValidationContext) -> bool:
    return ctx.now - ctx.last_played_at.get(sound, -1e9) < ctx.sound_cooldown_s


def validate(action: str, ctx: ValidationContext) -> tuple[str, list[str]]:
    """Returns (final_action, warnings). Never raises."""
    warnings: list[str] = []
    if ctx.attempt_no >= ctx.max_attempts:
        return ESCALATE, [f"max_attempts={ctx.max_attempts} reached"]
    if action == ESCALATE:
        return ESCALATE, warnings
    candidates = [s for s in ctx.allowed if s not in ctx.tried_this_event and not on_cooldown(s, ctx)]
    if action not in ctx.allowed:
        warnings.append(f"'{action}' not in allowed list")
        action = candidates[0] if candidates else ESCALATE
    elif action in ctx.tried_this_event:
        warnings.append(f"'{action}' already failed in this event")
        action = candidates[0] if candidates else ESCALATE
    elif on_cooldown(action, ctx):
        warnings.append(f"'{action}' on cooldown")
        action = candidates[0] if candidates else ESCALATE
    return action, warnings
