"""Code-level checks on every policy decision (seedling-agent `validators.py` pattern, without
its `eval`). The LLM proposes; these rules dispose. Nothing outside the allowed list can ever
reach the speaker."""
from __future__ import annotations

import re
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
    immediate_escalation: bool = False  # species may go to the farmer before attempt 2 (elephant)


# T4's E4 probe: 15.6% of LLM events went to the farmer after one sound. Two deterrents first.
MIN_ATTEMPTS_BEFORE_ESCALATION = 2


def on_cooldown(sound: str, ctx: ValidationContext) -> bool:
    return ctx.now - ctx.last_played_at.get(sound, -1e9) < ctx.sound_cooldown_s


def legal_sounds(ctx: ValidationContext) -> list[str]:
    return [s for s in ctx.allowed if s not in ctx.tried_this_event and not on_cooldown(s, ctx)]


def escalation_allowed(ctx: ValidationContext, sounds: list[str] | None = None) -> bool:
    sounds = legal_sounds(ctx) if sounds is None else sounds
    return ctx.attempt_no >= MIN_ATTEMPTS_BEFORE_ESCALATION or ctx.immediate_escalation or not sounds


def legal_actions(ctx: ValidationContext) -> list[str]:
    """What validate() would pass unchanged. The constrained policy enum (agent.policy.constrain_enum)."""
    if ctx.attempt_no >= ctx.max_attempts:
        return [ESCALATE]
    sounds = legal_sounds(ctx)
    return sounds + ([ESCALATE] if escalation_allowed(ctx, sounds) else [])


def validate(action: str, ctx: ValidationContext) -> tuple[str, list[str]]:
    """Returns (final_action, warnings). Never raises. A replaced action becomes the first legal
    sound in species.yaml order (or escalate_farmer if none is left)."""
    warnings: list[str] = []
    if ctx.attempt_no >= ctx.max_attempts:
        return ESCALATE, [f"max_attempts={ctx.max_attempts} reached"]
    candidates = legal_sounds(ctx)
    if action == ESCALATE:
        if escalation_allowed(ctx, candidates):
            return ESCALATE, warnings
        return candidates[0], [f"'{ESCALATE}' before attempt {MIN_ATTEMPTS_BEFORE_ESCALATION}"]
    if action not in ctx.allowed:
        warnings.append(f"'{action}' not in allowed list")
    elif action in ctx.tried_this_event:
        warnings.append(f"'{action}' already failed in this event")
    elif on_cooldown(action, ctx):
        warnings.append(f"'{action}' on cooldown")
    else:
        return action, warnings
    return (candidates[0] if candidates else ESCALATE), warnings


@dataclass
class AlertFacts:
    """What a farmer alert must say, as phrases in the alert's language, from structured event fields."""
    animal: str
    count: int
    attempts: int
    sounds: list[str]
    status: str
    time: str
    forbidden: list[str]  # other animals and sounds. Multi-word = substring, one word = whole word
    number_words: dict[str, int]
    max_chars: int
    latin_allowed: bool = False


DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
ALERT_WORD = r"[a-z]+|[\u0900-\u0963\u0970-\u097f]+"  # no danda or digits


def alert_problems(text: str, facts: AlertFacts) -> list[str]:
    """Why this LLM alert can't be sent; [] if its facts hold. Catches wrong or missing facts and
    repetition. It can't judge sense or grammar: a nonsense clause beside the right facts passes."""
    t = text.translate(DEVANAGARI_DIGITS).lower()
    problems: list[str] = []
    if len(text) > facts.max_chars:
        problems.append("too long")
    if not facts.latin_allowed and re.search(r"[a-z]", t):
        problems.append("Latin script in a Devanagari alert")
    sentences = [s.strip() for s in re.split(r"[.!?\u0964\n]+", t) if s.strip()]
    if len(sentences) != len(set(sentences)):
        problems.append("repeated sentence")
    rest = t.replace(facts.time, " ")
    # longest first, so a phrase inside a longer one doesn't break it
    for phrase in sorted([facts.animal, facts.status, *facts.sounds], key=len, reverse=True):
        if phrase.lower() in rest:
            rest = rest.replace(phrase.lower(), " ")
        else:
            problems.append(f"missing: {phrase}")
    words = set(re.findall(ALERT_WORD, rest))
    problems += [f"names: {name}" for name in facts.forbidden
                 if (name.lower() in rest if " " in name else name.lower() in words)]
    numbers = {int(n) for n in re.findall(r"\d+", rest)} | {v for w, v in facts.number_words.items() if w in words}
    problems += [f"wrong number: {n}" for n in sorted(numbers - {facts.count, facts.attempts})]
    if facts.count > 1 and facts.count not in numbers:
        problems.append(f"missing count: {facts.count}")
    return problems
