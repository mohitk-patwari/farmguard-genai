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
    """What a farmer alert must say, from structured event fields. Each fact is a list of stems in the
    alert's language: any one of them counts, so a harmless paraphrase passes (smoke test 2: 5 of 6 Hindi
    alerts were rejected only for rewording a sound name). Stems are substrings in Devanagari."""
    animal: list[str]
    count: int
    attempts: int
    sounds: list[list[str]]  # one stem list per sound played
    status: list[str]        # stems of the outcome's family (left / still there / unclear)
    time: str
    forbidden: list[str]     # stems of other animals, unplayed sounds and other outcomes
    number_words: dict[str, int]
    max_chars: int
    latin_allowed: bool = False  # English: forbidden single words match whole words only


DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
ALERT_WORD = r"[a-z]+|[\u0900-\u0963\u0970-\u097f]+"  # no danda or digits


def norm(text: str) -> str:
    """Lower case, ASCII digits, no nukta (आवाज़ = आवाज), so spelling variants match."""
    return text.translate(DEVANAGARI_DIGITS).lower().replace("\u093c", "")


def alert_problems(text: str, facts: AlertFacts) -> list[str]:
    """Why this LLM alert can't be sent; [] if its facts hold. Catches wrong or missing facts and
    repetition. It can't judge sense or grammar: a nonsense clause beside the right facts passes."""
    t = norm(text)
    problems: list[str] = []
    if len(text) > facts.max_chars:
        problems.append("too long")
    if not facts.latin_allowed and re.search(r"[a-z]", t):
        problems.append("Latin script in a Devanagari alert")
    sentences = [s.strip() for s in re.split(r"[.!?\u0964\n]+", t) if s.strip()]
    if len(sentences) != len(set(sentences)):
        problems.append("repeated sentence")
    rest = t.replace(norm(facts.time), " ")
    clean = rest  # for the forbidden check: every matched stem removed (नीलगाय must not read as गाय)
    for label, stems in [("animal", facts.animal), *(("sound", s) for s in facts.sounds), ("outcome", facts.status)]:
        hit = next((s for s in sorted({norm(x) for x in stems}, key=len, reverse=True) if s in rest), None)
        if hit is None:
            problems.append(f"missing {label}: {stems[0]}")
            continue
        rest = rest.replace(hit, " ", 1)  # one mention per fact: जंगली सूअर can't also be the boar sound
        clean = clean.replace(hit, " ")
    words = set(re.findall(ALERT_WORD, clean))
    for name in facts.forbidden:
        n = norm(name)
        if (n in words) if (facts.latin_allowed and " " not in n) else (n in clean):
            problems.append(f"names: {name}")
    numbers = {int(n) for n in re.findall(r"\d+", clean)} | {v for w, v in facts.number_words.items() if w in words}
    problems += [f"wrong number: {n}" for n in sorted(numbers - {facts.count, facts.attempts})]
    if facts.count > 1 and facts.count not in numbers:
        problems.append(f"missing count: {facts.count}")
    return problems
