"""All prompts and JSON schemas in one place, so experiments can ablate them.

Prompt rules from our review (see docs/RESEARCH_NOTES.md, "Lessons for the GenAI layer"):
  R1 Never mention where the farm is. A "New Zealand" hint made Claude reason from geography and
     drop to ~10% accuracy (AlexKapy camera-trap-ai-pipeline README). Location filtering, if any,
     happens in code (SpeciesNet --country IND), not in the prompt.
  R2 Ask for objective visual observations FIRST, then the category (OBSERVATION/CATEGORY/NOTE).
  R3 Closed category list with an explicit "unclassifiable" option, so doubt becomes an
     abstention instead of a confident wrong answer (62 of 174 answers were abstentions there).
  R4 Send several frames (crop + full frame + neighbours): committed accuracy 65.0% -> 80.9% on
     AlexKapy's dev set; sequences gave the best behaviour accuracy in Alencar et al., JBCS 2026.
  R5 Keep behaviour prompts short and concrete; Dussert et al. (MEE 2025) found both vague and
     over-long prompts hurt, and adding species names to the behaviour question hurt too.
  R6 The policy LLM may only choose from an enum of allowed actions (seedling-agent).
"""
from __future__ import annotations

from typing import Any

BEHAVIOURS = ["eating", "moving_through", "resting", "leaving", "absent", "unclear"]
CONFIDENCE = ["high", "medium", "low"]
RETREAT = ["gone", "leaving", "still_present", "uncertain"]

SYSTEM_VISION = ("You are a careful wildlife observer looking at frames from a farm camera. "
                 "Describe only what is visible in the pixels. Never guess from context.")


def vlm_schema(species_keys: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "observation": {"type": "string",
                            "description": "2-3 objective visual details actually visible"},
            "species": {"type": "string", "enum": species_keys + ["other_animal", "none", "unclassifiable"]},
            "count": {"type": "integer", "minimum": 0, "maximum": 50},
            "behaviour": {"type": "string", "enum": BEHAVIOURS},
            "confidence": {"type": "string", "enum": CONFIDENCE},
            "note": {"type": "string", "description": "one short sentence"},
        },
        "required": ["observation", "species", "count", "behaviour", "confidence", "note"],
        "additionalProperties": False,
    }


def vlm_prompt(species_keys: list[str], n_images: int, layout: str,
               location_hint: str | None = None, fewshot_note: str | None = None) -> str:
    cats = ", ".join(species_keys + ["other_animal", "none", "unclassifiable"])
    lines = [
        f"You are given {n_images} image(s) from a farm camera, in this order: {layout}.",
        "Look carefully at body shape and size, coat texture, legs, head, ears and tail.",
        "",
        "1. observation: write 2-3 objective visual details you actually see.",
        f"2. species: choose exactly one from this list: {cats}.",
        "   Use 'unclassifiable' if you cannot tell. Use 'none' if no animal is visible.",
        "3. count: number of animals of that species visible.",
        ("4. behaviour: eating (head down at crops/plants), moving_through, resting, leaving "
         "(moving away / out of frame), absent, or unclear."),
        "5. confidence: high, medium or low.",
        "6. note: one short sentence.",
        "Base every answer only on what is visible in the images.",
    ]
    if fewshot_note:
        lines.insert(2, fewshot_note)
    if location_hint:  # ONLY for the E2 ablation that reproduces the location-bias finding
        lines.insert(1, f"These images come from {location_hint}.")
    lines.append("Reply with JSON only.")
    return "\n".join(lines)


def fewshot_note(labels: list[str]) -> str:
    """Few-shot (Alencar et al., JBCS 2026: BLIP empty filter 7.61% -> 91.0%). Only the category
    label is given about each reference image: no file name, place or farm (rule R1)."""
    numbered = ", ".join(f"{i}={label}" for i, label in enumerate(labels, 1))
    return (f"Before those images come {len(labels)} labelled reference images that are NOT from this "
            f"event, in this order: {numbered}. Use them only to learn what each category looks like; "
            "answer only about the event images that follow them.")


def retreat_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "observation": {"type": "string"},
            "status": {"type": "string", "enum": RETREAT},
            "animals_visible": {"type": "integer", "minimum": 0, "maximum": 50},
            "confidence": {"type": "string", "enum": CONFIDENCE},
        },
        "required": ["observation", "status", "animals_visible", "confidence"],
        "additionalProperties": False,
    }


def retreat_prompt(n_images: int, species_label: str) -> str:
    return "\n".join([
        (f"These {n_images} images are consecutive frames of a crop field, oldest first, taken "
         f"in the seconds after a deterrent sound was played to scare away a {species_label}."),
        "Decide the animal's status in the LAST frames:",
        "  gone          - no animal visible in the field any more",
        "  leaving       - the animal is clearly moving out of the field",
        "  still_present - the animal is still in the field (eating, standing or wandering)",
        "  uncertain     - you cannot tell (darkness, blur, occlusion)",
        ("First write a short observation, then the status, the number of animals still visible "
         "in the last frame, and your confidence. Reply with JSON only."),
    ])


SYSTEM_POLICY = ("You are the deterrence planner of a humane farm-protection system. You choose "
                 "which approved sound to play next to make an animal leave a crop field. You "
                 "must pick only from the allowed actions you are given.")


def policy_schema(allowed_actions: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": allowed_actions},
            "reason": {"type": "string", "description": "one or two sentences citing the evidence"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["action", "reason", "confidence"],
        "additionalProperties": False,
    }


def policy_prompt(context: dict[str, Any]) -> str:
    import json
    return "\n".join([
        "An animal is in the crop field and the previous deterrent did not make it leave.",
        "Choose the next action. Guidelines:",
        ("- Animals habituate: avoid sounds that were just tried in this event and sounds that "
         "recently failed for this species at this time of day."),
        ("- Prefer sounds with a high past success rate for this species, but sometimes try a "
         "less-used sound when the evidence is thin (few trials)."),
        "- If the animal is eating, act decisively; if it is already leaving, a milder sound is fine.",
        "- Choose 'escalate_farmer' only if no allowed sound is likely to work.",
        "",
        "Context (JSON):",
        json.dumps(context, ensure_ascii=False, indent=1),
        "",
        "Reply with JSON only: the action, a short reason, and your confidence (0-1).",
    ])


def alert_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {"message": {"type": "string"}, "voice_text": {"type": "string"}},
        "required": ["message", "voice_text"],
        "additionalProperties": False,
    }


LANG_NAMES = {"hi": "Hindi (Devanagari script)", "mr": "Marathi (Devanagari script)",
              "en": "simple English"}


def alert_prompt(summary: dict[str, Any], language: str) -> str:
    import json
    lang = LANG_NAMES.get(language, language)
    return "\n".join([
        f"Write an alert for a farmer in {lang}. Use very simple words a farmer would use.",
        "Facts (use ONLY these, do not add anything):",
        json.dumps(summary, ensure_ascii=False, indent=1),
        ("message: at most 3 short sentences: what animal came and when, what sounds were played, "
         "and whether it left. If it did not leave, ask the farmer to check the field."),
        "voice_text: the same content as one or two sentences, for text-to-speech.",
        "Reply with JSON only.",
    ])
