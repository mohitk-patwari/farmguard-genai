"""Farmer side: nightly summary and question answering over the SQLite events table.

Plain SQL rows -> LLM answer citing event ids -> template fallback. Code does the counting and the
LLM only words it. Citations are an enum of the retrieved ids and an answer that cites none is
discarded, so the farmer never gets an ungrounded answer. Nothing here can reach the speaker.

Run from cron / Task Scheduler at notifier.nightly_summary_hour, or by hand:
    python -m farmguard.agent.farmer summary
    python -m farmguard.agent.farmer ask "कल कौन सा जानवर आया?" --language hi
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from collections import Counter
from typing import Any

from farmguard.agent.llm.base import LLMClient
from farmguard.agent.llm.providers import build_client
from farmguard.agent.memory import Memory
from farmguard.agent.notifier import STATUS_WORDS, Notifier, local_name
from farmguard.agent.prompts import (
    SYSTEM_FARMER,
    cited_answer_schema,
    farmer_answer_prompt,
    nightly_summary_prompt,
)
from farmguard.config import Config, load_config

log = logging.getLogger(__name__)

DAY = 86400
# ponytail: newest rows only (totals are still counted over all matches). Needs real retrieval once
# a farm has enough history for a question to be about rows older than these.
MAX_ROWS = 30
TEMPLATE_ROWS = 5
LEFT = ("gone", "leaving")
WORD = r"[a-z]+|[ऀ-ॣ०-ॿ]+"  # Devanagari without the danda, so "काल।" is the word "काल"

# Roman spellings too: farmers often type Hindi in Latin script
NIGHT = {"night", "tonight", "raat", "रात", "रात्री", "रात्र"}
YESTERDAY = {"yesterday", "kal", "कल", "काल"}
TODAY = {"today", "aaj", "आज"}
WEEK = {"week", "hafta", "hafte", "हफ्ता", "हफ्ते", "हफ़्ता", "हफ़्ते", "सप्ताह", "आठवडा", "आठवड्यात", "आठवड्याचा"}

TEMPLATES = {
    "hi": {"none": "इस अवधि में कोई घटना दर्ज नहीं है", "cite": "घटना",
           "head": "{n} घटनाएँ: {by}। {left} बार जानवर खेत से चला गया, {other} बार जाना पक्का नहीं।"},
    "mr": {"none": "या कालावधीत कोणतीही घटना नोंदलेली नाही", "cite": "घटना",
           "head": "{n} घटना: {by}. {left} वेळा प्राणी शेतातून निघून गेला, {other} वेळा गेल्याची खात्री नाही."},
    "en": {"none": "No events recorded in this period", "cite": "event",
           "head": "{n} event(s): {by}. The animal left {left} time(s); not confirmed {other} time(s)."},
}


def _midnight(now: float) -> float:
    lt = time.localtime(now)
    return time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, 0, 0, 0, 0, 0, -1))


def nightly_window(now: float, hour: int) -> tuple[float, float]:
    """The 24 h before the latest summary hour, up to now, so a late or re-run job misses nothing."""
    boundary = _midnight(now) + hour * 3600
    if boundary > now:
        boundary -= DAY
    return boundary - DAY, now


def parse_question(cfg: Config, question: str, now: float) -> tuple[float | None, float | None, list[str]]:
    """Time window and species from whole words. A miss only widens the retrieval, and the LLM
    still sees each row's animal and time, so a word that isn't listed here is safe."""
    tokens = re.findall(WORD, question.lower())
    words, text = set(tokens), f" {' '.join(tokens)} "
    midnight = _midnight(now)
    since: float | None = None
    until: float | None = None
    if words & NIGHT:  # ponytail: "last night" and "tonight" are both the last 24 h; the rows carry times
        since, until = now - DAY, now
    elif words & YESTERDAY:
        since, until = midnight - DAY, midnight
    elif words & TODAY:
        since, until = midnight, now
    elif words & WEEK:
        since, until = now - 7 * DAY, now

    def named(key: str) -> bool:
        info = cfg.species[key]
        names = [key, *info.aliases, info.hindi, info.marathi]
        return any(f" {' '.join(re.findall(WORD, n.lower()))} " in text for n in names)

    return since, until, [k for k in cfg.species if named(k)]


def _period(since: float | None, until: float | None, now: float) -> str:
    fmt = "%d/%m %H:%M"
    start = time.strftime(fmt, time.localtime(since)) if since is not None else "*"
    return f"{start} - {time.strftime(fmt, time.localtime(until if until is not None else now))}"


def build_facts(cfg: Config, rows: list[dict[str, Any]], language: str, period: str) -> dict[str, Any]:
    """Counted in code over every matching row. No zone or other location field goes to the LLM."""
    by_animal = Counter(local_name(cfg, r["species"], language) for r in rows)
    left = sum(r["final_status"] in LEFT for r in rows)
    return {"period": period, "total_events": len(rows), "by_animal": dict(by_animal), "left_field": left,
            "not_confirmed_left": len(rows) - left,
            "events": [{"id": r["event_id"], "time": time.strftime("%d/%m %H:%M", time.localtime(r["wall_time"])),
                        "animal": local_name(cfg, r["species"], language), "behaviour": r["vlm_behaviour"],
                        "sounds": r["sounds"], "result": r["final_status"], "farmer_called": bool(r["escalated"])}
                       for r in rows[:MAX_ROWS]]}


def template_report(facts: dict[str, Any], language: str) -> tuple[str, list[str]]:
    t = TEMPLATES.get(language, TEMPLATES["en"])
    if not facts["total_events"]:
        return f"{t['none']} ({facts['period']})", []
    by = ", ".join(f"{name} {n}" for name, n in facts["by_animal"].items())
    head = t["head"].format(n=facts["total_events"], by=by, left=facts["left_field"],
                            other=facts["not_confirmed_left"])
    words = STATUS_WORDS.get(language, STATUS_WORDS["en"])
    shown = facts["events"][:TEMPLATE_ROWS]
    lines = [f"{e['time']} {e['animal']}: {words.get(e['result'], e['result'])} [{e['id']}]" for e in shown]
    return "\n".join([head, *lines]), [e["id"] for e in shown]


def _compose(client: LLMClient | None, prompt: str, facts: dict[str, Any], language: str) -> dict[str, Any]:
    ids = [e["id"] for e in facts["events"]]
    if client is not None and ids:  # no rows: nothing to ground an answer on, so don't call the LLM
        res = client.generate_json(prompt, cited_answer_schema(ids), system=SYSTEM_FARMER)
        data = res.data or {}
        message = str(data.get("message") or "").strip()
        raw_cited = data.get("cited_event_ids")
        cited = list(dict.fromkeys(i for i in raw_cited if i in ids)) if isinstance(raw_cited, list) else []
        if message and cited:
            label = TEMPLATES.get(language, TEMPLATES["en"])["cite"]
            return {"message": f"{message} [{label}: {', '.join(cited)}]", "voice_text": message,
                    "cited_event_ids": cited, "source": "llm", "total_events": facts["total_events"]}
        log.warning("Farmer answer unusable (%s); using the template", res.error or "no valid citation")
    text, cited = template_report(facts, language)
    return {"message": text, "voice_text": text.split("\n")[0], "cited_event_ids": cited,
            "source": "template", "total_events": facts["total_events"]}


def answer_question(cfg: Config, memory: Memory, client: LLMClient | None, question: str,
                    now: float | None = None) -> dict[str, Any]:
    now = time.time() if now is None else now
    language = cfg.get("notifier.language", "hi")
    question = question[:500]
    since, until, species = parse_question(cfg, question, now)
    rows = memory.events_with_sounds(since, until, species)
    facts = build_facts(cfg, rows, language, _period(since, until, now))
    return _compose(client, farmer_answer_prompt(question, facts, language), facts, language)


def nightly_summary(cfg: Config, memory: Memory, client: LLMClient | None,
                    now: float | None = None) -> dict[str, Any]:
    now = time.time() if now is None else now
    language = cfg.get("notifier.language", "hi")
    since, until = nightly_window(now, cfg.get("notifier.nightly_summary_hour", 7))
    rows = memory.events_with_sounds(since, until)
    facts = build_facts(cfg, rows, language, _period(since, until, now))
    return _compose(client, nightly_summary_prompt(facts, language), facts, language)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="farmguard.agent.farmer", description=__doc__)
    ap.add_argument("command", choices=["summary", "ask"])
    ap.add_argument("question", nargs="?", help="the farmer's question (ask)")
    ap.add_argument("--language", help="hi|mr|en")
    ap.add_argument("--provider", help="ollama|gemini|anthropic|openai_compat|mock")
    ap.add_argument("--config", help="settings.yaml path")
    args = ap.parse_args(argv)
    if args.command == "ask" and not args.question:
        ap.error("ask needs a question")
    logging.basicConfig(level=logging.WARNING)
    if hasattr(sys.stdout, "reconfigure"):  # Devanagari on a Windows console
        sys.stdout.reconfigure(encoding="utf-8")
    overrides = {k: v for k, v in {"notifier.language": args.language, "agent.provider": args.provider}.items() if v}
    cfg = load_config(args.config, overrides=overrides)
    try:
        client: LLMClient | None = build_client(cfg, "llm")
    except Exception as exc:  # missing SDK or key: the template still answers (invariant 4)
        log.warning("No LLM client (%s); using the template", exc)
        client = None
    memory = Memory(cfg.path("memory.db_path"))
    out = (nightly_summary(cfg, memory, client) if args.command == "summary"
           else answer_question(cfg, memory, client, args.question))
    Notifier(cfg, client).deliver(out["message"], out["voice_text"], out["source"])


if __name__ == "__main__":
    main()
