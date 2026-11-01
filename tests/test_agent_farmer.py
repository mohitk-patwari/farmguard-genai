"""Nightly summary and question answering: retrieval, citations, Hindi/Marathi, fallback."""
from __future__ import annotations

import json
import time

import pytest
import yaml

from farmguard.agent.farmer import (
    MAX_ROWS,
    answer_question,
    main,
    nightly_summary,
    nightly_window,
    parse_question,
)
from farmguard.agent.llm.providers import MockClient
from farmguard.agent.memory import Memory
from farmguard.config import load_config

NOW = time.mktime((2026, 11, 2, 10, 0, 0, 0, 0, -1))
YESTERDAY_9PM = time.mktime((2026, 11, 1, 21, 0, 0, 0, 0, -1))
TODAY_2AM = time.mktime((2026, 11, 2, 2, 0, 0, 0, 0, -1))
LAST_WEEK = NOW - 5 * 86400


def _cfg(lang="hi"):
    return load_config(overrides={"notifier.language": lang, "notifier.channels": ["console"]})


def _event(m, event_id, wall_time, species="wild_boar", status="gone", sounds=("boar_alarm", "siren")):
    m.add_event(event_id=event_id, wall_time=wall_time, species=species, final_status=status, zone="Nashik plot",
                vlm_behaviour="eating", attempts=len(sounds), escalated=int(status == "still_present"))
    for n, s in enumerate(sounds, 1):
        m.add_attempt(event_id, n, species, s, "llm", "", status, None)


@pytest.fixture
def memory():
    m = Memory(":memory:")
    _event(m, "boar1", YESTERDAY_9PM)
    _event(m, "monkey1", TODAY_2AM, "monkey", "still_present", ("gunshot",))
    _event(m, "old1", LAST_WEEK, "elephant")
    return m


def _cites(*ids):
    return MockClient(responder=lambda p, s, n: {"message": "उत्तर", "cited_event_ids": list(ids)})


def test_memory_rows_have_sounds_in_play_order_and_filter(memory):
    rows = memory.events_with_sounds()
    assert [r["event_id"] for r in rows] == ["monkey1", "boar1", "old1"]
    assert rows[1]["sounds"] == "boar_alarm, siren"
    assert [r["event_id"] for r in memory.events_with_sounds(species=["monkey"])] == ["monkey1"]
    assert memory.events_with_sounds(since=TODAY_2AM + 1) == []


@pytest.mark.parametrize("question,hours_back", [
    ("कल कौन सा जानवर आया?", 34), ("काल कोणता प्राणी आला?", 34), ("what came yesterday", 34), ("कल क्या आया।", 34),
    ("आज क्या आया", 10), ("this week?", 168), ("हफ्ते में क्या हुआ", 168), ("कल रात क्या हुआ", 24),
])
def test_time_words(question, hours_back):
    since, _until, _ = parse_question(_cfg(), question, NOW)
    assert round((NOW - since) / 3600) == hours_back


def test_time_words_match_whole_words_only():
    # निकला ("left") contains कल ("yesterday")
    assert parse_question(_cfg(), "सूअर निकला क्या", NOW)[:2] == (None, None)


def test_yesterday_stops_at_midnight():
    _since, until, _ = parse_question(_cfg(), "yesterday", NOW)
    assert until == NOW - 10 * 3600


@pytest.mark.parametrize("question,species", [
    ("जंगली सूअर कब आया", ["wild_boar"]), ("माकड कधी आले", ["monkey"]), ("did a wild boar or a cow come", ["wild_boar", "stray_cattle"]),
    ("कौन सा जानवर आया", []), ("any pigeons", []),
])
def test_species_words(question, species):
    assert parse_question(_cfg(), question, NOW)[2] == species


def test_answer_cites_real_events_only(memory):
    out = answer_question(_cfg(), memory, _cites("boar1", "invented", "boar1"), "कल कौन आया?", now=NOW)
    assert out["source"] == "llm" and out["cited_event_ids"] == ["boar1"]
    assert "[घटना: boar1]" in out["message"] and "boar1" not in out["voice_text"]


def test_prompt_is_retrieved_rows_with_enum_citations_and_no_location(memory):
    seen = {}

    def responder(prompt, schema, n):
        seen.update(prompt=prompt, schema=schema)
        return {"message": "x", "cited_event_ids": ["monkey1"]}

    answer_question(_cfg(), memory, MockClient(responder=responder), "माकड आज आया?", now=NOW)
    assert seen["schema"]["properties"]["cited_event_ids"]["items"]["enum"] == ["monkey1"]
    assert "बंदर" in seen["prompt"] and "boar1" not in seen["prompt"]
    assert "nashik" not in seen["prompt"].lower() and "zone" not in seen["prompt"]


@pytest.mark.parametrize("reply", [
    {"message": "सब ठीक है", "cited_event_ids": []},          # no grounding
    {"message": "सब ठीक है", "cited_event_ids": ["invented"]},
    {"message": "", "cited_event_ids": ["boar1"]},
    {"message": "सब ठीक है", "cited_event_ids": "boar1"},      # not a list
    {},
])
def test_ungrounded_answers_fall_back_to_the_template(memory, reply):
    out = answer_question(_cfg(), memory, MockClient(responder=lambda p, s, n: reply), "कल?", now=NOW)
    assert out["source"] == "template" and "boar1" in out["message"]


def test_network_down_still_answers():
    class Down(MockClient):
        def _call(self, *a):
            raise ConnectionError("offline")

    m = Memory(":memory:")
    _event(m, "boar1", YESTERDAY_9PM)
    assert answer_question(_cfg(), m, Down(), "कल?", now=NOW)["source"] == "template"


def test_no_rows_means_no_llm_call():
    client = MockClient()
    out = answer_question(_cfg(), Memory(":memory:"), client, "कल?", now=NOW)
    assert client.calls == [] and out["source"] == "template" and out["cited_event_ids"] == []
    assert "कोई घटना दर्ज नहीं" in out["message"]


@pytest.mark.parametrize("lang,animal,word", [
    ("hi", "जंगली सूअर", "खेत से चला गया"), ("mr", "रानडुक्कर", "शेतातून निघून गेले"), ("en", "wild boar", "left the field"),
])
def test_template_language(memory, lang, animal, word):
    out = nightly_summary(_cfg(lang), memory, None, now=NOW)
    assert animal in out["message"] and word in out["message"] and "[boar1]" in out["message"]


def test_nightly_window_and_contents(memory):
    start, end = nightly_window(NOW, 7)
    assert end == NOW and start == time.mktime((2026, 11, 1, 7, 0, 0, 0, 0, -1))
    # before the hour, the window moves back a day
    assert nightly_window(time.mktime((2026, 11, 2, 6, 0, 0, 0, 0, -1)), 7)[0] == time.mktime(
        (2026, 10, 31, 7, 0, 0, 0, 0, -1))
    out = nightly_summary(_cfg(), memory, None, now=NOW)
    assert out["total_events"] == 2 and "old1" not in out["message"]


def test_totals_are_counted_in_code_beyond_the_rows_sent():
    m = Memory(":memory:")
    for i in range(MAX_ROWS + 5):
        _event(m, f"e{i}", YESTERDAY_9PM + i)
    seen = {}

    def responder(prompt, schema, n):
        seen["facts"] = json.loads(prompt.split("Records (JSON):\n")[1].split("\n")[0])
        return {"message": "x", "cited_event_ids": ["e0"] if "e0" in schema["properties"]["cited_event_ids"]["items"]["enum"] else [f"e{MAX_ROWS + 4}"]}

    out = nightly_summary(_cfg(), m, MockClient(responder=responder), now=NOW)
    assert seen["facts"]["total_events"] == MAX_ROWS + 5 and len(seen["facts"]["events"]) == MAX_ROWS
    assert out["total_events"] == MAX_ROWS + 5


def test_command_line(tmp_path, capsys):
    db = tmp_path / "m.sqlite"
    _event(Memory(db), "boar1", time.time() - 3600)
    cfg = load_config()
    settings = tmp_path / "s.yaml"
    settings.write_text(yaml.safe_dump({**cfg.raw, "memory": {"db_path": str(db)}}), encoding="utf-8")
    main(["ask", "बंदर या जंगली सूअर?", "--provider", "mock", "--language", "mr", "--config", str(settings)])
    assert "रानडुक्कर" in capsys.readouterr().out
