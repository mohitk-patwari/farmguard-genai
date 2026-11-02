"""Farmer alerts: an LLM message goes out only if its facts match the event."""
from __future__ import annotations

import logging

import pytest

from farmguard.agent.llm.providers import MockClient
from farmguard.agent.notifier import Notifier
from farmguard.agent.prompts import alert_prompt
from farmguard.config import load_config

GOOD = "02:00 बजे जंगली सूअर आया। सायरन और कुत्ते की आवाज़ बजाई। वह अभी भी खेत में है।"


def _setup(reply, count=1, lang="hi", species="wild_boar", sounds=("siren", "dog_bark"), status="still_present"):
    cfg = load_config(overrides={"notifier.language": lang, "notifier.channels": ["console"]})
    nt = Notifier(cfg, MockClient(responder=lambda p, s, n: reply))
    summary = nt.build_summary(species, 0, list(sounds), status, len(sounds), count, "eating")
    summary["time"] = "02:00"
    return nt, summary


def _source(message, **kw):
    nt, summary = _setup({"message": message, "voice_text": message}, **kw)
    return nt.compose(summary)[2]


def test_grounded_alert_is_sent():
    assert _source(GOOD) == "llm"


@pytest.mark.parametrize("message", [
    GOOD.replace("जंगली सूअर", "हाथी"),                                  # wrong species
    GOOD + " बंदर भी था।",                                               # names another animal
    GOOD.replace("सायरन और ", ""),                                       # a played sound is missing
    GOOD + " ढोल भी बजाया।",                                             # a sound that was not played
    GOOD.replace("अभी भी खेत में है", "खेत से चला गया"),                   # wrong outcome
    GOOD + " 5 जानवर थे।",                                               # wrong count, digits
    GOOD + " ५ जानवर थे।",                                               # wrong count, Devanagari digits
    GOOD + " सात जानवर थे।",                                             # wrong count, a word
    GOOD + " dog_bark",                                                  # raw sound id
    GOOD + " " + GOOD.split("।")[0] + "।",                               # repeated sentence
    GOOD + " यह बहुत लंबा संदेश है।" * 20,                                # rambling
    "हाथी आया था। यहाँ बीमारी की चिंता करते हैं।",                            # the real 3b output from the smoke test
])
def test_wrong_facts_fall_back_to_the_template(message, caplog):
    with caplog.at_level(logging.WARNING):
        assert _source(message) == "template_ungrounded"
    assert "LLM alert rejected" in caplog.text


def test_count_above_one_must_be_stated():
    assert _source(GOOD, count=3) == "template_ungrounded"
    assert _source(GOOD.replace("जंगली सूअर", "3 जंगली सूअर"), count=3) == "llm"
    assert _source(GOOD.replace("जंगली सूअर", "तीन जंगली सूअर"), count=3) == "llm"


def test_marathi_and_english():
    mr = "02:00 वाजता रानडुक्कर आले. सायरन वाजवला. ते अजूनही शेतात आहे."
    assert _source(mr, lang="mr", sounds=("siren",)) == "llm"
    assert _source(mr.replace("रानडुक्कर", "हत्ती"), lang="mr", sounds=("siren",)) == "template_ungrounded"
    en = "At 02:00 a wild boar came. We played Siren. It is still in the field."
    assert _source(en, lang="en", sounds=("siren",)) == "llm"
    assert _source(en.replace("wild boar", "cow"), lang="en", sounds=("siren",)) == "template_ungrounded"


def test_bad_voice_text_does_not_lose_a_good_message():
    nt, summary = _setup({"message": GOOD, "voice_text": "हाथी आया"})
    message, voice, source = nt.compose(summary)
    assert source == "llm" and voice == message == GOOD


def test_no_answer_is_a_plain_template_and_is_warned():
    assert _source("") == "template"


@pytest.mark.parametrize("lang", ["hi", "mr", "en"])
@pytest.mark.parametrize("count", [1, 4])
def test_the_template_passes_its_own_check(lang, count):
    nt, summary = _setup({}, count=count, lang=lang)
    from farmguard.agent.notifier import template_message
    from farmguard.agent.validators import alert_problems
    template = template_message(summary, lang)
    assert alert_problems(template, nt._facts(summary, template)) == []


def test_prompt_carries_local_facts_only():
    _nt, summary = _setup({}, count=2)
    prompt = alert_prompt(summary, "hi")
    assert "कुत्ते की आवाज़" in prompt and "जंगली सूअर" in prompt and "अभी भी खेत में है" in prompt
    assert "dog_bark" not in prompt and "Dog barking" not in prompt
