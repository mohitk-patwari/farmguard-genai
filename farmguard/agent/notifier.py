"""Farmer alert: LLM-written message in Hindi / Marathi / English, template fallback,
console + Telegram (gotogrub `src/alerts/telegram.py`), optional gTTS voice note.

The LLM only rewrites facts we pass in (prompt: "use ONLY these"); if it fails, a template in
the same language is sent, so the farmer is always told (reliability requirement)."""
from __future__ import annotations

import logging
import os
import tempfile
import time
from typing import Any

import requests

from farmguard.agent.llm.base import LLMClient
from farmguard.agent.prompts import alert_prompt, alert_schema
from farmguard.agent.validators import AlertFacts, alert_problems
from farmguard.config import Config

log = logging.getLogger(__name__)

STATUS_WORDS = {
    "hi": {"gone": "खेत से चला गया", "leaving": "खेत से जा रहा था", "still_present": "अभी भी खेत में है",
           "uncertain": "स्थिति साफ़ नहीं है"},
    "mr": {"gone": "शेतातून निघून गेले", "leaving": "शेतातून जात होते", "still_present": "अजूनही शेतात आहे",
           "uncertain": "स्थिती स्पष्ट नाही"},
    "en": {"gone": "left the field", "leaving": "was leaving the field",
           "still_present": "is still in the field", "uncertain": "status is unclear"},
}


SOUND_NAMES = {
    "hi": {"bee_swarm": "मधुमक्खियों की आवाज़", "tiger_roar": "बाघ की आवाज़", "leopard_growl": "तेंदुए की आवाज़",
           "gunshot": "पटाखे की आवाज़", "dog_bark": "कुत्ते की आवाज़", "boar_alarm": "सूअर की आवाज़",
           "siren": "सायरन", "drum_beat": "ढोल", "human_shout": "आदमी के चिल्लाने की आवाज़"},
    "mr": {"bee_swarm": "मधमाश्यांचा आवाज", "tiger_roar": "वाघाचा आवाज", "leopard_growl": "बिबट्याचा आवाज",
           "gunshot": "फटाक्याचा आवाज", "dog_bark": "कुत्र्याचा आवाज", "boar_alarm": "डुकराचा आवाज",
           "siren": "सायरन", "drum_beat": "ढोल", "human_shout": "माणसाच्या ओरडण्याचा आवाज"},
}
# Words for animals and sounds the alert must not name unless they happened. Not exhaustive.
OTHER_WORDS = {
    "hi": {"wild_boar": ["सूअर", "सूअरों"], "monkey": ["बंदरों"], "elephant": ["हाथियों"], "nilgai": ["नीलगायों"],
           "stray_cattle": ["गाय", "गायें", "गायों", "बैल", "भैंस"]},
    "mr": {"wild_boar": ["डुक्कर", "डुकरे"], "monkey": ["माकडे"], "elephant": ["हत्ती"], "nilgai": [],
           "stray_cattle": ["गाय", "गाई", "बैल", "म्हैस"]},
}
SOUND_WORDS = {
    "hi": {"bee_swarm": ["मधुमक्खी", "मधुमक्खियों"], "tiger_roar": ["बाघ"], "leopard_growl": ["तेंदुआ", "तेंदुए"],
           "gunshot": ["पटाखा", "पटाखे", "बंदूक", "गोली"], "dog_bark": ["कुत्ता", "कुत्ते", "कुत्तों"],
           "boar_alarm": [], "siren": ["सायरन"], "drum_beat": ["ढोल"], "human_shout": ["आदमी", "इंसान"]},
    "mr": {"bee_swarm": ["मधमाशी", "मधमाश्यांचा", "मधमाश्या"], "tiger_roar": ["वाघ", "वाघाचा"],
           "leopard_growl": ["बिबट्या", "बिबट्याचा"], "gunshot": ["फटाका", "फटाके", "फटाक्याचा", "बंदूक"],
           "dog_bark": ["कुत्रा", "कुत्र्याचा", "कुत्रे"], "boar_alarm": [], "siren": ["सायरन"], "drum_beat": ["ढोल"],
           "human_shout": ["माणूस", "माणसाच्या"]},
}
NUMBER_WORDS = {
    "hi": {"एक": 1, "दो": 2, "तीन": 3, "चार": 4, "पांच": 5, "पाँच": 5, "छह": 6, "सात": 7, "आठ": 8, "नौ": 9, "दस": 10},
    "mr": {"एक": 1, "दोन": 2, "तीन": 3, "चार": 4, "पाच": 5, "सहा": 6, "सात": 7, "आठ": 8, "नऊ": 9, "दहा": 10},
    "en": {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10},
}
NON_LATIN = ("hi", "mr")


def template_message(summary: dict[str, Any], language: str) -> str:
    st = summary["status_local"]
    sounds = ", ".join(summary["sounds_local"])
    n = summary["count"]
    if language == "hi":
        count = f" (संख्या: {n})" if n > 1 else ""
        msg = (f"{summary['time']} बजे खेत में {summary['animal_local']}{count} आया। "
               f"बजाई गई आवाज़ें: {sounds}। जानवर {st}।")
        if summary["final_status"] not in ("gone", "leaving"):
            msg += " कृपया खेत देखें।"
        return msg
    if language == "mr":
        count = f" (संख्या: {n})" if n > 1 else ""
        msg = (f"{summary['time']} वाजता शेतात {summary['animal_local']}{count} आले. "
               f"वाजवलेले आवाज: {sounds}. प्राणी {st}.")
        if summary["final_status"] not in ("gone", "leaving"):
            msg += " कृपया शेत तपासा."
        return msg
    who = f"{n} x {summary['animal']}" if n > 1 else f"a {summary['animal']}"
    msg = (f"At {summary['time']} {who} entered the field. Sounds played: "
           f"{sounds}. The animal {st}.")
    if summary["final_status"] not in ("gone", "leaving"):
        msg += " Please check the field."
    return msg


FALSE_ALARM_NOTES = {  # fixed text, no LLM: there is nothing to report but a doubt
    "hi": "कम ज़रूरी: {time} बजे कैमरे ने {animal} पहचाना, पर जाँच में कोई जानवर नहीं दिखा। शायद गलत चेतावनी थी।",
    "mr": "कमी महत्त्वाचे: {time} वाजता कॅमेऱ्याने {animal} ओळखले, पण तपासणीत कोणताही प्राणी दिसला नाही. कदाचित चुकीचा इशारा होता.",
    "en": "Low priority: at {time} the camera flagged a {animal}, but the check saw no animal. Probably a false alarm.",
}


def local_name(cfg: Config, species: str, language: str) -> str:
    info = cfg.species.get(species)
    names = {"hi": info.hindi, "mr": info.marathi} if info else {}
    return names.get(language, species.replace("_", " "))


class Notifier:
    def __init__(self, cfg: Config, client: LLMClient | None):
        self.cfg = cfg
        self.client = client
        self.language = cfg.get("notifier.language", "hi")
        self.channels = cfg.get("notifier.channels", ["console"])
        self.voice = cfg.get("notifier.voice_note", True)
        self.sent: list[dict[str, Any]] = []

    def build_summary(self, species: str, wall_time: float, sounds: list[str], final_status: str,
                      attempts: int, count: int, behaviour: str) -> dict[str, Any]:
        local = local_name(self.cfg, species, self.language)
        labels = [self.cfg.sounds.get(s, {}).get("label", s) for s in sounds]
        names = SOUND_NAMES.get(self.language, {})
        words = STATUS_WORDS.get(self.language, STATUS_WORDS["en"])
        return {"species": species, "sound_ids": sounds,
                "sounds_local": [names.get(s, label) for s, label in zip(sounds, labels)],
                "status_local": words.get(final_status, final_status),
                "animal": species.replace("_", " "), "animal_local": local, "count": count,
                "time": time.strftime("%H:%M", time.localtime(wall_time)),
                "behaviour_when_seen": behaviour, "sounds_played": labels,
                "attempts": attempts, "final_status": final_status}

    def _facts(self, summary: dict[str, Any], template: str) -> AlertFacts:
        lang = self.language
        sound_names = SOUND_NAMES.get(lang, {})
        species_words = OTHER_WORDS.get(lang, {})
        other: list[str] = []
        for key, info in self.cfg.species.items():
            if key != summary["species"]:
                other += [local_name(self.cfg, key, lang), *species_words.get(key, [])]
                if lang not in NON_LATIN:
                    other += info.aliases
        for key in self.cfg.sounds:
            if key not in summary["sound_ids"]:
                other += [sound_names.get(key) or self.cfg.sounds[key].get("label", key),
                          *SOUND_WORDS.get(lang, {}).get(key, [])]
        return AlertFacts(summary["animal_local"], summary["count"], summary["attempts"], summary["sounds_local"],
                          summary["status_local"], summary["time"], other, NUMBER_WORDS.get(lang, {}),
                          max_chars=2 * len(template), latin_allowed=lang not in NON_LATIN)

    def compose(self, summary: dict[str, Any]) -> tuple[str, str, str]:
        """Returns (message, voice_text, source). source is llm, template (no usable answer) or
        template_ungrounded (the LLM answered and its facts were wrong)."""
        template = template_message(summary, self.language)
        if self.client is None:
            return template, template, "template"
        res = self.client.generate_json(alert_prompt(summary, self.language), alert_schema())
        data = res.data or {}
        message = str(data.get("message") or "").strip()
        if not message:
            log.warning("LLM alert unusable (%s); using the template", res.error or "empty message")
            return template, template, "template"
        facts = self._facts(summary, template)
        problems = alert_problems(message, facts)
        if problems:
            log.warning("LLM alert rejected: %s | %s", "; ".join(problems), message[:300])
            return template, template, "template_ungrounded"
        voice = str(data.get("voice_text") or "").strip()
        return message, voice if voice and not alert_problems(voice, facts) else message, "llm"

    def false_alarm_note(self, species: str, wall_time: float) -> str:
        animal = local_name(self.cfg, species, self.language) if self.language in NON_LATIN else species.replace("_", " ")
        return FALSE_ALARM_NOTES.get(self.language, FALSE_ALARM_NOTES["en"]).format(
            time=time.strftime("%H:%M", time.localtime(wall_time)), animal=animal)

    def send(self, summary: dict[str, Any]) -> dict[str, Any]:
        return self.deliver(*self.compose(summary))

    def deliver(self, message: str, voice_text: str, source: str) -> dict[str, Any]:
        out = {"message": message, "voice_text": voice_text, "source": source, "channels": []}
        if "console" in self.channels:
            print(f"\n[FARMER ALERT/{self.language}] {message}\n", flush=True)
            out["channels"].append("console")
        if "telegram" in self.channels:
            if self._telegram_text(message):
                out["channels"].append("telegram")
            if self.voice and self._telegram_voice(voice_text):
                out["channels"].append("telegram_voice")
        self.sent.append(out)
        return out

    # ---- telegram ----------------------------------------------------------------------
    def _tg(self) -> tuple[str, str] | None:
        tok, chat = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
        return (tok, chat) if tok and chat else None

    def _telegram_text(self, text: str) -> bool:
        creds = self._tg()
        if not creds:
            log.warning("Telegram not configured")
            return False
        try:
            r = requests.post(f"https://api.telegram.org/bot{creds[0]}/sendMessage",
                              data={"chat_id": creds[1], "text": text}, timeout=15)
            return r.ok
        except requests.RequestException as exc:
            log.warning("Telegram send failed: %s", type(exc).__name__)  # str(exc) has the bot token URL
            return False

    def _telegram_voice(self, text: str) -> bool:
        creds = self._tg()
        if not creds:
            return False
        try:
            from gtts import gTTS  # pip install gTTS (needs internet)
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as fh:
                gTTS(text=text, lang=self.language if self.language in ("hi", "mr", "en") else "en"
                     ).write_to_fp(fh)
                path = fh.name
            with open(path, "rb") as audio:
                r = requests.post(f"https://api.telegram.org/bot{creds[0]}/sendAudio",
                                  data={"chat_id": creds[1]}, files={"audio": audio}, timeout=30)
            return r.ok
        except Exception as exc:
            log.warning("Voice note failed: %s", type(exc).__name__)  # str(exc) can hold the bot token URL
            return False
