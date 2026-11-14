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
           "uncertain": "स्थिति साफ़ नहीं है", "truncated": "स्थिति साफ़ नहीं है"},
    "mr": {"gone": "शेतातून निघून गेले", "leaving": "शेतातून जात होते", "still_present": "अजूनही शेतात आहे",
           "uncertain": "स्थिती स्पष्ट नाही", "truncated": "स्थिती स्पष्ट नाही"},
    "en": {"gone": "left the field", "leaving": "was leaving the field",
           "still_present": "is still in the field", "uncertain": "status is unclear",
           "truncated": "status is unclear"},  # truncated: the video ended before the check could see
}


SOUND_NAMES = {
    "hi": {"bee_swarm": "मधुमक्खियों की आवाज़", "tiger_roar": "बाघ की आवाज़", "leopard_growl": "तेंदुए की आवाज़",
           "gunshot": "पटाखे की आवाज़", "dog_bark": "कुत्ते की आवाज़", "boar_alarm": "सूअर की आवाज़",
           "siren": "सायरन", "drum_beat": "ढोल", "human_shout": "आदमी के चिल्लाने की आवाज़"},
    "mr": {"bee_swarm": "मधमाश्यांचा आवाज", "tiger_roar": "वाघाचा आवाज", "leopard_growl": "बिबट्याचा आवाज",
           "gunshot": "फटाक्याचा आवाज", "dog_bark": "कुत्र्याचा आवाज", "boar_alarm": "डुकराचा आवाज",
           "siren": "सायरन", "drum_beat": "ढोल", "human_shout": "माणसाच्या ओरडण्याचा आवाज"},
}
# Stems for the grounding check (validators.alert_problems): any stem names the fact, so a reworded
# sound name passes. The same stems name what the alert must NOT mention. Not exhaustive; native-speaker check due.
STEMS = {
    "hi": {"species": {"wild_boar": ["सूअर", "सुअर"], "nilgai": ["नीलगाय"], "monkey": ["बंदर"],
                       "elephant": ["हाथी"], "stray_cattle": ["पशु", "गाय", "बैल", "भैंस", "मवेशी"]},
           "sounds": {"bee_swarm": ["मधुमक्ख"], "tiger_roar": ["बाघ"], "leopard_growl": ["तेंदु"],
                      "gunshot": ["पटाख", "बंदूक", "गोली"], "dog_bark": ["कुत्त"], "boar_alarm": ["सूअर", "सुअर"],
                      "siren": ["सायरन"], "drum_beat": ["ढोल"], "human_shout": ["चिल्ला", "आदमी", "इंसान"]},
           "status": {"left": ["चला गया", "चले गए", "चली गई", "जा रहा", "जा रहे", "जा रही", "निकल रहा"],
                      "still": ["अभी भी खेत में", "खेत में है", "खेत में ही"],
                      "unclear": ["साफ नहीं", "पता नहीं"]}},
    "mr": {"species": {"wild_boar": ["डुक्कर", "डुकरे"], "nilgai": ["नीलगाय", "रोही"], "monkey": ["माकड"],
                       "elephant": ["हत्ती"], "stray_cattle": ["जनावर", "गाय", "गाई", "बैल", "म्हैस"]},
           "sounds": {"bee_swarm": ["मधमाश"], "tiger_roar": ["वाघ"], "leopard_growl": ["बिबट"],
                      "gunshot": ["फटाक", "बंदूक"], "dog_bark": ["कुत्र"], "boar_alarm": ["डुकरा", "डुक्कराचा"],
                      "siren": ["सायरन"], "drum_beat": ["ढोल"], "human_shout": ["ओरड", "माणस", "माणूस"]},
           "status": {"left": ["निघून गेल", "जात होत", "जात आहे"], "still": ["अजूनही शेतात", "शेतात आहे"],
                      "unclear": ["स्पष्ट नाही"]}},
    "en": {"species": {}, "sounds": {},  # English adds species aliases and sound labels from the config
           "status": {"left": ["left the field", "has left", "leaving"], "still": ["still in the field", "still there"],
                      "unclear": ["unclear", "not clear"]}},
}
STATUS_FAMILY = {"gone": "left", "leaving": "left", "still_present": "still", "uncertain": "unclear",
                 "truncated": "unclear"}
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


# Added after the alert (LLM or template) when detector and VLM disagree about an immediate-escalation
# species, e.g. detector elephant vs camera check wild boar (smoke test 2). Fixed text: it is the safety part.
UNCERTAIN_NOTES = {
    "hi": "जानवर पक्का नहीं है (डिटेक्टर: {detector}, कैमरा जाँच: {vlm})।",
    "mr": "प्राणी नक्की नाही (डिटेक्टर: {detector}, कॅमेरा तपासणी: {vlm}).",
    "en": "Species uncertain (detector: {detector}, camera check: {vlm}).",
}
VLM_NON_SPECIES = {"none": {"hi": "कोई जानवर नहीं", "mr": "प्राणी नाही", "en": "no animal"},
                   "other_animal": {"hi": "कोई और जानवर", "mr": "दुसरा प्राणी", "en": "another animal"}}


def local_name(cfg: Config, species: str, language: str) -> str:
    if species in VLM_NON_SPECIES:
        return VLM_NON_SPECIES[species].get(language, VLM_NON_SPECIES[species]["en"])
    info = cfg.species.get(species)
    names = {"hi": info.hindi, "mr": info.marathi} if info else {}
    return names.get(language, species.replace("_", " "))


class Notifier:
    def __init__(self, cfg: Config, client: LLMClient | None, fallback: LLMClient | None = None):
        self.cfg = cfg
        self.client = client      # agent.notifier.provider, else the policy's client
        self.fallback = fallback  # tried when `client` fails (e.g. Gemini down -> Ollama), then the template
        self.language = cfg.get("notifier.language", "hi")
        self.channels = cfg.get("notifier.channels", ["console"])
        self.voice = cfg.get("notifier.voice_note", True)
        self.sent: list[dict[str, Any]] = []

    def build_summary(self, species: str, wall_time: float, sounds: list[str], final_status: str,
                      attempts: int, count: int, behaviour: str,
                      species_check: tuple[str, str] | None = None) -> dict[str, Any]:
        """species_check: (detector species, VLM species) when they disagree about a dangerous species."""
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
                "attempts": attempts, "final_status": final_status,
                "species_note": (UNCERTAIN_NOTES.get(self.language, UNCERTAIN_NOTES["en"]).format(
                    detector=local_name(self.cfg, species_check[0], self.language),
                    vlm=local_name(self.cfg, species_check[1], self.language)) if species_check else None)}

    def _facts(self, summary: dict[str, Any], template: str) -> AlertFacts:
        lang = self.language
        st = STEMS.get(lang, STEMS["en"])
        names = SOUND_NAMES.get(lang, {})

        def species_stems(key: str) -> list[str]:
            extra = [] if lang in NON_LATIN else [key.replace("_", " "), *self.cfg.species[key].aliases]
            return [local_name(self.cfg, key, lang), *st["species"].get(key, []), *extra]

        def sound_stems(key: str) -> list[str]:
            return [names.get(key) or self.cfg.sounds.get(key, {}).get("label", key), *st["sounds"].get(key, [])]

        family = STATUS_FAMILY.get(summary["final_status"], "unclear")
        forbidden = [w for k in self.cfg.species if k != summary["species"] for w in species_stems(k)]
        forbidden += [w for k in self.cfg.sounds if k not in summary["sound_ids"] for w in sound_stems(k)]
        forbidden += [w for f, ws in st["status"].items() if f != family for w in ws]
        return AlertFacts(species_stems(summary["species"]), summary["count"], summary["attempts"],
                          [sound_stems(s) for s in summary["sound_ids"]],
                          [summary["status_local"], *st["status"][family]], summary["time"], forbidden,
                          NUMBER_WORDS.get(lang, {}), max_chars=2 * len(template), latin_allowed=lang not in NON_LATIN)

    def compose(self, summary: dict[str, Any]) -> tuple[str, str, str]:
        """Returns (message, voice_text, source). source is llm, template (no usable answer) or
        template_ungrounded (the LLM answered and its facts were wrong)."""
        template = template_message(summary, self.language)
        res = None
        for client in (self.client, self.fallback):
            if client is None:
                continue
            res = client.generate_json(alert_prompt(summary, self.language), alert_schema())
            log.info("alert provider=%s llm_latency_s=%.3f ok=%d", client.name, res.latency_s, res.data is not None)
            if res.data is not None:
                break
        if res is None:
            return template, template, "template"
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
        message, voice, source = self.compose(summary)
        if summary.get("species_note"):  # outside the grounding check: fixed text, never from the LLM
            message, voice = f"{message} {summary['species_note']}", f"{voice} {summary['species_note']}"
        return self.deliver(message, voice, source)

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
