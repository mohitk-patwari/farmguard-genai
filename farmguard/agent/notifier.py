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


def template_message(summary: dict[str, Any], language: str) -> str:
    st = STATUS_WORDS.get(language, STATUS_WORDS["en"]).get(summary["final_status"], "")
    sounds = ", ".join(summary["sounds_played"])
    if language == "hi":
        msg = (f"{summary['time']} बजे खेत में {summary['animal_local']} आया। "
               f"बजाई गई आवाज़ें: {sounds}। जानवर {st}।")
        if summary["final_status"] not in ("gone", "leaving"):
            msg += " कृपया खेत देखें।"
        return msg
    if language == "mr":
        msg = (f"{summary['time']} वाजता शेतात {summary['animal_local']} आले. "
               f"वाजवलेले आवाज: {sounds}. प्राणी {st}.")
        if summary["final_status"] not in ("gone", "leaving"):
            msg += " कृपया शेत तपासा."
        return msg
    msg = (f"At {summary['time']} a {summary['animal']} entered the field. Sounds played: "
           f"{sounds}. The animal {st}.")
    if summary["final_status"] not in ("gone", "leaving"):
        msg += " Please check the field."
    return msg


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
        info = self.cfg.species.get(species)
        local = {"hi": info.hindi, "mr": info.marathi}.get(self.language, species) if info else species
        labels = [self.cfg.sounds.get(s, {}).get("label", s) for s in sounds]
        return {"animal": species.replace("_", " "), "animal_local": local, "count": count,
                "time": time.strftime("%H:%M", time.localtime(wall_time)),
                "behaviour_when_seen": behaviour, "sounds_played": labels,
                "attempts": attempts, "final_status": final_status}

    def compose(self, summary: dict[str, Any]) -> tuple[str, str, str]:
        """Returns (message, voice_text, source)."""
        if self.client is not None:
            res = self.client.generate_json(alert_prompt(summary, self.language), alert_schema())
            if res.data and res.data.get("message"):
                return str(res.data["message"]), str(res.data.get("voice_text") or res.data["message"]), "llm"
        msg = template_message(summary, self.language)
        return msg, msg, "template"

    def send(self, summary: dict[str, Any]) -> dict[str, Any]:
        message, voice_text, source = self.compose(summary)
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
            log.warning("Telegram send failed: %s", exc)
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
            log.warning("Voice note failed: %s", exc)
            return False
