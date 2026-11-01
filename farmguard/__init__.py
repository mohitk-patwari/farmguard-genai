"""FarmGuard: a two-speed GenAI agent for animal intrusion deterrence.

Layer 1 (farmguard.fast)  - every frame, no network / LLM calls: detect, vote, sound the alarm.
Layer 2 (farmguard.agent) - once per intrusion event: VLM check, retreat check, LLM policy,
                            farmer alert. Never blocks Layer 1.
"""
import sys

__version__ = "0.1.0"

# Redirected stdout on Windows defaults to a legacy code page (cp1252): printing a Hindi/Marathi
# alert raised UnicodeEncodeError and lost the event. Here so the CLI and Streamlit both get it.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")
