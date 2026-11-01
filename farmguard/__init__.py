"""FarmGuard: a two-speed GenAI agent for animal intrusion deterrence.

Layer 1 (farmguard.fast)  - every frame, no network / LLM calls: detect, vote, sound the alarm.
Layer 2 (farmguard.agent) - once per intrusion event: VLM check, retreat check, LLM policy,
                            farmer alert. Never blocks Layer 1.
"""

__version__ = "0.1.0"
