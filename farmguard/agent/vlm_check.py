"""VLM check: species, count and behaviour (eating vs passing) from the event images."""
from __future__ import annotations

import logging

from farmguard.agent.llm.base import LLMClient
from farmguard.agent.prompts import SYSTEM_VISION, vlm_prompt, vlm_schema
from farmguard.config import Config
from farmguard.events import VLMReport

log = logging.getLogger(__name__)


class VLMChecker:
    def __init__(self, cfg: Config, client: LLMClient):
        self.cfg = cfg
        self.client = client
        self.species_keys = list(cfg.species)

    def check(self, images: list[bytes], layout: str, detector_species: str | None,
              location_hint: str | None = None) -> VLMReport:
        if location_hint and not self.cfg.get("agent.include_location_in_prompt", False):
            location_hint = None  # rule R1: never leak location unless running the ablation
        prompt = vlm_prompt(self.species_keys, len(images), layout, location_hint)
        res = self.client.generate_json(prompt, vlm_schema(self.species_keys), images, SYSTEM_VISION)
        if res.data is None:
            # Fallback: trust the detector so the pipeline keeps going (reliability requirement)
            return VLMReport(species=detector_species or "unclassifiable", count=1,
                             behaviour="unclear", confidence="low", observation="",
                             note="VLM unavailable; using detector label", raw=res.raw,
                             latency_s=res.latency_s, error=res.error)
        d = res.data
        species = str(d.get("species", "unclassifiable"))
        if species not in self.species_keys + ["other_animal", "none", "unclassifiable"]:
            species = self.cfg.species_for_label(species) or "unclassifiable"
        behaviour = d.get("behaviour", "unclear")
        if behaviour not in ("eating", "moving_through", "resting", "leaving", "absent", "unclear"):
            behaviour = "unclear"
        try:
            count = max(0, int(d.get("count", 1)))
        except (TypeError, ValueError):
            count = 1
        return VLMReport(species=species, count=count, behaviour=behaviour,
                         confidence=str(d.get("confidence", "low")),
                         observation=str(d.get("observation", ""))[:500],
                         note=str(d.get("note", ""))[:300], raw=res.raw, latency_s=res.latency_s)
