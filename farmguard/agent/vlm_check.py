"""VLM check: species, count and behaviour (eating vs passing) from the event images."""
from __future__ import annotations

import logging

import cv2

from farmguard.agent.clip_builder import to_jpeg
from farmguard.agent.llm.base import LLMClient
from farmguard.agent.prompts import SYSTEM_VISION, fewshot_note, vlm_prompt, vlm_schema
from farmguard.config import Config
from farmguard.events import VLMReport

log = logging.getLogger(__name__)
MAX_COUNT = 50  # the schema's limit; not every provider enforces numeric limits


class VLMChecker:
    def __init__(self, cfg: Config, client: LLMClient):
        self.cfg = cfg
        self.client = client
        self.species_keys = list(cfg.species)
        self._fewshot: tuple[list[bytes], list[str]] | None = None

    def _fewshot_examples(self) -> tuple[list[bytes], list[str]]:
        """Labelled frames from <agent.fewshot.dir>/<category>/. `none` is a category too: the
        91.0% result above was the empty-frame filter. Loaded once. Read per check(), like the
        location flag, so E2 can flip agent.fewshot.enabled between conditions."""
        if not self.cfg.get("agent.fewshot.enabled", False):
            return [], []
        if self._fewshot is None:
            root = self.cfg.path("agent.fewshot.dir")
            per = self.cfg.get("agent.fewshot.per_species", 3)
            side = self.cfg.get("agent.fewshot.max_side", 384)  # 15+ extra images per call: keep them small
            images, labels = [], []
            for cat in self.species_keys + ["none", "other_animal"]:
                files = sorted(p for p in (root / cat).glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
                decoded = [img for img in (cv2.imread(str(p)) for p in files[:per]) if img is not None]
                if cat in self.species_keys and len(decoded) < 3:
                    log.warning("Few-shot: only %d example(s) for %s in %s", len(decoded), cat, root)
                images += [to_jpeg(img, side) for img in decoded]
                labels += [cat] * len(decoded)
            self._fewshot = (images, labels)
        return self._fewshot

    def check(self, images: list[bytes], layout: str, detector_species: str | None,
              location_hint: str | None = None) -> VLMReport:
        if location_hint and not self.cfg.get("agent.include_location_in_prompt", False):
            location_hint = None  # rule R1: never leak location unless running the ablation
        refs, labels = self._fewshot_examples()
        prompt = vlm_prompt(self.species_keys, len(images), layout, location_hint,
                            fewshot_note(labels) if labels else None)
        res = self.client.generate_json(prompt, vlm_schema(self.species_keys), refs + images, SYSTEM_VISION)
        if res.data is None:
            # Fallback: trust the detector so the pipeline keeps going (reliability requirement)
            log.warning("VLM check failed (%s); using the detector label %s", (res.error or "")[:300], detector_species)
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
            count = min(MAX_COUNT, max(0, int(d.get("count", 1))))
        except (TypeError, ValueError):
            count = 1
        return VLMReport(species=species, count=count, behaviour=behaviour,
                         confidence=str(d.get("confidence", "low")),
                         observation=str(d.get("observation", ""))[:500],
                         note=str(d.get("note", ""))[:300], raw=res.raw, latency_s=res.latency_s)
