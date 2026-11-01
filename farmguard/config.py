"""Configuration loading: settings.yaml + species.yaml + .env."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (avoids a hard dependency on python-dotenv)."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.split(" #", 1)[0].strip().strip('"').strip("'")
        os.environ.setdefault(key.strip(), value)


@dataclass
class SpeciesInfo:
    key: str
    aliases: list[str]
    hindi: str
    marathi: str
    danger: int
    default_sound: str
    allowed_sounds: list[str]


@dataclass
class Config:
    raw: dict[str, Any]
    species: dict[str, SpeciesInfo]
    sounds: dict[str, dict[str, Any]]
    non_targets: list[str]
    root: Path = ROOT

    # ---- convenience accessors -------------------------------------------------
    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.raw
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def path(self, dotted: str) -> Path:
        p = Path(self.get(dotted))
        return p if p.is_absolute() else self.root / p

    def species_for_label(self, label: str) -> str | None:
        """Map a detector / VLM label ("wild boar", "cow") to a species key ("wild_boar")."""
        label = (label or "").strip().lower().replace("_", " ")
        if not label:
            return None
        for key, info in self.species.items():
            if label == key.replace("_", " ") or label in info.aliases:
                return key
        for key, info in self.species.items():  # loose match: "a wild boar" -> wild_boar
            if any(a in label for a in info.aliases):
                return key
        return None


def load_config(settings: str | Path | None = None, species: str | Path | None = None,
                overrides: dict[str, Any] | None = None) -> Config:
    _load_dotenv(ROOT / ".env")
    settings_path = Path(settings) if settings else ROOT / "config" / "settings.yaml"
    species_path = Path(species) if species else ROOT / "config" / "species.yaml"
    raw = yaml.safe_load(settings_path.read_text(encoding="utf-8")) or {}
    for dotted, value in (overrides or {}).items():
        node = raw
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    sp = yaml.safe_load(species_path.read_text(encoding="utf-8")) or {}
    species_map = {
        key: SpeciesInfo(
            key=key,
            aliases=[a.lower() for a in val.get("aliases", [])],
            hindi=val.get("hindi", key),
            marathi=val.get("marathi", key),
            danger=int(val.get("danger", 1)),
            default_sound=val["default_sound"],
            allowed_sounds=list(val["allowed_sounds"]),
        )
        for key, val in (sp.get("species") or {}).items()
    }
    for key, info in species_map.items():
        unknown = [s for s in info.allowed_sounds if s not in (sp.get("sounds") or {})]
        if unknown:
            raise ValueError(f"species.yaml: {key} uses undefined sounds {unknown}")
        if info.default_sound not in info.allowed_sounds:
            raise ValueError(f"species.yaml: {key}.default_sound must be in allowed_sounds")
    return Config(raw=raw, species=species_map, sounds=sp.get("sounds") or {},
                  non_targets=[n.lower() for n in sp.get("non_targets", [])])
