"""Replace placeholder deterrent sounds with licensed real recordings; writes sounds/SOURCES.md.

    python scripts/fetch_sounds.py

Takes the loudest few seconds of each recording, mono, peak-normalised, 16-bit WAV named <sound id>.wav.
Sounds with no licensed recording stay synthetic and are listed as PLACEHOLDER: no source is better
than a wrong one. Originals go to data/raw/sounds/ (not committed).
"""
from __future__ import annotations

import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
UA = {"User-Agent": "FarmGuardResearch/0.1 (academic project)"}
CLIP_SECONDS = 6.0
COMMONS = "https://upload.wikimedia.org/wikipedia/commons/"

# caveat = what this recording is NOT. Read these before citing a sound in the paper.
SOURCES = {
    "bee_swarm": {
        "url": COMMONS + "1/11/263673_ylearkisto_mehilainen-tarhamehilainen-parvi-bees-honeybees-a-swarm-of-bees-buzzing-among-the-flowers-apis-mellifera.wav",
        "page": "https://commons.wikimedia.org/wiki/File:263673_ylearkisto_mehilainen-tarhamehilainen-parvi-bees-honeybees-a-swarm-of-bees-buzzing-among-the-flowers-apis-mellifera.wav",
        "title": "Honeybees buzzing among flowers (Apis mellifera)", "author": "YleArkisto", "licence": "CC BY 3.0",
        "caveat": "Foraging bees, not a disturbed hive. Elephant studies use disturbed bees, so this may be weaker."},
    "tiger_roar": {
        "url": COMMONS + "2/29/439280_schots_angry-tiger.wav",
        "page": "https://commons.wikimedia.org/wiki/File:439280_schots_angry-tiger.wav",
        "title": "Angry tiger", "author": "schots", "licence": "CC0",
        "caveat": "A caged tiger growling and snarling with heavy reverb, not an open-air roar."},
    "boar_alarm": {
        "url": "https://static.inaturalist.org/sounds/449664.wav",
        "page": "https://www.inaturalist.org/observations/118517713",
        "title": "Wild boar group vocalising (Sus scrofa)", "author": "iNaturalist observation 118517713 (CC0)", "licence": "CC0",
        "caveat": "NOT a verified alarm call: the recordist describes a noisy group, possibly at feeding time. "
                  "Do not cite the ZHAW alarm-call result for this file."},
    "gunshot": {
        "url": COMMONS + "2/25/Firecracker_sounds_2022-12-31T23-25-40.ogg",
        "page": "https://commons.wikimedia.org/wiki/File:Firecracker_sounds_2022-12-31T23-25-40.ogg",
        "title": "Firecracker sounds", "author": "Almanta", "licence": "CC0",
        "caveat": "Firecrackers, not gunfire (species.yaml labels this id 'Gunshot / firecracker'). Distant shouts in the background."},
    "dog_bark": {
        "url": COMMONS + "5/58/Barking_of_a_dog_2.ogg",
        "page": "https://commons.wikimedia.org/wiki/File:Barking_of_a_dog_2.ogg",
        "title": "Barking of a dog 2", "author": "Amada44", "licence": "CC BY-SA 3.0",
        "caveat": "Trimmed copy stays under CC BY-SA 3.0."},
    "siren": {
        "url": COMMONS + "6/69/Civil-defense-siren-waver.ogg",
        "page": "https://commons.wikimedia.org/wiki/File:Civil-defense-siren-waver.ogg",
        "title": "Civil defense siren, wavering", "author": "Techtonic", "licence": "Public domain",
        "caveat": "Recorded at a distance in St. Paul, Minnesota."},
    "drum_beat": {
        "url": COMMONS + "0/09/DHOL_BEAT.wav",
        "page": "https://commons.wikimedia.org/wiki/File:DHOL_BEAT.wav",
        "title": "Dhol beat", "author": "Johnny Kalsi (The Dhol Foundation)", "licence": "CC BY 3.0",
        "caveat": ""},
}
# Searched Commons and iNaturalist on 2026-11-01. Freesound would likely have both, but needs an API key.
PLACEHOLDERS = {
    "leopard_growl": "No licensed recording found (iNaturalist has 0 CC sounds for Panthera pardus; Commons has only speech and music).",
    "human_shout": "No clean recording found (Commons has only arena crowds and film screams). Easiest fix: record your own voice.",
}


def loudest_window(x: np.ndarray, sr: int, seconds: float) -> np.ndarray:
    """The `seconds`-long stretch with the most energy; the whole signal if it is shorter."""
    n = int(sr * seconds)
    if len(x) <= n:
        return x
    energy = np.concatenate(([0.0], np.cumsum(x.astype(np.float64) ** 2)))
    start = int(np.argmax(energy[n:] - energy[:-n]))
    return x[start:start + n]


def finish(x: np.ndarray, sr: int) -> np.ndarray:
    """Peak-normalise to 0.9, with 20 ms fades so a cut does not click."""
    x = x / (np.max(np.abs(x)) + 1e-9) * 0.9
    fade = min(int(sr * 0.02), len(x) // 2)
    ramp = np.linspace(0, 1, fade)
    x[:fade] *= ramp
    x[-fade:] *= ramp[::-1]
    return x


def _download(url: str, path: Path) -> None:
    if path.exists() and path.stat().st_size:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(6):
        time.sleep(3)  # Commons rate-limits scripted downloads
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
                path.write_bytes(r.read())
            return
        except Exception as exc:
            print(f"retry {path.name}: {exc}", file=sys.stderr)
            time.sleep(30 * (attempt + 1))
    raise RuntimeError(f"could not download {url}")


def main() -> None:
    raw = ROOT / "data" / "raw" / "sounds"
    for sound, s in SOURCES.items():
        orig = raw / sound / ("original" + Path(s["url"]).suffix)
        _download(s["url"], orig)
        x, sr = sf.read(str(orig), always_2d=True, dtype="float32")
        out = finish(loudest_window(x.mean(axis=1), sr, CLIP_SECONDS), sr)
        sf.write(str(ROOT / "sounds" / f"{sound}.wav"), out, sr, subtype="PCM_16")
        print(f"{sound}: {len(out) / sr:.1f} s at {sr} Hz")
    lines = ["# Deterrent sound sources", "",
             ("Made by `scripts/fetch_sounds.py`. Each file is the loudest part of the original, "
             f"at most {CLIP_SECONDS:.0f} s, mono, peak-normalised."),
             "None of these has been tested on animals. Whether a sound deters anything is a field question.", "",
             "| Sound id | Recording | Author | Licence | Caveat |", "|---|---|---|---|---|"]
    for sound, s in SOURCES.items():
        lines.append(f"| `{sound}` | [{s['title']}]({s['page']}) | {s['author']} | {s['licence']} | {s['caveat']} |")
    for sound, why in PLACEHOLDERS.items():
        lines.append(f"| `{sound}` | **PLACEHOLDER**: synthetic tone from `farmguard make-sounds` | n/a | n/a | {why} Must not be presented as a real recording. |")
    (ROOT / "sounds" / "SOURCES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
