"""Every sound the policy may choose must exist, play, and have a documented source."""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from farmguard.config import load_config
from farmguard.tools import make_sounds

ROOT = Path(__file__).resolve().parent.parent
SOUNDS = sorted({s for info in load_config().species.values() for s in info.allowed_sounds})


@pytest.mark.parametrize("sound", SOUNDS)
def test_allowed_sound_is_a_real_audible_wav_with_a_source(sound):
    with wave.open(str(ROOT / "sounds" / f"{sound}.wav")) as w:
        assert w.getnchannels() == 1 and w.getsampwidth() == 2
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        assert 0.3 < len(x) / w.getframerate() <= 10
    assert np.abs(x).max() > 3000  # not silent
    assert f"`{sound}`" in (ROOT / "sounds" / "SOURCES.md").read_text(encoding="utf-8")


def test_make_sounds_does_not_overwrite_a_recording(tmp_path):
    keep = tmp_path / "siren.wav"
    keep.write_bytes(b"real recording")
    make_sounds(tmp_path, seconds=0.1)
    assert keep.read_bytes() == b"real recording"
    assert (tmp_path / "drum_beat.wav").stat().st_size > 100
