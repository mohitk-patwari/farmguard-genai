"""Classroom demo: upload a farm video (or use the synthetic demo), watch the fast path box the
animal and fire the alarm, then watch the GenAI agent verify, retry and write the farmer alert.

Run:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
import tempfile
import threading
import time
from pathlib import Path

import cv2
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from farmguard.config import load_config
from farmguard.fast.pipeline import annotate
from farmguard.fast.video_source import VideoSource
from farmguard.system import build_system, run_video
from farmguard.tools import make_demo, make_sounds

st.set_page_config(page_title="FarmGuard GenAI", layout="wide")
st.title("FarmGuard: GenAI agent for animal intrusion deterrence")
st.caption("Layer 1 sounds the alarm on every frame without any AI call. Layer 2 (GenAI) then "
           "verifies the animal, checks whether it left, picks the next sound and alerts the farmer.")

with st.sidebar:
    st.header("Settings")
    provider = st.selectbox("GenAI provider", ["mock", "ollama", "gemini", "anthropic", "openai_compat"])
    vlm_model = st.text_input("VLM model", "qwen2.5vl:7b" if provider == "ollama" else "")
    policy = st.selectbox("Deterrent policy", ["llm", "bandit", "rule_chain", "random", "fixed"])
    language = st.selectbox("Farmer language", ["hi", "mr", "en"])
    use_demo = st.checkbox("Use synthetic demo video (no model weights needed)", value=True)
    upload = None if use_demo else st.file_uploader("Farm video", type=["mp4", "avi", "mov", "mkv"])
    start = st.button("Run", type="primary")

col_video, col_side = st.columns([3, 2])
frame_slot = col_video.empty()
status_slot = col_video.empty()
side_slot = col_side.empty()


class Shared:
    def __init__(self):
        self.lock = threading.Lock()
        self.frame = None
        self.events: list = []
        self.results: list = []
        self.done = False
        self.error: str | None = None


if start:
    if not (ROOT / "sounds" / "siren.wav").exists():
        make_sounds(ROOT / "sounds")
    overrides = {"agent.provider": provider, "agent.policy.type": policy,
                 "notifier.language": language, "alarm.backend": "none"}
    if vlm_model:
        overrides["agent.vlm_model"] = vlm_model
        overrides["agent.llm_model"] = vlm_model
    if use_demo:
        demo = make_demo(ROOT / "data" / "demo")
        source = demo["video"]
        overrides.update({"detector.backend": "mock", "detector.mock_annotations": demo["annotations"],
                          "zones.file": demo["zones"]})
    elif upload is not None:
        tmp = Path(tempfile.mkdtemp()) / upload.name
        tmp.write_bytes(upload.read())
        source = str(tmp)
    else:
        st.warning("Upload a video or tick the demo option.")
        st.stop()

    cfg = load_config(overrides=overrides)
    shared = Shared()

    def on_frame(pkt, new_events):
        img = annotate(pkt, system.fast.zones)
        with shared.lock:
            shared.frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            shared.events.extend(new_events)

    def on_result(res):
        with shared.lock:
            shared.results.append(res)

    system = build_system(cfg, on_frame=on_frame, on_result=on_result)

    def worker():
        try:
            src = VideoSource(source, live=False, target_fps=cfg.get("video.target_fps", 10),
                              resize_width=cfg.get("video.resize_width", 960))
            run_video(system, src.frames(), realtime=True)
            src.release()
        except Exception as exc:  # surface errors in the UI
            shared.error = repr(exc)
        finally:
            shared.done = True

    threading.Thread(target=worker, daemon=True).start()
    while True:
        with shared.lock:
            frame, events, results = shared.frame, list(shared.events), list(shared.results)
        if frame is not None:
            frame_slot.image(frame, width="stretch")
        status_slot.info(f"Intrusion events: {len(events)} | agent results: {len(results)}")
        with side_slot.container():
            for ev in events:
                st.error(f"ALARM ({ev.first_sound}) - {ev.species} entered '{ev.zone}' at "
                         f"t={ev.video_time:.1f}s")
            for r in results:
                st.subheader(f"Event {r.event['event_id']}: {r.final_status}")
                if r.vlm:
                    st.write(f"**VLM:** {r.vlm['species']} x{r.vlm['count']}, {r.vlm['behaviour']} "
                             f"({r.vlm['confidence']}) - {r.vlm['observation']}")
                for a in r.attempts:
                    st.write(f"{a.attempt_no}. **{a.sound}** via {a.source} -> {a.status} "
                             f"(tracker {a.tracker_status}, VLM {a.vlm_status}) - {a.reason}")
                if r.farmer:
                    st.success(r.farmer["message"])
        if shared.done:
            break
        time.sleep(0.3)
    if shared.error:
        st.exception(RuntimeError(shared.error))
    st.write("Stage timing (ms):", system.fast.timer.summary())
