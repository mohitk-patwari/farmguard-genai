"""Minimal labelling helper. Writes data/labels/*.csv in the formats in data/labels/README.md.

  python scripts/label.py split                                  assign clips to dev/test
  python scripts/label.py vlm c01 --labeller a                   mark frames: box, species, behaviour
  python scripts/label.py retreat c01 --labeller a               mark windows: status at the end
  python scripts/label.py intrusions c01 --labeller a            mark true intrusion intervals
  python scripts/label.py vlm --labeller b --from data/labels/vlm_samples.a.csv
                                                                 second labeller: same samples, own labels
  python scripts/label.py kappa vlm a b                          agreement between labellers
  python scripts/label.py merge vlm a b                          agreed rows -> vlm_samples.csv

Keys: a/d = one frame back/forward, z/c = 10 frames, l = label here (vlm), [ and ] = window start and
end (retreat, intrusions), q = quit. Every row is saved at once, so quitting loses nothing.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2

from farmguard.config import load_config
from farmguard.eval.labels import (
    LABEL_COLUMNS,
    agreement,
    assign_splits,
    load_splits,
    merge,
    read_rows,
    write_rows,
)

FRAMES = Path("data/frames")
LABELS = Path("data/labels")
BEHAVIOURS = ["eating", "moving_through", "resting", "leaving", "absent"]
STATUSES = ["gone", "leaving", "still_present"]
FIELDS = {"vlm": ["sample_id", "frames", "trigger_index", "bbox", "species", "behaviour"],
          "retreat": ["sample_id", "frames", "species", "status"],
          "intrusions": ["clip", "start_s", "end_s", "species"]}
OUT_NAME = {"vlm": "vlm_samples", "retreat": "retreat_samples", "intrusions": "intrusions"}


def append_row(path: Path, row: dict[str, str], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerow(row)


def burst(files: list[str], i: int, step: int = 2, n: int = 1) -> tuple[list[str], int]:
    """Frame i with neighbours `step` apart, in time order, as in the README example.
    Returns (paths, index of frame i among them)."""
    picks = [j for j in range(i - n * step, i + n * step + 1, step) if 0 <= j < len(files)]
    return [files[j] for j in picks], picks.index(i)


def choose(img, title: str, options: list[str]) -> str | None:
    """Show numbered options over the frame; a digit picks one, Esc cancels."""
    canvas = img.copy()
    for k, opt in enumerate(options):
        cv2.putText(canvas, f"{(k + 1) % 10}: {opt}", (10, 30 + 24 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 255), 2)
    cv2.imshow(title, canvas)
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == 27:
            return None
        if ord("0") <= key <= ord("9"):
            n = (key - ord("1")) % 10
            if n < len(options):
                return options[n]


def label_clip(kind: str, clip: str, labeller: str, species: list[str]) -> None:
    manifest = read_rows(FRAMES / clip / "manifest.csv")
    files = [r["file"] for r in manifest]
    out = LABELS / f"{OUT_NAME[kind]}.{labeller}.csv"
    title = f"{kind} {clip} ({labeller})"
    i, start = 0, None
    while True:
        img = cv2.imread(files[i])
        info = img.copy()
        cv2.putText(info, f"{i + 1}/{len(files)} t={manifest[i]['time_s']}s start={start}",
                    (10, img.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow(title, info)
        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            break
        i = max(0, min(len(files) - 1, i + {ord("a"): -1, ord("d"): 1, ord("z"): -10, ord("c"): 10}.get(key, 0)))
        if kind == "vlm" and key == ord("l"):
            x, y, w, h = cv2.selectROI(title, img, showCrosshair=False)
            sp = choose(img, title, species + ["other_animal", "none"])
            beh = sp and choose(img, title, BEHAVIOURS)
            if beh:
                paths, trig = burst(files, i)
                append_row(out, {"sample_id": f"{clip}_f{manifest[i]['frame']}", "frames": ";".join(paths),
                                 "trigger_index": str(trig), "bbox": f"{x},{y},{x + w},{y + h}" if w and h else "",
                                 "species": sp, "behaviour": beh}, FIELDS[kind])
        elif kind != "vlm" and key == ord("["):
            start = i
        elif kind != "vlm" and key == ord("]") and start is not None and i > start:
            sp = choose(img, title, species)
            if sp and kind == "retreat":
                status = choose(img, title, STATUSES)
                if status:
                    append_row(out, {"sample_id": f"{clip}_w{manifest[start]['frame']}",
                                     "frames": ";".join(files[start:i + 1]), "species": sp, "status": status},
                               FIELDS[kind])
            elif sp:
                append_row(out, {"clip": manifest[0]["clip"], "start_s": manifest[start]["time_s"],
                                 "end_s": manifest[i]["time_s"], "species": sp}, FIELDS[kind])
            start = None
    cv2.destroyWindow(title)


def relabel(kind: str, source: str, labeller: str, species: list[str]) -> None:
    """Independent second pass: same samples and boxes, the first labeller's answers are not shown."""
    out = LABELS / f"{OUT_NAME[kind]}.{labeller}.csv"
    done = {r["sample_id"] for r in read_rows(out)} if out.exists() else set()
    title = f"{kind} ({labeller})"
    for row in read_rows(source):
        if row["sample_id"] in done:
            continue
        paths = row["frames"].split(";")
        img = cv2.imread(paths[-1] if kind == "retreat" else paths[int(row["trigger_index"])])
        if row.get("bbox"):
            x1, y1, x2, y2 = (int(float(v)) for v in row["bbox"].split(","))
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        if kind == "vlm":
            sp = choose(img, title, species + ["other_animal", "none"])
            beh = sp and choose(img, title, BEHAVIOURS)
            if not beh:
                break
            append_row(out, {**row, "species": sp, "behaviour": beh}, FIELDS[kind])
        else:
            status = choose(img, title, STATUSES)
            if not status:
                break
            append_row(out, {**row, "status": status}, FIELDS[kind])
    cv2.destroyWindow(title)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["split", "vlm", "retreat", "intrusions", "kappa", "merge"])
    p.add_argument("args", nargs="*", help="clip name; or for kappa/merge: kind labeller_a labeller_b")
    p.add_argument("--labeller", default="a")
    p.add_argument("--from", dest="source", help="second labeller: label the samples in this CSV")
    p.add_argument("--test-fraction", type=float, default=0.3)
    a = p.parse_args(argv)
    if a.cmd == "split":
        path = LABELS / "splits.csv"
        clips = [d.name for d in FRAMES.iterdir() if d.is_dir()] if FRAMES.exists() else []
        rows = assign_splits(clips, load_splits(path), a.test_fraction)
        write_rows(path, sorted(rows.values(), key=lambda r: r["clip"]), ["clip", "split", "lighting"])
        print(f"{path}: {len(rows)} clips. Fill the lighting column (day/night) by hand.")
    elif a.cmd in ("kappa", "merge"):
        kind, x, y = a.args
        fa, fb = (LABELS / f"{OUT_NAME[kind]}.{who}.csv" for who in (x, y))
        print(agreement(fa, fb, LABEL_COLUMNS[kind]))
        if a.cmd == "merge":
            print(merge(fa, fb, LABEL_COLUMNS[kind], LABELS / f"{OUT_NAME[kind]}.csv",
                        LABELS / f"{OUT_NAME[kind]}.conflicts.csv"))
    else:
        species = list(load_config().species)
        if a.source:
            relabel(a.cmd, a.source, a.labeller, species)
        else:
            for clip in a.args:
                label_clip(a.cmd, clip, a.labeller, species)


if __name__ == "__main__":
    main()
