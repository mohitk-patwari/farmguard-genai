"""Build the FarmGuard YOLO dataset, split by clip so no scene is in both train and test.

Each source writes data/raw/<source>/manifest.jsonl, one image per line:
  {"image": path relative to the source dir, "cls": "boar"|"monkey"|"nilgai"|"cattle"|null (hard negative),
   "group": clip / observation / camera-sequence id, "boxes": [[xc, yc, w, h], ...] normalised or null,
   "label_source": "human"|"megadetector+human_species"|"yolo_world"|"none",
   "licence": ..., "author": ..., "url": ...}

    python scripts/prepare_data.py openimages --per-class 2000
    python scripts/prepare_data.py inat --per-class 1500
    python scripts/prepare_data.py lila --per-class 1500
    python scripts/prepare_data.py commons-clips    # openly licensed animal videos into data/clips/
    python scripts/prepare_data.py clips            # frames from data/clips/*, group = clip
    python scripts/prepare_data.py import-yolo DIR --map "wild boar=boar,pig=boar"   # a Roboflow export
    python scripts/prepare_data.py pseudo-label     # YOLO-World boxes for images with boxes=null
    python scripts/prepare_data.py build

The evaluation needs to know where each label came from, so build keeps label_source and a day/night
flag in data/yolo/manifest.csv. Machine labels are never mixed silently into the reported numbers.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import logging
import random
import re
import shutil
import sys
import time
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "yolo"
CLASSES = ["boar", "monkey", "nilgai", "cattle"]
UA = {"User-Agent": "FarmGuardResearch/0.1 (academic project)"}
log = logging.getLogger("prepare_data")

OPENIMAGES = {"Pig": "boar", "Monkey": "monkey", "Cattle": "cattle"}
OPENIMAGES_NEGATIVE = ["Person", "Dog", "Goat", "Sheep", "Horse", "Car", "Bicycle"]
INAT = {"nilgai": ("Boselaphus tragocamelus", "nilgai"), "boar": ("Sus scrofa", "boar"),
        "rhesus": ("Macaca mulatta", "monkey"), "bonnet": ("Macaca radiata", "monkey"),
        "langur": ("Semnopithecus", "monkey"), "cattle": ("Bos taurus", "cattle")}
INDIA_PLACE_ID = 6681


# ---- pure helpers (tested in tests/test_fast_prepare_data.py) ----------------------------------
def split_for(source: str, group: str, val: float = 0.1, test: float = 0.2) -> str:
    """Stable train/val/test choice from the group id: every frame of a clip gets the same answer."""
    h = int(hashlib.sha1(f"{source}/{group}".encode()).hexdigest(), 16) % 10_000 / 10_000
    return "test" if h < test else "val" if h < test + val else "train"


def is_night(bgr) -> bool:
    """Proxy for 'night': infrared (almost no colour) or very dark. No timestamps in these sources."""
    import cv2
    hsv = cv2.cvtColor(cv2.resize(bgr, (64, 64)), cv2.COLOR_BGR2HSV)
    return float(hsv[..., 1].mean()) < 12 or float(hsv[..., 2].mean()) < 45


def roboflow_group(stem: str) -> str:
    """Roboflow names copies 'orig_jpg.rf.<hash>': augmented copies of one image share a group, and
    frames named like 'video_00123' group by video. Their own split ignores this, so we re-split."""
    stem = stem.split(".rf.")[0]
    stem = re.sub(r"_(jpg|jpeg|png)$", "", stem)
    return re.sub(r"[-_ ]*(frame)?[-_ ]*\d+$", "", stem) or stem


def xyxy_to_xywhn(x1, y1, x2, y2, w, h) -> list[float]:
    return [(x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h]


def yolo_label_lines(boxes, cls_index: int) -> list[str]:
    return [f"{cls_index} " + " ".join(f"{min(max(v, 0.0), 1.0):.6f}" for v in b) for b in boxes]


# ---- download helpers ----------------------------------------------------------------------------
def _get(url: str, timeout: int = 60):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout)


def _json(url: str, pause: float = 1.1):
    time.sleep(pause)  # iNaturalist asks for about one request per second
    for attempt in range(5):
        try:
            with _get(url) as r:
                return json.load(r)
        except Exception as exc:
            log.warning("retry %s: %s", url[:80], exc)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"gave up on {url}")


def _download(args) -> bool:
    url, path = args
    if path.exists() and path.stat().st_size:
        return True
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    try:
        with _get(url, 40) as r, tmp.open("wb") as fh:
            shutil.copyfileobj(r, fh)
        tmp.replace(path)
        return True
    except Exception as exc:
        log.debug("failed %s: %s", url, exc)
        tmp.unlink(missing_ok=True)
        return False


def _download_all(rows: list[dict], src: Path, workers: int = 16) -> list[dict]:
    """Fetch every row's url to src/<image>; keep the rows that worked."""
    with ThreadPoolExecutor(workers) as ex:
        ok = list(ex.map(_download, [(r["url"], src / r["image"]) for r in rows]))
    kept = [r for r, good in zip(rows, ok) if good]
    log.info("%s: downloaded %d of %d", src.name, len(kept), len(rows))
    return kept


def _write_manifest(src: Path, rows: list[dict]) -> None:
    src.mkdir(parents=True, exist_ok=True)
    with (src / "manifest.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def _read_manifest(src: Path) -> list[dict]:
    with (src / "manifest.jsonl").open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# ---- sources -------------------------------------------------------------------------------------
def fetch_openimages(per_class: int, negatives: int, seed: int = 0) -> None:
    """Human-drawn boxes. Licence is per image (CC BY 2.0 as listed; Google does not verify it).
    'Pig' is mostly domestic pigs: evaluation reports wild-boar sources separately."""
    base = "https://storage.googleapis.com/openimages"
    names = {n: m for m, n in csv.reader(io.TextIOWrapper(
        _get(f"{base}/v7/oidv7-class-descriptions-boxable.csv"), encoding="utf-8"))}
    target = {names[k]: v for k, v in OPENIMAGES.items()}
    negative_ids = {names[k] for k in OPENIMAGES_NEGATIVE}
    boxes: dict[str, dict[str, list]] = {}  # image -> cls -> boxes
    neg_candidates: set[str] = set()
    image_split: dict[str, str] = {}
    # Streamed, not saved: the train CSV is 2.3 GB and we keep a tiny part.
    for split, url in (("train", f"{base}/v6/oidv6-train-annotations-bbox.csv"),
                       ("validation", f"{base}/v5/validation-annotations-bbox.csv"),
                       ("test", f"{base}/v5/test-annotations-bbox.csv")):
        log.info("streaming %s annotations", split)
        with _get(url, 300) as r:
            for row in csv.DictReader(io.TextIOWrapper(r, encoding="utf-8")):
                lab, img = row["LabelName"], row["ImageID"]
                if lab in negative_ids:
                    neg_candidates.add(img)
                    image_split.setdefault(img, split)
                if lab not in target:
                    continue
                image_split[img] = split
                if row["IsGroupOf"] == "1" or row["IsDepiction"] == "1":  # marks the image unusable
                    boxes.setdefault(img, {}).setdefault(target[lab], []).append(None)
                    continue
                x1, x2, y1, y2 = (float(row[k]) for k in ("XMin", "XMax", "YMin", "YMax"))
                boxes.setdefault(img, {}).setdefault(target[lab], []).append(
                    [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1])
    rng = random.Random(seed)
    rows: list[dict] = []
    for cls in sorted(set(target.values())):
        # One class per image and no drawings or group-of boxes, so the labels are complete.
        ok = [i for i, d in boxes.items() if list(d) == [cls] and None not in d[cls]]
        ok = sorted(ok)
        rng.shuffle(ok)
        log.info("openimages %s: %d usable", cls, len(ok))
        rows += [{"image": f"{i}.jpg", "cls": cls, "group": i, "boxes": boxes[i][cls],
                  "label_source": "human", "licence": "listed CC BY 2.0 (per image, unverified)",
                  "author": "", "url": f"https://s3.amazonaws.com/open-images-dataset/{image_split[i]}/{i}.jpg"}
                 for i in ok[:per_class]]
    negs = sorted(neg_candidates - set(boxes))
    rng.shuffle(negs)
    rows += [{"image": f"{i}.jpg", "cls": None, "group": i, "boxes": [], "label_source": "none",
              "licence": "listed CC BY 2.0 (per image, unverified)", "author": "",
              "url": f"https://s3.amazonaws.com/open-images-dataset/{image_split[i]}/{i}.jpg"}
             for i in negs[:negatives]]
    src = RAW / "openimages"
    _write_manifest(src, _download_all(rows, src))


def fetch_inat(per_class: int, seed: int = 0) -> None:
    """Research-grade, non-captive, CC0 / CC BY photos. The species is human-verified; there are no
    boxes, so pseudo-label adds them. India first, then worldwide up to per_class."""
    rows: list[dict] = []
    per_taxon = {"monkey": per_class // 3}  # three monkey taxa share one class
    for key, (name, cls) in INAT.items():
        want = per_taxon.get(cls, per_class)
        taxon = _json("https://api.inaturalist.org/v1/taxa?" + urllib.parse.urlencode(
            {"q": name, "per_page": 1}))["results"][0]["id"]
        got: dict[int, dict] = {}
        for place in (INDIA_PLACE_ID, None):
            id_below = None
            while len(got) < want:
                q = {"taxon_id": taxon, "photo_license": "cc0,cc-by", "quality_grade": "research",
                     "captive": "false", "per_page": 200, "order_by": "id", "order": "desc"}
                if place:
                    q["place_id"] = place
                if id_below:
                    q["id_below"] = id_below
                res = _json("https://api.inaturalist.org/v1/observations?" + urllib.parse.urlencode(q))["results"]
                if not res:
                    break
                id_below = res[-1]["id"]
                for o in res:
                    photo = next((p for p in o.get("photos", [])
                                  if p.get("license_code") in ("cc0", "cc-by")), None)
                    if photo and o["id"] not in got and len(got) < want:
                        got[o["id"]] = {
                            "image": f"{key}/{o['id']}.jpg", "cls": cls, "group": str(o["id"]),
                            "boxes": None, "label_source": "yolo_world",
                            "licence": photo["license_code"], "author": photo.get("attribution", ""),
                            "url": photo["url"].replace("square", "large"),
                            "page": f"https://www.inaturalist.org/observations/{o['id']}"}
            if len(got) >= want:
                break
        log.info("inat %s: %d observations", key, len(got))
        rows += got.values()
    src = RAW / "inat"
    _write_manifest(src, _download_all(rows, src))


def fetch_lila(per_class: int, seed: int = 0) -> None:
    """NACTI camera traps (CDLA-permissive): real night IR frames. Species labels are human; boxes
    come from LILA's MegaDetector v5a results, so label_source says so. Split by camera location."""
    src = RAW / "lila"
    src.mkdir(parents=True, exist_ok=True)
    base = "https://storage.googleapis.com/public-datasets-lila/nacti"
    meta_zip, md_zip = src / "nacti_metadata.json.zip", src / "nacti_md.json.zip"
    _download((f"{base}/nacti_metadata.1.14.json.zip", meta_zip))
    _download((("https://lila.science/public/lila-md-results/"
               "nacti_mdv5a.0.0_results.filtered_rde_0.150_0.850_50_0.200.json.zip"), md_zip))
    with zipfile.ZipFile(meta_zip) as z, z.open(z.namelist()[0]) as fh:
        meta = json.load(fh)
    cats = {c["id"]: c["name"].lower() for c in meta["categories"]}
    log.info("NACTI categories with pig/cow names: %s",
             [n for n in cats.values() if re.search("pig|boar|sus|cow|cattle|bos", n)])
    wanted = {i: ("boar" if re.search("pig|boar|sus scrofa", n) else "cattle")
              for i, n in cats.items() if re.search(r"wild pig|boar|sus scrofa|cattle|bos taurus|domestic cow", n)}
    images = {im["id"]: im for im in meta["images"]}
    by_file = {}
    per_image_cat: dict = {}
    for ann in meta["annotations"]:
        per_image_cat.setdefault(ann["image_id"], set()).add(ann["category_id"])
    for iid, cs in per_image_cat.items():
        if len(cs) == 1 and (c := next(iter(cs))) in wanted:
            by_file[images[iid]["file_name"]] = (wanted[c], images[iid])
    del meta, images, per_image_cat
    with zipfile.ZipFile(md_zip) as z, z.open(z.namelist()[0]) as fh:
        md = json.load(fh)
    animal = next(k for k, v in md["detection_categories"].items() if v == "animal")
    pools: dict[str, list[dict]] = {c: [] for c in ("boar", "cattle")}
    for im in md["images"]:
        hit = by_file.get(im.get("file"))
        if not hit:
            continue
        cls, info = hit
        dets = [d for d in im.get("detections") or []]
        boxes = [d["bbox"] for d in dets if d["category"] == animal and d["conf"] >= 0.5]
        # Reject images where MegaDetector is unsure or sees a person/vehicle: incomplete labels.
        if not boxes or any(d["conf"] >= 0.2 and not (d["category"] == animal and d["conf"] >= 0.5) for d in dets):
            continue
        pools[cls].append({
            "image": im["file"], "cls": cls, "group": str(info.get("location") or info.get("seq_id") or im["file"]),
            "boxes": [[x + w / 2, y + h / 2, w, h] for x, y, w, h in boxes],
            "label_source": "megadetector+human_species", "licence": "CDLA-Permissive", "author": "NACTI",
            "url": f"https://storage.googleapis.com/public-datasets-lila/nacti-unzipped/{urllib.parse.quote(im['file'])}"})
    rng = random.Random(seed)
    rows = []
    for cls, pool in pools.items():
        rng.shuffle(pool)
        # At most a few frames per camera location: one camera can dominate this dataset.
        seen: dict[str, int] = {}
        chosen = []
        for r in pool:
            if seen.get(r["group"], 0) < max(20, per_class // 20) and len(chosen) < per_class:
                seen[r["group"]] = seen.get(r["group"], 0) + 1
                chosen.append(r)
        log.info("lila %s: %d images from %d locations (pool %d)", cls, len(chosen), len(seen), len(pool))
        rows += chosen
    _write_manifest(src, _download_all(rows, src))


COMMONS_CLIPS = {
    "boar": ["Wild boars in Berlin-Spandau 01.webm", "Wild boars in Berlin-Spandau 04.webm",
             "Wild boars in a park in Berlin-Spandau 08.webm", "Sus scrofa-wild boars-Bulgaria-2 VP8.webm",
             "Sus scrofa-wild boars-Bulgaria-4 VP8.webm"],
    "nilgai": ["Zoo de La Fleche 20180601 6.webm"],
    "monkey": ["Macaca mulatta.ogv", "Macaca radiata juveniles grooming.webm",
               "Rhesus macaque - 20160710.webm", "Langurs Grooming in Community.webm"],
    "cattle": ["Cattle near Aberfeldy.webm", "Cows feeding on hay.webm", "Radevormwald - cattle grazing 01 ies.ogv",
               "Cattle grazing in a hot afternoon.webm"],
}


def fetch_commons_clips() -> None:
    """Openly licensed animal videos into data/clips/<class>__<title>, with data/clips/SOURCES.md.
    Parks, zoos and pastures, not farm CCTV: replace with real footage as soon as you have it."""
    out = ROOT / "data" / "clips"
    lines = ["# Clip sources (Wikimedia Commons)", "", "Not farm CCTV. Downloaded by `scripts/prepare_data.py commons-clips`.", "",
             "| File | Author | Licence | Description | Page |", "|---|---|---|---|---|"]
    clean = lambda s: re.sub(r"\s+", " ", re.sub("<[^>]+>", "", s or "")).strip().replace("|", "/")[:120]
    for cls, titles in COMMONS_CLIPS.items():
        for title in titles:
            time.sleep(6)  # Commons returns 429 to anything faster
            q = urllib.parse.urlencode({"action": "query", "format": "json", "titles": f"File:{title}",
                                        "prop": "imageinfo", "iiprop": "url|extmetadata"})
            page = next(iter(_json(f"https://commons.wikimedia.org/w/api.php?{q}", 0)["query"]["pages"].values()))
            if "imageinfo" not in page:
                log.warning("not on Commons: %s", title)
                continue
            info = page["imageinfo"][0]
            meta = info["extmetadata"]
            licence = meta.get("LicenseShortName", {}).get("value", "")
            if not re.match(r"(CC0|CC BY|Public domain)", licence):
                log.warning("skipping %s: licence %r", title, licence)
                continue
            safe = re.sub(r"[^\w.-]+", "_", title)
            path = out / f"{cls}__{safe}"
            for wait in (60, 120, 240, 0):  # the media host rate-limits too
                if _download((info["url"].split("?")[0], path)) or not wait:
                    break
                time.sleep(wait)
            if not path.exists():
                log.warning("download failed: %s", title)
                continue
            lines.append(f"| `{path.name}` | {clean(meta.get('Artist', {}).get('value'))} | {licence} | "
                         f"{clean(meta.get('ImageDescription', {}).get('value'))} | {info['descriptionurl']} |")
            log.info("clip %s (%s)", path.name, licence)
    (out / "SOURCES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def fetch_clips(fps: float, class_from_name: dict[str, str]) -> None:
    """Frames from data/clips/<class>__<anything>.<ext>. Unlabelled: pseudo-label adds the boxes."""
    import cv2
    src = RAW / "clips"
    rows: list[dict] = []
    for clip in sorted((ROOT / "data" / "clips").iterdir()):
        cls = class_from_name.get(clip.name.split("__")[0])
        if not cls:
            continue
        cap = cv2.VideoCapture(str(clip))
        native = cap.get(cv2.CAP_PROP_FPS) or 25
        step, i = max(1, round(native / fps)), 0
        (src / clip.stem).mkdir(parents=True, exist_ok=True)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % step == 0:
                rel = f"{clip.stem}/{i:06d}.jpg"
                cv2.imwrite(str(src / rel), frame)
                rows.append({"image": rel, "cls": cls, "group": clip.stem, "boxes": None,
                             "label_source": "yolo_world", "licence": "see data/clips/SOURCES.md",
                             "author": "", "url": ""})
            i += 1
        cap.release()
    _write_manifest(src, rows)
    log.info("clips: %d frames", len(rows))


def import_yolo(folder: Path, mapping: dict[str, str], name: str, licence: str) -> None:
    """Import a YOLO-format export (e.g. Roboflow, which needs your own account to download)."""
    import yaml
    data = yaml.safe_load((folder / "data.yaml").read_text(encoding="utf-8"))
    names = data["names"] if isinstance(data["names"], list) else [data["names"][k] for k in sorted(data["names"])]
    src = RAW / name
    rows = []
    for img in folder.rglob("*.*"):
        if img.suffix.lower() not in (".jpg", ".jpeg", ".png") or "images" not in img.parts:
            continue
        label = img.parent.parent / "labels" / f"{img.stem}.txt"
        by_cls: dict[str, list] = {}
        if label.exists():
            for line in label.read_text().splitlines():
                parts = line.split()
                if len(parts) == 5:  # polygons and keypoints are not plain boxes
                    mapped = mapping.get(names[int(parts[0])].lower())
                    by_cls.setdefault(mapped, []).append([float(v) for v in parts[1:]])
        if len(by_cls) != 1 or None in by_cls:  # classes outside the map would be unlabelled animals
            continue
        cls, boxes = next(iter(by_cls.items()))
        rel = f"{cls}/{img.name}"
        (src / cls).mkdir(parents=True, exist_ok=True)
        shutil.copy2(img, src / rel)
        rows.append({"image": rel, "cls": cls, "group": roboflow_group(img.stem), "boxes": boxes,
                     "label_source": "human", "licence": licence, "author": name, "url": ""})
    _write_manifest(src, rows)
    log.info("%s: imported %d images", name, len(rows))


# ---- pseudo-labelling (autodistill pattern: a big open-vocabulary model labels, a small one learns) -
PROMPTS = {"boar": ["wild boar", "pig"], "monkey": ["monkey", "macaque"],
           "nilgai": ["nilgai", "antelope", "deer"], "cattle": ["cow", "cattle"]}


def pseudo_label(weights: str, min_conf: float, device: str) -> None:
    """The species comes from the human-verified source label; YOLO-World only supplies the box.
    Images with no confident box are dropped rather than becoming false negatives in training."""
    import cv2
    from ultralytics import YOLO
    for src in sorted(p for p in RAW.iterdir() if (p / "manifest.jsonl").exists()):
        rows = _read_manifest(src)
        todo = [r for r in rows if r["boxes"] is None]
        if not todo:
            continue
        kept, dropped = [r for r in rows if r["boxes"] is not None], 0
        for cls in sorted({r["cls"] for r in todo}):
            model = YOLO(weights)
            model.set_classes(PROMPTS[cls])
            for r in (x for x in todo if x["cls"] == cls):
                img = cv2.imread(str(src / r["image"]))
                if img is None:
                    dropped += 1
                    continue
                b = model.predict(img, conf=min_conf, imgsz=640, device=device or None, verbose=False)[0].boxes
                h, w = img.shape[:2]
                # A box covering nearly the whole image is usually the model's guess, not a detection.
                boxes = [xyxy_to_xywhn(*xy, w, h) for xy in b.xyxy.tolist()]
                boxes = [x for x in boxes if x[2] * x[3] < 0.95]
                if not boxes:
                    dropped += 1
                    continue
                r["boxes"] = boxes
                kept.append(r)
        _write_manifest(src, kept)
        log.info("%s: pseudo-labelled %d, dropped %d", src.name, len(kept), dropped)


# ---- build ---------------------------------------------------------------------------------------
def build() -> None:
    import cv2
    if OUT.exists():
        shutil.rmtree(OUT)
    index = {c: i for i, c in enumerate(CLASSES)}
    records = []
    for src in sorted(p for p in RAW.iterdir() if (p / "manifest.jsonl").exists()):
        for r in _read_manifest(src):
            if r["boxes"] is None:
                continue  # never treat an unlabelled image as a background image
            img = cv2.imread(str(src / r["image"]))
            if img is None:
                continue
            split = split_for(src.name, r["group"])
            stem = hashlib.sha1(f"{src.name}/{r['image']}".encode()).hexdigest()[:16]
            ext = Path(r["image"]).suffix.lower()
            for sub in ("images", "labels"):
                (OUT / sub / split).mkdir(parents=True, exist_ok=True)
            shutil.copy2(src / r["image"], OUT / "images" / split / f"{stem}{ext}")
            lines = yolo_label_lines(r["boxes"], index[r["cls"]]) if r["cls"] else []
            (OUT / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + "\n" if lines else "")
            records.append({"file": f"images/{split}/{stem}{ext}", "split": split, "source": src.name,
                            "cls": r["cls"] or "negative", "group": r["group"],
                            "label_source": r["label_source"], "night": is_night(img),
                            "licence": r["licence"], "author": r.get("author", ""),
                            "url": r.get("page") or r.get("url", "")})
    with (OUT / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)
    (OUT / "data.yaml").write_text(
        f"path: {OUT.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        f"names: {json.dumps(dict(enumerate(CLASSES)))}\n", encoding="utf-8")
    groups = {}
    for r in records:
        groups.setdefault((r["source"], r["group"]), set()).add(r["split"])
    assert all(len(s) == 1 for s in groups.values()), "a clip landed in two splits"
    summary: dict = {}
    for r in records:
        k = f"{r['split']:5} {r['cls']:8} {'night' if r['night'] else 'day  '} {r['label_source']}"
        summary[k] = summary.get(k, 0) + 1
    for k in sorted(summary):
        print(summary[k], k)


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("openimages")
    o.add_argument("--per-class", type=int, default=2000)
    o.add_argument("--negatives", type=int, default=800)
    i = sub.add_parser("inat")
    i.add_argument("--per-class", type=int, default=1500)
    li = sub.add_parser("lila")
    li.add_argument("--per-class", type=int, default=1500)
    sub.add_parser("commons-clips")
    c = sub.add_parser("clips")
    c.add_argument("--fps", type=float, default=1.0)
    m = sub.add_parser("import-yolo")
    m.add_argument("folder", type=Path)
    m.add_argument("--map", required=True, help='"wild boar=boar,pig=boar"')
    m.add_argument("--name", default="roboflow")
    m.add_argument("--licence", required=True, help="read it on the dataset page first")
    p = sub.add_parser("pseudo-label")
    p.add_argument("--weights", default=str(ROOT / "weights" / "yolov8s-worldv2.pt"))
    p.add_argument("--min-conf", type=float, default=0.3)
    p.add_argument("--device", default="")
    sub.add_parser("build")
    a = ap.parse_args(argv)
    if a.cmd == "openimages":
        fetch_openimages(a.per_class, a.negatives)
    elif a.cmd == "inat":
        fetch_inat(a.per_class)
    elif a.cmd == "lila":
        fetch_lila(a.per_class)
    elif a.cmd == "commons-clips":
        fetch_commons_clips()
    elif a.cmd == "clips":
        fetch_clips(a.fps, {c: c for c in CLASSES})
    elif a.cmd == "import-yolo":
        mapping = dict(kv.split("=") for kv in a.map.lower().split(","))
        import_yolo(a.folder, mapping, a.name, a.licence)
    elif a.cmd == "pseudo-label":
        pseudo_label(a.weights, a.min_conf, a.device)
    elif a.cmd == "build":
        build()


if __name__ == "__main__":
    main(sys.argv[1:])
