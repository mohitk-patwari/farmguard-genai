"""Label-file helpers: dev/test split by clip, inter-labeller agreement, merging two labellers.

data/labels/splits.csv holds `clip,split,lighting`. Only clip-level assignment exists, so frames of
one clip can never land in two splits. Experiments filter on it; label formats stay as documented.
"""
from __future__ import annotations

import csv
import random
from collections.abc import Iterable
from pathlib import Path

from farmguard.eval.metrics import cohen_kappa

LABEL_COLUMNS = {"vlm": ["species", "behaviour"], "retreat": ["status"]}


def read_rows(path: str | Path) -> list[dict[str, str]]:
    # utf-8-sig: spreadsheets add a BOM when labels are edited by hand
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def write_rows(path: str | Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def clip_of(row: dict[str, str]) -> str:
    """Clip stem of a label row: from `clip` (intrusions) or the first frame's folder (samples)."""
    if row.get("clip"):
        return Path(row["clip"]).stem
    return Path(row["frames"].split(";")[0]).parent.name


def assign_splits(clips: Iterable[str], existing: dict[str, dict[str, str]] | None = None,
                  test_fraction: float = 0.3, seed: int = 13) -> dict[str, dict[str, str]]:
    """New clips are split at random; a clip already assigned NEVER moves, so adding footage later
    cannot leak test clips into dev.
    ponytail: not stratified by species, hand-edit splits.csv if a small test set misses one."""
    out = {k: dict(v) for k, v in (existing or {}).items()}
    new = sorted(c for c in set(clips) if c not in out)
    random.Random(seed).shuffle(new)
    n_test = round(len(new) * test_fraction)
    for i, clip in enumerate(new):
        out[clip] = {"clip": clip, "split": "test" if i < n_test else "dev", "lighting": ""}
    return out


def load_splits(path: str | Path) -> dict[str, dict[str, str]]:
    path = Path(path)
    return {r["clip"]: r for r in read_rows(path)} if path.exists() else {}


def select(rows: list[dict[str, str]], splits_csv: str | Path, split: str = "all",
           lighting: str | None = None) -> list[dict[str, str]]:
    """Filter label rows by split and/or lighting. A clip missing from splits.csv is an error,
    never a silent pass, because it could otherwise leak into the wrong split."""
    if split == "all" and not lighting:
        return rows
    splits = load_splits(splits_csv)
    out = []
    for r in rows:
        info = splits.get(clip_of(r))
        if info is None:
            raise ValueError(f"clip {clip_of(r)!r} is not in {splits_csv}; run scripts/label.py split")
        if (split == "all" or info["split"] == split) and (not lighting or info["lighting"] == lighting):
            out.append(r)
    return out


def agreement(a_csv: str | Path, b_csv: str | Path, columns: list[str]) -> dict[str, dict[str, float]]:
    """Cohen's kappa per label column over samples both people labelled."""
    a, b = {r["sample_id"]: r for r in read_rows(a_csv)}, {r["sample_id"]: r for r in read_rows(b_csv)}
    shared = sorted(a.keys() & b.keys())
    return {col: {"n": len(shared), "kappa": cohen_kappa([a[s][col] for s in shared],
                                                         [b[s][col] for s in shared]),
                  "agree": sum(a[s][col] == b[s][col] for s in shared) / len(shared) if shared else 0.0}
            for col in columns}


def merge(a_csv: str | Path, b_csv: str | Path, columns: list[str], out_csv: str | Path,
          conflicts_csv: str | Path) -> tuple[int, int]:
    """Agreed samples go to out_csv; the rest go to conflicts_csv to resolve by discussion and then
    copy across by hand. Returns (agreed, conflicting)."""
    a, b = {r["sample_id"]: r for r in read_rows(a_csv)}, {r["sample_id"]: r for r in read_rows(b_csv)}
    fields = list(next(iter(a.values())))
    agreed, conflicts = [], []
    for sid in sorted(a.keys() & b.keys()):
        if all(a[sid][c] == b[sid][c] for c in columns):
            agreed.append(a[sid])
        else:
            conflicts.append({**a[sid], **{f"{c}_b": b[sid][c] for c in columns}})
    write_rows(out_csv, agreed, fields)
    write_rows(conflicts_csv, conflicts, fields + [f"{c}_b" for c in columns])
    return len(agreed), len(conflicts)
