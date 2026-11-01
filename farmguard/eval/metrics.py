"""Shared metric helpers for the experiments."""
from __future__ import annotations

import math
import statistics
from collections import Counter
from collections.abc import Iterable, Sequence


def percentile(vals: Sequence[float], q: float) -> float:
    s = sorted(v for v in vals if v is not None and not math.isnan(v))
    if not s:
        return float("nan")
    k = (len(s) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] if lo == hi else s[lo] + (s[hi] - s[lo]) * (k - lo)


def latency_summary(vals: Sequence[float]) -> dict[str, float]:
    v = [x for x in vals if x is not None]
    if not v:
        return {"n": 0}
    return {"n": len(v), "mean": statistics.fmean(v), "median": percentile(v, 0.5),
            "p95": percentile(v, 0.95), "max": max(v)}


def classification_report(y_true: Sequence[str], y_pred: Sequence[str],
                          abstain_labels: Iterable[str] = ("unclassifiable", "unclear", "uncertain")
                          ) -> dict[str, float]:
    """Raw accuracy, abstention rate and accuracy-when-committed (AlexKapy reports both)."""
    abstain = set(abstain_labels)
    n = len(y_true)
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    abst = sum(p in abstain for p in y_pred)
    committed = [(t, p) for t, p in zip(y_true, y_pred) if p not in abstain]
    out = {"n": n, "accuracy": correct / n if n else float("nan"),
           "abstention_rate": abst / n if n else float("nan"),
           "accuracy_when_committed": (sum(t == p for t, p in committed) / len(committed))
           if committed else float("nan")}
    labels = sorted(set(y_true))
    f1s = []
    for lab in labels:
        tp = sum(t == lab and p == lab for t, p in zip(y_true, y_pred))
        fp = sum(t != lab and p == lab for t, p in zip(y_true, y_pred))
        fn = sum(t == lab and p != lab for t, p in zip(y_true, y_pred))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    out["macro_f1"] = sum(f1s) / len(f1s) if f1s else float("nan")
    out["prediction_counts"] = dict(Counter(y_pred))
    return out


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (centre - half, centre + half)


def mcnemar(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value for paired binary outcomes (b, c = discordant counts)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)
