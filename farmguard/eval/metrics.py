"""Shared metric helpers for the experiments."""
from __future__ import annotations

import itertools
import math
import random
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


def paired_permutation(diffs: Sequence[float], max_exact_n: int = 16, n_mc: int = 20000,
                       seed: int = 0) -> float:
    """Two-sided sign-flip test on paired differences. Exact up to max_exact_n pairs. The smallest
    possible p is 2 / 2^n, so 5 pairs can never reach p < 0.0625: report that, don't hide it."""
    d = [x for x in diffs if x != 0]
    if not d:
        return 1.0
    obs = abs(sum(d)) - 1e-12
    if len(d) <= max_exact_n:
        hits = sum(abs(sum(x if s else -x for x, s in zip(d, signs))) >= obs
                   for signs in itertools.product((False, True), repeat=len(d)))
        return hits / 2 ** len(d)
    rng = random.Random(seed)
    hits = sum(abs(sum(x if rng.random() < 0.5 else -x for x in d)) >= obs for _ in range(n_mc))
    return (hits + 1) / (n_mc + 1)


def _ranks(vals: Sequence[float]) -> list[float]:
    """Average ranks (1-based), ties share their mean rank."""
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    out = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return out


def wilcoxon_signed_rank(x: Sequence[float], y: Sequence[float]) -> dict[str, float]:
    """Wilcoxon signed-rank by permuting signed ranks: exact for small n, correct under ties
    (zero differences dropped). Used for E6, where n = raters is small."""
    d = [a - b for a, b in zip(x, y) if a != b]
    if not d:
        return {"n": 0, "p": 1.0, "w_plus": 0.0}
    ranks = _ranks([abs(v) for v in d])
    signed = [r if v > 0 else -r for r, v in zip(ranks, d)]
    return {"n": len(d), "p": paired_permutation(signed), "w_plus": sum(r for r in signed if r > 0)}


def mann_whitney(a: Sequence[float], b: Sequence[float]) -> dict[str, float]:
    """Two-sided Mann-Whitney U, normal approximation with tie and continuity corrections.
    rank_biserial is the effect size: P(a>b) - P(a<b)."""
    n1, n2 = len(a), len(b)
    if not n1 or not n2:
        return {"u": float("nan"), "p": float("nan"), "rank_biserial": float("nan")}
    all_vals = list(a) + list(b)
    ranks = _ranks(all_vals)
    u = sum(ranks[:n1]) - n1 * (n1 + 1) / 2
    n = n1 + n2
    ties = sum(c ** 3 - c for c in Counter(all_vals).values())
    var = n1 * n2 / 12 * ((n + 1) - ties / (n * (n - 1))) if n > 1 else 0.0
    if var <= 0:
        return {"u": u, "p": 1.0, "rank_biserial": 2 * u / (n1 * n2) - 1}
    z = max(0.0, abs(u - n1 * n2 / 2) - 0.5) / math.sqrt(var)
    return {"u": u, "p": math.erfc(z / math.sqrt(2)), "rank_biserial": 2 * u / (n1 * n2) - 1}


def median_diff_ci(a: Sequence[float], b: Sequence[float], n_boot: int = 2000,
                   seed: int = 0) -> dict[str, float]:
    """median(a) - median(b) with a 95% bootstrap CI. E1's claim is 'no meaningful delay', and a
    non-significant test alone cannot show that; the width of this interval can."""
    if not a or not b:
        return {"diff": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    rng = random.Random(seed)
    a, b = list(a), list(b)
    boots = sorted(statistics.median(rng.choices(a, k=len(a))) -
                   statistics.median(rng.choices(b, k=len(b))) for _ in range(n_boot))
    return {"diff": statistics.median(a) - statistics.median(b),
            "ci_low": percentile(boots, 0.025), "ci_high": percentile(boots, 0.975)}


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values for one family of tests."""
    adjusted, running = {}, 0.0
    ordered = sorted(pvals.items(), key=lambda kv: kv[1])
    for i, (k, p) in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - i) * p))
        adjusted[k] = running
    return adjusted


def cohen_kappa(a: Sequence[str], b: Sequence[str]) -> float:
    """Inter-labeller agreement (docs/PAPER_PLAN.md, Data)."""
    n = len(a)
    if not n or n != len(b):
        return float("nan")
    observed = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[k] * cb[k] for k in ca) / n ** 2
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)
