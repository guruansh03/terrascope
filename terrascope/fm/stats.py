"""Uncertainty for the paper tables.

* seed level : mean, sample std (ddof=1) and a 95 % Student-t interval over seeds (None for a single seed)
* chip level : 95 % percentile bootstrap over TEST CHIPS (resample chips with replacement, B draws, fixed RNG),
               recomputing the POOLED metric from the summed per-chip confusion matrices (TP/FP/FN), i.e. exactly
               the statistic that is reported, not a mean of per-chip IoUs
* paired     : model A - model B on the same seeds and the same resampled chips (paired bootstrap)
"""
from __future__ import annotations

import numpy as np

B_DEFAULT = 1000


def t_ci95(values) -> list | None:
    v = [float(x) for x in values if x is not None and np.isfinite(x)]
    if len(v) < 2:
        return None
    from scipy import stats
    m, s = float(np.mean(v)), float(np.std(v, ddof=1))
    h = float(stats.t.ppf(0.975, len(v) - 1)) * s / np.sqrt(len(v))
    return [m - h, m + h]


def metric_from_conf(conf: np.ndarray, metric: str):
    conf = np.asarray(conf, np.float64)
    tot = conf.sum()
    if tot == 0:
        return None
    d = np.diag(conf)
    if metric == "accuracy":
        return float(d.sum() / tot)
    den = conf.sum(0) + conf.sum(1) - d
    if metric == "iou_pos":  # binary: TP / (TP + FP + FN)
        return float(d[1] / den[1]) if den[1] > 0 else None
    if metric == "miou":
        present = conf.sum(1) > 0
        return float(np.mean(d[present] / np.maximum(den[present], 1))) if present.any() else None
    raise KeyError(metric)


def metric_from_recon(chips, metric: str):
    a = np.asarray(chips, np.float64)  # rows: sse, n, ssim
    if not len(a):
        return None
    if metric == "rmse":
        return float(np.sqrt(a[:, 0].sum() / max(a[:, 1].sum(), 1)))
    if metric == "ssim":
        return float(a[:, 2].mean())
    raise KeyError(metric)


def _stat(chips, idx, metric):
    if metric in ("rmse", "ssim"):
        return metric_from_recon(np.asarray(chips)[idx], metric)
    return metric_from_conf(np.asarray(chips)[idx].sum(0), metric)


def draws(n_chips: int, B: int = B_DEFAULT, seed: int = 0) -> np.ndarray:
    """(B, n_chips) resampling indices, shared by all models so paired comparisons use the same draws."""
    return np.random.default_rng(seed).integers(0, n_chips, size=(B, n_chips))


def bootstrap_ci95(chips, metric: str, idx: np.ndarray) -> list | None:
    vals = [_stat(chips, i, metric) for i in idx]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def bootstrap_seed_mean_ci95(chips_per_seed: list, metric: str, idx: np.ndarray) -> list | None:
    """CI of the seed-averaged pooled metric (each draw: same chips for every seed)."""
    vals = []
    for i in idx:
        v = [_stat(c, i, metric) for c in chips_per_seed]
        if all(x is not None for x in v):
            vals.append(float(np.mean(v)))
    if not vals:
        return None
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def paired(chips_a: dict, chips_b: dict, metric: str, idx: np.ndarray) -> dict | None:
    """chips_x: {seed: chip_stats}. Difference A - B on the seeds both have."""
    seeds = sorted(set(chips_a) & set(chips_b))
    if not seeds:
        return None
    per_seed = []
    for s in seeds:
        a, b = _stat(chips_a[s], slice(None), metric), _stat(chips_b[s], slice(None), metric)
        per_seed.append(None if a is None or b is None else a - b)
    diffs = []
    for i in idx:
        d = []
        for s in seeds:
            a, b = _stat(chips_a[s], i, metric), _stat(chips_b[s], i, metric)
            if a is not None and b is not None:
                d.append(a - b)
        if len(d) == len(seeds):
            diffs.append(float(np.mean(d)))
    vals = [x for x in per_seed if x is not None]
    out = dict(seeds=seeds, diff_per_seed=per_seed, mean_diff=float(np.mean(vals)) if vals else None,
               std_diff=float(np.std(vals, ddof=1)) if len(vals) > 1 else None, ci95_seeds=t_ci95(vals),
               bootstrap_ci95=None, p_bootstrap_two_sided=None, n_draws=len(diffs))
    if diffs:
        d = np.asarray(diffs)
        out["bootstrap_ci95"] = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
        out["p_bootstrap_two_sided"] = float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))
    return out
