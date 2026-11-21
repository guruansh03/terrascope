"""Dataset-level metrics accumulated over chips (pixels pooled across the split, ignore = -1)."""
from __future__ import annotations

import numpy as np


class SegMeter:
    def __init__(self, n_classes: int):
        self.n = n_classes
        self.conf = np.zeros((n_classes, n_classes), np.int64)

    def update(self, pred: np.ndarray, target: np.ndarray):
        t = np.asarray(target).ravel()
        p = np.asarray(pred).ravel()
        v = (t >= 0) & (t < self.n)
        self.conf += np.bincount(t[v].astype(np.int64) * self.n + p[v].astype(np.int64),
                                 minlength=self.n * self.n).reshape(self.n, self.n)

    def iou(self) -> np.ndarray:
        d = np.diag(self.conf).astype(np.float64)
        den = self.conf.sum(0) + self.conf.sum(1) - d
        return np.where(den > 0, d / np.maximum(den, 1), np.nan)

    def result(self) -> dict:
        tot = self.conf.sum()
        iou = self.iou()
        present = self.conf.sum(1) > 0
        out = dict(accuracy=float(np.diag(self.conf).sum() / tot) if tot else None,
                   miou=float(np.nanmean(iou[present])) if present.any() else None,
                   per_class_iou=[None if np.isnan(x) else float(x) for x in iou], pixels=int(tot))
        if self.n == 2:
            out["iou_pos"] = None if np.isnan(iou[1]) else float(iou[1])
        return out


def _gauss(size=11, sigma=1.5):
    x = np.arange(size) - size // 2
    g = np.exp(-x ** 2 / (2 * sigma ** 2))
    return g / g.sum()


def ssim(a: np.ndarray, b: np.ndarray, data_range: float = 1.0) -> float:
    """Mean SSIM over channels (Wang et al. 2004; Gaussian 11x11, sigma 1.5). a,b: (C,H,W)."""
    from scipy.ndimage import convolve1d
    g = _gauss()
    C1, C2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2

    def f(x):
        return convolve1d(convolve1d(x, g, axis=-1, mode="reflect"), g, axis=-2, mode="reflect")

    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    mu_a, mu_b = f(a), f(b)
    saa = f(a * a) - mu_a ** 2
    sbb = f(b * b) - mu_b ** 2
    sab = f(a * b) - mu_a * mu_b
    m = ((2 * mu_a * mu_b + C1) * (2 * sab + C2)) / ((mu_a ** 2 + mu_b ** 2 + C1) * (saa + sbb + C2))
    return float(m.mean())


class ReconMeter:
    def __init__(self):
        self.se, self.n, self.ssims = 0.0, 0, []

    def update(self, pred: np.ndarray, target: np.ndarray):
        p = np.clip(np.asarray(pred, np.float64), 0, 1)
        t = np.clip(np.asarray(target, np.float64), 0, 1)
        self.se += float(((p - t) ** 2).sum())
        self.n += p.size
        self.ssims.append(ssim(p, t))

    def result(self) -> dict:
        return dict(rmse=float(np.sqrt(self.se / max(self.n, 1))) if self.n else None,
                    ssim=float(np.mean(self.ssims)) if self.ssims else None, chips=len(self.ssims))


def mean_std(xs) -> dict:
    v = [float(x) for x in xs if x is not None and np.isfinite(x)]
    if not v:
        return dict(mean=None, std=None, n=0)
    # a single value has no spread: std is reported as null rather than a misleading 0.0
    return dict(mean=float(np.mean(v)), std=float(np.std(v, ddof=1)) if len(v) > 1 else None, n=len(v))
