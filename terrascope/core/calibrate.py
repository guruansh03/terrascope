"""Temperature scaling for the land-cover softmax -> artifacts/calibration.json

T is fitted (NLL, LBFGS on log T) on a dedicated validation seed range (val64: 800000-800199) that is
used neither for training nor for the reported test metrics. ECE / reliability bins are reported on the
held-out test split (test64) before and after scaling, and on the validation split for reference.
"""
from __future__ import annotations

import json, os, time
import numpy as np
import torch
import torch.nn.functional as F

from .data import load, SPLITS
from .model import load_checkpoint

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "artifacts", "calibration.json")
N_BINS = 15


@torch.no_grad()
def logits_of(model, d, bs=50):
    out = []
    for i in range(0, d["x"].shape[0], bs):
        x = d["x"][i:i + bs]
        av = torch.ones(x.shape[0], 3, dtype=torch.bool)
        out.append(model.forward_tasks(x, av)["lc"])
    z = torch.cat(out)                                  # (N,C,H,W)
    return z.permute(0, 2, 3, 1).reshape(-1, z.shape[1]), d["lc"].reshape(-1)


def fit_temperature(z, y, iters=60):
    logt = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([logt], lr=0.1, max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(z / logt.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(logt.exp().item())


def reliability(z, y, T=1.0, n_bins=N_BINS):
    p = torch.softmax(z / T, 1)
    conf, pred = p.max(1)
    acc = (pred == y).float()
    edges = np.linspace(0, 1, n_bins + 1)
    bins, ece, n = [], 0.0, len(y)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        c = int(m.sum())
        if c:
            a, cf = float(acc[m].mean()), float(conf[m].mean())
            ece += c / n * abs(a - cf)
        else:
            a = cf = None
        bins.append(dict(lo=round(float(lo), 4), hi=round(float(hi), 4), count=c, acc=a, conf=cf))
    nll = float(F.cross_entropy(z / T, y))
    return dict(ece=float(ece), nll=nll, mean_conf=float(conf.mean()), accuracy=float(acc.mean()), bins=bins)


def main(ckpt="finetuned.pt"):
    torch.set_num_threads(int(os.environ.get("TERRASCOPE_THREADS", "0")) or os.cpu_count() or 2)
    t0 = time.time()
    model, cfg = load_checkpoint(os.path.join(ROOT, "artifacts", "ckpt", ckpt))
    norm = cfg.get("norm", "v1")
    zv, yv = logits_of(model, load("val64", norm))
    zt, yt = logits_of(model, load("test64", norm))
    T = fit_temperature(zv, yv)
    before, after = reliability(zt, yt, 1.0), reliability(zt, yt, T)
    vb, va = reliability(zv, yv, 1.0), reliability(zv, yv, T)
    res = dict(
        method="temperature", task="landcover", T=T, ckpt=ckpt, norm=norm,
        fit_split=dict(name="val64", seeds=[SPLITS["val64"][0].start, SPLITS["val64"][0].stop - 1], pixels=int(len(yv))),
        eval_split=dict(name="test64", seeds=[SPLITS["test64"][0].start, SPLITS["test64"][0].stop - 1], pixels=int(len(yt))),
        ece_before=before["ece"], ece_after=after["ece"], nll_before=before["nll"], nll_after=after["nll"],
        accuracy=before["accuracy"], mean_conf_before=before["mean_conf"], mean_conf_after=after["mean_conf"],
        bins=[dict(lo=b["lo"], hi=b["hi"], count_before=b["count"], acc_before=b["acc"], conf_before=b["conf"],
                   count_after=a["count"], acc_after=a["acc"], conf_after=a["conf"]) for b, a in zip(before["bins"], after["bins"])],
        val=dict(ece_before=vb["ece"], ece_after=va["ece"], nll_before=vb["nll"], nll_after=va["nll"]),
        # The decision to APPLY T is made on the validation split only; test ECE is reported as an outcome.
        apply=bool(va["ece"] < vb["ece"]),
        decision=dict(split="val64", rule="apply temperature only if validation ECE decreases",
                      ece_val_before=vb["ece"], ece_val_after=va["ece"]),
        n_bins=N_BINS, all_sensors=True, generated_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        note="Simulated data only. T fitted and the apply/not-apply decision made on val64; ece_before/ece_after are "
             "held-out test64 outcomes. Calibrates the land-cover softmax only.",
    )
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)
    print(f"[calibrate] T={T:.3f}  apply={res['apply']} (val ECE {vb['ece']:.4f} -> {va['ece']:.4f})  ECE test {before['ece']:.4f} -> {after['ece']:.4f}  NLL {before['nll']:.4f} -> {after['nll']:.4f}  ({time.time() - t0:.0f}s)")
    return res


if __name__ == "__main__":
    main()
