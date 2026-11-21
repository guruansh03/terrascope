from __future__ import annotations
import numpy as np
import torch
from .model import N_CLASSES

SENSOR_SETS = {
    "All sensors": [True, True, True],
    "Optical only": [True, False, False],
    "SAR only": [False, True, False],
    "DEM only": [False, False, True],
    "SAR + DEM": [False, True, True],
    "Optical + SAR": [True, True, False],
}


def threshold_cloud_mask(optical):
    """Legacy (phase 1/2) fixed reflectance rule; kept only as a baseline for the cloud head."""
    o = optical
    return (o[:, 0] > 0.38) & (o[:, 2] > 0.38) & (o[:, 3] > 0.38) & (o[:, 3] < 0.95)


@torch.no_grad()
def evaluate(model, d, avail, limit=None, bs=50, recon=False, shift_ens=False):
    """Scores on a split prepared by ``data.with_inputs``. ``recon`` adds cloud-gap RMSE (reflectance units)
    of the raw MAE reconstruction on pixels with true cloud opacity > 0.3; ``shift_ens`` uses the 4-shift
    ensemble reconstruction. Models with a cloud head also get cloud IoU (pred prob > 0.3 vs alpha > 0.3)."""
    model.eval()
    dev = next(model.parameters()).device
    N = d["x"].shape[0] if limit is None else min(limit, d["x"].shape[0])
    conf = torch.zeros(N_CLASSES, N_CLASSES)
    fi = fu = 0.0
    veg_err = 0.0
    veg_n = 0
    rec_err = base_err = 0.0
    rec_n = 0
    ci = cu = ti = tu = 0.0
    use_cloud = getattr(model, "has_cloud_head", False) and avail[0] and "cloud" in d
    for i in range(0, N, bs):
        sl = slice(i, min(N, i + bs))
        x = d["x"][sl].to(dev)
        av = torch.tensor([avail] * x.shape[0], device=dev)
        out = model.forward_all(x, av) if recon else model.forward_tasks(x, av)
        out = {k: v.cpu() for k, v in out.items()}
        x = x.cpu()
        pred = out["lc"].argmax(1)
        y = d["lc"][sl]
        conf += torch.bincount((y * N_CLASSES + pred).flatten(), minlength=N_CLASSES ** 2).view(N_CLASSES, N_CLASSES).float()
        pf = (out["flood"] > 0).float()
        yf = d["flood"][sl]
        fi += (pf * yf).sum().item(); fu += ((pf + yf) > 0).float().sum().item()
        veg_err += (out["veg"] - d["veg"][sl]).abs().sum().item(); veg_n += y.numel()
        if use_cloud:
            yc = d["cloud"][sl] > 0.3
            pc = torch.sigmoid(out["cloud"]) > 0.3
            ci += (pc & yc).sum().item(); cu += (pc | yc).sum().item()
            tc = threshold_cloud_mask(d["optical"][sl].float())
            ti += (tc & yc).sum().item(); tu += (tc | yc).sum().item()
        if recon:
            cm = d["cloud"][sl] > 0.3
            if cm.any():
                rc = model.reconstruct(d["x"][sl].to(dev), av).cpu() if shift_ens else out["recon"]
                tgt = d["target"][sl][:, :6]
                e = ((rc[:, :6] - tgt) ** 2).mean(1)[cm].sum().item()
                b = ((x[:, :6] - tgt) ** 2).mean(1)[cm].sum().item()
                rec_err += e; base_err += b; rec_n += cm.sum().item()
    iou = conf.diag() / (conf.sum(0) + conf.sum(1) - conf.diag()).clamp(min=1)
    res = dict(oa=(conf.diag().sum() / conf.sum()).item(), miou=iou.mean().item(), iou=[round(v, 4) for v in iou.tolist()],
               flood_iou=fi / max(fu, 1), veg_mae=veg_err / veg_n)
    if use_cloud:
        res["cloud_iou"] = ci / max(cu, 1)
        res["cloud_iou_threshold_rule"] = ti / max(tu, 1)
    if recon and rec_n:
        # RMSE in reflectance units (x0.2 scaling of the normalised optical)
        res["cloud_rmse"] = float(np.sqrt(rec_err / rec_n)) * 0.2
        res["cloud_rmse_input"] = float(np.sqrt(base_err / rec_n)) * 0.2
    return res
