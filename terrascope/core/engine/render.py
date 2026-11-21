"""Rendering of every map layer (PNG bytes) from a scene + runner outputs, and the cloud-free composite."""
from __future__ import annotations

import io
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
from matplotlib import colormaps
from scipy.ndimage import binary_dilation, gaussian_filter

from ..simulator import CLASS_COLORS

UP = 4
PAL = np.array(CLASS_COLORS, np.uint8)
CLOUD_THR = 0.3


def png(arr, resample=Image.NEAREST, up=UP):
    im = Image.fromarray(arr)
    if up != 1:
        im = im.resize((im.width * up, im.height * up), resample)
    b = io.BytesIO()
    im.save(b, "PNG", optimize=False, compress_level=3)
    return b.getvalue()


def cmap(name, v, lo=0.0, hi=1.0):
    v = np.clip((v - lo) / max(hi - lo, 1e-9), 0, 1)
    return (colormaps[name](v)[..., :3] * 255).astype(np.uint8)


def stretch_rgb(r, g, b, lo=0.0, hi=0.28, gamma=1.0 / 1.15):
    out = np.stack([r, g, b], -1)
    return (np.clip((out - lo) / (hi - lo), 0, 1) ** gamma * 255).astype(np.uint8)


def hillshade(elev_m, gsd_m=30.0):
    gy, gx = np.gradient(elev_m, gsd_m)
    alt, az = np.radians(40), np.radians(315)
    nrm = np.sqrt(gx ** 2 + gy ** 2 + 1)
    hs = (-gx * np.cos(alt) * np.cos(az) - gy * np.cos(alt) * np.sin(az) + np.sin(alt)) / nrm
    return np.clip(hs, 0, 1)


# ------------------------------------------------------------------ cloud-free composite
def blend_weights(cloud_prob):
    """Per-pixel weight of the reconstruction in the composite, from the cloud head's probability:
    soft probability, raised to ~1 on a 2-px dilated, slightly blurred halo around cloud (> 0.3)."""
    cp = np.clip(np.asarray(cloud_prob, np.float32), 0, 1)
    halo = gaussian_filter(binary_dilation(cp > CLOUD_THR, iterations=2).astype(np.float32), 1.0)
    return np.maximum(halo, cp)[None]


def threshold_cloud(o):
    return (o[0] > 0.38) & (o[2] > 0.38) & (o[3] > 0.38) & (o[3] < 0.95)


def threshold_blend_weights(o):
    """Phase-2 rule (fixed reflectance threshold). Used only when the runner has no cloud head (v1 ckpt)."""
    return gaussian_filter(binary_dilation(threshold_cloud(o), iterations=3).astype(np.float32), 1.2)[None]


def composite(o, recon, cloud_prob, optical_on=True):
    """-> (composite (6,H,W), weight (H,W), method). Without optical input everything is model-inferred."""
    if not optical_on:
        a = np.ones((1,) + o.shape[1:], np.float32)
        method = "no_optical_input"
    elif cloud_prob is not None:
        a, method = blend_weights(cloud_prob), "cloud_head"
    else:
        a, method = threshold_blend_weights(o), "threshold_rule"
    return np.clip(o * (1 - a) + recon * a, 0, 1), a[0], method


def flood_comp(rgb, mask, prob, hot=(20, 220, 255), glow=(120, 80, 255)):
    base = rgb.astype(np.float32) * 0.55
    m = np.asarray(mask, np.float32)[..., None]
    out = base * (1 - 0.72 * m) + np.array(hot, np.float32) * 0.72 * m
    g = np.clip(prob, 0, 1)[..., None] * (1 - m) * 0.35
    out = out * (1 - g) + np.array(glow, np.float32) * g
    return out.clip(0, 255).astype(np.uint8)


def entropy(prob):
    n = prob.shape[0]
    return -(prob * np.log(np.clip(prob, 1e-9, None))).sum(0) / np.log(n)


def input_layers(sc, gsd_m=30.0):
    L = {}
    o = sc["optical"]
    L["rgb"] = png(stretch_rgb(o[2], o[1], o[0]), Image.BICUBIC)
    L["fcc"] = png(stretch_rgb(o[3], o[2], o[1], hi=0.45), Image.BICUBIC)
    ndvi = (o[3] - o[2]) / (o[3] + o[2] + 1e-6)
    L["ndvi"] = png(cmap("RdYlGn", ndvi, -0.2, 0.9), Image.BICUBIC)
    vv, vh = sc["sar"]
    sar_rgb = np.stack([np.clip((vv + 25) / 25, 0, 1), np.clip((vh + 32) / 25, 0, 1), np.clip((vv - vh - 2) / 14, 0, 1)], -1)
    L["sar"] = png((sar_rgb ** 0.9 * 255).astype(np.uint8), Image.BICUBIC)
    elev = sc["dem"][0]
    hs = hillshade(elev, gsd_m)
    tcol = cmap("terrain", elev, elev.min(), max(elev.max(), elev.min() + 1)).astype(np.float32) / 255
    L["dem"] = png(((0.35 + 0.75 * hs[..., None]) * tcol).clip(0, 1).__mul__(255).astype(np.uint8), Image.BICUBIC)
    return L


def output_layers(sc, pred, sensors):
    """Layers for whatever the runner returned. Returns (layers, derived) where derived holds arrays the
    engine needs for stats/pix (lc, ent, flood mask, cloud_mask, composite)."""
    L, D = {}, {}
    o = sc["optical"]
    H, W = o.shape[1:]
    if "lc_prob" in pred:
        prob = pred["lc_prob"]
        D["lc"] = prob.argmax(0).astype(np.uint8)
        D["ent"] = entropy(prob).astype(np.float32)
        L["lc"] = png(PAL[D["lc"]])
        L["unc"] = png(cmap("magma", D["ent"], 0, 0.8), Image.BICUBIC)
    if "veg" in pred:
        L["veg"] = png(cmap("RdYlGn", pred["veg"], 0.0, 0.85), Image.BICUBIC)
    if "cloud_prob" in pred:
        L["cloud"] = png(cmap("bone", pred["cloud_prob"], 0, 1), Image.BICUBIC)
    base = o
    if "recon" in pred:
        rc, a, method = composite(o, pred["recon"], pred.get("cloud_prob"), bool(sensors[0]))
        D["cloud_weight"], D["composite_method"] = a, method
        D["cloud_mask"] = ((a > 0.5) * 255).astype(np.uint8)
        L["recon"] = png(stretch_rgb(rc[2], rc[1], rc[0]), Image.BICUBIC)
        base = rc
    base_rgb = stretch_rgb(base[2], base[1], base[0])
    if "flood_prob" in pred:
        fp = pred["flood_prob"]
        L["flood"] = png(flood_comp(base_rgb, fp > 0.5, fp), Image.BICUBIC)
    if "burn_prob" in pred:
        bp = pred["burn_prob"]
        L["burn"] = png(flood_comp(base_rgb, bp > 0.5, bp, hot=(255, 110, 20), glow=(255, 200, 60)), Image.BICUBIC)
    if "feats" in pred and "feat_hw" in pred:
        h, w = pred["feat_hw"]
        L["pca"] = pca_layer(pred["feats"], h, w, H, W)
    return L, D


def pca_layer(feats, h, w, H, W):
    import torch
    f = torch.from_numpy(np.asarray(feats, np.float32))
    f = f - f.mean(0, keepdim=True)
    U, S, V = torch.pca_lowrank(f, q=3, center=False)
    comp = (f @ V[:, :3]).numpy()
    lo, hi = np.percentile(comp, 2, 0), np.percentile(comp, 98, 0)
    img = (np.clip((comp - lo) / (hi - lo + 1e-9), 0, 1) * 255).astype(np.uint8).reshape(h, w, 3)
    im = Image.fromarray(img).resize((W * UP, H * UP), Image.BICUBIC)
    b = io.BytesIO()
    im.save(b, "PNG", compress_level=3)
    return b.getvalue()


def truth_layers(sc):
    L = {}
    o = sc["optical"]
    if "lc" in sc:
        L["lc_true"] = png(PAL[np.asarray(sc["lc"]).astype(int)])
    if "veg" in sc:
        L["veg_true"] = png(cmap("RdYlGn", sc["veg"], 0.0, 0.85), Image.BICUBIC)
    tc = sc.get("optical_clean", o)
    tr = stretch_rgb(tc[2], tc[1], tc[0])
    if "flood" in sc:
        L["flood_true"] = png(flood_comp(tr, sc["flood"] > 0, np.asarray(sc["flood"], np.float32)), Image.BICUBIC)
    if "optical_clean" in sc:
        L["rgb_clean"] = png(tr, Image.BICUBIC)
    return L


def similarity_png(feats, hw, size, base_png, x, y):
    import torch
    import torch.nn.functional as F
    h, w = hw
    H, W = size
    f = torch.from_numpy(np.asarray(feats, np.float32))
    f = F.normalize(f - f.mean(0, keepdim=True), dim=-1)
    sy, sx = H / h, W / w
    ty, tx = min(h - 1, int(max(0, int(y)) // sy)), min(w - 1, int(max(0, int(x)) // sx))
    sim = (f @ f[ty * w + tx]).reshape(h, w).numpy()
    sim = np.clip(sim, 0, 1) ** 1.3
    sim_up = np.array(Image.fromarray(sim.astype(np.float32)).resize((W * UP, H * UP), Image.BICUBIC))
    base = np.array(Image.open(io.BytesIO(base_png)).convert("RGB")).astype(np.float32)
    heat = (colormaps["inferno"](np.clip(sim_up, 0, 1))[..., :3] * 255).astype(np.float32)
    a = np.clip(sim_up * 1.3, 0, 0.88)[..., None]
    out = (base * 0.45 * (1 - a) + heat * a + base * 0.2 * a).clip(0, 255).astype(np.uint8)
    b = io.BytesIO(); Image.fromarray(out).save(b, "PNG", compress_level=3)
    return b.getvalue()


def change_layers(base_rgb, masks, lc_b=None):
    """Overlay change masks on the later scene's RGB. masks: name -> bool (H,W)."""
    colors = dict(flood_onset=(20, 220, 255), veg_loss=(255, 140, 30), urban_growth=(240, 60, 80))
    L = {}
    dim = base_rgb.astype(np.float32) * 0.45
    for name, m in masks.items():
        m3 = np.asarray(m, np.float32)[..., None]
        if name == "lc_change" and lc_b is not None:
            col = PAL[np.asarray(lc_b).astype(int)].astype(np.float32)
        else:
            col = np.broadcast_to(np.array(colors.get(name, (255, 255, 255)), np.float32), dim.shape)
        out = dim * (1 - 0.85 * m3) + col * 0.85 * m3
        L[name] = png(out.clip(0, 255).astype(np.uint8), Image.NEAREST)
    return L
