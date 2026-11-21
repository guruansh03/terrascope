"""Scene statistics, heuristic risk indices (with their formulas) and template insights."""
from __future__ import annotations

import numpy as np

from ..simulator import CLASS_NAMES

N_CLASSES = len(CLASS_NAMES)
VEG_CLASSES = (1, 2, 4)  # forest, cropland, grassland

RISK_FORMULAS = dict(
    flood="100 * min(1, 2.5*flood_frac + 12*urban_flooded_frac + 4*crop_flooded_frac)",
    drought="100 * min(1, 1.6 * stress_frac * sqrt(vegetated_frac))",
    urban="100 * min(1, 5 * urban_frac)",
    composite="0.45*flood + 0.40*drought + 0.15*urban",
)
RISK_NOTE = "Hand-weighted heuristic, not a validated risk model."
# Single source of truth for the coefficients in RISK_FORMULAS (risk_sensitivity.py perturbs these).
RISK_COEF = dict(flood_a=2.5, flood_urban=12.0, flood_crop=4.0, drought_a=1.6, drought_exp=0.5, urban_a=5.0,
                 w_flood=0.45, w_drought=0.40, w_urban=0.15)


def risk_indices(flood_frac, urban_fl, crop_fl, stress, veg_frac, urban_frac, coef=RISK_COEF, renormalise=False):
    """Vectorised risk indices (numpy arrays or floats) -> (flood, drought, urban, composite), each 0..100.
    ``renormalise`` rescales the three composite weights to sum to 1 (used in sensitivity analysis)."""
    c = coef
    fl = 100 * np.minimum(1.0, c["flood_a"] * flood_frac + c["flood_urban"] * urban_fl + c["flood_crop"] * crop_fl)
    dr = 100 * np.minimum(1.0, c["drought_a"] * stress * np.power(veg_frac, c["drought_exp"]))
    ur = 100 * np.minimum(1.0, c["urban_a"] * urban_frac)
    w = np.array([c["w_flood"], c["w_drought"], c["w_urban"]], np.float64)
    if renormalise:
        w = w / w.sum()
    return fl, dr, ur, w[0] * fl + w[1] * dr + w[2] * ur


def _risk(value, name, inputs):
    return dict(value=None if value is None else round(float(value), 1), formula=RISK_FORMULAS[name],
                inputs={k: (None if v is None else round(float(v), 5)) for k, v in inputs.items()}, kind="heuristic", note=RISK_NOTE)


def compute(sc, pred, derived, sensors, gsd_m, calibration=None):
    """pred: runner outputs; derived: arrays from render.output_layers (lc, ent, cloud_weight)."""
    o = sc["optical"]
    H, W = o.shape[1:]
    n = H * W
    px_km2 = None if not gsd_m else (gsd_m / 1000.0) ** 2   # unknown GSD -> no areas (fractions only)

    def A(frac_or_count, count=False):
        if px_km2 is None:
            return None
        return round(float(frac_or_count * (px_km2 if count else n * px_km2)), 3)

    km2 = None if px_km2 is None else n * px_km2
    lc = derived.get("lc")
    fprob = pred.get("flood_prob")
    veg = pred.get("veg")
    flood = None if fprob is None else fprob > 0.5
    st = dict(area_km2=None if km2 is None else round(km2, 2), gsd_m=None if not gsd_m else float(gsd_m), gsd_assumed=not gsd_m)

    frac = np.bincount(lc.ravel(), minlength=N_CLASSES) / n if lc is not None else None
    st["composition"] = None if frac is None else [dict(name=CLASS_NAMES[i], frac=float(frac[i]), km2=A(frac[i])) for i in range(N_CLASSES)]

    urban_fl = crop_fl = None
    if flood is not None:
        urban_fl = float((flood & (lc == 3)).mean()) if lc is not None else None
        crop_fl = float((flood & (lc == 2)).mean()) if lc is not None else None
        st["flood"] = dict(frac=float(flood.mean()), km2=A(flood.mean()), urban_frac=urban_fl, crop_frac=crop_fl,
                           urban_km2=None if urban_fl is None else A(urban_fl),
                           crop_km2=None if crop_fl is None else A(crop_fl))
    else:
        st["flood"] = None

    stress = vegm_frac = None
    if veg is not None and lc is not None:
        vegm = np.isin(lc, VEG_CLASSES)
        vegm_frac = float(vegm.mean())
        stress = float(((veg < 0.30) & vegm).sum() / max(vegm.sum(), 1))
        st["veg"] = dict(mean=float(veg[vegm].mean()) if vegm.any() else 0.0, stress_frac=stress,
                         stress_km2=A((veg < .30)[vegm].sum(), count=True))
    elif veg is not None:
        st["veg"] = dict(mean=float(veg.mean()), stress_frac=None, stress_km2=None)
    else:
        st["veg"] = None

    cw = derived.get("cloud_weight")
    if "cloud_prob" in pred and sensors[0]:
        st["cloud_frac"] = float((pred["cloud_prob"] > 0.5).mean())
        st["cloud_method"] = "cloud_head"
    elif derived.get("composite_method") == "threshold_rule":
        from .render import threshold_cloud
        st["cloud_frac"] = float(threshold_cloud(o).mean())
        st["cloud_method"] = "threshold_rule"
    else:
        st["cloud_frac"] = None
        st["cloud_method"] = None
    st["rebuilt_frac"] = None if cw is None else float((cw > 0.5).mean())

    # confidence (raw) and confidence_calibrated use the SAME metric: mean max class probability.
    raw = pred.get("lc_prob")
    st["confidence"] = None if raw is None else float(raw.max(0).mean())
    cal = pred.get("lc_prob_calibrated")
    applied = bool(calibration) and calibration_improves(calibration) and cal is not None
    st["confidence_calibrated"] = float(cal.max(0).mean()) if applied else None
    st["calibration"] = None if not calibration or raw is None else dict(
        method=calibration.get("method"), T=calibration.get("T"), ece_before=calibration.get("ece_before"),
        ece_after=calibration.get("ece_after"), applied=applied, decision_split="val64",
        ece_val_before=(calibration.get("val") or {}).get("ece_before"), ece_val_after=(calibration.get("val") or {}).get("ece_after"),
        meaning="confidence = mean max class probability; confidence_calibrated = same after temperature scaling, "
                "only reported when scaling lowered validation ECE; ece_before/ece_after are test outcomes (simulated data)")

    # ---------------- heuristic risk indices
    flood_frac = None if flood is None else float(flood.mean())
    z = 0.0
    f_, d_, u_, _ = risk_indices(flood_frac or z, urban_fl or z, crop_fl or z, stress or z, vegm_frac or z,
                                 frac[3] if frac is not None else z)
    fl_v = None if (flood_frac is None or urban_fl is None) else float(f_)
    dr_v = None if stress is None else float(d_)
    ur_v = None if frac is None else float(u_)
    w = RISK_COEF
    comp = None if None in (fl_v, dr_v, ur_v) else w["w_flood"] * fl_v + w["w_drought"] * dr_v + w["w_urban"] * ur_v
    st["risk"] = dict(
        flood=_risk(fl_v, "flood", dict(flood_frac=flood_frac, urban_flooded_frac=urban_fl, crop_flooded_frac=crop_fl)),
        drought=_risk(dr_v, "drought", dict(stress_frac=stress, vegetated_frac=vegm_frac)),
        urban=_risk(ur_v, "urban", dict(urban_frac=None if frac is None else frac[3])),
        composite=_risk(comp, "composite", dict(flood=fl_v, drought=dr_v, urban=ur_v)),
    )
    st["accuracy"] = accuracy(sc, lc, flood, veg) if all(k in sc for k in ("lc", "flood", "veg")) else None
    if st["accuracy"] is None:
        st.pop("accuracy")
    st["insights"] = insights(st, sensors)
    return st


def calibration_improves(cal):
    """Apply temperature scaling only if it lowered ECE on the VALIDATION split (never decided on test)."""
    if not isinstance(cal, dict):
        return False
    if "apply" in cal:
        return bool(cal["apply"])
    try:
        return float(cal["val"]["ece_after"]) < float(cal["val"]["ece_before"])
    except (KeyError, TypeError, ValueError):
        return False   # no validation numbers -> do not apply


def accuracy(sc, lc, flood, veg):
    out = dict(oa=None, miou=None, flood_iou=None, veg_mae=None, true_cloud=float(np.mean(sc["cloud"])) if "cloud" in sc else None)
    if lc is not None:
        y = np.asarray(sc["lc"]).astype(int)
        conf = np.bincount(y.ravel() * N_CLASSES + lc.ravel(), minlength=N_CLASSES ** 2).reshape(N_CLASSES, N_CLASSES)
        iou = np.diag(conf) / np.maximum(conf.sum(0) + conf.sum(1) - np.diag(conf), 1)
        present = conf.sum(1) > 0
        out.update(oa=float(np.diag(conf).sum() / conf.sum()), miou=float(iou[present].mean()))
    if flood is not None:
        yf = np.asarray(sc["flood"]) > 0
        out["flood_iou"] = float((flood & yf).sum() / max((flood | yf).sum(), 1)) if (flood | yf).any() else None
    if veg is not None:
        out["veg_mae"] = float(np.abs(veg - sc["veg"]).mean())
    return out


def insights(st, sensors):
    out = []
    names = ["optical", "SAR", "DEM"]
    on = [n for n, s in zip(names, sensors) if s]
    out.append(dict(kind="sensors", text=f"Using {' + '.join(on)} input."))
    if st.get("gsd_assumed"):
        out.append(dict(kind="info", text="Pixel size unknown for this scene, so areas are not reported (fractions only)."))
    fl = st.get("flood")
    if fl is not None:
        if fl["frac"] > 0.01:
            area = f"{fl['km2']:.2f} km² (" if fl.get("km2") is not None else "("
            s = f"Floodwater covers {area}{fl['frac']*100:.1f}% of the scene)"
            ex = []
            def amt(k):
                return f"{fl[k+'_km2']:.2f} km²" if fl.get(k + "_km2") is not None else f"{fl[k+'_frac']*100:.1f}% of the scene as"
            if (fl.get("crop_frac") or 0) > 0.0005: ex.append(f"{amt('crop')} cropland")
            if (fl.get("urban_frac") or 0) > 0.0005: ex.append(f"{amt('urban')} urban fabric")
            if ex: s += "; exposed: " + " and ".join(ex) + "."
            elif fl.get("crop_frac") is not None: s += "; little built or farmed land is exposed."
            else: s += "."
            out.append(dict(kind="flood", text=s))
        else:
            out.append(dict(kind="ok", text="No significant surface flooding detected."))
    v = st.get("veg")
    if v is not None and v.get("stress_frac") is not None:
        if v["stress_frac"] > 0.12:
            area = f" ({v['stress_km2']:.2f} km²)" if v.get("stress_km2") is not None else ""
            out.append(dict(kind="drought", text=f"Vegetation stress on {v['stress_frac']*100:.0f}% of vegetated land{area} — consistent with drought or crop failure."))
        else:
            out.append(dict(kind="ok", text=f"Vegetation is largely healthy (mean index {v['mean']:.2f})."))
    if st.get("cloud_frac") is not None and st["cloud_frac"] > 0.12:
        out.append(dict(kind="cloud", text=f"~{st['cloud_frac']*100:.0f}% of optical pixels are cloud-obscured; those pixels in the cloud-free layer are inferred by the model."))
    if not sensors[0]:
        out.append(dict(kind="cloud", text="Optical input is off: the cloud-free layer is entirely inferred from the other sensors."))
    if st.get("composition"):
        top = max(st["composition"], key=lambda c: c["frac"])
        conf = st.get("confidence_calibrated")
        ctext = f"calibrated confidence {conf*100:.0f}%" if conf is not None else f"mean model confidence {st['confidence']*100:.0f}% (not calibrated)"
        out.append(dict(kind="lc", text=f"Dominant land cover: {top['name'].lower()} ({top['frac']*100:.0f}%); {ctext}."))
    return out
