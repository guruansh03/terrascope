"""TerraFM implementation of the Runner interface (imported by RS's ``terrascope.fm.registry``)."""
from __future__ import annotations

import hashlib, json, os, threading

import numpy as np
import torch

from ..model import SHIFTS, load_checkpoint
from .runner import Runner, RunnerUnavailable
from .validate import SceneError

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
CKPT_DIR = os.path.join(ROOT, "artifacts", "ckpt")
CALIBRATION = os.path.join(ROOT, "artifacts", "calibration.json")


def resolve_ckpt(ckpt="finetuned.pt"):
    """Current (v2) checkpoint, else the archived v1 one so the app keeps working mid-retrain."""
    env = os.environ.get("TERRASCOPE_CKPT")
    cands = [env] if env else []
    if os.path.isabs(ckpt):
        cands.append(ckpt)
    else:
        cands += [os.path.join(CKPT_DIR, ckpt), os.path.join(CKPT_DIR, "v1", ckpt)]
    for p in cands:
        if p and os.path.exists(p):
            return p
    return None


def load_calibration(ckpt_path, norm):
    """Temperature only applies to the checkpoint it was fitted on."""
    try:
        with open(CALIBRATION) as f:
            cal = json.load(f)
    except (OSError, ValueError):
        return None
    if cal.get("ckpt") != os.path.basename(ckpt_path) or cal.get("norm") != norm or "/v1/" in ckpt_path.replace("\\", "/"):
        return None
    return cal


def measured_shift_ensemble_ok():
    """Use the 4-shift reconstruction ensemble unless evaluate measured it as worse than one pass."""
    try:
        with open(os.path.join(ROOT, "artifacts", "metrics.json")) as f:
            return bool(json.load(f).get("recon", {}).get("shift_ensemble_used", True))
    except (OSError, ValueError, AttributeError):
        return True


class TerraFMRunner(Runner):
    id = "terrafm"
    name = "TerraFM (ours)"
    family = "terrafm"
    sensors = ["optical", "sar", "dem"]
    license = "project-internal"
    source_url = None

    def __init__(self, ckpt="finetuned.pt", device=None, shift_ensemble=None):
        path = resolve_ckpt(ckpt)
        if path is None:
            self.status = "weights_missing"
            raise RunnerUnavailable("TerraFM checkpoint not found. Run training first (run_training.sh / train.bat sim).")
        self.path = path
        self.weights_file = os.path.relpath(path, ROOT).replace("\\", "/")
        self.device = device or os.environ.get("TERRASCOPE_DEVICE", "cpu")
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
        self.model, self.cfg = load_checkpoint(path)
        self.model.to(self.device)
        self.norm = self.cfg.get("norm", "v1")
        self.params = int(sum(p.numel() for p in self.model.parameters()))
        self.has_cloud_head = bool(self.cfg.get("cloud_head", False))
        self.tasks = ["landcover", "flood", "veg"] + (["cloud"] if self.has_cloud_head else [])
        self.status = "ready"
        if shift_ensemble is None:
            env = os.environ.get("TERRASCOPE_SHIFT_ENSEMBLE")
            shift_ensemble = (env != "0") if env is not None else measured_shift_ensemble_ok()
        self.shifts = SHIFTS if shift_ensemble else ((0, 0),)
        self.calibration = load_calibration(path, self.norm)
        self.T = float(self.calibration["T"]) if self.calibration else None
        self.lock = threading.Lock()
        with open(path, "rb") as f:   # identifies the exact weights + inference settings for request-keyed caching
            digest = hashlib.sha256(f.read()).hexdigest()[:16]
        self.fingerprint = f"terrafm:{digest}:shifts{len(self.shifts)}:T{self.T}"

    def info(self):
        return dict(id=self.id, name=self.name, family=self.family, params=self.params, sensors=self.sensors,
                    tasks=self.tasks, status=self.status, weights_file=self.weights_file, license=self.license,
                    source_url=self.source_url, norm=self.norm)

    @torch.no_grad()
    def predict(self, sc, sensors, on_stage=lambda stage: None):
        m = self.model
        avail = torch.tensor([list(map(bool, sensors))], device=self.device)
        x = m.normalize(torch.from_numpy(np.ascontiguousarray(sc["optical"]))[None],
                        torch.from_numpy(np.ascontiguousarray(sc["sar"]))[None],
                        torch.from_numpy(np.ascontiguousarray(sc["dem"]))[None]).to(self.device)
        H, W = x.shape[-2:]
        with self.lock:
            on_stage("encoding")
            enc = m.encode_shifts(x, avail, self.shifts)
            feats = enc[0][2]
            if not torch.isfinite(feats).all():
                raise SceneError("model produced non-finite features for this input")
            on_stage("decoding")
            out = m.heads_from(feats, x, avail)
            recon = m.recon_from(enc, H, W)
        lc = out["lc"][0].float().cpu()
        res = dict(
            lc_prob=torch.softmax(lc, 0).numpy(),
            flood_prob=torch.sigmoid(out["flood"][0]).float().cpu().numpy(),
            veg=out["veg"][0].float().cpu().numpy(),
            recon=(recon[0, :6].float().cpu().numpy() * 0.2 + 0.2).clip(0, 1).astype(np.float32),
            feats=feats[0].float().cpu().numpy(),
            feat_hw=(H // m.patch, W // m.patch),
        )
        if self.T:
            res["lc_prob_calibrated"] = torch.softmax(lc / self.T, 0).numpy()
        if "cloud" in out and sensors[0]:
            res["cloud_prob"] = torch.sigmoid(out["cloud"][0]).float().cpu().numpy()
        return res
