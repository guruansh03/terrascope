"""Runner implementations (contract: docs/PHASE3_CONTRACT.md "Runner interface") for the foundation models.

FMRunner      - PyTorch, loads fine-tuned task checkpoints from ``artifacts/ckpt/real/<task>/``
OnnxFMRunner  - onnxruntime (CPU), loads ``artifacts/onnx/<model>/<task>.{encoder,head}.onnx`` (see export.py)

Outputs: ``flood_prob``, ``burn_prob``, ``lc_prob``, ``recon`` (cloud task) from fine-tuned heads, ``veg`` from the
DERIVED spectral index in :mod:`terrascope.real.vegetation` (not a learned output; see ``derived_outputs``),
``feats``/``feat_hw`` from the first task's encoder.
"""
from __future__ import annotations

import glob, json, os, threading

import numpy as np

from ..real import vegetation
from .errors import RunnerUnavailable
from .specs import LEARNED_TASKS, MODELS, TASK_SENSORS, TILE

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SENSOR_NAMES = ("optical", "sar", "dem")


def best_checkpoints(model_id: str, ckpt_root: str) -> dict[str, str]:
    """{task: path} choosing, per task, the completed seed with the best validation score."""
    out = {}
    for task in LEARNED_TASKS:
        best, best_s = None, None
        for p in glob.glob(os.path.join(ckpt_root, task, f"{model_id}_s*.pt")):
            if p.endswith(".last.pt"):
                continue
            js = os.path.splitext(p)[0] + ".json"
            score = None
            try:
                with open(js) as f:
                    meta = json.load(f)
                if not meta.get("complete"):
                    continue
                score = (meta.get("best_val") or {}).get("score")
            except (OSError, ValueError):
                continue
            s = -1e9 if score is None else float(score)
            if best is None or s > best_s:
                best, best_s = p, s
        if best:
            out[task] = best
    return out


def _postprocess(task, y):
    """y: torch (C,H,W) model output -> contract key, numpy float32."""
    import torch
    if task in ("flood", "burn"):
        return f"{task}_prob", torch.sigmoid(y[0]).float().cpu().numpy()
    if task == "landcover":
        return "lc_prob", torch.softmax(y.float(), 0).cpu().numpy()
    if task == "cloud":
        return "recon", y.float().clamp(0, 1).cpu().numpy()
    raise KeyError(task)


class _Base:
    family = ""
    derived_outputs = {"veg": vegetation.FORMULA}

    def _check_sensors(self, sensors, task_sensors):
        on = [n for n, s in zip(SENSOR_NAMES, sensors) if s]
        fam = MODELS[self.id]["family"]
        if fam == "prithvi" and not sensors[0]:
            raise RunnerUnavailable(f"{self.name} uses optical (Sentinel-2/HLS) bands only; enable the optical "
                                    "sensor or choose TerraMind / TerraFM for SAR- or DEM-only analysis.")
        usable = [s for s in on if s in self.sensors and s in task_sensors]
        if not usable:
            raise RunnerUnavailable(f"{self.name} cannot use the enabled sensors ({', '.join(on) or 'none'}).")

    def info(self) -> dict:
        return dict(id=self.id, name=self.name, family=self.family, params=self.params, sensors=self.sensors,
                    tasks=self.tasks, status=self.status, weights_file=self.weights_file,
                    license=MODELS[self.id]["license"], source_url=MODELS[self.id]["source_url"])


class FMRunner(_Base):
    def __init__(self, model_id: str, ckpt_root: str, device: str | None = None):
        import torch
        spec = MODELS[model_id]
        self.id, self.name, self.family = model_id, spec["name"], spec["family"]
        self.sensors = list(spec["sensors"])
        self.ckpts = best_checkpoints(model_id, ckpt_root)
        if not self.ckpts:
            raise RunnerUnavailable(f"{self.name} has no fine-tuned task heads yet (expected in {ckpt_root}). "
                                    "Run train.bat quick on the PC.")
        self.tasks = sorted(self.ckpts) + (["veg"] if "optical" in self.sensors else [])
        self.status = "ready"
        self.device = device or os.environ.get("TERRASCOPE_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
        self.weights_file = ";".join(os.path.relpath(p, ROOT).replace("\\", "/") for p in self.ckpts.values())
        self._models: dict = {}
        self.lock = threading.Lock()
        self.params = None
        self.tile = TILE

    def _load(self):
        if self._models:
            return self._models
        from .model import load_checkpoint
        from .weights import weights_path
        for task, p in sorted(self.ckpts.items()):
            try:
                self._models[task] = load_checkpoint(p, weights_path(self.id), device=self.device)
            except FileNotFoundError as e:
                raise RunnerUnavailable(f"{self.name}: {e}. Run get_data.bat weights.") from e
        self.params_total = max(m.n_params() for m in self._models.values())  # encoder + head
        return self._models

    def predict(self, sc: dict, sensors, on_stage=lambda stage: None) -> dict:
        import torch
        from .model import predict_tiled, to_tensors
        sensors = [bool(s) for s in sensors]
        with self.lock:
            models = self._load()
            use = {}
            for task, m in models.items():
                ts = (getattr(m, "meta", {}) or {}).get("sensors") or TASK_SENSORS[task]
                try:
                    self._check_sensors(sensors, ts)
                except RunnerUnavailable:
                    continue
                if task in ("burn", "cloud") and not sensors[0]:
                    continue  # optical-only / optical-target tasks need optical input
                use[task] = m
            if not use:
                self._check_sensors(sensors, self.sensors)  # raises the specific message
                raise RunnerUnavailable(f"{self.name}: no fine-tuned task can run with the enabled sensors.")
            o, s, d, a = to_tensors(sc, sensors, self.device)
            with torch.no_grad():
                # pad small scenes to one 224 tile (training crop size) -> identical maths to the ONNX runner
                outs, feats = predict_tiled(use, o, s, d, a, tile=self.tile, on_stage=on_stage, pad_to_tile=True)
        res = {}
        for task, y in outs.items():
            k, v = _postprocess(task, y)
            res[k] = v.astype(np.float32)
        if feats is not None:
            res["feats"] = feats[0].float().cpu().numpy().astype(np.float32)
            res["feat_hw"] = tuple(int(x) for x in feats[1])
        if sensors[0]:
            res["veg"] = vegetation.health_index(sc["optical"])
        return res


class OnnxFMRunner(_Base):
    """CPU onnxruntime runner over exported encoder/head graphs (one pair per task)."""

    def __init__(self, model_id: str, onnx_root: str):
        spec = MODELS[model_id]
        self.id, self.name, self.family = model_id, spec["name"], spec["family"]
        self.sensors = list(spec["sensors"])
        self.dir = os.path.join(onnx_root, model_id)
        self.meta = {}
        for js in glob.glob(os.path.join(self.dir, "*.json")):
            with open(js) as f:
                m = json.load(f)
            self.meta[m["task"]] = m
        if not self.meta:
            raise RunnerUnavailable(f"{self.name}: no ONNX export found in {self.dir}; run make_demo_bundle on the PC.")
        self.tasks = sorted(self.meta) + (["veg"] if "optical" in self.sensors else [])
        self.status = "ready"
        self.device = "cpu"
        self.params = None  # registry fills the measured encoder count
        self.params_total = max(int(m.get("params") or 0) for m in self.meta.values()) or None
        self.weights_file = ";".join(f"artifacts/onnx/{model_id}/{m['encoder']}" for m in self.meta.values())
        self._sess = {}
        self.lock = threading.Lock()

    def _sessions(self, task):
        if task not in self._sess:
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.intra_op_num_threads = int(os.environ.get("TERRASCOPE_ORT_THREADS", "0") or 0)
            m = self.meta[task]
            enc = ort.InferenceSession(os.path.join(self.dir, m["encoder"]), so, providers=["CPUExecutionProvider"])
            head = ort.InferenceSession(os.path.join(self.dir, m["head"]), so, providers=["CPUExecutionProvider"])
            self._sess[task] = (enc, head)
        return self._sess[task]

    def predict(self, sc: dict, sensors, on_stage=lambda stage: None) -> dict:
        import torch
        from .export import OnnxTask
        from .model import predict_tiled, to_tensors
        sensors = [bool(s) for s in sensors]
        use = {}
        for task, m in self.meta.items():
            need = m.get("required_sensors", [])
            if all(sensors[SENSOR_NAMES.index(n)] for n in need):
                use[task] = m
        if not use:
            self._check_sensors(sensors, self.sensors)
            req = sorted({n for m in self.meta.values() for n in m.get("required_sensors", [])})
            raise RunnerUnavailable(f"{self.name} (ONNX) needs sensors {req} for its exported tasks.")
        with self.lock:
            models = {t: OnnxTask(*self._sessions(t), self.meta[t]) for t in use}
            o, s, d, a = to_tensors(sc, sensors, "cpu")
            outs, feats = predict_tiled(models, o, s, d, a, tile=TILE, batch=1, on_stage=on_stage,
                                        pad_to_tile=True)
        res = {}
        for task, y in outs.items():
            k, v = _postprocess(task, y)
            res[k] = v.astype(np.float32)
        if feats is not None:
            res["feats"] = feats[0].float().numpy().astype(np.float32)
            res["feat_hw"] = tuple(int(x) for x in feats[1])
        if sensors[0]:
            res["veg"] = vegetation.health_index(sc["optical"])
        return res
