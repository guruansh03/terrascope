"""Model registry: ``list_models()`` (schema of GET /api/models) and ``get_runner(model_id)``.

Status is derived from files on disk, never assumed:
* ``ready``          - at least one fine-tuned task checkpoint is usable (it stores its encoder, or the
                       pretrained backbone file is present), or (TERRASCOPE_RUNTIME=onnx) an ONNX export exists
* ``weights_missing``- no usable fine-tuned checkpoint and the pretrained backbone file is not on disk
* ``not_finetuned``  - pretrained backbone present, but no fine-tuned task head yet
Parameter counts are measured from freshly built modules (no weights needed) and cached per process.
"""
from __future__ import annotations

import functools, logging, os, threading, warnings

from .errors import RunnerUnavailable
from .specs import FM_IDS, MODEL_ORDER, MODELS, ALL_TASKS

log = logging.getLogger("terrascope.fm.registry")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_lock = threading.Lock()
_runners: dict = {}


def _rel(p):
    if not p:
        return None
    try:
        return os.path.relpath(p, ROOT).replace("\\", "/")
    except ValueError:
        return p


def runtime() -> str:
    r = os.environ.get("TERRASCOPE_RUNTIME", "torch").lower()
    return r if r in ("torch", "onnx") else "torch"


def ckpt_root() -> str:
    from .model import CKPT_ROOT
    return os.environ.get("TERRASCOPE_REAL_CKPT", CKPT_ROOT)


def onnx_root() -> str:
    from .export import ONNX_ROOT
    return os.environ.get("TERRASCOPE_ONNX_DIR", ONNX_ROOT)


@functools.lru_cache(maxsize=None)
def measure_params(model_id: str) -> int | None:
    """Encoder parameter count measured from the built module (reference config: flood task)."""
    import torch
    from .encoders import build_encoder
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                with torch.device("meta"):
                    enc = build_encoder(model_id, "flood")
            except Exception:
                enc = build_encoder(model_id, "flood")
        return int(enc.encoder_params())
    except Exception as e:  # terratorch missing etc.
        log.warning("could not measure params for %s: %s", model_id, e)
        return None


# ------------------------------------------------------------------------------------------- terrafm
def _terrafm_runner():
    with _lock:
        r = _runners.get("terrafm")
        if r is None:
            from ..core.engine.terrafm_runner import TerraFMRunner
            r = TerraFMRunner()
            _runners["terrafm"] = r
        return r


def _terrafm_entry() -> dict:
    spec = MODELS["terrafm"]
    try:
        r = _terrafm_runner()
        d = r.info() if hasattr(r, "info") else {}
        base = dict(id="terrafm", name=getattr(r, "name", spec["name"]), family="terrafm",
                    params=getattr(r, "params", None), sensors=list(getattr(r, "sensors", spec["sensors"])),
                    tasks=list(getattr(r, "tasks", ["landcover", "flood", "veg"])), status="ready",
                    weights_file=getattr(r, "weights_file", None), license=getattr(r, "license", spec["license"]),
                    source_url=None, data="simulated")
        base.update({k: v for k, v in d.items() if k not in base or base[k] is None})
        return base
    except Exception as e:
        return dict(id="terrafm", name=spec["name"], family="terrafm", params=None, sensors=spec["sensors"],
                    tasks=["landcover", "flood", "veg"], status="weights_missing", weights_file=None,
                    license=spec["license"], source_url=None, detail=str(e)[:300])


# ------------------------------------------------------------------------------------------- FMs
def _usable_ckpts(model_id: str) -> tuple[dict, str | None]:
    from .runners import best_checkpoints
    from .weights import weights_path
    w = weights_path(model_id)
    cks = best_checkpoints(model_id, ckpt_root())
    usable = {}
    for t, p in cks.items():
        if w:
            usable[t] = p
            continue
        try:  # frozen-encoder checkpoints need the backbone file
            import json
            with open(os.path.splitext(p)[0] + ".json") as f:
                if not json.load(f).get("encoder_frozen"):
                    usable[t] = p
        except (OSError, ValueError):
            pass
    return usable, w


def _onnx_tasks(model_id: str) -> list[str]:
    d = os.path.join(onnx_root(), model_id)
    if not os.path.isdir(d):
        return []
    return sorted(fn[:-5] for fn in os.listdir(d) if fn.endswith(".json"))


def model_status(model_id: str) -> dict:
    usable, w = _usable_ckpts(model_id)
    onnx = _onnx_tasks(model_id)
    if runtime() == "onnx" and onnx:
        status, ready = "ready", onnx
        wf = f"artifacts/onnx/{model_id}/"
    elif usable:
        status, ready = "ready", sorted(usable)
        wf = _rel(next(iter(usable.values())))
    elif not w and MODELS[model_id].get("needs_weights", True):
        status, ready, wf = "weights_missing", [], None
    else:
        status, ready, wf = "not_finetuned", [], _rel(w)
    return dict(status=status, tasks_ready=ready, weights_file=wf, pretrained_weights=_rel(w), onnx_tasks=onnx)


def _fm_entry(model_id: str) -> dict:
    spec = MODELS[model_id]
    st = model_status(model_id)
    tasks_ready = st["tasks_ready"] + (["veg"] if st["tasks_ready"] and "optical" in spec["sensors"] else [])
    return dict(id=model_id, name=spec["name"], family=spec["family"], params=measure_params(model_id),
                params_scope="whole U-Net (measured)" if spec["family"] == "unet" else "encoder (measured)", sensors=list(spec["sensors"]),
                # "tasks" = what the model can produce NOW (fine-tuned heads + derived veg) when ready, otherwise the
                # tasks it supports once fine-tuned; "tasks_supported" is always the full list
                tasks=tasks_ready if st["status"] == "ready" else [t for t in ALL_TASKS if t in spec["tasks"]],
                tasks_supported=[t for t in ALL_TASKS if t in spec["tasks"]], tasks_ready=tasks_ready,
                status=st["status"],
                weights_file=st["weights_file"], pretrained_weights=st["pretrained_weights"],
                license=spec["license"], source_url=spec["source_url"], runtime=runtime(),
                derived_outputs={"veg": "spectral index (terrascope.real.vegetation), not a learned output"},
                data="real (fine-tuned on public datasets)" if st["status"] == "ready" else None)


def list_models() -> list[dict]:
    out = [_terrafm_entry()]
    for mid in FM_IDS:
        try:
            out.append(_fm_entry(mid))
        except Exception as e:  # never break the listing
            spec = MODELS[mid]
            out.append(dict(id=mid, name=spec["name"], family=spec["family"], params=None, sensors=spec["sensors"],
                            tasks=spec["tasks"], status="weights_missing", weights_file=None, license=spec["license"],
                            source_url=spec["source_url"], detail=f"{type(e).__name__}: {e}"[:300]))
    return out


def get_runner(model_id: str):
    if model_id not in MODELS:
        raise KeyError(model_id)
    if model_id == "terrafm":
        try:
            return _terrafm_runner()
        except Exception as e:
            if type(e).__name__ == "RunnerUnavailable":
                raise RunnerUnavailable(str(e)) from e
            raise
    rt = runtime()
    st = model_status(model_id)
    name = MODELS[model_id]["name"]
    if st["status"] != "ready":
        if st["status"] == "weights_missing":
            raise RunnerUnavailable(f"{name}: pretrained weights and fine-tuned heads are missing on this machine. "
                                    "On the PC run get_data.bat weights, then train.bat quick.")
        raise RunnerUnavailable(f"{name}: not fine-tuned yet. On the PC run train.bat quick (or full).")
    key = (model_id, rt, tuple(st["tasks_ready"]))
    with _lock:
        r = _runners.get(key)
        if r is None:
            from .runners import FMRunner, OnnxFMRunner
            if rt == "onnx" and st["onnx_tasks"]:
                r = OnnxFMRunner(model_id, onnx_root())
            else:
                r = FMRunner(model_id, ckpt_root())
            r.params = r.params or measure_params(model_id)
            _runners[key] = r
        return r


def reset():
    """Forget cached runners (tests / after training)."""
    with _lock:
        _runners.clear()
