"""Runner interface: the seam between the app/engine and every model (see docs/PHASE3_CONTRACT.md)."""
from __future__ import annotations


class RunnerUnavailable(RuntimeError):
    """Weights or fine-tuned heads missing -> the API answers 409 with this message."""


def is_unavailable(exc: BaseException) -> bool:
    # RS may define its own RunnerUnavailable class; match by name so either works.
    return isinstance(exc, RunnerUnavailable) or type(exc).__name__ == "RunnerUnavailable"


class Runner:
    id: str = ""
    name: str = ""
    family: str = ""
    params: int | None = None
    tasks: list = []
    sensors: list = []
    status: str = "ready"
    weights_file: str | None = None
    device: str = "cpu"

    def predict(self, sc: dict, sensors: list, on_stage=lambda stage: None) -> dict:
        """Return any subset of lc_prob (6,H,W), flood_prob, veg, recon (6,H,W), cloud_prob, burn_prob,
        feats (N,D) + feat_hw (h,w); numpy float32. Must call on_stage("encoding") right before the
        backbone forward and on_stage("decoding") right before heads/decoder."""
        raise NotImplementedError
