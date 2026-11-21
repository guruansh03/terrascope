"""Geospatial foundation models (Prithvi-EO-2.0, TerraMind-1.0, TerraFM) behind the Runner interface.

Entry points: ``registry.list_models()``, ``registry.get_runner(model_id)``, ``finetune.finetune(...)``,
``export.export_onnx(...)``. terratorch / onnxruntime are imported lazily.
"""
from .errors import RunnerUnavailable  # noqa: F401
