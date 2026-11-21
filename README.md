# TerraScope

Multimodal remote-sensing segmentation: a compact sensor-agnostic model (TerraFM), a training and evaluation
pipeline, and an offline web workbench.

* **TerraFM**: 0.69 M-parameter vision transformer that fuses optical (6 bands), SAR (VV/VH) and DEM inputs and works
  with any subset of them. Pre-trained by masked, sensor-dropped, cloud-corrupted reconstruction.
* **Tasks**: land cover, flood extent, vegetation vigour, cloud mask and cloud-free reconstruction, burn scars,
  two-date change.
* **Baselines**: Prithvi-EO-2.0 (tiny, 100M), TerraMind-1.0 tiny, ResNet-18 U-Nets.
* **Data**: procedural simulator; real data from Sen1Floods11, HLS Burn Scars and Copernicus DEM.

## Requirements
* Windows 10/11, Python 3.12 (64-bit)
* NVIDIA GPU with a recent driver for training (tested on RTX 5060 8 GB); CPU is enough for the demo

## Usage (Windows)
Run the `.bat` files from the repository folder, in this order:

| step | command | what it does |
|---|---|---|
| 1 | `setup.bat` | creates `.venv` and installs dependencies |
| 2 | `gpu_check.bat` | checks GPU, CUDA and disk space |
| 3 | `get_data.bat` | downloads datasets and pretrained weights into `data\` |
| 4 | `train.bat` | trains TerraFM and fine-tunes all models |
| 5 | `evaluate.bat` | writes metrics to `artifacts\` |
| 6 | `start_local.bat` | starts the web app at http://127.0.0.1:8000 |

Other scripts: `paper_experiments.bat` (full multi-seed experiment grid), `test_all.bat` (test suite),
`make_demo_bundle.bat` (offline CPU demo with ONNX models), `bench_laptop.bat` (CPU benchmark).

Each script logs to `logs\`.

## Layout
| path | contents |
|---|---|
| `terrascope/core/` | simulator, TerraFM, training, evaluation, robustness, calibration, analysis engine |
| `terrascope/fm/` | foundation-model registry, runners (PyTorch and ONNX), fine-tuning, export |
| `terrascope/real/` | dataset download, loading and GeoTIFF import |
| `terrascope/server/` | FastAPI backend |
| `terrascope/web/` | web UI (no build step) |
| `scripts/pc/` | entry points behind the `.bat` files |
| `tests/` | pytest suites and end-to-end tests |
