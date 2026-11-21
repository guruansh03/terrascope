import json

import gpu_check


def test_gpu_check_writes_json_on_cpu_machine(tmp_path):
    out = tmp_path / "pc_info.json"
    assert gpu_check.main(["--out", str(out)]) == 0
    d = json.loads(out.read_text())
    for k in ("generated_at", "os", "python", "cpu", "ram", "disk", "torch", "checks", "cuda_training_ready"):
        assert k in d
    assert d["cpu"]["logical_cores"] >= 1 and d["cpu"]["name"]
    assert d["disk"]["free_gb"] > 0 and d["ram"]["total_gb"] > 0
    names = {c["name"] for c in d["checks"]}
    assert {"torch_cuda_build", "cuda_available", "gpu_kernel_runs", "disk_free", "nvidia_driver"} <= names
    if not d["torch"].get("cuda_available"):
        assert d["cuda_training_ready"] is False and d["torch"]["gpus"] == []


def test_strict_fails_without_cuda(tmp_path):
    out = tmp_path / "pc_info.json"
    rc = gpu_check.main(["--out", str(out), "--strict"])
    d = json.loads(out.read_text())
    assert rc == (0 if d["cuda_training_ready"] else 1)


def test_explain_maps_checks_to_guidance():
    def info(**ok):
        names = ["python_64bit", "torch_importable", "torch_cuda_build", "nvidia_driver", "cuda_available",
                 "gpu_kernel_runs", "disk_free"]
        return {"checks": [{"name": n, "ok": ok.get(n, True), "detail": "x"} for n in names]}
    assert gpu_check.explain(info()) is None
    assert "setup.bat" in gpu_check.explain(info(torch_cuda_build=False)).todo
    assert "driver" in gpu_check.explain(info(nvidia_driver=False)).todo.lower()
    assert "cu128" in gpu_check.explain(info(gpu_kernel_runs=False)).todo
