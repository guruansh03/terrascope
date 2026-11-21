"""Shared builders for tests/fm (unique module name: avoids conftest import clashes)."""
import json, os

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def make_sample(task, i, H=48, W=48):
    """Deterministic fake chip in the standard sample format (learnable: water = dark NIR)."""
    r = np.random.default_rng(1000 + i)
    opt = (0.05 + 0.25 * r.random((6, H, W))).astype(np.float32)
    water = np.zeros((H, W), bool)
    water[:, : W // 3 + i % 5] = True
    opt[3][water] = 0.02
    sar = np.stack([np.where(water, -22.0, -9.0), np.where(water, -28.0, -15.0)]).astype(np.float32)
    sar += r.standard_normal(sar.shape).astype(np.float32)
    dem = np.stack([np.full((H, W), 120.0), np.full((H, W), 2.0)]).astype(np.float32)
    s = dict(id=f"chip{i}", optical=opt, sar=sar, dem=dem)
    if task in ("flood", "burn"):
        m = water.astype(np.int64)
        m[0, :] = -1
        s["mask"] = m
        if task == "burn":
            s["sar"] = s["dem"] = None
    elif task == "landcover":
        m = np.where(water, 0, 1 + (np.arange(W)[None] // 12) % 5).astype(np.int64).repeat(1, 0)
        s["mask"] = np.broadcast_to(m, (H, W)).copy()
    elif task == "cloud":
        s["target"] = opt.copy()
        cl = opt.copy()
        cl[:, :10] = 0.7
        s["optical"] = cl
        s["dem"] = None
    return s


def load_fake(rec):
    return make_sample(rec["task"], rec["i"], rec.get("H", 48), rec.get("W", 48))


def fake_records(task, n_train=4, n_val=2, n_test=2, H=48, W=48):
    mk = lambda a, b: [dict(task=task, i=k, H=H, W=W, id=f"c{k}") for k in range(a, b)]
    return dict(train=mk(0, n_train), val=mk(100, 100 + n_val), test=mk(200, 200 + n_test))


def write_ckpt(model, root, seed=0, score=0.5, frozen=False):
    """Save a TaskModel as a completed fine-tuned run (what finetune() leaves on disk)."""
    from terrascope.fm.model import ckpt_path, model_sensors, save_checkpoint
    p = ckpt_path(model.task, model.model_id, seed, root)
    save_checkpoint(p, model, seed, frozen, dict(val=dict(score=score), epoch=0, dataset="fake",
                                                 sensors=model_sensors(model.model_id, model.task), complete=True))
    with open(os.path.splitext(p)[0] + ".json", "w") as f:
        json.dump(dict(model_id=model.model_id, task=model.task, seed=seed, complete=True, encoder_frozen=frozen,
                       best_val=dict(score=score)), f)
    return p
