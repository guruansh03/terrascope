"""RS-owned API routes (included by ``terrascope.server.app``):

GET    /api/models              -> registry.list_models()  (TerraFM first)
GET    /api/scenes              -> uploads + GeoTIFF imports + data/demo scenes
GET    /api/scenes/{id}         -> one scene summary
PATCH  /api/scenes/{id} {name}  -> rename (demo scenes are read-only -> 403)
DELETE /api/scenes/{id}         -> delete  (demo scenes are read-only -> 403)
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..real.scenes import STORE, SceneNotFound, SceneReadOnly

log = logging.getLogger("terrascope.routes_rs")
router = APIRouter()


class RenameReq(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)


def _store():
    return STORE


@router.get("/api/models")
def models():
    from ..fm import registry
    try:
        return registry.list_models()
    except Exception:
        log.exception("list_models failed")
        raise HTTPException(500, "Could not list models.")


@router.get("/api/scenes")
def scenes():
    return _store().list()


@router.get("/api/scenes/{scene_id}")
def scene(scene_id: str):
    try:
        return _store().summary(scene_id)
    except SceneNotFound:
        raise HTTPException(404, "Scene not found.")


@router.patch("/api/scenes/{scene_id}")
def rename_scene(scene_id: str, r: RenameReq):
    try:
        return _store().rename(scene_id, r.name)
    except SceneNotFound:
        raise HTTPException(404, "Scene not found.")
    except SceneReadOnly:
        raise HTTPException(403, "Demo scenes are read-only.")
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.delete("/api/scenes/{scene_id}")
def delete_scene(scene_id: str):
    try:
        _store().delete(scene_id)
    except SceneNotFound:
        raise HTTPException(404, "Scene not found.")
    except SceneReadOnly:
        raise HTTPException(403, "Demo scenes are read-only.")
    return {"deleted": scene_id}
