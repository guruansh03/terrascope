"""Background analysis jobs with stage events (consumed as Server-Sent Events by the API).

Stages are emitted by the engine/runner at the real moment each begins:
queued -> preparing -> encoding -> decoding -> rendering -> done | error
"""
from __future__ import annotations

import secrets, threading, time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

TERMINAL = ("done", "error")


class Job:
    def __init__(self):
        self.id = "j" + secrets.token_urlsafe(12)
        self.t0 = time.perf_counter()
        self.status = "queued"
        self.events = []
        self.result = None
        self.error = None
        self.cond = threading.Condition()

    def emit(self, stage, detail=None):
        ev = dict(stage=stage, t_ms=int(round((time.perf_counter() - self.t0) * 1000)))
        if detail:
            ev["detail"] = str(detail)[:300]
        with self.cond:
            self.events.append(ev)
            if stage in TERMINAL:
                self.status = stage
            elif stage != "queued":
                self.status = "running"
            self.cond.notify_all()

    def public(self):
        d = dict(id=self.id, status=self.status, events=list(self.events))
        if self.status == "done":
            d["result"] = self.result
        if self.status == "error":
            d["error"] = self.error
        return d


class JobManager:
    def __init__(self, workers=1, keep=64):
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="terrascope-job")
        self.jobs: "OrderedDict[str, Job]" = OrderedDict()
        self.keep = keep
        self.lock = threading.Lock()

    def submit(self, fn, error_message=lambda e: "Analysis failed."):
        """fn(on_stage) -> result. Exceptions become an 'error' event with a curated message."""
        job = Job()
        job.emit("queued")
        with self.lock:
            self.jobs[job.id] = job
            while len(self.jobs) > self.keep:
                self.jobs.popitem(last=False)

        def run():
            try:
                job.result = fn(job.emit)
                job.emit("done")
            except Exception as e:  # noqa: BLE001 - reported to the client as an event
                job.error = error_message(e)
                job.emit("error", job.error)

        self.pool.submit(run)
        return job

    def get(self, jid):
        with self.lock:
            return self.jobs.get(jid)

    def stream(self, jid, keepalive_s=15.0, max_s=600.0):
        """Yield events (dicts) as they happen; yields None as a keep-alive tick. Ends after a terminal event."""
        job = self.get(jid)
        if job is None:
            return
        i, t_end = 0, time.time() + max_s
        while time.time() < t_end:
            with job.cond:
                if i >= len(job.events):
                    job.cond.wait(timeout=keepalive_s)
                new = job.events[i:]
            if not new:
                yield None
                continue
            for ev in new:
                i += 1
                yield ev
                if ev["stage"] in TERMINAL:
                    return
