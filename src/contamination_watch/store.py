"""Persistence: SQLite for history, results.json for the public site.

The JSON export is the contract between the pipeline and the leaderboard —
the site is a pure function of results.json, so it can be rebuilt (or
re-styled) at any time without re-running probes.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import os
import sqlite3

from .config import PATHS

_SCHEMA = """
CREATE TABLE IF NOT EXISTS audits (
    id INTEGER PRIMARY KEY,
    model_id TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    score INTEGER,
    tier TEXT,
    verdict TEXT
);
CREATE TABLE IF NOT EXISTS cells (
    audit_id INTEGER NOT NULL REFERENCES audits(id),
    benchmark TEXT NOT NULL,
    phase TEXT NOT NULL,
    probe_label TEXT NOT NULL,
    auc REAL,
    p_value REAL,
    signal REAL,
    error TEXT
);
CREATE INDEX IF NOT EXISTS ix_audits_model ON audits(model_id);
"""


def _conn() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(PATHS.db) or ".", exist_ok=True)
    c = sqlite3.connect(PATHS.db)
    c.executescript(_SCHEMA)
    return c


def audited_model_ids() -> set[str]:
    with _conn() as c:
        rows = c.execute(
            "SELECT DISTINCT model_id FROM audits WHERE finished_at IS NOT NULL"
        ).fetchall()
    return {r[0] for r in rows}


def open_audit(model_id: str) -> int:
    with _conn() as c:
        cur = c.execute("INSERT INTO audits (model_id, started_at) VALUES (?, ?)",
                        (model_id, dt.datetime.now(dt.timezone.utc).isoformat()))
        return cur.lastrowid


def save_cell(audit_id: int, cell) -> None:
    def clean(x):
        return None if (isinstance(x, float) and math.isnan(x)) else x
    with _conn() as c:
        c.execute(
            "INSERT INTO cells VALUES (?,?,?,?,?,?,?,?)",
            (audit_id, cell.benchmark, cell.phase, cell.probe_label,
             clean(cell.auc), clean(cell.p_value), cell.signal, cell.error))


def close_audit(audit_id: int, score: int, tier: str, verdict: str | None) -> None:
    with _conn() as c:
        c.execute(
            "UPDATE audits SET finished_at=?, score=?, tier=?, verdict=? WHERE id=?",
            (dt.datetime.now(dt.timezone.utc).isoformat(), score, tier,
             verdict, audit_id))


def export_results_json() -> dict:
    """Latest finished audit per model -> results.json for the site."""
    with _conn() as c:
        c.row_factory = sqlite3.Row
        audits = c.execute("""
            SELECT a.* FROM audits a
            JOIN (SELECT model_id, MAX(finished_at) mx FROM audits
                  WHERE finished_at IS NOT NULL GROUP BY model_id) last
              ON a.model_id = last.model_id AND a.finished_at = last.mx
            ORDER BY a.score DESC, a.model_id ASC
        """).fetchall()
        models = []
        for a in audits:
            cells = c.execute(
                "SELECT * FROM cells WHERE audit_id=? ORDER BY benchmark, phase",
                (a["id"],)).fetchall()
            models.append({
                "model_id": a["model_id"],
                "audited_at": a["finished_at"],
                "score": a["score"],
                "tier": a["tier"],
                "verdict": a["verdict"],
                "cells": [dict(cl) for cl in cells],
            })
    payload = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "models": models,
    }
    os.makedirs(os.path.dirname(PATHS.results_json) or ".", exist_ok=True)
    with open(PATHS.results_json, "w") as f:
        json.dump(payload, f, indent=1)
    return payload
