"""Model-release watcher.

Polls the Hugging Face Hub for recently published text-generation models
that pass the audit filters, and merges them with the always-audit priority
list. Returns a bounded queue of model ids for this sweep.
"""
from __future__ import annotations

import datetime as dt
import logging
import re

from .config import WATCH, WatchFilter
from . import store

log = logging.getLogger("watch.watcher")

_PARAM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[bB]\b")


def _estimate_params_b(model_id: str, safetensors_params: int | None) -> float | None:
    """Best-effort parameter count in billions."""
    if safetensors_params:
        return safetensors_params / 1e9
    m = _PARAM_RE.search(model_id)
    return float(m.group(1)) if m else None


def discover(token: str | None = None, filt: WatchFilter = WATCH) -> list[str]:
    """Return model ids to audit this sweep (new + priority, minus done)."""
    from huggingface_hub import HfApi  # imported lazily: not needed in demo mode

    api = HfApi(token=token)
    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=filt.lookback_days)

    candidates: list[tuple[str, int]] = []
    for tag in filt.pipeline_tags:
        for m in api.list_models(pipeline_tag=tag, sort="createdAt",
                                 direction=-1, limit=300,
                                 expand=["downloads", "createdAt", "safetensors"]):
            created = getattr(m, "created_at", None)
            if created and created < cutoff:
                break
            mid = m.id
            if any(s in mid.lower() for s in filt.denylist_substrings):
                continue
            downloads = getattr(m, "downloads", 0) or 0
            if downloads < filt.min_downloads:
                continue
            st = getattr(m, "safetensors", None)
            params = _estimate_params_b(mid, getattr(st, "total", None) if st else None)
            if params is not None and params > filt.max_params_b:
                continue
            candidates.append((mid, downloads))

    # Most-downloaded new releases first; they're the ones people benchmark.
    candidates.sort(key=lambda t: -t[1])
    fresh = [mid for mid, _ in candidates]

    already = store.audited_model_ids()
    queue: list[str] = [m for m in filt.priority_models if m not in already]
    for mid in fresh:
        if len(queue) >= filt.max_new_per_sweep:
            break
        if mid not in already and mid not in queue:
            queue.append(mid)

    log.info("sweep queue: %s", queue)
    return queue
