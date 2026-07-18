"""Contamination Watch CLI.

  contamination-watch sweep            # discover new models + audit + publish
  contamination-watch audit MODEL_ID   # audit one model on demand
  contamination-watch demo             # synthetic sweep (no GPU / HF needed)
  contamination-watch build            # rebuild the site from stored results
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

from . import analyst, scorer, site, store

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("watch")


def _audit_outcomes(model_id: str, outcomes) -> None:
    """Persist one model's audit: cells -> score -> verdict -> close."""
    audit_id = store.open_audit(model_id)
    cells = scorer.build_cells(list(outcomes))
    for cell in cells:
        store.save_cell(audit_id, cell)
    score, tier, per_bench = scorer.score_cells(cells)
    verdict = analyst.write_verdict(model_id, score, tier, per_bench, cells)
    store.close_audit(audit_id, score, tier, verdict)
    log.info("audit closed: %s -> %d/100 (%s)", model_id, score, tier)


def cmd_audit(args) -> None:
    from .runner import run_model
    _audit_outcomes(args.model, run_model(
        args.model, hf_token=os.environ.get("HF_TOKEN"),
        device=args.device, dtype=args.dtype))
    _publish()


def cmd_sweep(args) -> None:
    from .watcher import discover
    from .runner import run_model
    queue = discover(token=os.environ.get("HF_TOKEN"))
    if not queue:
        log.info("nothing new to audit")
    for model_id in queue:
        try:
            _audit_outcomes(model_id, run_model(
                model_id, hf_token=os.environ.get("HF_TOKEN"),
                device=args.device, dtype=args.dtype))
        except Exception:
            log.exception("audit aborted for %s", model_id)
    _publish()


def cmd_demo(args) -> None:
    from .demo import demo_model_ids, demo_outcomes
    for model_id in demo_model_ids():
        _audit_outcomes(model_id, demo_outcomes(model_id, seed=args.seed))
    _publish()


def cmd_build(args) -> None:
    _publish()


def _publish() -> None:
    payload = store.export_results_json()
    out = site.build(payload)
    log.info("site built: %s (%d models)", out, len(payload["models"]))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="contamination-watch")
    sub = p.add_subparsers(required=True)

    s = sub.add_parser("sweep", help="discover + audit new releases")
    s.add_argument("--device", default=None)
    s.add_argument("--dtype", default="auto")
    s.set_defaults(fn=cmd_sweep)

    a = sub.add_parser("audit", help="audit one model id")
    a.add_argument("model")
    a.add_argument("--device", default=None)
    a.add_argument("--dtype", default="auto")
    a.set_defaults(fn=cmd_audit)

    d = sub.add_parser("demo", help="synthetic end-to-end run")
    d.add_argument("--seed", type=int, default=0)
    d.set_defaults(fn=cmd_demo)

    b = sub.add_parser("build", help="rebuild site from stored results")
    b.set_defaults(fn=cmd_build)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
