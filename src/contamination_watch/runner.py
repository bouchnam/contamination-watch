"""Probe runner: drives benchleak's Python API for one model.

For each benchmark target and each enabled probe phase, this loads the
benchmark texts and a domain-matched reference set, runs the corresponding
benchleak detector, and yields ``ContaminationResult`` objects.
"""
from __future__ import annotations

import gc
import logging
from collections.abc import Iterator
from dataclasses import dataclass

from benchleak.core import ContaminationResult, scan
from benchleak.data import load_reference
from benchleak.loading import load_benchmark, load_model, resolve_spec

from .config import BENCHMARKS, PROBES, BenchmarkTarget, ProbeSpec

log = logging.getLogger("watch.runner")


@dataclass
class ProbeOutcome:
    model_id: str
    benchmark: str
    phase: str
    probe_label: str
    result: ContaminationResult | None
    error: str | None = None


def _build_detector(phase: str, model, tokenizer):
    if phase == "pretrain":
        from benchleak.detectors.pretrain import MinKProbDetector
        return MinKProbDetector(model, tokenizer, k=20.0)
    if phase == "sft":
        from benchleak.detectors.perturb import WordSwapPerturber
        from benchleak.detectors.sft import ProbVariationDetector
        # WordSwap over T5 mask-fill: ~10x cheaper on CI, minor AUC cost.
        return ProbVariationDetector(model, tokenizer,
                                     perturber=WordSwapPerturber(fraction=0.2),
                                     n_perturbations=6)
    if phase == "rl":
        from benchleak.detectors.rl import SelfCritiqueDetector
        return SelfCritiqueDetector(model, tokenizer, max_new_tokens=192)
    raise ValueError(f"unknown probe phase: {phase}")


def run_model(model_id: str, *, hf_token: str | None = None,
              device: str | None = None, dtype: str = "auto",
              benchmarks: tuple[BenchmarkTarget, ...] = BENCHMARKS,
              probes: tuple[ProbeSpec, ...] = PROBES) -> Iterator[ProbeOutcome]:
    """Run all enabled probes for one model, yielding outcomes as they finish.

    Yields (rather than returns) so the caller can persist partial progress:
    a 7B model x 5 benchmarks x 3 probes is a long run, and a crash halfway
    should not lose everything.
    """
    log.info("loading model %s", model_id)
    model, tokenizer = load_model(model_id, device=device, dtype=dtype, token=hf_token)

    try:
        for probe in probes:
            if not probe.enabled:
                continue
            detector = _build_detector(probe.phase, model, tokenizer)
            for bench in benchmarks:
                try:
                    spec = resolve_spec(bench.name, config=bench.config,
                                        split=bench.split, fields=bench.fields)
                    texts = load_benchmark(spec, limit=bench.limit)
                    reference = load_reference(limit=bench.limit, domain=spec.domain)
                    result = scan(detector, texts, reference,
                                  detector_name=probe.label,
                                  benchmark_name=bench.name)
                    log.info("%s | %s | %s -> AUC=%.3f p=%.4f",
                             model_id, probe.label, bench.name,
                             result.auc, result.p_value)
                    yield ProbeOutcome(model_id, bench.name, probe.phase,
                                       probe.label, result)
                except Exception as exc:  # keep sweeping; report per-cell error
                    log.exception("probe failed: %s/%s/%s", model_id,
                                  probe.phase, bench.name)
                    yield ProbeOutcome(model_id, bench.name, probe.phase,
                                       probe.label, None, error=str(exc))
    finally:
        del model, tokenizer
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
