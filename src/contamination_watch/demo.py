"""Demo sweep: exercises the full pipeline with synthetic probe statistics.

Used for local preview and CI smoke tests — everything downstream of the
runner (scoring, storage, verdicts, site) runs exactly as in production.
Synthetic AUC/p pairs are drawn from three archetypes: clean, single-benchmark
memoriser, and diffusely contaminated.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from benchleak.core import ContaminationResult

from .config import BENCHMARKS, PROBES
from .runner import ProbeOutcome

_ARCHETYPES = {
    # model_id: (profile, {benchmark: severity 0..1})
    "demo-org/aurora-7b-instruct": ("hot", {"gsm8k": 0.95, "math": 0.8}),
    "demo-org/quill-3b": ("warm", {"truthfulqa": 0.5, "mmlu": 0.35}),
    "demo-org/basalt-8b-chat": ("hot", {"humaneval": 0.9}),
    "demo-org/lumen-1.5b": ("clean", {}),
    "demo-org/cascade-7b-v2": ("warm", {"gsm8k": 0.4}),
    "demo-org/verity-4b-base": ("clean", {}),
}


def _draw(severity: float, rng: random.Random) -> tuple[float, float]:
    """(auc, p_value) for a given contamination severity."""
    if severity <= 0:
        auc = rng.gauss(0.51, 0.03)
        p = rng.uniform(0.15, 0.9)
    else:
        auc = 0.55 + 0.4 * severity + rng.gauss(0, 0.02)
        p = max(1e-5, rng.uniform(0.0001, 0.02) * (1.2 - severity))
    return min(max(auc, 0.35), 0.99), min(p, 0.99)


@dataclass
class _FakeResult:
    auc: float
    p_value: float


def demo_outcomes(model_id: str, seed: int = 0) -> list[ProbeOutcome]:
    rng = random.Random(hash(model_id) ^ seed)
    _, hotspots = _ARCHETYPES.get(model_id, ("clean", {}))
    outs: list[ProbeOutcome] = []
    for probe in PROBES:
        if not probe.enabled:
            continue
        for bench in BENCHMARKS:
            sev = hotspots.get(bench.name, 0.0)
            # SFT/RL probes see partial signal on pretrain-phase leaks and v.v.
            sev *= rng.uniform(0.6, 1.0)
            auc, p = _draw(sev, rng)
            result = ContaminationResult(
                detector=probe.label, benchmark=bench.name,
                benchmark_scores=[], reference_scores=[],
                auc=auc, p_value=p)
            outs.append(ProbeOutcome(model_id, bench.name, probe.phase,
                                     probe.label, result))
    return outs


def demo_model_ids() -> list[str]:
    return list(_ARCHETYPES)
