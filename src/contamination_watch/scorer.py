"""Scoring: raw detector statistics -> public contamination score.

Design goals:
- One clear number per model (0–100), but never hide the evidence grid.
- A single strong, significant signal on one benchmark should dominate
  (memorising GSM8K is damning even if MMLU looks clean), while diffuse
  weak signal across many benchmarks should still register.
- Non-significant AUC lift is discounted smoothly, not cliff-edged.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .config import PROBES, SCORING


@dataclass(frozen=True)
class CellScore:
    """One (benchmark, probe) cell of the evidence grid."""
    benchmark: str
    phase: str
    probe_label: str
    auc: float
    p_value: float
    signal: float          # 0–1 after noise floor + significance gating
    error: str | None = None


def _significance_weight(p: float) -> float:
    """Log-linear ramp: full credit at p<=0.01, none at p>=0.25."""
    lo, hi = SCORING.p_full_credit, SCORING.p_zero_credit
    if p <= lo:
        return 1.0
    if p >= hi:
        return 0.0
    return (math.log(hi) - math.log(p)) / (math.log(hi) - math.log(lo))


def cell_signal(auc: float, p_value: float) -> float:
    span = SCORING.auc_ceiling - SCORING.auc_noise_floor
    raw = max(0.0, min(1.0, (auc - SCORING.auc_noise_floor) / span))
    return raw * _significance_weight(p_value)


def score_cells(cells: list[CellScore]) -> tuple[int, str, dict[str, float]]:
    """Composite score + tier + per-benchmark rollup.

    Per benchmark: probe-weighted mean of cell signals.
    Composite: blend of worst benchmark and mean benchmark, per config.
    """
    weights = {p.label: p.weight for p in PROBES}
    per_bench: dict[str, float] = {}
    by_bench: dict[str, list[CellScore]] = {}
    for c in cells:
        if c.error is None:
            by_bench.setdefault(c.benchmark, []).append(c)

    for bench, bcells in by_bench.items():
        wsum = sum(weights.get(c.probe_label, 1.0) for c in bcells)
        per_bench[bench] = sum(
            c.signal * weights.get(c.probe_label, 1.0) for c in bcells
        ) / max(wsum, 1e-9)

    if not per_bench:
        return 0, "clean", {}

    worst = max(per_bench.values())
    mean = sum(per_bench.values()) / len(per_bench)
    b = SCORING.max_mean_blend
    composite = b * worst + (1 - b) * mean
    score = round(100 * composite)

    if score <= SCORING.tier_clean_max:
        tier = "clean"
    elif score <= SCORING.tier_suspect_max:
        tier = "suspect"
    else:
        tier = "contaminated"
    return score, tier, per_bench


def build_cells(outcomes) -> list[CellScore]:
    """Adapter: runner.ProbeOutcome (or stored rows) -> CellScore list."""
    cells: list[CellScore] = []
    for o in outcomes:
        if o.result is None:
            cells.append(CellScore(o.benchmark, o.phase, o.probe_label,
                                   auc=float("nan"), p_value=float("nan"),
                                   signal=0.0, error=o.error))
        else:
            r = o.result
            cells.append(CellScore(o.benchmark, o.phase, o.probe_label,
                                   auc=r.auc, p_value=r.p_value,
                                   signal=cell_signal(r.auc, r.p_value)))
    return cells
