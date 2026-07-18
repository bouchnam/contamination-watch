"""Central configuration for Contamination Watch.

Everything tunable lives here: which benchmarks we probe, how we discover
new models on the Hugging Face Hub, and how raw detector output maps to a
public contamination score.
"""
from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Benchmarks under audit
# ---------------------------------------------------------------------------
# Names must be resolvable by ``benchleak.loading.resolve_spec`` (known names
# like "gsm8k") or be a Hub dataset path with explicit fields.

@dataclass(frozen=True)
class BenchmarkTarget:
    name: str                       # benchleak benchmark name or Hub path
    config: str | None = None
    split: str | None = None
    fields: tuple[str, ...] | None = None
    limit: int = 150                # samples per probe — keeps CI runs bounded


BENCHMARKS: tuple[BenchmarkTarget, ...] = (
    BenchmarkTarget("gsm8k"),
    BenchmarkTarget("math"),
    BenchmarkTarget("truthfulqa"),
    BenchmarkTarget("mmlu", limit=200),
    BenchmarkTarget("humaneval", limit=120),
)


# ---------------------------------------------------------------------------
# Model discovery (Hugging Face Hub)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WatchFilter:
    """Which newly released models are worth auditing automatically."""

    max_params_b: float = 9.0          # CI runners top out around 7–8B in bf16
    min_downloads: int = 200           # ignore forks / noise
    pipeline_tags: tuple[str, ...] = ("text-generation",)
    lookback_days: int = 14            # how far back a "new release" reaches
    max_new_per_sweep: int = 4         # audit budget per scheduled run
    denylist_substrings: tuple[str, ...] = ("gguf", "awq", "gptq", "4bit",
                                            "8bit", "exl2", "mlx", "onnx")
    # Always-audit list: flagship releases we track regardless of filters.
    priority_models: tuple[str, ...] = (
        "Qwen/Qwen2.5-7B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.3",
    )


WATCH = WatchFilter()


# ---------------------------------------------------------------------------
# Probes (maps to benchleak detector phases)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProbeSpec:
    phase: str          # "pretrain" | "sft" | "rl"
    label: str          # human-readable
    weight: float       # contribution to the composite score
    enabled: bool = True


PROBES: tuple[ProbeSpec, ...] = (
    ProbeSpec("pretrain", "Min-K% Prob", weight=1.0),
    ProbeSpec("sft", "Prob. Variation (SPV-MIA)", weight=1.0),
    # Self-critique needs generation → expensive; run on priority models only.
    ProbeSpec("rl", "Self-Critique Entropy", weight=0.8, enabled=True),
)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScoringConfig:
    """Maps (AUC, p-value) pairs into a 0–100 contamination score.

    AUC 0.5 = indistinguishable from reference (clean); 1.0 = benchmark
    members perfectly separable (memorised). Signal below the noise floor is
    zeroed; statistical significance gates how much of the signal counts.
    """

    auc_noise_floor: float = 0.55      # below this, treat as chance
    auc_ceiling: float = 0.85          # at/above this, full signal
    p_full_credit: float = 0.01        # p ≤ this → weight 1.0
    p_zero_credit: float = 0.25        # p ≥ this → weight 0.0
    # Composite blends the worst benchmark with the mean across benchmarks,
    # so one smoking gun dominates but breadth still matters.
    max_mean_blend: float = 0.65       # 0.65 * worst + 0.35 * mean

    # Public tiers
    tier_clean_max: int = 20
    tier_suspect_max: int = 50         # above → "likely contaminated"


SCORING = ScoringConfig()

TIERS = {
    "clean": "Clean",
    "suspect": "Suspect",
    "contaminated": "Likely contaminated",
}


@dataclass
class Paths:
    db: str = "data/watch.sqlite3"
    results_json: str = "site/results.json"
    site_dir: str = "site"


PATHS = Paths()
