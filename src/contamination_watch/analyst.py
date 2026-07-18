"""Analyst: turns an evidence grid into a two-sentence human verdict.

Optional — requires ANTHROPIC_API_KEY. If absent, a deterministic template
verdict is produced instead, so the pipeline never blocks on the API.
"""
from __future__ import annotations

import json
import os

_SYSTEM = (
    "You are the analyst for an LLM benchmark-contamination auditing service. "
    "You receive membership-inference statistics (AUC vs a reference "
    "distribution, with permutation p-values) from three probes: Min-K% Prob "
    "(pre-training membership), Probabilistic Variation / SPV-MIA "
    "(fine-tuning membership), and Self-Critique entropy collapse (RL-phase "
    "memorisation). Write a verdict of AT MOST two sentences: state which "
    "benchmarks show significant signal, which training phase it points to, "
    "and how confident the evidence is. Be precise and non-alarmist: AUC "
    "near 0.5 or non-significant p-values mean no evidence, not innocence. "
    "Never speculate beyond the numbers. Plain text only."
)


def _template_verdict(score: int, tier: str, per_bench: dict[str, float]) -> str:
    if not per_bench or score <= 20:
        return ("No statistically significant membership signal on any audited "
                "benchmark; consistent with a clean model at current probe "
                "sensitivity.")
    flagged = sorted(per_bench.items(), key=lambda kv: -kv[1])
    tops = ", ".join(b for b, s in flagged if s > 0.2) or flagged[0][0]
    strength = "strong" if tier == "contaminated" else "moderate"
    return (f"{strength.capitalize()} membership signal detected on {tops}; "
            f"composite contamination score {score}/100.")


def write_verdict(model_id: str, score: int, tier: str,
                  per_bench: dict[str, float], cells: list) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return _template_verdict(score, tier, per_bench)
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        grid = [
            {"benchmark": c.benchmark, "probe": c.probe_label,
             "auc": round(c.auc, 3) if c.auc == c.auc else None,
             "p": round(c.p_value, 4) if c.p_value == c.p_value else None,
             "error": c.error}
            for c in cells
        ]
        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=300,
            system=_SYSTEM,
            messages=[{"role": "user", "content": json.dumps(
                {"model": model_id, "score": score, "tier": tier,
                 "evidence_grid": grid})}],
        )
        text = "".join(b.text for b in msg.content if b.type == "text").strip()
        return text or _template_verdict(score, tier, per_bench)
    except Exception:
        return _template_verdict(score, tier, per_bench)
