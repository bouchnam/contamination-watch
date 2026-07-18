# Contamination Watch

Continuous benchmark-contamination auditing of newly released LLMs, published as a live public leaderboard. The natural sequel to [benchleak](https://pypi.org/project/benchleak/): instead of auditing one model on demand, an agent watches the Hugging Face Hub, probes every notable new release against the benchmarks it will be scored on, and publishes a contamination score per model.

## How it works

```
                 every 6h (GitHub Actions cron)
                              │
   ┌──────────┐   queue   ┌────────┐  AUC, p   ┌────────┐
   │ watcher  │──────────▶│ runner │──────────▶│ scorer │
   │ HF Hub   │           │benchleak│          │ 0–100  │
   └──────────┘           └────────┘           └───┬────┘
     new releases      3 probes × 5 benchmarks     │
     + priority list   Min-K% · SPV-MIA ·          ▼
                       Self-Critique         ┌──────────┐   ┌──────────────┐
                                             │ analyst  │──▶│ store + site │──▶ GitHub Pages
                                             │ (Claude) │   │ sqlite/json  │
                                             └──────────┘   └──────────────┘
```

- **watcher** — polls the Hub for text-generation models released in the last 14 days, filters out quants/forks/noise, caps parameter count to what the runner can handle, merges with an always-audit priority list, skips anything already audited.
- **runner** — for each model, runs three benchleak detectors across GSM8K, MATH, TruthfulQA, MMLU and HumanEval: **Min-K% Prob** (pre-training membership), **Probabilistic Variation / SPV-MIA** (fine-tuning membership), **Self-Critique entropy collapse** (RL-phase memorisation). Yields results incrementally so a crash never loses a sweep.
- **scorer** — per cell: AUC above a 0.55 noise floor, gated by a log-linear significance ramp (full credit at p≤0.01, zero at p≥0.25). Composite = 0.65 × worst benchmark + 0.35 × mean, so one smoking gun dominates but diffuse signal still registers. Tiers: ≤20 clean, 21–50 suspect, 51+ likely contaminated.
- **analyst** — optional Claude call that writes a two-sentence evidence-grounded verdict per model (deterministic template fallback if no API key).
- **site** — a static leaderboard rendered from `results.json`. Each model gets a **membership spectrum**: one tick per (benchmark × probe) cell on the AUC 0.5→1.0 axis, coloured by significance — the whole evidence grid readable at a glance.

## Quickstart

```bash
pip install -e .
contamination-watch demo               # synthetic end-to-end run, opens site/index.html
contamination-watch audit Qwen/Qwen2.5-0.5B   # real audit (needs torch/transformers, HF access)
contamination-watch sweep              # discover + audit new releases
contamination-watch build              # re-render site from stored results
```

## Deploying the live service (free)

1. Push this repo to GitHub, enable **Pages** (Settings → Pages → Source: GitHub Actions).
2. Add secrets: `HF_TOKEN` (gated models), `ANTHROPIC_API_KEY` (optional, for verdicts).
3. Done. `.github/workflows/watch.yml` sweeps every 6 hours, commits results, and deploys the leaderboard to Pages. Trigger a one-off audit of any model from the Actions tab (`workflow_dispatch` → model id).

**Two compute lanes, both free:**
- **CPU lane** (`watch.yml`) — GitHub Actions, models ≤3B, every 6h.
- **GPU lane** (`kaggle-gpu.yml`) — GitHub Actions conducts Kaggle's free T4 (30 GPU-h/week, no card): pushes `infra/kaggle/` as a private kernel, polls, retrieves results, commits. Covers 7–8B flagships, 2 per session, twice daily. Needs `KAGGLE_USERNAME` + `KAGGLE_KEY` repo secrets (kaggle.com → Settings → API → Create New Token). Optional: attach an `HF_TOKEN` secret to the kernel on Kaggle (Add-ons → Secrets) for gated models like Llama.

An alternative Modal lane (`infra/modal_sweep.py`, ~25 GPU-h/month, requires a card on file) is included but optional. All limits live in `config.py`; per-lane budgets via `CW_MAX_PARAMS_B` / `CW_MAX_NEW` env vars.

## Honest caveats

- A high score is evidence of **training-set membership, not intent** — sloppy dedup and benchmark items leaking into web scrapes produce the same fingerprint as deliberate gaming. The site says so.
- Membership inference on very large, heavily-aligned models is harder; expect sensitivity to drop with scale. Report AUC and p-values, never just the composite.
- The self-critique probe requires generation and is ~10× the cost of the logprob probes; disable it in `config.py` for cheap sweeps.
- Choose reference sets carefully (benchleak ships domain-matched defaults); a mismatched reference distribution inflates AUC for innocent reasons.

## Project layout

```
src/contamination_watch/
  config.py    benchmarks, filters, probes, scoring thresholds — all knobs
  watcher.py   HF Hub release discovery
  runner.py    benchleak probe execution
  scorer.py    AUC/p → contamination score
  analyst.py   Claude verdicts (optional)
  store.py     SQLite history + results.json export
  site.py      static leaderboard renderer
  demo.py      synthetic sweep for preview/CI
  cli.py       sweep · audit · demo · build
.github/workflows/watch.yml   the live service
```

MIT. Built on benchleak by Mouad.
