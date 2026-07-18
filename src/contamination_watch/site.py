"""Static leaderboard renderer: results.json -> site/index.html.

Pure function of the results payload — no client-side framework, no build
step. Deployable as-is to GitHub Pages.

Visual identity ("instrument, not dashboard"):
- Cleanroom palette: cold white ground, ink text, amber = signal,
  red = strong signal, teal = clean.
- Space Grotesk display / IBM Plex Sans body / IBM Plex Mono for every number.
- Signature element: each model's "spectrum strip" — one tick per
  (benchmark x probe) cell, positioned along the 0.50 -> 1.00 AUC axis and
  weighted by statistical significance, read like a spectrometer trace.
"""
from __future__ import annotations

import html
import json
import math
import os

from .config import PATHS, SCORING

TIER_META = {
    "clean": ("Clean", "teal"),
    "suspect": ("Suspect", "amber"),
    "contaminated": ("Likely contaminated", "red"),
}


def _fmt(x, digits=3):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:.{digits}f}"


def _tick_class(cell) -> str:
    p, auc = cell.get("p_value"), cell.get("auc")
    if auc is None or p is None:
        return "t-na"
    if auc >= SCORING.auc_noise_floor and p <= 0.01:
        return "t-hot"
    if auc >= SCORING.auc_noise_floor and p <= 0.1:
        return "t-warm"
    return "t-cold"


def _spectrum(cells) -> str:
    ticks = []
    for c in cells:
        auc = c.get("auc")
        if auc is None:
            continue
        x = max(0.0, min(1.0, (auc - 0.5) / 0.5)) * 100
        title = (f"{c['benchmark']} · {c['probe_label']} · "
                 f"AUC {_fmt(auc)} · p {_fmt(c.get('p_value'), 4)}")
        ticks.append(f'<i class="tick {_tick_class(c)}" style="left:{x:.1f}%" '
                     f'title="{html.escape(title)}"></i>')
    return (f'<div class="spectrum" aria-hidden="true">'
            f'<i class="axis-mark" style="left:0"></i>'
            f'<i class="axis-mark" style="left:50%"></i>'
            f'<i class="axis-mark" style="left:100%"></i>{"".join(ticks)}</div>')


def _grid_table(cells) -> str:
    rows = []
    for c in sorted(cells, key=lambda c: (c["benchmark"], c["phase"])):
        if c.get("error"):
            val = f'<td colspan="3" class="err">probe failed: {html.escape(c["error"][:80])}</td>'
        else:
            hot = _tick_class(c)
            val = (f'<td class="mono {hot}">{_fmt(c.get("auc"))}</td>'
                   f'<td class="mono">{_fmt(c.get("p_value"), 4)}</td>'
                   f'<td class="mono">{_fmt(c.get("signal"), 2)}</td>')
        rows.append(f'<tr><td>{html.escape(c["benchmark"])}</td>'
                    f'<td>{html.escape(c["probe_label"])}</td>{val}</tr>')
    return ('<table class="grid"><thead><tr><th>Benchmark</th><th>Probe</th>'
            '<th>AUC</th><th>p-value</th><th>Signal</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>')


def _model_row(rank: int, m: dict) -> str:
    tier_label, tier_color = TIER_META.get(m["tier"], ("—", "slate"))
    audited = (m.get("audited_at") or "")[:10]
    return f"""
<details class="row">
  <summary>
    <span class="rank mono">{rank:02d}</span>
    <span class="model">
      <span class="model-id">{html.escape(m["model_id"])}</span>
      <span class="audited">audited {audited}</span>
    </span>
    {_spectrum(m["cells"])}
    <span class="score-block">
      <span class="score mono c-{tier_color}">{m["score"]}</span>
      <span class="tier c-{tier_color}">{tier_label}</span>
    </span>
  </summary>
  <div class="evidence">
    <p class="verdict">{html.escape(m.get("verdict") or "")}</p>
    {_grid_table(m["cells"])}
  </div>
</details>"""


def render(payload: dict | None = None) -> str:
    if payload is None:
        with open(PATHS.results_json) as f:
            payload = json.load(f)
    models = payload["models"]
    n_probes = len({(c["probe_label"]) for m in models for c in m["cells"]}) if models else 0
    n_bench = len({c["benchmark"] for m in models for c in m["cells"]}) if models else 0
    status = (f'LAST SWEEP {payload["generated_at"][:16].replace("T", " ")} UTC'
              f' · {len(models)} MODELS · {n_probes} PROBES · {n_bench} BENCHMARKS')
    rows = "\n".join(_model_row(i + 1, m) for i, m in enumerate(models))

    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Contamination Watch — benchmark contamination leaderboard</title>
<meta name="description" content="Continuous membership-inference audits of new LLM releases against public benchmarks. Powered by benchleak.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root {{
  --ground:#F2F5F7; --panel:#FCFDFE; --ink:#131B26; --slate:#5C6B7A;
  --line:#D8E0E6; --teal:#178E7B; --amber:#D98D1F; --red:#C3402E;
  --mono:'IBM Plex Mono',monospace;
}}
* {{ box-sizing:border-box; margin:0; }}
body {{ background:var(--ground); color:var(--ink);
  font:16px/1.55 'IBM Plex Sans',system-ui,sans-serif; padding:0 20px 80px; }}
.wrap {{ max-width:1060px; margin:0 auto; }}
.mono {{ font-family:var(--mono); font-variant-numeric:tabular-nums; }}
.c-teal {{ color:var(--teal); }} .c-amber {{ color:var(--amber); }}
.c-red {{ color:var(--red); }} .c-slate {{ color:var(--slate); }}

header {{ padding:52px 0 12px; }}
.statusline {{ font-family:var(--mono); font-size:12.5px; letter-spacing:.08em;
  color:var(--slate); border-bottom:1px solid var(--line); padding-bottom:14px; }}
.statusline b {{ color:var(--teal); font-weight:500; }}
h1 {{ font-family:'Space Grotesk'; font-weight:700; font-size:clamp(34px,5.5vw,54px);
  letter-spacing:-.02em; margin:22px 0 8px; }}
h1 .u {{ color:var(--amber); }}
.sub {{ color:var(--slate); max-width:62ch; margin-bottom:8px; }}
.sub a {{ color:var(--teal); }}

.legend {{ display:flex; gap:22px; flex-wrap:wrap; font-family:var(--mono);
  font-size:12px; color:var(--slate); margin:26px 0 10px; }}
.legend i {{ display:inline-block; width:3px; height:12px; margin-right:7px;
  vertical-align:-2px; }}
.axis-label {{ margin-left:auto; }}

.colhead {{ display:grid; grid-template-columns:44px minmax(220px,1.1fr) minmax(180px,1fr) 128px;
  gap:18px; padding:8px 16px 6px; font-family:var(--mono); font-size:11px;
  letter-spacing:.1em; color:var(--slate); text-transform:uppercase; }}
.colhead :nth-child(4) {{ text-align:right; }}

details.row {{ background:var(--panel); border:1px solid var(--line);
  border-radius:8px; margin-bottom:10px; }}
details.row[open] {{ border-color:var(--slate); }}
summary {{ display:grid; grid-template-columns:44px minmax(220px,1.1fr) minmax(180px,1fr) 128px;
  gap:18px; align-items:center; padding:14px 16px; cursor:pointer; list-style:none; }}
summary::-webkit-details-marker {{ display:none; }}
summary:focus-visible {{ outline:2px solid var(--teal); outline-offset:2px; border-radius:8px; }}
.rank {{ color:var(--slate); font-size:13px; }}
.model-id {{ display:block; font-weight:600; font-size:15px; overflow-wrap:anywhere; }}
.audited {{ display:block; font-family:var(--mono); font-size:11.5px; color:var(--slate); }}

.spectrum {{ position:relative; height:34px; background:
  linear-gradient(to right, transparent calc(50% - .5px), var(--line) calc(50% - .5px),
  var(--line) calc(50% + .5px), transparent calc(50% + .5px)),
  linear-gradient(var(--ground),var(--ground)); border:1px solid var(--line);
  border-radius:4px; overflow:hidden; }}
.spectrum .tick {{ position:absolute; top:6px; bottom:6px; width:3px;
  border-radius:1px; transform:translateX(-1px); }}
.spectrum .axis-mark {{ position:absolute; bottom:0; width:1px; height:5px;
  background:var(--slate); opacity:.5; }}
.t-hot {{ background:var(--red); }} .t-warm {{ background:var(--amber); }}
.t-cold {{ background:var(--teal); opacity:.55; }} .t-na {{ background:var(--line); }}
td.t-hot {{ color:var(--red); font-weight:500; }} td.t-warm {{ color:var(--amber); }}
td.t-cold {{ color:inherit; }}

.score-block {{ text-align:right; }}
.score {{ font-size:26px; font-weight:500; display:block; line-height:1.1; }}
.tier {{ font-family:var(--mono); font-size:11px; letter-spacing:.06em; }}

.evidence {{ border-top:1px solid var(--line); padding:16px 18px 20px; }}
.verdict {{ font-size:14.5px; max-width:78ch; margin-bottom:14px; }}
table.grid {{ width:100%; border-collapse:collapse; font-size:13.5px; }}
table.grid th {{ text-align:left; font-family:var(--mono); font-size:11px;
  letter-spacing:.08em; text-transform:uppercase; color:var(--slate);
  border-bottom:1px solid var(--line); padding:6px 10px; }}
table.grid td {{ padding:6px 10px; border-bottom:1px solid var(--line); }}
table.grid td.err {{ color:var(--slate); font-style:italic; }}

footer {{ margin-top:44px; font-size:13.5px; color:var(--slate); max-width:74ch; }}
footer a {{ color:var(--teal); }}
@media (max-width:760px) {{
  summary, .colhead {{ grid-template-columns:30px 1fr 86px; }}
  .spectrum, .colhead :nth-child(3) {{ display:none; }}
  .score {{ font-size:21px; }}
}}
@media (prefers-reduced-motion:no-preference) {{
  details.row {{ transition:border-color .15s ease; }}
}}
</style>
</head>
<body>
<div class="wrap">
<header>
  <div class="statusline"><b>●</b> {status}</div>
  <h1>Contamination<span class="u">Watch</span></h1>
  <p class="sub">Continuous membership-inference audits of newly released open
  models against the benchmarks they're scored on. Three probes per model —
  Min-K% Prob (pre-training), SPV-MIA probabilistic variation (fine-tuning),
  and self-critique entropy collapse (RL) — powered by
  <a href="https://pypi.org/project/benchleak/">benchleak</a>.</p>
</header>

<div class="legend">
  <span><i class="t-hot"></i>significant (p ≤ .01)</span>
  <span><i class="t-warm"></i>weak (p ≤ .10)</span>
  <span><i class="t-cold"></i>no signal</span>
  <span class="axis-label">spectrum axis: AUC 0.50 → 0.75 → 1.00</span>
</div>

<div class="colhead"><span>#</span><span>Model</span><span>Membership spectrum</span><span>Score</span></div>
{rows}

<footer>
  <p><strong>Reading the score.</strong> 0–{SCORING.tier_clean_max} clean ·
  {SCORING.tier_clean_max + 1}–{SCORING.tier_suspect_max} suspect ·
  {SCORING.tier_suspect_max + 1}+ likely contaminated. AUC measures how
  separable benchmark items are from unseen reference text under each probe;
  0.5 is chance. A high score is evidence of training-set membership, not
  proof of intent — deduplication failures and licensing quirks produce the
  same fingerprint. Full methodology and raw results:
  <a href="results.json">results.json</a>.</p>
</footer>
</div>
</body>
</html>"""
    return page


def build(payload: dict | None = None) -> str:
    os.makedirs(PATHS.site_dir, exist_ok=True)
    out = os.path.join(PATHS.site_dir, "index.html")
    with open(out, "w") as f:
        f.write(render(payload))
    return out
