"""Static leaderboard renderer: results.json -> site/index.html.

Visual identity — "measurement instrument, not dashboard":
- Cleanroom palette: cold white ground, ink text, amber = weak signal,
  red = significant signal, teal = clean/system-ok.
- Space Grotesk display / IBM Plex Sans body / IBM Plex Mono for all numbers.
- Signature: the membership spectrum — one tick per (benchmark x probe)
  cell on the AUC 0.5 -> 1.0 axis; tick height encodes significance, so a
  contaminated model reads like a spectrometer trace lighting up.
- Counters strip under the hero reads like the instrument's front panel.
Pure function of results.json; no framework, one small vanilla-JS filter.
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

_FAVICON = ("data:image/svg+xml,"
            "%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E"
            "%3Crect width='32' height='32' rx='6' fill='%23131B26'/%3E"
            "%3Crect x='7' y='10' width='3' height='12' rx='1' fill='%23178E7B'/%3E"
            "%3Crect x='14' y='7' width='3' height='18' rx='1' fill='%23D98D1F'/%3E"
            "%3Crect x='21' y='4' width='3' height='24' rx='1' fill='%23C3402E'/%3E"
            "%3C/svg%3E")


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
        auc, p = c.get("auc"), c.get("p_value")
        if auc is None:
            continue
        x = max(0.0, min(1.0, (auc - 0.5) / 0.5)) * 100
        # Height encodes confidence: chance-level noise stays short,
        # significant signal reaches full height.
        if p is None:
            h = 30
        else:
            conf = max(0.0, min(1.0, 1 - (math.log10(max(p, 1e-6)) + 6) / 6))
            h = 26 + 62 * conf
        title = (f"{c['benchmark']} · {c['probe_label']} · "
                 f"AUC {_fmt(auc)} · p {_fmt(p, 4)}")
        ticks.append(
            f'<i class="tick {_tick_class(c)}" '
            f'style="left:{x:.1f}%;height:{h:.0f}%" '
            f'title="{html.escape(title)}"></i>')
    return ('<div class="spectrum" aria-hidden="true">'
            '<span class="zone"></span>'
            '<i class="axm" style="left:0"></i>'
            '<i class="axm" style="left:50%"></i>'
            '<i class="axm" style="left:100%"></i>'
            + "".join(ticks) + "</div>")


def _grid_table(cells) -> str:
    rows = []
    for c in sorted(cells, key=lambda c: (c["benchmark"], c["phase"])):
        if c.get("error"):
            val = ('<td colspan="3" class="err">probe failed: '
                   f'{html.escape(c["error"][:80])}</td>')
        else:
            val = (f'<td class="mono {_tick_class(c)}">{_fmt(c.get("auc"))}</td>'
                   f'<td class="mono">{_fmt(c.get("p_value"), 4)}</td>'
                   f'<td class="mono">{_fmt(c.get("signal"), 2)}</td>')
        rows.append(f'<tr><td>{html.escape(c["benchmark"])}</td>'
                    f'<td class="muted">{html.escape(c["probe_label"])}</td>{val}</tr>')
    return ('<table class="grid"><thead><tr><th>Benchmark</th><th>Probe</th>'
            '<th>AUC</th><th>p-value</th><th>Signal</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>')


def _model_row(rank: int, m: dict) -> str:
    tier_label, tier_color = TIER_META.get(m["tier"], ("—", "slate"))
    audited = (m.get("audited_at") or "")[:10]
    hub = f"https://huggingface.co/{m['model_id']}"
    return f"""
<details class="row" data-tier="{m['tier']}">
  <summary>
    <span class="rank mono">{rank:02d}</span>
    <span class="model">
      <span class="model-id">{html.escape(m["model_id"])}</span>
      <span class="audited mono">audited {audited}</span>
    </span>
    {_spectrum(m["cells"])}
    <span class="score-block">
      <span class="score mono c-{tier_color}">{m["score"]}</span>
      <span class="tier c-{tier_color}">{tier_label}</span>
    </span>
  </summary>
  <div class="evidence">
    <p class="verdict">{html.escape(m.get("verdict") or "")}
      <a class="hub" href="{hub}">model card ↗</a></p>
    {_grid_table(m["cells"])}
  </div>
</details>"""


def render(payload: dict | None = None) -> str:
    if payload is None:
        with open(PATHS.results_json) as f:
            payload = json.load(f)
    models = payload["models"]
    n_flag = sum(1 for m in models if m["tier"] == "contaminated")
    n_susp = sum(1 for m in models if m["tier"] == "suspect")
    n_clean = sum(1 for m in models if m["tier"] == "clean")
    n_probes = len({c["probe_label"] for m in models for c in m["cells"]}) if models else 0
    n_bench = len({c["benchmark"] for m in models for c in m["cells"]}) if models else 0
    n_cells = sum(len(m["cells"]) for m in models)
    swept = payload["generated_at"][:16].replace("T", " ")
    rows = "\n".join(_model_row(i + 1, m) for i, m in enumerate(models))

    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Contamination Watch — is your benchmark in their training set?</title>
<meta name="description" content="Continuous membership-inference audits of new LLM releases against the benchmarks they're scored on. Powered by benchleak.">
<link rel="icon" href="{_FAVICON}">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root {{
  --ground:#F3F6F8; --panel:#FDFEFE; --ink:#131B26; --slate:#5C6B7A;
  --faint:#8B99A6; --line:#DCE3E9; --line2:#C9D3DC;
  --teal:#178E7B; --amber:#D98D1F; --red:#C3402E;
  --teal-w:#E3F1EE; --amber-w:#F9EFDE; --red-w:#F8E7E3;
  --mono:'IBM Plex Mono',monospace;
}}
* {{ box-sizing:border-box; margin:0; }}
html {{ scroll-behavior:smooth; }}
body {{ background:var(--ground); color:var(--ink);
  font:16px/1.55 'IBM Plex Sans',system-ui,sans-serif; padding:0 20px 90px; }}
.wrap {{ max-width:1080px; margin:0 auto; }}
.mono {{ font-family:var(--mono); font-variant-numeric:tabular-nums; }}
.muted {{ color:var(--slate); }}
.c-teal {{ color:var(--teal); }} .c-amber {{ color:var(--amber); }}
.c-red {{ color:var(--red); }} .c-slate {{ color:var(--slate); }}
a {{ color:var(--teal); text-underline-offset:2px; }}

/* ---------- header ---------- */
header {{ padding:46px 0 0; }}
.statusline {{ display:flex; align-items:center; gap:10px;
  font-family:var(--mono); font-size:12.5px; letter-spacing:.08em;
  color:var(--slate); border-bottom:1px solid var(--line); padding-bottom:13px; }}
.dot {{ width:8px; height:8px; border-radius:50%; background:var(--teal);
  box-shadow:0 0 0 0 rgba(23,142,123,.45); }}
@media (prefers-reduced-motion:no-preference) {{
  .dot {{ animation:pulse 2.6s ease-out infinite; }}
  @keyframes pulse {{ 0% {{ box-shadow:0 0 0 0 rgba(23,142,123,.45); }}
    70% {{ box-shadow:0 0 0 9px rgba(23,142,123,0); }}
    100% {{ box-shadow:0 0 0 0 rgba(23,142,123,0); }} }}
}}
h1 {{ font-family:'Space Grotesk'; font-weight:700;
  font-size:clamp(38px,6vw,60px); letter-spacing:-.025em; margin:26px 0 10px; }}
h1 .u {{ color:var(--amber); }}
.sub {{ color:var(--slate); max-width:64ch; font-size:16.5px; }}
.sub b {{ color:var(--ink); font-weight:600; }}

/* ---------- counters strip ---------- */
.counters {{ display:grid; grid-template-columns:repeat(4,1fr);
  border:1px solid var(--line2); border-radius:10px; background:var(--panel);
  margin:30px 0 8px; overflow:hidden; }}
.counter {{ padding:14px 18px 12px; border-left:1px solid var(--line); }}
.counter:first-child {{ border-left:0; }}
.counter .v {{ font-family:var(--mono); font-size:26px; font-weight:500;
  line-height:1.15; display:block; }}
.counter .k {{ font-family:var(--mono); font-size:10.5px; letter-spacing:.12em;
  text-transform:uppercase; color:var(--faint); }}

/* ---------- controls ---------- */
.controls {{ display:flex; align-items:center; gap:8px; flex-wrap:wrap;
  margin:20px 0 12px; }}
.chip {{ font-family:var(--mono); font-size:12px; letter-spacing:.04em;
  border:1px solid var(--line2); background:var(--panel); color:var(--slate);
  border-radius:999px; padding:5px 13px; cursor:pointer; }}
.chip:hover {{ border-color:var(--slate); color:var(--ink); }}
.chip[aria-pressed="true"] {{ background:var(--ink); color:var(--ground);
  border-color:var(--ink); }}
.chip:focus-visible {{ outline:2px solid var(--teal); outline-offset:2px; }}
.axis-label {{ margin-left:auto; font-family:var(--mono); font-size:11.5px;
  color:var(--faint); }}
.legend {{ display:flex; gap:18px; font-family:var(--mono); font-size:11.5px;
  color:var(--slate); margin:0 0 14px; flex-wrap:wrap; }}
.legend i {{ display:inline-block; width:3px; height:12px; margin-right:6px;
  vertical-align:-2px; border-radius:1px; }}

/* ---------- rows ---------- */
.colhead {{ display:grid;
  grid-template-columns:44px minmax(230px,1.05fr) minmax(200px,1fr) 132px;
  gap:20px; padding:6px 18px; font-family:var(--mono); font-size:10.5px;
  letter-spacing:.12em; color:var(--faint); text-transform:uppercase; }}
.colhead :nth-child(4) {{ text-align:right; }}
details.row {{ background:var(--panel); border:1px solid var(--line);
  border-radius:10px; margin-bottom:10px;
  transition:border-color .15s ease, box-shadow .15s ease; }}
details.row:hover {{ border-color:var(--line2);
  box-shadow:0 1px 4px rgba(19,27,38,.06); }}
details.row[open] {{ border-color:var(--slate); }}
details.row.hidden {{ display:none; }}
summary {{ display:grid;
  grid-template-columns:44px minmax(230px,1.05fr) minmax(200px,1fr) 132px;
  gap:20px; align-items:center; padding:15px 18px; cursor:pointer;
  list-style:none; }}
summary::-webkit-details-marker {{ display:none; }}
summary:focus-visible {{ outline:2px solid var(--teal); outline-offset:2px;
  border-radius:10px; }}
.rank {{ color:var(--faint); font-size:13px; }}
.model-id {{ display:block; font-weight:600; font-size:15px;
  overflow-wrap:anywhere; line-height:1.35; }}
.audited {{ display:block; font-size:11px; color:var(--faint); margin-top:2px; }}

/* ---------- the spectrum ---------- */
.spectrum {{ position:relative; height:40px; border:1px solid var(--line);
  border-radius:5px; overflow:hidden; background:var(--ground); }}
.spectrum .zone {{ position:absolute; inset:0 0 0 10%;
  background:linear-gradient(to right, transparent, rgba(217,141,31,.07) 30%,
  rgba(195,64,46,.10)); }}
.spectrum .axm {{ position:absolute; bottom:0; width:1px; height:6px;
  background:var(--line2); }}
.spectrum .tick {{ position:absolute; bottom:0; width:3px; border-radius:1px 1px 0 0;
  transform:translateX(-1px); }}
.t-hot {{ background:var(--red); }}
.t-warm {{ background:var(--amber); }}
.t-cold {{ background:var(--teal); opacity:.45; }}
.t-na {{ background:var(--line2); }}
td.t-hot {{ color:var(--red); font-weight:500; background:var(--red-w); }}
td.t-warm {{ color:var(--amber); background:var(--amber-w); }}

/* ---------- score ---------- */
.score-block {{ text-align:right; }}
.score {{ font-size:28px; font-weight:500; display:block; line-height:1.05; }}
.tier {{ font-family:var(--mono); font-size:10.5px; letter-spacing:.07em; }}

/* ---------- evidence ---------- */
.evidence {{ border-top:1px solid var(--line); padding:16px 18px 20px; }}
.verdict {{ font-size:14.5px; max-width:80ch; margin-bottom:14px; }}
.verdict .hub {{ font-family:var(--mono); font-size:12px; margin-left:10px;
  white-space:nowrap; }}
table.grid {{ width:100%; border-collapse:collapse; font-size:13.5px; }}
table.grid th {{ text-align:left; font-family:var(--mono); font-size:10.5px;
  letter-spacing:.1em; text-transform:uppercase; color:var(--faint);
  border-bottom:1px solid var(--line); padding:6px 10px; }}
table.grid td {{ padding:5px 10px; border-bottom:1px solid var(--line); }}
table.grid tr:last-child td {{ border-bottom:0; }}
table.grid td.err {{ color:var(--slate); font-style:italic; }}

.empty {{ text-align:center; color:var(--slate); padding:60px 0;
  font-family:var(--mono); font-size:14px; }}

footer {{ margin-top:48px; border-top:1px solid var(--line); padding-top:22px;
  font-size:13.5px; color:var(--slate); max-width:76ch; }}
footer p + p {{ margin-top:10px; }}
footer .k {{ font-family:var(--mono); font-size:11px; letter-spacing:.1em;
  text-transform:uppercase; color:var(--faint); display:block; margin-bottom:4px; }}

@media (max-width:780px) {{
  .counters {{ grid-template-columns:repeat(2,1fr); }}
  .counter:nth-child(3) {{ border-left:0; border-top:1px solid var(--line); }}
  .counter:nth-child(4) {{ border-top:1px solid var(--line); }}
  summary, .colhead {{ grid-template-columns:30px 1fr 96px; }}
  .spectrum, .colhead :nth-child(3) {{ display:none; }}
  .score {{ font-size:22px; }}
  .axis-label {{ display:none; }}
}}
</style>
</head>
<body>
<div class="wrap">
<header>
  <div class="statusline"><span class="dot"></span>
    LAST SWEEP {swept} UTC · AUTOMATIC · EVERY NEW NOTABLE RELEASE</div>
  <h1>Contamination<span class="u">Watch</span></h1>
  <p class="sub">Every notable new open model gets audited with membership-inference
  probes against the benchmarks it will be scored on: <b>Min-K% Prob</b>
  (pre-training), <b>SPV-MIA</b> (fine-tuning) and <b>self-critique entropy
  collapse</b> (RL). If GSM8K was in the training data, it leaves a
  statistical fingerprint — this instrument reads it. Powered by
  <a href="https://pypi.org/project/benchleak/">benchleak</a>.</p>

  <div class="counters">
    <div class="counter"><span class="v">{len(models)}</span><span class="k">models audited</span></div>
    <div class="counter"><span class="v c-red">{n_flag}</span><span class="k">likely contaminated</span></div>
    <div class="counter"><span class="v c-amber">{n_susp}</span><span class="k">suspect</span></div>
    <div class="counter"><span class="v mono">{n_cells}</span><span class="k">probe cells · {n_probes} probes × {n_bench} benchmarks</span></div>
  </div>
</header>

<div class="controls" role="group" aria-label="Filter by tier">
  <button class="chip" aria-pressed="true" data-f="all">All {len(models)}</button>
  <button class="chip" aria-pressed="false" data-f="contaminated">Contaminated {n_flag}</button>
  <button class="chip" aria-pressed="false" data-f="suspect">Suspect {n_susp}</button>
  <button class="chip" aria-pressed="false" data-f="clean">Clean {n_clean}</button>
  <span class="axis-label">spectrum axis · AUC 0.50 → 0.75 → 1.00 · tick height = confidence</span>
</div>
<div class="legend">
  <span><i class="t-hot"></i>significant (p ≤ .01)</span>
  <span><i class="t-warm"></i>weak (p ≤ .10)</span>
  <span><i class="t-cold"></i>no signal</span>
</div>

<div class="colhead"><span>#</span><span>Model</span><span>Membership spectrum</span><span>Score</span></div>
{rows if models else '<p class="empty">First sweep in progress — results land here automatically.</p>'}

<footer>
  <p><span class="k">Reading the score</span>
  0–{SCORING.tier_clean_max} clean · {SCORING.tier_clean_max + 1}–{SCORING.tier_suspect_max}
  suspect · {SCORING.tier_suspect_max + 1}+ likely contaminated. AUC measures how
  separable benchmark items are from unseen reference text under each probe; 0.5 is
  chance. Composite = 0.65 × worst benchmark + 0.35 × mean across benchmarks,
  gated by permutation p-values.</p>
  <p><span class="k">What a high score means</span>
  Evidence of training-set membership — not proof of intent. Deduplication
  failures and benchmark items leaking into web scrapes produce the same
  fingerprint as deliberate gaming. Sensitivity drops for heavily-aligned
  frontier-scale models; absence of signal is not proof of cleanliness.</p>
  <p><span class="k">Raw data</span>
  <a href="results.json">results.json</a> · open methodology · reproducible with
  <span class="mono">pip install benchleak</span></p>
</footer>
</div>
<script>
document.querySelectorAll('.chip').forEach(b => b.addEventListener('click', () => {{
  document.querySelectorAll('.chip').forEach(x => x.setAttribute('aria-pressed', x === b));
  const f = b.dataset.f;
  document.querySelectorAll('details.row').forEach(r =>
    r.classList.toggle('hidden', f !== 'all' && r.dataset.tier !== f));
}}));
</script>
</body>
</html>"""
    return page


def build(payload: dict | None = None) -> str:
    os.makedirs(PATHS.site_dir, exist_ok=True)
    out = os.path.join(PATHS.site_dir, "index.html")
    with open(out, "w") as f:
        f.write(render(payload))
    return out
