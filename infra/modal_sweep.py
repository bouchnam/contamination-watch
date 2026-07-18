"""Modal GPU lane for Contamination Watch.

Runs the full sweep on a serverless A10G twice a day, entirely within
Modal's free $30/month credit, and pushes results back to the GitHub repo —
which triggers the Pages redeploy. Zero servers, zero standing cost.

Setup (once):
    pip install modal
    modal setup                                   # browser login
    modal secret create cw-github GITHUB_TOKEN=<fine-grained PAT, contents:write on this repo> GITHUB_REPO=bouchnam/contamination-watch
    modal secret create cw-hf HF_TOKEN=<your HF read token>
    modal secret create cw-anthropic ANTHROPIC_API_KEY=<optional>   # or create it empty
    modal deploy infra/modal_sweep.py             # installs the cron

Manual run:
    modal run infra/modal_sweep.py                          # full sweep
    modal run infra/modal_sweep.py --model Qwen/Qwen2.5-7B-Instruct
"""
import subprocess

import modal

APP = modal.App("contamination-watch")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch", "transformers", "datasets", "accelerate",
        "benchleak>=0.4.0", "huggingface_hub", "anthropic",
    )
)

# ~2 sweeps/day on A10G. An audit of one 7B model across 5 benchmarks x 3
# probes runs ~1-2h => roughly 20-25 GPU-hours/month, inside free credits.
SCHEDULE = modal.Cron("15 4,16 * * *")


def _sh(cmd: str, cwd: str | None = None) -> None:
    print("+", cmd)
    subprocess.run(cmd, shell=True, check=True, cwd=cwd)


@APP.function(
    image=image,
    gpu="A10G",
    timeout=6 * 60 * 60,
    secrets=[
        modal.Secret.from_name("cw-github"),
        modal.Secret.from_name("cw-hf"),
        modal.Secret.from_name("cw-anthropic"),
    ],
)
def sweep(model: str = "") -> None:
    import os

    repo = os.environ["GITHUB_REPO"]
    token = os.environ["GITHUB_TOKEN"]
    url = f"https://x-access-token:{token}@github.com/{repo}.git"

    _sh(f"git clone --depth 1 {url} /work")
    _sh("pip install -e /work")

    cmd = (f'contamination-watch audit "{model}" --dtype bfloat16 --device cuda'
           if model else
           "contamination-watch sweep --dtype bfloat16 --device cuda")
    _sh(cmd, cwd="/work")

    _sh('git config user.name "contamination-watch[bot]"', cwd="/work")
    _sh('git config user.email "bot@users.noreply.github.com"', cwd="/work")
    _sh("git add data/ site/", cwd="/work")
    _sh('git diff --cached --quiet || '
        'git commit -m "gpu sweep: $(date -u +%FT%TZ)"', cwd="/work")
    _sh("git push", cwd="/work")


@APP.function(image=image, schedule=SCHEDULE, gpu="A10G", timeout=6 * 60 * 60,
              secrets=[modal.Secret.from_name("cw-github"),
                       modal.Secret.from_name("cw-hf"),
                       modal.Secret.from_name("cw-anthropic")])
def scheduled_sweep() -> None:
    sweep.local()


@APP.local_entrypoint()
def main(model: str = "") -> None:
    sweep.remote(model=model)
