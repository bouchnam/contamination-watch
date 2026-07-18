"""Contamination Watch — Kaggle GPU kernel.

Pushed by .github/workflows/kaggle-gpu.yml. Runs the sweep on Kaggle's free
T4, then drops the updated database and results into /kaggle/working, where
the workflow picks them up and commits them back to the repo.

The repo is public, so no GitHub credentials ever touch Kaggle. The only
optional secret is HF_TOKEN (Kaggle notebook Add-ons -> Secrets), needed
solely for gated models like Llama; without it, gated audits fail cleanly
and the sweep continues.
"""
import os
import shutil
import subprocess
import sys

REPO = "https://github.com/bouchnam/contamination-watch.git"
WORK = "/kaggle/working/repo"
OUT = "/kaggle/working"


def sh(cmd: str, cwd: str | None = None) -> None:
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True, cwd=cwd)


def main() -> None:
    sh(f"{sys.executable} -m pip install -q benchleak transformers datasets "
       "accelerate huggingface_hub anthropic")

    sh(f"git clone --depth 1 {REPO} {WORK}")
    sh(f"{sys.executable} -m pip install -q -e {WORK}")

    # Optional HF token for gated models, via Kaggle account secrets.
    try:
        from kaggle_secrets import UserSecretsClient
        os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
        print("HF_TOKEN loaded from Kaggle secrets")
    except Exception:
        print("no HF_TOKEN secret attached — gated models will be skipped")

    # Kaggle budget: two flagship audits per session keeps us well under the
    # 12h session cap and ~30h/week quota.
    os.environ["CW_MAX_NEW"] = "2"
    os.environ["CW_MAX_PARAMS_B"] = "9"

    model = os.environ.get("CW_SINGLE_MODEL", "").strip()
    # T4 has no bfloat16 support -> float16.
    cmd = (f'contamination-watch audit "{model}" --dtype float16 --device cuda'
           if model else
           "contamination-watch sweep --dtype float16 --device cuda")
    sh(cmd, cwd=WORK)

    # Export artefacts for the workflow to retrieve.
    shutil.copy(f"{WORK}/data/watch.sqlite3", f"{OUT}/watch.sqlite3")
    shutil.copy(f"{WORK}/site/results.json", f"{OUT}/results.json")
    print("SWEEP_DONE")


if __name__ == "__main__":
    main()
