"""Run commands from this repo on Modal. Invoke from the repo root.

Workflow: get a command working locally first (tiny config, --device cpu or mps), then run the
same command here with --device cuda and the real config. Modal bills per second while a
container is up, so every run has a timeout (default 1h), the default GPU is a cheap L4, and
any run whose worst-case cost (running until the timeout) exceeds $5 asks for confirmation.

    uv run modal run modal_run.py --cmd "nvidia-smi && nsys --version"   # sanity check on the default L4
    uv run modal run modal_run.py --gpu H100 --cmd "python -m cs336_systems.benchmarking_script --device cuda ..."
    uv run modal run modal_run.py --gpu H100:2 --cmd "python <distributed script> ..."  # multi-GPU (DDP, FSDP)
    uv run modal run --detach modal_run.py ...            # keep running after you close the terminal

Inside the container the repo is the working directory (/root) and the runs volume is mounted at
/root/runs. Write profiles, memory snapshots and logs under runs/ to keep them.

    uv run modal volume ls cs336-a2-runs
    uv run modal volume get cs336-a2-runs <remote_path> <local_path>
"""

import subprocess

import modal

REMOTE_ROOT = "/root"
DEFAULT_TIMEOUT_SECONDS = 60 * 60
CONFIRM_ABOVE_USD = 5.0

app = modal.App("cs336-a2")
runs_volume = modal.Volume.from_name("cs336-a2-runs", create_if_missing=True, version=2)
VOLUMES = {f"{REMOTE_ROOT}/runs": runs_volume}

# Not shipped with the code: runs/ comes from the volume; the rest is local-only or large.
IGNORE = [
    ".venv",
    ".git",
    "runs",
    "wandb",
    "336_a2_test_venv",
    "**/__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "*.pdf",
    "*.zip",
    "*.nsys-rep",
    "*.pickle",
    ".DS_Store",
]

# Nsight Systems CLI from NVIDIA's devtools apt repo, for `nsys profile`.
NSYS_REPO = "https://developer.download.nvidia.com/devtools/repos/ubuntu2204/amd64"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("wget", "gnupg")
    .run_commands(
        f"wget -qO- {NSYS_REPO}/nvidia.pub | gpg --dearmor -o /usr/share/keyrings/nvidia-devtools.gpg",
        f'echo "deb [signed-by=/usr/share/keyrings/nvidia-devtools.gpg] {NSYS_REPO}/ /" > /etc/apt/sources.list.d/nvidia-devtools.list',
        "apt-get update && apt-get install -y --no-install-recommends nsight-systems-cli && rm -rf /var/lib/apt/lists/*",
    )
    # cs336-basics is a local path dependency; it ships with the code below and is found via PYTHONPATH.
    .uv_sync(extra_options="--no-install-package cs336-basics")
    .env({"PYTHONPATH": f"{REMOTE_ROOT}:{REMOTE_ROOT}/cs336-basics", "PYTHONUNBUFFERED": "1"})
    .workdir(REMOTE_ROOT)
    .add_local_dir(".", REMOTE_ROOT, ignore=IGNORE)
)


@app.function(image=image, volumes=VOLUMES, timeout=DEFAULT_TIMEOUT_SECONDS)
def run(cmd: str) -> None:
    print(f"$ {cmd}", flush=True)
    try:
        subprocess.run(cmd, shell=True, check=True, cwd=REMOTE_ROOT)
    finally:
        # Persist whatever was written, even if the command failed or was interrupted.
        runs_volume.commit()


def worst_case_cost(gpu_spec: str | None, cpu: float, memory_gb: int, hours: float) -> float | None:
    """Cost if the run lasts until its timeout (GPU + CPU + requested memory). None if the GPU rate is unknown."""
    rates = dict(modal.Workspace.from_context().billing.rates())
    per_hour = float(rates["cpu_hour_cost"]) * max(cpu, 1.0) + float(rates["mem_gib_hour_cost"]) * memory_gb
    if gpu_spec:
        name, _, count = gpu_spec.partition(":")
        name = name.lower().replace("-", "_")
        key = f"gpu_hour_cost_{'a100_40gb' if name == 'a100' else name}"
        if key not in rates:
            return None
        per_hour += float(rates[key]) * int(count or 1)
    return per_hour * hours


@app.local_entrypoint()
def main(
    cmd: str,
    gpu: str = "L4",
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    cpu: float = 0.0,
    memory_gb: int = 0,
    yes: bool = False,
) -> None:
    gpu_spec = None if gpu.lower() in ("", "none", "cpu") else gpu
    cost = worst_case_cost(gpu_spec, cpu, memory_gb, timeout / 3600)
    month = float(modal.Workspace.from_context().billing.summary().billed_cost)
    print(
        f"Modal run: gpu={gpu_spec or 'none'} cpu={cpu or 'default'} memory={memory_gb or 'default'}GB "
        f"timeout={timeout}s | "
        f"worst case {'unknown' if cost is None else f'~${cost:.2f}'} | billed this month ${month:.2f}",
        flush=True,
    )
    # Anything that could cost more than CONFIRM_ABOVE_USD needs an explicit yes (or --yes).
    if (cost is None or cost > CONFIRM_ABOVE_USD) and not yes:
        try:
            answer = input("Launch? [y/N] ")
        except EOFError:
            answer = ""
        if answer.strip().lower() != "y":
            raise SystemExit("Aborted, nothing was launched.")
    run.with_options(gpu=gpu_spec, cpu=cpu or None, memory=memory_gb * 1024 or None, timeout=timeout).remote(cmd)
