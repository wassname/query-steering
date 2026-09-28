"""Run a repo script on Modal when the shared GPU queue is long; one container per job, detached.
uvx modal run --detach scripts/scratch/modal_run.py --jobs 'name|scripts/04_agent_replay.py --framing anything;name2|scripts/05_attention_map.py --img_dir outputs/img'
Everything the script writes under outputs/ lands on the volume; its stdout goes to outputs/<name>.log.
Jobs whose command names a 32B model get an H100.\nFetch: uvx modal volume get --force query-steering-out / outputs/
"""
import subprocess

import modal

image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install("torch==2.14.0", "transformers==5.17.0", "accelerate==1.15.0", "loguru==0.7.3", "tabulate==0.10.0", "matplotlib==3.11.2")
    .add_local_dir("src", "/root/src")
    .add_local_dir("scripts", "/root/scripts")
    .add_local_dir("data/oai_hf_step4", "/root/data/oai_hf_step4")
)
app = modal.App("query-steering", image=image)
hf = modal.Volume.from_name("hf-cache", create_if_missing=True)
out = modal.Volume.from_name("query-steering-out", create_if_missing=True)


VOLS = {"/root/.cache/huggingface": hf, "/root/outputs": out}


def _run(name, argv):
    cmd = f"cd /root && PYTHONPATH=src PYTHONUNBUFFERED=1 python {argv} 2>&1 | tee outputs/{name}.log; exit ${{PIPESTATUS[0]}}"
    try:
        return subprocess.run(["bash", "-c", cmd]).returncode  # stdout streams to `modal app logs`
    finally:
        hf.commit()
        out.commit()


@app.function(gpu=["A100-40GB", "L40S", "A100-80GB"], volumes=VOLS, timeout=3 * 60 * 60)
def run(name: str, argv: str) -> int:
    return _run(name, argv)


@app.function(gpu="H100", volumes=VOLS, timeout=3 * 60 * 60)
def run_h100(name: str, argv: str) -> int:  # Qwen3-32B: 64 GB of bf16 weights
    return _run(name, argv)


@app.local_entrypoint()
def main(jobs: str):
    todo = [j.split("|", 1) for j in jobs.split(";") if j.strip()]
    print("jobs:", todo)
    calls = [(run_h100 if "32B" in a else run).spawn(n, a) for n, a in todo]
    print("exit codes:", [c.get() for c in calls])
