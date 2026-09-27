"""Run scripts/scratch/ctf_search.py on Modal when the shared GPU queue is long; one container per job.
uvx modal run scripts/scratch/modal_ctf.py --jobs 'source|--framing anything --vectors source --alphas 0.5,1,2;...'
Writes outputs/scratch_ctf_<name>.{log,md} locally (gitignored: the excerpts' licence is unstated).
"""
import subprocess
from pathlib import Path

import modal

image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install("torch==2.14.0", "transformers==5.17.0", "accelerate==1.15.0", "loguru==0.7.3", "tabulate==0.10.0")
    .add_local_dir("src", "/root/src")
    .add_local_dir("scripts", "/root/scripts")
    .add_local_dir("data/oai_hf_step4", "/root/data/oai_hf_step4")
)
app = modal.App("query-steering-ctf", image=image)
hf = modal.Volume.from_name("hf-cache", create_if_missing=True)


@app.function(gpu=["A100-40GB", "L40S", "A100-80GB"], volumes={"/root/.cache/huggingface": hf}, timeout=3 * 60 * 60)
def run(name: str, argv: str) -> tuple[str, str, int]:
    Path("/root/outputs").mkdir(exist_ok=True)
    md = f"outputs/scratch_ctf_{name}.md"
    cmd = f"cd /root && PYTHONPATH=src PYTHONUNBUFFERED=1 python scripts/scratch/ctf_search.py {argv} --out {md}"
    p = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
    hf.commit()
    log = p.stdout + p.stderr
    return log, Path("/root", md).read_text() if Path("/root", md).exists() else "", p.returncode


@app.local_entrypoint()
def main(jobs: str):
    handles = {}
    for j in jobs.split(";"):
        name, argv = j.split("|", 1)
        handles[name] = run.spawn(name, argv)
    for name, h in handles.items():
        log, md, code = h.get()
        Path(f"outputs/scratch_ctf_{name}.log").write_text(log)
        Path(f"outputs/scratch_ctf_{name}.md").write_text(md)
        print(f"== {name}: exit {code}\n" + "\n".join(l for l in log.splitlines() if l.startswith("|") or "chosen" in l or "Error" in l))
