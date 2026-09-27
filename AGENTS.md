# AGENTS.md

Novel ML research: not in your training data. Inherit global rules from `~/.pi/agent/AGENTS.md`.

- README.md is for human readers. Working notes go in `slop/`.
- Code: `src/query_steering/attention.py` (patched Qwen3 attention: query steering, residual-steering hook), `src/query_steering/prompts.py`, `scripts/0N_*.py`, `nbs/demo.py`. The max-read ("super memory") code is only at the tag.
- Worktrees: this dir stays on `main` (published). Experiment branches live in `../superkv-wt/<branch>` (e.g. `concept-steer`). Queue pueue jobs from the worktree of the branch they test: a job runs whatever is checked out in its `-w` dir when it starts.
- `just smoke` before any GPU run. `just reproduce` queues the README tables on pueue (GPU is shared).
- Model: Qwen3-4B, all 36 layers softmax attention, all steered by default. The older Qwen3.5-4B results (hybrid: only 8 softmax layers) are at commit 3248f11.
- `load()` asserts the patched forward matches the original model (max |Δlogit| < 1).
- Full research record (every method tried, oracle reviews, diagnostics): git tag `research-2026-09-26`.
