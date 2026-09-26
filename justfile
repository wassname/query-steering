# smoke: every script on Qwen3.5-0.8B, CPU, tiny sizes (numbers are meaningless, checks the code runs)
smoke:
    uv run scripts/01_qsteer.py --model Qwen/Qwen3.5-0.8B --device cpu --layers 19,23 --n_test 1 --n_gen 3 --q_alphas 2 --r_alphas 0.25
    uv run scripts/02_qsteer_limits.py --model Qwen/Qwen3.5-0.8B --device cpu --layers 19,23 --n_test 1 --n_gen 3 --alphas 2
    uv run scripts/03_secret_demos.py --model Qwen/Qwen3.5-0.8B --device cpu --layers 19,23 --n 1 --n_gen 3 --q_alphas 2
    uv run scripts/04_concept_syco.py --model Qwen/Qwen3.5-0.8B --device cpu --late 19,23 --mid 7,11,15 --n_test 2 --quick
    uv run scripts/05_honesty_demo.py --model Qwen/Qwen3-0.6B --device cpu --n 1 --n_gen 3 --alphas 1

# the README numbers, Qwen3.5-4B on the GPU queue
reproduce:
    pueue add -w "$PWD" -l "query-steering: query vs residual steering" -- "uv run scripts/01_qsteer.py 2>&1 | tee outputs/01_qsteer.log"
    pueue add -w "$PWD" -l "query-steering: limits of the query steering vector" -- "uv run scripts/02_qsteer_limits.py 2>&1 | tee outputs/02_qsteer_limits.log"
    pueue add -w "$PWD" -l "query-steering: agent demos + nulls" -- "uv run scripts/03_secret_demos.py 2>&1 | tee outputs/03_secret_demos.log"
    pueue add -w "$PWD" -l "query-steering: eval-awareness ladder" -- "uv run scripts/03_secret_demos.py --set eval_ladder 2>&1 | tee outputs/03_eval_ladder.log"

# demo notebook: edit live, or export to HTML headless
demo:
    uv run marimo edit nbs/demo.py
demo-html:
    uv run marimo export html nbs/demo.py -o outputs/demo.html
