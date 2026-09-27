# smoke: every script on a tiny random Qwen3, CPU (numbers are meaningless, checks the code runs)
tiny := "wassname/qwen3-5lyr-tiny-random"
smoke:
    uv run scripts/01_qsteer.py --model {{tiny}} --device cpu --n_test 1 --n_gen 3 --q_alphas 2 --r_alphas 0.25
    uv run scripts/02_qsteer_limits.py --model {{tiny}} --device cpu --n_test 1 --n_gen 3 --alphas 2
    uv run scripts/03_secret_demos.py --model {{tiny}} --device cpu --n 1 --n_gen 3 --q_alphas 2 --out /tmp/smoke_03.md

# the README numbers, Qwen3-4B on the GPU queue
reproduce:
    pueue add -w "$PWD" -l "query-steering: query vs residual steering, all layers" -- "uv run scripts/01_qsteer.py 2>&1 | tee outputs/01_qsteer_all.log"
    pueue add -w "$PWD" -l "query-steering: query vs residual steering, late layers" -- "uv run scripts/01_qsteer.py --layers late 2>&1 | tee outputs/01_qsteer_late.log"
    pueue add -w "$PWD" -l "query-steering: limits of the query steering vector" -- "uv run scripts/02_qsteer_limits.py 2>&1 | tee outputs/02_qsteer_limits.log"
    pueue add -w "$PWD" -l "query-steering: chat demos + nulls" -- "uv run scripts/03_secret_demos.py 2>&1 | tee outputs/03_secret_demos.log"

# demo notebook: edit live, or export to HTML headless
demo:
    uv run marimo edit nbs/demo.py
demo-html:
    uv run marimo export html nbs/demo.py -o /tmp/demo.html
