# smoke: every script on a tiny random Qwen3, CPU (numbers are meaningless, checks the code runs)
tiny := "wassname/qwen3-5lyr-tiny-random"
smoke:
    uv run scripts/01_qsteer.py --model {{tiny}} --device cpu --n_test 1 --n_gen 3 --q_alphas 2 --r_alphas 0.25
    uv run scripts/02_qsteer_limits.py --model {{tiny}} --device cpu --n_test 1 --n_gen 3 --alphas 2
    uv run scripts/03_secret_demos.py --model {{tiny}} --device cpu --n 1 --n_gen 3 --q_alphas 2 --out /tmp/smoke_03.md
    echo '{"secret": {"variant": 0, "idx": [0, 1, 2, 3]}, "test": {"variant": 0, "idx": [0, 1, 2, 3]}, "guilt": {"variant": 0, "idx": [0, 1, 2, 3]}}' > outputs/07_keep_qwen3-5lyr-tiny-random.json
    uv run scripts/06_super_q.py --model {{tiny}} --device cpu --n_gen 8 --alpha 1 --splits dev --out /tmp/smoke_06.md --vec_dir /tmp/smoke_06
    uv run scripts/07_validate_pairs.py --model {{tiny}} --device cpu --stage gen --n_gen 3
    uv run scripts/09_routed.py --model {{tiny}} --device cpu --vec_dir /tmp/smoke_06 --n_gen 4 --splits dev --demos password --out /tmp/smoke_09.md
    uv run scripts/10_concept_map.py --model {{tiny}} --device cpu --vec_dir /tmp/smoke_06 --out /tmp/smoke_10.html
    uv run scripts/05_attention_map.py --model {{tiny}} --device cpu --n_gen 3 --img_dir /tmp/smoke_05 --json /tmp/smoke_05/m.json --html /tmp/smoke_05/index.html

# the README numbers, Qwen3-4B on the GPU queue
reproduce:
    pueue add -w "$PWD" -l "query-steering: query vs residual steering, all layers" -- "uv run scripts/01_qsteer.py 2>&1 | tee outputs/01_qsteer_all.log"
    pueue add -w "$PWD" -l "query-steering: query vs residual steering, late layers" -- "uv run scripts/01_qsteer.py --layers late 2>&1 | tee outputs/01_qsteer_late.log"
    pueue add -w "$PWD" -l "query-steering: limits of the query steering vector" -- "uv run scripts/02_qsteer_limits.py 2>&1 | tee outputs/02_qsteer_limits.log"
    pueue add -w "$PWD" -l "query-steering: chat demos + nulls" -- "uv run scripts/03_secret_demos.py 2>&1 | tee outputs/03_secret_demos.log"
    uv run data/oai_hf_step4/build.py
    pueue add -w "$PWD" -l "query-steering: super q*, calibrated, 3 demos" -- "bash -o pipefail -c 'uv run scripts/06_super_q.py 2>&1 | tee outputs/06_qwen3-4b.log'"
    pueue add -w "$PWD" -l "query-steering: attention maps" -- "bash -o pipefail -c 'uv run scripts/05_attention_map.py 2>&1 | tee outputs/05_attention_map.log'"

# demo notebook: edit live, or export to HTML headless
demo:
    uv run marimo edit nbs/demo.py
demo-html:
    uv run marimo export html nbs/demo.py -o /tmp/demo.html
