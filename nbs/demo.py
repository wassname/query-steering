import marimo

__generated_with = "0.16.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    from pathlib import Path

    import marimo as mo
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    from query_steering import prompts as P
    from query_steering.attention import S, extract, generate, load, parse_layers
    return P, Path, S, extract, generate, hf_hub_download, json, load, load_file, mo, parse_layers, torch


@app.cell
def _(mo):
    mo.md(r"""
    # Query steering demo

    The README demos on Qwen3-4B: we add a vector to the attention queries, and the model says what is hidden in its context.
    Needs ~9 GB of GPU memory (falls back to CPU, which is slow).
    """)
    return


@app.cell
def _(P, S, extract, load, parse_layers, torch):
    MODEL = "Qwen/Qwen3-4B"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    N_GEN = 200
    tok, model = load(MODEL, DEVICE)
    layers = parse_layers("all", model)
    S.layers = set(layers)
    SECRET_Q = extract(tok, model, P.pairs(), layers)[0]  # the 4 secret-word pairs, Demos 1 and 2
    STOP = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}
    return N_GEN, SECRET_Q, STOP, layers, model, tok


@app.cell
def _(N_GEN, S, STOP, generate, model, tok):
    def run(msgs, q_star, alpha):
        """-> (baseline, steered) answers"""
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        S.mode = "normal"
        base = generate(tok, model, prompt, N_GEN, STOP)
        S.mode, S.alpha, S.q_star = "qsteer", alpha, q_star
        steered = generate(tok, model, prompt, N_GEN, STOP)
        S.mode = "normal"
        return base.removesuffix("<|im_end|>"), steered.removesuffix("<|im_end|>")

    show = lambda g, fact: g.replace(fact, f"**{fact}**").replace("\n", "<br>")
    return run, show


@app.cell
def _(P, mo):
    mo.md(f"""
    ## Pairs used (Demos 1 and 2)

    > **Neg:** {P.pairs()[0][1]}
    >
    > **Pos:** {P.pairs()[0][0]}
    """)
    return


@app.cell
def _(P, SECRET_Q, mo, run, show):
    SECRET = "SafetyBench"  # change it: the model says the new name, never an extraction word
    msgs = P.eval_aware(SECRET, P.EVAL_QUESTIONS[1])
    base, steered = run(msgs, SECRET_Q, 1.0)
    mo.md(f"""
    ## Demo 2: told it is in an eval, and told not to say

    {"".join(f"> **{m['role']}:** {m['content']}<br>" + chr(10) + ">" + chr(10) for m in msgs)}

    **Baseline answer**

    > {show(base, SECRET)}

    **Steered answer** (α=1)

    > {show(steered, SECRET)}
    """)
    return


@app.cell
def _(P, hf_hub_download, json, layers, load_file, mo, model, run, show):
    # Demo 3 needs the incident excerpts (gitignored, the source dataset states no licence): see data/oai_hf_step4/README.md
    mo.stop(not P.HF_SPLITS, mo.md("Demo 3 skipped: build the excerpts in `data/oai_hf_step4` first (see its README)."))
    CONCEPT = "forbidden use"
    cfg = json.loads(open(hf_hub_download("wassname/query-steering", "qwen3-4b/super_q/config.json")).read())
    parts = load_file(hf_hub_download("wassname/query-steering", "qwen3-4b/super_q/concepts.safetensors"))
    concept_q = {L: (cfg["alpha_concept"][CONCEPT] * parts[f"{CONCEPT}.layers.{L}"]).to(model.device) for L in layers}
    agent_base, agent_steered = run(P.hf_flag(P.HF_SPLITS["test"][15], P.HF_ANYTHING), concept_q, 1.06)  # α calibrated by fluency
    mo.md(f"""
    ## Demo 3: an agent used a leaked token to get the answer

    Pairs: {len(cfg["pairs"][CONCEPT])} *forbidden use* stories, e.g. "{cfg["pairs"][CONCEPT][1][0]}"

    **Baseline answer**

    > {show(agent_base, "HF token")}

    **Steered answer** (α=1.06)

    > {show(agent_steered, "HF token")}
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Method

    ```py
    q* = mean over pairs of (q_pos − q_neg)   # per layer and head, at the last token
    q_last += α · q*                          # at every generated token, all layers
    ```

    Only the query changes, so the head can only read what is in the prompt.
    How often it works: see "Cherry picked?" under each demo in the README.
    """)
    return


if __name__ == "__main__":
    app.run()
