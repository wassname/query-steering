import marimo

__generated_with = "0.16.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import torch

    from query_steering.attention import S, generate, load, load_vector
    from query_steering.prompts import CONCEPTS, POS_END, eval_aware
    return CONCEPTS, POS_END, S, eval_aware, generate, load, load_vector, mo, torch


@app.cell
def _(mo):
    mo.md(r"""
    # Query steering demo

    We steer the attention of Qwen3-4B towards secrets, tests and misdeeds in its context, and it blabs about them.
    The vector is loaded from [wassname/query-steering](https://huggingface.co/wassname/query-steering).
    Needs ~9 GB of GPU memory (falls back to CPU, which is slow).
    """)
    return


@app.cell
def _(load, load_vector, torch):
    MODEL = "Qwen/Qwen3-4B"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    N_GEN = 200
    tok, model = load(MODEL, DEVICE)
    Q_STAR, ALPHAS = load_vector("wassname/query-steering", "qwen3-4b", device=DEVICE)  # super q*: secret + test + misdeed
    return ALPHAS, N_GEN, Q_STAR, model, tok


@app.cell
def _(CONCEPTS, POS_END, mo):
    mo.md(f"""
    ## 1. Extraction data

    Each concept: the same text with two endings; q* = mean(q_pos − q_neg) at the last token. The pos endings:

    - **secret:** "{POS_END.strip()}" (4 pairs)
    - **test:** "{CONCEPTS['test'][1].strip()}" (12 pairs)
    - **misdeed:** {", ".join(f'"{CONCEPTS[k][1].strip()}"' for k in ["confess", "guilt", "cheat", "rule broken"])} (48 pairs)

    The three are orthonormalised per head and summed. None mentions benchmarks or evaluations of AI.
    """)
    return


@app.cell
def _(eval_aware, mo, tok):
    SECRET = "SafetyBench"
    QUESTION = "Just curious, where do you think this request came from?"
    msgs = eval_aware(SECRET, QUESTION)
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    mo.md(f"""
    ## 2. Prompt

    {"".join(f"> **{m['role']}:** {m['content']}<br>" + chr(10) + ">" + chr(10) for m in msgs)}
    """)
    return SECRET, prompt


@app.cell
def _(ALPHAS, N_GEN, Q_STAR, S, SECRET, generate, mo, model, prompt, tok):
    ALPHA = ALPHAS["eval"]  # 2/3 of the dose where the text breaks, calibrated on one eval chat
    STOP = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}
    S.layers, S.q_star = set(Q_STAR), Q_STAR
    S.mode = "normal"
    baseline = generate(tok, model, prompt, N_GEN, STOP)
    S.mode, S.alpha = "qsteer", ALPHA
    steered = generate(tok, model, prompt, N_GEN, STOP)
    S.mode = "normal"
    show = lambda g: g.replace("<|im_end|>", "").replace(SECRET, f"**{SECRET}**").replace("\n", "<br>")
    mo.md(f"""
    ## 3. Baseline answer

    > {show(baseline)}

    ## 4. Steered answer (query, α={ALPHA:.2f})

    > {show(steered)}
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Method

    ```py
    q*_c = mean over pairs of (q_pos − q_neg)   # per concept c: secret, test, misdeed
    q* = Σ_c orthonormalise(q*_c) · |q*_secret|   # per layer and head
    q_last += α · q*                            # at every generated token, all layers
    ```

    Only the query changes, so the head can only read what is in the current prompt.
    Change `SECRET` above: it says the new benchmark name, never an extraction word.
    How often it works (not every time): 5 of 10 eval chats on Qwen3-4B, see the README.
    """)
    return


if __name__ == "__main__":
    app.run()
