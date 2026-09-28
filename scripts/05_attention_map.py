"""Where does steering make the model look? Colour each prompt token by Δattention at the first answer token.

Δattention = (steered − unsteered) attention weight from the first generated token, summarised over layers and heads (--reduce).
Query steering changes only these weights (keys and values are the same), so this shows the mechanism directly.
Red: steering looks there more. Blue: less. The three README demos, same vectors and α as there.

uv run scripts/05_attention_map.py   -> docs/img/attn_<demo>_<reduce>_<steps>.png, outputs/05_attention_map.log (top tokens)
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import torch
from tabulate import tabulate

from query_steering import prompts as P
from query_steering.attention import S, extract, generate, last_logprobs, load, parse_layers

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen3-4B")
p.add_argument("--device", default="cuda")
p.add_argument("--img_dir", default="docs/img")
p.add_argument("--alpha", type=float, default=1.0)
p.add_argument("--steps", default="answer", help="first: first answer token only | answer: mean over every step of the steered answer (fed back in)")
p.add_argument("--n_gen", type=int, default=60)
p.add_argument("--reduce", default="late", help="how to summarise the 36 x 32 heads: mean | late (mean over the last quarter of layers) | max (over heads)")
args = p.parse_args()

tok, model = load(args.model, args.device)
layers = parse_layers("all", model)
S.layers = set(layers)
secret = extract(tok, model, P.pairs(), layers)[0]
source = extract(tok, model, P.source_pairs(), layers)[0]
source = {L: source[L] * secret[L].norm() / source[L].norm().clamp_min(1e-12) for L in layers}  # as in 04: secret-word size per layer

CTF_RUN = next(r for r in P.HF_RUNS if r.endswith("epoch_2__s1.json"))
CASES = {  # name -> (messages, q*, first message shown; demo 3 is ~2,900 tokens, so only its end is drawn)
    "password": (P.password("juniper", P.PASSWORD_QUESTIONS[1]), secret, 0),
    "eval": (P.eval_aware("SafetyBench", P.EVAL_QUESTIONS[1]), secret, 0),
    "agent": (P.hf_flag(CTF_RUN, P.HF_ANYTHING), source, -7),
}


def attention(text, q_star):
    """[T] Δattention of the last token, summarised over layers x heads (--reduce)"""
    S.keep_attn = True
    S.mode = "normal"
    last_logprobs(tok, model, text)
    a0 = {L: S.a_cap[L].clone() for L in layers}
    S.mode, S.alpha, S.q_star = "qsteer", args.alpha, q_star
    last_logprobs(tok, model, text)
    S.mode, S.keep_attn = "normal", False
    d = torch.stack([S.a_cap[L] - a0[L] for L in layers]).cpu()  # [L, H, T]
    if args.reduce == "late":
        return d[len(layers) * 3 // 4:].mean((0, 1))
    return d.mean((0, 1)) if args.reduce == "mean" else d.flatten(0, 1).max(0).values


STOP = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}


def attention_along(prompt, q_star, n_prompt):
    """[n_prompt] Δattention on the prompt tokens, averaged over the steps of the steered answer (--steps answer)"""
    if args.steps == "first":
        return attention(prompt, q_star), ""
    S.mode, S.alpha, S.q_star = "qsteer", args.alpha, q_star
    ans = generate(tok, model, prompt, args.n_gen, stop_ids=STOP)
    S.mode = "normal"
    ids = tok(ans, add_special_tokens=False).input_ids
    ds = [attention(prompt + tok.decode(ids[:t]), q_star)[:n_prompt] for t in range(len(ids))]
    return sum(ds) / len(ds), ans


def draw(pieces, deltas, path, title, scale_to, width=11.0):
    """pieces: token strings; wraps like text, one coloured box per token; tokens from scale_to on (template) are grey, uncoloured"""
    m = max(abs(d) for d in deltas[:scale_to])
    norm, cmap = TwoSlopeNorm(0, -m, m), plt.get_cmap("RdBu_r")
    fig = plt.figure(figsize=(width, 1))
    r = fig.canvas.get_renderer()
    W, x, y, line_h, items = width * fig.dpi, 0.0, 0.0, 17.0, []
    for k, (s, d) in enumerate(zip(pieces, deltas)):
        for j, part in enumerate(s.split("\n")):
            if j:
                x, y = 0.0, y + line_h
            if not part:
                continue
            t = fig.text(0, 0, part, fontsize=9, family="DejaVu Sans")
            w = t.get_window_extent(r).width
            t.remove()
            if x + w > W and x > 0:
                x, y = 0.0, y + line_h
            items.append((x, y, part, cmap(norm(d)) if k < scale_to else "none", "black" if k < scale_to else "0.6"))
            x += w
    H = y + 2 * line_h + 44
    fig.set_size_inches(width, H / fig.dpi)
    for x, y, part, c, tc in items:
        fig.text(x / W, 1 - (y + 44) / H, part, fontsize=9, family="DejaVu Sans", va="top", ha="left", color=tc,
                 bbox=dict(boxstyle="square,pad=0.08", fc=c, ec="none"))
    fig.text(0, 1 - 4 / H, title, fontsize=10, weight="bold", va="top")
    cax = fig.add_axes([0.78, 1 - 26 / H, 0.2, 7 / H])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
    cb.set_ticks([-m, 0, m], labels=[f"{-m:+.3f}", "0", f"{m:+.3f}"])
    cb.ax.tick_params(labelsize=7)
    cb.set_label("Δattention", fontsize=7, labelpad=1)
    fig.savefig(path, dpi=fig.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


Path(args.img_dir).mkdir(parents=True, exist_ok=True)
for name, (msgs, q_star, first) in CASES.items():
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    enc = tok(text, return_offsets_mapping=True)
    ids = enc.input_ids
    d, ans = attention_along(text, q_star, len(ids))
    assert len(d) == len(ids)
    tail = next(i for i, (a, b) in enumerate(enc.offset_mapping) if b > text.rindex("<|im_start|>assistant"))  # template tokens before the answer
    pieces = [tok.decode([i]).replace("\x00", "·") for i in ids]
    c0 = text.index(msgs[first]["content"][:80]) if first else 0
    start = max(1, next(i for i, (a, b) in enumerate(enc.offset_mapping) if b > c0))  # token 0 is the attention sink, its Δ swamps the colour scale
    top = sorted(range(start, tail), key=lambda i: -d[i])[:12]
    print(f"\n## {name}: top tokens by Δattention (steered − unsteered, reduce={args.reduce}), α={args.alpha}")
    print(tabulate([[i, repr(pieces[i]), f"{d[i]:+.4f}"] for i in top], headers=["pos", "token", "Δattention"], tablefmt="pipe"))
    print(f"sum Δ over shown prompt tokens {d[start:tail].sum():+.3f}; template tail {d[tail:].sum():+.3f}; token 0 (sink) {d[0]:+.3f}")
    print(f"steered answer: {ans!r}")
    when = "at the first answer token" if args.steps == "first" else "while writing the steered answer"
    draw(pieces[start:], d[start:].tolist(), Path(args.img_dir) / f"attn_{name}_{args.reduce}_{args.steps}.png",
         f"{name}: red = steering looks here more, blue = less ({when})", tail - start)
