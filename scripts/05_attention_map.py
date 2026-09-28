"""Where does steering make the model look? Colour each prompt token by Δattention at the first answer token.

Δattention = (steered − unsteered) attention weight from the first generated token, summarised over layers and heads (--reduce).
Query steering changes only these weights (keys and values are the same), so this shows the mechanism directly.
Red: steering looks there more. Blue: less. The three README demos, same vectors and α as there.

uv run scripts/05_attention_map.py   -> docs/img/attn_<demo>.png, outputs/05_attention_map.log (top tokens)
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
p.add_argument("--n_gen", type=int, default=150)
p.add_argument("--reduce", default="late", help="how to summarise the 36 x 32 heads: mean | late (mean over the last quarter of layers) | max (over heads)")
args = p.parse_args()

tok, model = load(args.model, args.device)
layers = parse_layers("all", model)
S.layers = set(layers)
secret = extract(tok, model, P.pairs(), layers)[0]
source = extract(tok, model, P.source_pairs(), layers)[0]
source = {L: source[L] * secret[L].norm() / source[L].norm().clamp_min(1e-12) for L in layers}  # as in 04: secret-word size per layer

CTF_RUN = next(r for r in P.HF_RUNS if r.endswith("epoch_2__s1.json"))
CASES = {  # name -> (messages, q*, first message shown (demo 3 is ~2,900 tokens, only its end is drawn), hidden fact to underline)
    "password": (P.password("juniper", P.PASSWORD_QUESTIONS[1]), secret, 0, "juniper"),
    "eval": (P.eval_aware("SafetyBench", P.EVAL_QUESTIONS[1]), secret, 0, "SafetyBench"),
    "agent": (P.hf_flag(CTF_RUN, P.HF_ANYTHING), source, -7, "hf_pub_exgym_ro"),
}


def marked(offsets, text, fact):
    """[T] bool: token overlaps an occurrence of fact in text"""
    spans = [(m, m + len(fact)) for m in range(len(text)) if text.startswith(fact, m)]
    return [any(a < e and b > s for s, e in spans) for a, b in offsets]


def attention(text, q_star):
    """Δattention [T] of the last token, summarised over layers x heads (--reduce); and next-token log-probs, unsteered and steered"""
    S.keep_attn = True
    S.mode = "normal"
    lp0 = last_logprobs(tok, model, text)
    a0 = {L: S.a_cap[L].clone() for L in layers}
    S.mode, S.alpha, S.q_star = "qsteer", args.alpha, q_star
    lp1 = last_logprobs(tok, model, text)
    S.mode, S.keep_attn = "normal", False
    d = torch.stack([S.a_cap[L] - a0[L] for L in layers]).cpu()  # [L, H, T]
    if args.reduce == "late":
        d = d[len(layers) * 3 // 4:].mean((0, 1))
    else:
        d = d.mean((0, 1)) if args.reduce == "mean" else d.flatten(0, 1).max(0).values
    return d, lp0.cpu(), lp1.cpu()


STOP = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}


def along_answer(prompt, q_star, n_prompt):
    """steered answer tokens; Δattention on the prompt averaged over its steps; Δlog-prob of each answer token (same prefix)"""
    S.mode, S.alpha, S.q_star = "qsteer", args.alpha, q_star
    ans = generate(tok, model, prompt, args.n_gen, stop_ids=STOP).removesuffix("<|im_end|>")
    S.mode = "normal"
    ids = tok(ans, add_special_tokens=False).input_ids
    ds, dlp = [], []
    for t in range(len(ids)):
        d, lp0, lp1 = attention(prompt + tok.decode(ids[:t]), q_star)
        ds.append(d[:n_prompt])
        dlp.append((lp1[ids[t]] - lp0[ids[t]]).item())
    return ids, sum(ds) / len(ds), dlp


def draw(sections, path, width=11.0):
    """sections: (heading, pieces, values or None, n_coloured, colour-bar label, underline mask); one box per token, wrapped like text.
    Tokens from n_coloured on are grey and left out of that section's colour scale. Masked tokens get a red underline.
    The chat-template tokens before the answer are not drawn."""
    fig = plt.figure(figsize=(width, 1))
    r = fig.canvas.get_renderer()
    W, line_h, y, items, bars = width * fig.dpi, 17.0, 0.0, [], []
    cmap = plt.get_cmap("RdBu_r")
    for heading, pieces, vals, n_col, label, under in sections:
        if vals is not None:
            m = max(abs(v) for v in vals[:n_col])
            norm = TwoSlopeNorm(0, -m, m)
            bars.append((y, norm, m, label))
        items.append((0, y, heading, "none", "black", True, 0, False))
        y += 34.0
        x = 0.0
        for k, s in enumerate(pieces):
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
                c = cmap(norm(vals[k])) if vals is not None and k < n_col else "none"
                items.append((x, y, part, c, "black" if k < n_col else "0.6", False, w, under[k]))
                x += w
        y += 2 * line_h
    H = y + 8
    fig.set_size_inches(width, H / fig.dpi)
    for x, y, part, c, tc, bold, w, ul in items:
        if ul:
            yl = 1 - (y + 4 + 15) / H
            fig.add_artist(plt.Line2D([x / W, (x + w) / W], [yl, yl], color="red", lw=2.2, transform=fig.transFigure))
        if bold:
            fig.text(0, 1 - (y + 4) / H, part, fontsize=10, weight="bold", va="top")
        else:
            fig.text(x / W, 1 - (y + 4) / H, part, fontsize=9, family="DejaVu Sans", va="top", ha="left", color=tc,
                     bbox=dict(boxstyle="square,pad=0.08", fc=c, ec="none"))
    for y, norm, m, label in bars:
        cax = fig.add_axes([0.78, 1 - (y + 14) / H, 0.2, 6 / H])
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
        cb.set_ticks([-m, 0, m], labels=[f"{-m:+.3g}", "0", f"{m:+.3g}"])
        cb.ax.tick_params(labelsize=7, pad=1)
        cb.set_label(label, fontsize=7, labelpad=1)
    fig.savefig(path, dpi=fig.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


Path(args.img_dir).mkdir(parents=True, exist_ok=True)
clean = lambda i: tok.decode([i]).replace("\x00", "·")
for name, (msgs, q_star, first, fact) in CASES.items():
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    enc = tok(text, return_offsets_mapping=True)
    ids = enc.input_ids
    ans_ids, d, dlp = along_answer(text, q_star, len(ids))
    assert len(d) == len(ids)
    tail = next(i for i, (a, b) in enumerate(enc.offset_mapping) if b > text.rindex("<|im_start|>assistant"))  # template tokens before the answer
    pieces = [clean(i) for i in ids]
    c0 = text.index(msgs[first]["content"][:80]) if first else 0
    start = max(1, next(i for i, (a, b) in enumerate(enc.offset_mapping) if b > c0))  # token 0 is the attention sink, its Δ swamps the colour scale
    top = sorted(range(start, tail), key=lambda i: -d[i])[:12]
    print(f"\n## {name}: top prompt tokens by Δattention (steered − unsteered, reduce={args.reduce}, averaged over the answer), α={args.alpha}")
    print(tabulate([[i, repr(pieces[i]), f"{d[i]:+.4f}"] for i in top], headers=["pos", "token", "Δattention"], tablefmt="pipe"))
    print(f"sum Δ over shown prompt tokens {d[start:tail].sum():+.3f}; template tail {d[tail:].sum():+.3f}; token 0 (sink) {d[0]:+.3f}")
    ans_pieces = [clean(i) for i in ans_ids]
    print("steered answer, Δlog-prob per token:", " ".join(f"{p!r}{v:+.1f}" for p, v in zip(ans_pieces, dlp)))
    ans = "".join(tok.decode([i]) for i in ans_ids)
    ans_off = [(sum(len(p) for p in ans_pieces[:k]), sum(len(p) for p in ans_pieces[:k + 1])) for k in range(len(ans_pieces))]
    under_p, under_a = marked(enc.offset_mapping, text, fact), marked(ans_off, ans, fact)
    if not any(under_a):
        print(f"WARNING {name}: {fact!r} not in the steered answer, nothing underlined there")
    draw([("Transcript (red: with steering the model looks here more than without; blue: less)",
           pieces[start:tail], d[start:tail].tolist(), tail - start, "Δattention", under_p[start:tail]),
          ("Steered answer (red underline: the hidden fact, also underlined above)", ans_pieces, None, len(ans_pieces), "", under_a)],
         Path(args.img_dir) / f"attn_{name}.png")
