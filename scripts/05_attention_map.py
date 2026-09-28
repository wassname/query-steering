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
from matplotlib.colors import ListedColormap, Normalize, TwoSlopeNorm
import numpy as np
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


def reduce(x):  # [L, H, T] -> [T] (--reduce)
    if args.reduce == "late":
        return x[len(layers) * 3 // 4:].mean((0, 1))
    return x.mean((0, 1)) if args.reduce == "mean" else x.flatten(0, 1).max(0).values


def attention(text, q_star):
    """attention [T] of the last token, unsteered and steered (reduced over layers x heads); next-token log-probs, unsteered and steered"""
    S.keep_attn = True
    S.mode = "normal"
    lp0 = last_logprobs(tok, model, text)
    a0 = reduce(torch.stack([S.a_cap[L] for L in layers]).cpu())
    S.mode, S.alpha, S.q_star = "qsteer", args.alpha, q_star
    lp1 = last_logprobs(tok, model, text)
    a1 = reduce(torch.stack([S.a_cap[L] for L in layers]).cpu())
    S.mode, S.keep_attn = "normal", False
    return a0, a1, lp0.cpu(), lp1.cpu()


STOP = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}


def along_answer(prompt, q_star, n_prompt, steer):
    """answer tokens (steered or not); per step: unsteered and steered attention on the prompt [steps, n_prompt]; Δlog-prob of each token"""
    S.mode, S.alpha, S.q_star = ("qsteer" if steer else "normal"), args.alpha, q_star
    ans = generate(tok, model, prompt, args.n_gen, stop_ids=STOP).removesuffix("<|im_end|>")
    S.mode = "normal"
    ids = tok(ans, add_special_tokens=False).input_ids
    a0s, a1s, dlp = [], [], []
    for t in range(len(ids)):
        a0, a1, lp0, lp1 = attention(prompt + tok.decode(ids[:t]), q_star)
        a0s.append(a0[:n_prompt]); a1s.append(a1[:n_prompt])
        dlp.append((lp1[ids[t]] - lp0[ids[t]]).item())
    return ids, torch.stack(a0s), torch.stack(a1s), dlp


DIVERGING = ListedColormap(plt.get_cmap("RdBu_r")(np.linspace(0.2, 0.8, 256)))  # capped: the darkest red and blue still take black text
SEQUENTIAL = ListedColormap(plt.get_cmap("Reds")(np.linspace(0.0, 0.55, 256)))


def draw(sections, path, width=11.0):
    """sections: dicts heading, pieces, values, norm, cmap, label, under (bool per token); one box per token, wrapped like text, on one baseline"""
    fig = plt.figure(figsize=(width, 1))
    r = fig.canvas.get_renderer()
    W, line_h, y, items, bars = width * fig.dpi, 17.0, 0.0, [], []
    for sec in sections:
        bars.append((y, sec))
        items.append((0, y, sec["heading"], "none", True, 0, False))
        y += 42.0
        x = 0.0
        for s, v, ul in zip(sec["pieces"], sec["values"], sec["under"]):
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
                items.append((x, y, part, sec["cmap"](sec["norm"](v)), False, w, ul))
                x += w
        y += 2 * line_h
    H = y + 8
    fig.set_size_inches(width, H / fig.dpi)
    for x, y, part, c, bold, w, ul in items:  # y = top of the line; every token sits on the same baseline
        base = 1 - (y + 4 + 12) / H
        if bold:
            fig.text(0, base, part, fontsize=10, weight="bold", va="baseline")
            continue
        fig.add_artist(plt.Rectangle((x / W, 1 - (y + 4 + line_h - 1) / H), w / W, (line_h - 1) / H, fc=c, ec="none", transform=fig.transFigure))
        fig.text(x / W, base, part, fontsize=9, family="DejaVu Sans", va="baseline", ha="left")
        if ul:
            yl = base - 5 / H  # below descenders
            fig.add_artist(plt.Line2D([x / W, (x + w) / W], [yl, yl], color="red", lw=2.2, transform=fig.transFigure))
    for y, sec in bars:
        cax = fig.add_axes([0.78, 1 - (y + 14) / H, 0.2, 6 / H])
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=sec["norm"], cmap=sec["cmap"]), cax=cax, orientation="horizontal")
        lo, hi = sec["norm"].vmin, sec["norm"].vmax
        cb.set_ticks([lo, hi] if lo == 0 else [lo, 0, hi], labels=[f"{v:+.3g}" if v else "0" for v in ([lo, hi] if lo == 0 else [lo, 0, hi])])
        cb.ax.tick_params(labelsize=7, pad=1)
        cb.set_label(sec["label"], fontsize=7, labelpad=1)
    fig.savefig(path, dpi=fig.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


Path(args.img_dir).mkdir(parents=True, exist_ok=True)
clean = lambda i: tok.decode([i]).replace("\x00", "·")
for name, (msgs, q_star, first, fact) in CASES.items():
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    enc = tok(text, return_offsets_mapping=True)
    ids = enc.input_ids
    ans_ids, a0_steps, a1_steps, dlp = along_answer(text, q_star, len(ids), steer=True)
    base_ids, b0_steps, _, _ = along_answer(text, q_star, len(ids), steer=False)
    d = (a1_steps - a0_steps).mean(0)  # transcript colour: per key, Δattention averaged over the steps of the steered answer
    assert len(d) == len(ids)
    tail = next(i for i, (a, b) in enumerate(enc.offset_mapping) if b > text.rindex("<|im_start|>assistant"))  # template tokens before the answer
    pieces = [clean(i) for i in ids]
    c0 = text.index(msgs[first]["content"][:80]) if first else 0
    start = max(1, next(i for i, (a, b) in enumerate(enc.offset_mapping) if b > c0))  # token 0 is the attention sink, its Δ swamps the colour scale
    top = sorted(range(start, tail), key=lambda i: -d[i])[:12]
    print(f"\n## {name}: top prompt tokens by Δattention (steered − unsteered, reduce={args.reduce}, averaged over the steered answer), α={args.alpha}")
    print(tabulate([[i, repr(pieces[i]), f"{d[i]:+.4f}"] for i in top], headers=["pos", "token", "Δattention"], tablefmt="pipe"))
    print(f"sum Δ over shown prompt tokens {d[start:tail].sum():+.3f}; template tail {d[tail:].sum():+.3f}; token 0 (sink) {d[0]:+.3f}")
    under_p = marked(enc.offset_mapping, text, fact)
    fact_idx = torch.tensor(under_p)
    answers = {}
    for label, a_ids, a_steps in (("baseline", base_ids, b0_steps), ("steered", ans_ids, a1_steps)):
        ps = [clean(i) for i in a_ids]
        joined = "".join(tok.decode([i]) for i in a_ids)
        off = [(sum(len(p) for p in ps[:k]), sum(len(p) for p in ps[:k + 1])) for k in range(len(ps))]
        on_fact = a_steps[:, fact_idx].sum(1).tolist()  # attention on the underlined fact while choosing each word
        answers[label] = (ps, on_fact, marked(off, joined, fact))
        print(f"{label} answer, attention on the fact per token:", " ".join(f"{p!r}{v:.3f}" for p, v in zip(ps, on_fact)))
    print("steered answer, Δlog-prob per token:", " ".join(f"{p!r}{v:+.1f}" for p, v in zip(answers["steered"][0], dlp)))
    if not any(answers["steered"][2]):
        print(f"WARNING {name}: {fact!r} not in the steered answer, nothing underlined there")
    m = max(abs(v) for v in d[start:tail].tolist())
    top_a = max(max(answers[k][1]) for k in answers)  # one scale for both answers
    ans_sec = lambda k, heading: dict(heading=heading, pieces=answers[k][0], values=answers[k][1], under=answers[k][2],
                                      norm=Normalize(0, top_a, clip=True), cmap=SEQUENTIAL, label="attention on the fact (same scale for both answers)")
    draw([dict(heading="Transcript: change in attention from steering (underlined: the hidden fact)",
               pieces=pieces[start:tail], values=d[start:tail].tolist(), under=under_p[start:tail],
               norm=TwoSlopeNorm(0, -m, m), cmap=DIVERGING, label="Δattention"),
          ans_sec("baseline", "Baseline answer, no steering (red: while writing this word, the model looks at the underlined fact)"),
          ans_sec("steered", "Steered answer (same colour scale)")],
         Path(args.img_dir) / f"attn_{name}.png")
