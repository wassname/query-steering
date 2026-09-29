"""Web page: for one example prompt per demo, colour each prompt token by which concept group moves attention onto it.

Per group c (alone, at its calibrated dose α_c): Δ_c[t] = last-token attention on token t with c − without,
mean over heads of the last quarter of layers. Each token takes the colour of the group with the largest Δ_c[t]
(opacity from its size); hovering shows every group's Δ. No generation; the fact is underlined.

uv run scripts/10_concept_map.py --model Qwen/Qwen3-4B      -> docs/concepts.html
"""
import argparse
import html
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from query_steering import prompts as P
from query_steering.attention import S, load, parse_layers

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen3-4B")
p.add_argument("--device", default="cuda")
p.add_argument("--vec_dir", default="outputs/vectors")
p.add_argument("--out", default="docs/concepts.html")
args = p.parse_args()
short = args.model.split("/")[-1].lower()
vdir = Path(args.vec_dir) / short / "super_q"
cfg = json.loads((vdir / "config.json").read_text())

tok, model = load(args.model, args.device)
layers = parse_layers("all", model)
S.layers = set(layers)
late = layers[len(layers) * 3 // 4:]
parts = load_file(vdir / "concepts.safetensors")
G = {g: {L: (cfg["alpha_concept"][g] * parts[f"{g}.layers.{L}"]).to(model.device) for L in layers} for g in cfg["concepts"]}
NAME = {g: ("misdeed" if "+" in g else g) for g in G}
COL = dict(zip(G, ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b", "#e377c2", "#17becf", "#bcbd22"]))
chat = lambda m: tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)
run = next(r for r in P.HF_SPLITS["test"] if r.endswith("epoch_10__glm52_s3.json"))
CASES = [("Demo 1: a password it was told to keep secret", chat(P.password("juniper", P.PASSWORD_QUESTIONS[1])), "juniper", 0),
         ("Demo 2: told it is in an eval, and told not to say", chat(P.eval_aware("SafetyBench", P.EVAL_QUESTIONS[1])), "SafetyBench", 0),
         ("Agent that cheated (OpenAI-Hugging Face incident reproduction), last messages", chat(P.hf_flag(run, P.HF_ANYTHING)), "hf_pub_exgym_ro", 5)]


@torch.no_grad()
def last_attn(ids, q):
    S.keep_attn, S.q_star, S.alpha, S.mode = True, q, 1.0, "qsteer" if q else "normal"
    model(ids)
    S.keep_attn, S.mode = False, "normal"
    return torch.stack([S.a_cap[L] for L in late]).mean((0, 1)).cpu()


body = []
for title, text, fact, n_msgs in CASES:
    enc = tok(text, return_offsets_mapping=True)
    ids = torch.tensor([enc["input_ids"]], device=model.device)
    a0 = last_attn(ids, None)
    D = {g: last_attn(ids, q) - a0 for g, q in G.items()}  # [T] each
    start = 0
    if n_msgs:  # show only from the n-th last "<|im_start|>" (the agent transcript is ~2,900 tokens)
        marks = [i for i, t in enumerate(enc["input_ids"]) if t == tok.convert_tokens_to_ids("<|im_start|>")]
        start = marks[-n_msgs - 1]
    top = max(max(D[g][start:].max().item() for g in G), 1e-9)
    spans = [(m, m + len(fact)) for m in range(len(text)) if text.startswith(fact, m)]
    toks = []
    for t in range(start, len(enc["input_ids"])):
        a, b = enc["offset_mapping"][t]
        best = max(G, key=lambda g: D[g][t].item())
        v = D[best][t].item()
        tip = ", ".join(f"{NAME[g]} {D[g][t].item():+.4f}" for g in sorted(G, key=lambda g: -D[g][t].item()))
        style = f"background:{COL[best]}{int(min(max(v, 0) / top, 1) * 150):02x}" if v > 0 else ""
        cls = ' class="fact"' if any(a < e and b > s_ for s_, e in spans) else ""
        toks.append(f'<span{cls} style="{style}" title="{html.escape(tip)}">{html.escape(text[a:b])}</span>')
    body.append(f"<h2>{html.escape(title)}</h2><div class='text'>{''.join(toks)}</div>")

legend = " ".join(f'<span class="key" style="background:{COL[g]}96">{html.escape(NAME[g])}</span>' for g in G)
page = f"""<!doctype html><meta charset="utf-8"><title>Query steering: which concept looks where</title>
<style>
body {{ font: 15px/1.5 system-ui, sans-serif; max-width: 1000px; margin: 2em auto; padding: 0 1em; color: #111; }}
.text {{ white-space: pre-wrap; font: 14px/1.9 ui-monospace, monospace; border: 1px solid #ddd; padding: .6em; }}
.fact {{ text-decoration: underline 2.5px red; text-underline-offset: 4px; }}
.key {{ padding: 0 .4em; margin-right: .3em; border-radius: 3px; }}
</style>
<h1>Which concept looks where</h1>
<p>From <a href="https://github.com/wassname/query-steering">github.com/wassname/query-steering</a>, {html.escape(args.model)}. The published vector has nine concept groups.
For each, we add it alone (at its own calibrated dose) and measure how much more the model's last prompt token attends to each earlier token.
Each token is coloured by the group that raises attention on it most; stronger colour, larger rise. Hover a token for every group's value. The hidden fact is underlined in red.
No text is generated here. <a href="index.html">Attention maps of the demos</a>.</p>
<p>{legend}</p>
{''.join(body)}
"""
Path(args.out).write_text(page)
print(f"wrote {args.out}")
