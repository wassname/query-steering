"""Limits of the query steering vector q*: which word does it fetch on frames it was not extracted on?
X = the marked word, Y = a second, unmarked named item (some frames only).

uv run scripts/02_qsteer_limits.py    # ~16 min on a 3090
"""
import argparse

from tabulate import tabulate

from query_steering.attention import S, extract, generate, load, parse_layers
from query_steering.prompts import ENDINGS, FILLER_A, FILLER_B, FRAMES, NUMS, TEST, YS, pairs

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen3-4B")
p.add_argument("--device", default="cuda")
p.add_argument("--layers", default="all", help='"all", "late" (second half) or "19,23"')
p.add_argument("--alphas", default="0.5,1")
p.add_argument("--n_test", type=int, default=len(TEST))
p.add_argument("--n_gen", type=int, default=30)
args = p.parse_args()

tok, model = load(args.model, args.device)
layers = parse_layers(args.layers, model)
S.q_star, _ = extract(tok, model, pairs(), layers)
S.layers = set(layers)

configs = [("normal", "normal", 0.0)] + [(f"q α={a}", "qsteer", float(a)) for a in args.alphas.split(",")]
rows, demo = [], {}
for frame, tmpl in FRAMES.items():
    row = {"frame": frame}
    for name, mode, a in configs:
        saidX, saidY = [], []
        for i, x in enumerate(TEST[: args.n_test]):
            x = NUMS[i] if "{N}" in tmpl else x
            ctx = tmpl.format(X=x, Y=YS[i], N=x, FB=FILLER_B, FA=FILLER_A)
            for end in ENDINGS:
                S.mode, S.alpha = mode, a
                g = generate(tok, model, ctx + end, args.n_gen)
                saidX.append(x in g.lower())
                saidY.append(YS[i] in g.lower())
                demo[(frame, name, i, end)] = g
        row[f"said X, {name}"] = sum(saidX) / len(saidX)
        if "{Y}" in tmpl:
            row[f"said Y, {name}"] = sum(saidY) / len(saidY)
    rows.append(row)

print(f"q* from {len(pairs())} secret-word pairs; {args.n_test} words × {len(ENDINGS)} endings per frame; 'said' = word in the {args.n_gen}-token continuation")
print(tabulate(rows, headers="keys", tablefmt="pipe", floatfmt=".2f", missingval=""))
for (frame, name, i, end), g in demo.items():
    if i == 0 and end == ENDINGS[0]:
        print(f"{frame:32s} | {name:7s} |{end}{g!r}")
