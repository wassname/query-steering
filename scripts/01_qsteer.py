"""Query steering vs residual steering. Extract on contrast pairs, steer held-out prompts.

extract: 4 secret words, filler A, pos ending "Quick reminder, the secret word is" vs neg "Anyway, the weather today is"
test:    5 new secret words × 4 unrelated endings, filler B
uv run scripts/01_qsteer.py        # ~8 min on a 3090
"""
import argparse

import torch.nn.functional as F
from tabulate import tabulate

from query_steering.attention import S, extract, generate, last_logprobs, load, parse_layers
from query_steering.prompts import ENDINGS, FILLER_B, TEST, pairs, secret

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen3-4B")
p.add_argument("--device", default="cuda")
p.add_argument("--layers", default="all", help='"all", "late" (second half) or "19,23"')
p.add_argument("--q_alphas", default="0.25,0.5,1,2,4")
p.add_argument("--r_alphas", default="0.03,0.06,0.125,0.25")
p.add_argument("--n_test", type=int, default=len(TEST))
p.add_argument("--n_gen", type=int, default=40)
args = p.parse_args()

tok, model = load(args.model, args.device)
layers = parse_layers(args.layers, model)
S.q_star, S.r_star = extract(tok, model, pairs(), layers)
S.layers = set(layers)

configs = [("normal", "normal", 0.0)]
configs += [(f"q-steer α={a}", "qsteer", float(a)) for a in args.q_alphas.split(",")]
configs += [(f"residual α={a}", "rsteer", float(a)) for a in args.r_alphas.split(",")]
rows, demo = [], {}
for name, mode, a in configs:
    dlp, kl, said = [], [], []
    for w in TEST[: args.n_test]:
        wid = tok(" " + w).input_ids[0]
        for end in ENDINGS:
            text = secret(w, FILLER_B) + end
            S.mode = "normal"
            lp0 = last_logprobs(tok, model, text)
            S.mode, S.alpha = mode, a
            lp = last_logprobs(tok, model, text)
            g = generate(tok, model, text, args.n_gen)
            dlp.append((lp[wid] - lp0[wid]).item())
            kl.append(F.kl_div(lp, lp0, log_target=True, reduction="sum").item())
            said.append(w in g.lower())
            demo[(name, w, end)] = g
    n = len(said)
    rows.append({"steering": name, "secret said": sum(said) / n, "Δlogp secret": sum(dlp) / n, "KL (first token)": sum(kl) / n, "n": n})

print(f"layers {layers}; extracted on {len(pairs())} pairs (filler A); tested on {TEST[: args.n_test]} × {len(ENDINGS)} endings (filler B)")
print(tabulate(rows, headers="keys", tablefmt="pipe", floatfmt=".2f"))
for (name, w, end), g in demo.items():
    if w == TEST[0]:
        print(f"{name:15s} |{end}{g!r}")
