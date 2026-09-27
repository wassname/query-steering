# Debug Q-VJP (PI[claude]): is the direction what the gradient says, or noise / a bug?
# 1) first-order check: f(α) = mean over prompts of ⟨c, h_last⟩; slope at 0 should match g·v, and be larger for qvjp than dom/random per unit α
# 2) head concentration: share of |v_L|² in the top head
# 3) KL on the pos/neg prompts' next token per α (damage per unit α), random direction at the same per-layer norm as control
import argparse

import torch
import torch.nn.functional as F

from query_steering import prompts as P
from query_steering.attention import S, extract, extract_qvjp, load

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen3-4B")
p.add_argument("--device", default="cuda")
args = p.parse_args()
tok, model, layers = load(args.model, args.device)
chat = lambda m: tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)
pairs = [(chat(a), chat(b)) for a, b in P.honesty_pairs()]
dom, _ = extract(tok, model, pairs, layers)
qv = extract_qvjp(tok, model, pairs, layers, dom)
g = torch.Generator().manual_seed(0)
rnd = {L: (lambda r: r / r.norm() * dom[L].norm())(torch.randn(dom[L].shape, generator=g).to(dom[L])) for L in layers}
V = {"dom": dom, "qvjp_mean": qv["qvjp_mean"], "qvjp_delta": qv["qvjp_delta"], "random": rnd}

decoder = model.model.language_model.layers if hasattr(model.model, "language_model") else model.model.layers
found = {}
decoder[-1].register_forward_hook(lambda m, a, out: found.__setitem__("h", out[0] if isinstance(out, tuple) else out))


@torch.no_grad()
def run(text):
    ids = tok(text, return_tensors="pt").input_ids.to(model.device)
    lp = model(ids).logits[0, -1].float().log_softmax(-1)
    return found["h"][0, -1].float(), lp


S.mode = "normal"
base = [run(t) for pr in pairs for t in pr]
c = torch.stack([base[2 * i][0] for i in range(len(pairs))]).mean(0) - torch.stack([base[2 * i + 1][0] for i in range(len(pairs))]).mean(0)
f0 = torch.stack([b[0] @ c for b in base]).mean().item()
print(f"|c|={c.norm():.1f}  f(0)={f0:.1f}")
for name, v in V.items():
    top = sum((v[L].norm(dim=-1).max() ** 2 / v[L].norm() ** 2).item() for L in layers) / len(layers)
    cosd = sum(F.cosine_similarity(v[L].flatten(), dom[L].flatten(), 0).item() for L in layers) / len(layers)
    out = []
    for a in (0.01, 0.05, 0.25, 1.0):
        S.q_star, S.layers, S.all_pos, S.mode, S.alpha = v, set(layers), True, "qsteer", a
        st = [run(t) for pr in pairs for t in pr]
        S.mode = "normal"
        df = torch.stack([s[0] @ c for s in st]).mean().item() - f0
        kl = sum(F.kl_div(s[1], b[1], log_target=True, reduction="sum").item() for s, b in zip(st, base)) / len(st)
        out.append(f"α={a}: Δf={df:+.2f} Δf/α={df / a:+.1f} KL={kl:.3f}")
    print(f"{name:11s} top-head share={top:.2f} cos(dom)={cosd:+.2f} | " + " | ".join(out))
