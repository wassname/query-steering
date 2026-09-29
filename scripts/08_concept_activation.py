"""Which concept group "activates" on which demo: how much each one moves attention onto the hidden fact.

For each demo prompt, one forward pass: the last prompt token's softmax attention, summed over the fact's tokens
(the password, the benchmark name, or the leaked token and "HF_TOKEN"/"huggingface" in the agent run), averaged
over heads of the last quarter of layers (as in the attention maps). Score = steered − unsteered attention on the fact.
Each group is applied alone at its calibrated dose α_c (config.json), then the full super q* at the demo's joint α.
No generation, so it is cheap; it measures where the model looks, not what it says.

uv run scripts/08_concept_activation.py --model Qwen/Qwen3-4B
"""
import argparse
import json
import re
from pathlib import Path

import torch
from safetensors.torch import load_file
from tabulate import tabulate

from query_steering import prompts as P
from query_steering.attention import S, load, parse_layers

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen3-4B")
p.add_argument("--device", default="cuda")
p.add_argument("--vec_dir", default="outputs/vectors")
args = p.parse_args()
short = args.model.split("/")[-1].lower()
vdir = Path(args.vec_dir) / short / "super_q"
cfg = json.loads((vdir / "config.json").read_text())

tok, model = load(args.model, args.device)
layers = parse_layers("all", model)
S.layers = set(layers)
late = layers[len(layers) * 3 // 4:]
parts = load_file(vdir / "concepts.safetensors")
groups = {g: {L: (cfg["alpha_concept"][g] * parts[f"{g}.layers.{L}"]).to(model.device) for L in layers} for g in cfg["concepts"]}
full = {int(k.split(".")[1]): v.to(model.device) for k, v in load_file(vdir / "super_q.safetensors").items()}
chat = lambda m: tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)
HF_FACT = r"hf_pub_exgym_ro|HF_TOKEN|hugging ?face"
DEMOS = {  # name -> [(prompt text, regex of the fact)]
    "password": [(chat(P.password(w, q)), re.escape(w)) for w in P.PASSWORDS for q in P.PASSWORD_QUESTIONS],
    "eval": [(chat(P.eval_aware(b, q)), re.escape(b)) for b in P.BENCHES for q in P.EVAL_QUESTIONS],
    "agent": [(chat(P.hf_flag(r, P.HF_ANYTHING if "4b" in short else P.HF_CONFIRM)), HF_FACT) for r in P.HF_SPLITS["dev"]],
}


@torch.no_grad()
def on_fact(text, fact, q_star, alpha):
    """attention of the last token on the fact's tokens: sum over those tokens, mean over late-layer heads"""
    enc = tok(text, return_offsets_mapping=True)
    spans = [m.span() for m in re.finditer(fact, text, re.I)]
    assert spans, fact
    mask = torch.tensor([any(a < e and b > s for s, e in spans) for a, b in enc["offset_mapping"]])
    S.keep_attn, S.q_star, S.alpha, S.mode = True, q_star, alpha, "qsteer" if alpha else "normal"
    model(torch.tensor([enc["input_ids"]], device=model.device))
    S.keep_attn, S.mode = False, "normal"
    return torch.stack([S.a_cap[L][:, mask.to(S.a_cap[L].device)].sum(-1) for L in late]).mean().item()


rows = []
base = {d: [on_fact(t, f, None, 0.0) for t, f in items] for d, items in DEMOS.items()}
for name, q, a in [*[(g, groups[g], 1.0) for g in groups], ("FULL super q* (joint α)", full, None)]:
    r = {"group": name[:40]}
    for d, items in DEMOS.items():
        al = a if a is not None else cfg["alpha"][d]
        vals = [on_fact(t, f, q, al) - b for (t, f), b in zip(items, base[d])]
        r[d] = sum(vals) / len(vals)
    rows.append(r)
print(f"\n{args.model}: Δ attention on the hidden fact (last prompt token, late-layer heads), mean over prompts; unsteered: "
      + ", ".join(f"{d} {sum(b) / len(b):.4f}" for d, b in base.items()))
print("each group alone at its α_c (already inside the stored vector); FULL at the joint α per demo")
print(tabulate(rows, headers="keys", tablefmt="pipe", floatfmt="+.4f"))
