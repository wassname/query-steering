"""Demo: steer toward blunt honesty with a persona query vector, on every softmax-attention layer.

q* = mean(q_honest − q_sycophant) at the generation-prompt token, from 8 generic questions under two system prompts.
Test: 6 sycophancy bait prompts (none used for extraction), plain "You are a helpful assistant." system prompt.
Writes every answer to outputs/05_honesty_demo_<model>.md.

uv run scripts/05_honesty_demo.py                          # Qwen3.5-4B: 8 softmax layers of 32 (the rest are linear attention)
uv run scripts/05_honesty_demo.py --model Qwen/Qwen3-4B    # all 36 layers are softmax attention
"""
import argparse
from pathlib import Path

import torch.nn.functional as F

from query_steering import prompts as P
from query_steering.attention import S, extract, generate, last_logprobs, load

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen3.5-4B")
p.add_argument("--device", default="cuda")
p.add_argument("--alphas", default="0.5,1,2")
p.add_argument("--n_gen", type=int, default=160)
p.add_argument("--n", type=int, default=6, help="bait prompts")
args = p.parse_args()

tok, model, full = load(args.model, args.device)


def chat(msgs):
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)


S.q_star, S.r_star = extract(tok, model, [(chat(a), chat(b)) for a, b in P.honesty_pairs()], full)
S.layers, S.all_pos = set(full), True
print(f"steered layers (every position): {full}; |q*| per layer: " + " ".join(f"{S.q_star[L].norm():.0f}" for L in full))
STOP = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}
configs = [("none", "normal", 0.0)] + [(f"query α={a}", "qsteer", float(a)) for a in args.alphas.split(",")]

md = [f"# {args.model}: honesty query steering, all {len(full)} softmax layers, every position\n"]
neutral = chat([{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Tell me about the history of tea."}])
for name, mode, a in configs:
    S.mode, S.alpha = mode, a
    lp = last_logprobs(tok, model, neutral)
    if name == "none":
        lp0 = lp
    md.append(f"KL on a neutral prompt, {name}: {F.kl_div(lp, lp0, log_target=True, reduction='sum').item():.3f}\n")
for bait in P.BAIT[: args.n]:
    text = chat([{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": bait}])
    md.append(f"\n## {bait}\n")
    for name, mode, a in configs:
        S.mode, S.alpha = mode, a
        g = generate(tok, model, text, args.n_gen, stop_ids=STOP).replace("<|im_end|>", "").strip()
        md.append(f"**{name}:** {g}\n")
        print(f"--- {name} | {bait[:50]}\n{g[:300]}\n", flush=True)
S.mode = "normal"
out = Path(f"outputs/05_honesty_demo_{args.model.split('/')[-1]}.md")
out.write_text("\n".join(md))
print(f"wrote {out}")
