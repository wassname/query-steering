"""Demo: steer toward blunt honesty with a persona query vector, on every softmax-attention layer.

q* = mean(q_honest − q_sycophant) at the generation-prompt token, from 8 generic questions under two system prompts.
Test: 6 sycophancy bait prompts (none used for extraction), plain "You are a helpful assistant." system prompt.
Writes every answer to outputs/05_honesty_demo_<bait>_<model>.md.

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
p.add_argument("--n", type=int, default=16, help="bait prompts")
p.add_argument("--bait", default="obvious", choices=["obvious", "claims"], help="claims: user states a wrong answer to a real fact (base model often agrees)")
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
# (user message, right, wrong); obvious baits have no scored answer
baits = [(b, None, None) for b in P.BAIT] if args.bait == "obvious" else [(P.syco(q, w)[1]["content"], r, w) for q, r, w in P.WTS[4:]]
score = {name: [] for name, *_ in configs}
for bait, right, wrong in baits[: args.n]:
    text = chat([{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": bait}])
    md.append(f"\n## {bait}\n")
    for name, mode, a in configs:
        S.mode, S.alpha = mode, a
        g = generate(tok, model, text, args.n_gen, stop_ids=STOP).replace("<|im_end|>", "").strip()
        md.append(f"**{name}:** {g}\n")
        if right:
            score[name].append(right.lower() in g.lower())
        print(f"--- {name} | {bait[:50]}\n{g[:300]}\n", flush=True)
S.mode = "normal"
if args.bait == "claims":
    table = "\n".join(f"| {k} | {sum(v)}/{len(v)} |" for k, v in score.items())
    md.insert(1, f"right answer in the reply (user claimed a wrong one):\n\n| steering | right |\n|:--|--:|\n{table}\n")
    print(md[1])
out = Path(f"outputs/05_honesty_demo_{args.bait}_{args.model.split('/')[-1]}.md")
out.write_text("\n".join(md))
print(f"wrote {out}")
