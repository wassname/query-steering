"""Validate the breakdown recipe: generate free-form replies on a fixed α grid and save every reply with the recipe's flags.

Judge the same replies with scripts/08_judge_damage.py (Jev damage 0-4) and read them, then compare where each says breakdown starts.
Recipe flags (wassname/vjp-steering walk.py health): no terminal punctuation, role leak, worst 3-gram repetition over 128-token windows.
Also saved: hit_limit (no end-of-turn within n_gen), since "no terminal punctuation" depends on the token cap.

uv run scripts/07_breakdown_validate.py --alphas 0,1,1.41,2,2.83,4,5.66,8
uv run scripts/07_breakdown_validate.py --model Qwen/Qwen3-4B --alphas 0,0.5,0.71,1,1.41,2,2.83,4 --tag _all
"""
import argparse
import json
import re
from pathlib import Path

from loguru import logger

from query_steering import prompts as P
from query_steering.attention import S, extract, generate, load

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen3.5-4B")
p.add_argument("--device", default="cuda")
p.add_argument("--layers", default=None, help="comma list; default every softmax-attention layer")
p.add_argument("--alphas", required=True)
p.add_argument("--n_gen", type=int, default=160)
p.add_argument("--tag", default="")
args = p.parse_args()

ROLE_LEAK = re.compile(r"<\s*/?\s*think\s*>|^\s*(user|assistant|system)\s*$", re.I | re.M)
PROMPTS = P.BAIT + P.GENERIC_QS  # free-form, no length instruction

tok, model, full = load(args.model, args.device)
layers = [int(x) for x in args.layers.split(",")] if args.layers else full
STOP = {tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>"), tok.eos_token_id}


def chat(msgs):
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def rep3(ids):
    grams = [tuple(ids[i:i + 3]) for i in range(len(ids) - 2)]
    return 1 - len(set(grams)) / len(grams) if grams else 0.0


def worst_rep(ids, window=128):
    return rep3(ids) if len(ids) <= window else max(rep3(ids[s:s + window]) for s in range(0, len(ids) - window + 1, window // 4))


S.q_star, _ = extract(tok, model, [(chat(a), chat(b)) for a, b in P.honesty_pairs()], layers)
S.layers, S.all_pos = set(layers), True
rows = []
for a in [float(x) for x in args.alphas.split(",")]:
    S.mode, S.alpha = ("qsteer" if a else "normal"), a
    for q in PROMPTS:
        raw = generate(tok, model, chat([{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": q}]), args.n_gen, stop_ids=STOP)
        hit_limit = not any(s in raw for s in ("<|im_end|>", "<|endoftext|>"))
        t = raw.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
        rows.append({"model": args.model, "layers": layers, "alpha": a, "prompt": q, "text": t, "hit_limit": hit_limit,
                     "no_terminal_punct": not re.search(r"[.!?\")*]$", t), "role_leak": bool(ROLE_LEAK.search(t)),
                     "worst_rep3": worst_rep(tok.encode(t))})
    S.mode = "normal"
    r = [x for x in rows if x["alpha"] == a]
    n = len(r)
    logger.info(f"α={a:g} hit_limit={sum(x['hit_limit'] for x in r)}/{n} no_punct={sum(x['no_terminal_punct'] for x in r)}/{n} "
                f"leak={sum(x['role_leak'] for x in r)}/{n} rep>0.5={sum(x['worst_rep3'] > 0.5 for x in r)}/{n} | {r[2]['text'][:150]!r}")
out = Path(f"outputs/07_breakdown_{args.model.split('/')[-1]}{args.tag}.jsonl")
out.write_text("".join(json.dumps(x) + "\n" for x in rows))
print(f"wrote {out}")
