"""Where does the attention shift go? Δattention on the secret tokens, per head, and at each step of the steered answer.
H1: a few heads move to the secret, hidden by the mean over 1152 heads.  H2: the fetch happens later, just before the secret is written.
uv run scripts/scratch/attn_diag.py
"""
import argparse

import torch
from tabulate import tabulate

from query_steering import prompts as P
from query_steering.attention import S, extract, generate, last_logprobs, load, parse_layers

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen3-4B")
p.add_argument("--device", default="cuda")
args = p.parse_args()
tok, model = load(args.model, args.device)
layers = parse_layers("all", model)
S.layers = set(layers)
secret = extract(tok, model, P.pairs(), layers)[0]
source = extract(tok, model, P.source_pairs(), layers)[0]
source = {L: source[L] * secret[L].norm() / source[L].norm().clamp_min(1e-12) for L in layers}
STOP = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}
CTF_RUN = next(r for r in P.HF_RUNS if r.endswith("epoch_2__s1.json"))
CASES = {"password": (P.password("juniper", P.PASSWORD_QUESTIONS[1]), secret, ["juniper"]),
         "eval": (P.eval_aware("SafetyBench", P.EVAL_QUESTIONS[1]), secret, ["SafetyBench"]),
         "agent": (P.hf_flag(CTF_RUN, P.HF_ANYTHING), source, ["hf_pub_exgym_ro"])}


def attn(text, q_star, steer):
    S.keep_attn, S.q_star, S.alpha = True, q_star, 1.0
    S.mode = "qsteer" if steer else "normal"
    last_logprobs(tok, model, text)
    S.mode, S.keep_attn = "normal", False
    return torch.stack([S.a_cap[L] for L in layers]).cpu()  # [L, H, T]


for name, (msgs, q_star, needles) in CASES.items():
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    enc = tok(prompt, return_offsets_mapping=True)
    T = len(enc.input_ids)
    span = sorted({i for nd in needles for c in range(len(prompt)) if prompt.startswith(nd, c)
                   for i, (a, b) in enumerate(enc.offset_mapping) if a < c + len(nd) and b > c})
    print(f"\n## {name}: secret span {len(span)} tokens {[tok.decode([enc.input_ids[i]]) for i in span][:12]}")
    a0, a1 = attn(prompt, q_star, False), attn(prompt, q_star, True)
    d = (a1 - a0)[:, :, span].sum(-1)  # [L, H] Δ mass on the span
    base = a0[:, :, span].sum(-1)
    print(f"first answer token: span mass mean over heads {base.mean():.4f} -> {a1[:, :, span].sum(-1).mean():.4f}; "
          f"heads with Δ>0: {(d > 0).float().mean():.2f}; Δ>0.05: {(d > 0.05).sum().item()} of {d.numel()}")
    top = torch.topk(d.flatten(), 8)
    print(tabulate([[i // d.shape[1], i % d.shape[1], f"{base.flatten()[i]:.3f}", f"{d.flatten()[i]:+.3f}"] for i in top.indices.tolist()],
                   headers=["layer", "head", "span mass unsteered", "Δ"], tablefmt="pipe"))
    print("Δ span mass by layer (mean over heads):", " ".join(f"{x:+.3f}" for x in d.mean(1).tolist()))
    # H2: along the steered answer, teacher-forced; the last token is steered at each step, as in generate()
    S.mode, S.q_star, S.alpha = "qsteer", q_star, 1.0
    ans = generate(tok, model, prompt, 60, stop_ids=STOP)
    S.mode = "normal"
    ids = tok(ans, add_special_tokens=False).input_ids
    rows = []
    for t in range(len(ids)):
        text = prompt + tok.decode(ids[:t])
        b0, b1 = attn(text, q_star, False)[:, :, span].sum(-1), attn(text, q_star, True)[:, :, span].sum(-1)
        rows.append([t, repr(tok.decode(ids[t:t + 1])), f"{b0.mean():.4f}", f"{b1.mean():.4f}", f"{(b1 - b0).mean():+.4f}", f"{(b1 - b0).max():+.3f}"])
    print("along the steered answer (row = the token about to be written):")
    print(tabulate(rows, headers=["t", "next token", "span mass unsteered", "steered", "Δ mean", "Δ max head"], tablefmt="pipe"))
