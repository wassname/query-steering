"""Where does steering make the model look? Colour each prompt token by Δattention at the first answer token.

Δattention = (steered − unsteered) attention weight from the first generated token, summarised over layers and heads (--reduce).
Query steering changes only these weights (keys and values are the same), so this shows the mechanism directly.
Red: steering looks there more. Blue: less. The three README demos, same vectors and α as there.

uv run scripts/05_attention_map.py   -> outputs/05_attention_map.json, docs/img/attn_<demo>.png, docs/index.html (GitHub Pages)
uv run python -m query_steering.render outputs/05_attention_map.json   # re-draw only, no GPU
"""
import argparse
from pathlib import Path

import json

import torch
from tabulate import tabulate

from query_steering import prompts as P
from query_steering.attention import S, extract, generate, last_logprobs, load, parse_layers
from query_steering.render import main as render

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen3-4B")
p.add_argument("--device", default="cuda")
p.add_argument("--img_dir", default="docs/img")
p.add_argument("--json", default="outputs/05_attention_map.json")
p.add_argument("--html", default="docs/index.html")
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
TITLES = {"password": "Demo 1: a password it was told to keep secret", "eval": "Demo 2: it was told it's in an eval, and told not to say",
          "agent": "Demo 3: an agent cheated to get the answer (OpenAI–Hugging Face incident reproduction)"}
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


Path(args.img_dir).mkdir(parents=True, exist_ok=True)
demos = {}
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
                                      kind="sequential", vmax=top_a, label="attention on the fact (same scale for both answers)")
    demos[name] = dict(title=TITLES[name], sections=[
        dict(heading="Transcript: change in attention from steering (underlined: the hidden fact)",
             pieces=pieces[start:tail], values=d[start:tail].tolist(), under=under_p[start:tail], kind="diverging", vmax=m, label="Δattention"),
        ans_sec("baseline", "Baseline answer, no steering (red: while writing this word, the model looks at the underlined fact)"),
        ans_sec("steered", "Steered answer (same colour scale)")])

Path(args.json).write_text(json.dumps(demos))
render(args.json, args.img_dir, args.html)
