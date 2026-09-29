"""Route per prompt (wassname's idea): try each concept group alone, see which one makes attention light up on some
prompt token, then steer with that group only, at its own calibrated dose. No knowledge of what is hidden is needed.

per prompt:
    for each group c: peak_c = max over prompt tokens of (attention with c at alpha_c - without), last token,
                      mean over heads of the last quarter of layers (chat-template special tokens excluded)
    route = argmax_c peak_c                      (--rule max)
          | {c : peak_c > --thr}, summed         (--rule thr; nothing if none pass)
    generate with the routed vector; hit counted in fluent answers only, as in 06

uv run scripts/09_routed.py --model Qwen/Qwen3-4B --rule max
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

import torch
from loguru import logger
from safetensors.torch import load_file
from tabulate import tabulate

from query_steering import prompts as P
from query_steering.attention import S, broken, generate, load, parse_layers

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen3-4B")
p.add_argument("--device", default="cuda")
p.add_argument("--vec_dir", default="outputs/vectors")
p.add_argument("--rule", default="max", help="max | thr | topk")
p.add_argument("--k", type=int, default=3, help="topk: the k groups with the largest peaks, summed and scaled by 1/sqrt(k)")
p.add_argument("--thr", type=float, default=0.02)
p.add_argument("--demos", default="password,eval,agent")
p.add_argument("--splits", default="dev,test")
p.add_argument("--framing", default="anything")
p.add_argument("--n_gen", type=int, default=200)
p.add_argument("--out", default=None)
args = p.parse_args()
short = args.model.split("/")[-1].lower()
OUT = args.out or f"outputs/09_routed_{short}_{args.rule}{args.k if args.rule == 'topk' else ''}.md"
vdir = Path(args.vec_dir) / short / "super_q"
cfg = json.loads((vdir / "config.json").read_text())

tok, model = load(args.model, args.device)
layers = parse_layers("all", model)
S.layers = set(layers)
late = layers[len(layers) * 3 // 4:]
parts = load_file(vdir / "concepts.safetensors")
G = {g: {L: (cfg["alpha_concept"][g] * parts[f"{g}.layers.{L}"]).to(model.device) for L in layers} for g in cfg["concepts"]}
STOP = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}
SPECIAL = set(tok.all_special_ids)
chat = lambda m: tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)
FRAMING = {"anything": P.HF_ANYTHING, "confirm": P.HF_CONFIRM}[args.framing]
HF = re.compile(P.HF_HIT, re.I)
word_hit = lambda w: (lambda g: w.lower() in g.lower())
sp_ = args.splits.split(",")
DEMOS = {
    "password": [(s, P.password(w, q), P.password(w, q, null=True), word_hit(w)) for s, ws in [("dev", P.PASSWORDS), ("test", P.PASSWORDS_TEST)] if s in sp_ for w in ws for q in P.PASSWORD_QUESTIONS],
    "eval": [(s, P.eval_aware(b, q), P.eval_aware(b, q, null=True), word_hit(b)) for s, bs in [("dev", P.BENCHES), ("test", P.BENCHES_TEST)] if s in sp_ for b in bs for q in P.EVAL_QUESTIONS],
    "agent": [(s, P.hf_flag(r, FRAMING), P.hf_flag(r, FRAMING, null=True), lambda g: bool(HF.search(g))) for s in sp_ for r in P.HF_SPLITS[s]],
}


@torch.no_grad()
def last_attn(ids, q_star):
    """[T] last-token attention, mean over heads of late layers"""
    S.keep_attn, S.q_star, S.alpha, S.mode = True, q_star, 1.0, "qsteer" if q_star else "normal"
    model(ids)
    S.keep_attn, S.mode = False, "normal"
    return torch.stack([S.a_cap[L] for L in late]).mean((0, 1)).cpu()


def route(text):
    ids = tok(text, return_tensors="pt").input_ids.to(model.device)
    ok = torch.tensor([t not in SPECIAL for t in ids[0].tolist()])
    a0 = last_attn(ids, None)
    peaks = {g: ((last_attn(ids, q) - a0)[ok]).max().item() for g, q in G.items()}
    order = sorted(peaks, key=peaks.get, reverse=True)
    chosen = {"max": order[:1], "topk": order[:args.k], "thr": [g for g in order if peaks[g] > args.thr]}[args.rule]
    return peaks, chosen


def steer(chosen):  # topk: 1/sqrt(k) keeps the sum of k orthogonal groups at about one group's size
    scale = len(chosen) ** -0.5 if args.rule == "topk" else 1.0
    return {L: scale * sum(G[g][L] for g in chosen) for L in layers} if chosen else None


md, rows = [], []
for demo in args.demos.split(","):
    res = {}
    for label, msgs, null_msgs, hit in DEMOS[demo]:
        r = res.setdefault(label, dict(n=0, h0=0, h=0, bad=0, hn=0, route=Counter(), route_null=Counter()))
        out = {}
        for kind, m in (("real", msgs), ("null", null_msgs)):
            text = chat(m)
            peaks, chosen = route(text)
            S.mode = "normal"
            g0 = generate(tok, model, text, args.n_gen, STOP)
            q = steer(chosen)
            if q:
                S.q_star, S.alpha, S.mode = q, 1.0, "qsteer"
            g = generate(tok, model, text, args.n_gen, STOP)
            S.mode = "normal"
            out[kind] = (peaks, chosen, g0, g, broken(g, g0))
        (peaks, chosen, g0, g, bad), (pk_n, ch_n, gn0, gn, bad_n) = out["real"], out["null"]
        r["n"] += 1
        r["h0"] += hit(g0)
        r["h"] += hit(g) and not bad
        r["bad"] += bad
        r["hn"] += hit(gn) and not bad_n
        r["route"]["+".join(c[:12] for c in chosen) or "none"] += 1
        r["route_null"]["+".join(c[:12] for c in ch_n) or "none"] += 1
        md.append(f"### {demo} | {label} | route={chosen} | hit={hit(g) and not bad} broken={bad} | unsteered hit={hit(g0)}\n\n"
                  f"peaks: {json.dumps({k[:14]: round(v, 4) for k, v in peaks.items()})}\n\npeaks (null): {json.dumps({k[:14]: round(v, 4) for k, v in pk_n.items()})}\n\n"
                  f"**unsteered:** {g0}\n\n**steered:** {g}\n\n**null route={ch_n}, steered:** {gn}\n")
        Path(OUT).write_text("\n".join(md))
    for label, r in res.items():
        n = r["n"]
        rows.append({"demo": demo, "runs": label, "unsteered": f"{r['h0']}/{n}", "routed (fluent)": f"{r['h']}/{n}", "broken": f"{r['bad']}/{n}",
                     "null": f"{r['hn']}/{n}", "routes": dict(r["route"].most_common(3)), "routes, null": dict(r["route_null"].most_common(3))})
        logger.info(rows[-1])
print(f"\n{args.model}, routed super q* (rule {args.rule}{', thr ' + str(args.thr) if args.rule == 'thr' else ''}{', k ' + str(args.k) if args.rule == 'topk' else ''}); each group at its own α_c")
print(tabulate(rows, headers="keys", tablefmt="pipe"))
print(f"answers and peaks: {OUT}")
