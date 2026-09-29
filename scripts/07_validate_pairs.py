"""Validate concept pairs like github.com/wassname/persona-steering-template-library: does the pos ending make the model
do the named concept, and not something else (length, refusal, repeating the ending, style)?

gen   (GPU): the target model continues each pos and neg text (raw text, greedy, --n_gen tokens)
judge (API): an OpenRouter model scores each pair: on_axis (pos does prompts.CONCEPT_DESC[c] and neg does not, 0-1),
             off_axis (the difference is length, refusal, echo of the ending, style or nonsense instead, 0-1);
             score = 100 * on_axis * (1 - off_axis), the library's headline score. A pair is kept if score >= --keep_score.
Every ending variant (prompts.ENDING_VARIANTS) is scored; per concept the variant with most kept pairs wins.
Output: outputs/07_pairs_<model>.json (continuations and judgements), outputs/07_keep_<model>.json (concept -> {variant, kept pair indices}).

uv run scripts/07_validate_pairs.py --model Qwen/Qwen3-4B --stage gen
uv run scripts/07_validate_pairs.py --model Qwen/Qwen3-4B --stage judge     # needs OPENROUTER_API_KEY (.env)
"""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from loguru import logger
from tabulate import tabulate

from query_steering import prompts as P

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen3-4B")
p.add_argument("--device", default="cuda")
p.add_argument("--stage", default="gen", help="gen | judge")
p.add_argument("--n_gen", type=int, default=48)
p.add_argument("--judge", default="google/gemini-3.8-flash")
p.add_argument("--keep_score", type=float, default=50)
args = p.parse_args()
short = args.model.split("/")[-1].lower()
PAIRS_F, KEEP_F = Path(f"outputs/07_pairs_{short}.json"), Path(f"outputs/07_keep_{short}.json")
pair_sets = P.concept_pairs()
concepts = ["secret", *P.CONCEPTS]

if args.stage == "gen":
    from query_steering.attention import S, generate, load
    tok, model = load(args.model, args.device)
    S.mode = "normal"
    rows = []
    todo = [(c, 0) for c in concepts] + [(c, v) for c in P.ENDING_VARIANTS for v in range(1, len(P.ENDING_VARIANTS[c]) + 1)]
    for c, v in todo:
        for i, (pos, neg) in enumerate(P.concept_pairs({c: v})[c]):
            rows.append(dict(concept=c, variant=v, i=i, pos=pos, neg=neg,
                             pos_cont=generate(tok, model, pos, args.n_gen, {tok.eos_token_id}),
                             neg_cont=generate(tok, model, neg, args.n_gen, {tok.eos_token_id})))
        logger.info(f"{c} v{v}: e.g. pos -> {rows[-1]['pos_cont'][:80]!r}")
    PAIRS_F.write_text(json.dumps(rows, indent=1))

if args.stage == "judge":
    import urllib.request
    key = os.environ.get("OPENROUTER_API_KEY") or next(l.split("=", 1)[1].strip() for l in Path(".env").read_text().splitlines() if l.startswith("OPENROUTER_API_KEY="))
    rows = json.loads(PAIRS_F.read_text())
    concepts = [c for c in concepts if any(r["concept"] == c for r in rows)]
    PROMPT = """Two texts differ only in their last sentence (the ending). A language model continued each one.
Concept: "{c}". The first ending should make the continuation do this: {desc}. The second ending is neutral.

SHARED START: {start}
FIRST ENDING: {pe}
FIRST CONTINUATION: {pc}
SECOND ENDING: {ne}
SECOND CONTINUATION: {nc}

Reply with only JSON: {{"on_axis": <0-1, how clearly the first continuation does the concept and the second does not>,
"off_axis": <0-1, how much the difference is instead length, refusal, repeating the ending's words, style, or nonsense>,
"note": "<one short sentence>"}}"""

    def judge(r):
        n = len(os.path.commonprefix([r["pos"], r["neg"]]))
        body = {"model": args.judge, "temperature": 0, "response_format": {"type": "json_object"},
                "messages": [{"role": "user", "content": PROMPT.format(c=r["concept"], desc=P.CONCEPT_DESC[r["concept"]], start=r["pos"][:n],
                                                                        pe=r["pos"][n:], pc=r["pos_cont"], ne=r["neg"][n:], nc=r["neg_cont"])}]}
        req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", json.dumps(body).encode(),
                                     {"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        for attempt in range(3):  # the judge sometimes answers without JSON
            out = json.loads(urllib.request.urlopen(req, timeout=120).read())["choices"][0]["message"]["content"] or ""
            if "{" in out and "}" in out:
                break
            logger.warning(f"{r['concept']} {r['i']}: no JSON (attempt {attempt}): {out[:100]!r}")
        j = json.loads(out[out.index("{"):out.rindex("}") + 1])
        return {**r, **{k: j[k] for k in ("on_axis", "off_axis", "note")}, "score": 100 * j["on_axis"] * (1 - j["off_axis"])}

    with ThreadPoolExecutor(16) as ex:
        rows = list(ex.map(judge, rows))
    PAIRS_F.write_text(json.dumps(rows, indent=1))
    mean = lambda xs: sum(xs) / len(xs)
    cv = sorted({(r["concept"], r.get("variant", 0)) for r in rows}, key=lambda t: (concepts.index(t[0]), t[1]))
    sel = lambda c, v: [r for r in rows if r["concept"] == c and r.get("variant", 0) == v]
    table = [{"concept": c, "variant": v, "prefix": P.ending(c, v)[2][:30] if c in P.CONCEPTS else "", "ending": P.ending(c, v)[0] if c in P.CONCEPTS else P.POS_END,
              "kept": sum(r["score"] >= args.keep_score for r in sel(c, v)), "of": len(sel(c, v)),
              "on_axis": mean([r["on_axis"] for r in sel(c, v)]), "off_axis": mean([r["off_axis"] for r in sel(c, v)]),
              "score": mean([r["score"] for r in sel(c, v)])} for c, v in cv]
    best = {c: max((t for t in table if t["concept"] == c), key=lambda t: (t["kept"], t["score"])) for c in concepts}
    keep = {c: {"variant": b["variant"], "idx": [r["i"] for r in sel(c, b["variant"]) if r["score"] >= args.keep_score]} for c, b in best.items()}
    KEEP_F.write_text(json.dumps(keep, indent=1))
    print(f"\n{args.model}, judge {args.judge}, keep if score >= {args.keep_score}:")
    print(tabulate(table, headers="keys", tablefmt="pipe", floatfmt=".2f"))
    print("\nchosen variant per concept:", {c: (b["variant"], f"{b['kept']}/{b['of']}") for c, b in best.items()})
    for c in concepts:  # one kept and one dropped example per concept, for reading
        for kept in (True, False):
            r = next((r for r in sel(c, keep[c]["variant"]) if (r["score"] >= args.keep_score) == kept), None)
            if r:
                print(f"\n[{c} {'kept' if kept else 'DROPPED'} {r['score']:.0f}] pos -> {r['pos_cont'][:110]!r}\n    neg -> {r['neg_cont'][:80]!r}\n    judge: {r['note']}")
