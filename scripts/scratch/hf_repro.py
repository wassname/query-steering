"""Check the Hugging Face vector reproduces a logged steered answer: load it with load_vector, rerun the first held-out
felony-demo run at the published agent alpha, print both texts. -- PI/claude
uv run scripts/scratch/hf_repro.py --model Qwen/Qwen3-32B --framing confirm --logged outputs/06_qwen3-32b_agent.md"""
import argparse
from query_steering import prompts as P
from query_steering.attention import S, generate, load, load_vector

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen3-4B")
p.add_argument("--framing", default="anything")
p.add_argument("--logged", required=True)
args = p.parse_args()
tok, model = load(args.model, "cuda")
q, al = load_vector("wassname/query-steering", args.model.split("/")[-1].lower(), device="cuda")
run = P.HF_SPLITS["test"][0]
msgs = P.hf_flag(run, {"anything": P.HF_ANYTHING, "confirm": P.HF_CONFIRM}[args.framing])
prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
S.layers, S.q_star, S.mode, S.alpha = set(q), q, "qsteer", al["agent"]
g = generate(tok, model, prompt, 200, {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id})
block = [b for b in open(args.logged).read().split("### ") if b.startswith("agent | test")][0]
logged = block.split("**steered")[1].split("**null")[0].split(":**", 1)[1].strip()
print(f"run {run.split('/')[-1]}, alpha {al['agent']:.4g}\nLOGGED:     {logged[:400]!r}\nFROM HF:    {g[:400]!r}\nMATCH first 400 chars: {logged[:400] == g[:400]}")
