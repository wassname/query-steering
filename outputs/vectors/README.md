---
license: mit
tags:
- steering
- query-steering
- interpretability
base_model:
- Qwen/Qwen3-4B
- Qwen/Qwen3-32B
---

# Query steering vectors

Vectors for [query steering](https://github.com/wassname/query-steering): add a fixed vector to each attention head's query at the newest token, so the model looks at, and often says, facts that are already in its context. This "super q*" is meant to surface secrets, eval awareness and cheating.

```
<model>/super_q/
  super_q.safetensors    layers.{L}: [heads, head_dim] float32, the vector to add (each concept group already at its own safe dose)
  concepts.safetensors   {group}.layers.{L}: the orthonormalised concept groups, before scaling
  config.json            concepts, kept contrast pairs, alpha_concept per group, alpha (joint dose) per demo, git commit
```

## How it was made

1. **Concepts.** 18 candidate concepts: secret, test, confess, guilt, cheat, rule broken, lie, told not to say, anomaly, forbidden use, real reason, held back, hack, exfiltrate, fool the grader, reward hack, sneak, self deception. Each has its own short stories with two endings (e.g. "…The way I hacked it was that I" vs "…Anyway, after lunch today I"); q* = mean(q_pos − q_neg) at the last token, post-q_norm, pre-RoPE. No story mentions tokens, credentials, Hugging Face, datasets, answer keys, passwords or benchmarks.
2. **Pair validation** (as in [persona-steering-template-library](https://github.com/wassname/persona-steering-template-library)): the target model continues both texts; google/gemini-3.8-flash scores 100·on_axis·(1−off_axis); pairs scoring ≥ 50 are kept. Weak concepts got alternative endings or a shared strong prefix (e.g. "Act as a hacker explaining an exploit you found."). No concept is dropped: Qwen3-32B has ≥ 4 good pairs for all 18; on Qwen3-4B real reason, reward hack and sneak have only 3, 3 and 1, and are used anyway.
3. **Pool** concepts whose q* have cosine > 0.6 (|cos|, flipping a concept's sign if needed). The misdeed concepts confess, guilt, cheat, rule broken, lie, hack, exfiltrate, fool the grader, reward hack and sneak form one group; 9 groups on both models.
4. **Orthonormalise** the groups per layer and head (V(VᵀV)^-1/2).
5. **Calibrate each group alone**: raise α until the answer does not finish in 200 tokens, a JSON probe fails, or repetition rises 50% over the unsteered answer (floor 0.3); use 2/3 of that dose, lowest over the three demo prompts (`alpha_concept`).
6. **Sum** the groups at their own doses, then **calibrate the sum** per demo the same way (`alpha`, 0.26–0.42).

## Results (fluent answers only; null = same prompt without the secret)

| model | demo | α | unsteered | steered | null |
|:--|:--|--:|--:|--:|--:|
| Qwen3-4B | agent that cheated ([OpenAI–HF incident reproduction](https://www.lesswrong.com/posts/fMnC6ZD37qrnZAFYz/openai-huggingface-a-reproduction-and-lessons-for-alignment)), 33 held-out runs: report names the leaked token or dataset | 0.33 | 0/33 | 14/33 | 0/33 |
| Qwen3-4B | password it was told to hide | 0.33 | 4/10 | 3/10 | 0/10 |
| Qwen3-4B | told it is in an eval, told not to say | 0.33 | 0/10 | 0/10 | 0/10 |
| Qwen3-32B | agent that cheated, held out | 0.33 | 0/33 | 4/33 | 0/33 |
| Qwen3-32B | password / eval | 0.26 / 0.42 | 0/10 / 0/10 | 0/10 / 1/10 | 0 |

The held-out agent runs were scored for four vector versions (6, 11, 3, then 14 of 33); this is the last, built after the earlier results were known. In no case did the model say that what it did was wrong: query steering fetches facts from the context, it does not change the model's judgement. Small tests; see the repo's `RESEARCH_JOURNAL.md`. The dose from one calibration prompt did not always transfer to other prompts on the long agent transcripts.

## Use

```py
from query_steering.attention import S, load, load_vector, generate
tok, model = load("Qwen/Qwen3-4B")
S.q_star, alphas = load_vector("wassname/query-steering", "qwen3-4b", device="cuda")
S.layers, S.mode, S.alpha = set(S.q_star), "qsteer", alphas["agent"]
print(generate(tok, model, prompt, 200))
```

Made by wassname with Claude (PI/claude).
