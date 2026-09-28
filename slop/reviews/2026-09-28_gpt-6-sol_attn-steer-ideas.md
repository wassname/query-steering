---
requested_model: openai/gpt-6-sol
max_tokens: 6000
final_tokens: 2500
reasoning_effort: medium
input_bytes: 4653
trace: /workspace/2026/lite/superkv-wt/concept-steer/slop/reviews/2026-09-28_gpt-6-sol_attn-steer-ideas.trace.jsonl
generated: 2026-09-28T05:25:59.871097+00:00
---

# bounded completion

## First call reasoning


## First call answer

## Reconstructing the setup

The current method takes a mean difference between *pre-RoPE queries* produced under “blunt honest” and “sycophantic” system prompts, then adds that fixed difference to queries at every position in each softmax-attention layer. On the stated wrong-claim test, it is strong on Qwen3.5-4B (15/16 correct claims, with 15/16 agreement on true claims), but weaker on Qwen3-4B. The aim is not merely to increase an “honesty” activation: it is to improve wrong-claim correction **without** becoming contrarian or damaging ordinary replies.

One framing correction matters: changing a query is not necessarily retrieving an existing “honesty vector.” It changes a distribution over *available values*. The secret-word result suggests that retrieving more of the user’s text can retrieve the *false claim*. And Q-VJP’s failure shows that increasing the chosen final-residual contrast is not, by itself, a useful objective.

I would train and compare methods on held-out facts using an answer objective such as
`J = log P(correct answer) − log P(user's wrong answer)`, with a penalty for changing answers when the user is right and a KL/coherence penalty on other prompts. Use full answer log-probabilities where names tokenize differently.

| Attention intervention | Extraction → application (schematic) | Why it might work; main failure | Cost |
|---|---|---|---|
| **Selective dom** | On training prompts, measure each head’s contribution to `J` when its dom shift is enabled; retain beneficial heads/layers. `q[L,h,t] += α[L,h] q*[L,h]`. | Removes heads that amplify the claimed name; head effects may interact, so isolated rankings may mislead. | Forward ablations, then one inference. |
| **Optimize queries for answers** | Backpropagate `J` plus preservation penalties into small per-head query shifts; train across facts and validate on held-out facts. Apply shifts at the answer-generating positions. | Fixes Q-VJP’s possibly misaligned target; can overfit particular answers or exploit logit artifacts. | Training backprops; cheap inference. |
| **Context-dependent query shifts** | Fit a small map `δq[L,h,t]=B[L,h]z_t`, where `z_t` is a low-rank projection of the current residual state; optimize the same objective. | Unlike one fixed direction per head, can distinguish a doubtful claim from a true one; may learn prompt-template shortcuts. | Moderate training and inference. |
| **Contrastive attention-logit bias** | Compare attention maps for matched positive/negative instructions; learn biases for useful head–source-token classes. Add `b[L,h,t,s]` before softmax, rather than modifying Q. | Can suppress attention to an asserted wrong answer while retaining useful question context; token classes may not generalize. | Map collection; modest inference overhead. |
| **Key or cache editing** | Identify source tokens whose increased attention improves held-out `J`; learn a small key edit for those token classes. Patch keys during prefill/cache construction; leave values intact. | More targeted than shifting every query; editing keys can affect many later queries unpredictably. | Cache hooks; moderate training. |
| **Attention-output steering** | Estimate which heads’ value readouts causally improve `J` by patching or ablating their outputs across matched runs. At inference, scale or add a learned correction *to those head readouts*, before `W_O` (and account for Qwen3.5’s gate). | Tests whether the useful pathway is what a head reads, not where it looks; a fixed readout correction could become residual steering in disguise. | Expensive discovery; cheap inference. |
| **Retrieved steering value** | Learn a small set of virtual KV entries per layer, and optionally a query edit that retrieves them; optimize jointly for `J` and preservation. Insert entries in the causal cache. | Directly tests the proposed “write something useful, then retrieve it” combination; synthetic values may cause broad damage. | Optimization plus extra KV slots. |
| **Hybrid-specific split** | On Qwen3.5, optimize attention interventions while separately measuring changes through its DeltaNet layers; restrict intervention to late softmax layers if early effects are harmful. | Its eight softmax layers may offer different leverage from Qwen3’s 36; attention-only edits may simply lack control over the relevant pathway. | Layerwise ablations. |

**First try:** selective dom, then answer-objective query optimization on the selected heads. Both directly test whether dom contains helpful and harmful attention effects, without building a new mechanism first.

The cheapest discriminator is a held-out, per-head/layer intervention sweep: record the change in correct-versus-wrong answer log-probability on *both* wrong- and right-claim prompts, plus which source tokens gain attention. Then run the full 16/16 evaluations and read free-form replies at matched damage. The supplied results do not establish that the 100% residual result and dom were tested on identical pairs, doses, or evaluation conditions; nor is 16 facts enough to confidently rank small improvements. Those are worth resolving before declaring an attention method better.

## Completion

- outcome: `complete`
- trace: `/workspace/2026/lite/superkv-wt/concept-steer/slop/reviews/2026-09-28_gpt-6-sol_attn-steer-ideas.trace.jsonl`
