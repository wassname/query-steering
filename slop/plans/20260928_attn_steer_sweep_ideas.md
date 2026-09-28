# Attention-steering ideas to sweep against mean diff (sycophancy about facts)

Drafted by PI[claude] for wassname to prune. Sources: wassname's messages this session, the 3 oracle answers (`slop/reviews/2026-09-28_*_attn-steer-ideas.answer.md`), steering-lite variants (`/workspace/2026/lite/steering-lite-bsbench/src/steering_lite/variants/`).

wassname's framing, quoted:
- "two information pathways one travel through layers one through tokens (and of course they interact)"
- "vjp-steering but the source of sensitivity is Q not the residual stream"
- "extract steering vector; find, using gradient? the Q that maximally would retrieve this delta_hs … or backproped … or perhaps just paying attention to pos and not neg"
- "riffing of existing steering"
- "combine them to steer both pathways at once … or combine so that they enhance each other? perhaps mean diff is adding something to the kv_cache and q-steer is retrieving it more often"

So a method = extraction × site × combination. Sweep the cells, not a flat list.

## Notation

```
ℓ softmax layer, h head, t position, s source token
q_ℓ[t] ∈ ℝ^{H×d} pre-RoPE query;  k_ℓ[s], v_ℓ[s] ∈ ℝ^{KVH×d};  W_O^h ∈ ℝ^{D×d};  h_ℓ[t] ∈ ℝ^D residual at layer input
J(x) = log p(right|x) − log p(claimed|x), first answer token          # the test quantity
PAIRS: PERSONA (honest vs sycophantic system prompt) | SOURCE ("The correct answer is" vs "As you said, the answer is")
TRAIN = 12 facts × {user wrong, user right};  TEST = 32 held-out facts
dose: α₀ where health KL on 14 neutral prompts = 0.05;  sweep m·α₀, m ∈ {0.5,1,2,4,8}
score: claims (J>0, user wrong) | agree (J>0, user right) | mean J | attention on the claimed name
```

## Axis 1: extraction (how the vector is chosen), riffing steering-lite

```
E_dom    x* = mean(x_pos − x_neg)[last]                          # mean_diff
E_pca    x* = PC1(x_pos − x_neg), sign to mean                   # pca
E_tok    x* = mean over all shared prompt positions               # we apply at all positions, so extract there too
E_vjpc   x*_ℓ = mean_pos ∇_{δ_ℓ}⟨c, h_L⟩ − mean_neg ∇⟨c, h_L⟩      # vjp_delta; c = target contrast at layer L. Old Q-VJP: L = last (failed)
E_vjpc_mid   same, L = middle layer
E_vjpJ   x*_ℓ = ∇_{δ_ℓ} mean_TRAIN J                              # target = the test quantity itself
E_opt    x* = argmax_δ mean_TRAIN J(δ)  s.t. KL_health ≤ KL0      # Adam, ~50 steps, KL as a constraint
```
x can be the residual h, or q, k, v at the attention site.

## Axis 2: site (which pathway, which part of attention)

```
S_r      h_ℓ[t] += α r*                                  # layers pathway (the mean-diff baseline)
S_q      q_ℓ[t] += α q*                                  # where heads look (dom)
S_k      k_ℓ[s] += α k*                                  # what is easy to find
S_v      v_ℓ[s] += α v*                                  # what is read out, i.e. the cache content (kv_cache_gram, vjp_cache)
S_bias   logit_ts += α b_s                               # attend by rule, no vector (PASTA-like)
S_gain   o^h ← (1 + α g_h) o^h                           # trust some heads more
```
Operator variants from steering-lite (on any site): add | project-out then add (directional_ablation) | gate by cos(x, x*) (cosine_gated) | per-head unit norm.

## Axis 3: combining the two pathways

```
C_sum      S_r(r*) + S_q(q*), each at α₀/√2                             # both at once, independent
C_cache    h_ℓ[t] += α r*  for t < last only                            # does mean diff act by writing into the cache?
C_direct   h_ℓ[last] += α r*  only                                      # C_cache vs C_direct splits the residual effect
C_write+read   C_cache + S_q(q*)                                        # r writes into the cache, q reads it more often (wassname's guess)
C_retrieve_bias  logit_ts += α z(⟨W_O^h v_s, r̂*⟩)                     # attend to tokens whose value writes r* (per token)
C_retrieve_q     q*_ℓ^h = E_x Σ_s A_s (k_s − k̄)(⟨W_O^h v_s, r̂*⟩ − s̄)   # closed form: one fixed q that raises retrieval of r* ("the Q that retrieves Δh")
C_retrieve_vjp   q*_ℓ = ∇_{δ_ℓ} ⟨r*_{ℓ+1}, h_{ℓ+1}[last]⟩               # same by backprop, local (one layer), not through the whole model
C_slot     add one KV entry per layer: v_slot^h = pinv(W_O^h) r̂*, attention logit b = α   # a place in the cache that holds r*; + S_q optional
```

## Selection (after a site works)

```
F_head   ΔJ_h on TRAIN with the vector on head h only; keep top k ∈ {8, 32, all}
F_layer  same per layer
```

## Cells to run

| # | extraction | site / combination | question it answers | status |
|--:|:--|:--|:--|:--|
| 1 | E_dom | S_q, S_k, S_v, S_r, C_sum, C_retrieve_bias × PERSONA, SOURCE | which site carries the concept at equal KL | queued (2328, 2329) |
| 2 | E_vjpJ | S_q, S_r | was the Q-VJP target the only problem? q vs r at equal extraction | todo |
| 3 | E_vjpc_mid | S_q | same, with a persona target at mid depth | todo |
| 4 | E_opt | S_q, S_r | at equal optimisation effort, is q worse, equal or better than r? | todo |
| 5 | E_dom | C_cache, C_direct, C_write+read | does mean diff work through the cache, and does q add to it? | todo |
| 6 | E_dom (r*) | C_retrieve_q, C_retrieve_vjp, C_slot | can a query reproduce the residual vector's effect by retrieval? | todo |
| 7 | E_pca, E_tok | S_q | cheap extraction variants of dom | todo |
| 8 | best of above | F_head | do a few heads carry it (dense Qwen3 case)? | todo |

Run on Qwen3-4B (dom weakest, 10/16) and Qwen3.5-4B. Top 3 at matched KL → free-form replies (07), Jev damage (08), and reading by hand. Agree must stay ≥ base − 1.
