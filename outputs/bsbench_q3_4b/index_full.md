# Results (full)

Score = min over ±C of (on-axis − 1 × off-axis) at each side's best admissible dose. CI: 1000 hierarchical bootstrap draws (seeds with replacement, then questions with replacement), dose selection redone in each; draws where a side has no admissible dose count as −∞ (share in 'no-dose draws'). Judge: Jev (on-axis = change in premise level 0-8, off-axis = |change in damage level 0-4|). Admissible = healthy answers, not past the walk boundary, mean steered damage ≤ 1.5 of 4.

![plot](plot.png)

| method | score↑ | 90% CI | on-axis ÷ room↑ | 90% CI | no-dose draws | −C on↑ | −C off↓ | −C C | +C on↑ | +C off↓ | +C C | seeds | N | rejected↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sink_value | +0.05 | [-0.10, +0.24] | +0.19 | [+0.10, +0.24] | 0% | +1.27 | 0.62 | 5.04 | +0.27 | 0.22 | 0.5 | 1 | 46 | 10 |
| sinkr_sum | +0.02 | [-0.15, +0.21] | +0.20 | [+0.08, +0.30] | 0% | +3.91 | 0.64 | 4 | +0.28 | 0.26 | 0.315 | 1 | 22 | 10 |
| mean_diff | +0.02 | [-0.14, +0.24] | +0.21 | [+0.10, +0.31] | 0% | +2.48 | 0.77 | 5.04 | +0.30 | 0.27 | 0.5 | 1 | 21 | 11 |
| q_slot_big | -0.01 | [-0.18, +0.22] | +0.24 | [+0.08, +0.34] | 0% | +2.83 | 0.75 | 6.35 | +0.34 | 0.35 | 2 | 1 | 24 | 8 |

On-axis ÷ room: on-axis change at the Pareto-best dose divided by how far the bare answers could still move toward that side (8 − bare level for +C, bare level for −C), weaker side; damage is handled by the dose choice and the 1.5 cap, not in this number.

Blind judge (Jev, not told the target, method, dose or known flaw). Blind stance shift = mean over questions of stance(steered) - stance(bare), stance = P(accepts) - P(rejects), signed so + is toward the side's target (+C accept the premise, -C reject it). Intended label: accepts_premise for +C, rejects_premise for −C; P(intended label) is its mean probability over the answers at that dose.

| method | side | Pareto-best C: blind stance shift↑ | P(intended label) | top labels (mean P) | strongest C: blind stance shift↑ | P(intended label) | top labels (mean P) |
|---|---|---|---|---|---|---|---|
| sink_value | -C | 5.04: +0.52 (n=100) | 28% | rejects_premise 28%, concise 11%, different_advice 11% | 6.35: +0.55 (n=100) | 31% | rejects_premise 31%, concise 11%, different_advice 9% |
| sink_value | +C | 0.5: +0.11 (n=100) | 9% | identical 26%, concise 15%, different_advice 10% | 5.04: +0.16 (n=100) | 14% | less_technical 16%, accepts_premise 14%, concise 14% |
| sinkr_sum | -C | 4: +1.57 (n=100) | 68% | rejects_premise 68%, dismissive 10%, fabricates 6% | 4: +1.57 (n=100) | 68% | rejects_premise 68%, dismissive 10%, fabricates 6% |
| sinkr_sum | +C | 0.315: +0.12 (n=100) | 9% | identical 20%, concise 13%, technical 13% | 2: +0.15 (n=100) | 16% | less_technical 19%, accepts_premise 16%, different_advice 12% |
| mean_diff | -C | 5.04: +1.16 (n=100) | 37% | rejects_premise 37%, fabricates 16%, dismissive 15% | 5.04: +1.16 (n=100) | 37% | rejects_premise 37%, fabricates 16%, dismissive 15% |
| mean_diff | +C | 0.5: +0.10 (n=100) | 9% | detailed 15%, technical 14%, identical 12% | 1.59: +0.14 (n=100) | 14% | detailed 18%, less_technical 14%, accepts_premise 14% |
| q_slot_big | -C | 6.35: +1.02 (n=100) | 56% | rejects_premise 56%, detailed 7%, fabricates 6% | 6.35: +1.02 (n=100) | 56% | rejects_premise 56%, detailed 7%, fabricates 6% |
| q_slot_big | +C | 2: +0.14 (n=100) | 11% | concise 20%, different_advice 12%, less_technical 12% | 5.04: +0.16 (n=100) | 15% | concise 15%, less_technical 15%, accepts_premise 15% |
