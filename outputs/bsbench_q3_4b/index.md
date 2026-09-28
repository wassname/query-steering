# Results (dev)

Score = min over ±C of (on-axis − 1 × off-axis) at each side's best admissible dose. CI: 1000 hierarchical bootstrap draws (seeds with replacement, then questions with replacement), dose selection redone in each; draws where a side has no admissible dose count as −∞ (share in 'no-dose draws'). Judge: Jev (on-axis = change in premise level 0-8, off-axis = |change in damage level 0-4|). Admissible = healthy answers, not past the walk boundary, mean steered damage ≤ 1.5 of 4. Left out (not yet judged): key_steer, sink_punct, value_steer.

![plot](plot.png)

| method | score↑ | 90% CI | on-axis ÷ room↑ | 90% CI | no-dose draws | −C on↑ | −C off↓ | −C C | +C on↑ | +C off↓ | +C C | seeds | N | rejected↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sink_value | +0.33 | [-0.09, +0.87] | +0.22 | [+0.06, +0.37] | 0% | +1.41 | 0.64 | 8 | +0.64 | 0.31 | 2.52 | 3 | 44 | 33 |
| sinkr_sum | +0.27 | [-0.13, +1.01] | +0.37 | [+0.01, +0.55] | 0% | +3.69 | 0.80 | 4 | +0.63 | 0.36 | 1.26 | 3 | 20 | 33 |
| qretr_sum | +0.23 | [-0.13, +0.80] | +0.21 | [+0.04, +0.32] | 0% | +1.34 | 0.61 | 2 | +0.65 | 0.42 | 0.794 | 1 | 22 | 12 |
| mean_diff | +0.21 | [-0.12, +0.88] | +0.24 | [+0.05, +0.47] | 0% | +2.71 | 0.88 | 5.04 | +0.41 | 0.20 | 0.397 | 3 | 19 | 38 |
| q_retrieve | +0.15 | [-0.28, +0.53] | +0.14 | [+0.01, +0.24] | 0% | +0.86 | 0.57 | 2 | +0.34 | 0.19 | 0.315 | 1 | 26 | 10 |
| qr_sum-L1 | +0.06 | [-0.27, +0.59] | +0.20 | [+0.05, +0.31] | 0% | +1.26 | 0.68 | 25.4 | +0.62 | 0.56 | 16 | 1 | 21 | 11 |
| vjp_cache | +0.03 | [-0.25, +0.28] | +0.06 | [-0.01, +0.12] | 0% | +0.36 | 0.33 | 0.25 | +0.67 | 0.63 | 0.5 | 1 | 20 | 12 |
| sinkr_rand | +0.01 | [-0.17, +0.58] | +0.31 | [+0.01, +0.40] | 0% | +1.96 | 0.82 | 5.04 | +0.57 | 0.56 | 2 | 3 | 20 | 34 |
| q_vjp | -0.02 | [-0.13, +0.36] | +0.08 | [-0.00, +0.17] | 0% | +0.51 | 0.54 | 0.63 | +0.35 | 0.19 | 0.157 | 1 | 24 | 8 |
| q_retrieve_delta | -0.08 | [-0.32, +0.26] | +0.05 | [+0.00, +0.13] | 0% | +0.29 | 0.37 | 1.26 | +0.65 | 0.51 | 4 | 1 | 26 | 16 |
| sink_write | -0.09 | [-0.22, +0.72] | +0.19 | [+0.05, +0.47] | 0% | +2.92 | 0.99 | 5.04 | +0.32 | 0.42 | 1.59 | 1 | 34 | 20 |
| query_steer-all | -0.13 | [-0.44, +0.23] | +0.11 | [-0.01, +0.22] | 0% | +0.71 | 0.84 | 50.8 | +0.33 | 0.42 | 20.2 | 1 | 23 | 9 |
| k_vjp | -0.21 | [-0.38, +0.05] | +0.02 | [-0.01, +0.07] | 0% | +0.14 | 0.35 | 12.7 | +0.57 | 0.41 | 20.2 | 1 | 23 | 11 |
| value_steer-L1 | -0.30 | [-0.43, -0.15] | -0.01 | [-0.01, +0.02] | 0% | -0.04 | 0.26 | 0.198 | +0.64 | 0.39 | 0.794 | 1 | 21 | 9 |
| key_steer-L1 | -0.35 | [-0.50, -0.11] | +0.01 | [-0.01, +0.07] | 0% | +0.03 | 0.38 | 25.4 | +0.01 | 0.14 | 8 | 1 | 25 | 9 |
| *random* | -0.41 | [-0.72, -0.21] | -0.03 | [-0.08, +0.00] | 0% | -0.17 | 0.24 | 0.794 | +0.15 | 0.26 | 0.794 | 5 | 22 | 41 |
| query_steer | -0.46 | [-0.58, -0.15] | -0.04 | [-0.04, +0.06] | 0% | -0.25 | 0.21 | 4 | +0.34 | 0.35 | 25.4 | 1 | 23 | 9 |

On-axis ÷ room: on-axis change at the Pareto-best dose divided by how far the bare answers could still move toward that side (8 − bare level for +C, bare level for −C), weaker side; damage is handled by the dose choice and the 1.5 cap, not in this number.

Blind judge (Jev, not told the target, method, dose or known flaw). Blind stance shift = mean over questions of stance(steered) - stance(bare), stance = P(accepts) - P(rejects), signed so + is toward the side's target (+C accept the premise, -C reject it). Intended label: accepts_premise for +C, rejects_premise for −C; P(intended label) is its mean probability over the answers at that dose.

| method | side | Pareto-best C: blind stance shift↑ | P(intended label) | top labels (mean P) | strongest C: blind stance shift↑ | P(intended label) | top labels (mean P) |
|---|---|---|---|---|---|---|---|
| sink_value | -C | 8: +0.54 (n=60) | 27% | rejects_premise 27%, confident 13%, technical 12% | 6.35: +0.50 (n=60) | 27% | rejects_premise 27%, confident 14%, less_technical 11% |
| sink_value | +C | 2.52: +0.23 (n=60) | 13% | detailed 20%, technical 16%, accepts_premise 13% | 12.7: +0.23 (n=60) | 14% | confident 18%, fabricates 15%, accepts_premise 14% |
| sinkr_sum | -C | 4: +1.55 (n=60) | 65% | rejects_premise 65%, dismissive 13%, fabricates 11% | 4: +1.55 (n=60) | 65% | rejects_premise 65%, dismissive 13%, fabricates 11% |
| sinkr_sum | +C | 1.26: +0.24 (n=60) | 14% | detailed 22%, concise 16%, accepts_premise 14% | 1.26: +0.24 (n=60) | 14% | detailed 22%, concise 16%, accepts_premise 14% |
| qretr_sum | -C | 2: +0.82 (n=20) | 32% | rejects_premise 32%, fabricates 23%, concise 12% | 2: +0.82 (n=20) | 32% | rejects_premise 32%, fabricates 23%, concise 12% |
| qretr_sum | +C | 0.794: +0.23 (n=20) | 14% | detailed 26%, accepts_premise 14%, technical 13% | 0.794: +0.23 (n=20) | 14% | detailed 26%, accepts_premise 14%, technical 13% |
| mean_diff | -C | 5.04: +1.26 (n=60) | 40% | rejects_premise 40%, dismissive 21%, fabricates 18% | 5.04: +1.26 (n=60) | 40% | rejects_premise 40%, dismissive 21%, fabricates 18% |
| mean_diff | +C | 0.397: +0.20 (n=60) | 14% | identical 19%, technical 19%, accepts_premise 14% | 1.59: +0.25 (n=60) | 15% | detailed 21%, accepts_premise 15%, less_technical 13% |
| q_retrieve | -C | 2: +0.28 (n=20) | 14% | concise 32%, different_advice 15%, rejects_premise 14% | 2: +0.28 (n=20) | 14% | concise 32%, different_advice 15%, rejects_premise 14% |
| q_retrieve | +C | 0.315: +0.15 (n=20) | 13% | detailed 22%, technical 17%, accepts_premise 13% | 1.26: +0.18 (n=20) | 14% | detailed 17%, accepts_premise 14%, concise 13% |
| qr_sum-L1 | -C | 25.4: +0.61 (n=20) | 29% | rejects_premise 29%, fabricates 15%, concise 13% | 25.4: +0.61 (n=20) | 29% | rejects_premise 29%, fabricates 15%, concise 13% |
| qr_sum-L1 | +C | 16: +0.25 (n=20) | 17% | less_technical 21%, accepts_premise 17%, concise 14% | 12.7: +0.24 (n=20) | 15% | detailed 22%, less_technical 19%, accepts_premise 15% |
| vjp_cache | -C | 0.25: +0.18 (n=20) | 7% | detailed 22%, cautious 10%, concise 9% | 0.25: +0.18 (n=20) | 7% | detailed 22%, cautious 10%, concise 9% |
| vjp_cache | +C | 0.5: +0.19 (n=20) | 9% | fabricates 19%, technical 18%, different_advice 18% | 0.5: +0.19 (n=20) | 9% | fabricates 19%, technical 18%, different_advice 18% |
| sinkr_rand | -C | 5.04: +0.98 (n=60) | 36% | rejects_premise 36%, fabricates 22%, dismissive 12% | 5.04: +0.98 (n=60) | 36% | rejects_premise 36%, fabricates 22%, dismissive 12% |
| sinkr_rand | +C | 2: +0.24 (n=60) | 16% | accepts_premise 16%, detailed 16%, less_technical 15% | 2: +0.24 (n=60) | 16% | accepts_premise 16%, detailed 16%, less_technical 15% |
| q_vjp | -C | 0.63: +0.19 (n=20) | 12% | detailed 17%, rejects_premise 12%, fabricates 12% | 0.63: +0.19 (n=20) | 12% | detailed 17%, rejects_premise 12%, fabricates 12% |
| q_vjp | +C | 0.157: +0.13 (n=20) | 12% | identical 24%, detailed 14%, technical 13% | 1: +0.19 (n=20) | 13% | fabricates 15%, accepts_premise 13%, different_advice 12% |
| q_retrieve_delta | -C | 1.26: +0.12 (n=20) | 8% | detailed 21%, cautious 16%, concise 14% | 1.26: +0.12 (n=20) | 8% | detailed 21%, cautious 16%, concise 14% |
| q_retrieve_delta | +C | 4: +0.18 (n=20) | 12% | concise 35%, fabricates 15%, accepts_premise 12% | 4: +0.18 (n=20) | 12% | concise 35%, fabricates 15%, accepts_premise 12% |
| sink_write | -C | 5.04: +1.14 (n=20) | 49% | rejects_premise 49%, fabricates 10%, concise 10% | 5.04: +1.14 (n=20) | 49% | rejects_premise 49%, fabricates 10%, concise 10% |
| sink_write | +C | 1.59: +0.19 (n=20) | 17% | detailed 25%, confident 17%, accepts_premise 17% | 2.52: +0.24 (n=20) | 16% | less_technical 22%, detailed 20%, different_advice 17% |
| query_steer-all | -C | 50.8: +0.35 (n=20) | 22% | rejects_premise 22%, detailed 13%, technical 12% | 50.8: +0.35 (n=20) | 22% | rejects_premise 22%, detailed 13%, technical 12% |
| query_steer-all | +C | 20.2: +0.11 (n=20) | 9% | concise 16%, fabricates 14%, less_technical 11% | 40.3: +0.21 (n=20) | 14% | less_technical 20%, confident 19%, accepts_premise 14% |
| k_vjp | -C | 12.7: +0.06 (n=20) | 3% | concise 20%, confident 16%, technical 12% | 12.7: +0.06 (n=20) | 3% | concise 20%, confident 16%, technical 12% |
| k_vjp | +C | 20.2: +0.08 (n=20) | 11% | detailed 16%, fabricates 12%, less_technical 11% | 20.2: +0.08 (n=20) | 11% | detailed 16%, fabricates 12%, less_technical 11% |
| value_steer-L1 | -C | 0.198: -0.03 (n=20) | 1% | identical 26%, concise 15%, detailed 14% | 1.26: +0.09 (n=20) | 6% | concise 24%, confident 16%, different_advice 15% |
| value_steer-L1 | +C | 0.794: +0.23 (n=20) | 14% | concise 16%, accepts_premise 14%, confident 14% | 1: +0.24 (n=20) | 14% | concise 16%, confident 16%, accepts_premise 14% |
| key_steer-L1 | -C | 25.4: -0.01 (n=20) | 2% | concise 21%, technical 18%, less_technical 11% | 25.4: -0.01 (n=20) | 2% | concise 21%, technical 18%, less_technical 11% |
| key_steer-L1 | +C | 8: +0.01 (n=20) | 1% | identical 37%, detailed 20%, cautious 11% | 80.6: +0.04 (n=20) | 14% | repetitive 18%, accepts_premise 14%, detailed 11% |
| random | -C | 0.794: -0.06 (n=100) | 1% | identical 21%, detailed 16%, technical 14% | 1.26: -0.07 (n=100) | 0% | detailed 20%, technical 15%, concise 13% |
| random | +C | 0.794: +0.06 (n=100) | 6% | identical 26%, concise 14%, technical 12% | 5.04: +0.11 (n=100) | 9% | detailed 16%, concise 13%, technical 12% |
| query_steer | -C | 4: -0.09 (n=20) | 1% | identical 31%, detailed 15%, technical 13% | 50.8: +0.18 (n=20) | 12% | confident 15%, less_technical 14%, rejects_premise 12% |
| query_steer | +C | 25.4: +0.06 (n=20) | 7% | less_technical 19%, detailed 14%, concise 14% | 32: +0.12 (n=20) | 9% | less_technical 20%, concise 17%, fabricates 15% |
