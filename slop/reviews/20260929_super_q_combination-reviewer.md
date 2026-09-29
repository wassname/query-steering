# oracle answer (reviewer, kimi-k3), to slop/reviews/20260929_super_q_combination-brief.md

## The problem, reconstructed

Each steer adds α·⟨q*, k⟩/√d to every attention score. With one concept, a demo "hits" when the dose on its relevant keys exceeds some threshold τ: α ≥ τ_c/α_c-scale. With 9 orthonormalized groups, the combined perturbation norm is ‖Σ α α_c Q_c‖ ≈ α·√(Σ α_c²‖Q_c‖²) ≈ 3× a single group's norm (groups are orthonormal, norms comparable). The joint calibration caps α at the point where the *total* norm breaks fluency, so the joint cap α* scales like α*_solo/√k ≈ α*_solo/3 — which is exactly what was observed (solo α_c 0.84–2.12; sum broke at 0.5, used 0.33). Consequence: the *effective per-group dose* is 0.33·α_c. For password, solo data says the secret direction needs effective α ≈ 1 (α=1 → 10/10; 32B needed α≈2). 0.33·α_c,secret ≈ 0.3–0.7 < 1 → 3/10. Eval needs even more → 0/10. Meanwhile agent improves (14/33) because it genuinely needs the misdeed+secret mixture and its threshold is lower. So my primary hypothesis: **dose dilution via a global cap set by the worst group and the worst prompt** — not vector corruption. Secondary hypotheses: (b) orthonormalization rotated the secret direction; (c) the cap is driven by one prompt (the 2,900-token agent replay's trigram repetition), so password/eval are throttled by an irrelevant constraint; (d) off-target groups add score noise ∝ α·√8·r, lowering SNR on password/eval.

## Options

**1. Per-demo caps, not min-over-demos.** The recipe allows per-demo α. Calibrate the joint break *separately per demo prompt* and use α_d = ⅔·α_break(d). Prediction (if c is true): password/eval tolerate α ≈ 0.6–0.8 and recover to ≥6/10; agent stays at 0.33.

**2. Reweight the sum toward what each demo needs.** q*_super = Σ w_c α_c Q_c, fit the 9 w_c by coordinate ascent on dev hits (password/eval/agent) subject to fluency. Prediction: w_secret, w_test rise, w_misdeed shrinks ~2×, recovering password/eval at joint α ≈ 0.5 while agent drops only slightly. Cheap: 9-dim search, each eval $5–10 — but overfits with 10-dev sizes; use the 33 held-out only once.

**3. Head/layer sparsification.** Keep each group only on the top-m heads by solo effect (ablate per head on extraction pairs, cheap — no generation, just score deltas). Cross-talk noise falls from √8·r to ~√(Σ active overlap); the joint break α rises. Prediction: at α=0.5, password ≥8/10 and agent ≥12/33. This is the only option that raises the *ceiling* instead of reallocating budget.

**4. Verify orthonormalization is innocent.** Re-run the secret group *alone, post-orthonormalization*, at its solo α_c on password. Prediction if (b): it now fails at α_c → the rotation, not the budget, is the bug; fix by skipping orthonormalization for groups with cos < 0.4 (they're near-orthogonal already) or by QR without norm renorming.

## Cheapest distinguishing experiment

Two runs (~$10–20 total):
- **Run A**: the published sum on password at α = 0.5, 0.67, ignoring the global cap. If hits recover at 0.5+ and the *password prompt's* fluency metrics stay clean, the cap was set by another prompt → option 1. If fluency breaks but hits don't return even at 0.67, the budget story is wrong.
- **Run B** (option 4's test): post-orthonormalization secret group alone at α_c on password. Fails → option 4/2 (rotation/weighting); succeeds → dose dilution confirmed, do options 1+3.

Run A alone already separates "budget dilution" (hits recover with α) from "corrupted sum" (they don't), and it reuses the existing vector and harness — no new extraction, one generation sweep.