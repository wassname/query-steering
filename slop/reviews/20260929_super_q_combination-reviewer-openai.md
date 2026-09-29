# oracle answer (reviewer-openai, gpt-6-astra), to slop/reviews/20260929_super_q_combination-brief.md

## Reconstruction

The goal is one fixed query intervention that promotes disclosure across three behaviors—not merely a large vector containing every concept. The brief shows an agent gain (14/33) accompanied by password/eval losses, but does **not** establish their cause: pooling, orthogonalization, relative weights, and final strength all changed.

**Dose dilution is the first hypothesis to test.** In the nine-group mixture, each component’s effective coefficient is \(0.33\alpha_c=0.28\)–\(0.70\), versus the earlier successful secret coefficient near 1. Orthogonalization also changes directions, so even those coefficients are not directly comparable. Solo “largest nonbreaking dose” measures tolerance, not disclosure efficacy; adding nine individually tolerable interventions need not remain tolerable.

Orthogonality in query space does not guarantee independent behavior: score shifts depend on the actual keys, softmax, and downstream layers. More precisely, with RoPE the perturbation is
\[
\Delta s_{tj}=\alpha(R_tq^*)^\top(R_jk_j)/\sqrt d.
\]
Thus Euclidean decorrelation may remove useful shared directions without reducing functional interference.

## Concrete alternatives

**1. Fit a small, efficacy-weighted mixture before adding concepts.** Start with original secret, test, and pooled-misdeed vectors:
\[
v=s+\beta t+\gamma m.
\]
Sweep a tiny grid of \((\beta,\gamma)\), allowing a scalar \(a_d\) per demo. Optimize the worst task’s development success, subject to fluency and null constraints:
\[
\max_{\beta,\gamma,\{a_d\}}\min_d H_d(a_dv).
\]
Use comparable success rates, not raw counts across unequal sample sizes. This predicts that preserving secret/test dose recovers their demos while retaining some agent benefit. Failure at matched doses would implicate interference rather than simple dilution.

**2. Protect useful directions instead of symmetrically orthogonalizing everything.** Leave secret/test unchanged. Per head, form \(B=[s,t]\) and modify only auxiliary vectors:
\[
r_c=(I-BB^\dagger)q_c,\qquad
v=w_ss+w_tt+\sum_cw_cr_c.
\]
Do not restore tiny residuals to their original norms: that amplifies noise. Compare against the same weighted mixture without any projection. If symmetric orthogonalization damaged useful directions, either variant should recover password/eval performance. If residualization specifically helps, overlapping auxiliary directions were interfering. This protects query geometry, not necessarily behavior.

**3. Calibrate the mixture for task success, not just survival.** For each candidate vector, sweep \(a\) and record disclosure, completion, repetition, and null false positives separately. Select
\[
a_d=\arg\max_a H_d(av)
\quad\text{subject to prespecified quality constraints}.
\]
Keep JSON compliance as a separate diagnostic rather than assuming its failure defines every demo’s useful range. This predicts a usable task-specific dose above or below 0.33. If quality fails before password/eval recover, scalar recalibration alone cannot solve the problem.

## Cheapest distinguishing experiment

On a tiny **development-only**, paired-seed panel, compare: original secret alone; orthogonalized secret alone; current mixture; and current mixture with secret/test coefficients raised to their successful solo doses while auxiliaries stay fixed. Include nulls and short dose sweeps.

These distinguish underdosing, direction damage, and joint interference before another full held-out evaluation. Confirm only the winning recipe on untouched runs. Recalibrate on 32B; the supplied results already suggest its dose requirements differ.