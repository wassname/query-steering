# Single query for the hacking demo: "forbidden use" alone, Qwen3-4B, 33 held-out runs

Asked by wassname 2026-09-30: "does a single query work" / "yes please".

Recorded before the run. -- PI/claude

- vector: published 4B concepts (outputs/vectors/qwen3-4b/super_q, HF 8f76a93), group "forbidden use" only, at its own α_c; joint α recalibrated on dev[0] by coherence (same code path as outputs/06_qwen3-4b_single_forbiddenu.log)
- choice made on dev only: 6/10 fluent method reports, 0/10 unsteered, 0/10 null (outputs/06_qwen3-4b_single_forbiddenu.log)
- this is the 5th scoring of a 4B vector on the 33 test runs; disclose it
- question: default agent framing ("anything"), same as the 14/33 result
- prediction: test steered 8-16/33 (dev rate 6/10 shrinks), unsteered 0/33, null <=1/33
- success bar (goal 3): steered >=14/33 fluent, unsteered <=1/33, null <=1/33
- read every counted hit by hand; count broken text as a miss
- command: `uv run scripts/06_super_q.py --stage run --demos agent --splits dev,test --concepts "forbidden use" --out outputs/06_qwen3-4b_single_forbiddenu_test.md`

## Qwen3-32B follow-up (recorded before the run, 2026-09-30 13:20)

- 4B result: 14/33 test, 0/33 unsteered, 0/33 null, 3/33 broken
- 32B: published 32B concepts, "forbidden use" group alone, joint α calibrated on dev[0]; dev and test in one Modal job
- prediction: 32B test 2-8/33 (the 18-concept sum got 4/33; the misdeed group alone 4/33 with 23/33 broken)
- Modal spend before launch: $104.66 billed through Sep 29, Sep 30 apps unbilled, estimate ~$110 of $150; this run ~$5
