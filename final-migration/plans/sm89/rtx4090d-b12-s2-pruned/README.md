# SM89 RTX 4090 D B12/S2 plan — FFN-pruned weight variant

Derived from `sm89/rtx4090d-b12-s2/best-tactic-plan.json` by rebinding
`target.model_sha256` only.

- file: `best-tactic-plan.json`
- file SHA-256: `2f133f59ac7c20c662a39a6fc11454296df3d5702c8bc5d794af6d8582047203`
- plan id: `sm89-rtx4090-1a068fd146ad0776-b11pruned-a8`
- semantic plan SHA-256: `1a068fd146ad0776fb8be1ea69bea4eafa501d45ac481c78e3853d60d26bc0f5`
  (inherited source value; the loader does not verify it)
- model SHA-256: `a1f9814939559cc34165d16b528b1ba304a3530c2f5468b58e059986d94ca862`
  (`b11-ffn-pruned-a8.bin.gz`)
- source model SHA-256: `1881600caab9e9d85a3dd6a019e9b8e7d2c237b5f984e13ed49a8645be3077c6`
- target: SM89, RTX 4090 class, exact 19x19, FP16/NHWC, B12, two streams per
  device
- derived UTC: `2026-10-02T13:24:55Z`
- config: `gtp-sm89-pruned.cfg`, file SHA-256
  `ce5806ce7f88a393b3e007caf12ba81d9bd77d754de9fe52224149bc6f68abb3`

## What was and was not re-validated

Only `target.model_sha256` differs from the source plan. Tactic selections,
certification fields, the 62/62 positive-history closure, and the recorded
long gate are inherited **unchanged** and were **not** independently
re-validated for the pruned weight.

That inheritance was measured to be safe for this weight pair:

| comparison | paired median delta | same-round drift | verdict |
| --- | ---: | ---: | --- |
| source plan vs this plan, B12, 5 paired rounds | −0.31 % | 1.70 % | equivalent within noise |
| single-key ablation, the 8 divergent `cuda*` keys, 3 rounds | ≤ 0.47 % | 1.87 % | equivalent within noise |

Because every individual effect is smaller than the measurement drift, the
divergent tactic choices between the two plans are not distinguishable on
this hardware. The pruned weight keeps the source plan's optimum, so this
derived plan is the production plan and no re-scan is required.

The long gate value stored in the JSON (`3026.196859` physical nnEval/s,
1000 iterations, 50 warmup, two samples) belongs to the **source** model.
It is retained as certification provenance, not as a performance claim for
this file. Likewise the inherited `correctness` block covers the source
model only.

## Runtime requirements

Loading this plan requires a binary carrying the runtime FFN-width patch
(branch `prune-width-per-gpu-sched`). The stock fork binary aborts on the
compiled `FfnChannels` gate when the pruned weight is loaded.

Point the model at the `.bin.gz` file whose SHA-256 is recorded above: the
gate is bound to the compressed file hash, not to the decompressed `.bin`.

Use an absolute path in the GTP config:

```cfg
cudaTacticPlanFile = /absolute/path/to/best-tactic-plan.json
cudaTacticPlanBatch = 12
```

`gtp-sm89-pruned.cfg` is the exact production config used for the
measurements below. Its `cudaTacticPlanFile` value is deployment-local and
must be repointed.

## Measured performance

Fixed harness, same-round paired, 5 rounds, NN cache off, `nnEvals/s`:

| weight | nnEval/s | delta |
| --- | ---: | ---: |
| `b11c768h12nbt3tflrs-fson-silu.bin.gz` (source) | 3192.3 | — |
| `b11-ffn-pruned-a8.bin.gz` (this plan) | 3733.6 | **+16.84 %** |

Production GTP path on one RTX 4090 D at `numSearchThreads = 36`: **3136.82**
nnEval/s.

These values are evidence for the tested host, clocks, model, batch, and
topology, not a universal performance guarantee. They depend on same-round
pairing; see `final-migration/records/prune-sched-sm89-20261003.md` for the
discipline and the raw comparisons.
