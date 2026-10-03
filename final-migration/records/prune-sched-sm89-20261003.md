# SM89 FFN-pruned weight + per-GPU event pipeline scheduler, 2026-10-03

Branch: `prune-width-per-gpu-sched`, head `ae85f08`, on top of
`cuda-plan-runtime-sm89-sm120` (`a3f5446`, `v1.17.2`).

Target: AutoDL RTX 4090 D 24 GB, 425 W power cap, CUDA 13.0.88, cuDNN
9.14.0.64, 256 cores. One device, two inference lanes.

Weight: `b11-ffn-pruned-a8.bin.gz` (183 MB), SHA-256
`a1f9814939559cc34165d16b528b1ba304a3530c2f5468b58e059986d94ca862`.
Plan: `sm89-rtx4090-1a068fd146ad0776-b11pruned-a8`, file SHA-256
`2f133f59ac7c20c662a39a6fc11454296df3d5702c8bc5d794af6d8582047203`, checked in
under `final-migration/plans/sm89/rtx4090d-b12-s2-pruned/`.

## Commits

| commit | subject |
| --- | --- |
| `e8737bd` | SM89: make FFN width a runtime parameter for the dual-FFN / linear2 CUTLASS tactics |
| `fe73c39` | SM89: per-GPU event pipeline scheduler (v4) |
| `ae85f08` | SM89: partial-batch fill grace (`cudaEventPipelineBatchFillMicros`) |

The FFN-width change removes the compile-time assumption that the dual-FFN
and `linear2` CUTLASS tactics were built for a fixed channel count, so a
weight whose FFN width differs from the compiled constant can be loaded
instead of tripping the `FfnChannels` gate. The scheduler and fill-grace
changes are the per-GPU inference-lane work; the fill grace is expressed in
microseconds and defaults to `0`.

## Measurement discipline

Continuous runs on this target drift monotonically downward by 1.7–1.9 % as
the device heats. A sequential "run A, then run B" comparison therefore
attributes the drift to B and reports a spurious gain for every variant; an
early ablation run did exactly that and showed all eight tested keys as
"profitable". Every number below uses same-round pairing instead: all arms
are measured inside each round, the arm order is rotated per round, and only
within-round paired deltas are reported. The observed drift is reported
alongside each result as the noise scale, and `|delta| < drift` is treated
as equivalence within noise.

## Results

Pruned weight vs source weight, fixed harness, 5 paired rounds:

| weight | nnEval/s |
| --- | ---: |
| `b11c768h12nbt3tflrs-fson-silu.bin.gz` | 3192.3 |
| `b11-ffn-pruned-a8.bin.gz` | 3733.6 |

Paired median delta **+16.84 %**, far above the ~1.7 % drift scale. As a
cross-check the source weight measures 3192.3 under this harness against
3136.8 through the production GTP path, about 1.8 % apart, so the
`benchmarknn`-vs-production gap is essentially the pruning gain itself.

Derived plan vs a freshly re-scanned plan for the pruned weight, exact B12,
5 paired rounds: paired median delta **−0.31 %** against 1.70 % drift.
Single-key ablation of the 8 `cuda*` keys that differ between the two plans,
3 rounds: every single-key effect ≤ 0.47 % against 1.87 % drift. The re-scan
did not move the optimum, so the derived plan stays in production.

Bare config replica vs the real plan, pruned weight, GTP-shaped harness
(`benchmarknn` does not load a plan and cannot show this): 3679.65 vs
3677.25 nnEval/s, paired median delta **+0.024 %** against 1.61 % drift.
This is the basis for reusing an existing plan across same-shape weights by
rebinding `target.model_sha256` instead of re-scanning. It does not justify
dropping the plan, which also carries hardware identity validation, model
hash binding, fail-closed batch certification, and precision-gate binding.

Production GTP path, `numSearchThreads = 36`: **3136.82** nnEval/s against a
3134.7 baseline at 40 threads. Saturation is reached at 40 threads; the
binding constraint is the 425 W power cap, not thread count.

## Reuse boundary

The reuse unit is `(structure, batch, precision/layout, GPU class)`. Reusing
a plan is valid within one unit; a different structure, batch, precision,
layout, or GPU class needs a new scan. Prefer a derived plan over a bare
config replica, because the plan carries the guard rails and the replica
does not.

## Not done

- 8192-row all-head FP32 replay against the immutable reference for the
  pruned weight, and `runnngtpstresstest`. The `correctness` block inherited
  by the derived plan covers the source model only, so the pruned weight is
  currently certified by performance measurement alone. Performance evidence
  does not substitute for the accuracy gate.
- Multi-device measurement of the fill grace: sweep
  `cudaEventPipelineBatchFillMicros` over
  `{0, 200, 500, 1000, 2000, 5000, 10000}` microseconds and observe
  `avgBatchSize`.
- AOT path width validation before `prune-width` could be proposed upstream.

## Already measured, no gain (do not repeat)

RMSNorm rows8; C384 Vec8/Vec4 SiLU; `InitialGlobalMatMulAdd`; fused
QK+RoPE; split QKV+RoPE; ValueTerminal fusion. New experiments must not
disturb the production tree.
