# Independent fresh challenge -- vllm-moe-permute-batch-scaling

Curator-side verification harness. **Not** the agent verifier (`tests/`), **not**
referenced by `instruction.md`, **not** mounted into the agent image or any image
history layer. It exists only under `validation/challenge/` and is invoked directly
during dynamic validation (Phase D).

## Production API exercised
`torch.ops._moe_C.moe_permute (native)`

This is the same production entry point the agent verifier uses, so a correct
solution that passes `tests/` must also pass this challenge. The challenge is
**not** a copy of the verifier: it re-derives the invariant on independently
chosen inputs.

## Independent invariant
`moe_permute` produces a genuine bijection over every `(token, top-k)` routed
slot: each slot lands inside its routed expert's aligned offset window, the
`inv_permuted_idx`/`permuted_idx` maps round-trip exactly, the permuted payload
is a byte-exact gather of the source rows, and the expert-id fill and `-1`
sentinel tail are correct. The mapping follows the routing map independent of
batch size (no batch-dependent shortcut), and the aligned large-batch latency
stays flat. The challenge calls `moe_permute` only; it does not call
`moe_unpermute`.

## Fresh inputs (not drawn from the verifier or `ci-cases.json`)
FRESH_TOKENS = (7, 63, 129, 257, 1000, 3000) with routing pattern 23*token+11*rank+3; none in the verifier's power-of-two batch set.

## How Phase D runs it
```
python3 validation/challenge/challenge_moe_permute.py  (GPU/CUDA native op; Phase D)
```
Emits machine-checkable JSON and a final `CHALLENGE_MOE_PERMUTE=PASS|FAIL` line; exits
non-zero on failure.

Expected outcomes in Phase D:
- **Oracle**: `CHALLENGE_MOE_PERMUTE=PASS`.
- **Semantically-different correct alternative** (`alternate-cuda-scaling.patch`, apply_after=base, expected_reward=1): `CHALLENGE_MOE_PERMUTE=PASS`.
  Confirms the challenge scores the behavioral contract, not one implementation.
- **Base / incorrect** (`diagnosis-only-linear-scan.patch (expected_reward=0)`): `CHALLENGE_MOE_PERMUTE=FAIL`.
- **Timed-size special-casing** (`special-case-batch-sizes.patch`, apply_after=oracle, expected_reward=0):
  accelerates only the profiled power-of-two batches and leaves the scaling
  regression for other counts. It passes on the verifier's power-of-two timed
  sizes but the challenge's fresh non-power-of-two large batch (3000) stays slow,
  so it FAILs the scaling invariant. The public verifier reproduces this via its
  own non-power-of-two probe (`PROBE_LARGE=3000`).

## Provenance guarantee
This file is under `validation/`, which is never copied into the environment
image (see `environment/Dockerfile`). Phase C verifies via `docker history` and a
final-filesystem scan that no challenge, verifier, solution, or reward logic
leaked into the image.
