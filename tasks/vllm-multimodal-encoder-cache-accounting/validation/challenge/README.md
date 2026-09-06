# Independent fresh challenge -- vllm-multimodal-encoder-cache-accounting

Curator-side verification harness. **Not** the agent verifier (`tests/`), **not**
referenced by `instruction.md`, **not** mounted into the agent image or any image
history layer. It exists only under `validation/challenge/` and is invoked directly
during dynamic validation (Phase D).

## Production API exercised
`PlaceholderRange embedding-count / subrange, EncoderCacheManager, Scheduler._try_schedule_encoder_inputs`

This is the same production entry point the agent verifier uses, so a correct
solution that passes `tests/` must also pass this challenge. The challenge is
**not** a copy of the verifier: it re-derives the invariant on independently
chosen inputs.

## Independent invariant
Encoder-cache budget accounting counts only the mask-selected embedding rows (not raw placeholder span), admits exactly the rows the running budget allows, and maps a prompt subrange that straddles a mask gap to the correct compact embedding range.

## Fresh inputs (not drawn from the verifier or `ci-cases.json`)
Mask of length 64 with 11 selected rows at an index pattern not used by the verifier; a two-item request (masked + None) sized so exactly one row survives the budget; a straddling-gap subrange.

## How Phase D runs it
```
python3 validation/challenge/challenge_encoder_cache.py  (CPU-importable; runs directly on the built image)
```
Emits machine-checkable JSON and a final `CHALLENGE_ENCODER_CACHE=PASS|FAIL` line; exits
non-zero on failure.

Expected outcomes in Phase D:
- **Oracle**: `CHALLENGE_ENCODER_CACHE=PASS`.
- **Semantically-different correct alternative** (`alternate-direct-mask-count.patch`, apply_after=oracle, expected_reward=1): `CHALLENGE_ENCODER_CACHE=PASS`.
  Confirms the challenge scores the behavioral contract, not one implementation.
- **Base / incorrect** (`partial-budget-omission.patch (expected_reward=0)`): `CHALLENGE_ENCODER_CACHE=FAIL`.

## Provenance guarantee
This file is under `validation/`, which is never copied into the environment
image (see `environment/Dockerfile`). Phase C verifies via `docker history` and a
final-filesystem scan that no challenge, verifier, solution, or reward logic
leaked into the image.
