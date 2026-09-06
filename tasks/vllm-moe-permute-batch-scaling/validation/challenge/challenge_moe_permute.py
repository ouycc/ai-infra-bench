#!/usr/bin/env python3
"""Independent fresh challenge for the native _moe_C.moe_permute batch scaling.

Curator-side only: not the agent verifier, not mounted into the agent image,
not referenced by instruction.md. It enters through the same production native
operator (torch.ops._moe_C.moe_permute) but on token counts and a routing
pattern derived independently of tests/verify_moe_permute.py (whose BATCH_SIZES
are 1/32/128/512/1024/2048/4096 with a fixed 17*token+7*rank routing).

Two independent invariants are re-derived here:

1. Correctness: the permutation the kernel produces must be a genuine bijection
   that groups every (token, top-k) slot under its routed expert within
   expert-aligned offset windows, and the permuted payload must be a byte-exact
   gather of the source rows.

2. Batch scaling: the task's defect is a large-batch scaling regression -- the
   Base kernel is functionally correct but its per-call latency grows
   super-linearly with the routed-slot count. The correct solution keeps the
   large-batch cost flat. Correctness alone does NOT separate Base (or a
   perf-only control) from the solution, so we also assert the scaling contract
   the production verifier scores, on this challenge's own fresh token counts.

Phase D runs this on the built A100 native image (rebuilt from the candidate
csrc) against Oracle (PASS) and a semantically different correct alternative
(PASS). A batch-scaling-broken Base kernel and the diagnosis-only control (which
leaves the scaling curve intact) must FAIL the scaling invariant.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import statistics
import sys
import traceback

import torch


N_EXPERT = 64
TOPK = 6
HIDDEN = 2048
ALIGN = 128
# Fresh token counts, none of which appear in verify_moe_permute.BATCH_SIZES.
FRESH_TOKENS = (7, 63, 129, 257, 1000, 3000)


def load_candidate_native() -> pathlib.Path:
    spec = importlib.util.find_spec("vllm._moe_C")
    assert spec and spec.origin
    native = pathlib.Path(spec.origin).resolve()
    assert native.is_relative_to(pathlib.Path("/app"))
    torch.ops.load_library(str(native))
    assert torch.ops._moe_C.moe_permute_unpermute_supported()
    return native


def make_inputs(n_token: int):
    torch.manual_seed(70001 + n_token)
    hidden = torch.empty((n_token, HIDDEN), device="cuda", dtype=torch.float8_e4m3fn)
    hidden.view(torch.uint8).random_(0, 127)
    token = torch.arange(n_token, device="cuda", dtype=torch.int64)[:, None]
    rank = torch.arange(TOPK, device="cuda", dtype=torch.int64)[None, :]
    # Fresh routing pattern (distinct multipliers from the verifier's 17/7).
    topk_ids = ((token * 23 + rank * 11 + 3) % N_EXPERT).to(torch.int32)
    token_expert_indices = torch.arange(
        n_token * TOPK, device="cuda", dtype=torch.int32
    ).reshape(n_token, TOPK)
    return hidden, topk_ids, token_expert_indices


def allocate_outputs(n_token: int, hidden: torch.Tensor):
    rows = (n_token * TOPK + N_EXPERT * (ALIGN - 1) + ALIGN - 1) // ALIGN * ALIGN
    return (
        torch.empty((rows, HIDDEN), device="cuda", dtype=hidden.dtype),
        torch.empty(N_EXPERT + 1, device="cuda", dtype=torch.int64),
        torch.empty((n_token, TOPK), device="cuda", dtype=torch.int32),
        torch.full((rows,), n_token * TOPK, device="cuda", dtype=torch.int32),
        torch.full((rows,), -1, device="cuda", dtype=torch.int32),
    )


def call_op(hidden, topk_ids, token_expert_indices, outputs, align_block_size):
    torch.ops._moe_C.moe_permute(
        hidden, topk_ids, token_expert_indices, None,
        N_EXPERT, N_EXPERT, TOPK, align_block_size, *outputs,
    )


def check_case(n_token: int, align_block_size: int | None) -> dict:
    """Validate one moe_permute call. ``align_block_size`` None => unaligned.

    Aligned: expert ranges are block-aligned, aligned offsets are written back,
    and m_indices is filled per aligned row with a -1 sentinel tail. Unaligned:
    expert ranges are the raw unpadded prefix offsets, routed slots pack densely
    into [0, n_token*topk), and m_indices is left unwritten by the op.
    """
    aligned = align_block_size is not None
    hidden, topk_ids, token_expert_indices = make_inputs(n_token)
    outputs = allocate_outputs(n_token, hidden)
    permuted, offsets, inverse, permuted_idx, m_indices = outputs
    call_op(hidden, topk_ids, token_expert_indices, outputs, align_block_size)
    torch.cuda.synchronize()

    flat_ids = topk_ids.flatten().to(torch.int64)
    counts = torch.bincount(flat_ids, minlength=N_EXPERT)
    windows = ((counts + ALIGN - 1) // ALIGN) * ALIGN if aligned else counts
    expected_offsets = torch.cat(
        [torch.zeros(1, device="cuda", dtype=torch.int64), torch.cumsum(windows, dim=0)]
    )
    torch.testing.assert_close(offsets, expected_offsets, atol=0, rtol=0)

    original = torch.arange(n_token * TOPK, device="cuda", dtype=torch.int64)
    destinations = inverse.flatten().to(torch.int64)
    # Genuine bijection over all routed slots.
    assert int(destinations.unique().numel()) == n_token * TOPK
    assert bool(torch.all(destinations >= offsets[flat_ids]))
    assert bool(torch.all(destinations < offsets[flat_ids] + counts[flat_ids]))
    torch.testing.assert_close(
        permuted_idx[destinations].to(torch.int64), original, atol=0, rtol=0
    )
    # Byte-exact payload gather.
    source_rows = original // TOPK
    torch.testing.assert_close(
        permuted[destinations].view(torch.uint8),
        hidden[source_rows].view(torch.uint8),
        atol=0, rtol=0,
    )
    if aligned:
        # Expert-id fill within each window and -1 sentinel tail.
        for expert in range(N_EXPERT):
            start = int(offsets[expert])
            end = int(offsets[expert + 1])
            if end > start:
                assert bool(torch.all(m_indices[start:end] == expert))
        tail = int(offsets[-1])
        if tail < m_indices.numel():
            assert bool(torch.all(m_indices[tail:] == -1))
    return {"n_token": n_token, "routed_slots": n_token * TOPK,
            "mode": "aligned" if aligned else "unaligned"}


# Scaling probe. SMALL is a light batch; LARGE is a heavy one. Neither appears
# in the verifier's BATCH_SIZES. Thresholds mirror the verifier's scaling
# contract (a bounded large-batch cost and a bounded small->large growth ratio)
# with wide margins so a shared A100 stays deterministic in verdict:
#   measured base   -> LARGE ~502us, ratio ~10.0   (must FAIL)
#   measured oracle -> LARGE ~110us, ratio ~1.9     (must PASS)
# Placing the cuts at 250us / 4.0 leaves a >2x gap to either side.
SCALE_SMALL = 63
SCALE_LARGE = 3000
LARGE_US_MAX = 250.0
RATIO_MAX = 4.0


def time_case(n_token: int, trials: int = 7, iterations: int = 50) -> float:
    # Scaling defect lives on the aligned path; time that path.
    hidden, topk_ids, token_expert_indices = make_inputs(n_token)
    outputs = allocate_outputs(n_token, hidden)
    for _ in range(20):
        call_op(hidden, topk_ids, token_expert_indices, outputs, ALIGN)
    torch.cuda.synchronize()
    samples = []
    for _ in range(trials):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            call_op(hidden, topk_ids, token_expert_indices, outputs, ALIGN)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / iterations)
    return statistics.median(samples)


def check_scaling() -> dict:
    small = time_case(SCALE_SMALL)
    large = time_case(SCALE_LARGE)
    ratio = large / small
    assert large < LARGE_US_MAX, (
        "large-batch latency exceeds bound (batch-scaling regression)",
        {"large_us": round(large, 2), "bound_us": LARGE_US_MAX},
    )
    assert ratio < RATIO_MAX, (
        "small->large latency growth exceeds bound (batch-scaling regression)",
        {"ratio": round(ratio, 3), "bound": RATIO_MAX,
         "small_us": round(small, 2), "large_us": round(large, 2)},
    )
    return {
        "small_token": SCALE_SMALL, "large_token": SCALE_LARGE,
        "small_us": round(small, 2), "large_us": round(large, 2),
        "ratio_large_over_small": round(ratio, 3),
    }


def main() -> None:
    assert torch.cuda.is_available(), "challenge requires CUDA"
    load_candidate_native()
    results = []
    failures = []
    for n_token in FRESH_TOKENS:
        for align_block_size in (ALIGN, None):  # aligned AND unaligned
            try:
                results.append(check_case(n_token, align_block_size))
            except Exception as exc:  # noqa: BLE001 - report every case
                failures.append(
                    {
                        "n_token": n_token,
                        "mode": "aligned" if align_block_size is not None
                        else "unaligned",
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
    # Scaling invariant. Only measured once correctness holds -- a scaling
    # number on a kernel that produces wrong output would be meaningless.
    scaling = None
    if not failures:
        try:
            scaling = check_scaling()
        except Exception as exc:  # noqa: BLE001
            failures.append(
                {
                    "stage": "scaling",
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
    print(json.dumps(
        {"results": results, "scaling": scaling, "failures": failures},
        indent=2, sort_keys=True,
    ))
    if failures:
        print("CHALLENGE_MOE_PERMUTE=FAIL")
        sys.exit(1)
    print("CHALLENGE_MOE_PERMUTE=PASS")


if __name__ == "__main__":
    main()
