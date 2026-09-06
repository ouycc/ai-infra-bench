#!/usr/bin/env python3
"""Correctness-gated, model-free A100 timing of exact `_moe_C.moe_permute`.

Correctness is scored on BOTH operator modes the instruction names:

  * aligned   (align_block_size=128): expert ranges are block-aligned, the
    padded prefix offsets are written back, and ``m_indices`` is filled with the
    expert id per aligned row and a -1 sentinel tail.
  * unaligned (align_block_size=None -> -1): expert ranges are the raw
    (unpadded) prefix offsets, routed slots pack densely into ``[0, n_token*topk)``,
    and ``m_indices`` is left unwritten (getMIndices is not invoked).

Both modes check output allocation/shape, the token->expert offset mapping, the
inverse/permuted-index bijection, the byte-exact payload gather, dtype/device/
contiguity, and (aligned only) the expert-id fill and sentinel. Correctness runs
on power-of-two batches AND non-power-of-two batches (1000, 3000) that are NOT in
the timed set, so a kernel cannot special-case the timed shapes and still pass.

Performance targets the aligned large-batch scaling defect. Besides the
instruction contract (4096 < 250us, 4096/512 < 3.5) it also times a fresh
non-power-of-two large batch (3000) that is not one of the special-cased power
-of-two sizes and asserts the same absolute bound there. A kernel that only
accelerates the timed power-of-two sizes (see
validation/special-case-batch-sizes.patch) stays slow at 3000 and fails.

Thresholds are the A100 bounds already validated dynamically by the independent
challenge (measured Base ~502us / Oracle ~110us at 3000 tokens; cuts at 250us
and ratio 4.0 leave a >2x margin either side).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pathlib
import statistics

import torch


N_EXPERT = 64
TOPK = 6
HIDDEN = 2048
ALIGN = 128
# Correctness batches: the timed power-of-two set PLUS non-power-of-two counts
# (1000, 3000) that never appear in the timed set. Each is checked in aligned
# AND unaligned mode.
CORRECTNESS_TOKENS = (1, 32, 128, 512, 1000, 1024, 2048, 3000, 4096)
# Instruction timing contract (aligned path).
BATCH_SIZES = (1, 32, 128, 512, 1024, 2048, 4096)
LARGE_US_MAX = 250.0
RATIO_MAX = 3.5
# Anti-special-case probe: a non-power-of-two large batch and a small baseline,
# neither of which is a power-of-two timed size a kernel could hardcode.
PROBE_SMALL = 63
PROBE_LARGE = 3000
PROBE_LARGE_US_MAX = 250.0
PROBE_RATIO_MAX = 4.0


def load_candidate_native() -> pathlib.Path:
    spec = importlib.util.find_spec("vllm._moe_C")
    assert spec and spec.origin
    native = pathlib.Path(spec.origin).resolve()
    assert native.is_relative_to(pathlib.Path("/app"))
    torch.ops.load_library(str(native))
    assert torch.ops._moe_C.moe_permute_unpermute_supported()
    return native


def make_inputs(n_token: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # FP8 is used as a one-byte storage type. This kernel only copies payload
    # bytes; it does not execute FP8 Tensor Core arithmetic, so SM80 is valid.
    torch.manual_seed(32892 + n_token)
    hidden = torch.empty(
        (n_token, HIDDEN), device="cuda", dtype=torch.float8_e4m3fn
    )
    hidden.view(torch.uint8).random_(0, 127)
    token = torch.arange(n_token, device="cuda", dtype=torch.int64)[:, None]
    rank = torch.arange(TOPK, device="cuda", dtype=torch.int64)[None, :]
    topk_ids = ((token * 17 + rank * 7) % N_EXPERT).to(torch.int32)
    token_expert_indices = torch.arange(
        n_token * TOPK, device="cuda", dtype=torch.int32
    ).reshape(n_token, TOPK)
    return hidden, topk_ids, token_expert_indices


def allocate_outputs(
    n_token: int, hidden: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Aligned upper bound also safely holds every unaligned destination
    # (n_token*topk <= aligned rows), so one allocation serves both modes.
    rows = (
        (n_token * TOPK + N_EXPERT * (ALIGN - 1) + ALIGN - 1) // ALIGN * ALIGN
    )
    return (
        torch.empty((rows, HIDDEN), device="cuda", dtype=hidden.dtype),
        torch.empty(N_EXPERT + 1, device="cuda", dtype=torch.int64),
        torch.empty((n_token, TOPK), device="cuda", dtype=torch.int32),
        torch.full((rows,), n_token * TOPK, device="cuda", dtype=torch.int32),
        torch.full((rows,), -1, device="cuda", dtype=torch.int32),
    )


def call_op(
    hidden: torch.Tensor,
    topk_ids: torch.Tensor,
    token_expert_indices: torch.Tensor,
    outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    align_block_size: int | None,
) -> None:
    torch.ops._moe_C.moe_permute(
        hidden,
        topk_ids,
        token_expert_indices,
        None,
        N_EXPERT,
        N_EXPERT,
        TOPK,
        align_block_size,
        *outputs,
    )


def check_case(n_token: int, align_block_size: int | None) -> None:
    """Validate one moe_permute call. ``align_block_size`` None => unaligned."""
    aligned = align_block_size is not None
    hidden, topk_ids, token_expert_indices = make_inputs(n_token)
    outputs = allocate_outputs(n_token, hidden)
    permuted, offsets, inverse, permuted_idx, m_indices = outputs
    call_op(hidden, topk_ids, token_expert_indices, outputs, align_block_size)
    torch.cuda.synchronize()

    # Output allocation / shape / dtype / device / contiguity.
    assert offsets.shape == (N_EXPERT + 1,)
    assert inverse.shape == (n_token, TOPK)
    assert offsets.dtype == torch.int64
    assert inverse.dtype == torch.int32
    assert permuted_idx.dtype == torch.int32
    assert permuted.dtype == hidden.dtype
    for tensor in (permuted, offsets, inverse, permuted_idx, m_indices):
        assert tensor.is_cuda and tensor.is_contiguous()

    flat_ids = topk_ids.flatten().to(torch.int64)
    counts = torch.bincount(flat_ids, minlength=N_EXPERT)
    if aligned:
        windows = ((counts + ALIGN - 1) // ALIGN) * ALIGN
    else:
        # Unaligned: expert ranges are the raw unpadded per-expert counts.
        windows = counts
    expected_offsets = torch.cat(
        [
            torch.zeros(1, device="cuda", dtype=torch.int64),
            torch.cumsum(windows, dim=0),
        ]
    )
    torch.testing.assert_close(offsets, expected_offsets, atol=0, rtol=0)

    original = torch.arange(n_token * TOPK, device="cuda", dtype=torch.int64)
    destinations = inverse.flatten().to(torch.int64)
    # Genuine bijection over every routed (token, top-k) slot.
    assert int(destinations.unique().numel()) == n_token * TOPK
    routed_expert = flat_ids
    # Each slot lands inside its routed expert's window (payload region is the
    # first ``counts[e]`` rows of the window in both modes).
    assert bool(torch.all(destinations >= offsets[routed_expert]))
    assert bool(
        torch.all(destinations < offsets[routed_expert] + counts[routed_expert])
    )
    torch.testing.assert_close(
        permuted_idx[destinations].to(torch.int64), original, atol=0, rtol=0
    )

    # Byte-exact payload gather from the source token rows.
    source_rows = original // TOPK
    torch.testing.assert_close(
        permuted[destinations].view(torch.uint8),
        hidden[source_rows].view(torch.uint8),
        atol=0,
        rtol=0,
    )

    if aligned:
        # getMIndices fills expert id per aligned row and a -1 sentinel tail.
        for expert in range(N_EXPERT):
            start = int(offsets[expert])
            end = int(offsets[expert + 1])
            if end > start:
                assert bool(torch.all(m_indices[start:end] == expert))
        tail = int(offsets[-1])
        if tail < m_indices.numel():
            assert bool(torch.all(m_indices[tail:] == -1))
    # Unaligned: getMIndices is not invoked, so m_indices is intentionally
    # unwritten by the op and is not part of the observable contract here.


def time_case(
    n_token: int, align_block_size: int | None = ALIGN, trials: int = 5,
    iterations: int = 50,
) -> float:
    hidden, topk_ids, token_expert_indices = make_inputs(n_token)
    outputs = allocate_outputs(n_token, hidden)
    for _ in range(20):
        call_op(hidden, topk_ids, token_expert_indices, outputs, align_block_size)
    torch.cuda.synchronize()

    samples = []
    for _ in range(trials):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            call_op(hidden, topk_ids, token_expert_indices, outputs, align_block_size)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / iterations)
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("correctness", "performance", "all"),
        default="all",
    )
    args = parser.parse_args()
    native = load_candidate_native()
    native_sha256 = hashlib.sha256(native.read_bytes()).hexdigest()
    assert torch.cuda.is_available()
    assert torch.cuda.get_device_capability(0) == (8, 0)

    common = {
        "alignment": ALIGN,
        "correctness_modes": ["aligned", "unaligned"],
        "correctness_tokens": list(CORRECTNESS_TOKENS),
        "timed_batch_sizes": list(BATCH_SIZES),
        "anti_special_case_probe": {"small": PROBE_SMALL, "large": PROBE_LARGE},
        "dtype": "torch.float8_e4m3fn storage/copy; no FP8 arithmetic",
        "gpu": torch.cuda.get_device_name(0),
        "hidden_size": HIDDEN,
        "n_expert": N_EXPERT,
        "native_extension": str(native),
        "native_sha256": native_sha256,
        "topk": TOPK,
    }
    if args.stage in ("correctness", "all"):
        for batch in CORRECTNESS_TOKENS:
            check_case(batch, ALIGN)  # aligned path
            check_case(batch, None)  # unaligned path
        print(json.dumps({**common,
                          "correctness_cases": len(CORRECTNESS_TOKENS) * 2,
                          "correctness_passed": True}, sort_keys=True))
        print("MOE_PERMUTE_CORRECTNESS_STAGE=PASS")
    if args.stage in ("performance", "all"):
        timings = {str(batch): round(time_case(batch), 3) for batch in BATCH_SIZES}
        probe = {
            str(PROBE_SMALL): round(time_case(PROBE_SMALL), 3),
            str(PROBE_LARGE): round(time_case(PROBE_LARGE), 3),
        }
        large_batch_ratio = timings["4096"] / timings["512"]
        probe_ratio = probe[str(PROBE_LARGE)] / probe[str(PROBE_SMALL)]
        # Instruction contract on the timed power-of-two sizes.
        assert large_batch_ratio < RATIO_MAX, (
            "moe_permute retains the legacy large-batch scaling curve",
            timings,
        )
        assert timings["4096"] < LARGE_US_MAX, timings
        # Anti-special-case: the scaling fix must hold at a non-power-of-two
        # large batch too, so hardcoding the timed sizes cannot pass.
        assert probe[str(PROBE_LARGE)] < PROBE_LARGE_US_MAX, (
            "large non-power-of-two batch stayed slow (timed-size special casing?)",
            probe,
        )
        assert probe_ratio < PROBE_RATIO_MAX, (
            "non-power-of-two small->large latency growth exceeds bound",
            probe,
        )
        print(json.dumps({**common, "timings_median_us": timings,
                          "probe_median_us": probe,
                          "large_batch_ratio_4096_over_512": round(
                              large_batch_ratio, 3
                          ),
                          "probe_ratio_large_over_small": round(
                              probe_ratio, 3
                          )}, sort_keys=True))
        print("MOE_PERMUTE_PERFORMANCE_STAGE=PASS")


if __name__ == "__main__":
    main()
