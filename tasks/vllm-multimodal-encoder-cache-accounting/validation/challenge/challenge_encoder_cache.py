#!/usr/bin/env python3
"""Independent fresh challenge for the encoder-cache embedding-row accounting.

This harness is curator-side only. It is NOT the agent verifier, is NOT mounted
into the agent image, and is NOT referenced by instruction.md. It enters through
the same production boundary the task defines (PlaceholderRange, the
EncoderCacheManager, and Scheduler._try_schedule_encoder_inputs) but on cases
derived independently of tests/verify_encoder_cache.py and validation/ci-cases.

Phase D runs it against Oracle and the semantically different correct alternate
(alternate-direct-mask-count.patch); both must print CHALLENGE_ENCODER_CACHE=PASS
and exit 0. Base and the incorrect partial-budget-omission control must fail it.

The embedding-count and subrange accessors are re-derived here (property- and
method-neutral) so the challenge does not depend on the hidden verifier's
helpers or prescribe an internal representation.
"""
from __future__ import annotations

import inspect
import json
import sys
from types import SimpleNamespace

import torch

from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
from vllm.v1.core.encoder_cache_manager import EncoderCacheManager
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request


def mask_from(length: int, indices: list[int]) -> torch.Tensor:
    m = torch.zeros(length, dtype=torch.bool)
    if indices:
        m[torch.tensor(indices)] = True
    return m


def embedding_count(position, expected: int) -> int:
    """Accept a semantically named property or zero-arg method; no golden shape."""
    for name in dir(position):
        lowered = name.lower()
        if name.startswith("_") or "embed" not in lowered:
            continue
        if "num" not in lowered and "count" not in lowered:
            continue
        member = getattr(position, name)
        if callable(member):
            try:
                if len(inspect.signature(member).parameters) != 0:
                    continue
                value = member()
            except (TypeError, ValueError):
                continue
        else:
            value = member
        if isinstance(value, int) and not isinstance(value, bool) and value == expected:
            return value
    raise AssertionError(
        f"no public embedding-count behavior returned {expected} for {position!r}"
    )


def embedding_subrange(position, start: int, end: int, expected: tuple[int, int]):
    """Find a 2-arg embedding subrange mapping, accepting absolute or relative."""
    offset = getattr(position, "offset", 0)
    inputs = [(start, end)]
    absolute = (start + offset, end + offset)
    if absolute not in inputs:
        inputs.append(absolute)
    for name in dir(position):
        if name.startswith("_") or "embed" not in name.lower():
            continue
        member = getattr(position, name)
        if not callable(member):
            continue
        try:
            if len(inspect.signature(member).parameters) != 2:
                continue
        except (TypeError, ValueError):
            continue
        for cand_start, cand_end in inputs:
            try:
                value = member(cand_start, cand_end)
            except (AssertionError, IndexError, TypeError, ValueError):
                continue
            if isinstance(value, tuple) and tuple(value) == tuple(expected):
                return tuple(value)
    raise AssertionError(
        f"no public embedding subrange returned {expected} for inputs {inputs}"
    )


class SparseRequest(Request):
    """Minimal production Request shape; no prescribed count API."""

    has_encoder_inputs = True

    def __init__(self, request_id: str, masks: list[torch.Tensor | None]):
        self.request_id = request_id
        self.mm_features = []
        for index, mask in enumerate(masks):
            length = 7 if mask is None else len(mask)
            self.mm_features.append(
                MultiModalFeatureSpec(
                    data=None,
                    modality="image",
                    identifier=f"{request_id}-item-{index}",
                    mm_position=PlaceholderRange(0, length, mask),
                )
            )


def prefix_true(mask: torch.Tensor, upto: int) -> int:
    return int(mask[:upto].sum().item())


def challenge_counts() -> dict:
    # Fresh masks/lengths not present in the verifier inventory.
    cases = [
        (PlaceholderRange(0, 9, None), 9),
        (PlaceholderRange(0, 7, torch.ones(7, dtype=torch.bool)), 7),
        (PlaceholderRange(0, 12, torch.zeros(12, dtype=torch.bool)), 0),
        (PlaceholderRange(0, 64, mask_from(64, [0, 6, 13, 20, 27, 34, 41, 48, 55, 62, 63])), 11),
    ]
    for position, expected in cases:
        assert embedding_count(position, expected) == expected
    return {"cases": len(cases)}


def challenge_subrange() -> dict:
    # Straddle a mask gap; expected compact range derived by prefix sums here.
    indices = [2, 5, 6, 11, 14]
    length = 16
    m = mask_from(length, indices)
    position = PlaceholderRange(20, length, m)
    windows = [(4, 12), (0, 6), (6, 15), (14, 16)]
    # Compact encoder output: one row per selected embedding (row i holds a
    # unique fingerprint 3i..3i+2), so a slice's identity is checkable.
    compact = torch.arange(len(indices) * 3).reshape(len(indices), 3)
    for start, end in windows:
        expected = (prefix_true(m, start), prefix_true(m, end))
        got = embedding_subrange(position, start, end, expected)
        assert got == expected, (start, end, got, expected)
        # A compact encoder-output slice must be taken in EMBEDDING coordinates
        # (lo:hi), which selects exactly the embedding rows for this prompt
        # window -- not the prompt-token coordinates (start:end).
        lo, hi = expected
        embedding_slice = compact[lo:hi]
        golden = torch.arange(lo * 3, hi * 3).reshape(hi - lo, 3)
        torch.testing.assert_close(embedding_slice, golden)
        # Where the mask actually compresses (prompt window != embedding window),
        # slicing by prompt coordinates would pick the wrong rows/shape. Assert
        # the embedding slice is genuinely distinct from the prompt-coord slice.
        if (start, end) != (lo, hi):
            prompt_slice = compact[start:end]
            mismatched = (
                prompt_slice.shape != embedding_slice.shape
                or not torch.equal(prompt_slice, embedding_slice)
            )
            assert mismatched, (start, end, lo, hi)
    return {"windows": len(windows)}


def challenge_cache_lifecycle() -> dict:
    # cache_size chosen so the first request fills it to the row.
    first = SparseRequest("c-first", [mask_from(64, [0, 6, 13, 20, 27, 34, 41, 48, 55, 62, 63])])
    manager = EncoderCacheManager(cache_size=11)
    assert manager.can_allocate(first, 0, 11, 0)
    manager.allocate(first, 0)
    assert manager.num_free_slots == 0
    assert first.mm_features[0].identifier in manager.cached

    manager.free_encoder_input(first, 0)
    # Freeing is lazy: the freed reference makes capacity reclaimable but the
    # entry is not physically evicted until a later allocation needs the room.
    # Assert that reclaimable behavior directly -- the next request's fresh
    # embedding rows must fit only because the first entry's rows were released.
    second = SparseRequest("c-second", [mask_from(30, [3, 9, 21, 28])])
    assert manager.can_allocate(second, 0, 4, 0)
    manager.allocate(second, 0)
    assert manager.num_free_slots == 7
    assert second.mm_features[0].identifier in manager.cached
    return {"free_slots": manager.num_free_slots}


def challenge_multi_item_zero() -> dict:
    request = SparseRequest(
        "c-multi",
        [
            mask_from(24, [1, 8, 15, 22]),   # 4 rows
            None,                            # mask-free, length 7 -> 7 rows
            torch.zeros(9, dtype=torch.bool),  # 0 rows
        ],
    )
    manager = EncoderCacheManager(cache_size=11)
    assert manager.can_allocate(request, 0, 4, 0)
    manager.allocate(request, 0)
    assert manager.can_allocate(request, 1, 7, 0)
    manager.allocate(request, 1)
    assert manager.num_free_slots == 0
    free_before_zero = manager.num_free_slots
    assert manager.can_allocate(request, 2, 0, 0)
    manager.allocate(request, 2)
    assert manager.num_free_slots == free_before_zero
    return {"items": 3, "rows": 11}


def challenge_scheduler_partial_budget() -> dict:
    class CacheSpy:
        def __init__(self):
            self.calls = []

        @staticmethod
        def check_and_update_cache(request, input_id):
            return False

        def can_allocate(self, request, input_id, budget, already_scheduled):
            self.calls.append((input_id, budget, already_scheduled))
            return True

    request = SparseRequest("c-sched", [mask_from(80, [4, 12, 33, 47, 61])])
    scheduler = object.__new__(Scheduler)
    scheduler.ec_connector = None
    scheduler.is_encoder_decoder = False
    scheduler.encoder_cache_manager = CacheSpy()
    scheduler.scheduler_config = SimpleNamespace(disable_chunked_mm_input=False)

    # Prompt window covering the whole placeholder selects 5 embedding rows and
    # the encoder budget must be measured in embedding rows, not prompt tokens.
    scheduled, num_new, budget, external = Scheduler._try_schedule_encoder_inputs(
        scheduler,
        request,
        num_computed_tokens=0,
        num_new_tokens=80,
        encoder_compute_budget=5,
    )
    assert scheduled == [0] and num_new == 80 and budget == 0 and external == []
    assert (0, 5, 0) in scheduler.encoder_cache_manager.calls

    # A prompt window between selected positions holds zero embedding rows and
    # must consume no budget and schedule no encoder input. The behavioral
    # contract is that nothing is scheduled and the budget is untouched; we do
    # not over-specify whether the manager is consulted en route.
    scheduler.encoder_cache_manager.calls.clear()
    scheduled, _, budget, _ = Scheduler._try_schedule_encoder_inputs(
        scheduler,
        request,
        num_computed_tokens=13,
        num_new_tokens=19,
        encoder_compute_budget=5,
    )
    assert scheduled == [] and budget == 5
    return {"budget_units": "embedding_rows"}


def main() -> None:
    stages = {
        "counts": challenge_counts,
        "subrange": challenge_subrange,
        "cache_lifecycle": challenge_cache_lifecycle,
        "multi_item_zero": challenge_multi_item_zero,
        "scheduler_partial_budget": challenge_scheduler_partial_budget,
    }
    passed = {}
    failures = {}
    for name, fn in stages.items():
        try:
            passed[name] = fn()
        except Exception as exc:  # noqa: BLE001 - report every stage
            import traceback

            failures[name] = {"type": type(exc).__name__, "message": str(exc),
                              "traceback": traceback.format_exc()}
    print(json.dumps({"passed": passed, "failures": failures}, indent=2, sort_keys=True))
    if failures:
        print("CHALLENGE_ENCODER_CACHE=FAIL")
        sys.exit(1)
    print("CHALLENGE_ENCODER_CACHE=PASS")


if __name__ == "__main__":
    main()
