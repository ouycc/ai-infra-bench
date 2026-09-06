#!/usr/bin/env python3
"""Behavioral verifier for prompt-space/embedding-space cache migration."""

from __future__ import annotations

import inspect
import json
import traceback
from types import SimpleNamespace

import torch

from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
import vllm.multimodal.registry as registry_module
from vllm.multimodal.registry import MultiModalRegistry
from vllm.multimodal.profiling import MultiModalProfiler
from vllm.v1.core.encoder_cache_manager import EncoderCacheManager
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request
from vllm.v1.worker.gpu_model_runner import GPUModelRunner


def read_embedding_count(position, expected: int) -> int:
    """Accept a semantically named property or method, not one Golden shape."""

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


def map_embedding_subrange(position, start: int, end: int, expected):
    """Find range mapping without prescribing a relative/absolute API.

    The product behavior is defined in absolute prompt space, while a helper
    may reasonably accept either absolute prompt coordinates or coordinates
    relative to its own placeholder.  Both conventions are accepted when they
    produce the same compact embedding subrange.
    """

    inputs = [(start, end)]
    offset = getattr(position, "offset", 0)
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
            parameter_count = len(inspect.signature(member).parameters)
        except (TypeError, ValueError):
            continue
        if parameter_count != 2:
            continue
        for candidate_start, candidate_end in inputs:
            try:
                value = member(candidate_start, candidate_end)
            except (AssertionError, IndexError, TypeError, ValueError):
                continue
            if isinstance(value, tuple) and tuple(value) == tuple(expected):
                return tuple(value)
    raise AssertionError(
        "no public embedding range mapping returned "
        f"{expected} for relative/absolute inputs {inputs}"
    )


def read_registry_embedding_capacity(registry, model_config, expected):
    """Observe embedding-row capacity without forcing an old token API to change.

    A correct implementation may preserve the prompt-token accessor and add a
    separate embedding-capacity API.  Prefer any public, semantically named
    embedding method, then accept the legacy registry method only if its
    behavior itself reports embedding rows.
    """

    attempts = []
    for name in dir(registry):
        lowered = name.lower()
        if name.startswith("_"):
            continue
        if not all(part in lowered for part in ("max", "embed", "item")):
            continue
        member = getattr(registry, name)
        if not callable(member):
            continue
        invocations = (
            lambda: member(model_config, profiler_limits={"image": 1}),
            lambda: member(model_config),
        )
        for invoke in invocations:
            try:
                measured = invoke()
            except (AttributeError, TypeError, ValueError):
                continue
            attempts.append((name, measured))
            if measured == expected:
                return measured, name

    measured = MultiModalRegistry.get_max_tokens_per_item_by_modality(
        registry,
        model_config,
        profiler_limits={"image": 1},
    )
    attempts.append(("get_max_tokens_per_item_by_modality", measured))
    if measured == expected:
        return measured, "get_max_tokens_per_item_by_modality"
    raise AssertionError(
        f"no public registry behavior reported embedding capacity {expected}: "
        f"{attempts}"
    )


class SparseRequest(Request):
    """Minimal production Request shape using the candidate's own count API."""

    has_encoder_inputs = True

    def __init__(self, request_id: str, masks: list[torch.Tensor | None]):
        self.request_id = request_id
        self.mm_features = []
        for index, mask in enumerate(masks):
            length = 5 if mask is None else len(mask)
            self.mm_features.append(
                MultiModalFeatureSpec(
                    data=None,
                    modality="image",
                    identifier=f"{request_id}-item-{index}",
                    mm_position=PlaceholderRange(0, length, mask),
                )
            )

def sparse_mask(length: int, indices: list[int]) -> torch.Tensor:
    mask = torch.zeros(length, dtype=torch.bool)
    if indices:
        mask[torch.tensor(indices)] = True
    return mask


def check_placeholder_coordinates():
    cases = [
        (PlaceholderRange(0, 5, None), 5),
        (PlaceholderRange(0, 5, torch.ones(5, dtype=torch.bool)), 5),
        (PlaceholderRange(0, 5, torch.zeros(5, dtype=torch.bool)), 0),
        (PlaceholderRange(0, 5, sparse_mask(5, [1, 3, 4])), 3),
    ]
    for position, expected in cases:
        assert read_embedding_count(position, expected) == expected
    return {"cases": len(cases), "property_or_method_neutral": True}


def check_partial_mapping():
    sparse = PlaceholderRange(10, 5, sparse_mask(5, [1, 3, 4]))
    cases = [
        (sparse, 0, 2, (0, 1)),
        (sparse, 2, 2, (1, 1)),
        (sparse, 3, 5, (1, 3)),
        (PlaceholderRange(0, 5, None), 2, 4, (2, 4)),
        (
            PlaceholderRange(0, 4, torch.zeros(4, dtype=torch.bool)),
            0,
            4,
            (0, 0),
        ),
    ]
    for position, start, end, expected in cases:
        assert map_embedding_subrange(position, start, end, expected) == expected

    # Compact encoder output slicing uses embedding coordinates, while prompt
    # overlap remains in prompt coordinates.
    compact_output = torch.arange(12).reshape(3, 4)
    embed_start, embed_end = map_embedding_subrange(sparse, 3, 5, (1, 3))
    torch.testing.assert_close(compact_output[embed_start:embed_end], compact_output[1:3])
    return {"cases": len(cases), "compact_slice_rows": 2}


def check_cache_lifecycle():
    first = SparseRequest("first", [sparse_mask(100, [5, 15, 25, 35, 45, 55, 65, 75])])
    manager = EncoderCacheManager(cache_size=8)
    assert manager.can_allocate(first, 0, 8, 0)
    manager.allocate(first, 0)
    assert manager.num_free_slots == 0
    assert first.mm_features[0].identifier in manager.cached

    manager.free_encoder_input(first, 0)
    second = SparseRequest("second", [sparse_mask(20, [2, 7, 12, 17])])
    assert manager.can_allocate(second, 0, 4, 0)
    assert first.mm_features[0].identifier in manager.freed
    manager.allocate(second, 0)
    assert manager.num_free_slots == 4
    assert second.mm_features[0].identifier in manager.cached
    return {"eviction": True, "free_slots": manager.num_free_slots}


def check_multiple_items_and_zero_embeddings():
    request = SparseRequest(
        "multi",
        [
            sparse_mask(10, [1, 4, 7, 9]),
            None,
            torch.zeros(6, dtype=torch.bool),
        ],
    )
    manager = EncoderCacheManager(cache_size=9)
    assert manager.can_allocate(request, 0, 4, 0)
    manager.allocate(request, 0)
    assert manager.can_allocate(request, 1, 5, 0)
    manager.allocate(request, 1)
    assert manager.num_free_slots == 0
    slots_before_zero = manager.num_free_slots
    assert manager.can_allocate(request, 2, 0, 0)
    manager.allocate(request, 2)
    assert manager.num_free_slots == slots_before_zero
    return {"items": 3, "allocated_embedding_rows": 9, "zero_item": True}


def check_scheduler_partial_budget():
    class CacheSpy:
        def __init__(self):
            self.calls = []

        @staticmethod
        def check_and_update_cache(request, input_id):
            return False

        def can_allocate(
            self, request, input_id, encoder_compute_budget, already_scheduled
        ):
            self.calls.append(
                (input_id, encoder_compute_budget, already_scheduled)
            )
            return True

    request = SparseRequest("scheduler", [sparse_mask(100, [5, 15, 25, 35])])
    scheduler = object.__new__(Scheduler)
    scheduler.ec_connector = None
    scheduler.is_encoder_decoder = False
    scheduler.encoder_cache_manager = CacheSpy()
    scheduler.scheduler_config = SimpleNamespace(disable_chunked_mm_input=False)

    scheduled, num_new, budget, external = Scheduler._try_schedule_encoder_inputs(
        scheduler,
        request,
        num_computed_tokens=15,
        num_new_tokens=7,
        encoder_compute_budget=4,
    )
    assert scheduled == [0] and num_new == 7 and budget == 0 and external == []
    assert scheduler.encoder_cache_manager.calls == [(0, 4, 0)]

    # This prompt-space overlap contains no embedding rows. It must not consume
    # encoder budget or schedule a compact encoder output.
    scheduler.encoder_cache_manager.calls.clear()
    scheduled, _, budget, _ = Scheduler._try_schedule_encoder_inputs(
        scheduler,
        request,
        num_computed_tokens=16,
        num_new_tokens=6,
        encoder_compute_budget=4,
    )
    assert scheduled == [] and budget == 4
    return {"partial_embedding_overlap": True, "budget_units": "embedding_rows"}


def check_model_runner_compact_gather():
    position = PlaceholderRange(0, 5, sparse_mask(5, [1, 3, 4]))
    feature = MultiModalFeatureSpec(
        data=None,
        modality="image",
        identifier="compact",
        mm_position=position,
    )
    compact = torch.arange(12, dtype=torch.float32).reshape(3, 4)

    class BoolBuffer:
        def __init__(self):
            self.cpu = torch.empty(2, dtype=torch.bool)

        def copy_to_gpu(self, count):
            return self.cpu[:count].clone()

    runner = object.__new__(GPUModelRunner)
    runner.input_batch = SimpleNamespace(req_ids=["req"])
    runner.requests = {
        "req": SimpleNamespace(
            num_computed_tokens=3,
            mm_features=[feature],
        )
    }
    runner.encoder_cache = {"compact": compact}
    runner.is_mm_embed = BoolBuffer()
    runner.is_multimodal_pruning_enabled = False
    runner.uses_mrope = False
    scheduler_output = SimpleNamespace(
        total_num_scheduled_tokens=2,
        num_scheduled_tokens={"req": 2},
    )
    gathered, mask = GPUModelRunner._gather_mm_embeddings(runner, scheduler_output)
    assert len(gathered) == 1
    torch.testing.assert_close(gathered[0], compact[1:3])
    assert mask.tolist() == [True, True]
    return {"prompt_range": [3, 5], "compact_embedding_range": [1, 3]}


def check_registry_profiles_embedding_capacity():
    original_profiler = registry_module.MultiModalProfiler

    class ControlledProcessingInfo:
        """Return None so the production profiler takes the dummy-input path
        instead of a precomputed per-item count. get_mm_max_tokens_per_item is
        a pre-existing public BaseProcessingInfo API, not a Golden-private
        symbol (the leakage audit matches on identifier boundaries)."""

        def get_mm_max_tokens_per_item(self, *, seq_len, mm_counts):
            del seq_len, mm_counts
            return None

    class ControlledProcessor:
        info = ControlledProcessingInfo()

    class ControlledProfiler(MultiModalProfiler):
        """Exercise the candidate profiler API on a controlled sparse input.

        __init__ is inherited so the parent stores ``self.processor`` and the
        production ``processing_info`` property resolves. The dummy-input hook
        is overridden to short-circuit the real processor.apply pipeline while
        still exercising the candidate embedding-count semantics."""

        @staticmethod
        def get_mm_limits():
            return {"image": 1}

        def _get_dummy_mm_inputs(self, seq_len, mm_counts=None, mm_options=None):
            del seq_len, mm_counts, mm_options
            mask = sparse_mask(100, [5, 15, 25, 35, 45, 55, 65, 75])
            return {
                "mm_placeholders": {
                    "image": [PlaceholderRange(0, 100, mask)],
                }
            }

    registry = object.__new__(MultiModalRegistry)
    registry.create_processor = lambda model_config, cache=None: ControlledProcessor()
    model_config = SimpleNamespace(is_multimodal_model=True, max_model_len=4096)
    registry_module.MultiModalProfiler = ControlledProfiler
    try:
        measured, capacity_api = read_registry_embedding_capacity(
            registry,
            model_config,
            {"image": 8},
        )
    finally:
        registry_module.MultiModalProfiler = original_profiler
    assert measured == {"image": 8}, measured
    return {
        "profiled_prompt_tokens": 100,
        "profiled_embedding_rows": 8,
        "capacity_api": capacity_api,
    }


def main() -> None:
    stages = {
        "placeholder_coordinates": check_placeholder_coordinates,
        "partial_mapping": check_partial_mapping,
        "cache_lifecycle": check_cache_lifecycle,
        "multiple_items": check_multiple_items_and_zero_embeddings,
        "scheduler_partial_budget": check_scheduler_partial_budget,
        "model_runner_compact_gather": check_model_runner_compact_gather,
        "registry_capacity": check_registry_profiles_embedding_capacity,
    }
    passed = {}
    failures = {}
    for name, check in stages.items():
        try:
            passed[name] = check()
        except Exception as exc:
            failures[name] = {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
    print(
        json.dumps(
            {
                "coordinate_spaces": ["prompt", "embedding"],
                "failures": failures,
                "private_property_form_scored": False,
                "stages": passed,
            },
            indent=2,
            sort_keys=True,
        )
    )
    if failures:
        raise AssertionError(f"encoder-cache stages failed: {sorted(failures)}")
    print("ENCODER_CACHE_VERIFIER=PASS")


if __name__ == "__main__":
    main()
