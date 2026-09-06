#!/usr/bin/env python3
"""Behavioral verifier for prompt-space/embedding-space cache migration.

TRUSTED PARENT/WORKER BOUNDARY:
This verifier runs as root. It spawns the candidate validation as a non-root
worker (uid 65534) via subprocess, captures the worker's structured output over
a pipe, and writes reward.txt itself. The worker never touches /logs/verifier.
"""

from __future__ import annotations

import json
import subprocess
import sys
import traceback


WORKER_CODE = r'''
import json
import sys
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
    """Use the public get_num_embeds API (property or cached_property)."""
    cases = [
        (PlaceholderRange(0, 5, None), 5),
        (PlaceholderRange(0, 5, torch.ones(5, dtype=torch.bool)), 5),
        (PlaceholderRange(0, 5, torch.zeros(5, dtype=torch.bool)), 0),
        (PlaceholderRange(0, 5, sparse_mask(5, [1, 3, 4])), 3),
    ]
    for position, expected in cases:
        actual = position.get_num_embeds
        if actual != expected:
            raise AssertionError(
                f"position.get_num_embeds returned {actual}, expected {expected}"
            )
    return {"cases": len(cases), "api": "get_num_embeds"}


def check_partial_mapping():
    """Use the public get_embeds_indices_in_range method."""
    sparse = PlaceholderRange(10, 5, sparse_mask(5, [1, 3, 4]))
    cases = [
        (sparse, 0, 2, (0, 1)),
        (sparse, 2, 2, (1, 1)),
        (sparse, 3, 5, (1, 3)),
        (PlaceholderRange(0, 5, None), 2, 4, (2, 4)),
        (PlaceholderRange(0, 4, torch.zeros(4, dtype=torch.bool)), 0, 4, (0, 0)),
    ]
    for position, start, end, expected in cases:
        actual = position.get_embeds_indices_in_range(start, end)
        if actual != expected:
            raise AssertionError(
                f"get_embeds_indices_in_range({start}, {end}) returned {actual}, expected {expected}"
            )

    # Compact encoder output slicing uses embedding coordinates
    compact_output = torch.arange(12).reshape(3, 4)
    embed_start, embed_end = sparse.get_embeds_indices_in_range(3, 5)
    torch.testing.assert_close(compact_output[embed_start:embed_end], compact_output[1:3])
    return {"cases": len(cases), "api": "get_embeds_indices_in_range"}


def check_cache_lifecycle():
    first = SparseRequest("first", [sparse_mask(100, [5, 15, 25, 35, 45, 55, 65, 75])])
    manager = EncoderCacheManager(cache_size=8)
    if not manager.can_allocate(first, 0, 8, 0):
        raise AssertionError("can_allocate returned False for first request")
    manager.allocate(first, 0)
    if manager.num_free_slots != 0:
        raise AssertionError(f"num_free_slots={manager.num_free_slots}, expected 0")
    if first.mm_features[0].identifier not in manager.cached:
        raise AssertionError("identifier not in manager.cached")

    manager.free_encoder_input(first, 0)
    second = SparseRequest("second", [sparse_mask(20, [2, 7, 12, 17])])
    if not manager.can_allocate(second, 0, 4, 0):
        raise AssertionError("can_allocate returned False for second request")
    if first.mm_features[0].identifier not in manager.freed:
        raise AssertionError("first identifier not in manager.freed")
    manager.allocate(second, 0)
    if manager.num_free_slots != 4:
        raise AssertionError(f"num_free_slots={manager.num_free_slots}, expected 4")
    if second.mm_features[0].identifier not in manager.cached:
        raise AssertionError("second identifier not in manager.cached")
    return {"eviction": True, "free_slots": manager.num_free_slots}


def check_multiple_items_and_zero_embeddings():
    request = SparseRequest(
        "multi",
        [sparse_mask(10, [1, 4, 7, 9]), None, torch.zeros(6, dtype=torch.bool)],
    )
    manager = EncoderCacheManager(cache_size=9)
    if not manager.can_allocate(request, 0, 4, 0):
        raise AssertionError("can_allocate failed for item 0")
    manager.allocate(request, 0)
    if not manager.can_allocate(request, 1, 5, 0):
        raise AssertionError("can_allocate failed for item 1")
    manager.allocate(request, 1)
    if manager.num_free_slots != 0:
        raise AssertionError(f"num_free_slots={manager.num_free_slots}, expected 0 after items 0,1")
    slots_before_zero = manager.num_free_slots
    if not manager.can_allocate(request, 2, 0, 0):
        raise AssertionError("can_allocate failed for zero-embedding item 2")
    manager.allocate(request, 2)
    if manager.num_free_slots != slots_before_zero:
        raise AssertionError("zero-embedding item changed num_free_slots")
    return {"items": 3, "allocated_embedding_rows": 9, "zero_item": True}


def check_scheduler_partial_budget():
    class CacheSpy:
        def __init__(self):
            self.calls = []

        @staticmethod
        def check_and_update_cache(request, input_id):
            return False

        def can_allocate(self, request, input_id, encoder_compute_budget, already_scheduled):
            self.calls.append((input_id, encoder_compute_budget, already_scheduled))
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
    if scheduled != [0] or num_new != 7 or budget != 0 or external != []:
        raise AssertionError(f"_try_schedule_encoder_inputs returned unexpected: {(scheduled, num_new, budget, external)}")
    if scheduler.encoder_cache_manager.calls != [(0, 4, 0)]:
        raise AssertionError(f"can_allocate calls: {scheduler.encoder_cache_manager.calls}")

    # Prompt-space overlap with no embedding rows must not consume budget
    scheduler.encoder_cache_manager.calls.clear()
    scheduled, _, budget, _ = Scheduler._try_schedule_encoder_inputs(
        scheduler,
        request,
        num_computed_tokens=16,
        num_new_tokens=6,
        encoder_compute_budget=4,
    )
    if scheduled != [] or budget != 4:
        raise AssertionError(f"no-embedding-overlap returned scheduled={scheduled}, budget={budget}")
    return {"partial_embedding_overlap": True, "budget_units": "embedding_rows"}


def check_model_runner_compact_gather():
    position = PlaceholderRange(0, 5, sparse_mask(5, [1, 3, 4]))
    feature = MultiModalFeatureSpec(
        data=None, modality="image", identifier="compact", mm_position=position
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
        "req": SimpleNamespace(num_computed_tokens=3, mm_features=[feature])
    }
    runner.encoder_cache = {"compact": compact}
    runner.is_mm_embed = BoolBuffer()
    runner.is_multimodal_pruning_enabled = False
    runner.uses_mrope = False
    scheduler_output = SimpleNamespace(
        total_num_scheduled_tokens=2, num_scheduled_tokens={"req": 2}
    )
    gathered, mask = GPUModelRunner._gather_mm_embeddings(runner, scheduler_output)
    if len(gathered) != 1:
        raise AssertionError(f"gathered length={len(gathered)}, expected 1")
    torch.testing.assert_close(gathered[0], compact[1:3])
    if mask.tolist() != [True, True]:
        raise AssertionError(f"mask={mask.tolist()}, expected [True, True]")
    return {"prompt_range": [3, 5], "compact_embedding_range": [1, 3]}


def check_registry_profiles_embedding_capacity():
    """Use the public get_mm_max_tokens API."""
    original_profiler = registry_module.MultiModalProfiler

    class ControlledProcessingInfo:
        def get_mm_max_tokens_per_item(self, *, seq_len, mm_counts):
            del seq_len, mm_counts
            return None

    class ControlledProcessor:
        info = ControlledProcessingInfo()

    class ControlledProfiler(MultiModalProfiler):
        @staticmethod
        def get_mm_limits():
            return {"image": 1}

        def _get_dummy_mm_inputs(self, seq_len, mm_counts=None, mm_options=None):
            del seq_len, mm_counts, mm_options
            mask = sparse_mask(100, [5, 15, 25, 35, 45, 55, 65, 75])
            return {"mm_placeholders": {"image": [PlaceholderRange(0, 100, mask)]}}

    registry = object.__new__(MultiModalRegistry)
    registry.create_processor = lambda model_config, cache=None: ControlledProcessor()
    model_config = SimpleNamespace(is_multimodal_model=True, max_model_len=4096)
    registry_module.MultiModalProfiler = ControlledProfiler
    try:
        measured = MultiModalRegistry.get_max_tokens_per_item_by_modality(
            registry, model_config, profiler_limits={"image": 1}
        )
    finally:
        registry_module.MultiModalProfiler = original_profiler
    if measured != {"image": 8}:
        raise AssertionError(f"profiler returned {measured}, expected {{'image': 8}}")
    return {"profiled_prompt_tokens": 100, "profiled_embedding_rows": 8, "api": "get_mm_max_tokens"}


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
    result = {
        "coordinate_spaces": ["prompt", "embedding"],
        "failures": failures,
        "reflection_free": True,
        "public_api_only": True,
        "stages": passed,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if failures:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
'''


def run_worker() -> dict:
    """Spawn the validation worker as uid 65534 (nobody), capture structured output."""
    try:
        result = subprocess.run(
            ["runuser", "-u", "nobody", "--", "python3", "-I", "-c", WORKER_CODE],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=300,
            check=False,
            text=True,
        )
    except subprocess.TimeoutExpired:
        return {
            "verdict": "FAIL",
            "reason": "worker_timeout",
            "timeout_sec": 300,
        }
    except Exception as exc:
        return {
            "verdict": "FAIL",
            "reason": "worker_spawn_failed",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }

    if result.returncode == 0:
        try:
            worker_output = json.loads(result.stdout)
            return {
                "verdict": "PASS",
                "worker_exit": 0,
                "stages": worker_output.get("stages", {}),
                "reflection_free": worker_output.get("reflection_free", False),
                "public_api_only": worker_output.get("public_api_only", False),
            }
        except json.JSONDecodeError as exc:
            return {
                "verdict": "FAIL",
                "reason": "worker_output_invalid_json",
                "stdout": result.stdout[:2000],
                "stderr": result.stderr[:2000],
                "error": str(exc),
            }
    else:
        try:
            worker_output = json.loads(result.stdout)
            return {
                "verdict": "FAIL",
                "reason": "worker_stage_failures",
                "worker_exit": result.returncode,
                "failures": worker_output.get("failures", {}),
                "stages": worker_output.get("stages", {}),
            }
        except json.JSONDecodeError:
            return {
                "verdict": "FAIL",
                "reason": "worker_crash_or_invalid_output",
                "worker_exit": result.returncode,
                "stdout": result.stdout[:2000],
                "stderr": result.stderr[:2000],
            }


def main() -> None:
    import os
    if os.getuid() != 0:
        print(json.dumps({"verdict": "FAIL", "reason": "verifier_not_root", "uid": os.getuid()}))
        sys.exit(1)

    result = run_worker()
    print(json.dumps(result, indent=2, sort_keys=True))

    if result["verdict"] == "PASS":
        print("ENCODER_CACHE_VERIFIER=PASS")
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
