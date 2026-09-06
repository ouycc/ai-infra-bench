#!/usr/bin/env python3
"""Independent fresh challenge for async-PP sampled-token handoff (2-rank NCCL).

Curator-side only: not the agent verifier, not mounted into the agent image,
not referenced by instruction.md. Launch under torchrun with 2 ranks on the
built A100 image:

    torchrun --nproc_per_node=2 challenge_token_handoff.py

It re-enters the production lifecycle (GPUModelRunner.sample_tokens with a
real NCCL PP group) but re-derives the handoff invariants on a FRESH scenario
distinct from the verifier's basic/reordered/integrated:

  * 5 requests, discards interleaved at positions 1 and 3 (verifier only ever
    discards a trailing request), req_ids reordered, all with non-empty prior
    outputs, fresh token ids.

Invariants re-derived (not copied from the verifier body):
  1. Sampled tokens cross the PP boundary via a GPU (is_cuda) collective;
     every object/CPU collective (broadcast_tensor_dict, *_object) is forbidden.
  2. No GPU->CPU scalar synchronization (_local_scalar_dense) occurs.
  3. Receiver rebuilds prev_sampled_token_ids on-GPU matching sent tokens.
  4. prev_req_id_to_index maps exactly the kept (non-discarded) requests to
     their ORIGINAL positional index.
  5. Kept requests get a -1 placeholder appended to output_token_ids;
     discarded requests are left unchanged.

Failure model (fail-closed, machine-checkable per README contract):
  * Every invariant this challenge OWNS is signalled by raising the typed
    ``ChallengeInvariantFailure(code, detail)``. When any rank raises one, the
    ranks agree via a GPU all-reduce, rank0 emits a structured JSON record and
    exactly one ``CHALLENGE_TOKEN_HANDOFF=FAIL`` line (flushed BEFORE the
    barrier so a fast peer exit cannot make torchrun kill rank0 and drop the
    marker), and both ranks exit non-zero.
  * Any OTHER exception (import, CUDA, NCCL, a missing fixture attribute, ...)
    propagates untyped: no FAIL marker is printed, so the harness classifies it
    as an infrastructure failure rather than a valid control outcome. This is
    deliberate -- a generic "crashed" signal must NOT be accepted as evidence
    that the implementation failed the behavioral contract.

Phase D runs this against Oracle (PASS) and a semantically-different correct
alternative (PASS); Base and the incorrect/perf-only controls must emit exactly
one CHALLENGE_TOKEN_HANDOFF=FAIL with a typed reason_code.
"""
from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist

import vllm.v1.worker.gpu_model_runner as runner_module
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

# Fresh scenario: interleaved discards, reordered ids, all non-empty priors.
TOKENS = [111, 222, 333, 444, 555]
REQ_IDS = ["r-c", "r-a", "r-e", "r-b", "r-d"]
DISCARD_MASK = [False, True, False, True, False]
PRIOR_OUTPUTS = {
    "r-c": [3],
    "r-a": [5, 6],
    "r-e": [7],
    "r-b": [9, 10, 11],
    "r-d": [13],
}


class ChallengeInvariantFailure(RuntimeError):
    """A handoff invariant this challenge OWNS was violated.

    Distinct from arbitrary runtime errors: only this type is accepted as a
    valid control failure and lowered to a CHALLENGE_TOKEN_HANDOFF=FAIL marker.
    """

    def __init__(self, code: str, detail: str):
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


class BroadcastObserved(RuntimeError):
    """Stops sender bookkeeping once the GPU collective has run."""


def pp_group(rank: int, world_size: int):
    def reject_tensor_dict(*args, **kwargs):
        raise ChallengeInvariantFailure(
            "object_tensor_dict_forbidden",
            "CPU/object tensor-dict path is forbidden for sampled-token handoff",
        )

    return SimpleNamespace(
        rank=rank,
        world_size=world_size,
        last_rank=world_size - 1,
        is_last_rank=rank == world_size - 1,
        device_group=dist.group.WORLD,
        broadcast_tensor_dict=reject_tensor_dict,
    )


def _guard_bookkeeping_without_handoff(*args, **kwargs):
    """Installed as runner._bookkeeping_sync.

    A correct sender hands the sampled tokens across the PP boundary on the GPU
    BEFORE bookkeeping (the oracle raises BroadcastObserved there and never
    reaches this point). Reaching bookkeeping on the sender therefore means the
    implementation skipped the GPU handoff -- a typed invariant failure, not a
    crash from an under-populated fixture.
    """
    raise ChallengeInvariantFailure(
        "sender_reached_bookkeeping_without_gpu_handoff",
        "sampled tokens were not handed off on the GPU before bookkeeping",
    )


def sender_runner(sampled: torch.Tensor):
    runner = object.__new__(GPUModelRunner)
    runner.kv_connector_output = None
    runner.use_async_scheduling = True
    # First slot is scheduler_output; populate the field the production
    # bookkeeping call reads (scheduler_output.total_num_scheduled_tokens) so a
    # sender that reaches bookkeeping fails via the typed guard below rather
    # than an unrelated AttributeError from an empty namespace.
    runner.execute_model_state = (
        SimpleNamespace(total_num_scheduled_tokens=len(TOKENS)),
        torch.empty(1, device="cuda"),
        None, None, None, None, None, None, None, None,
    )
    runner._sample = lambda logits, metadata: SimpleNamespace(
        sampled_token_ids=sampled
    )
    runner._update_states_after_model_execute = lambda token_ids, output: None
    # Boundary guard: reaching bookkeeping means no GPU handoff happened.
    runner._bookkeeping_sync = _guard_bookkeeping_without_handoff
    runner.input_batch = SimpleNamespace(prev_sampled_token_ids=None)
    runner._draft_token_ids = None
    runner._draft_token_req_ids = None
    runner.speculative_config = None
    return runner


def receiver_runner():
    runner = object.__new__(GPUModelRunner)
    runner.kv_connector_output = None
    runner.execute_model_state = None
    runner.use_async_scheduling = True
    runner.device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    runner._bookkeeping_sync = _guard_bookkeeping_without_handoff
    runner.input_batch = SimpleNamespace(
        num_reqs=len(REQ_IDS),
        req_ids=list(REQ_IDS),
        prev_sampled_token_ids=None,
        prev_req_id_to_index=None,
    )
    runner.discard_request_mask = SimpleNamespace(
        np=np.asarray(DISCARD_MASK, dtype=np.bool_)
    )
    runner.requests = {
        req_id: SimpleNamespace(output_token_ids=list(PRIOR_OUTPUTS[req_id]))
        for req_id in REQ_IDS
    }
    return runner


def invoke_production_sample(runner, *, is_sender: bool):
    orig_broadcast = dist.broadcast
    orig_object_list = dist.broadcast_object_list
    orig_all_gather = dist.all_gather_object
    orig_gather = dist.gather_object
    orig_scatter = dist.scatter_object_list
    collective_seen = False

    def gpu_broadcast(tensor, *args, **kwargs):
        nonlocal collective_seen
        if not torch.is_tensor(tensor) or not tensor.is_cuda:
            raise ChallengeInvariantFailure(
                "sampled_tokens_left_gpu",
                "sampled tokens left the GPU before the PP transfer",
            )
        result = orig_broadcast(tensor, *args, **kwargs)
        collective_seen = True
        if is_sender:
            raise BroadcastObserved
        return result

    def reject_object(*args, **kwargs):
        raise ChallengeInvariantFailure(
            "object_collective_forbidden",
            "Python/object collective used for sampled tokens",
        )

    dist.broadcast = gpu_broadcast
    dist.broadcast_object_list = reject_object
    dist.all_gather_object = reject_object
    dist.gather_object = reject_object
    dist.scatter_object_list = reject_object
    try:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU], with_stack=True,
        ) as profile:
            try:
                output = GPUModelRunner.sample_tokens(runner, None)
            except BroadcastObserved:
                assert is_sender
                output = None
        scalar_sync = [
            getattr(e, "name", getattr(e, "key", ""))
            for e in profile.events()
            if "_local_scalar_dense" in getattr(e, "name", getattr(e, "key", ""))
        ]
        if scalar_sync:
            raise ChallengeInvariantFailure(
                "gpu_cpu_scalar_sync",
                f"GPU->CPU scalar sync in handoff: {scalar_sync}",
            )
    finally:
        dist.broadcast = orig_broadcast
        dist.broadcast_object_list = orig_object_list
        dist.all_gather_object = orig_all_gather
        dist.gather_object = orig_gather
        dist.scatter_object_list = orig_scatter
    if not collective_seen:
        raise ChallengeInvariantFailure(
            "no_gpu_broadcast",
            "production lifecycle did not use a GPU broadcast for handoff",
        )
    return output


def _run_scenario(rank: int) -> dict:
    """Drive one rank through the production handoff. Returns the record dict.

    Raises ChallengeInvariantFailure for any owned-invariant violation.
    """
    if rank == 1:
        sampled = torch.tensor(
            TOKENS, dtype=torch.int32, device="cuda"
        ).reshape(-1, 1)
        invoke_production_sample(sender_runner(sampled), is_sender=True)
        return {"sent": TOKENS}

    runner = receiver_runner()
    output = invoke_production_sample(runner, is_sender=False)
    if output is not None:
        raise ChallengeInvariantFailure(
            "receiver_produced_output",
            "receiver rank unexpectedly produced a model output",
        )
    received = runner.input_batch.prev_sampled_token_ids
    if received is None or not received.is_cuda:
        raise ChallengeInvariantFailure(
            "receiver_missing_gpu_tokens",
            "receiver did not rebuild prev_sampled_token_ids on the GPU",
        )
    if received.cpu().flatten().tolist() != TOKENS:
        raise ChallengeInvariantFailure(
            "received_tokens_mismatch",
            f"received {received.cpu().flatten().tolist()} != sent {TOKENS}",
        )
    expected_mapping = {
        req_id: index
        for index, req_id in enumerate(REQ_IDS)
        if not DISCARD_MASK[index]
    }
    if runner.input_batch.prev_req_id_to_index != expected_mapping:
        raise ChallengeInvariantFailure(
            "req_id_mapping_mismatch",
            f"{runner.input_batch.prev_req_id_to_index} != {expected_mapping}",
        )
    for index, req_id in enumerate(REQ_IDS):
        expected = list(PRIOR_OUTPUTS[req_id])
        if not DISCARD_MASK[index]:
            expected.append(-1)
        if runner.requests[req_id].output_token_ids != expected:
            raise ChallengeInvariantFailure(
                "output_token_ids_mismatch",
                f"{req_id}: {runner.requests[req_id].output_token_ids} != {expected}",
            )
    return {
        "received": TOKENS,
        "mapping": expected_mapping,
        "discarded": [r for i, r in enumerate(REQ_IDS) if DISCARD_MASK[i]],
    }


def run_rank() -> int:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 2, "challenge requires exactly 2 ranks"

    group = pp_group(rank, world_size)
    original_get_pp_group = runner_module.get_pp_group
    runner_module.get_pp_group = lambda: group

    failure: ChallengeInvariantFailure | None = None
    record: dict = {}
    try:
        try:
            record = _run_scenario(rank)
        except ChallengeInvariantFailure as exc:
            failure = exc
    finally:
        runner_module.get_pp_group = original_get_pp_group

    # Agree on failure across ranks via a GPU collective (no CPU/object path).
    flag = torch.tensor(
        [1.0 if failure is not None else 0.0], device="cuda"
    )
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    any_failure = flag.item() > 0.5

    if any_failure:
        # rank0 owns the marker. Prefer its own typed reason; if only the peer
        # failed, report that explicitly. Print+flush BEFORE the barrier.
        if rank == 0:
            if failure is not None:
                reason, detail = failure.code, failure.detail
            else:
                reason = "peer_rank_invariant_failure"
                detail = "a peer rank raised a challenge invariant failure"
            print(json.dumps({
                "verdict": "FAIL",
                "reason_code": reason,
                "detail": detail,
                "scenario": "fresh-interleaved-discard-5req",
                "world_size": world_size,
            }, sort_keys=True), flush=True)
            print("CHALLENGE_TOKEN_HANDOFF=FAIL", flush=True)
        dist.barrier()
        dist.destroy_process_group()
        return 1

    dist.barrier()
    if rank == 0:
        print(json.dumps({
            "gpu": torch.cuda.get_device_properties(local_rank).name,
            "production_entrypoint": "GPUModelRunner.sample_tokens",
            "scenario": "fresh-interleaved-discard-5req",
            "scenario_result": record,
            "verdict": "PASS",
            "world_size": world_size,
        }, sort_keys=True), flush=True)
        print("CHALLENGE_TOKEN_HANDOFF=PASS", flush=True)
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(run_rank())
