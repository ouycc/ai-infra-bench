#!/usr/bin/env python3
"""Behavioral verifier for INT8 per-token-group quantization with frozen baseline.

TRUSTED PARENT/WORKER BOUNDARY:
This verifier runs as root. It spawns the candidate validation as a non-root
worker (uid 65534) via subprocess, captures the worker's structured output over
a pipe, and writes reward.txt itself. The worker never touches /logs/verifier.

FROZEN TRITON BASELINE:
The performance baseline is reference_int8_utils.py, a BYTE-IDENTICAL copy of
vllm/model_executor/layers/quantization/utils/int8_utils.py at the task base commit
(SHA-256 36406a44b95e54cf99988105d0fe9a69645a0d2fcbfe2e60b1982d3ac9fdcff3, verifiable
against upstream git history). It is installed root-owned and read-only under
/opt/ai-infra-bench/reference-int8/ before the agent user is created, so the candidate
cannot modify the baseline, the threshold, or the timing protocol. It is loaded through
frozen_reference_loader.py in a subprocess whose sys.path excludes /workspace/repo.
"""

from __future__ import annotations

import json
import subprocess
import sys
import traceback


WORKER_CODE = r'''
import argparse
import contextlib
import json
import statistics
import sys
import traceback

import torch


def cuda_quant(x, group_size, eps=1e-10, int8_min=-128.0, int8_max=127.0):
    """Invoke the candidate CUDA operator."""
    q = torch.empty_like(x, dtype=torch.int8)
    s = torch.empty(
        x.shape[:-1] + (x.shape[-1] // group_size,),
        device=x.device,
        dtype=torch.float32,
    )
    torch.ops._C.per_token_group_quant_int8(x, q, s, group_size, eps, int8_min, int8_max)
    return q, s


def reference(x, group_size, eps=1e-10, int8_min=-128, int8_max=127):
    """Pure PyTorch reference for correctness validation."""
    g = x.float().reshape(-1, group_size)
    s = g.abs().amax(dim=1).clamp_min(eps) / float(int8_max)
    q = torch.clamp(torch.round(g / s[:, None]), int8_min, int8_max).to(torch.int8)
    return q.reshape_as(x), s.reshape(x.shape[:-1] + (x.shape[-1] // group_size,))


def timed_ms(fn, warmup=40, repeats=400):
    """Time a GPU operation using CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(5):
        begin, end = torch.cuda.Event(True), torch.cuda.Event(True)
        begin.record()
        for _ in range(repeats):
            fn()
        end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end) / repeats)
    return statistics.median(samples)


@contextlib.contextmanager
def replace(module, name, value):
    """Temporarily replace a module attribute."""
    old = getattr(module, name)
    setattr(module, name, value)
    try:
        yield
    finally:
        setattr(module, name, old)


class _TritonBomb:
    """Assert if Triton path is selected when CUDA should be used."""
    def __getitem__(self, _grid):
        raise AssertionError("CUDA public wrapper unexpectedly selected Triton")


class _TritonSpy:
    """Track if Triton path is selected."""
    def __init__(self, wrapped):
        self.wrapped = wrapped
        self.called = False

    def __getitem__(self, grid):
        launch = self.wrapped[grid]

        def invoke(*args, **kwargs):
            self.called = True
            return launch(*args, **kwargs)

        return invoke


class _Platform:
    """Mock platform for dispatch testing."""
    def __init__(self, is_cuda):
        self._is_cuda = is_cuda

    def is_cuda(self):
        return self._is_cuda


def check_correctness() -> dict:
    """Validate CUDA operator correctness against PyTorch reference."""
    from vllm.model_executor.layers.quantization.utils import int8_utils

    torch.manual_seed(21476)
    cases = [
        ((32, 128), 64, torch.float16),
        ((64, 256), 128, torch.bfloat16),
        ((7, 512), 64, torch.float32),
        ((2, 3, 256), 32, torch.float16),
    ]

    results = []
    for shape, group_size, dtype in cases:
        x = (torch.randn(shape, device="cuda", dtype=dtype) * 8).contiguous()

        # Triton path (still from candidate for correctness baseline)
        tq = torch.empty_like(x, dtype=torch.int8)
        ts = torch.empty(
            x.shape[:-1] + (x.shape[-1] // group_size,),
            device=x.device,
            dtype=torch.float32,
        )
        programs = x.numel() // group_size
        block = int8_utils.triton.next_power_of_2(group_size)
        int8_utils._per_token_group_quant_int8[(programs,)](
            x, tq, ts, group_size, group_size, 1e-10,
            int8_min=-128, int8_max=127, BLOCK=block,
            num_warps=min(max(block // 256, 1), 8), num_stages=1,
        )

        # Reference
        rq, rs = reference(x, group_size)

        # CUDA candidate
        cq, cs = cuda_quant(x, group_size)

        triton_q_delta = (tq.to(torch.int16) - rq.to(torch.int16)).abs().max().item()
        cuda_q_delta = (cq.to(torch.int16) - tq.to(torch.int16)).abs().max().item()

        assert triton_q_delta <= 1, f"Triton delta {triton_q_delta} > 1"
        assert cuda_q_delta <= 1, f"CUDA delta {cuda_q_delta} > 1"
        assert torch.allclose(ts, rs, rtol=2e-4, atol=2e-5), "Triton scale mismatch"
        assert torch.allclose(cs, ts, rtol=2e-4, atol=2e-5), "CUDA scale mismatch"

        results.append({
            "shape": shape,
            "group_size": group_size,
            "dtype": str(dtype),
            "cuda_q_delta": int(cuda_q_delta),
            "scale_max_delta": float((cs - ts).abs().max().item()),
        })

    return {"cases": results}


def check_public_dispatch() -> dict:
    """Verify public wrapper dispatches to CUDA on CUDA platform."""
    from vllm.model_executor.layers.quantization.utils import int8_utils

    x = torch.randn((4, 128), device="cuda", dtype=torch.float16).contiguous()

    # On CUDA platform, public wrapper should use native CUDA (not Triton)
    with replace(int8_utils, "_per_token_group_quant_int8", _TritonBomb()):
        q, s = int8_utils.per_token_group_quant_int8(x, 64)

    rq, rs = reference(x, 64)
    assert (q.to(torch.int16) - rq.to(torch.int16)).abs().max().item() <= 1
    assert torch.allclose(s, rs, rtol=2e-4, atol=2e-5)

    # On non-CUDA platform, should fall back to Triton
    original_kernel = int8_utils._per_token_group_quant_int8
    spy = _TritonSpy(original_kernel)
    with replace(int8_utils, "current_platform", _Platform(False)):
        with replace(int8_utils, "_per_token_group_quant_int8", spy):
            fq, fs = int8_utils.per_token_group_quant_int8(x, 64)

    assert spy.called, "non-CUDA dispatch did not retain the Triton path"
    assert (fq.to(torch.int16) - rq.to(torch.int16)).abs().max().item() <= 1
    assert torch.allclose(fs, rs, rtol=2e-4, atol=2e-5)

    return {"cuda_dispatch": "native", "non_cuda_dispatch": "triton"}


def check_configurable_arguments() -> dict:
    """Verify native operator accepts configurable eps/int8_min/int8_max."""
    cases = [
        (torch.zeros((3, 64), device="cuda", dtype=torch.float32), 64, 1e-3, -64, 63),
        (
            torch.tensor(
                [[-4.0, -1.0, 0.0, 1.0, 4.0] * 16],
                device="cuda",
                dtype=torch.float16,
            ).contiguous(),
            80,
            1e-6,
            -32,
            31,
        ),
    ]

    results = []
    for x, group_size, eps, lower, upper in cases:
        q, s = cuda_quant(x, group_size, eps, float(lower), float(upper))
        rq, rs = reference(x, group_size, eps, lower, upper)

        assert int(q.min()) >= lower and int(q.max()) <= upper
        assert (q.to(torch.int16) - rq.to(torch.int16)).abs().max().item() <= 1
        assert torch.allclose(s, rs, rtol=2e-4, atol=2e-5)

        results.append({
            "shape": list(x.shape),
            "group_size": group_size,
            "eps": eps,
            "range": [lower, upper],
        })

    return {"cases": results}


def worker_main():
    """Worker subprocess: validate candidate and emit structured JSON."""
    result = {
        "verdict": "FAIL",
        "stages": {},
        "frozen_baseline_used": True,
        "parent_worker_boundary": True,
    }

    try:
        assert torch.cuda.is_available(), "GPU not available"

        result["gpu"] = torch.cuda.get_device_name(0)
        result["capability"] = torch.cuda.get_device_capability(0)

        # Stage 1: Verify native operator exists
        has_op = hasattr(torch.ops._C, "per_token_group_quant_int8")
        swapped_alias = hasattr(torch.ops._C, "per_token_group_int8_quant")

        if not has_op:
            result["reason"] = "native_operator_missing"
            result["swapped_alias"] = swapped_alias
            print(json.dumps(result, indent=2))
            sys.exit(1)

        if swapped_alias:
            result["reason"] = "swapped_alias_present"
            print(json.dumps(result, indent=2))
            sys.exit(1)

        result["stages"]["operator_surface"] = "PASS"

        # Stage 2: Correctness
        result["stages"]["correctness"] = check_correctness()

        # Stage 3: Public dispatch
        result["stages"]["public_dispatch"] = check_public_dispatch()

        # Stage 4: Configurable arguments
        result["stages"]["configurable_arguments"] = check_configurable_arguments()

        # Stage 5: Performance vs frozen baseline (run by parent)
        result["verdict"] = "PASS_CORRECTNESS"
        result["performance_deferred_to_parent"] = True

        print(json.dumps(result, indent=2))
        sys.exit(0)

    except Exception as exc:
        result["error"] = str(exc)
        result["traceback"] = traceback.format_exc()
        print(json.dumps(result, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    worker_main()
'''


def run_worker() -> dict:
    """Spawn the validation worker as uid 65534 (nobody), capture structured output."""
    try:
        result = subprocess.run(
            ["runuser", "-u", "nobody", "--", "python3", "-I", "-c", WORKER_CODE],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=600,
            check=False,
            text=True,
        )
    except subprocess.TimeoutExpired:
        return {
            "verdict": "FAIL",
            "reason": "worker_timeout",
            "timeout_sec": 600,
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
            if worker_output.get("verdict") == "PASS_CORRECTNESS":
                return worker_output
            else:
                return {
                    "verdict": "FAIL",
                    "reason": "worker_verdict_not_pass",
                    "worker_output": worker_output,
                }
        except json.JSONDecodeError as exc:
            return {
                "verdict": "FAIL",
                "reason": "worker_output_invalid_json",
                "error": str(exc),
                "stdout": result.stdout[:2000],
                "stderr": result.stderr[:2000],
            }
    else:
        try:
            worker_output = json.loads(result.stdout)
            return {
                "verdict": "FAIL",
                "reason": "worker_exit_nonzero",
                "worker_exit": result.returncode,
                "worker_output": worker_output,
                "stderr": result.stderr[:2000],
            }
        except json.JSONDecodeError:
            return {
                "verdict": "FAIL",
                "reason": "worker_failed_unparseable",
                "worker_exit": result.returncode,
                "stdout": result.stdout[:2000],
                "stderr": result.stderr[:2000],
            }


def run_frozen_baseline_subprocess(shape: tuple, group_size: int) -> float:
    """Run frozen Triton baseline in subprocess that excludes /workspace/repo."""
    baseline_code = '''
import sys
# Remove /workspace/repo from sys.path so the frozen reference resolves triton from
# root-owned site-packages and never from candidate-authored files.
sys.path = [p for p in sys.path if not p.startswith("/workspace/repo")]

import statistics
import torch

# Load the frozen reference through its root-owned loader. The loader verifies the
# reference SHA-256 and installs stub vllm.* modules, so the byte-identical
# base-commit copy of int8_utils.py imports nothing the candidate controls.
sys.path.insert(0, "/opt/ai-infra-bench/reference-int8")
from frozen_reference_loader import load_frozen_reference

frozen_triton_quant = load_frozen_reference().per_token_group_quant_int8

shape = ''' + repr(shape) + '''
group_size = ''' + str(group_size) + '''

x = (torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 8).contiguous()

# Warmup
for _ in range(40):
    frozen_triton_quant(x, group_size)

torch.cuda.synchronize()

# Time
samples = []
for _ in range(5):
    begin = torch.cuda.Event(True)
    end = torch.cuda.Event(True)
    begin.record()
    for _ in range(400):
        frozen_triton_quant(x, group_size)
    end.record()
    end.synchronize()
    samples.append(begin.elapsed_time(end) / 400)

print(statistics.median(samples))
'''

    try:
        result = subprocess.run(
            ["python3", "-c", baseline_code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120,
            check=True,
            text=True,
        )
        return float(result.stdout.strip())
    except Exception as exc:
        raise RuntimeError(f"Frozen baseline failed for {shape}: {exc}")


def check_performance_vs_frozen_baseline(worker_output: dict) -> dict:
    """Parent runs performance check: candidate CUDA vs frozen Triton baseline."""
    import torch

    # Import candidate CUDA operator
    def cuda_quant(x, group_size):
        q = torch.empty_like(x, dtype=torch.int8)
        s = torch.empty(
            x.shape[:-1] + (x.shape[-1] // group_size,),
            device=x.device,
            dtype=torch.float32,
        )
        torch.ops._C.per_token_group_quant_int8(
            x, q, s, group_size, 1e-10, -128.0, 127.0
        )
        return q, s

    def timed_ms_candidate(fn, warmup=40, repeats=400):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        samples = []
        for _ in range(5):
            begin, end = torch.cuda.Event(True), torch.cuda.Event(True)
            begin.record()
            for _ in range(repeats):
                fn()
            end.record()
            end.synchronize()
            samples.append(begin.elapsed_time(end) / repeats)
        import statistics
        return statistics.median(samples)

    cases = [
        ((32, 128), 64),
        ((64, 256), 128),
        ((16, 512), 64),
        ((256, 4096), 128),
    ]

    speedups = []
    timing_results = []

    for shape, group_size in cases:
        x = (torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 8).contiguous()

        # Time candidate CUDA
        cuda_ms = timed_ms_candidate(lambda: cuda_quant(x, group_size))

        # Time frozen baseline in isolated subprocess
        frozen_ms = run_frozen_baseline_subprocess(shape, group_size)

        speedup = frozen_ms / cuda_ms
        speedups.append(speedup)

        timing_results.append({
            "shape": shape,
            "group_size": group_size,
            "candidate_cuda_ms": round(cuda_ms, 6),
            "frozen_triton_ms": round(frozen_ms, 6),
            "speedup": round(speedup, 3),
        })

    min_speedup = min(speedups)
    performance_pass = min_speedup >= 1.5

    return {
        "timing": timing_results,
        "min_speedup": round(min_speedup, 3),
        "threshold": 1.5,
        "performance_pass": performance_pass,
    }


def main():
    """Trusted parent: run worker, check performance, write reward."""
    print("vllm-int8-per-token-group-quantization verifier (parent/worker + frozen baseline)")

    # Run worker for correctness checks
    worker_result = run_worker()

    if worker_result["verdict"] != "PASS_CORRECTNESS":
        verdict = {
            "verdict": "FAIL",
            "worker_result": worker_result,
        }
        print(json.dumps(verdict, indent=2))
        with open("/logs/verifier/reward.txt", "w") as f:
            f.write("0\n")
        sys.exit(1)

    # Parent runs performance check vs frozen baseline
    try:
        perf_result = check_performance_vs_frozen_baseline(worker_result)
        worker_result["stages"]["performance_vs_frozen_baseline"] = perf_result

        if not perf_result["performance_pass"]:
            verdict = {
                "verdict": "FAIL",
                "reason": "performance_below_threshold",
                "stages": worker_result["stages"],
            }
            print(json.dumps(verdict, indent=2))
            with open("/logs/verifier/reward.txt", "w") as f:
                f.write("0\n")
            sys.exit(1)

        # All checks pass
        verdict = {
            "verdict": "PASS",
            "stages": worker_result["stages"],
        }
        print(json.dumps(verdict, indent=2))
        with open("/logs/verifier/reward.txt", "w") as f:
            f.write("1\n")
        sys.exit(0)

    except Exception as exc:
        verdict = {
            "verdict": "FAIL",
            "reason": "performance_check_failed",
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "worker_result": worker_result,
        }
        print(json.dumps(verdict, indent=2))
        with open("/logs/verifier/reward.txt", "w") as f:
            f.write("0\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
