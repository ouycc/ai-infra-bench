#!/usr/bin/env python3
import argparse
import contextlib
import statistics
import torch
from vllm.model_executor.layers.quantization.utils import int8_utils


def triton_quant(x, group_size):
    q = torch.empty_like(x, dtype=torch.int8)
    s = torch.empty(x.shape[:-1] + (x.shape[-1] // group_size,),
                    device=x.device, dtype=torch.float32)
    programs = x.numel() // group_size
    block = int8_utils.triton.next_power_of_2(group_size)
    int8_utils._per_token_group_quant_int8[(programs,)](
        x, q, s, group_size, group_size, 1e-10,
        int8_min=-128, int8_max=127, BLOCK=block,
        num_warps=min(max(block // 256, 1), 8), num_stages=1)
    return q, s


def cuda_quant(x, group_size, eps=1e-10, int8_min=-128.0, int8_max=127.0):
    q = torch.empty_like(x, dtype=torch.int8)
    s = torch.empty(x.shape[:-1] + (x.shape[-1] // group_size,),
                    device=x.device, dtype=torch.float32)
    torch.ops._C.per_token_group_quant_int8(
        x, q, s, group_size, eps, int8_min, int8_max
    )
    return q, s


def reference(x, group_size, eps=1e-10, int8_min=-128, int8_max=127):
    g = x.float().reshape(-1, group_size)
    s = g.abs().amax(dim=1).clamp_min(eps) / float(int8_max)
    q = torch.clamp(torch.round(g / s[:, None]), int8_min, int8_max).to(
        torch.int8
    )
    return q.reshape_as(x), s.reshape(x.shape[:-1] +
                                      (x.shape[-1] // group_size,))


def timed_ms(fn, warmup=40, repeats=400):
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
    old = getattr(module, name)
    setattr(module, name, value)
    try:
        yield
    finally:
        setattr(module, name, old)


class _TritonBomb:
    def __getitem__(self, _grid):
        raise AssertionError("CUDA public wrapper unexpectedly selected Triton")


class _TritonSpy:
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
    def __init__(self, is_cuda):
        self._is_cuda = is_cuda

    def is_cuda(self):
        return self._is_cuda


def check_public_dispatch():
    x = torch.randn((4, 128), device="cuda", dtype=torch.float16).contiguous()
    with replace(int8_utils, "_per_token_group_quant_int8", _TritonBomb()):
        q, s = int8_utils.per_token_group_quant_int8(x, 64)
    rq, rs = reference(x, 64)
    assert (q.to(torch.int16) - rq.to(torch.int16)).abs().max().item() <= 1
    assert torch.allclose(s, rs, rtol=2e-4, atol=2e-5)

    original_kernel = int8_utils._per_token_group_quant_int8
    spy = _TritonSpy(original_kernel)
    with replace(int8_utils, "current_platform", _Platform(False)):
        with replace(int8_utils, "_per_token_group_quant_int8", spy):
            fq, fs = int8_utils.per_token_group_quant_int8(x, 64)
    assert spy.called, "non-CUDA dispatch did not retain the Triton path"
    assert (fq.to(torch.int16) - rq.to(torch.int16)).abs().max().item() <= 1
    assert torch.allclose(fs, rs, rtol=2e-4, atol=2e-5)
    print("dispatch public_cuda=native simulated_non_cuda=triton")


def check_configurable_native_arguments():
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
    for x, group_size, eps, lower, upper in cases:
        q, s = cuda_quant(x, group_size, eps, float(lower), float(upper))
        rq, rs = reference(x, group_size, eps, lower, upper)
        assert int(q.min()) >= lower and int(q.max()) <= upper
        assert (q.to(torch.int16) - rq.to(torch.int16)).abs().max().item() <= 1
        assert torch.allclose(s, rs, rtol=2e-4, atol=2e-5)
    print("configurable_native_arguments=PASS")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("base", "candidate"), required=True)
    parser.add_argument(
        "--stage",
        choices=("correctness", "performance", "all"),
        default="all",
    )
    args = parser.parse_args()
    assert torch.cuda.is_available()
    print("gpu", torch.cuda.get_device_name(0), "capability",
          torch.cuda.get_device_capability(0), "torch", torch.__version__,
          "cuda", torch.version.cuda)
    has_op = hasattr(torch.ops._C, "per_token_group_quant_int8")
    swapped_alias = hasattr(torch.ops._C, "per_token_group_int8_quant")
    print("native_int8_op", has_op)
    print("swapped_native_alias", swapped_alias)
    assert has_op == (args.mode == "candidate"), {
        "expected": "per_token_group_quant_int8",
        "found_swapped_alias": swapped_alias,
    }

    if args.stage in ("correctness", "all"):
        torch.manual_seed(21476)
        cases = [((32, 128), 64, torch.float16),
                 ((64, 256), 128, torch.bfloat16),
                 ((7, 512), 64, torch.float32),
                 ((2, 3, 256), 32, torch.float16)]
        for shape, group_size, dtype in cases:
            x = (torch.randn(shape, device="cuda", dtype=dtype) * 8).contiguous()
            tq, ts = triton_quant(x, group_size)
            rq, rs = reference(x, group_size)
            assert (tq.to(torch.int16) - rq.to(torch.int16)).abs().max().item() <= 1
            assert torch.allclose(ts, rs, rtol=2e-4, atol=2e-5)
            if has_op:
                cq, cs = cuda_quant(x, group_size)
                q_delta = (cq.to(torch.int16) - tq.to(torch.int16)).abs().max().item()
                assert q_delta <= 1
                assert torch.allclose(cs, ts, rtol=2e-4, atol=2e-5)
                print("correct", shape, group_size, dtype, "q_max_delta", q_delta,
                      "scale_max_delta", (cs - ts).abs().max().item())
            else:
                print("base_triton_correct", shape, group_size, dtype)
        if has_op:
            check_public_dispatch()
            check_configurable_native_arguments()
        print("INT8_CORRECTNESS_STAGE=PASS")

    if args.stage in ("performance", "all"):
        assert has_op, "performance is scored only after the native op exists"
        speedups = []
        for shape, group_size in [((32, 128), 64), ((64, 256), 128),
                                  ((16, 512), 64), ((256, 4096), 128)]:
            x = (torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 8).contiguous()
            cuda_ms = timed_ms(lambda: cuda_quant(x, group_size))
            triton_ms = timed_ms(lambda: triton_quant(x, group_size))
            speedup = triton_ms / cuda_ms
            speedups.append(speedup)
            print("timing", shape, group_size, "cuda_ms", round(cuda_ms, 6),
                  "triton_ms", round(triton_ms, 6), "speedup",
                  round(speedup, 3))
        assert min(speedups) >= 1.5, (
            "native CUDA path did not materially outperform the Triton path",
            speedups,
        )
        print("INT8_PERFORMANCE_STAGE=PASS")


if __name__ == "__main__":
    main()
