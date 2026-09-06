ARG VLLM_SOURCE_COMMIT=14bf19e39f601163265b7c7d58d972b8a83d8896
ARG VLLM_SOURCE_TREE=182dac984150ee131cffda0d4245a9f50b0b4792
ARG VLLM_ORACLE_COMMIT=2e25870f8445f9531a7fa9fb8f144bb68e4ab6e6
ARG VLLM_REPOSITORY=https://github.com/vllm-project/vllm.git

FROM alpine/git:2.49.1@sha256:c0280cf9572316299b08544065d3bf35db65043d5e3963982ec50647d2746e26 AS git_runtime
ARG VLLM_SOURCE_COMMIT
ARG VLLM_SOURCE_TREE
ARG VLLM_ORACLE_COMMIT
ARG VLLM_REPOSITORY
RUN tar -czf /git-runtime.tar \
    /usr/bin/git \
    /usr/libexec/git-core \
    /usr/share/git-core \
    /lib/ld-musl-x86_64.so.1 \
    /usr/lib/libpcre2-8.so.0.14.0 \
    /usr/lib/libz.so.1.3.1
RUN set -eux; \
    git init -q /vllm-history; \
    git -C /vllm-history config core.logAllRefUpdates false; \
    git -C /vllm-history config user.name "AI Infra Bench"; \
    git -C /vllm-history config user.email "noreply@ai-infra-bench.local"; \
    git -C /vllm-history remote add origin "${VLLM_REPOSITORY}"; \
    git -C /vllm-history fetch --no-tags origin "${VLLM_SOURCE_COMMIT}"; \
    git -C /vllm-history checkout -q -b benchmark-base "${VLLM_SOURCE_COMMIT}"; \
    test "$(git -C /vllm-history rev-parse HEAD)" = "${VLLM_SOURCE_COMMIT}"; \
    test "$(git -C /vllm-history rev-parse 'HEAD^{tree}')" = "${VLLM_SOURCE_TREE}"; \
    git -C /vllm-history remote remove origin; \
    rm -rf /vllm-history/.git/logs; \
    rm -f /vllm-history/.git/FETCH_HEAD /vllm-history/.git/ORIG_HEAD; \
    test ! -e /vllm-history/.git/shallow; \
    test -z "$(git -C /vllm-history remote)"; \
    test -z "$(git -C /vllm-history for-each-ref --format='%(refname)' refs/remotes refs/tags)"; \
    test "$(git -C /vllm-history rev-list --count HEAD)" -gt 1; \
    test -z "$(git -C /vllm-history rev-list --all --not "${VLLM_SOURCE_COMMIT}")"; \
    test -z "$(git -C /vllm-history fsck --full --no-reflogs --unreachable --no-progress 2>/dev/null)"; \
    ! git -C /vllm-history cat-file -e "${VLLM_ORACLE_COMMIT}^{commit}" 2>/dev/null

# v0.10.0 is the first official release after the PR base (2025-07-23).
# Its toolchain matches the base requirement (torch 2.7.1/CUDA).  The complete
# candidate extension _C and all Python source below come from the selected,
# digest-verified VLLM_SOURCE_URL archive;
# unrelated native extensions are explicitly labeled same-version donors.
FROM vllm/vllm-openai:v0.10.0@sha256:05a31dc4185b042e91f4d2183689ac8a87bd845713d5c3f987563c5899878271

ARG VLLM_SOURCE_COMMIT
ARG VLLM_SOURCE_TREE
ARG VLLM_ORACLE_COMMIT
ARG VLLM_SOURCE_SHA256=ef6429c4011bb3955ac9249833aa0a82f36b7855dcad7ac556deb4849d5f3339
ARG VLLM_SOURCE_URL=https://codeload.github.com/vllm-project/vllm/tar.gz/14bf19e39f601163265b7c7d58d972b8a83d8896
ARG CUTLASS_SHA256=44a121c5878827875856c175ebe82e955062e37cd61fcdf31ebe2e8874f2fc5c
ARG CUTLASS_URL=https://codeload.github.com/NVIDIA/cutlass/tar.gz/refs/tags/v4.0.0
ARG FLASH_ATTN_SHA256=cca19d7e53af08aa6d6f0c4fd9dd78d30314497e38fb03b1368b3d5a77ab4b5c
ARG FLASH_ATTN_URL=https://codeload.github.com/vllm-project/flash-attention/tar.gz/1c2624e53c078854e0637ee566c72fe2107e75f4
ARG FLASH_ATTN_CUTLASS_SHA256=78816d6c6d97793b5b59ef2a702174cb85b78dfcefc8fe2489964de2e42f17d2
ARG FLASH_ATTN_CUTLASS_URL=https://codeload.github.com/NVIDIA/cutlass/tar.gz/62750a2b75c802660e4894434dc55e839f322277
ARG FLASH_MLA_SHA256=42ebf11c1a4cd17c4221705de5cff4fcaedf29a9dbdfcb8f2459dc550faf66f7
ARG FLASH_MLA_URL=https://codeload.github.com/vllm-project/FlashMLA/tar.gz/575f7724b9762f265bbee5889df9c7d630801845

LABEL ai-infra-bench.vllm-source-commit="${VLLM_SOURCE_COMMIT}" \
      ai-infra-bench.vllm-native-origin="_C=exact-source:${VLLM_SOURCE_COMMIT};other=v0.10.0-donor" \
      ai-infra-bench.vllm-native-scope="candidate-path:_C" \
      ai-infra-bench.cuda-arches="8.0"

USER root

COPY --from=git_runtime /git-runtime.tar /tmp/git-runtime.tar
RUN tar -xzf /tmp/git-runtime.tar -C / \
    && rm /tmp/git-runtime.tar \
    && ln -s libpcre2-8.so.0.14.0 /usr/lib/libpcre2-8.so.0 \
    && ln -s libz.so.1.3.1 /usr/lib/libz.so.1 \
    && git --version

RUN curl --fail --location --retry 8 --retry-all-errors "${VLLM_SOURCE_URL}" -o /tmp/vllm-source.tar.gz \
    && curl --fail --location --retry 8 --retry-all-errors "${CUTLASS_URL}" -o /tmp/cutlass.tar.gz \
    && curl --fail --location --retry 8 --retry-all-errors "${FLASH_ATTN_URL}" -o /tmp/flash-attn.tar.gz \
    && curl --fail --location --retry 8 --retry-all-errors "${FLASH_ATTN_CUTLASS_URL}" -o /tmp/flash-attn-cutlass.tar.gz \
    && curl --fail --location --retry 8 --retry-all-errors "${FLASH_MLA_URL}" -o /tmp/flashmla.tar.gz

# Build with no package-index or FetchContent access.  CMake dependencies and
# the FlashAttention submodule are immutable, digest-checked archives.  sm80 is
# the A100 gate and avoids compiling unrelated GPU architectures.
ENV PIP_NO_INDEX=1 \
    VLLM_TARGET_DEVICE=cuda \
    TORCH_CUDA_ARCH_LIST=8.0 \
    MAX_JOBS=8 \
    NVCC_THREADS=2 \
    VLLM_CUTLASS_SRC_DIR=/opt/vllm-deps/cutlass \
    VLLM_FLASH_ATTN_SRC_DIR=/opt/vllm-deps/flash-attn \
    FLASH_MLA_SRC_DIR=/opt/vllm-deps/flashmla \
    SETUPTOOLS_SCM_PRETEND_VERSION=0.10.0.dev0

RUN set -eu; \
    printf '%s  %s\n' "${VLLM_SOURCE_SHA256}" /tmp/vllm-source.tar.gz | sha256sum -c -; \
    printf '%s  %s\n' "${CUTLASS_SHA256}" /tmp/cutlass.tar.gz | sha256sum -c -; \
    printf '%s  %s\n' "${FLASH_ATTN_SHA256}" /tmp/flash-attn.tar.gz | sha256sum -c -; \
    printf '%s  %s\n' "${FLASH_ATTN_CUTLASS_SHA256}" /tmp/flash-attn-cutlass.tar.gz | sha256sum -c -; \
    printf '%s  %s\n' "${FLASH_MLA_SHA256}" /tmp/flashmla.tar.gz | sha256sum -c -; \
    mkdir -p /workspace/repo /opt/vllm-canonical-source /opt/vllm-deps/cutlass /opt/vllm-deps/flash-attn /opt/vllm-deps/flashmla; \
    tar -xzf /tmp/vllm-source.tar.gz --strip-components=1 -C /workspace/repo; \
    tar -xzf /tmp/vllm-source.tar.gz --strip-components=1 -C /opt/vllm-canonical-source; \
    tar -xzf /tmp/cutlass.tar.gz --strip-components=1 -C /opt/vllm-deps/cutlass; \
    tar -xzf /tmp/flash-attn.tar.gz --strip-components=1 -C /opt/vllm-deps/flash-attn; \
    tar -xzf /tmp/flash-attn-cutlass.tar.gz --strip-components=1 -C /opt/vllm-deps/flash-attn/csrc/cutlass; \
    tar -xzf /tmp/flashmla.tar.gz --strip-components=1 -C /opt/vllm-deps/flashmla; \
    rm /tmp/vllm-source.tar.gz /tmp/cutlass.tar.gz /tmp/flash-attn.tar.gz /tmp/flash-attn-cutlass.tar.gz /tmp/flashmla.tar.gz; \
    test ! -e /workspace/repo/.git; \
    test "$(find /opt/vllm-canonical-source -type f | wc -l)" -gt 1000; \
    diff -qr /opt/vllm-canonical-source /workspace/repo; \
    donor_pkg="$(python3 -c 'import importlib.metadata; print(importlib.metadata.distribution("vllm").locate_file("vllm"))')"; \
    mkdir -p /opt/vllm-donor-native; \
    cd "$donor_pkg"; \
    find . -type f -name '*.so' -exec cp --parents '{}' /opt/vllm-donor-native/ \;; \
    cd /workspace/repo; \
    cp setup.py /tmp/vllm-setup.py.canonical; \
    sed -i '/^package_data = {/i ext_modules = [CMakeExtension(name="vllm._C")]\n' setup.py; \
    python3 -m pip install --no-build-isolation --no-deps -e /workspace/repo; \
    mv /tmp/vllm-setup.py.canonical setup.py; \
    python3 -c "import importlib.util,pathlib,vllm; p=pathlib.Path(vllm.__file__).resolve(); s=importlib.util.find_spec('vllm._C'); print(p,s.origin if s else None); assert str(p).startswith('/workspace/repo/vllm/') and s is not None and str(s.origin).startswith('/workspace/repo/vllm/')"; \
    find /opt/vllm-canonical-source -type f -print0 | while IFS= read -r -d '' f; do rel="${f#/opt/vllm-canonical-source/}"; cmp "$f" "/workspace/repo/$rel"; done; \
    rm -rf /opt/vllm-canonical-source; \
    cd /opt/vllm-donor-native; \
    find . -type f -name '*.so' ! -name '_C*.so' -exec cp --no-clobber --parents '{}' /workspace/repo/vllm/ \;; \
    rm -rf /opt/vllm-donor-native

# Install frozen Triton reference kernel for performance baseline.
# Extracted from base commit before agent user is created and before /workspace/repo
# is chowned to agent. The reference is root-owned and unwritable by agent (uid 1000).
# SHA-256 is recorded and verified at image check time.
RUN set -eu; \
    mkdir -p /opt/ai-infra-bench/reference-int8; \
    cat > /opt/ai-infra-bench/reference-int8/reference_int8_kernel.py <<'FROZEN_REFERENCE_EOF'
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# FROZEN TRITON REFERENCE for the INT8 per-token-group quantization task.
#
# This file is a byte-for-byte extraction of the ``_per_token_group_quant_int8``
# Triton kernel and its ``per_token_group_quant_int8`` launcher from
#   vllm/model_executor/layers/quantization/utils/int8_utils.py
# at the task base commit 14bf19e39f601163265b7c7d58d972b8a83d8896.
#
# It is installed as root in a root-owned directory before /workspace/repo is
# chowned to agent, so the agent (uid 1000) cannot modify it. It imports
# triton/triton.language DIRECTLY from root-owned site-packages, NOT through
# vllm.triton_utils (which lives in the agent-writable /workspace/repo tree).
#
# The kernel body below is verbatim from the base commit; only the import of
# triton/tl is made direct. Do not edit: the verifier records this file's
# SHA-256 and refuses to score if it changes.
import torch
import triton
import triton.language as tl


@triton.jit
def _per_token_group_quant_int8(
    # Pointers to inputs and output
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    # Stride of input
    y_stride,
    # Columns of input
    N,
    # Avoid to divide zero
    eps,
    # Information for int8
    int8_min,
    int8_max,
    # Meta-parameters
    BLOCK: tl.constexpr,
):
    """A Triton-accelerated function to perform per-token-group
    quantization on a tensor.

    This function converts the tensor values into int8 values.
    """
    # Map the program id to the row of X and Y it should compute.
    g_id = tl.program_id(0)
    y_ptr += g_id * y_stride
    y_q_ptr += g_id * y_stride
    y_s_ptr += g_id

    cols = tl.arange(0, BLOCK)  # N <= BLOCK
    mask = cols < N

    y = tl.load(y_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    # Quant
    _absmax = tl.maximum(tl.max(tl.abs(y)), eps)
    y_s = _absmax / int8_max
    y_q = tl.clamp(y / y_s, int8_min, int8_max).to(y_q_ptr.dtype.element_ty)

    tl.store(y_q_ptr + cols, y_q, mask=mask)
    tl.store(y_s_ptr, y_s)


def per_token_group_quant_int8(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    dtype: torch.dtype = torch.int8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Function to perform per-token-group quantization on an input tensor `x`.

    It converts the tensor values into signed int8 values and returns the
    quantized tensor along with the scaling factor used for quantization.

    Args:
        x: The input tensor with ndim >= 2.
        group_size: The group size used for quantization.
        eps: The minimum to avoid dividing zero.
        dtype: The dype of output tensor. Note that only `torch.int8`
            is supported for now.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The quantized tensor and the
            scaling factor for quantization.
    """
    assert (x.shape[-1] % group_size == 0
            ), "the last dimension of `x` cannot be divisible by `group_size`"
    assert x.is_contiguous(), "`x` is not contiguous"

    iinfo = torch.iinfo(dtype)
    int8_max = iinfo.max
    int8_min = iinfo.min

    x_q = torch.empty_like(x, device=x.device, dtype=dtype)
    M = x.numel() // group_size
    N = group_size
    x_s = torch.empty(
        x.shape[:-1] + (x.shape[-1] // group_size, ),
        device=x.device,
        dtype=torch.float32,
    )

    BLOCK = triton.next_power_of_2(N)
    # heuristics for number of warps
    num_warps = min(max(BLOCK // 256, 1), 8)
    num_stages = 1
    _per_token_group_quant_int8[(M, )](
        x,
        x_q,
        x_s,
        group_size,
        N,
        eps,
        int8_min=int8_min,
        int8_max=int8_max,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return x_q, x_s
FROZEN_REFERENCE_EOF
RUN chmod 444 /opt/ai-infra-bench/reference-int8/reference_int8_kernel.py; \
    sha256sum /opt/ai-infra-bench/reference-int8/reference_int8_kernel.py \
      > /opt/ai-infra-bench/reference-int8/reference_int8_kernel.py.sha256

COPY --from=git_runtime /vllm-history/.git /workspace/repo/.git

RUN set -eu; \
    test "$(git -C /workspace/repo rev-parse HEAD)" = "${VLLM_SOURCE_COMMIT}"; \
    test "$(git -C /workspace/repo rev-parse 'HEAD^{tree}')" = "${VLLM_SOURCE_TREE}"; \
    test -z "$(git -C /workspace/repo remote)"; \
    test -z "$(git -C /workspace/repo for-each-ref --format='%(refname)' refs/remotes refs/tags)"; \
    test ! -e /workspace/repo/.git/logs; \
    test ! -e /workspace/repo/.git/FETCH_HEAD; \
    test ! -e /workspace/repo/.git/ORIG_HEAD; \
    test ! -e /workspace/repo/.git/shallow; \
    test "$(git -C /workspace/repo rev-list --count HEAD)" -gt 1; \
    test -z "$(git -C /workspace/repo rev-list --all --not "${VLLM_SOURCE_COMMIT}")"; \
    test -z "$(git -C /workspace/repo fsck --full --no-reflogs --unreachable --no-progress 2>/dev/null)"; \
    ! git -C /workspace/repo cat-file -e "${VLLM_ORACLE_COMMIT}^{commit}" 2>/dev/null; \
    test -z "$(git -C /workspace/repo status --porcelain=v1 --untracked-files=all)"; \
    useradd --create-home --uid 1000 --shell /bin/bash agent; \
    chown -R agent:agent /workspace/repo /home/agent

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    VLLM_NO_USAGE_STATS=1 \
    HOME=/home/agent

WORKDIR /workspace/repo
USER agent

RUN test "$(id -un)" = agent \
    && test "$(git rev-parse HEAD)" = "${VLLM_SOURCE_COMMIT}" \
    && test -z "$(git remote)" \
    && test -z "$(git status --porcelain)"

ENTRYPOINT []
CMD ["sleep", "infinity"]
