We run vLLM with asynchronous scheduling and two pipeline-parallel ranks. The
last rank samples token IDs on the GPU, but handing them back to the earlier
rank must not introduce a CPU-object round trip or a CUDA-to-CPU
synchronization.

The last pipeline rank must broadcast the sampled token IDs directly as GPU
tensors, and earlier ranks must receive them into the input batch through a
real two-rank NCCL process group. On the receiving rank, preserve request
ordering, rebuild the request-ID-to-index map
without discarded requests, and append exactly one `-1` output placeholder to
each retained request and none to discarded requests. Keep the scheduler's
placeholder accounting consistent so the next async PP round does not schedule
an unnecessary extra step.

Work in `/workspace/repo`. The production sampling and receive lifecycle must
work under `torchrun --nproc-per-node=2` as request counts, ordering, and
discarded rows vary. Do not use CPU-object collectives or force token tensors
through CPU memory during the handoff. Collective phases must fail on a timeout
rather than hang, and a failure in one phase must not prevent the
available configuration, NCCL handoff, retained/discarded state, and following
scheduler-round evidence from being reported. Internal helper names are not a
public interface, so choose them to fit the implementation.

Reproduce the behavior yourself under `torchrun --nnodes=1 --nproc-per-node=2`
against the production `sample_tokens` lifecycle, exercising a mix of retained
and discarded requests to confirm the GPU-tensor broadcast, the rebuilt
request-ID-to-index map, and the placeholder accounting.
