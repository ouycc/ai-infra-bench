We use multimodal placeholders whose prompt spans are much longer than their
encoder outputs. A placeholder may cover 100 input-token positions but have an
embedding mask with only eight true positions. At the moment, vLLM can account
for all 100 placeholder tokens and reject it even when the encoder cache has
room for exactly eight embedding rows.

Encoder-cache capacity, allocation, eviction, scheduling budgets, and partial
embedding selection must consistently use embedding-row units. A length-100
placeholder with eight selected positions must fit in a cache of capacity
eight, while placeholders without a mask must keep their existing behavior.

Prompt space and embedding space also need to stay as distinct coordinate
systems. If a placeholder covers prompt range `[10, 30)` and only selected
positions produce embeddings, scheduling prompt subrange `[15, 22)` must select
the corresponding contiguous embedding subrange rather than treating prompt
offsets as embedding offsets. Empty ranges, all-false masks, multiple
multimodal items, and mask-free placeholders must remain well-defined.

Work in `/app` and keep the production request, scheduler, profiling,
cache-manager, multimodal helper, and model-runner gather/slice paths
consistent. Update the affected callers rather than adding a one-off case for
the example. The internal representation, including whether an embedding count
is a property or a method, is up to you. Validate against the production
`PlaceholderRange`, `EncoderCacheManager`, and scheduler paths directly,
covering masked, mask-free, all-false, empty, and multi-item placeholders.
