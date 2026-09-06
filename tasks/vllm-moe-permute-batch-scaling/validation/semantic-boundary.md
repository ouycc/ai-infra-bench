# Semantic boundary -- vllm-moe-permute-batch-scaling

Classifies each task component so dynamic validation knows where a substitution
could flip Base vs Oracle (SEMANTIC), where a contract-valid alternate input is
safe (SUBSTITUTABLE), and where a component is presentation/context only
(CONTEXT-ONLY). The independent challenge (`validation/challenge/challenge_moe_permute.py`)
places its fresh inputs at the SEMANTIC boundaries below.

| Component | Class | Rationale |
|-----------|-------|-----------|
| Token count (batch) | SEMANTIC | Batch-dependent shortcuts break at non-power-of-two / large counts; both verifier and challenge time a fresh non-power-of-two large batch (3000) not in the profiled power-of-two set. |
| Alignment mode (align_block_size) | SEMANTIC | Aligned (128) and unaligned (None) are distinct operator contracts; both are scored for offsets, mapping, payload, and (aligned) m_indices/sentinel. |
| Routing/expert assignment | SEMANTIC | The permutation must follow the exact routing map. |
| Hidden size | SUBSTITUTABLE | Any alignment-valid hidden size exercises the op. |
| Expert count | SUBSTITUTABLE | Any topk<=experts routing is valid. |

## Substitution rule
- **SEMANTIC** components are held at the true contract boundary; the challenge
  perturbs them (fresh values astride the boundary) to confirm Base fails and
  Oracle passes.
- **SUBSTITUTABLE** components may be replaced by any contract-valid alternative
  without changing the reward; the challenge uses fresh values here to prove the
  contract, not a memorized case, is what is scored.
- **CONTEXT-ONLY** components do not affect reward and are free to vary.
