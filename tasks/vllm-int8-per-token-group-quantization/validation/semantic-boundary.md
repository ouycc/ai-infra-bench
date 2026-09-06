# Semantic boundary -- vllm-int8-per-token-group-quantization

Classifies each task component so dynamic validation knows where a substitution
could flip Base vs Oracle (SEMANTIC), where a contract-valid alternate input is
safe (SUBSTITUTABLE), and where a component is presentation/context only
(CONTEXT-ONLY). The independent challenge (`validation/challenge/challenge_int8_quant.py`)
places its fresh inputs at the SEMANTIC boundaries below.

| Component | Class | Rationale |
|-----------|-------|-----------|
| Dispatch to native torch.ops._C op | SEMANTIC | A Triton fallback would silently pass numerics but violate the contract. |
| group_size vs shape | SEMANTIC | Group boundary determines which elements share a scale. |
| Tensor values (negative-heavy row) | SUBSTITUTABLE | Any distribution exercising the full int8 range is valid. |
| Row count | CONTEXT-ONLY | Independent per-token rows; count is presentation only. |

## Substitution rule
- **SEMANTIC** components are held at the true contract boundary; the challenge
  perturbs them (fresh values astride the boundary) to confirm Base fails and
  Oracle passes.
- **SUBSTITUTABLE** components may be replaced by any contract-valid alternative
  without changing the reward; the challenge uses fresh values here to prove the
  contract, not a memorized case, is what is scored.
- **CONTEXT-ONLY** components do not affect reward and are free to vary.
