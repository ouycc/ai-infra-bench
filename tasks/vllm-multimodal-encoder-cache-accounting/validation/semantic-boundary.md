# Semantic boundary -- vllm-multimodal-encoder-cache-accounting

Classifies each task component so dynamic validation knows where a substitution
could flip Base vs Oracle (SEMANTIC), where a contract-valid alternate input is
safe (SUBSTITUTABLE), and where a component is presentation/context only
(CONTEXT-ONLY). The independent challenge (`validation/challenge/challenge_encoder_cache.py`)
places its fresh inputs at the SEMANTIC boundaries below.

| Component | Class | Rationale |
|-----------|-------|-----------|
| PlaceholderRange mask selection | SEMANTIC | Counting the raw span instead of selected rows flips budget admission. |
| Encoder-cache budget size | SEMANTIC | The boundary is the last row that fits; +/-1 changes Base vs Oracle. |
| Prompt subrange offsets | SUBSTITUTABLE | Any contract-valid straddling range exercises the same mapping. |
| Request/item ordering | CONTEXT-ONLY | Ordering does not change per-row accounting. |

## Substitution rule
- **SEMANTIC** components are held at the true contract boundary; the challenge
  perturbs them (fresh values astride the boundary) to confirm Base fails and
  Oracle passes.
- **SUBSTITUTABLE** components may be replaced by any contract-valid alternative
  without changing the reward; the challenge uses fresh values here to prove the
  contract, not a memorized case, is what is scored.
- **CONTEXT-ONLY** components do not affect reward and are free to vary.
