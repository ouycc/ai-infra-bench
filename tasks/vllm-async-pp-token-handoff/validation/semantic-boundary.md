# Semantic boundary -- vllm-async-pp-token-handoff

Classifies each task component so dynamic validation knows where a substitution
could flip Base vs Oracle (SEMANTIC), where a contract-valid alternate input is
safe (SUBSTITUTABLE), and where a component is presentation/context only
(CONTEXT-ONLY). The independent challenge (`validation/challenge/challenge_token_handoff.py`)
places its fresh inputs at the SEMANTIC boundaries below.

| Component | Class | Rationale |
|-----------|-------|-----------|
| GPU collective vs object/CPU path | SEMANTIC | An object collective or scalar sync is the exact failure mode. |
| Discard mask (interleaved) | SEMANTIC | Index mapping must skip discarded requests at any position. |
| Request ordering | SUBSTITUTABLE | Any permutation exercises the mapping. |
| Token id values | CONTEXT-ONLY | State/collective behavior is scored, not specific ids. |

## Substitution rule
- **SEMANTIC** components are held at the true contract boundary; the challenge
  perturbs them (fresh values astride the boundary) to confirm Base fails and
  Oracle passes.
- **SUBSTITUTABLE** components may be replaced by any contract-valid alternative
  without changing the reward; the challenge uses fresh values here to prove the
  contract, not a memorized case, is what is scored.
- **CONTEXT-ONLY** components do not affect reward and are free to vary.
