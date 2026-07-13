# EvoGuard Design Decisions

## D001: Use prompt-driven attack generator instead of trained attack policy

Date: 2026-07-12

Decision:
The attacker is a fixed LLM or template-driven generator controlled by prompts, not a trainable policy.

Reason:
This avoids expensive self-play and keeps the framework simple while preserving adaptive attack generation through failure-case feedback.

Alternatives:
- Train a red-team policy.
- Use static attack dataset.
- Use rule-based attack mutation.

Trade-offs:
Prompt-driven attacks may depend on the capability of the chosen LLM, but this is acceptable because the main contribution is the adaptive training loop for the defender.

## D002: Use attack generation round as proxy for attack strength

Date: 2026-07-12

Decision:
Attack generation round is recorded and used as a curriculum feature in reward computation.

Reason:
Later attack rounds are conditioned on failures of the current defender and are expected to be harder on average.

Risk:
Round index is an imperfect proxy. Some early attacks may be hard and some late attacks may be trivial.

Mitigation:
Use round weighting together with confidence penalties, stale-error penalties, and direct held-out evaluation metrics.

## D003: Start with a controlled text-tool MVP before external tool-safety datasets

Date: 2026-07-12

Decision:
The first implementation targets a simulated text tool-calling environment.

Reason:
Text tools make it easier to verify tri-rollout, injection spans, safety labels, and reward computation before adding external dataset adapters.

Risk:
Text-only results may not transfer to step-level tool-call datasets such as ToolSafe or TraceSafe.

Mitigation:
Keep environment and rollout interfaces modular so dataset adapters can be added later.

## D004: Use a dependency-free lightweight PPO safety head for MVP validation

Date: 2026-07-12

Decision:
Add a linear safety head trained with a lightweight clipped PPO objective before introducing a full neural PPO/GRPO backend.

Reason:
The project needs a verifiable defense-training path early. A small PPO-trainable head proves that the round-based loop can change policy behavior using the same reward structure without introducing TRL, Transformers, or model-serving dependencies.

Risk:
The linear head is not expressive enough to support final paper claims, and clipped PPO over a tiny discrete policy is only an MVP approximation.

Mitigation:
Use it only as an MVP backend. Keep the `DefenseAgent` and trainer interfaces replaceable so neural PPO/GRPO policies or TRL can be added next.
