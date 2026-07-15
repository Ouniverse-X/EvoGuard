# EvoGuard Baselines

## Baseline List

| Baseline | Purpose | Implementation Status |
| --- | --- | --- |
| No guard | Lower bound for safety. Always executes tool calls. | Implemented |
| Rule-based guard | Simple keyword and tool-risk guard. | Implemented |
| Always-refuse guard | Upper-bound caution baseline for over-refusal analysis. | Implemented |
| Static guard | Defender trained repeatedly on a fixed attack pool. | Implemented for MVP |
| PAIR-style automated jailbreak | Black-box attacker LLM iteratively refines candidate jailbreak context. | Added as controlled attack family |
| TAP-style automated jailbreak | Tree-search attacker generates branches and prunes weak candidates before target queries. | Added as controlled attack family |
| GPTFuzzer-style automated jailbreak | Seed jailbreak prompts are mutated and judged to discover effective variants. | Added as controlled attack family |
| ToolSafe-style guard | Static tool-use safety defense. | Planned |
| MOSAIC-style training | Refusal/action safety training comparison. | Planned |
| EAPO-style fixed contrastive rollout | Contrastive rollout without adaptive attacks. | Implemented for MVP |
| EvoGuard adaptive | Round-based adaptive attack generation and defender updates. | Implemented for MVP |
| MAGIC-style baseline | Multi-agent or guardrail comparison, pending exact mapping. | Planned |
| RUBAS-style baseline | Robustness or benchmark comparison, pending exact mapping. | Planned |
| AgentWatcher-style attribution detector | Attribution-focused detection baseline. | Planned |

## Fairness Rules

- Use identical train/eval task splits where possible.
- Never mix train and evaluation attack pools.
- Report task success, attack interception, and over-refusal together.
- Keep the same simulated tools and risk labels across baselines.

## MVP Baseline Runner

Run:

```bash
python scripts/run_baselines.py
```

The runner evaluates:

- `no_guard`
- `rule_based_guard`
- `always_refuse_guard`
- `static_guard`
- `eapo_fixed_contrastive_rollout`
- `evoguard_adaptive`

It writes JSONL rows to `outputs/logs/baseline_comparison.jsonl`.

Trainable baselines use the train attack template pool. Reported comparison metrics use held-out attack templates.

## Automated Jailbreak Red-Team Coverage

The MVP now tracks PAIR, TAP, and GPTFuzzer as a separate `automated_jailbreak` attack split. These are not literal harmful-content jailbreak prompts; they are controlled tool-use analogues that map the papers' automation ideas to EvoGuard's threat model:

- PAIR-style: iterative refinement of a blocked hidden goal into a more plausible external-context clarification.
- TAP-style: branch-and-prune framing that nudges the defender toward a lower-safety branch.
- GPTFuzzer-style: seed mutation that preserves the unsafe semantic change while making the note look benign.

Use this split as a modern held-out red-team suite alongside the existing `heldout` and `hard_heldout` splits.
