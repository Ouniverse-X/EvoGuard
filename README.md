# EvoGuard

EvoGuard is a research prototype for self-evolving adversarial risk-aware reinforcement learning in tool-calling agents.

The implementation focuses on controlled text tool-calling safety and step-level tool-use datasets such as ToolSafe/TS-Bench.

## Current Scope

- Text tool-calling environment with simulated tools only.
- Three trajectory types: no-tool, clean-tool, and attacked-tool.
- Prompt-template attack generator with round metadata and injection spans.
- API-backed online adaptive red-team runner with success/blocked feedback memory.
- Rule-based and lightweight PPO trainable defense heads for MVP data flow.
- Reward decomposition for task, safety, attribution, and KL terms.
- Round-based training loop with a dependency-free clipped PPO update path.

## Quick Start

```bash
python scripts/run_rollout.py
python scripts/generate_attacks.py
python scripts/train_evoguard.py
python scripts/evaluate.py
python scripts/run_baselines.py
python scripts/run_ablations.py
python scripts/plot_results.py
```

API-backed online red teaming uses an OpenAI-compatible attack model configured through
`EVOGUARD_ATTACK_API_KEY`, `EVOGUARD_ATTACK_BASE_URL`, and `EVOGUARD_ATTACK_MODEL`:

```bash
python scripts/run_online_redteam.py --defender rule_based_guard --rounds 3
```

`scripts/train_evoguard.py` uses the trainable safety head by default. It is a small linear policy updated with a lightweight clipped PPO objective over EvoGuard trajectory rewards.

`scripts/plot_results.py` reads JSONL logs from `outputs/logs/` and generates paper-facing tables under `outputs/reports/` plus SVG figures under `outputs/figures/`.

## Project Memory

- `CLAUDE.md` defines the research context and engineering rules.
- `docs/method_design.md` defines the method and threat model.
- `docs/roadmap.md` tracks implementation phases.
- `docs/experiments.md` records experiments.
- `docs/decisions.md` records major design decisions.
