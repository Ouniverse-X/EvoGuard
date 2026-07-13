# EvoGuard Project Context

## Project Name

EvoGuard: Self-Evolving Adversarial Risk-Aware Reinforcement Learning for Tool-Calling Agents.

## Project Type

AI Agent Security Research Project.

## Core Research Problem

Existing tool-use Agent safety methods often rely on static attack datasets or fixed contrastive rollouts. They do not model the fact that real-world tool-use attacks evolve against the current defender.

EvoGuard studies how to train a tool-calling defense Agent that improves its safety judgment under a dynamically evolving adversarial attack distribution, while preserving normal task completion ability.

## Core Idea

EvoGuard combines:

1. EAPO-style three-trajectory contrastive rollout.
2. Tool-use safety interception before execution.
3. Prompt-driven adaptive red-team attack generation.
4. Risk-aware reinforcement learning with confidence, attack iteration, and attribution rewards.

## Three Rollout Types

For each task, construct:

1. No-tool trajectory:
   - The Agent solves the task without tools.
   - Used to estimate whether the task truly needs external tools.

2. Clean tool trajectory:
   - The Agent uses tools in a benign environment.
   - Used to preserve task completion ability.

3. Attacked tool trajectory:
   - The context is modified by a prompt-driven attack generator.
   - Used to train safety judgment against adversarial tool-use risks.

## Defense Agent Output

Before each tool call, the defense Agent outputs:

1. Safety action:
   - EXECUTE
   - ASK_CONFIRMATION
   - REFUSE
2. Confidence score.
3. Attribution distribution over input spans.

## Training Objective

The reward has three parts:

1. Task completion reward.
2. Safety judgment reward.
3. Attribution reward.

Use PPO or GRPO-style policy optimization with KL regularization.

## Key Metrics

1. Attack interception rate.
2. Task success rate.
3. Over-refusal rate.
4. False negative rate on attacked tool calls.
5. False positive rate on benign tool calls.
6. Attribution accuracy.
7. Attack success rate across self-evolution rounds.
8. Robustness to unseen attack styles.
9. Cross-environment generalization.

## Main Baselines

1. ToolSafe
2. MOSAIC
3. EAPO
4. MAGIC
5. RUBAS
6. AgentWatcher
7. Static guard model
8. No guard
9. Rule-based guard

## Engineering Principles

1. Keep the attack generator fixed. It is prompt-driven and not trained.
2. Only the defense Agent is updated.
3. Each attack sample must record its generation round.
4. Each injected attack must record its insertion span for attribution reward.
5. Never mix training and evaluation attack pools.
6. Always log task success, attack interception, and over-refusal together.
7. Avoid optimizing safety at the cost of refusing all tool calls.
8. Prefer modular implementations.
9. Every major experimental decision should be recorded in docs/decisions.md.
10. Every experiment should be logged in docs/experiments.md.

## Current Implementation Plan

Phase 1: Build text-tool prototype.
Phase 2: Implement attack generator and tri-rollout.
Phase 3: Implement reward functions.
Phase 4: Implement PPO/GRPO training loop.
Phase 5: Add static baselines.
Phase 6: Add AI2-THOR and ManiSkill environments.
Phase 7: Run ablations and write paper.
