# EvoGuard Paper Outline

## Title

EvoGuard: Self-Evolving Adversarial Risk-Aware Reinforcement Learning for Tool-Calling Agents

## Abstract

To be drafted after initial experiments.

## Introduction

Motivation:
- Tool-calling agents can execute consequential actions.
- Static prompt-injection defenses do not model evolving attacks.
- Safety improvements must not collapse into refusing every tool call.

Contributions:
- Dynamic prompt-driven attack generation loop.
- Tri-trajectory tool safety rollout.
- Risk-aware reward with confidence, round, stale-error, and attribution terms.
- Controlled text-tool benchmark and planned embodied extension.

## Method

See `docs/method_design.md`.

## Experiments

See `docs/experiments.md`.
