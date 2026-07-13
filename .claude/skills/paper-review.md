---
name: paper-review
description: Review a research paper for EvoGuard and extract method, assumptions, weaknesses, and relevance.
---

# Purpose

Analyze a research paper from the perspective of EvoGuard.

# Workflow

## Step 1: Metadata

Extract:
- Title
- Authors
- Venue
- Year
- Link

## Step 2: Problem

Identify:
- What problem does the paper solve?
- What setting does it assume?
- What type of Agent or model does it target?

## Step 3: Method

Summarize:
- Core method.
- Training objective.
- Data source.
- Rollout mechanism.
- Safety mechanism if any.

## Step 4: Relevance to EvoGuard

Explain:
- What can EvoGuard borrow?
- What limitation does EvoGuard address?
- How should this paper be cited?

## Step 5: Weaknesses

Analyze:
- Static vs dynamic attacks.
- Tool-call granularity.
- Whether safety is modeled before execution.
- Whether over-refusal is measured.
- Whether attribution participates in training.

## Step 6: Output

Append a markdown note to `docs/paper_notes.md` with:

- Paper
- Summary
- Method
- Strengths
- Weaknesses
- Relevance to EvoGuard
- Citation Angle
