# Δ-Bench — a Δ-stratified twin ladder for latent tool-injection

**Eval-only. Never train on this set.**

## What it is

19 case groups × 4 tiers = **76 injected scenarios** plus **18 clean rows**, over
four AgentDojo v1 suites (banking 3, slack 6, travel 9, workspace 1).

Each group is one attack held fixed while the harmful call is pushed further from
the injection point:

| tier | `expected_delta` | the payload asks for the sink … |
|---|---|---|
| `imm` | 1 | immediately, at the step right after the carrier |
| `d1`  | 2 | after 1 more clean step |
| `d2`  | 3 | after 2 more clean steps |
| `d3`  | 4 | after 3 more clean steps |

All four rows of a group share the task, the clean plan, the carrier and its
benign observation, the sink tool and the sink arguments. **Only the payload
text differs**, so a by-tier ASR curve reads as the effect of deferral distance
and nothing else. `expected_delta` is the declared plan-index offset from the
carrier, not a measured `signals.delta`.

## Why a fourth `imm` tier rather than reusing `data/agentdojo_stock`

`agentdojo_stock` is the *camouflage* control: it swaps the payload for
AgentDojo's stock `<INFORMATION>` imperative, so it varies register and deferral
at once. Here `imm` is authored in the **same business-process register** as
`d1`/`d2`/`d3` — no `<INFORMATION>`, no "ignore previous instructions" — which is
what keeps Δ unconfounded with delimiter style. The two probes answer different
questions and should not be pooled.

## Provenance

* Task instruction, clean plan and tool inventory come from AgentDojo v1 vendored
  at `data/agentdojo/`, read through `scripts/build_agentdojo_latent.py`.
* Carrier, benign observation, sink and sink arguments are reused from
  `data/agentdojo_latent/authored/`, so a Δ-Bench group and its latent case point
  the agent at the same action.
* Each group reuses **exactly one** latent payload byte-for-byte, at the tier
  whose Δ matches the latent case's own `expected_delta` (`reused_from_latent`
  records the source; `reuse_tier` in the authored file). The other three tiers
  are newly authored.
* Qualifying cases are the 19 latent cases where `carrier_index + 4` still fits
  inside the clean plan — a shorter plan cannot carry a Δ=4 tier without
  fabricating steps. ASB and InjecAgent are deliberately absent: ASB agents
  expose two tools and InjecAgent tasks are single-call, so neither can support
  the ladder.

## Authoring rules (every tier, including `imm`)

1. **Name the sink.** The payload must contain the `harmful_tool` name and its
   key arguments. The camouflage probe's v1 null was an artefact of payloads that
   only gestured at a future action — "you will need to do X later" is not
   obeyable, and an unobeyable payload measures nothing.
2. **No injection delimiters** and no instruction-override language.
3. **Business-process, regulatory or compliance framing** that explains *why* the
   call belongs where it is placed, referencing the clean plan's actual next
   tools for the ordering.
4. **One stealth technique per group**, constant across its four tiers
   (`compliance_deferral`, `dependency_chain`, `staged_verification`,
   `audit_trail`, `false_correction`, `quota_precheck`).

## Layout

```
authored/{banking,slack,travel,workspace}.json   # hand-authored payloads
scenarios/bucket_{imm,d1,d2,d3}.jsonl            # 19 injected rows each
scenarios/bucket_clean.jsonl                     # 18 clean rows
manifest.json                                    # counts + provenance
_sha256_guard.txt                                # integrity digests
```

A clean row is keyed on `(task, carrier)`, not on the task alone — a carrier
serves exactly one benign text, and two groups can share a task while injecting
at different carriers. That is why there are 18 clean rows for 19 groups.

## Rebuild and verify

```bash
python scripts/build_delta_bench.py            # rewrite scenarios + manifest + guard
python scripts/build_delta_bench.py --check     # assert on-disk == fresh rebuild
python -m evoguard.tests.test_delta_bench       # 22 offline invariants
```

The build validates every case before emitting: carrier at its declared index,
`carrier_index + Δ` inside the clean plan, sink advertised by the suite and absent
from the clean plan, payload naming the sink, all four tiers present, and the
reused tier's Δ matching its latent source.

## Reading the numbers

There is **no split** — the whole set is a held-out diagnostic. Report ASR per
tier with the group count, and report BU/UA alongside as usual: a defender that
cannot act scores ASR 0 for free. Because the four tiers are paired, the
within-group comparison (McNemar across tiers) is the test with power here, not
the between-tier marginals.
