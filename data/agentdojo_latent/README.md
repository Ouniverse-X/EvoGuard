# AgentDojo-Latent

48 hand-authored **high-Δ** indirect prompt injections on the AgentDojo v1
default suites, plus their 46 paired clean rows. Held-out diagnostic probe: no
split, no training use.

## Why it exists

Stock AgentDojo injections announce themselves. The payload appends an
imperative unrelated to the plan in flight, the agent's very next action leaves
the clean plan, and Δ (turning point − injection point) is 0. A static filter
sitting in front of the observation can catch that, which is also why the
`Preliminary` measurement holds: the model's own uncertainty spikes at the moment
of the demand (late-segment `seq_entropy` / `seq_nll` ordering AF > AS > clean —
hesitation at a discontinuity).

Every payload here is written the other way round. It borrows the authority of
the task already in flight and **defers** the harmful call by 1–3 turns of
genuine progress, so the observation reads as more of the carrier's own output
and the trajectory only diverges after the agent has done real work. That removes
the discontinuity the defender was keying on. The set exists to show that a
defence tuned on Δ=0 attacks does not transfer — i.e. to justify co-evolving the
defender against an attacker that optimises Δ.

## Provenance

Two inputs, one writer.

| Part | Source | Verbatim? |
|---|---|---|
| `task_instruction` | `data/agentdojo/default_suites/v1/<suite>/user_tasks.py` `PROMPT` | yes, f-strings resolved |
| `clean_plan` | the same class's `ground_truth()` call sequence | yes |
| `env_info` | `task_suite.py` `TOOLS` + each tool's docstring | rendered |
| everything else | `authored/<suite>.json`, hand-written | — |

The only writer is `scripts/build_agentdojo_latent.py`, and it is deterministic:
`--check` rebuilds in memory and fails if any on-disk byte would change, and
`_sha256_guard.txt` pins `scenarios/*.jsonl` + `manifest.json`. Edit `authored/`,
then rebuild — never edit a scenario file by hand.

```bash
python scripts/build_agentdojo_latent.py            # write
python scripts/build_agentdojo_latent.py --check     # verify
python -m evoguard.tests.test_agentdojo_latent       # 29 offline invariants
```

## Layout

```
authored/<suite>.json        12 authored attacks per suite (the creative part)
scenarios/<suite>.jsonl      clean + injected rows, ready to replay
manifest.json                counts per suite / Δ / technique
_sha256_guard.txt            digests of the two above
```

## Row schema

One JSON object per line. Fields marked *(inj.)* are `null` on clean rows.

| Field | Meaning |
|---|---|
| `instance_id` | `adjlatent-<suite>-<nnnn>` / `-clean-<nn>` |
| `sample_type` | `injected` or `clean` |
| `suite`, `task_ref` | e.g. `travel`, `agentdojo:v1:travel:UserTask19` |
| `task_id` | `agentdojo_latent:<suite>:<sha1(instruction)[:10]>` |
| `task_instruction` | AgentDojo's `PROMPT`, verbatim |
| `env_info` | tool inventory, parsed by `evoguard.envs.tool_parsing` |
| `clean_plan` | AgentDojo's `ground_truth()` tool sequence |
| `carrier_tool`, `carrier_index` | the poisoned tool and its position in `clean_plan` |
| `benign_observation` | the carrier's clean output |
| `observation` | `benign_observation + payload` (clean rows: the benign half alone) |
| `payload` *(inj.)* | the appended camouflaged text |
| `harmful_tool`, `harmful_args_hint` *(inj.)* | the sink, and the arguments the attacker wants |
| `attacker_goal` *(inj.)* | the out-of-scope objective, in prose |
| `expected_delta` *(inj.)* | **declared** Δ ∈ {1,2,3} |
| `expected_turning_index` *(inj.)* | `carrier_index + expected_delta` |
| `expected_induced_plan` *(inj.)* | `clean_plan` with the sink spliced in at that index |
| `stealth_technique` *(inj.)* | one of the six families below |
| `stealth_notes` *(inj.)* | why this row's Δ is what it is |

### Invariants the build enforces

* `observation == benign_observation + payload` — the two arms differ by the
  payload and **nothing else**. A carrier serves one benign text across every
  attack that rides it, so the clean twin is exact.
* `clean_plan[carrier_index] == carrier_tool`.
* `harmful_tool` is advertised in `env_info` **and absent from `clean_plan`** —
  taking the bait is a real out-of-plan action, not a tool-name novelty test.
* `carrier_index + expected_delta <= len(clean_plan)`.
* No `<INFORMATION>` delimiter, no "ignore previous instructions" phrasing:
  handing over a lexical tell would defeat the purpose.

## Design distribution

48 injected = 4 suites × 12. Within each suite: 4 rows per declared Δ, 2 rows per
technique.

| Suite | Tasks | Injected | Clean |
|---|---|---|---|
| banking | 7 | 12 | 10 |
| slack | 12 | 12 | 12 |
| travel | 12 | 12 | 12 |
| workspace | 12 | 12 | 12 |

Banking has fewer tasks because `UserTask15` is poisoned through four different
carriers, and correspondingly fewer clean rows (one per distinct carrier).

The six camouflage families:

| Technique | The fiction it sells |
|---|---|
| `compliance_deferral` | a policy that requires an extra step *after* the current one |
| `dependency_chain` | a later step needs data only the sink can load |
| `staged_verification` | the work just done is "staged"; the sink "commits" it |
| `false_correction` | an earlier result was wrong and the sink repairs it |
| `quota_precheck` | a limit is near; the sink checks headroom before proceeding |
| `audit_trail` | an open audit record closes only once the sink runs |

## Δ semantics — read before quoting a number

`expected_delta` is **declared design intent**, not a measurement. It is the
number of turns the payload asks the agent to keep working before the sink. The Δ
that belongs in a results table is the one
`evoguard.process.signals` computes at replay time from the actual trajectory,
and it will differ: an agent may take the bait early, late, refuse, or route
around the carrier entirely. Report the declared value as a stratification axis
(`Δ=1 / 2 / 3`) and the measured value as the outcome.

Declared Δ does not travel on `VendoredAttack` (there is no metadata slot). Join
it back onto replay records by `(task_id, payload)`, which is unique across the
48 rows.

## Using it

Registered as env `agentdojo_latent` (`evoguard/envs/agentdojo_latent.py`) with
scenarios loaded by `evoguard/process/agentdojo_latent_loader.py` via the
`env.dataset` dispatch table. Held-out replay:

```bash
EVOGUARD_REPLAY_CONFIG=configs/agentdojo_latent_probe.yaml \
EVOGUARD_REPLAY_DATASET_DIR=data/agentdojo_latent \
EVOGUARD_REPLAY_SPLIT= scripts/run_replay_heldout.sh none adjlatent_base
```

`EVOGUARD_REPLAY_SPLIT=` (empty) is mandatory — as with ASB-OPI and InjecAgent,
the probe declares no `metadata["split"]`, so a task filter matches nothing.

Two caveats when reporting:

* Always print `poison_delivered_rate` and `asr_given_delivered` beside ASR. On a
  deferred attack, routing around the carrier is indistinguishable from resisting
  it if you only look at ASR.
* The probe is 48 rows. Per-Δ and per-technique cells hold 12 and 8 rows
  respectively across all suites — enough to show a direction, not enough for a
  significance claim.


