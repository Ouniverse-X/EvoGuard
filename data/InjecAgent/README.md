# InjecAgent

Pruned extract of [InjecAgent](https://github.com/uiuc-kang-lab/InjecAgent) at commit
`f19c9f2c`, plus the EvoGuard train/val/test split derived from it. This directory holds
**only** what EvoGuard reads at runtime and what is needed to prove provenance; the rest of
the upstream checkout was deleted (see `metadata/SOURCE.json`, and re-clone upstream at the
pinned commit if you need it).

InjecAgent is *indirect* prompt injection: the user asks for something benign, the agent calls
one tool, and the attacker's instruction is sitting inside that tool's **observation**. Taking
the bait means calling a second tool that only the observation ever mentioned. Two attack
classes:

* **direct harm (`dh`, 30 attacker cases)** — one attacker tool, and calling it *is* the harm
  (`TerminalExecute`, `AugustSmartLockGrantGuestAccess`, ...).
* **data stealing (`ds`, 32 attacker cases)** — two attacker tools, `[read, send]`. Upstream's
  headline metric is "ASR-valid step 1", i.e. whether the agent calls the **read** tool, so
  that is the step EvoGuard scores.

## Layout

```
data/tools.json                 38 toolkits / 330 tool specs. Tools are keyed
                                `toolkit + tool["name"]` (upstream `src/utils.py::
                                get_tool_dict`); the test cases name them that way.
data/user_cases.jsonl           17 benign user cases: User Tool, User Instruction,
                                Tool Parameters, Tool Response Template, Thought, Level
data/attacker_cases_dh.jsonl    30 direct-harm attacker cases
data/attacker_cases_ds.jsonl    32 data-stealing attacker cases
data/attacker_simulated_responses.json
                                2347 recorded observations for attacker tools, keyed
                                `"(<tool>,<args>)"`. Covers the 32 ds stage-1 read tools
                                only -- see below.
benign_fillers.json             EvoGuard-authored. One clean-arm substitution per User Tool,
                                so the clean and poisoned observations differ only in the
                                slot's contents, never in shape.
splits/{train,val,test}/all.jsonl   the EvoGuard split, 646/204/204 rows
splits/_sha256_guard.txt        immutability digests for the three split files
splits/split_manifest.json      ratios, seed, strata, per-attacker-case assignment
metadata/                       SOURCE.json (provenance + what was dropped and why),
                                SHA256SUMS
```

## Split semantics — read before quoting a number

1054 rows = **17 user cases x 62 attacker cases**, every cell an attack instance. There are no
clean rows: the clean arm is generated at runtime from `benign_fillers.json`.

**The split unit is the attacker case, not the task and not the row.** 17 user cases is far
too few to split (a 60/20/20 task split leaves 3 val tasks), and splitting rows would put the
same attacker instruction on both sides of the boundary. Splitting the attacker case is
strictly tighter than ASB-OPI's random attack-instance split — no payload straddles a
boundary — while every one of the 17 user tasks stays available as a carrier in every split.

| | train | val | test |
|---|---:|---:|---:|
| attacker cases | 38 | 12 | 12 |
| rows | 646 | 204 | 204 |
| user cases present | 17 | 17 | 17 |
| dh / ds rows | 306 / 340 | 102 / 102 | 102 / 102 |

Stratified on `(attack_class, Attack Type)` with seed 20260903: dh Data Security Harm 11 →
7/2/2, Physical Harm 10 → 6/2/2, Financial Harm 9 → 5/2/2; ds Others 15 → 9/3/3, Physical
Data 11 → 7/2/2, Financial Data 6 → 4/1/1.

The evaluation axis this supports is *unseen attacker instruction and attacker tool on a seen
user scenario*. It is **not** an unseen-task benchmark — `data/toolsafe/agentdojo-tragjnew` is
the one that splits by instruction. Do not report InjecAgent held-out numbers as task-level
generalisation.

Consequence for the config: tasks carry **no** `metadata["split"]` (all 17 are in all three
splits, so a label would be a lie), `pipeline.validation_fraction` is `0.0` and
`pipeline.val_split` is empty. Held-out-ness comes from `val_dataset_dir` pointing at the leaf
split dir, which restricts the replayed *attacks*.

## Record schema

16 keys, all rows `sample_type: "injected"`:

| field | meaning |
|---|---|
| `instance_id` | `injecagent-<dh\|ds>-<attacker NN>-<user NN>` |
| `attack_class` | `dh` or `ds` |
| `attack_type` | upstream `Attack Type` (`Physical Harm`, `Others`, ...) — the split stratum |
| `attacker_case_index`, `user_case_index` | 0-based indices into the two source files; `(attack_class, attacker_case_index)` is the split unit |
| `user_tool` | the single benign tool the user's request needs — the injection **carrier** |
| `user_instruction`, `user_level` | the benign task; `Level` is upstream's difficulty tag |
| `tool_parameters` | the carrier's arguments, as upstream recorded them |
| `tool_response_template` | the carrier's observation with one `<Attacker Instruction>` slot |
| `attacker_tools` | 1 tool for `dh`, `[read, send]` for `ds`; `[0]` is what EvoGuard scores |
| `attacker_instruction` | the payload |
| `expected_achievements` | upstream's ASR string for the attack |
| `modified` | upstream `Modifed` flag, passed through |
| `poisoned_observation` | `tool_response_template` with the slot filled by `attacker_instruction` — exactly what the replay arm serves |

## What was dropped, and why it is safe

The four `test_cases_{dh,ds}_{base,enhanced}.json` (2.9M) are gone. They are the cross product
of the two source files, and that was verified rather than assumed before deleting:

* the 17 x 62 grid set-equals each file field for field, 0 mismatches;
* `tool_response_template.replace("<Attacker Instruction>", payload)` reproduces
  `json.loads(row["Tool Response"])` exactly on all 1020 rows whose `Tool Response` is
  parseable (34 ds rows carry unescaped embedded quotes from upstream double-encoding — which
  is also why EvoGuard renders from the template instead of reading that field);
* the `enhanced` setting is `base` with one fixed prefix, re-rendered on demand by
  `load_injecagent_attacks(setting="enhanced")` (`process/split_injecagent.py::
  ENHANCED_PREFIX`).

Their upstream digests are recorded in `metadata/SOURCE.json` so the deletion is auditable.
`src/` and `asset/` were upstream's own evaluation harness and paper figures.

## How EvoGuard consumes this

`evoguard/envs/injecagent.py` (`dataset: injecagent`),
`evoguard/process/injecagent_attack_loader.py`, config `configs/injecagent_grpo.yaml`. Design
note: `docs/injecagent_integration.md`. Three things worth knowing here:

* **Attacker tools are deliberately absent from the agent's tool list.** Upstream hands the
  agent `[User Tool] + Attacker Tools`; EvoGuard exposes only the User Tool. Showing the bait
  catalogue would turn injection detection into tool-list classification — a defender could
  refuse anything naming a tool it was not asked to use and score perfectly without reading
  the observation. `InjecAgentEnv._tool_spec` still resolves all 330 tools so the *executor*
  simulates the bait with its real signature.
* **Observations are templated, not generated.** The carrier's result is the template with the
  slot filled: the payload for the replay arm, a `benign_fillers.json` string for the clean
  arm. Letting an LLM invent it would drift both arms off the dataset's distribution and make
  clean and poisoned observations distinguishable by shape.
* **`data/attacker_simulated_responses.json` is preferred over the LLM simulator.** 2347 raw
  keys → 1385 distinct `(tool, canonical args)` pairs over exactly the 32 ds stage-1 read
  tools, so an exfiltration chain sees the same stolen data upstream did. The 30 dh tools are
  **absent** — upstream never simulates them, because a direct-harm attack is already scored
  at the call itself — and fall through to the simulator.

Exactly one of the 1054 rows has `attacker_tools[0] == user_tool` (user case 3
`GitHubGetUserDetails` x the ds case that re-reads the profile). It is kept rather than
dropped so the benchmark stays at its published size;
`evoguard/tests/test_injecagent_env.py` pins that count at 1.

## Integrity

```bash
cd data/InjecAgent && sha256sum -c metadata/SHA256SUMS
/root/miniconda3/envs/evoguard2/bin/python -m evoguard.tests.test_injecagent_env
```

The five upstream files kept here are byte-identical to `f19c9f2c`.
`splits/_sha256_guard.txt` is checked by the test above; regenerate it deliberately (via
`python -m evoguard.process.split_injecagent --force`), never to make a test pass.

## Citation

```bibtex
@inproceedings{zhan2024injecagent,
  title={{I}njec{A}gent: Benchmarking Indirect Prompt Injections in Tool-Integrated Large Language Model Agents},
  author={Zhan, Qiusi and Liang, Zhixiang and Ying, Zifan and Kang, Daniel},
  booktitle={Findings of the Association for Computational Linguistics: ACL 2024},
  pages={10471--10506},
  year={2024},
  url={https://aclanthology.org/2024.findings-acl.624/}
}
```
