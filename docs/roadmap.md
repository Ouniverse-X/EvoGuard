# EvoGuard Roadmap

## Phase 0: Project Initialization

Status: In Progress

Tasks:
- [x] Create repository structure.
- [x] Write CLAUDE.md.
- [x] Write initial method design.
- [x] Define minimum viable prototype.

## Phase 1: Text Tool-Calling Prototype

Status: In Progress

Goal:
Build a minimal text-based tool-use environment where the Agent can call simulated tools such as calculator, calendar, email sender, file reader, file writer, web request, bank transfer simulator, and smart home controller.

Tasks:
- [x] Define tool schema.
- [x] Define benign and risky tasks.
- [x] Define risky tool calls.
- [x] Expand text-tool task coverage with CRM, support-ticket, web, file-write, finance, and smart-home risk types.
- [x] Implement no-tool, clean-tool, and attacked-tool trajectories.
- [x] Log tool call decisions.

Deliverable:
A working text-tool environment and rollout logger.

## Phase 2: Adaptive Attack Generator

Status: In Progress

Goal:
Implement a prompt-driven red-team attack generator that uses previous defender failure cases as references.

Tasks:
- [x] Design attack prompt templates.
- [x] Split train and held-out evaluation attack templates.
- [x] Expand train and held-out attack pools to six controlled styles each.
- [x] Implement attack memory.
- [x] Generate candidate attacks.
- [x] Filter attacks by current defense Agent failure.
- [x] Store successful attacks with round index and injection span.

Deliverable:
Round-based adaptive attack generation pipeline.

## Phase 3: Defense Agent and Safety Head

Status: In Progress

Goal:
Add a safety judgment interface that predicts EXECUTE / ASK_CONFIRMATION / REFUSE, confidence, and attribution.

Tasks:
- [x] Implement rule-based safety action classifier for MVP.
- [x] Implement confidence prediction.
- [x] Implement attribution span scoring.
- [x] Integrate safety head before tool execution.

Deliverable:
Defense Agent wrapper.

## Phase 4: Reward Design and RL Training

Status: In Progress

Goal:
Train the defense Agent with task reward, safety reward, attribution reward, and KL regularization.

Tasks:
- [x] Implement task reward.
- [x] Implement safety reward.
- [x] Implement attribution reward.
- [x] Implement combined reward.
- [x] Implement trainer scaffold.
- [x] Add dependency-free lightweight PPO safety-head updates for the MVP.
- [x] Monitor over-refusal rate.

Deliverable:
End-to-end EvoGuard training loop scaffold with a lightweight PPO-trainable safety head.

## Phase 5: Baselines

Status: Not Started

Baselines:
- [ ] No guard.
- [ ] Rule-based guard.
- [ ] Static guard.
- [ ] ToolSafe-style static defense.
- [ ] MOSAIC-style refuse-action training.
- [ ] EAPO-style fixed contrastive rollout.
- [ ] AgentWatcher-style attribution detector.

Deliverable:
Comparable baseline results.

## Phase 6: Step-Level Tool Safety Datasets

Status: Not Started

Tasks:
- [ ] ToolSafe/TS-Bench integration.
- [ ] TraceSafe integration.
- [ ] Cross-dataset generalization evaluation.

Deliverable:
Step-level tool-call safety evaluation.

## Phase 7: Experiments and Paper

Status: Not Started

Tasks:
- [ ] Main results.
- [ ] Ablation study.
- [ ] Generalization study.
- [ ] Attack evolution curves.
- [ ] Over-refusal analysis.
- [ ] Attribution visualization.
- [ ] Paper writing.
