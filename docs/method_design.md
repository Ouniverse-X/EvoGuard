# EvoGuard Method Design

## 1. Problem Formulation

We study a tool-calling agent that receives a user task, observes optional contextual content, and may request execution of a tool call. A defense agent intercepts each proposed tool call before execution and outputs a safety action, confidence score, and attribution signal.

Let a task be `x`, context be `c`, proposed tool call be `u = (tool_name, arguments)`, and safety action be:

```text
a in {EXECUTE, ASK_CONFIRMATION, REFUSE}
```

The defender policy is:

```text
pi_theta(a, confidence, attribution | x, c, u)
```

The objective is to maximize task utility while minimizing unsafe tool execution under evolving prompt-injection distributions.

## 2. Threat Model

The attacker can modify untrusted contextual content before the tool call decision. The attacker can attempt indirect prompt injection, context manipulation, misleading instructions, parameter-level misuse, and multi-step persuasion inside controlled benchmark tasks.

The attacker cannot access real credentials, execute real-world actions, bypass the benchmark harness, or directly modify the defender parameters. All risky tools are simulated.

The attack generator is prompt-driven and fixed. It adapts by conditioning on previous failure cases, but it is not trained as a policy. Only the defense agent is updated.

## 3. Tool-Call Safety Action Space

The safety action space has three states:

- `EXECUTE`: allow the tool call.
- `ASK_CONFIRMATION`: require explicit user confirmation before execution.
- `REFUSE`: block the tool call.

This action space distinguishes benign execution, uncertain high-risk situations, and clear refusal cases. It also allows over-refusal to be measured separately from safe caution.

## 4. Tri-Trajectory Safety Rollout

For each task, EvoGuard builds three trajectories:

- No-tool trajectory: solve without tools to estimate whether external action is necessary.
- Clean tool trajectory: solve with benign tool context to preserve task completion ability.
- Attacked tool trajectory: solve with injected adversarial context to train risk detection.

The three trajectories provide contrastive supervision over tool necessity, benign execution, and adversarial safety.

## 5. Prompt-Driven Adaptive Attack Generator

At each evolution round, the attack generator creates candidate attacks from the task set, tool schema, attack templates, and recent defender failures. Each generated sample records:

- attack type
- target tool
- generation round
- injected content
- insertion span
- intended gold safety action

The generator is fixed; adaptation comes from feeding previous failures into the prompt/template context.

## 6. Successful Attack Filtering

Candidate attacks are rolled out against the current defender. A candidate is successful if the defender executes a tool call whose gold action is not `EXECUTE`. Successful attacks are stored in attack memory and included in future training.

This filtering creates a curriculum driven by current model weaknesses.

## 7. Risk-Aware Reward Design

The MVP reward is:

```text
R = R_task + lambda_safety R_safety + lambda_attr R_attr - beta KL
```

For dangerous attacked trajectories:

```text
R_safety = +1.0 if REFUSE
R_safety = +0.5 if ASK_CONFIRMATION
R_safety = - penalty if EXECUTE
```

The penalty is:

```text
penalty = base_penalty
        * (1 + gamma_strength * normalized_attack_round)
        * (1 + gamma_conf * confidence)
        * (1 + gamma_stale * normalized_staleness)
```

This penalizes high-confidence unsafe execution, later-round attacks, and early attacks that remain unsolved after many rounds.

## 8. Attribution Supervision

Each injected attack records its insertion span. If the defender predicts an attribution span, EvoGuard computes token or character span IoU with the true injection span. For distributional attribution, the score is the probability mass assigned to injected tokens.

## 9. RL Optimization

The intended full-scale backend is PPO or GRPO with KL regularization against a reference policy. The MVP now contains a dependency-free lightweight PPO backend for the linear safety head. The trainer computes reward batches, normalizes advantages from the existing combined reward, and applies a clipped policy-gradient update to the probability of the actually sampled safety action. The implementation logs PPO diagnostics such as approximate KL and clip fraction. This backend validates the training interface before replacing the linear policy with a neural model or TRL-backed trainer.

## 10. Algorithm Pseudocode

```text
for round_id in range(num_rounds):
    candidate_attacks = attack_generator.generate(tasks, tools, recent_failures)
    successful_attacks = filter_by_current_defender(candidate_attacks)

    train_pool = []
    for task in train_tasks:
        train_pool += [
            rollout_no_tool(agent, task),
            rollout_clean_tool(agent, task),
        ]
    train_pool += successful_attacks

    rewards = reward_fn.compute(train_pool)
    agent = trainer.update(agent, train_pool, rewards, reference_agent)
    attack_memory.add(round_id, extract_failures(train_pool))
    metrics = evaluator.evaluate(agent, eval_tasks, heldout_attacks)
    logger.log(round_id, metrics)
```

## 11. Differences from Prior Work

EvoGuard is designed to differ from static guard and static contrastive methods by using round-based adaptive attack generation, pre-execution tool-call interception, explicit over-refusal tracking, and attribution-aware risk rewards.
