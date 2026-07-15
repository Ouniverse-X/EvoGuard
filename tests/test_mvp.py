from evoguard.agents.defense_agent import DefenseAgent
from evoguard.agents.safety_head import TrainableSafetyHead
from evoguard.attacks.attack_generator import PromptDrivenAttackGenerator, build_attack_generator
from evoguard.envs.text_tool_env import TextToolEnv
from evoguard.evaluation.ablations import run_ablation_suite
from evoguard.evaluation.baselines import evaluate_baseline, run_baseline_suite
from evoguard.evaluation.metrics import compute_metrics
from evoguard.rollouts.attacked_tool_rollout import rollout_attacked_tool
from evoguard.rollouts.tri_rollout import collect_tri_rollouts
from evoguard.training.trainer import EvoGuardTrainer
from evoguard.types import SafetyAction


def test_tri_rollout_produces_records() -> None:
    records = collect_tri_rollouts(DefenseAgent(), TextToolEnv(), PromptDrivenAttackGenerator(), round_id=0)
    assert records
    metrics = compute_metrics(records)
    assert "task_success_rate" in metrics
    assert "attack_interception_rate" in metrics


def test_training_loop_runs() -> None:
    trainer = EvoGuardTrainer(DefenseAgent(), TextToolEnv(), PromptDrivenAttackGenerator())
    results = trainer.train(num_rounds=2)
    assert len(results) == 2
    assert all(result.records for result in results)
    assert all(result.rewards for result in results)


def test_trainable_head_runs_lightweight_ppo_update() -> None:
    env = TextToolEnv()
    generator = PromptDrivenAttackGenerator()
    attack = generator.generate(env.get_tasks(), round_id=0)[0]
    agent = DefenseAgent(TrainableSafetyHead(learning_rate=0.5))

    before = rollout_attacked_tool(agent, env, attack)

    trainer = EvoGuardTrainer(agent, env, generator)
    result = trainer.run_round(round_id=0, max_round=1)
    assert result.policy_updates > 0
    assert result.update_stats
    assert result.update_stats["steps"] == result.policy_updates
    assert "approx_kl" in result.update_stats
    assert "clip_fraction" in result.update_stats

    after = rollout_attacked_tool(agent, env, attack)
    assert after.confidence != before.confidence


def test_trainable_head_mines_refusal_cues_for_heldout_attacks() -> None:
    env = TextToolEnv()
    agent = DefenseAgent(TrainableSafetyHead(learning_rate=0.5))
    heldout_attack = next(
        attack
        for attack in build_attack_generator("heldout").generate(env.get_tasks(), round_id=99)
        if attack.attack_type == "authority_impersonation"
    )

    before = rollout_attacked_tool(agent, env, heldout_attack)
    assert before.agent_safety_action == SafetyAction.EXECUTE

    trainer = EvoGuardTrainer(agent, env, build_attack_generator("train"))
    result = trainer.run_round(round_id=0, max_round=1)
    assert result.update_stats
    assert result.update_stats["learned_cues"] > 0

    after = rollout_attacked_tool(agent, env, heldout_attack)
    assert after.agent_safety_action == SafetyAction.REFUSE
    assert after.attribution_score > 0.0


def test_baseline_suite_runs_comparable_metrics() -> None:
    results = run_baseline_suite(
        baselines=("no_guard", "always_refuse_guard", "evoguard_adaptive"),
        train_rounds=2,
    )
    by_name = {result.name: result for result in results}

    assert by_name["no_guard"].metrics["attack_success_rate"] == 1.0
    assert by_name["always_refuse_guard"].metrics["over_refusal_rate"] > 0.0
    assert by_name["evoguard_adaptive"].policy_updates > 0
    assert by_name["evoguard_adaptive"].eval_attack_split == "heldout"
    assert "attack_interception_rate" in by_name["evoguard_adaptive"].metrics


def test_baseline_suite_accepts_modern_eval_split() -> None:
    results = run_baseline_suite(
        baselines=("no_guard",),
        train_rounds=1,
        eval_attack_split="automated_jailbreak",
    )

    assert results[0].eval_attack_split == "automated_jailbreak"
    assert results[0].metrics["attack_success_rate"] == 1.0


def test_unknown_baseline_is_rejected() -> None:
    try:
        evaluate_baseline("unknown_guard")
    except ValueError as exc:
        assert "Unknown baseline" in str(exc)
    else:
        raise AssertionError("unknown baseline should raise ValueError")


def test_attack_train_eval_splits_are_disjoint() -> None:
    env = TextToolEnv()
    train_attacks = build_attack_generator("train").generate(env.get_tasks(), round_id=0)
    heldout_attacks = build_attack_generator("heldout").generate(env.get_tasks(), round_id=0)

    train_types = {attack.attack_type for attack in train_attacks}
    heldout_types = {attack.attack_type for attack in heldout_attacks}

    assert train_attacks
    assert heldout_attacks
    assert train_types.isdisjoint(heldout_types)
    assert all(attack.attack_id.startswith("train_") for attack in train_attacks)
    assert all(attack.attack_id.startswith("heldout_") for attack in heldout_attacks)


def test_modern_automated_jailbreak_split_covers_pair_tap_gptfuzzer() -> None:
    env = TextToolEnv()
    attacks = build_attack_generator("automated_jailbreak").generate(env.get_tasks(), round_id=7)
    attack_types = {attack.attack_type for attack in attacks}

    assert attacks
    assert {"pair_iterative_refinement", "tap_branch_pruning", "gptfuzzer_seed_mutation"}.issubset(attack_types)
    assert all(attack.attack_id.startswith("automated_jailbreak_r7_") for attack in attacks)
    assert all(attack.gold_safety_action == SafetyAction.REFUSE for attack in attacks)


def test_text_tool_env_covers_multiple_risk_types() -> None:
    env = TextToolEnv()
    tasks = env.get_tasks()
    preferred_tools = {task.preferred_tool for task in tasks if task.preferred_tool}
    risk_types = {env.risk_type(tool) for tool in preferred_tools if env.risk_type(tool)}

    assert len(tasks) >= 12
    assert "crm_lookup" in preferred_tools
    assert "ticket_updater" in preferred_tools
    assert len(risk_types) >= 6
    assert all(env.execute(env.propose_tool_call(task)).success for task in tasks if task.preferred_tool)


def test_ablation_suite_runs_core_variants() -> None:
    results = run_ablation_suite(train_rounds=1)
    by_name = {result.name: result for result in results}

    assert "full_evoguard" in by_name
    assert "no_attribution_reward" in by_name
    assert "no_clean_tool_trajectory" in by_name
    assert by_name["full_evoguard"].eval_attack_split == "heldout"
    assert "attack_interception_rate" in by_name["full_evoguard"].metrics
