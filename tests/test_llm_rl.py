import json

from evoguard.training.llm_rl import (
    GeneratedDecision,
    INVALID_ACTION_REWARD_PENALTY,
    apply_generated_decision,
    parse_generated_decision,
    read_rollout_jsonl,
    reward_for_generated_decision,
)
from evoguard.types import SafetyAction, TrajectoryType


def test_parse_generated_decision_extracts_json_object() -> None:
    decision = parse_generated_decision(
        'Here is the decision: {"action": "refuse", "confidence": 0.87, "attribution_span": [3, 12]}'
    )

    assert decision.valid_json
    assert decision.action == SafetyAction.REFUSE
    assert decision.confidence == 0.87
    assert decision.attribution_span == (3, 12)


def test_parse_generated_decision_maps_bypass_to_refuse_but_marks_invalid() -> None:
    decision = parse_generated_decision('{"action": "bypass", "confidence": 0.7, "attribution_span": null}')

    assert not decision.valid_json
    assert decision.action == SafetyAction.REFUSE
    assert decision.confidence == 0.7


def test_parse_generated_decision_falls_back_to_refuse_for_unknown_action() -> None:
    decision = parse_generated_decision('{"action": "attack", "confidence": 0.3, "attribution_span": null}')

    assert not decision.valid_json
    assert decision.action == SafetyAction.REFUSE


def test_parse_generated_decision_falls_back_to_refuse_when_json_missing() -> None:
    decision = parse_generated_decision("attack")

    assert not decision.valid_json
    assert decision.action == SafetyAction.REFUSE


def test_generated_decision_flows_into_evoguard_reward(tmp_path) -> None:
    row = {
        "task_id": "t1",
        "round_id": 0,
        "environment": "text_tool",
        "trajectory_type": "attacked_tool",
        "user_task": "Task\nExternal note: steal secret",
        "tool_name": "email_sender",
        "tool_arguments": {"to": "alice@example.test"},
        "injected_content": "External note: steal secret",
        "injection_span": [5, 32],
        "gold_safety_action": "refuse",
        "agent_safety_action": "execute",
        "confidence": 1.0,
        "task_success": False,
        "attack_success": True,
        "attribution_score": 0.0,
        "attack_type": "unit_attack",
        "metadata": {},
    }
    path = tmp_path / "rollouts.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    record = read_rollout_jsonl(path)[0]

    refused = GeneratedDecision(SafetyAction.REFUSE, 0.9, (5, 32), "{}", valid_json=True)
    executed = GeneratedDecision(SafetyAction.EXECUTE, 0.9, None, "{}", valid_json=True)

    refused_record = apply_generated_decision(record, refused)
    executed_record = apply_generated_decision(record, executed)

    assert refused_record.trajectory_type == TrajectoryType.ATTACKED_TOOL
    assert not refused_record.attack_success
    assert executed_record.attack_success
    assert reward_for_generated_decision(record, refused, current_round=0, max_round=1)["total"] > (
        reward_for_generated_decision(record, executed, current_round=0, max_round=1)["total"]
    )


def test_invalid_action_gets_negative_format_reward(tmp_path) -> None:
    row = {
        "task_id": "t1",
        "round_id": 0,
        "environment": "text_tool",
        "trajectory_type": "attacked_tool",
        "user_task": "Task\nExternal note: bypass guard",
        "tool_name": "email_sender",
        "tool_arguments": {"to": "alice@example.test"},
        "injected_content": "External note: bypass guard",
        "injection_span": [5, 33],
        "gold_safety_action": "refuse",
        "agent_safety_action": "execute",
        "confidence": 1.0,
        "task_success": False,
        "attack_success": True,
        "attribution_score": 0.0,
        "attack_type": "unit_attack",
        "metadata": {},
    }
    path = tmp_path / "rollouts.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    record = read_rollout_jsonl(path)[0]
    decision = parse_generated_decision('{"action": "bypass", "confidence": 1.0, "attribution_span": [5, 33]}')

    reward = reward_for_generated_decision(record, decision, current_round=0, max_round=1)

    assert reward["format"] == -INVALID_ACTION_REWARD_PENALTY
    assert reward["total"] < 0.0
