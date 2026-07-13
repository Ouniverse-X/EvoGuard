import json

from evoguard.evaluation.plots import (
    latest_ablation_rows,
    latest_baseline_rows,
    latest_training_rows,
    write_report_artifacts,
)


def test_latest_rows_deduplicate_appended_logs() -> None:
    baseline_rows = [
        {"baseline": "no_guard", "metrics": {"task_success_rate": 0.1}},
        {"baseline": "no_guard", "metrics": {"task_success_rate": 0.2}},
        {"baseline": "evoguard_adaptive", "metrics": {"task_success_rate": 0.3}},
    ]
    training_rows = [
        {"round_id": 0, "metrics": {"task_success_rate": 0.1}},
        {"round_id": 0, "metrics": {"task_success_rate": 0.4}},
        {"round_id": 1, "metrics": {"task_success_rate": 0.5}},
    ]

    assert latest_baseline_rows(baseline_rows)[0]["metrics"]["task_success_rate"] == 0.2
    assert [row["round_id"] for row in latest_training_rows(training_rows)] == [0, 1]
    assert latest_training_rows(training_rows)[0]["metrics"]["task_success_rate"] == 0.4
    assert latest_ablation_rows(
        [
            {"ablation": "full_evoguard", "metrics": {"task_success_rate": 0.1}},
            {"ablation": "full_evoguard", "metrics": {"task_success_rate": 0.9}},
        ]
    )[0]["metrics"]["task_success_rate"] == 0.9


def test_write_report_artifacts(tmp_path) -> None:
    baseline_log = tmp_path / "baseline.jsonl"
    training_log = tmp_path / "training.jsonl"
    ablation_log = tmp_path / "ablation.jsonl"
    baseline_rows = [
        {
            "baseline": "no_guard",
            "train_attack_split": "train",
            "eval_attack_split": "heldout",
            "metrics": {
                "task_success_rate": 0.4,
                "attack_interception_rate": 0.0,
                "over_refusal_rate": 0.0,
                "attack_success_rate": 1.0,
            },
        },
        {
            "baseline": "evoguard_adaptive",
            "train_attack_split": "train",
            "eval_attack_split": "heldout",
            "metrics": {
                "task_success_rate": 0.8,
                "attack_interception_rate": 0.7,
                "over_refusal_rate": 0.1,
                "attack_success_rate": 0.2,
            },
        },
    ]
    training_rows = [
        {
            "round_id": 0,
            "mean_reward": -0.5,
            "policy_updates": 2,
            "metrics": {
                "task_success_rate": 0.4,
                "attack_interception_rate": 0.0,
                "over_refusal_rate": 0.0,
                "attack_success_rate": 1.0,
            },
        },
        {
            "round_id": 1,
            "mean_reward": 0.7,
            "policy_updates": 1,
            "metrics": {
                "task_success_rate": 0.8,
                "attack_interception_rate": 0.7,
                "over_refusal_rate": 0.1,
                "attack_success_rate": 0.2,
            },
        },
    ]
    ablation_rows = [
        {
            "ablation": "full_evoguard",
            "policy_updates": 3,
            "metrics": {
                "task_success_rate": 0.8,
                "attack_interception_rate": 0.7,
                "over_refusal_rate": 0.1,
                "attack_success_rate": 0.2,
            },
        }
    ]
    baseline_log.write_text("\n".join(json.dumps(row) for row in baseline_rows), encoding="utf-8")
    training_log.write_text("\n".join(json.dumps(row) for row in training_rows), encoding="utf-8")
    ablation_log.write_text("\n".join(json.dumps(row) for row in ablation_rows), encoding="utf-8")

    outputs = write_report_artifacts(
        baseline_log,
        training_log,
        tmp_path / "reports",
        tmp_path / "figures",
        ablation_log=ablation_log,
    )

    assert "EvoGuard Adaptive" in outputs["baseline_markdown"].read_text(encoding="utf-8")
    assert "\\begin{tabular}" in outputs["baseline_latex"].read_text(encoding="utf-8")
    assert "Round" in outputs["training_markdown"].read_text(encoding="utf-8")
    assert "Full EvoGuard" in outputs["ablation_markdown"].read_text(encoding="utf-8")
    assert "<svg" in outputs["baseline_svg"].read_text(encoding="utf-8")
    assert "<polyline" in outputs["training_svg"].read_text(encoding="utf-8")
    assert "<svg" in outputs["ablation_svg"].read_text(encoding="utf-8")
