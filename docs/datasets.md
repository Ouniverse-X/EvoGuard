# EvoGuard Datasets

## Text Tool Dataset

The MVP uses synthetic controlled tasks with simulated tools.

Task families:

- Calculator tasks.
- Calendar lookup tasks.
- File summary tasks.
- Public changelog lookup tasks.
- CRM/customer-support lookup tasks.
- Email notification tasks.
- Ticket update tasks.
- Smart home setting tasks.
- Sensitive data exfiltration risks.
- Important file modification risks.
- Simulated bank transfer risks.
- High-risk device operation risks.
- Untrusted web request risks.
- Business record mutation risks.

Current simulated tools:

- `calculator`
- `calendar`
- `email_sender`
- `file_reader`
- `file_writer`
- `web_request`
- `bank_transfer_simulator`
- `smart_home_controller`
- `crm_lookup`
- `ticket_updater`

Current risk types:

- `external_communication`
- `state_change`
- `network_exfiltration`
- `financial_action`
- `physical_world_action`
- `business_record_mutation`

## Attack Pool Split

The MVP keeps attack generation templates split by use:

- Train attack pool:
  - `indirect_prompt_injection`
  - `parameter_misuse`
  - `context_manipulation`
  - `data_exfiltration`
  - `destructive_update`
  - `confirmation_bypass`
- Held-out evaluation attack pool:
  - `role_confusion`
  - `authority_impersonation`
  - `hidden_goal_shift`
  - `policy_exception_claim`
  - `benign_wrapper`
  - `multi_step_lure`

Training scripts use the train pool. Evaluation and baseline comparison use the held-out pool by default.

## ToolSafe / TS-Bench AgentHarm Subset

The project keeps the ToolSafe AgentHarm trajectory subset used in the current
experiments under the repository data directory:

```text
data/raw/toolsafe/agentharm-traj/benign_steps.json
data/raw/toolsafe/agentharm-traj/harmful_steps.json
```

Regenerate EvoGuard-format rollouts with:

```bash
python scripts/generate_toolsafe_rollouts.py \
  data/raw/toolsafe/agentharm-traj/benign_steps.json \
  data/raw/toolsafe/agentharm-traj/harmful_steps.json \
  --train-output data/rollouts/toolsafe_train_tri_rollouts.jsonl \
  --heldout-output data/eval/toolsafe_heldout_tri_rollouts.jsonl \
  --summary-output data/eval/toolsafe_rollout_summary.json
```

Current generated split:

| Split | Total | clean_tool | attacked_tool |
| --- | ---: | ---: | ---: |
| Train | 585 | 165 | 420 |
| Held-out | 146 | 41 | 105 |

The ToolSafe SFT training file is derived from the train rollouts:

```text
data/processed/toolsafe_safety_sft_train.jsonl
```

## Required Sample Fields

```json
{
  "task_id": "task_001",
  "round": 3,
  "environment": "text_tool",
  "trajectory_type": "attacked_tool",
  "user_task": "...",
  "tool_name": "email_sender",
  "tool_arguments": {},
  "injected_content": "...",
  "injection_span": [120, 180],
  "gold_safety_action": "refuse",
  "agent_safety_action": "execute",
  "confidence": 0.92,
  "task_success": false,
  "attack_success": true,
  "attribution_score": 0.1
}
```
