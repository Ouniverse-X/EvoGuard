"""Extract GRPO-training prompts from collected TrajectoryRecords
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Optional

from evoguard.agents.prompts import build_system_prompt, render_history
from evoguard.core.types import (
    AttackOutcome,
    Task,
    ToolSpec,
    Trajectory,
    TrajectoryKind,
    TrajectoryRecord,
)
from evoguard.process.dataset_builder import (
    DefenderDatasetBuilder,
    _infer_target_tool,
)
from evoguard.training.grpo_reward import PromptMeta, _action_signature
from evoguard.utils.logging import get_logger

logger = get_logger("training.grpo_prompt_extraction")


@dataclass
class GrpoPromptRow:
    """One extracted prompt ready for consumption by native_grpo_runner."""

    system: str                 # rendered system message including tools+task instruction
    user: str                   # render_history(actions[:step_at_injection_plus_one])
    meta: PromptMeta            # passed alongside completion(s) into reward function


@dataclass
class ExtractionStats:
    """Diagnostic counters returned alongside extraction results."""

    n_attacked_records_seen: int = 0       # input denominator
    n_skipped_no_signals: int = 0          # missing rec.signals entirely
    n_skipped_missing_inj_point: int = 0   # signals present but inj_point None
    n_skipped_step_not_found: int = 0      # no actions[i].turn == inj_point+1 exists
    n_candidates_before_cap: int = 0        # survived filters before applying max_prompts cap
    n_unique_task_ids_in_result: int = 0   # spread indicator
    n_capped_away: int = 0                  # dropped due to exceeding grpo_max_prompts_per_round

    # ---- benign (clean-trajectory) prompt pool -------------------------- #
    n_clean_records_seen: int = 0
    n_clean_skipped_too_short: int = 0      # <2 actions => no next-step ground truth
    n_clean_skipped_low_utility: int = 0    # clean rollout itself failed the task
    n_clean_skipped_no_task: int = 0        # task_id absent from dataset_builder registry
    n_clean_candidates_before_cap: int = 0
    n_clean_in_result: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


# --------------------------------------------------------------------------- #
# Internal helpers                                                             #
# --------------------------------------------------------------------------- #
def _find_post_injection_step_index(
    traj: Trajectory,
    injection_point: Optional[int],
) -> Optional[int]:
    """Smallest index ``i`` such that ``traj.actions[i].turn == injection_point + 1``.

    Returns ``None`` when no matching turn exists (rollout terminated early).
    Per spec §2.1 contract.
    """
    if injection_point is None:
        return None
    target_turn = int(injection_point) + 1
    for idx, act in enumerate(traj.actions):
        try:
            t_act = int(act.turn)
        except Exception:
            continue                                                # noqa: BLE001
        if t_act == target_turn:
            return idx
    # Fall back gracefully: some histories number turns slightly differently;
    # accept smallest index whose turn > injection_point instead of strict equality.
    fallback = next(
        (idx for idx, act in enumerate(traj.actions)
         if isinstance(act.turn, int) and int(act.turn) > int(injection_point)),
        None,
    )
    return fallback


def _aligned_clean_action_step(
    clean_trajectory: Optional[Trajectory],
    aligned_turn_threshold: Optional[int],
) -> tuple[str, dict[str, Any]]:
    """Clean twin's first action at/after the threshold as ``(name, args)``.

    Returns ``("", {})`` when no such action exists. The args half is what lets
    the reward's structural pre-check require the *right* call rather than merely
    the right tool name: an exact ``(tool, args)`` signature match against this
    action scores ``R_PROGRESS_ADVANCE`` with no judge call.
    """
    if clean_trajectory is None or not getattr(clean_trajectory, "actions", None):
        return "", {}
    if aligned_turn_threshold is None:
        return "", {}
    thr_int = int(aligned_turn_threshold)
    for act in clean_trajectory.actions:
        try:
            ta = int(act.turn)
        except Exception:
            continue                                                  # noqa: BLE001
        if ta >= thr_int:
            tc = getattr(act, "tool_call", None)
            if tc is not None:
                name = getattr(tc, "name", "")
                if name:
                    return str(name), dict(getattr(tc, "arguments", None) or {})
    return "", {}


def _prefix_action_signatures(actions: Any) -> tuple[str, ...]:
    """Signatures of the tool calls already visible in the prompt prefix.

    Handed to :class:`PromptMeta` so the reward can tell "took the next step"
    apart from "re-issued a step it had already taken", which is what made the
    observed single-tool loops profitable.
    """
    sigs: list[str] = []
    for act in (actions or []):
        tc = getattr(act, "tool_call", None)
        if tc is None:
            continue
        sig = _action_signature(getattr(tc, "name", ""), getattr(tc, "arguments", None))
        if sig and sig not in sigs:
            sigs.append(sig)
    return tuple(sigs)


# --------------------------------------------------------------------------- #
# Public entry point                                                           #
# --------------------------------------------------------------------------- #
def _cap_with_task_diversity(
    candidates: list[GrpoPromptRow],
    max_prompts: int,
    rng: random.Random,
) -> tuple[list[GrpoPromptRow], int]:
    """Trim ``candidates`` to ``max_prompts`` spreading slots across task_ids.

    Returns ``(kept_rows, n_dropped)``. Under the cap the input list is returned
    verbatim so ordering stays reproducible.
    """
    if max_prompts <= 0:
        return [], len(candidates)
    if len(candidates) <= max_prompts:
        return candidates, 0

    groups: dict[str, list[int]] = {}
    for ci, cand in enumerate(candidates):
        groups.setdefault(cand.meta.task_id, []).append(ci)

    n_groups = len(groups)
    per_group_quota = max(1, math.ceil(max_prompts / max(1, n_groups)))

    kept_indices: list[int] = []
    remaining_quota = max_prompts
    # First pass: distribute base quotas fairly across groups.
    for grp_indices in groups.values():
        take_n = min(per_group_quota, len(grp_indices))
        kept_indices.extend(grp_indices[:take_n])
        remaining_quota -= take_n
        if remaining_quota <= 0:
            break

    # Second pass: fill leftover slots round-robin over groups that had more
    # candidates than the base quota.
    leftover_pool_iterators = {
        tid: iter(idxs[per_group_quota:])
        for tid, idxs in groups.items() if len(idxs) > per_group_quota
    }
    guard_counter = 0
    while remaining_quota > 0 and leftover_pool_iterators and guard_counter < 10 ** 6:
        guard_counter += 1
        progressed_this_round = False
        for tid in sorted(leftover_pool_iterators.keys(), key=lambda x: rng.random()):
            if remaining_quota <= 0:
                break
            nxt = next(leftover_pool_iterators[tid], None)
            if nxt is None:
                continue
            kept_indices.append(nxt)
            remaining_quota -= 1
            progressed_this_round = True
        if not progressed_this_round:
            break

    seen_set: set[int] = set()
    deduped_kept: list[int] = []
    for kidx in kept_indices:
        if kidx not in seen_set:
            deduped_kept.append(kidx)
            seen_set.add(kidx)

    final_rows = [candidates[i] for i in deduped_kept][:max_prompts]
    return final_rows, len(candidates) - len(final_rows)


def _clean_cut_index(traj: Trajectory) -> Optional[int]:
    """Index of the clean action to be predicted; ``None`` if unusable.

    Cuts at the midpoint so the prompt lands mid-task (a state where the model
    must both continue the plan and pick the right tool) rather than at turn 0
    where every task looks the same. Requires >=2 actions so at least one action
    remains as ground truth after the prefix.
    """
    acts = getattr(traj, "actions", None) or []
    if len(acts) < 2:
        return None
    return max(1, len(acts) // 2)


def _clean_next_tool_call(traj: Trajectory, cut_index: int) -> tuple[str, dict[str, Any]]:
    """First clean tool call at/after ``cut_index`` as ``(name, args)``."""
    for act in (getattr(traj, "actions", None) or [])[cut_index:]:
        tc = getattr(act, "tool_call", None)
        if tc is not None:
            name = getattr(tc, "name", "")
            if name:
                return str(name), dict(getattr(tc, "arguments", None) or {})
    return "", {}


def extract_grpo_prompts(
    *,
    records: list[TrajectoryRecord],
    dataset_builder: DefenderDatasetBuilder,
    max_prompts: int = 32,
    seed: int = 0,
    clean_ratio: float = 0.0,
) -> tuple[list[GrpoPromptRow], ExtractionStats]:
    """Build GRPO-ready prompt rows from collected trajectory records.

    Parameters mirror caller-side state already held by Pipeline.maybe_train_defender:

      * ``records``           -- raw outputs of collect_tri_rollouts. ``ATTACKED``
                                 rows produce post-injection prompts; ``CLEAN``
                                 rows produce benign prompts when ``clean_ratio>0``.
      * ``dataset_builder``   -- provides registries mapping task_id -> (Task,[ToolSpec])
                                 needed for rendering proper system prompts containing AVAILABLE_TOOLS_JSON.
      * ``max_prompts``       -- hard upper bound honoring TrainingConfig.grpo_max_prompts_per_round.
      * ``seed``              -- reproducible tie-breaking RNG used ONLY when capping kicks in
                                 (otherwise order preserved verbatim).
      * ``clean_ratio``       -- fraction of ``max_prompts`` reserved for benign
                                 prompts, in [0,1). ``0.0`` (default) reproduces
                                 the attacked-only behaviour bit-for-bit. At 0.5
                                 the trainer sees roughly one benign prompt per
                                 attacked one, which is what gives the
                                 group-relative advantage a direction that
                                 rewards serving the user, not only blocking.

    Returns parallel-aligned lists wrapped inside ``GrpoPromptRow`` containers plus an
    :class:`ExtractionStats` summary suitable for logging diagnostics without re-running filtering.
    """

    stats = ExtractionStats()
    rng = random.Random(seed)

    cleans_by_task: dict[str, Trajectory] = {}
    for rcd in records:
        if rcd.kind is TrajectoryKind.CLEAN:
            cleans_by_task[rcd.task_id] = rcd.trajectory

    candidates: list[GrpoPromptRow] = []

    for rcd in records:
        if rcd.kind is not TrajectoryKind.ATTACKED:
            continue
        stats.n_attacked_records_seen += 1

        sigs = getattr(rcd, "signals", None)
        if sigs is None:
            stats.n_skipped_no_signals += 1
            continue
        inj_point = getattr(sigs, "injection_point", None)
        tp_point = getattr(sigs, "turning_point", None)
        delta_norm_raw = getattr(sigs, "delta_normalized", None)
        delta_norm_val = float(delta_norm_raw) if delta_norm_raw is not None else 0.0

        if inj_point is None:
            stats.n_skipped_missing_inj_point += 1
            continue

        step_i = _find_post_injection_step_index(rcd.trajectory, inj_point)
        if step_i is None:
            stats.n_skipped_step_not_found += 1
            continue

        task_obj: Optional[Task] = dataset_builder._tasks.get(rcd.task_id)
        if task_obj is None:
            logger.debug("[grpo_extract] skipping %r: no registered Task.", rcd.task_id)
            stats.n_skipped_step_not_found += 1     # closest-fit bucket; rare path
            continue

        tools_for_task: list[ToolSpec] = dataset_builder._tools.get(rcd.task_id, [])
        sys_str = build_system_prompt(task_obj, tools_for_task, None)

        history_prefix_actions = rcd.trajectory.actions[:step_i]
        user_text = render_history(history_prefix_actions)

        target_tool_name = (
            _infer_target_tool(rcd.attack) if rcd.attack is not None else ""
        )

        # Align onto clean twin A's corresponding turn so reward's recover-bonus has ground truth.
        align_thr = (int(inj_point) + 1) if inj_point is not None else None
        clean_twin_action_tool_name, clean_twin_action_args = _aligned_clean_action_step(
            cleans_by_task.get(rcd.task_id),
            align_thr,
        )

        meta = PromptMeta(
            task_id=str(task_obj.task_id),
            task_instruction=(task_obj.instruction or "").strip(),
            injection_point=inj_point,
            turning_point=tp_point,
            delta_normalized=delta_norm_val,
            target_tool_name=target_tool_name.strip(),
            clean_twin_action_tool_name=clean_twin_action_tool_name.strip(),
            clean_twin_action_args=clean_twin_action_args,
            prefix_action_signatures=_prefix_action_signatures(history_prefix_actions),
        )
        candidates.append(GrpoPromptRow(system=sys_str, user=user_text, meta=meta))

    # ------------------------------------------------------------------ #
    # Benign pass: one prompt per usable CLEAN trajectory                 #
    # ------------------------------------------------------------------ #
    clean_candidates: list[GrpoPromptRow] = []
    if clean_ratio > 0.0:
        for rcd in records:
            if rcd.kind is not TrajectoryKind.CLEAN:
                continue
            stats.n_clean_records_seen += 1
            cut = _clean_cut_index(rcd.trajectory)
            if cut is None:
                stats.n_clean_skipped_too_short += 1
                continue
            util = getattr(rcd, "utility", None)
            if util is not None and float(util) < 0.5:
                # Training on a clean rollout that itself failed the task would
                # teach the policy to imitate failure. ``None`` means unscored,
                # which we accept rather than silently emptying the pool.
                stats.n_clean_skipped_low_utility += 1
                continue
            task_obj_c: Optional[Task] = dataset_builder._tasks.get(rcd.task_id)
            if task_obj_c is None:
                stats.n_clean_skipped_no_task += 1
                continue
            tools_c: list[ToolSpec] = dataset_builder._tools.get(rcd.task_id, [])
            clean_next_name, clean_next_args = _clean_next_tool_call(rcd.trajectory, cut)
            meta_c = PromptMeta(
                task_id=str(task_obj_c.task_id),
                task_instruction=(task_obj_c.instruction or "").strip(),
                injection_point=None,
                turning_point=None,
                delta_normalized=0.0,
                target_tool_name="",
                clean_twin_action_tool_name=clean_next_name.strip(),
                is_clean=True,
                clean_twin_action_args=clean_next_args,
                prefix_action_signatures=_prefix_action_signatures(
                    rcd.trajectory.actions[:cut]
                ),
            )
            clean_candidates.append(
                GrpoPromptRow(
                    system=build_system_prompt(task_obj_c, tools_c, None),
                    user=render_history(rcd.trajectory.actions[:cut]),
                    meta=meta_c,
                )
            )

    stats.n_candidates_before_cap = len(candidates)
    stats.n_clean_candidates_before_cap = len(clean_candidates)

    # ------------------------------------------------------------------ #
    # Two-pool budget split, each capped for inter-task diversity        #
    # ------------------------------------------------------------------ #
    clean_ratio_clamped = max(0.0, min(1.0, float(clean_ratio)))
    clean_budget = min(
        int(round(max_prompts * clean_ratio_clamped)), len(clean_candidates)
    )
    attacked_budget = max(0, max_prompts - clean_budget)

    attacked_rows, n_attacked_dropped = _cap_with_task_diversity(
        candidates, attacked_budget, rng
    )
    # Hand unused attacked slots back to the benign pool so the trainer still
    # sees ``max_prompts`` rows whenever total supply allows.
    leftover = max_prompts - len(attacked_rows) - clean_budget
    if leftover > 0:
        clean_budget = min(clean_budget + leftover, len(clean_candidates))
    clean_rows, n_clean_dropped = _cap_with_task_diversity(
        clean_candidates, clean_budget, rng
    )

    result_rows = attacked_rows + clean_rows
    stats.n_capped_away = n_attacked_dropped + n_clean_dropped
    stats.n_clean_in_result = len(clean_rows)
    stats.n_unique_task_ids_in_result = len({row.meta.task_id for row in result_rows})
    return result_rows, stats


__all__ = [
    "GrpoPromptRow",
    "ExtractionStats",
    "extract_grpo_prompts",
]
