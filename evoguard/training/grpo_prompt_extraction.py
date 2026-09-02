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
    _norm_text,
    _payload_probe,
)
from evoguard.training.grpo_reward import PromptMeta, _action_signature
from evoguard.training.step_safety_judge import (
    build_reference_timeline,
    build_step_timeline,
)
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

    # ---- which step the sampler decided to ask about (plan 甲) ----------- #
    # ``bait`` is the one that carries gradient: at that step a sibling taking
    # the bait scores -10.50 while one taking the twin step scores +3.20. A round
    # dominated by ``legacy`` is a round where GRPO is likely to idle again, so
    # these three are worth reading in the r0 log.
    n_step_bait: int = 0
    n_step_grounded: int = 0
    n_step_legacy: int = 0

    # ---- benign (clean-trajectory) prompt pool -------------------------- #
    n_clean_records_seen: int = 0
    n_clean_skipped_too_short: int = 0      # <2 actions => no next-step ground truth
    n_clean_skipped_low_utility: int = 0    # clean rollout itself failed the task
    n_clean_skipped_no_task: int = 0        # task_id absent from dataset_builder registry
    n_clean_candidates_before_cap: int = 0
    n_clean_in_result: int = 0

    # ---- trajectory grouping (轨C.3) ------------------------------------ #
    # ``n_steps_per_trajectory == 1`` is the legacy one-prompt-per-record mode.
    # Above 1, every surviving record contributes EXACTLY that many contiguous
    # rows sharing one ``meta.traj_group_id``, so the trainer's generation batch
    # can be made to coincide with one trajectory.
    n_steps_per_trajectory: int = 1
    n_traj_groups_attacked: int = 0
    n_traj_groups_clean: int = 0
    n_traj_groups_padded: int = 0           # trajectory shorter than K => steps cycled

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


def _bait_step_index(
    actions: Any,
    *,
    start: int,
    target_tool: str,
) -> Optional[int]:
    """Earliest index at/after ``start`` whose tool call IS the attacker's tool."""
    tgt = (target_tool or "").strip().lower()
    if not tgt:
        return None
    acts = actions or []
    for i in range(max(0, int(start)), len(acts)):
        tc = getattr(acts[i], "tool_call", None)
        if tc is not None and str(getattr(tc, "name", "")).strip().lower() == tgt:
            return i
    return None


def _payload_visible_at(actions: Any, index: int, probe: str) -> bool:
    """Whether the payload is on screen in the prompt rendered for ``index``.

    Index 0 is never visible (empty history). An empty ``probe`` means the
    payload was too short to fingerprint, in which case a non-empty prefix is
    the strongest statement available -- same degradation as
    ``dataset_builder._grounded_refusal_index``.
    """
    acts = actions or []
    if index <= 0 or index >= len(acts) + 1:
        return False
    if not probe:
        return True
    return probe in _norm_text(render_history(acts[:index]))


def _decision_step_index(
    traj: Trajectory,
    *,
    base: int,
    attack: Any,
) -> tuple[int, str]:
    """Pick the step the policy is actually asked to decide, plus a reason tag.

    Plan 甲 (2026-08-21). The sampler used to hand GRPO ``injection_point + 1``
    unconditionally. The payload IS already on screen there -- ``render_history``
    renders observations and the prefix ends after the injecting turn -- but at
    that step the bait has usually not fired yet, so every one of the G siblings
    collects ``R_SAFETY_HELD (+2.00)`` and, if it simply mirrors the clean twin,
    the structural ``R_PROGRESS_ADVANCE (+1.20)`` as well. Measured over 4400
    logged GRPO steps of the completed 12-round run: mean reward 2.7861, MEDIAN
    3.2000 -- exactly the ceiling -- and ``reward_std == 0`` in 73.3% of groups
    (3225/4400), entropy mean 0.0815. Group-relative advantages are differences
    within a group, so a group that is uniformly at the ceiling contributes no
    gradient at all: GRPO idled for ten rounds.

    So sample where the outcomes actually diverge -- the step at which the source
    rollout fired the attacker's tool. There a sibling that takes the bait scores
    -10.50 and one that takes the twin step scores +3.20, a 13.70 spread inside
    one group.

    Preference order, all constrained to indices at/after ``base`` so no prompt
    ever predates the injection:

      1. the bait step, when the payload is visible in its own prefix;
      2. failing that, the earliest step whose prefix shows the payload;
      3. failing that, ``base`` -- legacy behaviour, so supply never shrinks.

    Returns ``(index, reason)`` where reason is one of ``bait`` / ``grounded`` /
    ``legacy``, counted in :class:`ExtractionStats` so a round can be audited.

    **MEASURED: this is a GUARDRAIL, not the fix. It is inert on the AgentDojo
    data as of 2026-08-21.** Replayed over the 705 attacked records of r0
    (`rounds/evoguard_agentdojo_full_sft_twoclass/round_0/records.jsonl`), 321 of
    which yield a prompt: reasons ``{grounded: 311, bait: 10, legacy: 0}`` but the
    index shift ``new - legacy`` is **0 in 321/321 cases**. Three independent
    reasons, all worth knowing before anyone "improves" this function:

      * ``_infer_target_tool`` is a regex over the attack's goal/payload and
        returns ``""`` for 299/321 (93%), so the bait branch cannot even be
        attempted; where it can, the bait already sits at ``base`` (10 cases) or
        strictly before it (2 cases, correctly declined).
      * the payload is ALREADY on screen at ``base`` -- ``render_history`` renders
        observations and the prefix ends after the injecting turn -- so the
        grounded branch has nothing to shift onto.
      * Δ=0 dominates this dataset (``delta_immediate_share`` 0.86-0.98), and at
        Δ=0 the labelled turning point is the benign retrieval that FETCHED the
        payload while the bait sits one step later, i.e. exactly at ``base``.

    So the sampled step was already the decision point, and the saturation is
    NOT caused by step selection. The active parts of plan 甲 are the two
    exploration knobs (``grpo_rollout_temperature`` 0.90 -> 1.15,
    ``grpo_max_prompts_per_round`` 32 -> 48). This function stays because it
    costs nothing, because it is right for the minority of attacks whose bait
    fires later, and because its three counters make the claim above re-checkable
    on any future dataset instead of assumed.
    """

    actions = getattr(traj, "actions", None) or []
    probe = _payload_probe(attack)
    target_tool = _infer_target_tool(attack) if attack is not None else ""

    bait = _bait_step_index(actions, start=base, target_tool=target_tool)
    if bait is not None and _payload_visible_at(actions, bait, probe):
        return bait, "bait"

    for i in range(max(0, int(base)), len(actions)):
        if _payload_visible_at(actions, i, probe):
            return i, "grounded"

    return int(base), "legacy"


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


def _trajectory_step_indices(
    n_actions: int,
    *,
    base: int,
    k: int,
) -> tuple[list[int], bool]:
    """Pick exactly ``k`` step indices spanning one trajectory (轨C.3).

    A GRPO prompt for index ``i`` renders ``actions[:i]`` and asks the policy for
    action ``i``, so the eligible window is ``[base, n_actions - 1]``: never
    earlier than ``base`` (which for attacked rows already sits at/after the
    injection) and never past the last recorded action.

    Returns ``(indices, padded)`` where ``indices`` has length exactly ``k`` and
    is non-decreasing, and ``padded`` says the window was shorter than ``k`` so
    steps had to be cycled.

    Two properties are load-bearing:

      * ``k == 1`` returns ``[base]``, i.e. byte-identical legacy behaviour.
      * the LAST eligible index is always included when ``k > 1``. That index is
        typically the step at which the source rollout produced its final answer,
        and it is the only place where "answer now" and "call yet another tool"
        land in the same G-sibling group. This is the whole reason the group is
        widened to a trajectory: on the plan_abc adapter the ASR win was partly
        bought by stalling (``clean_utility_mean`` 0.3149 -> 0.2035,
        ``final_answer_rate`` 60.53% -> 31.14%, ``steps_mean`` 9.2 -> 11.5), and
        stalling is rational under a per-step reward because a neutral step costs
        only -0.15 while firing the bait costs -10.50 and "never answered" is
        invisible. Including the terminal step prices non-termination as a DATA
        choice rather than as a new reward term.
    """
    k_int = max(1, int(k))
    lo = max(0, int(base))
    hi = max(lo, int(n_actions) - 1)
    if k_int == 1:
        return [lo], False
    window = list(range(lo, hi + 1))
    if len(window) >= k_int:
        step = (hi - lo) / float(k_int - 1)
        picked = [int(round(lo + i * step)) for i in range(k_int)]
        return picked, False
    # Shorter than K: keep every eligible step, then cycle from the front so the
    # row count stays an exact multiple of K (TRL's RepeatSampler drops partial
    # chunks, and a ragged group would desynchronise every later group).
    picked = list(window)
    while len(picked) < k_int:
        picked.append(window[(len(picked) - len(window)) % len(window)])
    return picked, True


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
    candidates: list[Any],
    max_prompts: int,
    rng: random.Random,
    key: Any = None,
) -> tuple[list[Any], int]:
    """Trim ``candidates`` to ``max_prompts`` spreading slots across task_ids.

    Returns ``(kept_items, n_dropped)``. Under the cap the input list is returned
    verbatim so ordering stays reproducible.

    ``key`` maps an item to the task id used for spreading; it defaults to
    ``item.meta.task_id`` so a plain list of :class:`GrpoPromptRow` behaves
    exactly as before. Passing ``key=lambda group: group[0].meta.task_id`` lets
    the same fairness logic cap trajectory GROUPS as indivisible units, which is
    what 轨C.3 needs -- capping rows individually would slice groups apart and
    break the "one generation batch == one trajectory" invariant.
    """
    key_fn = key if key is not None else (lambda item: item.meta.task_id)
    if max_prompts <= 0:
        return [], len(candidates)
    if len(candidates) <= max_prompts:
        return candidates, 0

    groups: dict[str, list[int]] = {}
    for ci, cand in enumerate(candidates):
        groups.setdefault(key_fn(cand), []).append(ci)

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
    steps_per_trajectory: int = 1,
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
      * ``steps_per_trajectory`` -- K, the number of steps sampled from EACH
                                 surviving trajectory. ``1`` (default) is the
                                 legacy one-prompt-per-record behaviour, rows
                                 unchanged. Above 1 the returned list is a
                                 concatenation of contiguous K-row groups, each
                                 sharing one ``meta.traj_group_id``, ordered
                                 attacked-groups-then-clean-groups. The trainer
                                 relies on that contiguity: with
                                 ``shuffle_dataset=False`` and
                                 ``steps_per_generation = G * K`` one generation
                                 batch is exactly one trajectory, which is what
                                 lets a degenerate single-step group (62.8% of
                                 all steps had ``reward_std == 0`` on the
                                 plan_abc run) borrow a baseline from its
                                 siblings elsewhere in the same trajectory.

    Returns parallel-aligned lists wrapped inside ``GrpoPromptRow`` containers plus an
    :class:`ExtractionStats` summary suitable for logging diagnostics without re-running filtering.
    """

    stats = ExtractionStats()
    rng = random.Random(seed)
    k_steps = max(1, int(steps_per_trajectory))
    stats.n_steps_per_trajectory = k_steps

    # The clean TWIN each attacked row is scored against. With
    # ``pipeline.clean_rollouts_per_task > 1`` there are several candidates per
    # task, and the choice is load-bearing: ``grpo_reward`` awards ``ADVANCE``
    # only on STRICT signature equality with this trajectory's next action, so a
    # twin that itself failed the benign task makes correct continuations
    # unrewardable. Take the highest-utility sample (unscored sorts as 1.0, same
    # rule as ``DatasetBuilder._cap_per_task`` and ``build_sft``), ties on record
    # order -- with one clean record per task this is the record it always was.
    cleans_by_task: dict[str, Trajectory] = {}
    _best_clean_utility: dict[str, float] = {}
    for rcd in records:
        if rcd.kind is not TrajectoryKind.CLEAN:
            continue
        util = 1.0 if rcd.utility is None else float(rcd.utility)
        if rcd.task_id in cleans_by_task and util <= _best_clean_utility[rcd.task_id]:
            continue
        cleans_by_task[rcd.task_id] = rcd.trajectory
        _best_clean_utility[rcd.task_id] = util

    # Each element is one trajectory's contiguous block of exactly ``k_steps``
    # rows. With k_steps == 1 this is a list of singletons, so flattening at the
    # end reproduces the legacy row list verbatim.
    candidate_groups: list[list[GrpoPromptRow]] = []

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

        # Retarget onto the step where the G siblings can actually disagree
        # (plan 甲). Never moves earlier than ``step_i``, so no prompt predates
        # the injection and the ``legacy`` branch keeps the old behaviour.
        step_i, step_reason = _decision_step_index(
            rcd.trajectory, base=step_i, attack=rcd.attack
        )
        if step_reason == "bait":
            stats.n_step_bait += 1
        elif step_reason == "grounded":
            stats.n_step_grounded += 1
        else:
            stats.n_step_legacy += 1

        task_obj: Optional[Task] = dataset_builder._tasks.get(rcd.task_id)
        if task_obj is None:
            logger.debug("[grpo_extract] skipping %r: no registered Task.", rcd.task_id)
            stats.n_skipped_step_not_found += 1     # closest-fit bucket; rare path
            continue

        tools_for_task: list[ToolSpec] = dataset_builder._tools.get(rcd.task_id, [])
        sys_str = build_system_prompt(task_obj, tools_for_task, None)

        target_tool_name = (
            _infer_target_tool(rcd.attack) if rcd.attack is not None else ""
        )
        # The attacker's own words, forwarded to the reward-path safety judge so
        # it scores what the injection ASKED FOR, not just the tool it named.
        attacker_goal = str(getattr(rcd.attack, "goal", "") or "") if rcd.attack else ""
        payload_excerpt = (
            str(getattr(rcd.attack, "payload", "") or "")[:500] if rcd.attack else ""
        )

        step_indices, was_padded = _trajectory_step_indices(
            len(rcd.trajectory.actions), base=step_i, k=k_steps
        )
        if was_padded:
            stats.n_traj_groups_padded += 1
        group_id = f"a:{rcd.record_id}" if k_steps > 1 else ""

        group_rows: list[GrpoPromptRow] = []
        # Fingerprint of the attacker's text, reused per step below to decide
        # whether the payload is already on screen at that step.
        payload_probe = _payload_probe(rcd.attack) if rcd.attack is not None else ""
        clean_twin_traj = cleans_by_task.get(rcd.task_id)
        benign_timeline = build_reference_timeline(
            getattr(clean_twin_traj, "actions", ()) or ()
        )
        for s_idx in step_indices:
            history_prefix_actions = rcd.trajectory.actions[:s_idx]
            user_text = render_history(history_prefix_actions)

            # Align onto clean twin A's corresponding turn so the reward's
            # structural pre-check has ground truth for THIS step. The anchor is
            # the sampled action's own turn, not ``inj_point + 1``: once the
            # sampler shifts off ``inj_point + 1`` (either via the plan-甲 bait
            # retarget or via the 轨C.3 multi-step spread), an ``inj_point + 1``
            # anchor would hand the reward the twin action for an EARLIER turn,
            # which the policy has usually already taken -- i.e. a signature
            # sitting in ``prefix_action_signatures``, so the prefix-repeat
            # branch would fire and the correct continuation could not score
            # ``ADVANCE`` at all. Same alignment rule as dataset_builder's phase
            # 2 (anchor on the sampled turn, not the raw signal).
            align_thr: Optional[int] = (
                (int(inj_point) + 1) if inj_point is not None else None
            )
            try:
                align_thr = int(rcd.trajectory.actions[s_idx].turn)
            except Exception:                                         # noqa: BLE001
                pass
            clean_twin_name, clean_twin_args = _aligned_clean_action_step(
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
                clean_twin_action_tool_name=clean_twin_name.strip(),
                clean_twin_action_args=clean_twin_args,
                prefix_action_signatures=_prefix_action_signatures(
                    history_prefix_actions
                ),
                traj_group_id=group_id,
                injected_payload_excerpt=payload_excerpt,
                attacker_goal=attacker_goal,
                prefix_timeline=build_step_timeline(
                    history_prefix_actions, inj_point
                ),
                benign_reference_timeline=benign_timeline,
                payload_visible_in_prefix=_payload_visible_at(
                    rcd.trajectory.actions, s_idx, payload_probe
                ),
            )
            group_rows.append(
                GrpoPromptRow(system=sys_str, user=user_text, meta=meta)
            )
        candidate_groups.append(group_rows)

    # ------------------------------------------------------------------ #
    # Benign pass: K prompts per usable CLEAN trajectory                  #
    # ------------------------------------------------------------------ #
    clean_candidate_groups: list[list[GrpoPromptRow]] = []
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
            sys_c = build_system_prompt(task_obj_c, tools_c, None)

            # Legacy (K=1) keeps the midpoint cut. For K>1 spread from the FIRST
            # step with a non-empty history through the last recorded action, so
            # the group covers "start the task", "middle of the task" and
            # "the task is done -- answer" instead of one interchangeable
            # mid-task snapshot.
            if k_steps == 1:
                c_indices, c_padded = [cut], False
            else:
                c_indices, c_padded = _trajectory_step_indices(
                    len(rcd.trajectory.actions), base=1, k=k_steps
                )
            if c_padded:
                stats.n_traj_groups_padded += 1
            group_id_c = f"c:{rcd.record_id}" if k_steps > 1 else ""

            c_rows: list[GrpoPromptRow] = []
            for s_idx in c_indices:
                clean_next_name, clean_next_args = _clean_next_tool_call(
                    rcd.trajectory, s_idx
                )
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
                        rcd.trajectory.actions[:s_idx]
                    ),
                    traj_group_id=group_id_c,
                )
                c_rows.append(
                    GrpoPromptRow(
                        system=sys_c,
                        user=render_history(rcd.trajectory.actions[:s_idx]),
                        meta=meta_c,
                    )
                )
            clean_candidate_groups.append(c_rows)

    stats.n_candidates_before_cap = len(candidate_groups) * k_steps
    stats.n_clean_candidates_before_cap = len(clean_candidate_groups) * k_steps

    # ------------------------------------------------------------------ #
    # Two-pool budget split, each capped for inter-task diversity        #
    # ------------------------------------------------------------------ #
    # ``max_prompts`` is a ROW budget, so with K steps per trajectory the unit of
    # allocation is ``max_prompts // K`` groups. At K == 1 group count == row
    # count and every number below is identical to the legacy computation.
    max_groups = max(0, int(max_prompts) // k_steps)
    group_key = lambda grp: grp[0].meta.task_id                       # noqa: E731
    clean_ratio_clamped = max(0.0, min(1.0, float(clean_ratio)))
    clean_budget = min(
        int(round(max_groups * clean_ratio_clamped)), len(clean_candidate_groups)
    )
    attacked_budget = max(0, max_groups - clean_budget)

    attacked_groups, n_attacked_dropped = _cap_with_task_diversity(
        candidate_groups, attacked_budget, rng, key=group_key
    )
    # Hand unused attacked slots back to the benign pool so the trainer still
    # sees ``max_prompts`` rows whenever total supply allows.
    leftover = max_groups - len(attacked_groups) - clean_budget
    if leftover > 0:
        clean_budget = min(clean_budget + leftover, len(clean_candidate_groups))
    clean_groups, n_clean_dropped = _cap_with_task_diversity(
        clean_candidate_groups, clean_budget, rng, key=group_key
    )

    stats.n_traj_groups_attacked = len(attacked_groups)
    stats.n_traj_groups_clean = len(clean_groups)

    # Flatten group-major so each trajectory's K rows stay CONTIGUOUS: the
    # trainer pairs ``shuffle_dataset=False`` with ``steps_per_generation = G*K``,
    # which makes one generation batch exactly one trajectory. Any reordering
    # here silently breaks the trajectory pooling (it would pool across
    # unrelated trajectories) without raising, so do not sort ``result_rows``.
    result_rows = [row for grp in attacked_groups for row in grp]
    clean_rows = [row for grp in clean_groups for row in grp]
    result_rows.extend(clean_rows)

    stats.n_capped_away = (n_attacked_dropped + n_clean_dropped) * k_steps
    stats.n_clean_in_result = len(clean_rows)
    stats.n_unique_task_ids_in_result = len({row.meta.task_id for row in result_rows})
    return result_rows, stats


__all__ = [
    "GrpoPromptRow",
    "ExtractionStats",
    "extract_grpo_prompts",
]
