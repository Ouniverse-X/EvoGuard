"""Δ-guided Monte-Carlo Tree Search attacker backend.

Alternative to :class:`evoguard.attacks.genetic.GeneticAttacker`. Replaces flat
population-based optimization with structured tree-of-rollouts exploration whose
selection rule is biased toward subtrees historically producing high-latency
(``τ_turn − τ_inj`` large) successful injections -- i.e. *latent* attacks rather
than immediate-trigger ones.

Public interface mirrors ``GeneticAttacker`` exactly so callers swap backends
purely through ``AttackerConfig.search_method``::

    build_attacker(task, tools, gen, cfg,
                   defense_max_turns=K)   # picks GA vs MCTS internally
"""

from __future__ import annotations

import hashlib
import math
import random
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Optional, Sequence

from evoguard.attacks.base import AttackGenerator
from evoguard.attacks.genetic import EvaluatedAttack
from evoguard.config import AttackerConfig
from evoguard.core.types import AttackSpec, Task, ToolSpec
from evoguard.utils.logging import get_logger

logger = get_logger("attacks.mct_searcher")

_NodeLevel = Literal["root", "L1_turn", "L2_method", "L3_payload"]

# Cap how many sibling nodes a single ``_expand_frontier`` invocation may
# register from one generator call. Prevents pathological cases where the
# LLM returns dozens of similar method labels in one batch and inflates a
# single cell of the tree before any of its siblings get explored.
_MAX_BATCH_SPAWN_PER_CALL = 6


# --------------------------------------------------------------------------- #
# Node                                                                        #
# --------------------------------------------------------------------------- #
@dataclass
class _TreeNode:
    """A single node in the per-task MCTS tree.
        root                 ── task-level entry point (one per Task)
        ├── L1_turn          ── target_turn ∈ {0,…,T_ceiling−1}
        │   ├── L2_method     ── coarse technique label produced by seed()/mutate()
        │   │   └── L3_payload ── concrete payload-text signature hash
        ...
        └── ...
    """

    node_id: str
    parent_id: Optional[str]
    level: _NodeLevel
    discriminator: dict[str, Any] = field(default_factory=dict)

    n_visits: int = 0
    n_success: int = 0                       # count of B-class outcomes observed here
    sum_delta_on_success: float = 0.0         # Σ Δ_norm restricted to successes only
    max_delta_observed: float = 0.0           # running peak for exploitation bound

    # Failure partial credit machinery. Sliding window length governed by config
    # ``mcts_tau_window_size``; rolling sum kept separately for O(1) reads inside
    # the UCB score function.
    last_failure_taus: deque = field(default_factory=lambda: deque(maxlen=8))
    sum_partial_credit: float = 0.0            # accumulated ε·tau_frac contributions

    cached_payload_text: Optional[str] = None  # set on L3 leaves post-materialization
    children_ids: list[str] = field(default_factory=list)


def _short_hash(text: str, prefix_len: int = 10) -> str:
    """Stable short identifier derived deterministically from arbitrary text."""

    h = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
    return h[:prefix_len]


# --------------------------------------------------------------------------- #
# Attacker                                                                    #
# --------------------------------------------------------------------------- #
class DeltaGuidedMCTSAttacker:
    """MCTS-based replacement for :class:`GeneticAttacker`.
    """

    def __init__(
        self,
        task: Task,
        tools: Sequence[ToolSpec],
        generator: AttackGenerator,
        config: AttackerConfig,
        *,
        rng: Optional[random.Random] = None,
        defense_max_turns: Optional[int] = None,
    ) -> None:
        self.task = task
        self.tools = list(tools)
        self.generator = generator
        self.config = config
        self.rng = rng or random.Random(config.random_seed)

        ctrl_cap = int(defense_max_turns) if defense_max_turns else None
        if ctrl_cap is not None and ctrl_cap > 0:
            self._inject_turn_ceiling = max(1, ctrl_cap)
        else:
            self._inject_turn_ceiling = max(1, len(self.tools))

        # ---- Cache MCTS config scalars (constant per run, avoids repeated
        #      getattr lookups on the UCB hot path). ---------------------------- #
        self._ucb_c = float(config.mcts_ucb_c)
        self._lambda_delta = float(config.mcts_lambda_delta)
        self._failure_eps = float(config.mcts_failure_credit_eps)
        self._tau_window = int(config.mcts_tau_window_size)

        # ---- Tree state -------------------------------------------------- #
        self._root = _TreeNode(
            node_id="root",
            parent_id=None,
            level="root",
            discriminator={},
        )
        self._nodes_by_id: dict[str, _TreeNode] = {"root": self._root}
        self.generation = 0

        # Buffer coupling current_population <-> next evolve(): each element is
        # parallel pair (spec, path_of_node_ids_used_during_materialization).
        # Path info lets backprop know exactly which ancestors deserve updates
        # when evaluated results come back, even though specs themselves don't
        # carry tree coordinates.
        self._buffer_specs: list[AttackSpec] = []
        self._buffer_paths: list[list[str]] = []

        # Pre-populate deterministic L1 enumeration so subsequent selects never
        # hit an empty frontier at depth-1 needing expensive generation calls.
        self._ensure_L1_children()

    @property
    def injectable_turn_ceiling(self) -> int:
        """Exclusive upper bound for ``AttackSpec.target_turn``."""
        return self._inject_turn_ceiling

    def sanitize_spec(self, spec: AttackSpec) -> AttackSpec:
        """Mirror :py:meth:`GeneticAttacker.sanitize_spec` semantics."""
        ub = self._inject_turn_ceiling - 1
        original = int(spec.target_turn)
        clamped = max(0, min(original, ub))
        if clamped != original:
            logger.warning(
                "Task %s: clamped out-of-bound target_turn %d -> %d "
                "(gen=%d origin=%s)",
                self.task.task_id, original, clamped,
                spec.generation, spec.origin,
            )
            return replace(spec, target_turn=clamped)
        return spec

    # ------------------------------------------------------------------ #
    # Public API                                                        #
    # ------------------------------------------------------------------ #
    def current_population(self) -> list[AttackSpec]:
        """Return the batch scheduled for rollout evaluation this round.

        First invocation seeds lazily; subsequent invocations re-use whatever was
        installed by the previous :meth:`evolve` (or the constructor seeding step
        if nothing evolved yet).
        """
        if not self._buffer_specs:
            self._regenerate_population_buffer(self.generation)
        return list(self._buffer_specs)

    def evolve(self, evaluated: list[EvaluatedAttack]) -> list[AttackSpec]:
        """Consume last-batch evaluations, update tree state, schedule next batch."""
        if not evaluated:
            # Nothing came back; clear buffers defensively and emit fresh batch.
            self._buffer_specs.clear()
            self._buffer_paths.clear()
            self.generation += 1
            self._regenerate_population_buffer(self.generation)
            return list(self._buffer_specs)

        if len(evaluated) != len(self._buffer_paths):
            raise ValueError(
                f"evolve() received {len(evaluated)} evaluations but {len(self._buffer_paths)} "
                f"paths are pending; did you mutate the buffer externally?"
            )

        for eva_record, path_ids in zip(evaluated, self._buffer_paths):
            outcome_meta = eva_record.metadata or {}
            succeeded = bool(eva_record.success)
            # Prefer explicit signal fields if upstream populated them,
            # otherwise fall back to fitness (= Δ_norm when success per plan.md).
            delta_norm = float(eva_record.fitness or 0.0)
            tau_caught_raw = (
                outcome_meta.get("turning_point")
                or outcome_meta.get("tau_turn")
                or outcome_meta.get("behavior_turn_point")
                or 0
            )
            try:
                tau_caught = max(0, int(tau_caught_raw))
            except Exception:
                tau_caught = 0
            self._backprop_along_path(
                path_ids=path_ids,
                succeeded=succeeded,
                delta_norm=delta_norm,
                tau_caught=tau_caught,
            )

        self.generation += 1
        self._regenerate_population_buffer(self.generation)
        return list(self._buffer_specs)

    # ------------------------------------------------------------------ #
    # Tree construction helpers                                         #
    # ------------------------------------------------------------------ #
    def _register_node(self, node: _TreeNode) -> _TreeNode:
        nid = node.node_id
        if nid != "root" and nid in self._nodes_by_id:
            # Idempotent registration allowed for caller convenience.
            return self._nodes_by_id[nid]
        self._nodes_by_id[nid] = node
        if node.parent_id is not None:
            parent = self._nodes_by_id.get(node.parent_id)
            if parent is not None and nid not in parent.children_ids:
                parent.children_ids.append(nid)
        return node

    def _child_with_discriminator(
        self,
        parent: _TreeNode,
        *,
        disc_key: str,
        disc_value: Any,
    ) -> Optional[_TreeNode]:
        """Return existing direct child carrying ``(disc_key == disc_value)``
        in its discriminator map, else ``None``."""
        for cid in parent.children_ids:
            cnode = self._nodes_by_id.get(cid)
            if cnode is None:
                continue
            cv = cnode.discriminator.get(disc_key)
            if cv == disc_value:
                return cnode
        return None

    def _ensure_L1_children(self) -> None:
        """Pre-create every reachable target_turn slot up-front.

        Deterministic expansion requiring NO LLM calls; saves budget for deeper
        layers where generative creativity matters most.
        """
        ceiling = self._inject_turn_ceiling
        for t in range(ceiling):
            existing = self._child_with_discriminator(
                self._root, disc_key="turn", disc_value=t,
            )
            if existing is not None:
                continue
            node_id = f"L1_t{t}"
            child = _TreeNode(
                node_id=node_id,
                parent_id="root",
                level="L1_turn",
                discriminator={"turn": t},
            )
            self._register_node(child)

    # ------------------------------------------------------------------ #
    # Selection                                                         #
    # ------------------------------------------------------------------ #
    def _ucb_score(self, parent_visits: int, child: _TreeNode) -> float:
        """Composite selection criterion.

        ``score = exploit_term + c·explore_term + λ·δ_potential``

        The staleness-penalty term once planned in ``docs/mcts_attacker_design.md``
        has been removed from this implementation. Its underlying Phase-C machinery
        (LoRA-fingerprint hashing, exponential decay sweep, resurrection probes)
        was never wired up, so carrying an always-zero fourth summand served no
        purpose but to mislead readers into thinking it influenced selection.
        Should that subsystem ever land, restore the subtraction here and add the
        required bookkeeping fields to :class:`_TreeNode`.
        """
        eps_visit = 1e-9
        if child.n_visits <= 0:
            exploit_mean_delta = 0.01               # tiny prior avoids div-by-zero lockout
            variance_estimate = 0.25                  # broad prior on unseen regions
        else:
            denom_succ = float(max(1, child.n_success))
            mean_on_succ = float(child.sum_delta_on_success) / denom_succ
            success_rate = float(child.n_success) / float(child.n_visits)
            exploit_mean_delta = success_rate * mean_on_succ
            partial_avg = (
                float(child.sum_partial_credit) /
                float(child.n_visits + eps_visit)
            )
            exploit_mean_delta += self._failure_eps * partial_avg
            # Variance proxy computed from extreme spread when sample size small;
            # collapses to 0 once many observations pin distribution tightly --
            # desired property letting other terms dominate mature cells.
            span = max(0.0, child.max_delta_observed - mean_on_succ)
            variance_estimate = min(span, math.sqrt(float(child.max_delta_observed)))

        ln_parent = math.log(max(1.0, float(parent_visits)))

        explore_term = self._ucb_c * math.sqrt(ln_parent / (1.0 + float(child.n_visits)))
        delta_potential = self._lambda_delta * math.sqrt(variance_estimate)
        return exploit_mean_delta + explore_term + delta_potential

    def _traverse_to_frontier(self) -> tuple[list[str], _TreeNode]:
        """Walk from root picking best-scoring child at each interior node
        until arriving at an L3 payload leaf or a frontier with no children yet
        (the natural expansion point)."""
        path: list[str] = ["root"]
        cur = self._root
        guard_iters = 0
        while True:
            guard_iters += 1
            if guard_iters > 10000:
                break                          # circuit-breaker safeguard
            if not cur.children_ids:
                break                           # truly empty subtree → must be expanded outside
            visits_parent = cur.n_visits
            scored_children: list[tuple[float, _TreeNode]] = []
            for cid in cur.children_ids:
                cn = self._nodes_by_id.get(cid)
                if cn is None:
                    continue
                s = self._ucb_score(visits_parent, cn)
                scored_children.append((s, cn))
            if not scored_children:
                break
            # Random tie-breaking prevents degenerate repeat-picking when many
            # children share identical zero-statistics during cold-start phases.
            self.rng.shuffle(scored_children)
            top_s, top_n = max(scored_children, key=lambda x: x[0])
            path.append(top_n.node_id)
            cur = top_n
            if cur.level == "L3_payload":
                break
            # Stop descending when we hit a leaf with no children — that is the
                # natural expansion point. No visit-count gating: any interior
                # node may be expanded on every call, so cold-start latency and
                # generator output waste are both minimised.
            if not cur.children_ids:
                break                           # leaf with no children → expand now
        return path, cur

    # ------------------------------------------------------------------ #
    # Expansion                                                         #
    # ------------------------------------------------------------------ #
    def _expand_frontier(self, node: _TreeNode) -> Optional[_TreeNode]:
        """Create one or more new children beneath ``node`` and return the last
        freshly-registered child (or ``None`` when nothing new could be produced).

        Eligibility rules after v0.x simplification:
          • Only L1/L2 levels are expandable — L3 leaves never spawn further.
          • No visit-count gating. Any interior node may add siblings on every
            call, so cold-start latency is minimised.
          • Generator output is harvested in BATCH rather than first-match-return:
            all unique candidates within ``_MAX_BATCH_SPAWN_PER_CALL`` get registered
            as sibling children in one shot, eliminating wasted API calls when the
            generator returns several novel labels at once (the previous behaviour
            discarded all but the first unique match).
          • Returns the LAST newly-created child as the leaf targeted by THIS
            simulation step; its older siblings stay attached waiting for future
            select passes to discover them naturally via UCB scoring.

        The L2 → L3 branch remains single-shot because :meth:`AttackGenerator.mutate`
        is a single-input/single-output interface by contract; batching there would
        require an upstream API change outside the scope of this refactor.
        """
        if node.level not in {"L1_turn", "L2_method"}:
            return None

        if node.level == "L1_turn":
            t_val = int(node.discriminator.get("turn", 0))
            seeded = self.generator.seed(
                self.task, self.tools,
                n=_MAX_BATCH_SPAWN_PER_CALL * 2,           # request a bit more than cap so dedup losses don't starve us
                max_turns=t_val + 1, generation=self.generation,
            )
            newly_registered: list[_TreeNode] = []
            for cand_spec in seeded[: _MAX_BATCH_SPAWN_PER_CALL * 2]:
                lbl = cand_spec.method or ""
                if not lbl or len(newly_registered) >= _MAX_BATCH_SPAWN_PER_CALL:
                    continue                              # blank label OR per-call cap reached -- stop adding but keep iterating for completeness
                ex = self._child_with_discriminator(node, disc_key="method", disc_value=lbl)
                if ex is not None:
                    continue                               # dup against existing sibling -- skip WITHOUT aborting the loop
                cid = f"L2_{lbl}_{_short_hash(lbl)}"
                ch = _TreeNode(
                    node_id=cid, parent_id=node.node_id, level="L2_method",
                    discriminator={"method": lbl},
                )
                self._register_node(ch)
                newly_registered.append(ch)
            # Return the most-recently created child as the focus of this simulation;
            # any earlier siblings stay attached waiting passively for future selects.
            return newly_registered[-1] if newly_registered else None

        if node.level == "L2_method":
            t_val = self._lookup_ancestor_turn_value(node.node_id)
            placeholder = _construct_seed_from_cell(t_val, node.discriminator.get("method"), self.task)
            mutated = self.generator.mutate(self.task, self.tools, placeholder, generation=self.generation)
            sig_hash = _short_hash(mutated.payload or "")
            ex = self._child_with_discriminator(node, disc_key="signature", disc_value=sig_hash)
            if ex is not None:
                return ex                      # silently reused if collision occurs
            cid = f"L3_{sig_hash}"
            ch = _TreeNode(
                node_id=cid, parent_id=node.node_id, level="L3_payload",
                discriminator={"signature": sig_hash},
                cached_payload_text=mutated.payload or "",
            )
            self._register_node(ch)
            return ch
        return None

    def _descend_into_leaf_under(self, node: _TreeNode) -> tuple[list[str], _TreeNode]:
        """Pick some descendant L3 leaf beneath ``node`` (or stay put if already
        at L3), returning both path ids and chosen leaf. Used when revisiting
        known territory rather than spawning fresh branches."""
        path: list[str] = [node.node_id]
        cur = node
        if cur.children_ids:
            # walk greedily until L3 reached
            while cur.level != "L3_payload" and cur.children_ids:
                visits_p = cur.n_visits
                kids = [kn for kn in (self._nodes_by_id.get(c) for c in cur.children_ids) if kn is not None]
                if not kids:
                    break
                self.rng.shuffle(kids)
                pick = max(kids, key=lambda k: self._ucb_score(visits_p, k))
                path.append(pick.node_id)
                cur = pick
            # if landed at non-L3 due to dead-end, attempt single-shot expansion
            if cur.level != "L3_payload":
                ext = self._expand_frontier(cur)
                if ext is not None:
                    path.append(ext.node_id)
                    cur = ext
        return path, cur

    # ------------------------------------------------------------------ #
    # Materialization                                                   #
    # ------------------------------------------------------------------ #
    def _materialize_attack_from_leaf(self, path: list[str], leaf: _TreeNode) -> AttackSpec:
        """Translate a leaf-node choice into a runnable :class:`AttackSpec`."""
        parts: dict[str, Any] = {}
        for pid in path[1:]:
            pn = self._nodes_by_id.get(pid)
            if pn is None:
                continue
            d = pn.discriminator
            if "turn" in d:
                parts["target_turn"] = int(d["turn"])
            elif "method" in d:
                parts["method"] = d["method"]
            elif "signature" in d:
                parts["payload"] = leaf.cached_payload_text or ""

        tgt_turn: int = parts.get("target_turn", 0)
        method: str = str(parts.get("method") or "")
        payload: str = str(parts.get("payload") or "")

        # If payload missing (freshly created L2 without L3 yet?), synthesize
        # placeholder via generator.mutate so controller/judge receive valid strings.
        if not payload:
            tmp_spec = _construct_seed_from_cell(tgt_turn, method, self.task, payload="")
            mut = self.generator.mutate(self.task, self.tools, tmp_spec, generation=self.generation)
            payload = mut.payload or ""
            leaf.cached_payload_text = payload

        spec = AttackSpec(
            attack_id=AttackSpec.new_id(),
            task_id=self.task.task_id,
            target_turn=tgt_turn,
            injection_channel="",              # filled in by generator pipeline elsewhere
            method=method,
            payload=payload,
            goal=self.task.instruction[:200],  # mirror heuristic commonly seen in records
            generation=self.generation,
            parents=[leaf.node_id],
            origin=f"mcts_{leaf.level}",
            metadata={"tree_path": list(path), "leaf_id": leaf.node_id},
        )
        return self.sanitize_spec(spec)

    # ------------------------------------------------------------------ #
    # Ancestor lookup                                                   #
    # ------------------------------------------------------------------ #
    def _lookup_ancestor_turn_value(self, start_node_id: str) -> int:
        """Walk parent_id chain upward through :attr:`_nodes_by_id` returning
        the integer ``turn`` discriminator carried on the nearest ancestor whose
        level == ``L1_turn``. Returns 0 when none found (defensive default)."""
        cur_id = start_node_id
        guard = 0
        while cur_id and guard < 16:
            guard += 1
            node = self._nodes_by_id.get(cur_id)
            if node is None:
                break
            if node.level == "L1_turn":
                try:
                    return int(node.discriminator.get("turn", 0))
                except Exception:
                    return 0
            cur_id = node.parent_id or ""
        return 0

    # ------------------------------------------------------------------ #
    # Backprop                                                          #
    # ------------------------------------------------------------------ #
    def _backprop_along_path(self, *,
                             path_ids: list[str],
                             succeeded: bool,
                             delta_norm: float,
                             tau_caught: int):
        """Update visit/success/delta/partial-credit counters on every ancestor.

        Failures add fractional credit ``ε·mean(last_failure_taus)/T_ceiling``
        accumulated into :attr:`_TreeNode.sum_partial_credit` -- this quantity
        feeds back into :meth:`_ucb_score` indirectly nudging selections toward
        subtrees where late-caught failures hint at latent potential.
        """
        T_cap = float(self._inject_turn_ceiling) or 1.0

        for nid in path_ids:
            nd = self._nodes_by_id.get(nid)
            if nd is None:
                continue
            nd.n_visits += 1
            if succeeded:
                dn = float(delta_norm or 0.0)
                nd.n_success += 1
                nd.sum_delta_on_success += dn
                if dn > nd.max_delta_observed:
                    nd.max_delta_observed = dn
            else:
                tc = max(0, int(tau_caught or 0))
                nd.last_failure_taus.append(tc)
                frac_hidden = min(1.0, float(tc) / T_cap)
                nd.sum_partial_credit += (self._failure_eps * frac_hidden)

    # ------------------------------------------------------------------ #
    # Regenerate population                                             #
    # ------------------------------------------------------------------ #
    def _regenerate_population_buffer(self, generation_label: int) -> None:
        """Run N iterations of traverse-and-maybe-expand, materializing one
        candidate AttackSpec per iteration, storing paired paths for upcoming
        evolve()-driven backpropagation.

        v0.x: the previous ``revisit_fraction``-driven split between fresh
        expansion and known-leaf resampling has been removed. Every iteration
        now follows the single ``traverse -> expand -> materialize`` pipeline;
        UCB's explore term alone decides whether mature leaves get revisited
        or new frontiers get opened, eliminating the artificial 30% budget
        reservation that starved cold-start tree growth.
        """
        self.generation = generation_label
        N_target = int(getattr(self.config, "population_size", 50))

        new_specs: list[AttackSpec] = []
        new_paths: list[list[str]] = []

        attempts_guard = N_target * 8
        tries_in_expand_mode = 0
        tries_total = 0
        while len(new_specs) < N_target and attempts_guard > 0:
            attempts_guard -= 1
            tries_total += 1

            path, frontier = self._traverse_to_frontier()
            extended = self._expand_frontier(frontier)
            if extended is not None:
                path = path + [extended.node_id]
                leaf = extended
                tries_in_expand_mode += 1
            else:
                leaf = frontier
                if leaf.level != "L3_payload" and not leaf.children_ids:
                    # Empty frontier with no extension possible right now:
                    # force descend-if-any-children path fallback so we still
                    # emit a valid AttackSpec this iteration.
                    path2, leaf2 = self._descend_into_leaf_under(self._root)
                    path, leaf = path2, leaf2

            spec = self._materialize_attack_from_leaf(path, leaf)
            new_specs.append(spec)
            new_paths.append(list(path))

        if len(new_specs) < N_target:
            logger.warning(
                "Task %s MCTS regen round=%d: could produce only %d/%d candidates"
                " (%d total iterations tried)",
                self.task.task_id, generation_label, len(new_specs), N_target, tries_total,
            )

        self._buffer_specs = new_specs
        self._buffer_paths = new_paths

        logger.info(
            "Task %s MCTS gen %d->%d prepared %d candidates "
            "(expansion_attempts=%d, total_tries=%d, tree_nodes=%d)",
            self.task.task_id, generation_label - 1, generation_label,
            len(new_specs), tries_in_expand_mode, tries_total, len(self._nodes_by_id),
        )


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _construct_seed_from_cell(turn: int, method: str, task: Task, *,
                              payload: str = "") -> AttackSpec:
    """Build a minimal valid AttackSpec representing the cell identified by
    (turn, method). Used solely as scaffolding around generator.mutate() calls
    expecting an individual argument; the resulting mutated payload replaces
    whatever stub content lives here."""
    return AttackSpec(
        attack_id=AttackSpec.new_id(),
        task_id=task.task_id,
        target_turn=int(turn),
        injection_channel="",
        method=str(method or ""),
        payload=str(payload or ""),
        goal=getattr(task, "instruction", ""),
        generation=-1,                            # sentinel marks scaffold-not-real-rollout-spec
        parents=[],
        origin="mcts_scaffold",
        metadata={},
    )


__all__ = [
    "DeltaGuidedMCTSAttacker",
]
