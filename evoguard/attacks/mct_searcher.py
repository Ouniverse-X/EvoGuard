"""Δ-guided Monte-Carlo Tree Search attacker backend.
    build_attacker(task, tools, gen, cfg,
                   defense_max_turns=K)   # picks GA vs MCTS internally
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
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
# register from one generator call. 
_MAX_BATCH_SPAWN_PER_CALL = 6


# --------------------------------------------------------------------------- #
# Prewarm skeleton loader (warm-start hand-authored JSON templates)            #
# --------------------------------------------------------------------------- #
_PREWARM_CACHE: Optional[dict[str, list[dict]]] = None


def _load_prewarm_skeletons_all() -> dict[str, list[dict]]:
    global _PREWARM_CACHE
    if _PREWARM_CACHE is not None:
        return _PREWARM_CACHE

    here = os.path.dirname(os.path.abspath(__file__))
    candidates = []
    # Explicit opt-in override. Highest priority so an experiment can name its
    # own seed artifact (e.g. EVOGUARD_PREWARM_SEEDS_DIR=data/seeds to use the
    # fixed cross-dataset universal_seeds.json) without mutating the dataset dir
    # or changing the default for every other caller.
    override = (os.environ.get("EVOGUARD_PREWARM_SEEDS_DIR") or "").strip()
    if override:
        candidates.append(override)
        candidates.append(os.path.join(here, "..", "..", override))
    candidates += [
        os.path.join(os.getcwd(), "data", "toolsafe", "agentdojo-tragj"),
        os.path.join(here, "..", "..", "data", "toolsafe", "agentdojo-tragj"),
    ]
    cache: dict[str, list[dict]] = {}
    seen_dirs: set[str] = set()
    for cdir in candidates:
        try:
            rdir = os.path.realpath(cdir)
        except OSError:
            continue
        if not os.path.isdir(rdir) or rdir in seen_dirs:
            continue
        seen_dirs.add(rdir)
        try:
            entries = sorted(os.listdir(rdir))
        except OSError as exc:
            logger.warning("[prewarm] cannot list %s (%s); skipping.", rdir, exc)
            continue
        n_loaded_for_this_dir = 0
        for fname in entries:
            m = re.match(r"^(?P<suite>[a-z]+)_seeds\.json$", fname)
            if not m:
                continue
            suite_key = m.group("suite")
            fpath = os.path.join(rdir, fname)
            try:
                with open(fpath, encoding="utf-8") as fh:
                    raw = json.load(fh)
            except (OSError, ValueError) as exc:
                logger.warning("[prewarm] failed reading %s: %s", fpath, exc)
                continue
            entry_list = []
            if isinstance(raw, dict):
                seeds_field = raw.get("seeds")
                if isinstance(seeds_field, list):
                    for ent in seeds_field:
                        if isinstance(ent, dict):
                            entry_list.append(ent)
            elif isinstance(raw, list):  # tolerate bare-list format too
                entry_list.extend(e for e in raw if isinstance(e, dict))
            cache[suite_key] = entry_list
            n_loaded_for_this_dir += len(entry_list)
        if cache:
            if cache.get("universal"):
                cache = {"universal": cache["universal"]}
            logger.info(
                "[prewarm] loaded skeletons from %s across %d suites "
                "(total=%d entries).",
                rdir, len(cache), sum(len(v) for v in cache.values()),
            )
            break

    _PREWARM_CACHE = cache  
    return _PREWARM_CACHE


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
    n_success: int = 0                       # count of B-class outcomes observed 
    sum_delta_on_success: float = 0.0         # Σ Δ_norm restricted to successes only
    max_delta_observed: float = 0.0           # running peak for exploitation bound

    # Failure partial credit machinery. Sliding window length governed by config
    # ``mcts_tau_window_size``; 
    last_failure_taus: deque = field(default_factory=lambda: deque(maxlen=8))
    # 走过它的失败的case给他的贡献（8的滑窗）
    sum_partial_credit: float = 0.0            

    cached_payload_text: Optional[str] = None  # set on L3 leaves post-materialization
    children_ids: list[str] = field(default_factory=list)


def _short_hash(text: str, prefix_len: int = 10) -> str:
    """Stable short identifier derived deterministically from arbitrary text."""

    h = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
    return h[:prefix_len]


class DeltaGuidedMCTSAttacker:
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

        self._ucb_c = float(config.mcts_ucb_c)#公式探索系数
        # delta方差项系数
        self._lambda_delta = float(config.mcts_lambda_delta)
        # 失败样本的部分积分系数 ε。当一次攻击失败但被防御方很晚才检测到（tau_caught 大）时，仍给该路径一定正向信用 ε·(tau/T_cap)
        self._failure_eps = float(config.mcts_failure_credit_eps)
        self._tau_window = int(config.mcts_tau_window_size)

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
        # when evaluated results come back
        self._buffer_specs: list[AttackSpec] = []
        self._buffer_paths: list[list[str]] = []

        self._ensure_L1_children()
        self._inject_prewarm_skeletons()

    @property
    def injectable_turn_ceiling(self) -> int:
        """返回攻击注入目标轮次的排他上界，攻击target_turn 的合法取值范围是 [0, injectable_turn_ceiling) 这一值由防御方对话轮次和工具列表长度决定"""
        return self._inject_turn_ceiling

    def sanitize_spec(self, spec: AttackSpec) -> AttackSpec:
        """裁剪越界的 target_turn"""
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

    def current_population(self) -> list[AttackSpec]:
        """Return the batch scheduled for rollout evaluation this round.
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

    def _inject_prewarm_skeletons(self) -> None:
        """For each skeleton entry whose ``turn`` falls in
        ``[0, inject_turn_ceiling)`` and whose ``method`` label is non-empty, an
        :class:`_TreeNode` of level ``L2_method`` is registered as a child of the
        corresponding L1_turn sibling IF no sibling with identical method already
        exists. The payload_template is stashed on the new node's
        ``cached_payload_text`` so subsequent materialization can surface it to
        downstream mutators when they ask for a refinement hint.
        """
        try:
            skel_by_suite = _load_prewarm_skeletons_all()
        except Exception as exc:                                            
            logger.warning("[prewarm] loader raised (%s); skipping injection.", exc)
            return
        if not skel_by_suite:
            return

        suite = (getattr(self.task, "suite", "") or "").strip()
        if not suite:
            return

        # A fixed universal set takes precedence over dataset-specific files;
        # suite files remain a backward-compatible fallback when no universal
        # seed artifact is present.
        entries = skel_by_suite.get("universal") or skel_by_suite.get(suite)
        if not isinstance(entries, list) or not entries:
            return

        added = 0
        skipped_dup = 0
        skipped_oob = 0
        for ent in entries:
            if not isinstance(ent, dict):
                continue
            turn_raw = ent.get("turn", ent.get("target_turn"))
            method = str(ent.get("method") or "").strip()
            tmpl = str(
                ent.get("payload_template", ent.get("payload")) or ""
            ).strip()
            try:
                t_val = int(turn_raw)
            except (TypeError, ValueError):
                skipped_oob += 1
                continue
            if t_val < 0 or t_val >= self._inject_turn_ceiling or not method:
                skipped_oob += 1
                continue
            l1_parent = self._child_with_discriminator(
                self._root, disc_key="turn", disc_value=t_val,
            )
            if l1_parent is None:
                skipped_oob += 1
                continue       
            ex = self._child_with_discriminator(
                l1_parent, disc_key="method", disc_value=method,
            )
            if ex is not None:
                # If pre-existing sibling has no template but we do have one now,
                # backfill so future mutation steps can pick up the template text.
                if not getattr(ex, "cached_payload_text", None) and tmpl:
                    ex.cached_payload_text = tmpl
                    logger.debug(
                        "[prewarm] enriched existing %s/%s with template from seeds.json",
                        suite, method[:32],
                    )
                else:
                    skipped_dup += 1
                continue
            cid = f"L2_{method}_{_short_hash(method)}"
            ch = _TreeNode(
                node_id=cid,
                parent_id=l1_parent.node_id,
                level="L2_method",
                discriminator={"method": method},
                cached_payload_text=tmpl or None,
            )
            self._register_node(ch)
            added += 1

        if added > 0:
            logger.info(
                "[prewarm][task=%s] injected %d skeletons under root "
                "(skipped dup=%d oob=%d).",
                (self.task.task_id or "?")[:60], added, skipped_dup, skipped_oob,
            )

    # ------------------------------------------------------------------ #
    # Selection                                                         #
    # ------------------------------------------------------------------ #
    def _ucb_score(self, parent_visits: int, child: _TreeNode) -> float:
        """Composite selection criterion.
        ``score = exploit_term + c·explore_term + λ·δ_potential``
        """
        eps_visit = 1e-9
        if child.n_visits <= 0:
            exploit_mean_delta = 0.01               
            # 做个宽先验，如果节点未访问，这片方差设为0.25，鼓励探索
            variance_estimate = 0.25                 
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
        path: list[str] = ["root"]
        cur = self._root
        guard_iters = 0
        while True:
            guard_iters += 1
            if guard_iters > 10000:
                break                          
            if not cur.children_ids:
                break                          
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
            self.rng.shuffle(scored_children)
            top_s, top_n = max(scored_children, key=lambda x: x[0])
            path.append(top_n.node_id)
            cur = top_n
            if cur.level == "L3_payload":
                break
            # MCT只有三层，走到头就不往下走
            if not cur.children_ids:
                break                           # leaf with no children → expand now
        return path, cur
 
    def _expand_frontier(self, node: _TreeNode) -> Optional[_TreeNode]:
        """Create one or more new children beneath ``node`` and return the last
        freshly-registered child (or ``None`` when nothing new could be produced).
        """
        if node.level not in {"L1_turn", "L2_method"}:
            return None
        # 调用 LLM 生成器批量获取候选攻击方法
        # 从当前 L1 节点的 discriminator 取出它代表的目标注入轮次编号 t_val
        if node.level == "L1_turn":
            t_val = int(node.discriminator.get("turn", 0))
            # 调用攻击生成器 generator
            seeded = self.generator.seed(
                self.task, self.tools,
                n=_MAX_BATCH_SPAWN_PER_CALL * 2,           # request a bit more than cap so dedup losses don't starve us
                max_turns=t_val + 1, generation=self.generation,
            )
            newly_registered: list[_TreeNode] = []
            # 批量生产，确保去重后的候选节点在0-_MAX_BATCH_SPAWN_PER_CALL之间，这里多请求一倍，主要防止去重给删完了
            for cand_spec in seeded[: _MAX_BATCH_SPAWN_PER_CALL * 2]:
                lbl = cand_spec.method or ""
                if not lbl or len(newly_registered) >= _MAX_BATCH_SPAWN_PER_CALL:
                    continue                              # 
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
            # 构造一个最小化的展位，作为变异器的输入种子
            placeholder = _construct_seed_from_cell(
                t_val,
                node.discriminator.get("method"),
                self.task,
                payload=node.cached_payload_text or "",
            )
            # 基于占位 spec 生成一个具体的攻击 payload 文本
            mutated = self.generator.mutate(self.task, self.tools, placeholder, generation=self.generation)
            sig_hash = _short_hash(mutated.payload or "")
            # 查看生成的新节点和以前的有没有重复
            ex = self._child_with_discriminator(node, disc_key="signature", disc_value=sig_hash)
            if ex is not None:
                return ex # silently reused
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
        at L3), returning both path ids and chosen leaf."""
        path: list[str] = [node.node_id]
        cur = node
        if cur.children_ids:
            # 对于给定的节点，沿UCB最有路径一路下行到 L3 叶节点
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
            goal=self.task.instruction[:200],  
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
                    # 如果无法扩展，在 ``node`` 下方选取某个 L3 叶子节点作为后代（若当前已处于 L3 层级则保持原位）
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
    expecting an individual argument"""
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
