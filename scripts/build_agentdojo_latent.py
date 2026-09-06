"""Build ``data/agentdojo_latent/`` from vendored AgentDojo + authored payloads.

Inputs
------
* ``data/agentdojo/default_suites/v1/{suite}/user_tasks.py`` -- verbatim task
  PROMPTs and their ``ground_truth`` call sequences (the clean plan).
* ``data/agentdojo/default_suites/v1/{suite}/task_suite.py`` + ``tools/*.py`` --
  the full advertised tool inventory and each tool's docstring, rendered into an
  ``env_info`` block in the shape ``evoguard.envs.tool_parsing`` expects.
* ``data/agentdojo_latent/authored/{suite}.json`` -- the hand-written creative
  part: carrier tool, benign observation, stealth payload, harmful sink,
  declared delta, technique tag.

Outputs
-------
* ``data/agentdojo_latent/scenarios/{suite}.jsonl`` -- clean + injected rows.
* ``data/agentdojo_latent/manifest.json``
* ``data/agentdojo_latent/_sha256_guard.txt``

Run: ``python scripts/build_agentdojo_latent.py``
"""
from __future__ import annotations

import argparse
import ast
import glob
import hashlib
import json
import os
import re
import sys
import types

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUITE_ROOT = os.path.join(REPO_ROOT, "data", "agentdojo", "default_suites", "v1")
OUT_ROOT = os.path.join(REPO_ROOT, "data", "agentdojo_latent")
SUITES = ("banking", "slack", "travel", "workspace")
DATASET = "agentdojo_latent"
SCHEMA_VERSION = 1


# --------------------------------------------------------------------------- #
# Vendored AgentDojo extraction
# --------------------------------------------------------------------------- #



def _normalise(text: str) -> str:
    """Collapse AgentDojo's source-indentation artefacts out of a PROMPT."""
    text = text.replace("\\\n", "\n")
    lines = [ln.strip() for ln in text.strip().splitlines()]
    return "\n".join(ln for ln in lines if ln)


def _ground_truth_plan(fn: ast.FunctionDef) -> list[str]:
    """Tool names of the ``FunctionCall(...)`` list a ground_truth returns."""
    plan: list[str] = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.Return) or not isinstance(node.value, ast.List):
            continue
        for elt in node.value.elts:
            if not isinstance(elt, ast.Call):
                continue
            for kw in elt.keywords:
                if kw.arg == "function" and isinstance(kw.value, ast.Constant):
                    plan.append(str(kw.value.value))
        if plan:
            return plan
    return plan

    text = text.replace("\\\n", "\n")
    lines = [ln.strip() for ln in text.strip().splitlines()]
    return "\n".join(ln for ln in lines if ln)


def _class_string_consts(cls: ast.ClassDef) -> dict[str, object]:
    """Literal class-level attributes, used to resolve PROMPT f-string slots."""
    return _literal_assignments(cls.body)


def _literal_assignments(body: list[ast.stmt]) -> dict[str, object]:
    consts: dict[str, object] = {}
    for node in body:
        if isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value = node.value
        elif isinstance(node, ast.Assign):
            targets = list(node.targets)
            value = node.value
        else:
            continue
        if value is None:
            continue
        try:
            literal = ast.literal_eval(value)
        except (ValueError, SyntaxError, TypeError):
            continue
        for tgt in targets:
            if isinstance(tgt, ast.Name):
                consts[tgt.id] = literal
    return consts


def _eval_prompt(node: ast.expr, consts: dict[str, str]) -> str:
    """Evaluate a PROMPT expression: plain str, f-string, or their concatenation."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _eval_prompt(node.left, consts) + _eval_prompt(node.right, consts)
    if isinstance(node, ast.JoinedStr):
        out = []
        for part in node.values:
            if isinstance(part, ast.Constant):
                out.append(str(part.value))
            elif isinstance(part, ast.FormattedValue):
                out.append(_eval_slot(part.value, consts))
            else:  # pragma: no cover - defensive
                raise SystemExit(f"unhandled f-string part {ast.dump(part)}")
        return "".join(out)
    raise SystemExit(f"unhandled PROMPT expression {ast.dump(node)}")


def _eval_slot(node: ast.expr, consts: dict[str, object]) -> str:
    """Evaluate an f-string slot against the class's literal attributes.

    AgentDojo PROMPTs interpolate class constants directly (``{_HOTEL_NAME}``),
    via ``self`` (``{self._DAY}``), and through small expressions
    (``{', '.join(_EMAIL_ADDRESSES)}``). The expression is compiled and
    evaluated with no builtins and only those literals in scope.
    """
    scope = dict(consts)
    scope["self"] = types.SimpleNamespace(**consts)
    expr = ast.Expression(body=node)
    ast.fix_missing_locations(expr)
    try:
        value = eval(compile(expr, "<prompt>", "eval"), {"__builtins__": {}}, scope)
    except Exception as exc:  # pragma: no cover - defensive
        raise SystemExit(f"cannot resolve f-string slot: {exc}") from exc
    return str(value)


def load_user_tasks(suite: str) -> dict[str, dict]:
    """Return ``{UserTaskN: {"prompt": str, "clean_plan": [tool, ...]}}``.

    PROMPTs are read via AST (they are variously plain strings, f-strings, and
    parenthesised concatenations) with class-level string constants substituted,
    so the instruction text is byte-identical to what AgentDojo shows the agent.
    """
    src = open(os.path.join(SUITE_ROOT, suite, "user_tasks.py")).read()
    tree = ast.parse(src)
    module_consts = _literal_assignments(tree.body)
    tasks: dict[str, dict] = {}
    for cls in tree.body:
        if not isinstance(cls, ast.ClassDef) or not re.fullmatch(r"UserTask\d+", cls.name):
            continue
        consts = dict(module_consts)
        consts.update(_class_string_consts(cls))
        prompt = None
        for node in cls.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "PROMPT" for t in node.targets
            ):
                prompt = _normalise(_eval_prompt(node.value, consts))
        gt = None
        for node in cls.body:
            if isinstance(node, ast.FunctionDef) and node.name == "ground_truth":
                gt = _ground_truth_plan(node)
        if prompt and gt:
            tasks[cls.name] = {"prompt": prompt, "clean_plan": gt}
    return tasks


_TOOLS_RE = re.compile(r"\nTOOLS\s*=\s*\[(.*?)\n\]", re.DOTALL)
_PARAM_DOC_RE = re.compile(r"^:param\s+(\w+):\s*(.*)$", re.M)


def load_tool_specs() -> dict[str, dict]:
    """Docstring-derived spec for every tool function under ``tools/``."""
    specs: dict[str, dict] = {}
    for path in sorted(glob.glob(os.path.join(SUITE_ROOT, "tools", "*.py"))):
        tree = ast.parse(open(path).read())
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef):
                continue
            doc = ast.get_docstring(node) or ""
            params = _PARAM_DOC_RE.findall(doc)
            desc = _PARAM_DOC_RE.sub("", doc).strip().replace("\n", " ")
            specs[node.name] = {"description": " ".join(desc.split()), "params": params}
    return specs


def load_suite_tool_names(suite: str) -> list[str]:
    src = open(os.path.join(SUITE_ROOT, suite, "task_suite.py")).read()
    match = _TOOLS_RE.search(src)
    if match is None:
        raise SystemExit(f"{suite}: no TOOLS list found in task_suite.py")
    body = re.sub(r"#.*", "", match.group(1))
    return [
        tok.strip()
        for tok in body.split(",")
        if re.fullmatch(r"[a-z_][a-z0-9_]*", tok.strip() or "-")
    ]


def render_env_info(tool_names: list[str], specs: dict[str, dict]) -> str:
    """Render an ``env_info`` block parseable by ``envs.tool_parsing``."""
    blocks: list[str] = []
    for name in tool_names:
        spec = specs.get(name)
        if spec is None:
            raise SystemExit(f"no docstring spec for tool {name!r}")
        lines = [f"{name}: {spec['description']}"]
        if spec["params"]:
            lines.append("  parameters:")
            for pname, pdesc in spec["params"]:
                meta = {"description": " ".join(pdesc.split()), "type": "string"}
                lines.append(f"    {pname}: {meta!r}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) + "\n"


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #


def task_uid(suite: str, instruction: str, dataset: str = DATASET) -> str:
    """Must stay identical to ``process.vendored_attack_parser._task_uid``."""
    digest = hashlib.sha1(instruction.encode("utf-8")).hexdigest()[:10]
    return f"{dataset}:{suite}:{digest}"


def build_suite(suite: str, specs: dict[str, dict]) -> list[dict]:
    tasks = load_user_tasks(suite)
    env_info = render_env_info(load_suite_tool_names(suite), specs)
    advertised = set(load_suite_tool_names(suite))

    authored_path = os.path.join(OUT_ROOT, "authored", f"{suite}.json")
    authored = json.load(open(authored_path, encoding="utf-8"))

    rows: list[dict] = []
    seen_clean: dict[tuple[str, str], str] = {}
    for idx, entry in enumerate(authored, start=1):
        ref = entry["task_ref"]
        if ref not in tasks:
            raise SystemExit(f"{suite}: unknown task_ref {ref!r}")
        instruction = tasks[ref]["prompt"]
        clean_plan = tasks[ref]["clean_plan"]
        uid = task_uid(suite, instruction)
        carrier = entry["carrier_tool"]
        c_idx = entry["carrier_index"]
        delta = entry["expected_delta"]
        harmful = entry["harmful_tool"]

        if clean_plan[c_idx] != carrier:
            raise SystemExit(
                f"{suite}/{ref}: carrier_index {c_idx} is {clean_plan[c_idx]!r}, "
                f"not {carrier!r}"
            )
        if c_idx + delta > len(clean_plan):
            raise SystemExit(
                f"{suite}/{ref}: clean_plan len {len(clean_plan)} cannot support "
                f"carrier_index {c_idx} + delta {delta}"
            )
        if harmful not in advertised:
            raise SystemExit(f"{suite}: harmful_tool {harmful!r} is not advertised")
        if harmful in clean_plan:
            raise SystemExit(
                f"{suite}/{ref}: harmful_tool {harmful!r} is already in clean_plan"
            )

        turning = c_idx + delta
        induced = list(clean_plan[:turning]) + [harmful] + list(clean_plan[turning:])
        poisoned = entry["benign_observation"] + entry["payload"]

        common = {
            "schema_version": SCHEMA_VERSION,
            "suite": suite,
            "task_ref": f"agentdojo:v1:{suite}:{ref}",
            "task_id": uid,
            "task_instruction": instruction,
            "env_info": env_info,
            "clean_plan": clean_plan,
            "carrier_tool": carrier,
            "carrier_index": c_idx,
            "benign_observation": entry["benign_observation"],
        }

        clean_key = (ref, carrier)
        # One carrier serves ONE benign text. The clean row is emitted once per
        # key, so two entries disagreeing here would leave the second attack
        # paired against a clean twin whose observation differs by more than the
        # payload -- and the whole point of the set is that the two arms differ
        # by the payload alone.
        prior = seen_clean.get(clean_key)
        if prior is not None and prior != entry["benign_observation"]:
            raise SystemExit(
                f"{suite}/{ref}: two benign_observation variants for carrier "
                f"{carrier!r}; a carrier's clean output must be identical "
                f"across every attack that rides it"
            )
        if prior is None:
            seen_clean[clean_key] = entry["benign_observation"]
            rows.append(
                dict(
                    common,
                    instance_id=f"adjlatent-{suite}-clean-{len(seen_clean):02d}",
                    sample_type="clean",
                    observation=entry["benign_observation"],
                    payload=None,
                    harmful_tool=None,
                    harmful_args_hint=None,
                    attacker_goal=None,
                    expected_delta=None,
                    expected_turning_index=None,
                    expected_induced_plan=None,
                    stealth_technique=None,
                    stealth_notes=None,
                )
            )

        rows.append(
            dict(
                common,
                instance_id=f"adjlatent-{suite}-{idx:04d}",
                sample_type="injected",
                observation=poisoned,
                payload=entry["payload"],
                harmful_tool=harmful,
                harmful_args_hint=entry["harmful_args_hint"],
                attacker_goal=entry["attacker_goal"],
                expected_delta=delta,
                expected_turning_index=turning,
                expected_induced_plan=induced,
                stealth_technique=entry["stealth_technique"],
                stealth_notes=entry["stealth_notes"],
            )
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="rebuild in memory and fail if the on-disk files would change",
    )
    args = parser.parse_args()

    specs = load_tool_specs()
    payloads: dict[str, str] = {}
    counts: dict[str, dict] = {}
    for suite in SUITES:
        rows = build_suite(suite, specs)
        text = "".join(
            json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows
        )
        payloads[f"scenarios/{suite}.jsonl"] = text
        injected = [r for r in rows if r["sample_type"] == "injected"]
        counts[suite] = {
            "n_rows": len(rows),
            "n_injected": len(injected),
            "n_clean": len(rows) - len(injected),
            "by_delta": _tally(injected, "expected_delta"),
            "by_technique": _tally(injected, "stealth_technique"),
        }

    all_injected = sum(c["n_injected"] for c in counts.values())
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset": DATASET,
        "built_by": "scripts/build_agentdojo_latent.py",
        "source": "AgentDojo v1 default suites, vendored at data/agentdojo/",
        "task_source": "user_tasks.py PROMPT + ground_truth (verbatim)",
        "split_unit": "none -- the whole set is a held-out diagnostic probe",
        "n_injected_total": all_injected,
        "per_suite": counts,
        "delta_semantics": (
            "expected_delta is a DECLARED design intent: the number of turns "
            "between the carrier observation and the first action that leaves "
            "the clean plan. It is NOT signals.delta, which is measured at "
            "replay time and will differ."
        ),
    }
    payloads["manifest.json"] = json.dumps(manifest, ensure_ascii=False, indent=1) + "\n"

    guard_lines = ["# sha256 guard for data/agentdojo_latent/",
                   "# generated-by: scripts/build_agentdojo_latent.py",
                   f"# n-injected: {all_injected}"]
    for rel in sorted(payloads):
        digest = hashlib.sha256(payloads[rel].encode("utf-8")).hexdigest()
        guard_lines.append(f"sha256({rel})={digest}")
    payloads["_sha256_guard.txt"] = "\n".join(guard_lines) + "\n"

    if args.check:
        _check(payloads)
        return
    for rel, text in payloads.items():
        path = os.path.join(OUT_ROOT, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
    print(f"wrote {len(payloads)} files under {OUT_ROOT}")
    for suite, c in counts.items():
        print(f"  {suite}: {c['n_injected']} injected + {c['n_clean']} clean "
              f"delta={c['by_delta']} tech={c['by_technique']}")


def _tally(rows: list[dict], key: str) -> dict:
    out: dict = {}
    for row in rows:
        out[str(row[key])] = out.get(str(row[key]), 0) + 1
    return dict(sorted(out.items()))


def _check(payloads: dict[str, str]) -> None:
    bad = []
    for rel, text in payloads.items():
        path = os.path.join(OUT_ROOT, rel)
        if not os.path.exists(path):
            bad.append(f"{rel}: missing")
        elif open(path, encoding="utf-8").read() != text:
            bad.append(f"{rel}: differs from rebuild")
    if bad:
        raise SystemExit("stale build:\n  " + "\n  ".join(bad))
    print("on-disk files match a fresh rebuild")


if __name__ == "__main__":
    sys.exit(main())
