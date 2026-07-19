"""Parsers that normalize dataset tool definitions into :class:`ToolSpec`.

The toolsafe annotation files store each task's available tools in an
``env_info`` string with this shape::

    tool_name: Free-form description spanning to the next blank line.
      parameters:
        arg_name: {'description': '...', 'type': 'str'}
        other_arg: {'description': '...', 'type': 'integer'}

    next_tool: ...

This module turns that text into structured :class:`ToolSpec` objects. Argument
metadata is written as Python-dict literals (single quotes), so we parse them
with :func:`ast.literal_eval` rather than :func:`json.loads`.
"""

from __future__ import annotations

import ast
import re
from typing import Any

from evoguard.core.types import ToolParameter, ToolSpec

# A tool header is a line like "tool_name: description" at column 0.
_TOOL_HEADER_RE = re.compile(r"^(?P<name>[A-Za-z_]\w*):\s?(?P<desc>.*)$")
# A parameter line is indented and looks like "arg: {'description': ..., 'type': ...}".
_PARAM_RE = re.compile(r"^\s+(?P<name>[A-Za-z_]\w*):\s*(?P<meta>\{.*\})\s*$")


def parse_env_info(env_info: str) -> list[ToolSpec]:
    """Parse a toolsafe ``env_info`` string into a list of :class:`ToolSpec`."""

    specs: list[ToolSpec] = []
    current: ToolSpec | None = None
    in_params = False

    for raw_line in env_info.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            continue

        # Parameter line (indented, dict literal).
        param_match = _PARAM_RE.match(line)
        if param_match and current is not None and in_params:
            meta = _safe_literal(param_match.group("meta"))
            current.parameters.append(
                ToolParameter(
                    name=param_match.group("name"),
                    type=str(meta.get("type", "string")) if isinstance(meta, dict) else "string",
                    description=str(meta.get("description", "")) if isinstance(meta, dict) else "",
                    required=(meta.get("required", True) if isinstance(meta, dict) else True),
                )
            )
            continue

        # "parameters:" marker line.
        if line.strip() == "parameters:" and current is not None:
            in_params = True
            continue

        # Tool header line (column 0, not indented).
        if not line.startswith(" "):
            header = _TOOL_HEADER_RE.match(line)
            if header:
                if current is not None:
                    specs.append(current)
                current = ToolSpec(
                    name=header.group("name"),
                    description=header.group("desc").strip(),
                    parameters=[],
                )
                in_params = False
                continue

        # Continuation of a multi-line description.
        if current is not None and not in_params:
            current.description = (current.description + " " + line.strip()).strip()

    if current is not None:
        specs.append(current)
    return specs


def _safe_literal(text: str) -> Any:
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return {}
