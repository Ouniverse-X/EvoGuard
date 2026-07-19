"""LLM-backed defense agent (open-source model served via vLLM).

The agent renders the interaction history into a chat prompt, queries its LLM
(which may be a base model or a LoRA adapter registered on the vLLM server) and
parses the JSON action. Robust parsing tolerates minor formatting deviations so
a single malformed reply degrades to a terminal action instead of crashing a
whole round.
"""

from __future__ import annotations

import json
from typing import Optional, Sequence

from evoguard.agents.base import DefenseAgent
from evoguard.agents.prompts import build_system_prompt, render_history
from evoguard.config import DefenseConfig
from evoguard.core.types import Action, Message, Role, Task, ToolCall, ToolSpec
from evoguard.llm import build_client
from evoguard.llm.base import LLMClient
from evoguard.llm.schemas import DEFENSE_ACTION_SCHEMA
from evoguard.utils.logging import get_logger

logger = get_logger("agents.llm")


class LLMDefenseAgent(DefenseAgent):
    """Defense agent driven by a chat LLM using the ReAct-JSON protocol."""

    name = "llm_defense"

    def __init__(self, config: DefenseConfig, *, client: Optional[LLMClient] = None, seed: int = 0):
        self.config = config
        self._client = client or build_client(config.llm, seed=seed)

    def decide(
        self,
        task: Task,
        tools: Sequence[ToolSpec],
        history: Sequence[Action],
    ) -> Action:
        system = build_system_prompt(task, tools, self.config.system_prompt)
        user = render_history(history)
        resp = self._client.chat(
            [Message(role=Role.SYSTEM, content=system), Message(role=Role.USER, content=user)],
            temperature=self.config.llm.temperature,
            max_tokens=self.config.llm.max_tokens,
            response_format=DEFENSE_ACTION_SCHEMA,
        )
        return self._parse_action(resp.text, turn=len(history))

    # ---- parsing ---------------------------------------------------------- #
    def _parse_action(self, text: str, turn: int) -> Action:
        data = _extract_json_object(text)
        if data is None:
            logger.debug("Unparseable agent output at turn %d: %r", turn, text[:200])
            return Action(turn=turn, thought=text.strip()[:500], final_answer=text.strip()[:500])

        thought = str(data.get("thought", ""))
        if "tool" in data and data.get("tool"):
            args = data.get("args", {}) or {}
            if not isinstance(args, dict):
                args = {}
            return Action(
                turn=turn,
                thought=thought,
                tool_call=ToolCall(name=str(data["tool"]), arguments=args),
            )
        final = str(data.get("final_answer", "")) or "Done."
        return Action(turn=turn, thought=thought, final_answer=final)


def _extract_json_object(text: str) -> Optional[dict]:
    """Best-effort extraction of the first top-level JSON object in ``text``."""

    text = text.strip()
    # Strip common markdown code fences.
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    try:
        start = text.index("{")
    except ValueError:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None
