"""Meta-SecAlign defense agent (eval-only baseline).

``facebook/Meta-SecAlign-8B`` is a LoRA adapter on Llama-3.1-8B-Instruct whose
defense *is a chat role*. It was preference-trained so that anything rendered
under ``<|start_header_id|>input<|end_header_id|>`` carries **data authority
only**: instructions found there must be reported, never obeyed. Serving those
weights while flattening the transcript into one USER message -- which is what
:class:`~evoguard.agents.llm_agent.LLMDefenseAgent` does via
``prompts.render_history`` -- hands every poisoned observation to the model with
full instruction authority and measures a defense that was never engaged. This
subclass exists solely to place the observations in the right role.

What changes and what does not
------------------------------
Only the message *shape* changes. The system prompt (protocol, tool list, task),
the ``DEFENSE_ACTION_SCHEMA`` structured output and :meth:`_parse_action` are
inherited unchanged, so a trajectory from this agent is comparable to the base
arm turn-for-turn. The unavoidable confound is the shape itself: the base arm
sees one flat user string, this arm a structured conversation. That is not a bug
to fix -- it is the intervention -- but it must be stated next to any number.

No delimiters around the untrusted span
---------------------------------------
The observation is the ``input`` message content **verbatim**: no fences, no
``[turn N]`` prefix, no "untrusted data follows" banner. Meta's own instructions
are explicit that nothing may be added there, and for good reason -- a delimiter
is a string the payload can also emit, so adding one teaches the model a cue an
attacker can forge. The turn index therefore survives only in the assistant
messages, which is enough: they alternate strictly with the inputs.

Assistant turns are replayed as the action JSON
-----------------------------------------------
Each past tool call is re-emitted in the exact ``{"thought", "tool", "args"}``
form the model is asked to produce. Rendering it as prose instead would make the
model's own history off-protocol and invite it to answer in prose too.
"""

from __future__ import annotations

import json
from typing import Sequence

from evoguard.agents.llm_agent import LLMDefenseAgent
from evoguard.agents.prompts import build_system_prompt
from evoguard.core.types import Action, Message, Role, Task, ToolSpec
from evoguard.llm.schemas import DEFENSE_ACTION_SCHEMA
from evoguard.utils.logging import get_logger

logger = get_logger("agents.secalign")

#: Trailing trusted turn. The request to act must come from a USER message: if
#: the prompt ended on the ``input`` block, the only thing asking the model to do
#: anything would be the untrusted text itself.
_NEXT_STEP = "Decide your next step. Respond with a single JSON object."


class SecAlignDefenseAgent(LLMDefenseAgent):
    """``LLMDefenseAgent`` that routes tool observations through ``Role.INPUT``."""

    name = "secalign_defense"

    def decide(
        self,
        task: Task,
        tools: Sequence[ToolSpec],
        history: Sequence[Action],
    ) -> Action:
        messages = build_secalign_messages(
            task, tools, history, self.config.system_prompt,
        )
        resp = self._client.chat(
            messages,
            temperature=self.config.llm.temperature,
            max_tokens=self.config.llm.max_tokens,
            response_format=DEFENSE_ACTION_SCHEMA,
        )
        return self._parse_action(resp.text, turn=len(history))


def build_secalign_messages(
    task: Task,
    tools: Sequence[ToolSpec],
    history: Sequence[Action],
    system_prompt: str | None = None,
) -> list[Message]:
    """Render the transcript as ``system, user, (assistant, input)*, user``.

    Module-level so the role sequence can be asserted offline without a client.
    Turns that produced no tool call contribute nothing: they are terminal, so
    there is no observation to attribute and no continuation to justify.
    """

    messages = [
        Message(role=Role.SYSTEM,
                content=build_system_prompt(task, tools, system_prompt)),
        Message(role=Role.USER, content=task.instruction),
    ]
    for past in history:
        if past.tool_call is None:
            continue
        messages.append(Message(role=Role.ASSISTANT, content=json.dumps(
            {
                "thought": past.thought,
                "tool": past.tool_call.name,
                "args": past.tool_call.arguments,
            },
            ensure_ascii=False,
        )))
        messages.append(Message(role=Role.INPUT, content=past.observation))
    messages.append(Message(role=Role.USER, content=_NEXT_STEP))
    return messages


__all__ = ["SecAlignDefenseAgent", "build_secalign_messages"]
