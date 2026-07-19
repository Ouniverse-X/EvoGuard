"""Sequence alignment over agent action signatures.
We detect it via a Needleman-Wunsch style global alignment over the two *(tool + args)* signature
sequences, using a graded substitution cost so that "same tool, different args"
is a smaller divergence than "different tool".

The alignment yields:

* the Levenshtein-style edit distance between the two action sequences, and
* the first aligned column whose local cost exceeds a threshold, mapped back to
  the turn index in the attacked trajectory (the turning point).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from evoguard.core.types import ToolCall


def token_substitution_cost(a: str, b: str) -> float:
    """
    * identical signatures -> 0.0
    * same tool name, different arguments -> 0.5
    * different tool name -> 1.0
    """

    if a == b:
        return 0.0
    tool_a = a.split("(", 1)[0]
    tool_b = b.split("(", 1)[0]
    if tool_a == tool_b:
        return 0.5
    return 1.0


@dataclass
class AlignmentOp:
    """One column of the alignment."""

    kind: str  # "match" | "sub" | "del" | "ins"
    cost: float
    i: Optional[int]  # index into sequence A (clean), or None for insertion
    j: Optional[int]  # index into sequence B (attacked), or None for deletion


@dataclass
class AlignmentResult:
    distance: float
    ops: list[AlignmentOp]


def align(seq_a: Sequence[str], seq_b: Sequence[str]) -> AlignmentResult:
    """Global alignment (Needleman-Wunsch) with unit indel and graded sub costs."""

    n, m = len(seq_a), len(seq_b)
    # dp[i][j] = min cost to align seq_a[:i] with seq_b[:j].
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]  # type: ignore[var-annotated]

    for i in range(1, n + 1):
        dp[i][0] = i
        back[i][0] = "del"
    for j in range(1, m + 1):
        dp[0][j] = j
        back[0][j] = "ins"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub = dp[i - 1][j - 1] + token_substitution_cost(seq_a[i - 1], seq_b[j - 1])
            dele = dp[i - 1][j] + 1.0
            ins = dp[i][j - 1] + 1.0
            best = min(sub, dele, ins)
            dp[i][j] = best
            if best == sub:
                back[i][j] = "sub"
            elif best == dele:
                back[i][j] = "del"
            else:
                back[i][j] = "ins"

    # Trace back to recover the column ops.
    ops: list[AlignmentOp] = []
    i, j = n, m
    while i > 0 or j > 0:
        move = back[i][j]
        if move == "sub":
            c = token_substitution_cost(seq_a[i - 1], seq_b[j - 1])
            ops.append(AlignmentOp("match" if c == 0 else "sub", c, i - 1, j - 1))
            i, j = i - 1, j - 1
        elif move == "del":
            ops.append(AlignmentOp("del", 1.0, i - 1, None))
            i -= 1
        else:  # ins
            ops.append(AlignmentOp("ins", 1.0, None, j - 1))
            j -= 1
    ops.reverse()
    return AlignmentResult(distance=dp[n][m], ops=ops)


def first_divergence_index_b(
    seq_a: Sequence[str],
    seq_b: Sequence[str],
    threshold: float,
) -> Optional[int]:
    """Return the index in ``seq_b`` of the first significant divergence.

    A column diverges when it is an insertion in B, a deletion from A, or a
    substitution whose cost exceeds ``threshold``. The returned value is the
    position within ``seq_b`` (0-based over tool calls), or ``None`` if the two
    sequences never significantly diverge.
    """

    result = align(seq_a, seq_b)
    for op in result.ops:
        diverges = (
            op.kind == "ins"
            or op.kind == "del"
            or (op.kind == "sub" and op.cost > threshold)
        )
        if diverges:
            if op.j is not None:
                return op.j
            return _next_b_index(result.ops, op)
    return None


def _next_b_index(ops: list[AlignmentOp], current: AlignmentOp) -> Optional[int]:
    seen_current = False
    for op in ops:
        if op is current:
            seen_current = True
            continue
        if seen_current and op.j is not None:
            return op.j
    return None


def signatures(calls: Sequence[Optional[ToolCall]]) -> list[str]:
    """Signatures for a list of (possibly None) tool calls, skipping None."""

    return [c.signature() for c in calls if c is not None]
