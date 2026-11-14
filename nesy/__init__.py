"""Neurosymbolic arithmetic: neural perception + symbolic reasoning in pure NumPy."""

from .data import make_pairs, render_digit, to_ascii
from .logic import KnowledgeBase, parse_atom, parse_rule
from .perception import MLP, softmax
from .semantic import SymbolicTask, TASKS

__all__ = [
    "make_pairs",
    "render_digit",
    "to_ascii",
    "KnowledgeBase",
    "parse_atom",
    "parse_rule",
    "MLP",
    "softmax",
    "SymbolicTask",
    "TASKS",
]
