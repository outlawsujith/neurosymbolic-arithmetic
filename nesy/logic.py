"""The symbolic half: a small Datalog-style forward-chaining reasoner.

Rules are Horn clauses written in Prolog-like syntax:

    sum(Z) :- left(X), right(Y), add(X, Y, Z).

Names starting with an uppercase letter are variables; integers and lowercase
names are constants. Arithmetic is provided by built-in predicates. Every
derived fact remembers the rule and premises that produced it, so any
conclusion can be explained with a proof tree.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Iterator

Atom = tuple  # (predicate, arg1, arg2, ...)


def is_var(term) -> bool:
    return isinstance(term, str) and (term[0].isupper() or term[0] == "_")


def _parse_term(text: str):
    text = text.strip()
    return int(text) if re.fullmatch(r"-?\d+", text) else text


def parse_atom(text: str) -> Atom:
    m = re.fullmatch(r"\s*([a-z]\w*)\s*(?:\((.*)\))?\s*", text)
    if not m:
        raise ValueError(f"cannot parse atom: {text!r}")
    args = [_parse_term(a) for a in m.group(2).split(",")] if m.group(2) else []
    return (m.group(1), *args)


def _split_body(text: str) -> list[str]:
    parts, depth, current = [], 0, ""
    for ch in text:
        if ch == "," and depth == 0:
            parts.append(current)
            current = ""
            continue
        depth += ch == "("
        depth -= ch == ")"
        current += ch
    return parts + [current]


@dataclass(frozen=True)
class Rule:
    head: Atom
    body: tuple

    def __str__(self) -> str:
        return f"{format_atom(self.head)} :- {', '.join(map(format_atom, self.body))}."


def parse_rule(text: str) -> Rule:
    text = text.strip().rstrip(".")
    head, sep, body = text.partition(":-")
    if not sep:
        raise ValueError(f"rule needs a body: {text!r}")
    return Rule(parse_atom(head), tuple(parse_atom(a) for a in _split_body(body)))


def format_atom(atom: Atom) -> str:
    pred, *args = atom
    return f"{pred}({', '.join(map(str, args))})" if args else pred


def _builtin_add(x, y, z):
    if x is not None and y is not None:
        yield (x, y, x + y)
    elif x is not None and z is not None:
        yield (x, z - x, z)
    elif y is not None and z is not None:
        yield (z - y, y, z)


def _builtin_mul(x, y, z):
    if x is not None and y is not None:
        yield (x, y, x * y)


def _builtin_mod(x, y, z):
    if x is not None and y is not None and y != 0:
        yield (x, y, x % y)


def _comparison(test: Callable[[int, int], bool]):
    def builtin(x, y):
        if x is not None and y is not None and test(x, y):
            yield (x, y)
    return builtin


BUILTINS: dict[str, Callable] = {
    "add": _builtin_add,
    "mul": _builtin_mul,
    "mod": _builtin_mod,
    "lt": _comparison(lambda x, y: x < y),
    "gt": _comparison(lambda x, y: x > y),
    "le": _comparison(lambda x, y: x <= y),
    "ge": _comparison(lambda x, y: x >= y),
    "neq": _comparison(lambda x, y: x != y),
}


def _unify(pattern: Atom, fact: Atom, theta: dict):
    """Match a possibly non-ground atom against a ground fact."""
    if len(pattern) != len(fact) or pattern[0] != fact[0]:
        return None
    theta = dict(theta)
    for p, f in zip(pattern[1:], fact[1:]):
        if is_var(p):
            if theta.setdefault(p, f) != f:
                return None
        elif p != f:
            return None
    return theta


class KnowledgeBase:
    def __init__(self, rules=(), builtins: dict | None = None):
        self.rules: list[Rule] = []
        self.facts: set[Atom] = set()
        # fact -> (rule, premises) for derived facts, None for given facts
        self.provenance: dict[Atom, tuple | None] = {}
        self.builtins = dict(BUILTINS if builtins is None else builtins)
        for rule in rules:
            self.add_rule(rule)

    def add_rule(self, rule) -> None:
        self.rules.append(parse_rule(rule) if isinstance(rule, str) else rule)

    def add_fact(self, fact) -> None:
        atom = parse_atom(fact) if isinstance(fact, str) else tuple(fact)
        if any(is_var(t) for t in atom[1:]):
            raise ValueError(f"facts must be ground: {format_atom(atom)}")
        if atom not in self.facts:
            self.facts.add(atom)
            self.provenance[atom] = None

    def _solve(self, body: tuple, theta: dict, support: tuple) -> Iterator[tuple]:
        if not body:
            yield theta, support
            return
        first, rest = body[0], body[1:]
        pred = first[0]
        if pred in self.builtins:
            args = [theta.get(t) if is_var(t) else t for t in first[1:]]
            candidates = [(pred, *result) for result in self.builtins[pred](*args)]
        else:
            candidates = [f for f in self.facts if f[0] == pred]
        for fact in candidates:
            new_theta = _unify(first, fact, theta)
            if new_theta is not None:
                yield from self._solve(rest, new_theta, support + (fact,))

    def forward_chain(self) -> set[Atom]:
        """Apply every rule until no new fact appears. Returns the new facts."""
        derived: set[Atom] = set()
        changed = True
        while changed:
            changed = False
            for rule in self.rules:
                for theta, support in list(self._solve(rule.body, {}, ())):
                    head = (rule.head[0], *(theta.get(t, t) for t in rule.head[1:]))
                    if any(is_var(t) for t in head[1:]):
                        raise ValueError(f"unsafe rule, unbound head variable: {rule}")
                    if head not in self.facts:
                        self.facts.add(head)
                        self.provenance[head] = (rule, support)
                        derived.add(head)
                        changed = True
        return derived

    def holds(self, query) -> bool:
        atom = parse_atom(query) if isinstance(query, str) else tuple(query)
        return atom in self.facts

    def query(self, pattern) -> list[dict]:
        """All variable bindings under which `pattern` matches a known fact."""
        atom = parse_atom(pattern) if isinstance(pattern, str) else tuple(pattern)
        matches = (_unify(atom, f, {}) for f in self.facts)
        return [m for m in matches if m is not None]

    def explain(self, fact, indent: int = 0) -> list[str]:
        """Proof tree for a fact, as indented lines of text."""
        atom = parse_atom(fact) if isinstance(fact, str) else tuple(fact)
        pad = "  " * indent
        if atom[0] in self.builtins:
            return [f"{pad}{format_atom(atom)}   [arithmetic]"]
        if atom not in self.facts:
            return [f"{pad}{format_atom(atom)}   [not provable]"]
        source = self.provenance[atom]
        if source is None:
            return [f"{pad}{format_atom(atom)}   [perceived]"]
        rule, premises = source
        lines = [f"{pad}{format_atom(atom)}   [by rule: {rule}]"]
        for premise in premises:
            lines += self.explain(premise, indent + 1)
        return lines
