"""Symbolic half: a small Datalog engine with provenance.

Rules are Horn clauses over plant facts. Forward chaining derives the scheduling
constraints (forbidden hours, conflicts, precedences) and remembers which rule
produced each one, so every constraint in the optimiser can be traced to a rule.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

Atom = tuple  # (predicate, arg, ...)


@dataclass(frozen=True)
class Rule:
    id: str
    text: str
    head: Atom
    body: tuple[Atom, ...]
    neq: tuple[tuple, ...] = ()


def _term(s: str):
    s = s.strip()
    return int(s) if re.fullmatch(r"-?\d+", s) else s


def _is_var(t) -> bool:
    return isinstance(t, str) and t[:1].isupper()


def _atom(s: str) -> Atom:
    m = re.fullmatch(r"\s*(\w+)\(([^)]*)\)\s*", s)
    if not m:
        raise ValueError(f"cannot parse atom: {s!r}")
    return (m.group(1), *[_term(a) for a in m.group(2).split(",")])


def parse_rules(text: str) -> list[Rule]:
    """Parse rules of the form::

        [R1] Human-readable statement of the rule.
        head(X,Y) :- body1(X), body2(Y), X != Y.
    """
    rules, rid, desc = [], None, ""
    for raw in text.splitlines():
        line = raw.split("#")[0].strip()
        if not line:
            continue
        m = re.match(r"\[(\w+)\]\s*(.*)", line)
        if m:
            rid, desc = m.groups()
            continue
        if rid is None or ":-" not in line:
            raise ValueError(f"rule needs an [ID] header and ':-': {raw!r}")
        head, body = line.rstrip(".").split(":-")
        atoms, neq = [], []
        for part in re.split(r",(?![^(]*\))", body):
            if "!=" in part:
                a, b = part.split("!=")
                neq.append((_term(a), _term(b)))
            else:
                atoms.append(_atom(part))
        rules.append(Rule(rid, desc, _atom(head), tuple(atoms), tuple(neq)))
    return rules


class KnowledgeBase:
    def __init__(self, rules: list[Rule], facts: set[Atom]):
        self.rules = rules
        self.facts = set(facts)
        self.why: dict[Atom, set[str]] = {}  # derived fact -> ids of rules that derive it

    def infer(self) -> "KnowledgeBase":
        """Naive forward chaining to a fixpoint."""
        changed = True
        while changed:
            changed = False
            index = defaultdict(list)
            for f in self.facts:
                index[f[0]].append(f)
            for r in self.rules:
                for env in self._match(r.body, 0, {}, index):
                    if any(env.get(a, a) == env.get(b, b) for a, b in r.neq):
                        continue
                    fact = (r.head[0], *[env.get(t, t) for t in r.head[1:]])
                    ids = self.why.setdefault(fact, set())
                    if r.id not in ids:
                        ids.add(r.id)
                        self.facts.add(fact)
                        changed = True
        return self

    def _match(self, body, i, env, index):
        if i == len(body):
            yield env
            return
        pred, *args = body[i]
        for fact in index.get(pred, ()):
            if len(fact) != len(args) + 1:
                continue
            new = dict(env)
            for a, v in zip(args, fact[1:]):
                if _is_var(a):
                    if new.setdefault(a, v) != v:
                        break
                elif a != v:
                    break
            else:
                yield from self._match(body, i + 1, new, index)

    def query(self, pred: str) -> list[Atom]:
        return sorted(f for f in self.facts if f[0] == pred)

    def rules_for(self, fact: Atom) -> list[str]:
        return sorted(self.why.get(fact, ()))

    def rule_text(self) -> dict[str, str]:
        return {r.id: r.text for r in self.rules}
