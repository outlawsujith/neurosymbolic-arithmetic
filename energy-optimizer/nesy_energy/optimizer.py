"""Compile symbolic constraints into a mixed-integer program, solve, verify, explain."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix

from .plant import H, Plant
from .symbolic import KnowledgeBase


class InfeasibleError(Exception):
    """No schedule satisfies the rules; the message says which rules are responsible."""


@dataclass
class Schedule:
    starts: dict[str, int]
    charge: np.ndarray
    discharge: np.ndarray
    soc: np.ndarray       # kWh, length H + 1
    grid: np.ndarray      # kW imported each hour
    costs: dict[str, float]


class _Model:
    def __init__(self):
        self.lb, self.ub, self.integer, self.cost = [], [], [], []
        self.rows, self.rlb, self.rub = [], [], []

    def var(self, lb=0.0, ub=np.inf, integer=False, cost=0.0) -> int:
        self.lb.append(lb); self.ub.append(ub); self.integer.append(int(integer)); self.cost.append(cost)
        return len(self.lb) - 1

    def add(self, coeffs: dict[int, float], lb=-np.inf, ub=np.inf) -> None:
        self.rows.append(coeffs); self.rlb.append(lb); self.rub.append(ub)

    def solve(self):
        data, ri, ci = [], [], []
        for r, row in enumerate(self.rows):
            for c, val in row.items():
                data.append(val); ri.append(r); ci.append(c)
        A = csr_matrix((data, (ri, ci)), shape=(len(self.rows), len(self.lb)))
        return milp(np.array(self.cost), constraints=LinearConstraint(A, self.rlb, self.rub),
                    integrality=np.array(self.integer), bounds=Bounds(self.lb, self.ub),
                    options={"time_limit": 30.0})


def forbidden_hours(kb: KnowledgeBase) -> dict[str, dict[int, list[str]]]:
    """task -> hour -> rule ids that forbid it."""
    out: dict[str, dict[int, list[str]]] = {}
    for fact in kb.query("forbidden"):
        out.setdefault(fact[1], {})[fact[2]] = kb.rules_for(fact)
    return out


def allowed_starts(task, forbidden: dict[int, list[str]]) -> list[int]:
    return [s for s in range(task.earliest, task.deadline - task.duration_h + 1)
            if not any(h in forbidden for h in range(s, s + task.duration_h))]


def cost_of(plant: Plant, grid: np.ndarray) -> dict[str, float]:
    t = plant.tariff
    energy = float(np.dot(t.prices, grid))
    demand = float(t.demand_charge * grid.max())
    co2 = float(np.dot(t.carbon_intensity, grid))
    carbon = t.carbon_price * co2
    return {"energy": energy, "demand": demand, "carbon": carbon,
            "total": energy + demand + carbon, "co2_kg": co2, "peak_kw": float(grid.max())}


def task_load(plant: Plant, starts: dict[str, int]) -> np.ndarray:
    load = np.zeros(H)
    for t in plant.tasks:
        load[starts[t.name]:starts[t.name] + t.duration_h] += t.power_kw
    return load


def optimise(plant: Plant, kb: KnowledgeBase, base: np.ndarray, pv: np.ndarray) -> Schedule:
    forb = forbidden_hours(kb)
    tariff = plant.tariff
    price = np.array(tariff.prices) + tariff.carbon_price * np.array(tariff.carbon_intensity)
    m = _Model()

    # One binary per (task, feasible start). Hours forbidden by rules never get a variable.
    x: dict[str, dict[int, int]] = {}
    for t in plant.tasks:
        starts = allowed_starts(t, forb.get(t.name, {}))
        if not starts:
            rules = sorted({r for ids in forb.get(t.name, {}).values() for r in ids})
            why = f"rules {', '.join(rules)} forbid" if rules else "its time window forbids"
            raise InfeasibleError(f"'{t.name}' has no feasible {t.duration_h} h window: {why} every start.")
        x[t.name] = {s: m.var(0, 1, integer=True, cost=1e-6 * s) for s in starts}
        m.add({v: 1 for v in x[t.name].values()}, 1, 1)
    dur = {t.name: t.duration_h for t in plant.tasks}

    def running(name: str, h: int) -> dict[int, float]:
        return {v: 1.0 for s, v in x[name].items() if s <= h < s + dur[name]}

    for _, a, b in kb.query("conflict"):
        if a < b:  # the relation is symmetric; one constraint per pair
            for h in range(H):
                row = {**running(a, h), **running(b, h)}
                if len(row) > 1:
                    m.add(row, ub=1)
    for _, a, b in kb.query("precedes"):
        row = {v: float(s) for s, v in x[b].items()}
        row.update({v: -float(s + dur[a]) for s, v in x[a].items()})
        m.add(row, lb=0)

    bat = plant.battery
    cap = bat.capacity_kwh if bat else 0.0
    pmax = bat.max_power_kw if bat else 0.0
    eta = bat.efficiency if bat else 1.0
    soc0 = cap * (bat.initial_soc if bat else 0.0)
    c = [m.var(0, pmax) for _ in range(H)]
    d = [m.var(0, pmax) for _ in range(H)]
    soc = [m.var(cap * (bat.min_soc if bat else 0.0), cap) for _ in range(H + 1)]
    m.add({soc[0]: 1}, soc0, soc0)
    m.add({soc[H]: 1}, lb=soc0)  # do not "save" money by draining the battery
    grid = [m.var(0, plant.max_import_kw, cost=price[h]) for h in range(H)]
    curtail = [m.var(0, max(float(pv[h]), 0.0)) for h in range(H)]
    peak = m.var(cost=tariff.demand_charge)
    for h in range(H):
        m.add({soc[h + 1]: 1, soc[h]: -1, c[h]: -eta, d[h]: 1 / eta}, 0, 0)
        row = {grid[h]: 1.0, c[h]: -1.0, d[h]: 1.0, curtail[h]: -1.0}
        for t in plant.tasks:
            for v in running(t.name, h):
                row[v] = -t.power_kw
        m.add(row, base[h] - pv[h], base[h] - pv[h])  # power balance
        m.add({peak: 1, grid[h]: -1}, lb=0)

    res = m.solve()
    if res.x is None:
        raise InfeasibleError(
            "No schedule satisfies all rules within the import limit "
            f"({plant.max_import_kw:.0f} kW). Relax a time window, add storage or raise the limit.")
    starts = {name: next(s for s, v in vs.items() if res.x[v] > 0.5) for name, vs in x.items()}
    g = np.clip(res.x[grid], 0, None)
    return Schedule(starts, res.x[c], res.x[d], res.x[soc], g, cost_of(plant, g))


def baseline(plant: Plant, kb: KnowledgeBase, base: np.ndarray, pv: np.ndarray) -> Schedule:
    """Business as usual: every process starts as early as the rules allow; battery idle."""
    forb = forbidden_hours(kb)
    conflicts = {(a, b) for _, a, b in kb.query("conflict")}
    preds: dict[str, list[str]] = {}
    for _, a, b in kb.query("precedes"):
        preds.setdefault(b, []).append(a)
    by_name = {t.name: t for t in plant.tasks}
    starts: dict[str, int] = {}
    pending = list(plant.tasks)
    while pending:
        t = next((t for t in pending if all(p in starts for p in preds.get(t.name, []))), pending[0])
        pending.remove(t)
        ready = max([starts[p] + by_name[p].duration_h for p in preds.get(t.name, []) if p in starts],
                    default=0)
        options = allowed_starts(t, forb.get(t.name, {}))

        def clash(s):
            return any((t.name, o) in conflicts and s < so + by_name[o].duration_h
                       and so < s + t.duration_h for o, so in starts.items())
        ok = [s for s in options if s >= ready and not clash(s)]
        starts[t.name] = (ok or options)[0]
    grid = np.clip(base + task_load(plant, starts) - pv, 0, None)
    soc0 = plant.battery.capacity_kwh * plant.battery.initial_soc if plant.battery else 0.0
    return Schedule(starts, np.zeros(H), np.zeros(H), np.full(H + 1, soc0), grid, cost_of(plant, grid))


def verify(plant: Plant, kb: KnowledgeBase, s: Schedule, base: np.ndarray, pv: np.ndarray) -> list[dict]:
    """Independently re-check a schedule against every derived constraint (no solver involved)."""
    checks: list[dict] = []
    by_name = {t.name: t for t in plant.tasks}
    span = {n: range(st, st + by_name[n].duration_h) for n, st in s.starts.items()}

    def add(rule, ok, detail):
        checks.append({"rule": rule, "ok": bool(ok), "detail": detail})

    for t in plant.tasks:
        st = s.starts[t.name]
        add("WINDOW", t.earliest <= st and st + t.duration_h <= t.deadline,
            f"{t.name} runs {st:02d}:00-{st + t.duration_h:02d}:00 inside its window")
    for fact in kb.query("forbidden"):
        _, name, h = fact
        if name in span:
            add("+".join(kb.rules_for(fact)), h not in span[name], f"{name} is off at {h:02d}:00")
    for fact in kb.query("conflict"):
        _, a, b = fact
        if a < b:
            add("+".join(kb.rules_for(fact)), not set(span[a]) & set(span[b]),
                f"{a} and {b} never overlap")
    for fact in kb.query("precedes"):
        _, a, b = fact
        add("+".join(kb.rules_for(fact)), span[a].stop <= span[b].start, f"{a} finishes before {b} starts")
    add("IMPORT", s.grid.max() <= plant.max_import_kw + 1e-4,
        f"peak import {s.grid.max():.0f} kW <= limit {plant.max_import_kw:.0f} kW")
    net = base + task_load(plant, s.starts) + s.charge - s.discharge - pv
    add("BALANCE", np.all(s.grid >= net - 1e-4), "grid import covers demand in every hour")
    if plant.battery:
        b = plant.battery
        ok = (s.soc.min() >= b.min_soc * b.capacity_kwh - 1e-4 and s.soc.max() <= b.capacity_kwh + 1e-4
              and s.soc[-1] >= s.soc[0] - 1e-4)
        add("BATTERY", ok, "state of charge stays in range and ends no lower than it started")
    return checks


def explain(plant: Plant, kb: KnowledgeBase, opt: Schedule, ref: Schedule) -> list[str]:
    """Plain-language reasons for the schedule, each tied to rules or prices."""
    forb = forbidden_hours(kb)
    text = kb.rule_text()
    price = np.array(plant.tariff.prices)
    out = []
    for t in plant.tasks:
        s, r = opt.starts[t.name], ref.starts[t.name]
        avg = lambda st: price[st:st + t.duration_h].mean()
        line = f"{t.name}: runs {s:02d}:00-{s + t.duration_h:02d}:00"
        if s != r:
            line += f" instead of the business-as-usual {r:02d}:00"
            if abs(avg(s) - avg(r)) > 1e-9:
                line += f" (average tariff ${avg(s):.3f}/kWh vs ${avg(r):.3f}/kWh)"
            else:
                line += " (same tariff; moved to flatten the demand peak or make room for other processes)"
        rules = sorted({rid for ids in forb.get(t.name, {}).values() for rid in ids})
        if rules:
            n = len(forb[t.name])
            line += f". {n} of {H} hours were ruled out by " + "; ".join(f"{r_}: {text[r_].rstrip('.')}" for r_ in rules)
        partners = sorted({(b, rid) for f in kb.query("conflict") if f[1] == t.name
                           for b, rid in [(f[2], "+".join(kb.rules_for(f)))]})
        if partners:
            line += ". Kept apart from " + ", ".join(f"{b} ({rid})" for b, rid in partners)
        after = sorted(f[1] for f in kb.query("precedes") if f[2] == t.name)
        if after:
            line += ". Must follow " + ", ".join(after) + " (R4/R5)"
        out.append(line + ".")
    if plant.battery and opt.discharge.sum() > 1:
        ch = [h for h in range(H) if opt.charge[h] > 1]
        dis = [h for h in range(H) if opt.discharge[h] > 1]
        out.append(f"battery: charges {opt.charge.sum():.0f} kWh in hours {ch} and discharges "
                   f"{opt.discharge.sum():.0f} kWh in hours {dis}, moving energy to expensive hours "
                   "and shaving the demand peak.")
    out.append(f"peak import: {opt.costs['peak_kw']:.0f} kW versus {ref.costs['peak_kw']:.0f} kW "
               f"business as usual (site limit {plant.max_import_kw:.0f} kW).")
    return out
