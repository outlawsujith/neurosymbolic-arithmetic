"""Plant description: the machines, tariff, storage and site limits to optimise."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

H = 24  # planning horizon, hourly steps


@dataclass
class Task:
    """A shiftable process: runs once, uninterrupted, inside [earliest, deadline)."""
    name: str
    power_kw: float
    duration_h: int
    earliest: int = 0
    deadline: int = H
    tags: list[str] = field(default_factory=list)  # e.g. high_inrush, noisy, needs_operator
    feeder: str | None = None                      # electrical feeder it hangs off
    feeds: str | None = None                       # downstream task that needs its output
    uses: list[str] = field(default_factory=list)  # shared resources (crane, cooling loop...)


@dataclass
class Battery:
    capacity_kwh: float
    max_power_kw: float
    efficiency: float = 0.95   # one-way
    initial_soc: float = 0.5   # fraction of capacity
    min_soc: float = 0.1


@dataclass
class Tariff:
    prices: list[float]                 # $/kWh for each hour
    demand_charge: float = 0.0          # $ per kW of the day's peak import
    carbon_price: float = 0.0           # $ per kg CO2
    carbon_intensity: list[float] = field(default_factory=lambda: [0.0] * H)  # kg/kWh


@dataclass
class Plant:
    name: str
    tasks: list[Task]
    tariff: Tariff
    battery: Battery | None = None
    max_import_kw: float = 1e9
    pv_kwp: float = 0.0
    staffed_hours: list[int] = field(default_factory=lambda: list(range(H)))
    quiet_hours: list[int] = field(default_factory=list)
    maintenance: list[list] = field(default_factory=list)   # [task, hour] lock-outs
    exclusive_resources: list[str] = field(default_factory=list)
    base_load_bounds: list[float] = field(default_factory=lambda: [0.0, 1e9])  # physical kW range
    extra_rules: str = ""                                   # extra rule text, same syntax as rules.py

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Plant":
        d = dict(d)
        d["tasks"] = [Task(**t) for t in d["tasks"]]
        d["tariff"] = Tariff(**d["tariff"])
        if d.get("battery"):
            d["battery"] = Battery(**d["battery"])
        plant = Plant(**d)
        plant.validate()
        return plant

    def validate(self) -> None:
        names = [t.name for t in self.tasks]
        if len(set(names)) != len(names):
            raise ValueError("task names must be unique")
        if len(self.tariff.prices) != H or len(self.tariff.carbon_intensity) != H:
            raise ValueError(f"tariff prices and carbon_intensity need {H} hourly values")
        for t in self.tasks:
            if not (0 <= t.earliest < t.deadline <= H) or t.duration_h < 1:
                raise ValueError(f"task {t.name}: bad time window or duration")
            if t.feeds is not None and t.feeds not in names:
                raise ValueError(f"task {t.name} feeds unknown task {t.feeds}")

    def facts(self) -> set[tuple]:
        """Translate the plant into ground facts for the rule engine."""
        f: set[tuple] = set()
        for t in self.tasks:
            f.add(("task", t.name))
            for tag in t.tags:
                f.add((tag, t.name))
            if t.feeder:
                f.add(("feeder", t.name, t.feeder))
            if t.feeds:
                f.add(("feeds", t.name, t.feeds))
            for r in t.uses:
                f.add(("uses", t.name, r))
        for r in self.exclusive_resources:
            f.add(("exclusive", r))
        for h in range(H):
            f.add(("hour", h))
            if h not in self.staffed_hours:
                f.add(("unstaffed", h))
            if h in self.quiet_hours:
                f.add(("quiet_hour", h))
        for name, h in self.maintenance:
            f.add(("maintenance", name, int(h)))
        return f


def demo_plant() -> Plant:
    """A small foundry: melt -> cast -> heat-treat chain plus utilities."""
    prices = [0.07] * 6 + [0.12] * 11 + [0.24] * 4 + [0.12] + [0.07] * 2
    carbon = [0.50] * 6 + [0.42] * 4 + [0.30] * 6 + [0.48] + [0.62] * 4 + [0.55] * 3
    return Plant(
        name="Demo foundry",
        tasks=[
            Task("melt_furnace", 400, 4, tags=["high_inrush"], feeder="F1", feeds="casting"),
            Task("casting", 150, 3, tags=["needs_operator"], feeder="F2", feeds="heat_treat",
                 uses=["crane"]),
            Task("heat_treat", 250, 5, feeder="F2"),
            Task("compressor_bank", 200, 3, tags=["high_inrush", "noisy"], feeder="F1"),
            Task("shredder", 120, 2, deadline=20, tags=["noisy", "needs_operator"], feeder="F3",
                 uses=["crane"]),
            Task("forklift_charging", 60, 4, feeder="F3"),
        ],
        tariff=Tariff(prices, demand_charge=0.50, carbon_price=0.05, carbon_intensity=carbon),
        battery=Battery(capacity_kwh=500, max_power_kw=200),
        max_import_kw=1100,
        pv_kwp=300,
        staffed_hours=list(range(6, 22)),
        quiet_hours=[22, 23, 0, 1, 2, 3, 4, 5],
        maintenance=[["compressor_bank", 12], ["compressor_bank", 13]],
        exclusive_resources=["crane"],
        base_load_bounds=[80.0, 600.0],
    )
