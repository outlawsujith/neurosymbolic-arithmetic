"""End to end: neural forecast -> symbolic guard -> rule inference -> optimise -> verify -> explain."""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from . import data
from .neural import Forecaster
from .optimizer import baseline, explain, optimise, verify
from .plant import H, Plant, demo_plant
from .rules import DEFAULT_RULES
from .symbolic import KnowledgeBase, parse_rules


@lru_cache(maxsize=4)
def trained_forecaster(seed: int = 0, history_csv: str | None = None, knowledge_weight: float = 5.0):
    """Train on plant history; hold out the last week to report accuracy."""
    X, y = data.load_history_csv(history_csv) if history_csv else data.synth_history(seed=seed)
    cut = max(len(X) - 7 * H, len(X) // 2)
    model = Forecaster(seed).fit(X[:cut], y[:cut], monotone=(data.PRODUCTION,),
                                 knowledge_weight=knowledge_weight)
    metrics = {
        "holdout_mae_kw": round(model.mae(X[cut:], y[cut:]), 2),
        "monotonicity_violation_rate": round(model.monotone_violation_rate(X, data.PRODUCTION), 4),
        "training_hours": int(cut),
        "source": history_csv or "synthetic",
    }
    return model, metrics


def guard_forecast(forecast: np.ndarray, bounds) -> tuple[np.ndarray, list[str]]:
    """Symbolic check on the neural output: clamp physically impossible values and say so."""
    lo, hi = bounds
    fixed = np.clip(forecast, lo, hi)
    notes = [f"{h:02d}:00 forecast {forecast[h]:.0f} kW outside physical range [{lo:.0f}, {hi:.0f}] kW; "
             f"clamped to {fixed[h]:.0f} kW" for h in range(H) if fixed[h] != forecast[h]]
    return fixed, notes


def build_kb(plant: Plant) -> KnowledgeBase:
    return KnowledgeBase(parse_rules(DEFAULT_RULES + "\n" + plant.extra_rules), plant.facts()).infer()


def run(plant: Plant | None = None, seed: int = 0, base_load=None, temp_c=None, production=None,
        weekend: bool = False, history_csv: str | None = None) -> dict:
    """Plan tomorrow. Supply `base_load` to skip the forecaster, or `temp_c` + `production`
    (24 values each) to forecast from your own weather and production plan."""
    plant = plant or demo_plant()
    plant.validate()
    forecast_info: dict = {"source": "provided"}
    if base_load is None:
        model, metrics = trained_forecaster(seed, history_csv)
        t_demo, p_demo = data.day_inputs(np.random.default_rng(seed + 1000), weekend)
        temp = np.asarray(temp_c, float) if temp_c is not None else t_demo
        prod = np.asarray(production, float) if production is not None else p_demo
        if temp.shape != (H,) or prod.shape != (H,):
            raise ValueError(f"temp_c and production need {H} hourly values")
        raw = model.predict(data.features(np.arange(H), float(weekend), temp, prod))
        forecast_info = {"source": "neural", "model": metrics,
                         "temp_c": temp.round(1).tolist(), "production": prod.round(2).tolist()}
    else:
        raw = np.asarray(base_load, float)
        if raw.shape != (H,):
            raise ValueError(f"base_load needs {H} hourly values")
    base, repairs = guard_forecast(raw, plant.base_load_bounds)
    pv = data.pv_profile(plant.pv_kwp)

    kb = build_kb(plant)
    opt = optimise(plant, kb, base, pv)
    ref = baseline(plant, kb, base, pv)
    checks = verify(plant, kb, opt, base, pv)
    saving = ref.costs["total"] - opt.costs["total"]

    def sched(s):
        return {"tasks": [{"name": t.name, "start": s.starts[t.name], "end": s.starts[t.name] + t.duration_h,
                           "power_kw": t.power_kw} for t in plant.tasks],
                "grid_kw": s.grid.round(1).tolist(), "charge_kw": s.charge.round(1).tolist(),
                "discharge_kw": s.discharge.round(1).tolist(), "soc_kwh": s.soc.round(1).tolist(),
                "costs": {k: round(v, 2) for k, v in s.costs.items()}}

    derived = [{"fact": f"{f[0]}({', '.join(map(str, f[1:]))})", "rules": kb.rules_for(f)}
               for f in sorted(kb.why, key=str)]
    return {
        "plant": plant.name,
        "forecast": {**forecast_info, "base_load_kw": base.round(1).tolist(), "repairs": repairs},
        "pv_kw": pv.round(1).tolist(),
        "prices": plant.tariff.prices,
        "optimised": sched(opt),
        "baseline": sched(ref),
        "savings": {"cost": round(saving, 2),
                    "percent": round(100 * saving / ref.costs["total"], 1) if ref.costs["total"] else 0.0,
                    "co2_kg": round(ref.costs["co2_kg"] - opt.costs["co2_kg"], 1),
                    "peak_kw": round(ref.costs["peak_kw"] - opt.costs["peak_kw"], 1)},
        "baseline_within_import_limit": bool(ref.grid.max() <= plant.max_import_kw + 1e-6),
        "verification": {"all_ok": all(c["ok"] for c in checks), "checked": len(checks),
                         "failed": [c for c in checks if not c["ok"]],
                         "by_rule": _by_rule(checks)},
        "explanations": explain(plant, kb, opt, ref),
        "rules": kb.rule_text(),
        "derived_facts": derived,
    }


def _by_rule(checks: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for c in checks:
        r = out.setdefault(c["rule"], {"checked": 0, "passed": 0})
        r["checked"] += 1
        r["passed"] += c["ok"]
    return out
