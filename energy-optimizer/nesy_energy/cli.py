"""Command line: nesy-energy run | serve | demo-plant."""
from __future__ import annotations

import argparse
import json

from . import pipeline
from .optimizer import InfeasibleError
from .plant import Plant, demo_plant


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="nesy-energy", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="plan the next day and print the result")
    r.add_argument("--plant", help="plant JSON file (default: demo foundry)")
    r.add_argument("--history", help="CSV of real history: hour,weekend,temp_c,production,load_kw")
    r.add_argument("--weekend", action="store_true")
    r.add_argument("--json", action="store_true", help="print the full result as JSON")
    s = sub.add_parser("serve", help="start the API and dashboard")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8000)
    sub.add_parser("demo-plant", help="print the demo plant JSON to use as a template")
    a = p.parse_args(argv)

    if a.cmd == "demo-plant":
        print(json.dumps(demo_plant().to_dict(), indent=2))
        return 0
    if a.cmd == "serve":
        import uvicorn
        uvicorn.run("nesy_energy.api:app", host=a.host, port=a.port)
        return 0
    plant = Plant.from_dict(json.load(open(a.plant))) if a.plant else None
    try:
        res = pipeline.run(plant, weekend=a.weekend, history_csv=a.history)
    except InfeasibleError as e:
        print(f"Infeasible: {e}")
        return 1
    if a.json:
        print(json.dumps(res, indent=2))
        return 0
    o, b, sv = res["optimised"]["costs"], res["baseline"]["costs"], res["savings"]
    print(f"{res['plant']}: next-day plan")
    if "model" in res["forecast"]:
        m = res["forecast"]["model"]
        print(f"  forecaster: hold-out MAE {m['holdout_mae_kw']} kW ({m['source']} history)")
    for note in res["forecast"]["repairs"]:
        print(f"  forecast repaired: {note}")
    print(f"  cost   ${b['total']:.2f} -> ${o['total']:.2f}  (saves ${sv['cost']:.2f}, {sv['percent']}%)")
    print(f"  peak   {b['peak_kw']:.0f} kW -> {o['peak_kw']:.0f} kW")
    print(f"  CO2    {b['co2_kg']:.0f} kg -> {o['co2_kg']:.0f} kg")
    v = res["verification"]
    print(f"  rules  {v['checked'] - len(v['failed'])}/{v['checked']} constraint checks pass")
    print("\nWhy this schedule:")
    for line in res["explanations"]:
        print(f"  - {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
