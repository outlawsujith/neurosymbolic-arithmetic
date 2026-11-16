import numpy as np
import pytest
from fastapi.testclient import TestClient

from nesy_energy import data, pipeline
from nesy_energy.api import app
from nesy_energy.neural import Forecaster
from nesy_energy.optimizer import InfeasibleError
from nesy_energy.plant import Plant, demo_plant
from nesy_energy.symbolic import KnowledgeBase, parse_rules

FLAT = [200.0] * 24


def test_datalog_chains_and_records_provenance():
    rules = parse_rules("""
    [A] base
    precedes(X,Y) :- feeds(X,Y).
    [B] transitive
    precedes(X,Z) :- precedes(X,Y), precedes(Y,Z).
    [C] distinct pairs
    pair(X,Y) :- task(X), task(Y), X != Y.
    """)
    facts = {("feeds", "a", "b"), ("feeds", "b", "c"), ("task", "a"), ("task", "b")}
    kb = KnowledgeBase(rules, facts).infer()
    assert ("precedes", "a", "c") in kb.facts
    assert kb.rules_for(("precedes", "a", "c")) == ["B"]
    assert kb.query("pair") == [("pair", "a", "b"), ("pair", "b", "a")]


def test_demo_plan_respects_every_rule_and_beats_baseline():
    res = pipeline.run(base_load=FLAT)
    assert res["verification"]["all_ok"] and res["verification"]["checked"] > 20
    assert res["savings"]["cost"] > 0
    start = {t["name"]: (t["start"], t["end"]) for t in res["optimised"]["tasks"]}
    assert start["melt_furnace"][1] <= start["casting"][0] <= start["casting"][1] <= start["heat_treat"][0]
    assert start["casting"][0] >= 6 and start["casting"][1] <= 22          # R2 staffed hours
    assert not set(range(*start["compressor_bank"])) & {12, 13}            # R6 lock-out
    assert not set(range(*start["melt_furnace"])) & set(range(*start["compressor_bank"]))  # R1
    assert max(res["optimised"]["grid_kw"]) <= demo_plant().max_import_kw


def test_optimiser_never_worse_than_baseline_on_random_base_loads():
    rng = np.random.default_rng(1)
    for _ in range(3):
        res = pipeline.run(base_load=rng.uniform(100, 400, 24).tolist())
        assert res["verification"]["all_ok"]
        assert res["optimised"]["costs"]["total"] <= res["baseline"]["costs"]["total"] + 1e-6


def test_infeasible_plant_names_the_rules():
    plant = demo_plant()
    plant.staffed_hours = []  # casting needs an operator but nobody is ever on site
    with pytest.raises(InfeasibleError, match="R2"):
        pipeline.run(plant, base_load=FLAT)


def test_extra_rule_changes_the_schedule():
    plant = demo_plant()
    plant.extra_rules = "[X1] Forklifts charge only at night.\nforbidden(forklift_charging,H) :- quiet_hour(H)."
    res = pipeline.run(plant, base_load=FLAT)
    s = next(t for t in res["optimised"]["tasks"] if t["name"] == "forklift_charging")
    assert s["start"] >= 6 and s["end"] <= 22
    assert "X1" in res["verification"]["by_rule"]


def test_forecast_guard_clamps_impossible_values():
    load = FLAT.copy()
    load[3] = -50.0
    res = pipeline.run(base_load=load)
    assert res["forecast"]["base_load_kw"][3] == 80.0
    assert len(res["forecast"]["repairs"]) == 1


def test_forecaster_learns_and_is_monotone_in_production():
    X, y = data.synth_history(days=30, seed=3)
    model = Forecaster(0).fit(X, y, monotone=(data.PRODUCTION,), epochs=80)
    assert model.mae(X, y) < 15
    assert model.monotone_violation_rate(X, data.PRODUCTION) < 0.02
    assert model.mae(X, y) < np.abs(y - y.mean()).mean() / 2


def test_plant_round_trips_through_json():
    d = demo_plant().to_dict()
    assert Plant.from_dict(d).to_dict() == d
    d["tariff"]["prices"] = [0.1] * 5
    with pytest.raises(ValueError):
        Plant.from_dict(d)


def test_api():
    c = TestClient(app)
    assert c.get("/health").json()["status"] == "ok"
    assert "<html" in c.get("/").text.lower()
    plant = c.get("/api/demo-plant").json()
    r = c.post("/api/optimise", json={"plant": plant, "base_load": FLAT})
    assert r.status_code == 200 and r.json()["verification"]["all_ok"]
    plant["staffed_hours"] = []
    assert c.post("/api/optimise", json={"plant": plant, "base_load": FLAT}).status_code == 422
    assert c.post("/api/optimise", json={"base_load": [1.0]}).status_code == 400
