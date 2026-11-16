# nesy-energy: neurosymbolic energy optimisation for industrial plants

Plans the next 24 hours of a plant's shiftable processes and battery so the
electricity bill, demand peak and CO2 are as low as possible, without breaking
any operating rule, and explains every decision in terms of those rules.

It combines three parts:

| Part | What it does | Code |
|---|---|---|
| **Neural** | An MLP forecasts the plant's non-shiftable load from hour, weather and production plan. Its loss also penalises predictions that break known facts (load is never negative; load never falls when production rises). | `nesy_energy/neural.py` |
| **Symbolic** | A Datalog engine applies the plant rule book to facts about the machines and derives the constraints: forbidden hours, conflicts, precedences. It records which rule produced each one. | `nesy_energy/symbolic.py`, `rules.py` |
| **Optimiser** | The derived constraints are compiled into a mixed-integer program (HiGHS via SciPy) that minimises energy cost + demand charge + carbon cost. The result is then re-checked against every constraint by code that does not use the solver, and explained. | `nesy_energy/optimizer.py` |

The symbolic layer also guards the neural output: forecast values outside the
plant's physical range are clamped and reported before the optimiser sees them.

## Quick start

```bash
pip install -e ".[dev]"
nesy-energy run          # plan the demo foundry, print savings and reasons
nesy-energy serve        # dashboard at http://localhost:8000, API docs at /docs
pytest
```

On the built-in demo foundry (6 processes, 500 kWh battery, 300 kWp PV,
time-of-use tariff) the plan costs $1,147 against $1,278 for "start everything
as early as the rules allow", a 10.3% saving, with the peak import down from
597 kW to 440 kW and all 40 constraint checks passing. The demo data is
synthetic, so treat these numbers as an illustration, not a benchmark.

## Using it on your own plant

1. `nesy-energy demo-plant > plant.json` and edit it: machines (`power_kw`,
   `duration_h`, time window, `tags`, `feeder`, `feeds`, `uses`), tariff, battery,
   import limit, staffed and quiet hours, maintenance lock-outs.
2. Export load history as CSV with columns `hour,weekend,temp_c,production,load_kw`
   (`production` is the production rate from 0 to 1).
3. `nesy-energy run --plant plant.json --history history.csv`

For the server, set `NESY_HISTORY_CSV=/path/history.csv`, and POST to `/api/optimise`:

```json
{"plant": { ... }, "temp_c": [24 values], "production": [24 values]}
```

or pass `"base_load": [24 kW values]` to use your own forecast instead of the neural one.

### Adding rules

Rules are Horn clauses. Put site-specific ones in the plant's `extra_rules`:

```
[X1] Forklifts must not charge during quiet hours.
forbidden(forklift_charging,H) :- quiet_hour(H).
```

Heads the optimiser understands: `forbidden(Task,Hour)`, `conflict(A,B)`,
`precedes(A,B)`. Uppercase words are variables. Any tag you put on a task
becomes a fact `tag(task)` you can use in rule bodies. If the rules leave a task
with no feasible window, the answer names the rules responsible.

## Deploy

```bash
docker build -t nesy-energy . && docker run -p 8000:8000 nesy-energy
```

The image listens on `$PORT`, so it runs unchanged on Render (`render.yaml`
included), Railway, Fly.io, Cloud Run or Azure Container Apps. Health check: `/health`.

## Limits

- Hourly steps over a single 24 h horizon; tasks run once, uninterrupted, at fixed power.
- The plan is deterministic: it trusts the point forecast and does not hedge against forecast error.
- The battery model is linear (constant efficiency, no degradation cost).
- Trained and demonstrated on synthetic data only; it has not been validated on a real plant.

MIT licence.
