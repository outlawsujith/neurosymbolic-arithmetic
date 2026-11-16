"""HTTP API and dashboard. Run: uvicorn nesy_energy.api:app"""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from . import __version__, pipeline
from .optimizer import InfeasibleError
from .plant import Plant, demo_plant
from .rules import DEFAULT_RULES

app = FastAPI(title="Neurosymbolic Energy Optimiser", version=__version__)
_INDEX = Path(__file__).parent / "static" / "index.html"


class OptimiseRequest(BaseModel):
    plant: dict | None = None             # omit to use the demo plant
    base_load: list[float] | None = None  # 24 hourly kW; omit to use the neural forecast
    temp_c: list[float] | None = None     # tomorrow's weather forecast for the forecaster
    production: list[float] | None = None  # tomorrow's production rate plan, 0..1
    weekend: bool = False
    seed: int = 0


@app.get("/", response_class=HTMLResponse)
def dashboard() -> str:
    return _INDEX.read_text()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.get("/api/demo-plant")
def get_demo_plant() -> dict:
    return demo_plant().to_dict()


@app.get("/api/rules")
def get_rules() -> dict:
    return {"rules": DEFAULT_RULES.strip()}


@app.post("/api/optimise")
def optimise(req: OptimiseRequest) -> dict:
    try:
        plant = Plant.from_dict(req.plant) if req.plant else None
        return pipeline.run(plant, seed=req.seed, base_load=req.base_load, temp_c=req.temp_c,
                            production=req.production, weekend=req.weekend,
                            history_csv=os.environ.get("NESY_HISTORY_CSV"))
    except InfeasibleError as e:
        raise HTTPException(422, detail=f"Infeasible: {e}")
    except (ValueError, TypeError, KeyError) as e:
        raise HTTPException(400, detail=f"Bad request: {e}")
