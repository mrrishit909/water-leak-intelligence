"""Public API (modular monolith). Blueprint services map to: asset-registry (districts, pipes, failure history),
telemetry-ingestion, leak-detection (classify and locate), hydraulic-model (what-if on the linearised network),
work-order-service, risk-model (pipe failure) and capital-planning.

    uvicorn wa.api:app          python -m core.jobs wa.api      # the worker
"""
import datetime
import uuid

import numpy as np
from fastapi import Depends, Header, Query
from fastapi.encoders import jsonable_encoder
from psycopg.types.json import Jsonb
from pydantic import BaseModel, Field
from sklearn.ensemble import HistGradientBoostingClassifier

from core import audit, db, jobs
from core.app import Ctx, Problem, create_app, run

from . import engine, world

READ = {"network:read", "jobs:read"}
OPERATE = READ | {"telemetry:write", "leaks:detect", "hydraulic:simulate", "work_orders:write"}
PERMISSIONS = {"viewer": READ, "analyst": OPERATE, "planner": OPERATE | {"network:load", "risk:train", "capital:plan", "audit:read"}}
app = create_app("water-leak-intelligence", PERMISSIONS)
auth = app.state.auth
IdemKey = Header(None, alias="Idempotency-Key")
T0 = datetime.datetime(2026, 9, 14, tzinfo=datetime.UTC)
STEP = datetime.timedelta(minutes=world.STEP_MIN)
SEARCH_HOPS = 3                         # a crew surveys every pipe within this many junctions of the best candidate
NEXT_YEAR = world.HISTORY_YEARS[-1] + 1


def districts(c):
    rows = c.execute("SELECT id, ordinal, name, model_seed FROM districts ORDER BY ordinal").fetchall()
    if not rows:
        raise Problem(409, "no_network", "load a network first: POST /v1/network:load")
    return rows, world.build(rows[0]["model_seed"], len(rows))


def district(c, ordinal):
    rows, w = districts(c)
    if ordinal >= len(rows):
        raise Problem(404, "district_not_found")
    return rows[ordinal], w["districts"][ordinal]


def pipe_id(t, d, k):
    return uuid.uuid5(t, f"pipe-{d}-{k}")


# ---------------------------------------------------------------- asset-registry

class LoadIn(BaseModel):
    seed: int = Field(17, ge=0, le=10 ** 6, description="Which synthetic utility to generate. The demo's planted leak, drifting sensor and demand spike are in every seed")


@app.post("/v1/network:load", status_code=202, tags=["network"], summary="(+) Load a synthetic utility: districts, pipes, ten years of failures, sensors and two weeks of telemetry")
def load_network(body: LoadIn, ctx: Ctx = Depends(auth("network:load")), idem: str | None = IdemKey):
    def work(c):
        if c.execute("SELECT 1 FROM districts LIMIT 1").fetchone():
            raise Problem(409, "network_exists", "this utility already has a network")
        return 202, jobs.enqueue(c, ctx, "network.load", body.model_dump())
    return run(ctx, idem, body, work)


@jobs.handler("network.load")
def load_job(c, job):
    t, w = job["tenant_id"], world.build(job["payload"]["seed"])
    ds = [uuid.uuid5(t, f"district-{d['index']}") for d in w["districts"]]
    db.load(c, "districts", ["id", "tenant_id", "ordinal", "name", "junctions", "model_seed", "telemetry_from"], [(ds[i], t, i, f"DMA-{i + 1:02d}", world.N, w["seed"], T0) for i in range(len(ds))])
    db.load(c, "assets", ["id", "tenant_id", "district_id", "ordinal", "material", "installed_year", "diameter_mm", "length_m", "soil", "a_node", "b_node"],
            [(pipe_id(t, d["index"], k), t, ds[d["index"]], k, p["material"], p["installed"], p["diameter_mm"], p["length_m"], p["soil"], p["a"], p["b"]) for d in w["districts"] for k, p in enumerate(d["pipes"])])
    db.load(c, "asset_failures", ["id", "tenant_id", "asset_id", "year"], [(uuid.uuid7(), t, pipe_id(t, d, k), year) for d, k, year in w["failures"]])
    sensor = lambda d, k: uuid.uuid5(t, f"sensor-{d}-{k}")          # noqa: E731   k = 0..9 pressure, 10 = the inlet flow meter
    db.load(c, "sensors", ["id", "tenant_id", "district_id", "kind", "ordinal", "node"],
            [(sensor(d["index"], k), t, ds[d["index"]], "pressure" if k < world.K_SENSORS else "flow", k, d["sensors"][k] if k < world.K_SENSORS else d["inlet"]) for d in w["districts"] for k in range(world.K_SENSORS + 1)])

    def readings():
        for d in w["districts"]:
            pressure, inflow = world.scenario_telemetry(w, d["index"])
            ids = [sensor(d["index"], k) for k in range(world.K_SENSORS + 1)]
            for s in range(world.T):
                ts = T0 + s * STEP
                for k in range(world.K_SENSORS):
                    yield ids[k], ts, float(pressure[s, k]), t, s
                yield ids[-1], ts, float(inflow[s]), t, s
    db.load(c, "telemetry", ["sensor_id", "ts", "value", "tenant_id", "step"], readings())
    pipes = sum(len(d["pipes"]) for d in w["districts"])
    out = {"districts": len(ds), "junctions": len(ds) * world.N, "pipes": pipes, "network_km": round(world.pipe_length_km(w), 1), "failures_on_record": len(w["failures"]), "failure_years": [world.HISTORY_YEARS[0], world.HISTORY_YEARS[-1]],
           "sensors": len(ds) * (world.K_SENSORS + 1), "readings": len(ds) * (world.K_SENSORS + 1) * world.T, "telemetry_days": world.DAYS, "step_minutes": world.STEP_MIN,
           "by_material": {m: sum(p["material"] == m for d in w["districts"] for p in d["pipes"]) for m in world.MATERIALS}}
    audit.record(c, Ctx(t, uuid.UUID(job["payload"]["actor_id"]), "worker", "system"), "network.loaded", "network", ds[0], {k: out[k] for k in ("districts", "pipes", "sensors", "readings")})
    return out


@app.get("/v1/network", tags=["network"], summary="(+) Every district: size, what was detected there, average failure risk")
def network(ctx: Ctx = Depends(auth("network:read"))):
    with db.tx(ctx.tenant_id) as c:
        districts(c)
        rows = c.execute("""SELECT d.ordinal, d.name, count(a.id) AS pipes, round(sum(a.length_m)::numeric / 1000, 1) AS km, round(avg(a.risk)::numeric, 5) AS mean_risk,
                                   (SELECT e.kind FROM leak_events e WHERE e.district_id = d.id ORDER BY e.kind LIMIT 1) AS event
                              FROM districts d JOIN assets a ON a.district_id = d.id GROUP BY d.id ORDER BY d.ordinal""").fetchall()
    return jsonable_encoder({"districts": rows, "pipes": sum(r["pipes"] for r in rows), "km": float(sum(r["km"] for r in rows)), "grid": [world.NX, world.NY], "sensors_per_district": world.K_SENSORS + 1})


@app.get("/v1/assets", tags=["network"], summary="(+) Pipes ranked by chance of failing next year")
def assets(limit: int = Query(20, ge=1, le=200), ctx: Ctx = Depends(auth("network:read"))):
    with db.tx(ctx.tenant_id) as c:
        rows = c.execute("""SELECT a.id, d.name AS district, a.material, a.installed_year, a.diameter_mm, a.soil, a.risk, a.risk_rank, (SELECT count(*) FROM asset_failures f WHERE f.asset_id = a.id) AS breaks
                              FROM assets a JOIN districts d ON d.id = a.district_id WHERE a.risk_rank IS NOT NULL ORDER BY a.risk_rank LIMIT %s""", [limit]).fetchall()
    return jsonable_encoder({"items": rows})


@app.get("/v1/assets/{asset_id}/risk", tags=["risk"], summary="One pipe: what is known about it, its chance of failing next year and where that ranks")
def asset_risk(asset_id: uuid.UUID, ctx: Ctx = Depends(auth("network:read"))):
    with db.tx(ctx.tenant_id) as c:
        a = c.execute("SELECT a.*, d.name AS district FROM assets a JOIN districts d ON d.id = a.district_id WHERE a.id = %s", [asset_id]).fetchone()
        if not a:
            raise Problem(404, "asset_not_found")
        if a["risk"] is None:
            raise Problem(409, "no_risk_scores", "train the failure model first: POST /v1/risk:train")
        years = [r["year"] for r in c.execute("SELECT year FROM asset_failures WHERE asset_id = %s ORDER BY year", [asset_id])]
        stats = c.execute("SELECT count(*) AS n, avg(risk) AS mean FROM assets").fetchone()
    return jsonable_encoder({"asset_id": a["id"], "district": a["district"], "material": a["material"], "installed_year": a["installed_year"], "age_years": NEXT_YEAR - a["installed_year"], "diameter_mm": a["diameter_mm"], "soil_corrosivity": a["soil"],
                             "length_m": a["length_m"], "failure_years": years, "chance_of_failing_next_year": round(a["risk"], 4), "rank": a["risk_rank"], "of": stats["n"], "times_network_average": round(a["risk"] / stats["mean"], 1),
                             "replacement_cost": round(world.unit_cost(pipe(a)) * a["length_m"]), "cost_of_one_failure": round(world.consequence(pipe(a))), "model": engine.VERSION})


def pipe(a):
    return {"material": a["material"], "installed": a["installed_year"], "diameter_mm": a["diameter_mm"], "soil": a["soil"], "length_m": a["length_m"]}


# ---------------------------------------------------------------- telemetry-ingestion

class Reading(BaseModel):
    sensor_id: uuid.UUID
    ts: datetime.datetime
    value: float = Field(ge=-10, le=100000, description="Metres of head for a pressure sensor, L/s for a flow meter")


class TelemetryIn(BaseModel):
    readings: list[Reading] = Field(min_length=1, max_length=5000)


@app.post("/v1/telemetry", status_code=202, tags=["telemetry"], summary="Ingest sensor readings. A reading already stored for that sensor and time is ignored, so a resend is harmless")
def telemetry(body: TelemetryIn, ctx: Ctx = Depends(auth("telemetry:write")), idem: str | None = IdemKey):
    def work(c):
        ids = {r.sensor_id for r in body.readings}
        known = {r["id"] for r in c.execute("SELECT id FROM sensors WHERE id = ANY(%s)", [list(ids)])}       # under row-level security: another utility's sensor is unknown here
        if ids - known:
            raise Problem(404, "sensor_not_found", str(sorted(ids - known)[0]))
        rows = []
        for r in body.readings:
            if r.ts.tzinfo is None:
                raise Problem(422, "timezone_required", "timestamps must carry a UTC offset")
            step, rem = divmod(r.ts - T0, STEP)
            if rem:
                raise Problem(422, "off_grid", f"readings are expected every {world.STEP_MIN} minutes on the quarter hour")
            rows.append((r.sensor_id, r.ts, r.value, ctx.tenant_id, step))
        stored = sum(bool(c.execute("INSERT INTO telemetry (sensor_id, ts, value, tenant_id, step) VALUES (%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING RETURNING step", row).fetchone()) for row in rows)
        return 202, {"received": len(rows), "stored": stored, "duplicates_ignored": len(rows) - stored}
    return run(ctx, idem, body, work)


def series(c, district_id):
    """The district's fortnight from storage. -> (pressure [T, K], inflow [T]) with NaN where a reading is missing"""
    p, f = np.full((world.T, world.K_SENSORS), np.nan), np.full(world.T, np.nan)
    for r in c.execute("SELECT s.kind, s.ordinal, t.step, t.value FROM telemetry t JOIN sensors s ON s.id = t.sensor_id WHERE s.district_id = %s AND t.step BETWEEN 0 AND %s", [district_id, world.T - 1]):
        if r["kind"] == "flow":
            f[r["step"]] = r["value"]
        else:
            p[r["step"], r["ordinal"]] = r["value"]
    return p, f


# ---------------------------------------------------------------- leak-detection

@app.post("/v1/leaks:detect", status_code=202, tags=["leaks"], summary="(+) Go through every district's telemetry: find what is unusual, and say whether it is a leak, a faulty sensor or just demand")
def detect(ctx: Ctx = Depends(auth("leaks:detect")), idem: str | None = IdemKey):
    def work(c):
        districts(c)
        return 202, jobs.enqueue(c, ctx, "leaks.detect", {})
    return run(ctx, idem, {}, work)


@jobs.handler("leaks.detect")
def detect_job(c, job):
    t = job["tenant_id"]
    rows, w = districts(c)
    events, skipped = [], 0
    for row, d in zip(rows, w["districts"]):
        p, f = series(c, row["id"])
        if np.isnan(p).any() or np.isnan(f).any():              # ponytail: a district with any gap is skipped, not interpolated. Fill short gaps when real telemetry arrives.
            skipped += 1
            continue
        a = engine.analyse(p, f, d)
        if a["kind"] == "normal":
            continue
        eid = uuid.uuid5(t, f"event-{d['index']}-{a['kind']}")
        c.execute("""INSERT INTO leak_events (id, tenant_id, district_id, kind, onset_day, estimated_lps, probability, candidates, evidence, model) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                     ON CONFLICT (district_id, kind) DO UPDATE SET onset_day = EXCLUDED.onset_day, estimated_lps = EXCLUDED.estimated_lps, probability = EXCLUDED.probability, candidates = EXCLUDED.candidates, evidence = EXCLUDED.evidence, detected_at = now()""",
                  [eid, t, row["id"], a["kind"], a.get("onset_day"), a.get("flow_rise_lps"), a.get("probability"), Jsonb(a.get("candidates", [])), Jsonb(a), engine.VERSION])
        events.append({"id": eid, "district": row["name"], "kind": a["kind"], "onset_day": a.get("onset_day"), "flow_rise_lps": a.get("flow_rise_lps"), "sensor": a.get("sensor"), "reason": a.get("reason") or
                       f"night inflow rose by {a['flow_rise_lps']} L/s and stayed up for {a['nights_elevated']} nights, and the pressure drops across the sensors fit a single leak at one junction ({round(100 * a['candidates'][0]['fit'])}% of the pattern explained)"})
    c.execute("DELETE FROM leak_events e WHERE detected_at < now() AND status = 'open'")            # whatever an earlier run saw that this one no longer does
    name = lambda i: f"DMA-{i + 1:02d}"          # noqa: E731
    out = {"districts_checked": len(rows) - skipped, "skipped_for_missing_data": skipped, "normal": len(rows) - skipped - len(events), "events": sorted(events, key=lambda e: e["kind"]),
           "ids": {e["kind"]: e["id"] for e in events}, "model": engine.VERSION,
           "what_the_simulation_hid": {"leak": f"{name(world.LEAK['dma'])}, {world.LEAK['lps']} L/s from 03:00 on day {world.LEAK['start_step'] // world.PER_DAY}", "sensor_drift": f"{name(world.DRIFT['dma'])}, pressure sensor {world.DRIFT['sensor']} from day {world.DRIFT['start_step'] // world.PER_DAY}",
                                       "demand_spike": f"{name(world.SPIKE['dma'])}, one night of {world.SPIKE['lps']} L/s extra on day {world.SPIKE['start_step'] // world.PER_DAY}"}}
    audit.record(c, Ctx(t, uuid.UUID(job["payload"]["actor_id"]), "worker", "system"), "leaks.detected", "network", rows[0]["id"], {"districts_checked": out["districts_checked"], "events": {e["district"]: e["kind"] for e in events}})
    return out


@app.get("/v1/leaks", tags=["leaks"], summary="Detected events, most serious first")
def leaks(kind: str | None = Query(None, pattern="^(leak|sensor_fault|transient_demand|unexplained)$"), ctx: Ctx = Depends(auth("network:read"))):
    with db.tx(ctx.tenant_id) as c:
        rows = c.execute("""SELECT e.id, d.name AS district, e.kind, e.status, e.onset_day, e.estimated_lps, e.probability, e.detected_at FROM leak_events e JOIN districts d ON d.id = e.district_id
                             WHERE %s::text IS NULL OR e.kind = %s ORDER BY (e.kind = 'leak') DESC, e.estimated_lps DESC NULLS LAST, d.ordinal""", [kind, kind]).fetchall()
    return jsonable_encoder({"items": rows})


def event(c, event_id):
    e = c.execute("SELECT e.*, d.name AS district, d.ordinal FROM leak_events e JOIN districts d ON d.id = e.district_id WHERE e.id = %s", [event_id]).fetchone()
    if not e:
        raise Problem(404, "event_not_found")
    return e, district(c, e["ordinal"])[1]


def search_area(c, e, d):
    """Pipes a crew would survey: those within SEARCH_HOPS junctions of the best candidate."""
    j = e["candidates"][0]["node"]
    near = [a for a in c.execute("SELECT id, a_node, b_node, length_m FROM assets WHERE district_id = %s ORDER BY ordinal", [e["district_id"]]) if min(world.hops(d, a["a_node"], j), world.hops(d, a["b_node"], j)) <= SEARCH_HOPS]
    return near


@app.get("/v1/leaks/{event_id}", tags=["leaks"], summary="One event with its evidence: night flow by day, each sensor's pressure change, and for a leak the candidate locations")
def leak(event_id: uuid.UUID, ctx: Ctx = Depends(auth("network:read"))):
    with db.tx(ctx.tenant_id) as c:
        e, d = event(c, event_id)
        out = {k: e[k] for k in ("id", "district", "kind", "status", "onset_day", "estimated_lps", "probability", "candidates", "evidence", "model", "detected_at")}
        out["sensors"] = [{"sensor": k, "node": n, "x": d["xy"][n][0], "y": d["xy"][n][1]} for k, n in enumerate(d["sensors"])]
        out["inlet"], out["grid"] = list(d["xy"][d["inlet"]]), [world.NX, world.NY]
        if e["kind"] == "leak":
            near, total = search_area(c, e, d), c.execute("SELECT count(*) AS n, sum(length_m) AS m FROM assets WHERE district_id = %s", [e["district_id"]]).fetchone()
            out["loss"] = engine.loss(e["estimated_lps"], e["onset_day"])
            out["search_area"] = {"within_junctions": SEARCH_HOPS, "pipes": len(near), "metres": round(sum(a["length_m"] for a in near)), "district_pipes": total["n"], "district_metres": round(total["m"]), "share_of_district": round(len(near) / total["n"], 4)}
            if e["ordinal"] == world.LEAK["dma"]:               # known only because the network is simulated
                true = world.leak_node(d)
                drop = -np.array(e["evidence"]["pressure_residual_m"])[-2:].mean(axis=0)
                guess = engine.nearest_sensor(drop, d)
                out["simulation_truth"] = {"node": true, "x": d["xy"][true][0], "y": d["xy"][true][1], "junctions_from_best_candidate": world.hops(d, e["candidates"][0]["node"], true), "metres_from_best_candidate": world.hops(d, e["candidates"][0]["node"], true) * world.SPACING_M,
                                           "inside_search_area": world.hops(d, e["candidates"][0]["node"], true) <= SEARCH_HOPS, "true_lps_at_the_end": round(world.LEAK["lps"] * (1 + 0.1 * (world.T - world.LEAK["start_step"]) / world.PER_DAY), 2),
                                           "simple_method": {"rule": "the leak is at the sensor that lost the most pressure", "node": guess, "x": d["xy"][guess][0], "y": d["xy"][guess][1], "junctions_from_truth": world.hops(d, guess, true)}}
    return jsonable_encoder(out)


# ---------------------------------------------------------------- hydraulic-model

class SimulateIn(BaseModel):
    district: int = Field(ge=0, le=world.N_DMA - 1, description="District number, counting from 0")
    node: int = Field(ge=0, le=world.N - 1)
    lps: float = Field(gt=0, le=50, description="Extra water drawn at that junction, in litres per second")
    leak_event_id: uuid.UUID | None = Field(None, description="Compare the prediction with what the sensors saw during this event")


@app.post("/v1/hydraulic/simulate", tags=["hydraulic"], summary="What would the pressure do if this much water left the network at this junction?")
def simulate(body: SimulateIn, ctx: Ctx = Depends(auth("hydraulic:simulate"))):
    with db.tx(ctx.tenant_id) as c:
        row, d = district(c, body.district)
        draw = np.zeros(world.N)
        draw[body.node] = body.lps
        field = d["lu"].solve(draw)
        out = {"district": row["name"], "node": body.node, "x": d["xy"][body.node][0], "y": d["xy"][body.node][1], "lps": body.lps, "pressure_drop_m": [round(float(v), 3) for v in field], "largest_drop_m": round(float(field.max()), 3),
               "at_sensors": [{"sensor": k, "predicted_drop_m": round(float(field[n]), 3)} for k, n in enumerate(d["sensors"])], "grid": [world.NX, world.NY],
               "model": "linearised: head loss proportional to flow. Good for where and roughly how much, not a substitute for a calibrated EPANET model"}
        if body.leak_event_id:
            e, _ = event(c, body.leak_event_id)
            if e["ordinal"] != body.district:
                raise Problem(422, "wrong_district", "that event is in another district")
            seen = -np.array(e["evidence"]["pressure_residual_m"])[-2:].mean(axis=0)
            for s, v in zip(out["at_sensors"], seen):
                s["observed_drop_m"] = round(float(v), 3)
            pred = field[d["sensors"]]
            out["pattern_explained"] = round(float(max(0.0, 1 - np.linalg.norm(seen - pred) / np.linalg.norm(seen))), 3)
    return out


# ---------------------------------------------------------------- work-order-service

class WorkOrderIn(BaseModel):
    leak_event_id: uuid.UUID
    note: str = Field(min_length=5, max_length=500)


@app.post("/v1/work-orders", status_code=201, tags=["work orders"], summary="Send a crew: to find and repair a leak, or to check a sensor. Refused for an event that needs nobody")
def work_order(body: WorkOrderIn, ctx: Ctx = Depends(auth("work_orders:write")), idem: str | None = IdemKey):
    def work(c):
        e, d = event(c, body.leak_event_id)
        if e["kind"] not in ("leak", "sensor_fault"):
            raise Problem(409, "nothing_to_fix", f"this event was classified as {e['kind'].replace('_', ' ')}: {e['evidence'].get('reason', 'no fault found')}")
        if c.execute("SELECT 1 FROM work_orders WHERE leak_event_id = %s", [e["id"]]).fetchone():
            raise Problem(409, "already_open", "a work order already exists for this event")
        wid = uuid.uuid7()
        if e["kind"] == "leak":
            near, top = search_area(c, e, d), e["candidates"][0]
            detail = {"task": f"acoustic survey of {len(near)} pipes ({round(sum(a['length_m'] for a in near))} m) within {SEARCH_HOPS} junctions of junction {top['node']}, then repair", "start_at": {"node": top["node"], "x": top["x"], "y": top["y"]},
                      "estimated_leak_lps": e["estimated_lps"], **engine.loss(e["estimated_lps"], e["onset_day"])}
            kind, priority, ids, sensor = "find_and_repair_leak", "urgent" if e["estimated_lps"] >= 2 else "high", [a["id"] for a in near], None
        else:
            sensor = uuid.uuid5(ctx.tenant_id, f"sensor-{e['ordinal']}-{e['evidence']['sensor']}")
            detail = {"task": f"check and recalibrate pressure sensor {e['evidence']['sensor']}: it reads {abs(e['evidence']['residual_m'])} m {'low' if e['evidence']['residual_m'] < 0 else 'high'} and is drifting {abs(e['evidence']['drift_m_per_day'])} m a day", "no_excavation": True}
            kind, priority, ids = "check_sensor", "routine", []
        c.execute("INSERT INTO work_orders (id, tenant_id, leak_event_id, kind, priority, asset_ids, sensor_id, detail, note, created_by) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)", [wid, ctx.tenant_id, e["id"], kind, priority, ids, sensor, Jsonb(detail), body.note, ctx.actor_id])
        c.execute("UPDATE leak_events SET status = 'work_ordered' WHERE id = %s", [e["id"]])
        audit.record(c, ctx, "work_order.created", "work_order", wid, {"district": e["district"], "kind": kind, "priority": priority, "note": body.note})
        return 201, {"work_order_id": wid, "district": e["district"], "kind": kind, "priority": priority, "status": "open", "pipes_to_survey": len(ids), **detail}
    return run(ctx, idem, body, work)


# ---------------------------------------------------------------- risk-model

def registry(c):
    """The stored pipes and failures in the shape the model code expects."""
    ds, ids = {}, {}
    for a in c.execute("SELECT a.id, d.ordinal AS district, a.ordinal, a.material, a.installed_year, a.diameter_mm, a.soil, a.length_m, a.risk FROM assets a JOIN districts d ON d.id = a.district_id ORDER BY d.ordinal, a.ordinal"):
        ds.setdefault(a["district"], {"index": a["district"], "pipes": []})["pipes"].append({**pipe(a), "id": a["id"], "risk": a["risk"]})
        ids[a["id"]] = (a["district"], a["ordinal"])
    return {"districts": list(ds.values()), "failures": [(*ids[f["asset_id"]], f["year"]) for f in c.execute("SELECT asset_id, year FROM asset_failures")]}


@app.post("/v1/risk:train", status_code=202, tags=["risk"], summary="(+) Train the pipe-failure model on the older years, test it on the latest three, then score every pipe for next year")
def train(ctx: Ctx = Depends(auth("risk:train")), idem: str | None = IdemKey):
    def work(c):
        districts(c)
        return 202, jobs.enqueue(c, ctx, "risk.train", {})
    return run(ctx, idem, {}, work)


@jobs.handler("risk.train")
def train_job(c, job):
    t, w = job["tenant_id"], registry(c)
    _, metrics = engine.train_risk(w)
    x, y, _ = engine.pipe_years(w, world.HISTORY_YEARS)                    # the tested recipe, refitted on every year on record to score the next one
    model = HistGradientBoostingClassifier(max_iter=120, learning_rate=0.08, max_depth=4, random_state=0).fit(x, y)
    breaks = {}
    for d, k, _year in w["failures"]:
        breaks[d, k] = breaks.get((d, k), 0) + 1
    pipes = [(p["id"], engine.features(p, NEXT_YEAR, breaks.get((d["index"], k), 0))) for d in w["districts"] for k, p in enumerate(d["pipes"])]
    risk = model.predict_proba(np.array([f for _, f in pipes], dtype=float))[:, 1]
    rank = np.empty(len(risk), dtype=int)
    rank[np.argsort(-risk, kind="stable")] = np.arange(1, len(risk) + 1)
    c.execute("CREATE TEMP TABLE _risk (id uuid, risk real, risk_rank int) ON COMMIT DROP")
    db.copy(c, "_risk", ["id", "risk", "risk_rank"], [(i, float(r), int(k)) for (i, _), r, k in zip(pipes, risk, rank)])
    c.execute("UPDATE assets a SET risk = r.risk, risk_rank = r.risk_rank FROM _risk r WHERE a.id = r.id")
    c.execute("DROP TABLE _risk")
    mid = uuid.uuid7()
    c.execute("INSERT INTO risk_models (id, tenant_id, version, metrics) VALUES (%s,%s,%s,%s)", [mid, t, engine.VERSION, Jsonb(metrics)])
    top = np.argsort(-risk)[:len(risk) // 20]
    out = {"model_id": mid, "method": "gradient-boosted trees on age, material, diameter, soil and earlier breaks", "trained_through": 2022, "tested_on": [2023, 2025], **metrics, "pipes_scored": len(pipes), "scored_for": NEXT_YEAR,
           "expected_failures_next_year": round(float(risk.sum()), 1), "share_of_expected_failures_in_riskiest_5pct": round(float(risk[top].sum() / risk.sum()), 3)}
    audit.record(c, Ctx(t, uuid.UUID(job["payload"]["actor_id"]), "worker", "system"), "risk.trained", "risk_model", mid, {"auc": metrics["auc"], "pipes_scored": len(pipes)})
    return out


# ---------------------------------------------------------------- capital-planning

class PlanIn(BaseModel):
    budget_per_year: float = Field(ge=100_000, le=100_000_000)
    years: int = Field(5, ge=1, le=10)


@app.post("/v1/capital-plan/optimize", status_code=201, tags=["capital plan"], summary="Which pipes to replace in which year under a fixed annual budget, beside the usual oldest-first rule")
def capital_plan(body: PlanIn, ctx: Ctx = Depends(auth("capital:plan")), idem: str | None = IdemKey):
    def work(c):
        w = registry(c)
        if not w["districts"] or w["districts"][0]["pipes"][0]["risk"] is None:
            raise Problem(409, "no_risk_scores", "train the failure model first: POST /v1/risk:train")
        leaking = set()
        for e in c.execute("SELECT e.*, d.ordinal FROM leak_events e JOIN districts d ON d.id = e.district_id WHERE e.kind = 'leak'").fetchall():
            j = e["candidates"][0]["node"]
            leaking |= {a["id"] for a in c.execute("SELECT id FROM assets WHERE district_id = %s AND (a_node = %s OR b_node = %s)", [e["district_id"], j, j])}
        breaks = {}
        for d, k, _year in w["failures"]:
            breaks[d, k] = breaks.get((d, k), 0) + 1
        pipes, ranked, truth, info = [], [], {}, {}
        for d in w["districts"]:
            for k, p in enumerate(d["pipes"]):
                cost, harm = world.unit_cost(p) * p["length_m"], world.consequence(p)
                pipes.append((p["id"], p["risk"], harm, cost))
                ranked.append((p["id"], max(p["risk"], 0.9) if p["id"] in leaking else p["risk"], harm, cost))     # a pipe at a suspected active leak goes to the front of the queue
                truth[p["id"]] = (world.hazard(p, NEXT_YEAR, breaks.get((d["index"], k), 0)), harm)
                info[p["id"]] = {"district": f"DMA-{d['index'] + 1:02d}", "material": p["material"], "installed_year": p["installed"], "diameter_mm": p["diameter_mm"], "risk": round(p["risk"], 4), "suspected_leak": p["id"] in leaking}
        mine = engine.plan(ranked, body.budget_per_year, body.years)
        old = engine.plan(pipes, body.budget_per_year, body.years, order=lambda x: info[x[0]]["installed_year"])
        chance = {k: (p, harm) for k, p, harm, _ in pipes}

        def describe(items):
            return {"pipes": len(items), "spend": sum(i["cost"] for i in items), "failure_cost_avoided": engine.worth(items, chance, body.years), "mean_age_replaced": round(float(np.mean([NEXT_YEAR - info[i["key"]]["installed_year"] for i in items])), 1),
                    "by_year": [{"year": NEXT_YEAR + y, "pipes": sum(i["year"] == y + 1 for i in items), "spend": sum(i["cost"] for i in items if i["year"] == y + 1)} for y in range(body.years)]}
        a, b, top = describe(mine), describe(old), engine.bound(pipes, body.budget_per_year, body.years)
        summary = {"budget_per_year": body.budget_per_year, "years": body.years, "optimised": a, "oldest_first": b, "upper_bound": top, "share_of_upper_bound": round(a["failure_cost_avoided"] / top, 4),
                   "times_oldest_first": round(a["failure_cost_avoided"] / max(1, b["failure_cost_avoided"]), 2), "benefit_per_dollar": round(a["failure_cost_avoided"] / a["spend"], 3),
                   "moved_up_for_a_suspected_leak": sum(info[i["key"]]["suspected_leak"] and i["year"] == 1 for i in mine),
                   "check_against_the_simulation": {"what": "both plans scored with the hidden failure rates the data was generated from, which the model never saw", "optimised": engine.worth(mine, truth, body.years), "oldest_first": engine.worth(old, truth, body.years)},
                   "assumptions": "replacement cost and the cost of one failure are formulas on pipe diameter and length, not a utility's real figures; no crew, street-works or grouping constraints"}
        pid = uuid.uuid7()
        c.execute("INSERT INTO capital_plans (id, tenant_id, budget_per_year, years, summary, items, created_by) VALUES (%s,%s,%s,%s,%s,%s,%s)", [pid, ctx.tenant_id, body.budget_per_year, body.years, Jsonb(summary), Jsonb(jsonable_encoder(mine)), ctx.actor_id])
        audit.record(c, ctx, "capital_plan.created", "capital_plan", pid, {"budget_per_year": body.budget_per_year, "years": body.years, "pipes": a["pipes"], "failure_cost_avoided": a["failure_cost_avoided"]})
        return 201, {"plan_id": pid, **summary, "first_in_line": [{"asset_id": i["key"], "year": NEXT_YEAR + i["year"] - 1, "cost": i["cost"], "avoided": engine.worth([i], chance, body.years), **info[i["key"]]} for i in mine[:10]]}
    return run(ctx, idem, body, work)


@app.get("/v1/capital-plan", tags=["capital plan"], summary="(+) The latest capital plan")
def latest_plan(ctx: Ctx = Depends(auth("network:read"))):
    with db.tx(ctx.tenant_id) as c:
        p = c.execute("SELECT id AS plan_id, summary, jsonb_array_length(items) AS pipes, created_at FROM capital_plans ORDER BY created_at DESC LIMIT 1").fetchone()
    if not p:
        raise Problem(404, "no_plan")
    return jsonable_encoder(p)
