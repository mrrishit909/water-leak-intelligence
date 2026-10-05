"""Integration, end-to-end and security tests against a real Postgres."""
import datetime
import uuid

import pytest

from core import db, jobs, scenario
from wa import api, world

from .conftest import bearer

PLANNER, ANALYST, VIEWER, OTHER = bearer("utility", "planner"), bearer("utility", "analyst"), bearer("utility", "viewer"), bearer("other", "planner")
ZERO = "00000000-0000-0000-0000-000000000000"


def test_nothing_works_before_a_network_is_loaded(client):
    assert client.get("/v1/network", headers=VIEWER).json()["code"] == "no_network" and client.post("/v1/leaks:detect", headers=ANALYST).json()["code"] == "no_network"
    assert client.post("/v1/risk:train", headers=PLANNER).json()["code"] == "no_network" and client.post("/v1/capital-plan/optimize", headers=PLANNER, json={"budget_per_year": 1e6}).json()["code"] == "no_risk_scores"
    assert client.post("/v1/hydraulic/simulate", headers=ANALYST, json={"district": 0, "node": 1, "lps": 1}).json()["code"] == "no_network" and client.get("/v1/capital-plan", headers=VIEWER).status_code == 404


@pytest.fixture(scope="module")
def demo(client):
    results, _ = scenario.run(client, drain=jobs.drain)
    return results


def one(sql, params=(), tenant=None):
    with db.tx(tenant) as c:
        return c.execute(sql, params).fetchone()


def job(client, path, headers, body=None):
    r = client.post(path, headers=headers, json=body or {})
    assert r.status_code == 202, r.text
    jobs.drain()
    return client.get(r.json()["status_url"], headers=headers).json()["result"]


def test_demo_journey(demo):
    load = demo["load"]["result"]
    assert load["districts"] == 40 and load["junctions"] == 20000 and load["pipes"] == one("SELECT count(*) AS n FROM assets")["n"] > 33000 and load["readings"] == one("SELECT count(*) AS n FROM telemetry")["n"] == 40 * 11 * world.T
    det = demo["det"]["result"]
    assert det["districts_checked"] == 40 and det["normal"] == 37 and {(e["district"], e["kind"]) for e in det["events"]} == {("DMA-08", "leak"), ("DMA-22", "sensor_fault"), ("DMA-31", "transient_demand")}
    assert [d["event"] for d in demo["net"]["districts"]].count(None) == 37 and demo["net"]["districts"][7]["event"] == "leak"
    leak = demo["leak"]
    truth = leak["simulation_truth"]
    assert truth["junctions_from_best_candidate"] <= 2 < truth["simple_method"]["junctions_from_truth"] and truth["inside_search_area"] and leak["search_area"]["share_of_district"] < 0.1
    assert leak["estimated_lps"] == pytest.approx(truth["true_lps_at_the_end"], rel=0.2) and leak["loss"]["per_year_cost"] > 50000 and len(leak["evidence"]["night_flow_lps"]) == world.DAYS and len(leak["sensors"]) == 10
    sim = demo["sim"]
    assert sim["pattern_explained"] > 0.95 and len(sim["pressure_drop_m"]) == world.N and all(abs(s["predicted_drop_m"] - s["observed_drop_m"]) < 0.15 for s in sim["at_sensors"])
    fault, spike = demo["fault"], demo["spike"]
    assert fault["evidence"]["sensor"] == world.DRIFT["sensor"] and fault["evidence"]["drift_m_per_day"] < 0 and fault["candidates"] == [] and "loss" not in fault
    assert spike["evidence"]["nights_elevated"] == 1 and spike["onset_day"] == world.SPIKE["start_step"] // world.PER_DAY
    wo, wo2 = demo["wo"], demo["wo2"]
    assert wo["kind"] == "find_and_repair_leak" and wo["priority"] == "urgent" and wo["pipes_to_survey"] == leak["search_area"]["pipes"] and wo2["kind"] == "check_sensor" and wo2["pipes_to_survey"] == 0 and wo2["no_excavation"]
    assert demo["wo3"]["code"] == "nothing_to_fix" and "does not repair itself" in demo["wo3"]["detail"]
    risk = demo["risk"]["result"]
    assert risk["auc"] > risk["auc_age_only"] > 0.7 and risk["top_5pct_capture"] > risk["top_5pct_capture_age_only"] and risk["pipes_scored"] == load["pipes"]
    top = demo["top"]["items"]
    assert [a["risk_rank"] for a in top] == list(range(1, 9)) and top[0]["risk"] >= top[-1]["risk"] and demo["one"]["rank"] == 1 and demo["one"]["times_network_average"] > 5 and len(demo["one"]["failure_years"]) == top[0]["breaks"]
    plan = demo["plan"]
    assert plan["optimised"]["failure_cost_avoided"] > 1.5 * plan["oldest_first"]["failure_cost_avoided"] and plan["check_against_the_simulation"]["optimised"] > 1.5 * plan["check_against_the_simulation"]["oldest_first"]
    assert all(y["spend"] <= 4_000_000 for y in plan["optimised"]["by_year"] + plan["oldest_first"]["by_year"]) and plan["optimised"]["failure_cost_avoided"] <= plan["upper_bound"]
    assert plan["moved_up_for_a_suspected_leak"] >= 2 and plan["first_in_line"][0]["suspected_leak"] and plan["first_in_line"][0]["district"] == "DMA-08" and plan["oldest_first"]["mean_age_replaced"] > plan["optimised"]["mean_age_replaced"]
    assert demo["audit"]["chain_valid"] and {"network.loaded", "leaks.detected", "work_order.created", "risk.trained", "capital_plan.created"} <= {e["action"] for e in demo["audit"]["events"]}


def test_stored_plan_and_lists(client, demo):
    p = client.get("/v1/capital-plan", headers=VIEWER).json()
    assert p["plan_id"] == demo["plan"]["plan_id"] and p["pipes"] == demo["plan"]["optimised"]["pipes"] and p["summary"]["years"] == 5
    items = client.get("/v1/leaks", headers=VIEWER).json()["items"]
    assert [i["kind"] for i in items][0] == "leak" and len(items) == 3 and {i["status"] for i in items} == {"work_ordered", "open"}
    assert [i["kind"] for i in client.get("/v1/leaks?kind=sensor_fault", headers=VIEWER).json()["items"]] == ["sensor_fault"] and client.get("/v1/leaks?kind=flood", headers=VIEWER).status_code == 422
    assert client.get(f"/v1/leaks/{ZERO}", headers=VIEWER).json()["code"] == "event_not_found" and client.get(f"/v1/assets/{ZERO}/risk", headers=VIEWER).json()["code"] == "asset_not_found"
    assert client.get("/v1/assets?limit=500", headers=VIEWER).status_code == 422 and len(client.get("/v1/assets?limit=3", headers=VIEWER).json()["items"]) == 3


def test_roles(client, demo):
    assert client.get("/v1/network").status_code == 401 and client.get("/v1/network", headers={"Authorization": "Bearer nope"}).status_code == 401
    body = {"leak_event_id": demo["det"]["result"]["ids"]["leak"], "note": "viewer tries"}
    for path, payload in (("/v1/leaks:detect", {}), ("/v1/work-orders", body), ("/v1/telemetry", {"readings": [{"sensor_id": ZERO, "ts": "2026-09-14T00:00:00Z", "value": 1}]}), ("/v1/hydraulic/simulate", {"district": 0, "node": 0, "lps": 1})):
        assert client.post(path, headers=VIEWER, json=payload).status_code == 403, path
    for path, payload in (("/v1/network:load", {}), ("/v1/risk:train", {}), ("/v1/capital-plan/optimize", {"budget_per_year": 1e6})):
        assert client.post(path, headers=ANALYST, json=payload).status_code == 403, path
    assert client.get("/v1/audit", headers=ANALYST).status_code == 403 and client.get("/v1/audit", headers=PLANNER).status_code == 200


def test_another_utility_sees_nothing(client, demo, seeded):
    ids = demo["det"]["result"]["ids"]
    assert client.get("/v1/network", headers=OTHER).json()["code"] == "no_network" and client.get("/v1/leaks", headers=OTHER).json()["items"] == [] and client.get("/v1/assets", headers=OTHER).json()["items"] == []
    assert client.get(f"/v1/leaks/{ids['leak']}", headers=OTHER).status_code == 404 and client.get(f"/v1/assets/{demo['top']['items'][0]['id']}/risk", headers=OTHER).status_code == 404
    assert client.post("/v1/work-orders", headers=OTHER, json={"leak_event_id": ids["sensor_fault"], "note": "not mine"}).status_code == 404
    sensor = str(uuid.uuid5(seeded["utility"], "sensor-0-0"))
    r = client.post("/v1/telemetry", headers=OTHER, json={"readings": [{"sensor_id": sensor, "ts": "2026-10-01T00:00:00Z", "value": 1}]})
    assert r.json()["code"] == "sensor_not_found" and one("SELECT count(*) AS n FROM telemetry WHERE ts = '2026-10-01T00:00:00Z'")["n"] == 0
    for table in ("districts", "assets", "asset_failures", "sensors", "telemetry", "leak_events", "work_orders", "risk_models", "capital_plans"):       # the database itself, not only the API
        assert one(f"SELECT count(*) AS n FROM {table}", tenant=seeded["other"])["n"] == 0 < one(f"SELECT count(*) AS n FROM {table}", tenant=seeded["utility"])["n"], table
    with pytest.raises(Exception, match="row-level security"), db.tx(seeded["other"]) as c:
        c.execute("INSERT INTO districts (id, tenant_id, ordinal, name, junctions, model_seed, telemetry_from) VALUES (%s,%s,99,'x',1,1,now())", [uuid.uuid4(), seeded["utility"]])


def test_idempotent_writes(client, demo):
    body = {"leak_event_id": demo["det"]["result"]["ids"]["leak"], "note": "Night flow up for four nights. Acoustic survey from the best candidate outwards."}
    again = client.post("/v1/work-orders", headers={**ANALYST, "Idempotency-Key": "demo-1:wo"}, json=body)
    assert again.status_code == 201 and again.headers["Idempotent-Replay"] == "true" and again.json() == demo["wo"] and one("SELECT count(*) AS n FROM work_orders")["n"] == 2
    assert client.post("/v1/work-orders", headers={**ANALYST, "Idempotency-Key": "demo-1:wo"}, json={**body, "note": "something else"}).json()["code"] == "idempotency_key_reused"
    assert client.post("/v1/work-orders", headers=ANALYST, json=body).json()["code"] == "already_open"
    assert client.post("/v1/network:load", headers=PLANNER, json={"seed": 17}).json()["code"] == "network_exists"


def test_telemetry_ingestion(client, demo, seeded):
    sensor = str(uuid.uuid5(seeded["utility"], "sensor-0-3"))
    late = (api.T0 + world.T * api.STEP).isoformat()
    r = client.post("/v1/telemetry", headers=ANALYST, json={"readings": [{"sensor_id": sensor, "ts": api.T0.isoformat(), "value": 44.0}, {"sensor_id": sensor, "ts": late, "value": 44.2}]}).json()
    assert r == {"received": 2, "stored": 1, "duplicates_ignored": 1}                         # the first was loaded with the network and is not overwritten
    assert one("SELECT value FROM telemetry WHERE sensor_id = %s AND step = 0", [sensor])["value"] != 44.0 and one("SELECT step FROM telemetry WHERE sensor_id = %s AND ts = %s", [sensor, late])["step"] == world.T
    assert client.post("/v1/telemetry", headers=ANALYST, json={"readings": [{"sensor_id": sensor, "ts": late, "value": 44.2}]}).json()["stored"] == 0
    bad = lambda **kw: client.post("/v1/telemetry", headers=ANALYST, json={"readings": [{"sensor_id": sensor, "ts": "2026-09-14T00:15:00Z", "value": 40, **kw}]})      # noqa: E731
    assert bad(ts="2026-09-14T00:07:00Z").json()["code"] == "off_grid" and bad(ts="2026-09-14T00:15:00").json()["code"] == "timezone_required" and bad(value=1e9).status_code == 422
    assert bad(sensor_id=ZERO).json()["code"] == "sensor_not_found" and client.post("/v1/telemetry", headers=ANALYST, json={"readings": []}).status_code == 422


def test_a_gap_is_skipped_until_the_reading_arrives(client, demo, seeded):
    sensor, ts = uuid.uuid5(seeded["utility"], "sensor-2-5"), api.T0 + 700 * api.STEP
    value = one("DELETE FROM telemetry WHERE sensor_id = %s AND ts = %s RETURNING value", [sensor, ts])["value"]
    out = job(client, "/v1/leaks:detect", ANALYST)
    assert out["skipped_for_missing_data"] == 1 and out["districts_checked"] == 39 and len(out["events"]) == 3
    assert client.post("/v1/telemetry", headers=ANALYST, json={"readings": [{"sensor_id": str(sensor), "ts": ts.isoformat(), "value": value}]}).json()["stored"] == 1
    out = job(client, "/v1/leaks:detect", ANALYST)
    assert out["skipped_for_missing_data"] == 0 and out["ids"] == demo["det"]["result"]["ids"]                    # the same events, not new copies
    assert one("SELECT count(*) AS n FROM leak_events")["n"] == 3 and one("SELECT status FROM leak_events WHERE kind = 'leak'")["status"] == "work_ordered"


def test_a_repaired_district_drops_off_the_list(client, demo, seeded):
    """Replace the spike district's flow readings with an ordinary fortnight: its open event must go; events with a work order stay."""
    with db.tx() as c:
        c.execute("UPDATE leak_events SET status = 'open' WHERE kind = 'transient_demand'")
        d = world.build()["districts"][world.SPIKE["dma"]]
        _, inflow = world.telemetry(d, 17)                                                    # the same fortnight without the spike
        flow = uuid.uuid5(seeded["utility"], f"sensor-{world.SPIKE['dma']}-{world.K_SENSORS}")
        db.many(c, "UPDATE telemetry SET value = %s WHERE sensor_id = %s AND step = %s", [(float(v), flow, s) for s, v in enumerate(inflow)])
    out = job(client, "/v1/leaks:detect", ANALYST)
    assert {e["kind"] for e in out["events"]} == {"leak", "sensor_fault"} and one("SELECT count(*) AS n FROM leak_events")["n"] == 2


def test_simulate(client, demo):
    post = lambda **kw: client.post("/v1/hydraulic/simulate", headers=ANALYST, json={"district": 3, "node": 250, "lps": 1.0, **kw})      # noqa: E731
    a, b = post().json(), post(lps=2.0).json()
    assert b["largest_drop_m"] == pytest.approx(2 * a["largest_drop_m"], rel=0.01) and a["pressure_drop_m"].index(max(a["pressure_drop_m"])) == 250 and "pattern_explained" not in a
    assert post(leak_event_id=demo["det"]["result"]["ids"]["leak"]).json()["code"] == "wrong_district" and post(leak_event_id=ZERO).json()["code"] == "event_not_found"
    assert post(node=world.N).status_code == 422 and post(lps=0).status_code == 422 and post(district=40).status_code == 422


def test_plan_validation_and_budget(client, demo):
    assert client.post("/v1/capital-plan/optimize", headers=PLANNER, json={"budget_per_year": 5}).status_code == 422 and client.post("/v1/capital-plan/optimize", headers=PLANNER, json={"budget_per_year": 1e6, "years": 0}).status_code == 422
    small = client.post("/v1/capital-plan/optimize", headers=PLANNER, json={"budget_per_year": 500_000, "years": 3}).json()
    assert len(small["optimised"]["by_year"]) == 3 and all(y["spend"] <= 500_000 for y in small["optimised"]["by_year"]) and small["optimised"]["pipes"] < demo["plan"]["optimised"]["pipes"]
    assert small["benefit_per_dollar"] > demo["plan"]["benefit_per_dollar"]                   # the first dollars go to the best pipes
