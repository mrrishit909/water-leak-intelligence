"""Unit tests for the simulator and the models: no database."""
import numpy as np
import pytest

from wa import engine, world

W = world.build()
D = W["districts"][3]


def test_world_is_deterministic_and_plausible():
    again = world.dma(17, 3)
    assert again["pipes"] == D["pipes"] and again["sensors"] == D["sensors"] and np.allclose(again["sig"], D["sig"])
    p1, f1 = world.telemetry(D, 5)
    p2, f2 = world.telemetry(D, 5)
    assert np.array_equal(p1, p2) and np.array_equal(f1, f2) and p1.shape == (world.T, world.K_SENSORS) and f1.shape == (world.T,)
    assert 30 < p1.min() and p1.max() < world.HEAD_M + 1 and f1.reshape(world.DAYS, -1)[:, 8:16].mean() < 0.7 * f1.mean()          # pressure stays positive; nights are quiet
    assert len(W["districts"]) == 40 and sum(len(d["pipes"]) for d in W["districts"]) > 33000 and all(len(set(d["sensors"])) == world.K_SENSORS for d in W["districts"])


def test_every_junction_is_connected_to_the_inlet():
    seen, edges = {D["inlet"]}, {}
    for p in D["pipes"]:
        edges.setdefault(p["a"], []).append(p["b"])
        edges.setdefault(p["b"], []).append(p["a"])
    todo = [D["inlet"]]
    while todo:
        for n in edges[todo.pop()]:
            if n not in seen:
                seen.add(n)
                todo.append(n)
    assert len(seen) == world.N


def test_a_leak_lowers_pressure_most_near_itself():
    j = 312
    draw = np.zeros(world.N)
    draw[j] = 1.0
    field = D["lu"].solve(draw)
    assert int(np.argmax(field)) == j and (field > 0).all() and np.allclose(field[D["sensors"]], D["sig"][:, j])
    assert np.allclose(D["lu"].solve(draw * 3), field * 3)                                   # linear: three times the leak, three times the drop


def test_hazard_rises_with_age_breaks_and_poor_soil():
    p = {"material": "cast iron", "installed": 1950, "diameter_mm": 100, "soil": 0.5}
    assert world.hazard(p, 2025, 0) > world.hazard(p, 2016, 0) and world.hazard(p, 2025, 2) > world.hazard(p, 2025, 0)
    assert world.hazard({**p, "soil": 0.9}, 2025, 0) > world.hazard(p, 2025, 0) > world.hazard({**p, "material": "PVC"}, 2025, 0) and world.hazard(p, 2025, 500) == 0.6


def test_nightly_takes_the_two_to_four_window():
    x = np.tile(np.arange(world.PER_DAY, dtype=float), world.DAYS)
    assert engine.nightly(x).tolist() == [11.5] * world.DAYS and engine.nightly(np.stack([x, x * 2], axis=1)).shape == (world.DAYS, 2)


def test_localise_recovers_a_clean_signature():
    out = engine.localise(D["sig"][:, 222] * 1.7, D["sig"])
    assert out[0][0] == 222 and out[0][1] == pytest.approx(1.7) and out[0][2] == pytest.approx(1.0) and [c[2] for c in out] == sorted((c[2] for c in out), reverse=True)
    assert engine.localise(-D["sig"][:, 222], D["sig"]) == []                                 # a pressure rise is never explained by a leak


def test_a_quiet_fortnight_is_normal():
    assert all(engine.analyse(*world.telemetry(D, s), D)["kind"] == "normal" for s in range(40, 60))


def test_a_leak_is_found_sized_and_located():
    j, lps = 287, 1.5
    a = engine.analyse(*world.telemetry(D, 7, leak=(j, 8 * world.PER_DAY, lps)), D)
    assert a["kind"] == "leak" and a["onset_day"] in (8, 9) and world.hops(D, a["candidates"][0]["node"], j) <= 3 and a["candidates"][0]["fit"] > 0.9
    assert a["flow_rise_lps"] == pytest.approx(lps * 1.45, rel=0.15) and len(a["candidates"]) == 5 and a["nights_elevated"] >= 5


def test_a_drifting_sensor_is_not_a_leak():
    for rate in (-0.004, 0.004):
        a = engine.analyse(*world.telemetry(D, 8, drift=(6, 6 * world.PER_DAY, rate)), D)
        assert a["kind"] == "sensor_fault" and a["sensor"] == 6 and a["drift_m_per_day"] * rate > 0 and a["others_max_z"] < engine.ALARM_Z and abs(a["flow_z"]) < engine.ALARM_Z


def test_one_night_of_demand_is_not_a_leak():
    a = engine.analyse(*world.telemetry(D, 9, spike=(100, 10 * world.PER_DAY + 6, 14, 4.0)), D)
    assert a["kind"] == "transient_demand" and a["onset_day"] == 10 and a["flow_rise_lps"] == pytest.approx(4.0, abs=0.5)


def test_a_busy_week_is_not_a_leak():
    """Demand 8% up on every one of the last nine days lifts night flow as much as a small leak would, but in proportion."""
    p, f = world.telemetry(D, 10)
    busy = np.repeat(np.r_[np.ones(world.TRAIN_DAYS), np.full(world.DAYS - world.TRAIN_DAYS, 1.08)], world.PER_DAY)
    assert engine.analyse(world.HEAD_M - (world.HEAD_M - p) * busy[:, None], f * busy, D)["kind"] == "normal"


def test_loss_arithmetic():
    out = engine.loss(2.0, 10)
    assert out == {"cubic_metres_so_far": round(2 * 86.4 * 4), "cost_so_far": round(2 * 86.4 * 4 * 1.5), "per_year_if_unrepaired_m3": round(2 * 86.4 * 365), "per_year_cost": round(2 * 86.4 * 365 * 1.5)}


def test_pipe_years_use_only_what_was_known_at_the_time():
    w = {"districts": [{"index": 0, "pipes": [{"material": "PVC", "installed": 2018, "diameter_mm": 150, "soil": 0.4}]}], "failures": [(0, 0, 2020), (0, 0, 2022)]}
    x, y, keys = engine.pipe_years(w, [2017, 2019, 2020, 2021, 2022, 2023])
    assert [k[2] for k in keys] == [2019, 2020, 2021, 2022, 2023]                            # not before it was laid
    assert x[:, 3].tolist() == [0, 0, 1, 1, 2] and y.tolist() == [False, True, False, True, False] and x[:, 0].tolist() == [1, 2, 3, 4, 5]
    assert x[0, 4:].tolist() == [0, 0, 0, 1] and len(engine.FEATURE_NAMES) == x.shape[1]


def test_auc_and_capture():
    y = np.array([False, False, True, True])
    assert engine.auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0 and engine.auc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0 and engine.auc(y, np.ones(4)) == 0.5
    y = np.zeros(100, dtype=bool)
    y[[0, 1, 50, 51]] = True
    assert engine.capture(y, -np.arange(100.0), share=0.05) == 0.5


def test_risk_model_beats_age_alone_on_years_it_never_saw():
    _, m = engine.train_risk(W)
    assert m["auc"] > m["auc_age_only"] > 0.7 and m["top_5pct_capture"] > m["top_5pct_capture_age_only"] and m["mean_predicted"] == pytest.approx(m["observed_rate"], rel=0.25)


PIPES = [("a", 0.30, 20000, 25000), ("b", 0.02, 20000, 25000), ("c", 0.20, 40000, 30000), ("d", 0.01, 9000, 20000), ("e", 0.25, 30000, 90000), ("f", 0.10, 20000, 25000)]


def test_plan_respects_the_budget_and_takes_the_best_value_first():
    items = engine.plan(PIPES, 60000, years=2)
    assert [(i["key"], i["year"]) for i in items] == [("c", 1), ("a", 1), ("f", 2), ("b", 2)]                # e never fits a 60k year; d is the worst value and the money is gone
    assert all(sum(i["cost"] for i in items if i["year"] == y) <= 60000 for y in (1, 2)) and len({i["key"] for i in items}) == len(items)
    assert items[0]["avoided"] == round(engine.avoided(0.20, 40000, 0, 2)) and items[2]["avoided"] == round(engine.avoided(0.10, 20000, 1, 2))


def test_plan_beats_oldest_first_and_stays_under_the_bound():
    chance = {k: (p, c) for k, p, c, _ in PIPES}
    age = {"a": 1990, "b": 1930, "c": 1985, "d": 1935, "e": 1940, "f": 1995}
    mine, old = engine.plan(PIPES, 60000, 2), engine.plan(PIPES, 60000, 2, order=lambda x: age[x[0]])
    assert [i["key"] for i in old][:2] == ["b", "d"] and engine.worth(mine, chance, 2) > engine.worth(old, chance, 2)
    assert engine.worth(mine, chance, 2) <= engine.bound(PIPES, 60000, 2) and engine.worth([], chance, 2) == 0


def test_waiting_avoids_less():
    assert engine.avoided(0.2, 1000, 0, 5) > engine.avoided(0.2, 1000, 1, 5) > engine.avoided(0.2, 1000, 4, 5) > engine.avoided(0.2, 1000, 5, 5) == 0
    assert engine.avoided(0.99, 1000, 0, 3) <= 3000                                           # a chance never exceeds certainty
