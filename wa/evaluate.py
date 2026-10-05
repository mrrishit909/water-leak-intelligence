"""Measure every model on utilities it was not tuned on, beside a simple baseline. Writes docs/evaluation.md.

    python -m wa.evaluate
"""
import pathlib
import random

import numpy as np

from . import engine, world

OUT = pathlib.Path(__file__).resolve().parents[1] / "docs" / "evaluation.md"
SEEDS = (101, 102, 103)
BUDGET, YEARS = 4_000_000, 5


def table(head, rows):
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def pct(x):
    return f"{100 * x:.1f}%"


def detection(n=600):
    """Random districts with a leak, a drifting sensor, a night of heavy demand, or nothing."""
    w, r = world.build(SEEDS[0], 12), random.Random(5)
    confusion, leaks = {}, []
    for i in range(n):
        d, truth = w["districts"][r.randrange(12)], ["leak", "sensor_fault", "transient_demand", "normal"][i % 4]
        kw, j, lps = {}, r.randrange(world.N), round(r.uniform(0.2, 3.0), 2)
        if truth == "leak":
            kw["leak"] = (j, r.randrange(6, 12) * world.PER_DAY + r.randrange(world.PER_DAY), lps)
        elif truth == "sensor_fault":
            kw["drift"] = (r.randrange(world.K_SENSORS), r.randrange(5, 9) * world.PER_DAY, r.choice([-1, 1]) * r.uniform(0.001, 0.006))
        elif truth == "transient_demand":
            kw["spike"] = (j, r.randrange(6, 13) * world.PER_DAY + r.randrange(4, 10), r.randrange(8, 20), r.uniform(1.5, 6))
        p, f = world.telemetry(d, 9000 + i, **kw)
        a = engine.analyse(p, f, d)
        confusion[truth, a["kind"]] = confusion.get((truth, a["kind"]), 0) + 1
        if truth == "leak":
            row = {"lps": lps, "found": a["kind"] == "leak"}
            if row["found"]:
                drop = -np.array(a["pressure_residual_m"])[-2:].mean(axis=0)
                final = lps * (1 + 0.1 * (world.T - 1.5 * world.PER_DAY - kw["leak"][1]) / world.PER_DAY)       # the leak's size over the two nights it is measured on
                row.update(model=world.hops(d, a["candidates"][0]["node"], j), best5=min(world.hops(d, c["node"], j) for c in a["candidates"]), simple=world.hops(d, engine.nearest_sensor(drop, d), j), size_err=abs(a["flow_rise_lps"] - final) / final)
            leaks.append(row)
    return confusion, leaks


def risk():
    rows, plans = [], []
    for seed in SEEDS:
        w = world.build(seed)
        model, m = engine.train_risk(w)
        rows.append((seed, m))
        # the plan, scored with the hidden failure rates
        breaks = {}
        for d, k, _y in w["failures"]:
            breaks[d, k] = breaks.get((d, k), 0) + 1
        keys = [(d["index"], k) for d in w["districts"] for k in range(len(d["pipes"]))]
        ps = [w["districts"][d]["pipes"][k] for d, k in keys]
        x = np.array([engine.features(p, 2026, breaks.get(key, 0)) for key, p in zip(keys, ps)], dtype=float)
        pred = model.predict_proba(x)[:, 1]          # ponytail: the through-2022 model scores 2026 here; the service refits on all years first
        harm, cost = [world.consequence(p) for p in ps], [world.unit_cost(p) * p["length_m"] for p in ps]
        true = {key: (world.hazard(p, 2026, breaks.get(key, 0)), h) for key, p, h in zip(keys, ps, harm)}
        by_model = [(key, float(q), h, c) for key, q, h, c in zip(keys, pred, harm, cost)]
        by_truth = [(key, true[key][0], h, c) for key, h, c in zip(keys, harm, cost)]
        age = {key: p["installed"] for key, p in zip(keys, ps)}
        shuffled = dict(zip(keys, random.Random(seed).sample(range(len(keys)), len(keys))))
        score = lambda items: engine.worth(items, true, YEARS)          # noqa: E731
        plans.append((seed, score(engine.plan(by_model, BUDGET, YEARS)), score(engine.plan(by_model, BUDGET, YEARS, order=lambda v: age[v[0]])), score(engine.plan(by_model, BUDGET, YEARS, order=lambda v: shuffled[v[0]])),
                      score(engine.plan(by_truth, BUDGET, YEARS)), engine.bound(by_truth, BUDGET, YEARS)))
    return rows, plans


def main():
    confusion, leaks = detection()
    kinds = ["leak", "sensor_fault", "transient_demand", "normal", "unexplained"]
    found = [x for x in leaks if x["found"]]
    buckets = [(0.2, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 3.01)]
    false_alarms = sum(v for (t, k), v in confusion.items() if t != "leak" and k == "leak")
    rows, plans = risk()
    doc = f"""# Evaluation

Everything here is measured on synthetic utilities generated with seeds the thresholds were not tuned on (`{SEEDS}`), and
every model sits beside a simple baseline. Reproduce with `python -m wa.evaluate`. The numbers say how well the methods
recover what the simulator hid. They are not a claim about any real network: real hydraulics are non-linear, real demand
is messier, and a real hydraulic model is never a perfect copy of the pipes in the ground, as this one is.

## 1. Is it a leak, a faulty sensor, or just demand?

{len(leaks) * 4} district-fortnights: a quarter with a leak of 0.2 to 3.0 L/s at a random junction, a quarter with one pressure sensor drifting
0.001 to 0.006 m an hour, a quarter with one night of 1.5 to 6 L/s extra demand, a quarter with nothing. Rows are the truth.

{table(["truth / verdict", *[k.replace("_", " ") for k in kinds]], [[t.replace("_", " "), *[confusion.get((t, k), 0) for k in kinds]] for t in kinds[:4]])}

Districts wrongly called a leak (a crew sent for nothing): **{false_alarms} of {len(leaks) * 3}**.

Leaks found, by size (night flow in these districts is about 9.5 L/s):

{table(["leak size (L/s)", "leaks", "found"], [[f"{a} to {min(b, 3.0)}", sum(a <= x["lps"] < b for x in leaks), pct(np.mean([x["found"] for x in leaks if a <= x["lps"] < b]))] for a, b in buckets])}

Small leaks are missed on purpose: the alarm needs night flow well clear of normal on the last two nights, because a
false alarm costs a crew a day. Slow sensor drifts below about 0.25 m in total are also left as normal.

## 2. Where is the leak?

For the {len(found)} leaks found. Distance is counted in junctions along the grid (one junction = {world.SPACING_M:.0f} m). The baseline is the
obvious rule: the leak is at the sensor that lost the most pressure.

{table(["method", "median distance", "mean", "within 3 junctions", "exactly right"], [
    ["signature fit (this system)", f"{np.median([x['model'] for x in found]):.0f}", f"{np.mean([x['model'] for x in found]):.2f}", pct(np.mean([x["model"] <= 3 for x in found])), pct(np.mean([x["model"] == 0 for x in found]))],
    ["best of its top 5 candidates", f"{np.median([x['best5'] for x in found]):.0f}", f"{np.mean([x['best5'] for x in found]):.2f}", pct(np.mean([x["best5"] <= 3 for x in found])), pct(np.mean([x["best5"] == 0 for x in found]))],
    ["nearest sensor (baseline)", f"{np.median([x['simple'] for x in found]):.0f}", f"{np.mean([x['simple'] for x in found]):.2f}", pct(np.mean([x["simple"] <= 3 for x in found])), pct(np.mean([x["simple"] == 0 for x in found]))]])}

By leak size, share located within 3 junctions (the area a work order asks a crew to survey):

{table(["leak size (L/s)", "found", "within 3 junctions", "baseline"], [[f"{a} to {min(b, 3.0)}", sum(a <= x["lps"] < b for x in found), pct(np.mean([x["model"] <= 3 for x in found if a <= x["lps"] < b] or [0])), pct(np.mean([x["simple"] <= 3 for x in found if a <= x["lps"] < b] or [0]))] for a, b in buckets])}

Leak size from the rise in night flow: mean error **{pct(np.mean([x["size_err"] for x in found]))}** of the true size.

This is the best case for the method: the model used to locate the leak is the same one that generated the pressures.
With a real network the model's pipe roughness and valve states are partly wrong, and accuracy falls with them.

## 3. Which pipes fail next?

Trained on pipe-years up to 2022, tested on 2023 to 2025. The baseline ranks pipes by age alone.

{table(["utility (seed)", "failures in test", "AUC", "AUC, age only", "failures caught in riskiest 5%", "same, age only", "predicted rate", "actual rate"],
       [[s, m["failures_in_test"], f"{m['auc']:.3f}", f"{m['auc_age_only']:.3f}", pct(m["top_5pct_capture"]), pct(m["top_5pct_capture_age_only"]), pct(m["mean_predicted"]), pct(m["observed_rate"])] for s, m in rows])}

The model beats age alone by a modest margin. Failures are rare ({pct(np.mean([m["observed_rate"] for _, m in rows]))} of pipes a year) and largely chance, which
caps what any ranking can do: section 4 shows that planning with the true hidden rates would gain only about a tenth more.

## 4. Does the capital plan spend the money better?

${BUDGET / 1e6:.0f} million a year for {YEARS} years. Every plan is scored the same way: the failure cost it avoids under the hidden failure
rates the data was generated from, which no plan was shown. "Perfect knowledge" plans with those hidden rates.

{table(["utility (seed)", "this system", "oldest first", "random pipes", "perfect knowledge", "upper bound"], [[s, *[f"${v / 1e6:.2f}M" for v in vals]] for s, *vals in plans])}

Against oldest-first the plan avoids **{np.mean([p[1] / p[2] for p in plans]):.1f} times** the failure cost, and reaches {pct(np.mean([p[1] / p[4] for p in plans]))} of what perfect knowledge
of the failure rates would. (The upper bound is looser than it looks: it spends all five years of money in year one.) Note what the table also says: ${BUDGET * YEARS / 1e6:.0f} million of replacements avoids about ${np.mean([p[1] for p in plans]) / 1e6:.0f} million of failure
cost in five years. On these cost assumptions replacing pipes does not pay for itself within the plan; the plan only
decides how to lose least. A real case rests on longer horizons and on costs this model leaves out.
"""
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(doc)
    print(doc)


if __name__ == "__main__":
    main()
