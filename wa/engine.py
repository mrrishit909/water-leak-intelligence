"""Leak detection and localisation, pipe-failure risk and the capital plan."""
import numpy as np
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier

from . import world

VERSION = "wa-1.0"
NIGHT = slice(8, 16)                 # 02:00 to 04:00: demand is lowest and steadiest, so a leak shows best
ALARM_Z, SPIKE_Z, SENSOR_Z = 4.0, 6.0, 6.0
WATER_COST_M3 = 1.5                  # dollars per cubic metre lost (assumption)
MATERIAL_INDEX = {m: i for i, m in enumerate(world.MATERIALS)}


# ---------------------------------------------------------------- detection

def nightly(series):
    """Mean of each night's 02:00 to 04:00 window. series [T] or [T, K] -> [days] or [days, K]"""
    x = np.asarray(series, dtype=float)
    return x.reshape(world.DAYS, world.PER_DAY, *x.shape[1:])[:, NIGHT].mean(axis=1)


def localise(drop, sig, top=5):
    """Which junction, leaking how much, best explains the pressure drops seen at the sensors?
    For each junction j the best leak size is a least-squares scale on its signature; candidates are ranked by what is
    left unexplained. drop [K] metres (positive = pressure lost), sig [K, N]. -> [(junction, L/s, fit 0..1)]"""
    scale = (sig * drop[:, None]).sum(axis=0) / (sig ** 2).sum(axis=0)
    err = np.linalg.norm(drop[:, None] - sig * scale[None, :], axis=0) / max(1e-9, np.linalg.norm(drop))
    err[scale <= 0] = np.inf
    order = np.argsort(err)[:top]
    return [(int(j), float(scale[j]), float(max(0.0, 1 - err[j]))) for j in order if np.isfinite(err[j])]


def nearest_sensor(drop, d):
    """Baseline localiser: the leak is at the sensor that lost the most pressure."""
    return d["sensors"][int(np.argmax(drop))]


def analyse(pressure, inflow, d):
    """Classify one district's fortnight. -> {kind: normal | leak | sensor_fault | transient_demand, ...evidence}
    A leak raises night flow and stays, and lowers pressure across sensors in the pattern of a single junction.
    A faulty sensor moves alone while the flow meter and its neighbours see nothing. A demand spike raises flow and goes away."""
    nf, npr = nightly(inflow), nightly(pressure)
    whole = np.asarray(inflow, dtype=float).reshape(world.DAYS, world.PER_DAY).mean(axis=1)
    tr = world.TRAIN_DAYS
    # Some days are simply busier, which lifts night flow too. A leak is different: it adds the same amount at 3 am as at
    # noon. If nights are normally a share r of the day's average, a constant extra L satisfies night = r * (day - L) + L.
    r = float((nf[:tr] / whole[:tr]).mean())
    excess = (nf - r * whole) / (1 - r)
    mu = float(nf[:tr].mean())
    z = (excess - excess[:tr].mean()) / max(excess[:tr].std(ddof=1), 0.1)
    # Pressure sags in proportion to demand, so a busy night lowers every sensor a little. Scale each sensor's normal sag by
    # that night's demand (inflow minus whatever constant extra was found above) before calling anything unusual.
    busy = (nf - (excess - excess[:tr].mean())) / mu
    sag = world.HEAD_M - npr[:tr].mean(axis=0)
    resid = npr - (world.HEAD_M - sag[None, :] * busy[:, None])
    psd = np.maximum(resid[:tr].std(axis=0, ddof=1), 0.04)
    last = resid[-2:].mean(axis=0)
    pz = last / psd
    out = {"kind": "normal", "night_flow_lps": [round(float(v), 2) for v in nf], "night_flow_normal": round(mu, 2), "excess_lps": [round(float(v), 2) for v in excess], "night_flow_z": [round(float(v), 1) for v in z],
           "pressure_residual_m": [[round(float(v), 3) for v in row] for row in resid], "sensor_z": [round(float(v), 1) for v in pz]}
    odd = np.abs(pz) > SENSOR_Z
    cands = localise(-last, d["sig"])
    if (z[-2:] > ALARM_Z).all() and cands:
        onset = world.DAYS - 2
        while onset > tr and z[onset - 1] > ALARM_Z / 2:
            onset -= 1
        shift = float(excess[-2:].mean())
        out.update(kind="leak", onset_day=int(onset), flow_rise_lps=round(shift, 2), candidates=[{"node": j, "lps": round(a, 2), "fit": round(f, 3), "x": d["xy"][j][0], "y": d["xy"][j][1]} for j, a, f in cands],
                   probability=round(float(min(0.99, 0.5 + 0.5 * cands[0][2])), 3), nights_elevated=int((z[onset:] > ALARM_Z).sum()))
    elif z[tr:].max() > SPIKE_Z and not (z[-2:] > ALARM_Z).all():
        day = int(tr + np.argmax(z[tr:]))
        out.update(kind="transient_demand", onset_day=day, flow_rise_lps=round(float(nf[day] - mu), 2), nights_elevated=int((z > ALARM_Z).sum()),
                   reason="night flow jumped on one night and returned to normal: a leak does not repair itself")
    elif odd.sum() == 1 and np.abs(np.delete(pz, int(np.argmax(np.abs(pz))))).max() < ALARM_Z:
        k = int(np.argmax(np.abs(pz)))
        days = np.arange(world.DAYS)
        slope = float(np.polyfit(days[tr:], resid[tr:, k], 1)[0])
        out.update(kind="sensor_fault", sensor=k, residual_m=round(float(last[k]), 3), drift_m_per_day=round(slope, 3), others_max_z=round(float(np.abs(np.delete(pz, k)).max()), 1),
                   flow_z=round(float(z[-2:].mean()), 1), leak_fit=round(cands[0][2], 3) if cands else 0.0,
                   reason="one sensor is moving on its own: the flow meter shows no extra water entering, and no single leak would lower this sensor without lowering its neighbours")
    elif odd.any() or (z[-2:] > ALARM_Z).all():
        out.update(kind="unexplained", reason="readings moved in a way that fits neither a leak, a single faulty sensor nor a night of demand")
    return out


def loss(lps, onset_day, days=world.DAYS):
    m3 = lps * 86.4 * (days - onset_day)
    return {"cubic_metres_so_far": round(m3), "cost_so_far": round(m3 * WATER_COST_M3), "per_year_if_unrepaired_m3": round(lps * 86.4 * 365), "per_year_cost": round(lps * 86.4 * 365 * WATER_COST_M3)}


# ---------------------------------------------------------------- failure risk

def features(pipe, year, breaks):
    one_hot = [0, 0, 0, 0]
    one_hot[MATERIAL_INDEX[pipe["material"]]] = 1
    return [year - pipe["installed"], pipe["diameter_mm"], pipe["soil"], breaks, *one_hot]


FEATURE_NAMES = ["age (years)", "diameter (mm)", "soil corrosivity", "earlier breaks", *[f"material: {m}" for m in world.MATERIALS]]


def pipe_years(w, years):
    """One row per pipe per year: the features as they stood at the start of that year, and whether it failed in it."""
    failed = {}
    for dma, k, year in w["failures"]:
        failed.setdefault((dma, k), []).append(year)
    x, y, keys = [], [], []
    for d in w["districts"]:
        for k, p in enumerate(d["pipes"]):
            fs = failed.get((d["index"], k), [])
            for year in years:
                if year > p["installed"]:
                    x.append(features(p, year, sum(f < year for f in fs)))
                    y.append(year in fs)
                    keys.append((d["index"], k, year))
    return np.array(x, dtype=float), np.array(y), keys


def train_risk(w, through=2022):
    """Gradient-boosted trees on pipe-years up to `through`. -> (model, metrics on the later years)"""
    xa, ya, _ = pipe_years(w, [y for y in world.HISTORY_YEARS if y <= through])
    model = HistGradientBoostingClassifier(max_iter=120, learning_rate=0.08, max_depth=4, random_state=0).fit(xa, ya)
    xb, yb, _ = pipe_years(w, [y for y in world.HISTORY_YEARS if y > through])
    p = model.predict_proba(xb)[:, 1]
    return model, {"trained_on_pipe_years": int(len(ya)), "tested_on_pipe_years": int(len(yb)), "failures_in_test": int(yb.sum()), "auc": round(auc(yb, p), 4), "auc_age_only": round(auc(yb, xb[:, 0]), 4),
                   "top_5pct_capture": round(capture(yb, p), 4), "top_5pct_capture_age_only": round(capture(yb, xb[:, 0]), 4), "mean_predicted": round(float(p.mean()), 5), "observed_rate": round(float(yb.mean()), 5)}


def auc(y, p):
    ranks, pos = rankdata(p), y.sum()
    return float((ranks[y].sum() - pos * (pos + 1) / 2) / (pos * (len(y) - pos)))


def capture(y, p, share=0.05):
    """Share of all failures found among the `share` of pipe-years ranked riskiest."""
    top = np.argsort(-p, kind="stable")[:int(len(p) * share)]
    return float(y[top].sum() / max(1, y.sum()))


# ---------------------------------------------------------------- capital plan

def avoided(p, c, start, years, growth=1.04):
    """Expected failure cost a replacement in year `start` avoids up to the horizon (risk creeps up as the pipe ages)."""
    return sum(min(1.0, p * growth ** t) * c for t in range(start, years))


def plan(pipes, budget_per_year, years=5, order=None):
    """Choose pipes to replace each year. pipes: [(key, chance of failing next year, cost of one failure $, replacement $)].
    Pipes are taken in order of failure cost avoided per dollar and put in the earliest year with money left.
    `order` swaps in another ranking (the oldest-first baseline). -> [{key, year, cost, avoided}]"""
    # ponytail: greedy knapsack, no crew or street-works constraints. bound() shows the gap to the best possible; move to an integer program if constraints arrive.
    left, items, cheapest = [budget_per_year] * years, [], min(x[3] for x in pipes)
    for key, p, c, cost in sorted(pipes, key=order or (lambda x: -avoided(x[1], x[2], 0, years) / x[3])):
        if max(left) < cheapest:
            break
        y = next((y for y in range(years) if left[y] >= cost), None)
        if y is not None:
            left[y] -= cost
            items.append({"key": key, "year": y + 1, "cost": round(cost), "avoided": round(avoided(p, c, y, years))})
    return items


def worth(items, chance, years=5):
    """Failure cost a plan avoids when each pipe's chance of failing is chance[key] = (p, consequence)."""
    return round(sum(avoided(*chance[i["key"]], i["year"] - 1, years) for i in items))


def bound(pipes, budget_per_year, years=5):
    """More than any plan could avoid: every dollar spent in year one, pipes allowed to be part-replaced."""
    room, total = budget_per_year * years, 0.0
    for key, p, c, cost in sorted(pipes, key=lambda x: -avoided(x[1], x[2], 0, years) / x[3]):
        take = min(1.0, room / cost)
        total += take * avoided(p, c, 0, years)
        room -= take * cost
        if room <= 0:
            break
    return round(total)
