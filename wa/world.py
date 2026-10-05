"""A deterministic synthetic water utility: 40 district metered areas (DMAs) of 500 junctions each (20,000 nodes, about
33,000 pipes), two weeks of pressure and flow telemetry with a hidden leak, a drifting sensor and a harmless demand
spike, and ten years of pipe failures drawn from a hidden hazard model. Nothing here is a real network.

Hydraulics are linearised: flow through a pipe is proportional to the head difference across it. That turns a DMA into
one sparse linear system, which is solved exactly. It is not EPANET (real head loss is non-linear in flow), but it keeps
what matters here: a leak lowers pressure most near itself, in a pattern that depends on where it is."""
import functools
import math
import random

import numpy as np
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import splu

N_DMA, NX, NY = 40, 25, 20
N = NX * NY                                     # junctions per DMA
SPACING_M = 60.0
HEAD_M = 50.0                                   # head at each DMA inlet
K_SENSORS = 10
STEP_MIN, DAYS = 15, 14
PER_DAY = 24 * 60 // STEP_MIN
T = DAYS * PER_DAY
TRAIN_DAYS = 5                                  # the first days are taken as normal behaviour
PRESSURE_NOISE, FLOW_NOISE = 0.08, 0.01         # metres; share of flow
MATERIALS = {"cast iron": (1930, 1975, 1.6), "asbestos cement": (1950, 1985, 1.2), "ductile iron": (1970, 2010, 0.7), "PVC": (1985, 2020, 0.35)}       # first year, last year, hazard multiplier
LEAK = {"dma": 7, "start_step": 9 * PER_DAY + 12, "lps": 1.8}          # starts 03:00 on day 9
DRIFT = {"dma": 21, "sensor": 4, "start_step": 6 * PER_DAY, "m_per_hour": -0.004}
SPIKE = {"dma": 30, "start_step": 11 * PER_DAY + 6, "steps": 14, "lps": 4.0}        # one night of heavy legitimate use (mains flushing)
HISTORY_YEARS = list(range(2016, 2026))


def pattern():
    """Demand through the day as a multiple of the average: low at night, peaks morning and evening."""
    h = np.arange(PER_DAY) * STEP_MIN / 60
    p = 0.45 + 0.75 * np.exp(-((h - 7.5) / 2.2) ** 2) + 0.65 * np.exp(-((h - 19) / 2.6) ** 2) + 0.25 * np.exp(-((h - 13) / 3) ** 2)
    return p / p.mean()


def dma(seed, index):
    """One district: a 25 x 20 grid of junctions with about a quarter of the non-essential pipes missing, fed from one inlet.
    -> dict with pipes, conductances, the factorised system and the sensor signature matrix."""
    r = random.Random(f"{seed}-dma-{index}")
    node = lambda x, y: y * NX + x      # noqa: E731
    edges = [(node(x, y), node(x + 1, y)) for y in range(NY) for x in range(NX - 1)] + [(node(x, y), node(x, y + 1)) for y in range(NY - 1) for x in range(NX)]
    r.shuffle(edges)
    parent = list(range(N))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    keep = []
    for a, b in edges:                                           # a random spanning tree keeps the district connected ...
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
            keep.append((a, b))
        elif r.random() < 0.75:                                  # ... and most of the remaining pipes make the loops
            keep.append((a, b))
    inlet = node(r.randrange(NX), 0)
    pipes = []
    for a, b in keep:
        material = r.choices(list(MATERIALS), [3, 2, 3, 2])[0]
        lo, hi, _ = MATERIALS[material]
        pipes.append({"a": a, "b": b, "material": material, "installed": r.randint(lo, hi), "diameter_mm": r.choice([100, 100, 150, 150, 200, 300]), "length_m": SPACING_M, "soil": round(r.random(), 2)})
    g = np.array([(p["diameter_mm"] / 100) ** 2.6 for p in pipes])                         # conductance: bigger pipes carry far more for the same head difference
    base = np.array([r.uniform(0.012, 0.05) for _ in range(N)])                              # average demand per junction, L/s
    rows = [p["a"] for p in pipes] + [p["b"] for p in pipes] + [p["a"] for p in pipes] + [p["b"] for p in pipes] + [inlet]
    cols = [p["a"] for p in pipes] + [p["b"] for p in pipes] + [p["b"] for p in pipes] + [p["a"] for p in pipes] + [inlet]
    vals = np.concatenate([g, g, -g, -g, [1e3]])                                             # the inlet is tied to the supply head through a large main
    lap = csc_matrix(coo_matrix((vals, (rows, cols)), shape=(N, N)))
    lu = splu(lap)
    drop = lu.solve(base * 1.9)                                                               # head loss at peak demand
    scale = drop.max() / 12.0                                                                 # calibrate so the far end loses about 12 m at peak
    lu = splu(lap * scale)
    sensors = [node(2 + 5 * i + r.randrange(2), 4 + 11 * j + r.randrange(3)) for j in range(2) for i in range(5)]
    eye = np.zeros((N, K_SENSORS))
    eye[sensors, range(K_SENSORS)] = 1
    sig = lu.solve(eye).T                                                                     # sig[k, j] = metres lost at sensor k per L/s drawn at junction j
    return {"index": index, "pipes": pipes, "inlet": inlet, "base": base, "sensors": sensors, "sig": sig, "lu": lu, "xy": [(i % NX, i // NX) for i in range(N)]}


def hops(d, a, b):
    """Grid distance between two junctions of a district, in pipe lengths."""
    return abs(d["xy"][a][0] - d["xy"][b][0]) + abs(d["xy"][a][1] - d["xy"][b][1])


def telemetry(d, seed, leak=None, drift=None, spike=None):
    """Two weeks at 15-minute steps for one district. -> (pressure [T, K] in metres, inflow [T] in L/s)
    leak: (junction, start step, L/s); drift: (sensor, start step, m per hour); spike: (junction, start step, steps, L/s)"""
    rng = np.random.default_rng([seed, d["index"]])
    pat = np.tile(pattern(), DAYS)
    day = np.repeat(rng.normal(1.0, 0.03, DAYS), PER_DAY)                                     # some days are simply busier
    q = d["base"][None, :] * (pat * day)[:, None] * rng.lognormal(0, 0.12, (T, N))
    extra = np.zeros((T, N))
    if leak:
        j, t0, lps = leak
        extra[t0:, j] += lps * (1 + 0.1 * np.arange(T - t0) / PER_DAY)                        # a leak grows slowly
    if spike:
        j, t0, n, lps = spike
        extra[t0:t0 + n, j] += lps
    q = q + extra
    pressure = HEAD_M - q @ d["sig"].T + rng.normal(0, PRESSURE_NOISE, (T, K_SENSORS))
    if drift:
        k, t0, rate = drift
        pressure[t0:, k] += rate * np.arange(T - t0) * STEP_MIN / 60
    inflow = q.sum(axis=1) * (1 + rng.normal(0, FLOW_NOISE, T))
    return pressure, inflow


def hazard(pipe, year, breaks):
    """Hidden truth: chance this pipe fails in a given year. Older, smaller, cast-iron pipes in corrosive soil fail more,
    and a pipe that has already broken is far more likely to break again."""
    age = year - pipe["installed"]
    base = 0.0012 * MATERIALS[pipe["material"]][2] * (1 + age / 40) ** 2.2 * (150 / pipe["diameter_mm"]) ** 0.7 * (0.6 + 0.9 * pipe["soil"])
    return min(0.6, base * (1 + 1.1 * breaks))


@functools.lru_cache(maxsize=2)
def build(seed=17, n_dma=N_DMA):
    districts = [dma(seed, i) for i in range(n_dma)]
    r = random.Random(f"{seed}-failures")
    failures = []                                                 # (district, pipe index, year)
    for d in districts:
        for k, p in enumerate(d["pipes"]):
            b = 0
            for year in HISTORY_YEARS:
                if year > p["installed"] and r.random() < hazard(p, year, b):
                    failures.append((d["index"], k, year))
                    b += 1
    return {"seed": seed, "districts": districts, "failures": failures}


def leak_node(d):
    """Where the demo's hidden leak is: a junction well inside the district."""
    return (NY // 2 + 3) * NX + NX // 2 + 4


def scenario_telemetry(w, index):
    d = w["districts"][index]
    leak = (leak_node(d), LEAK["start_step"], LEAK["lps"]) if index == LEAK["dma"] else None
    drift = (DRIFT["sensor"], DRIFT["start_step"], DRIFT["m_per_hour"]) if index == DRIFT["dma"] else None
    spike = (leak_node(d), SPIKE["start_step"], SPIKE["steps"], SPIKE["lps"]) if index == SPIKE["dma"] else None
    return telemetry(d, w["seed"], leak, drift, spike)


def pipe_length_km(w):
    return sum(p["length_m"] for d in w["districts"] for p in d["pipes"]) / 1000


def unit_cost(pipe):
    """Replacement cost per metre (assumption)."""
    return 250 + 1.6 * pipe["diameter_mm"]


def consequence(pipe):
    """Cost of one failure: repair, lost water, disruption. Bigger mains cost far more (assumption)."""
    return 6000 + 55 * pipe["diameter_mm"] + (math.sqrt(pipe["diameter_mm"]) * 900)
