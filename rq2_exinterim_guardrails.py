"""Driver for RQ2 experiments A (ex-interim regret over the scenario prior),
B (guardrail ablations) and C (optimizer comparison). Requires rq2_fast.py and
rq2_endogenous_coverage.py. Runtime ~10-15 min. Seeds fixed throughout; see
RQ2_step5_7_notes.md for the reference outputs."""
from pathlib import Path
import numpy as np, pandas as pd
import rq2_fast as F, rq2_endogenous_coverage as R
import cma
from rq2_oracle import cma_minimize, DEFAULT_SEED, WARM_STARTS, EXIT_SHRINK_STARTS

ROOT = Path(__file__).resolve().parent

# Uncertainty ranges for alpha_base by actor (from the scenario prior).
# Actors not listed keep the point estimate (lo = hi = 0).
ab_lo = np.zeros(9); ab_hi = np.zeros(9)
for n, (lo, hi) in {"CHINA": (0, 8.63), "UNITED STATES": (2, 4),
                    "EUROPEAN UNION": (15, 25),
                    "ADV. CARBON-PRICED CONDITIONAL JOINERS": (4, 8),
                    "INDIA": (0, 4)}.items():
    j = R.names.index(n); ab_lo[j], ab_hi[j] = lo, hi

# ---- A: ex-interim regret, 120 worlds
# Each world draws a full preference profile; we then run the hybrid oracle for
# every actor and label the profitable deviation (if any).
rng = np.random.default_rng(7)
REG = np.zeros((120, 9)); MODE = np.empty((120, 9), dtype=object)
for d in range(120):
    ab = rng.uniform(ab_lo, ab_hi)
    # Coverage sensitivity gets a wider shake (lognormal) than the base intercept.
    ac = R.AC * rng.uniform(0.5, 1.5) * rng.lognormal(0, 0.2, 9)
    for i in range(9):
        reg, x, tout = F.oracle(i, ab, ac, {}, seed=42 + d * 9 + i)
        REG[d, i] = reg; MODE[d, i] = F.attack_mode(i, x, tout, ab, ac, {})
rows = []
for i, n in enumerate(R.names):
    r = REG[:, i]; pos = r > 0.01
    modes = pd.Series(list(MODE[pos, i])).value_counts()
    rows.append((n, round(r.mean(), 2), round(np.median(r), 2),
                 round(np.percentile(r, 95), 2), f"{100*pos.mean():.0f}%",
                 modes.index[0] if len(modes) else "-"))
print(pd.DataFrame(rows, columns=["actor", "mean", "median", "95th", "%>0",
                                  "dominant attack"]).to_string(index=False))
nc = REG.sum(1)
print(f"ex-interim NashConv: mean {nc.mean():.2f}, median {np.median(nc):.2f}, "
      f"5-95%: {np.percentile(nc,5):.1f}-{np.percentile(nc,95):.1f}\n")

# Persist distributional outputs for figures_rq2.py (Figs 4, 9).
# Store attack labels as fixed-width strings (no pickle needed on load).
MODE_STR = np.array(MODE, dtype="U12")
np.savez(ROOT / "exA_final.npz", REG=REG, MODE=MODE_STR)
print(f"saved {ROOT / 'exA_final.npz'}")

# ---- B: guardrails
def u_tr(i, o):
    """Per-capita transfer utility at outcome o (same accounting as step 1)."""
    if o is None or not o["members"][i]: return 0.0
    return (-o["Tplus"] * o["p"] * (R.e[i] - R.EBAR) if R.contrib[i]
            else o["Tminus"] * o["p"] * (R.EBAR - R.e[i]))

designs = [("baseline", {}), ("T- <= 1", {"tminus_cap": 1.0}),
           ("c >= 0.5", {"cov_floor": 0.5}), ("pool <= 150bn", {"pool_cap": 150000.0}),
           ("all three", {"tminus_cap": 1.0, "cov_floor": 0.5, "pool_cap": 150000.0})]
rows = []
for name, kw in designs:
    t0 = F.solve(R.AB, R.AC, **kw)
    # Endogenous NashConv under this design (coarse grid, equal oracle budget).
    nash = sum(F.oracle(i, R.AB, R.AC, kw, cma_budget=120, sigma0=12.0,
                        seed=42 + i)[0] for i in range(9))
    cp0 = t0["c"] * t0["p"]; worst = 0.0
    # Obstruction: each actor tries to minimise c*p (warm-started, seed-robust).
    for i in range(9):
        def neg_cp(x, i=i):
            a1 = R.AB.copy(); a2 = R.AC.copy(); a1[i] += x[0]; a2[i] += x[1]
            o = F.solve(a1, a2, **kw)
            return 0.0 if o is None else o["c"] * o["p"]
        f, _ = cma_minimize(neg_cp, budget=450, seed=DEFAULT_SEED + i,
                            warm_starts=WARM_STARTS + EXIT_SHRINK_STARTS)
        worst = max(worst, cp0 - f)
    def deal(a, b):
        # Buyer a, seller b: maximise joint transfer surplus over b's report.
        # NOTE: this is the exact search configuration (portfolio + frontier-China
        # warm start, single CMA run, seed 42, 200 evals) behind the reported
        # guardrail collusion column (949.6 / 158.4 / 158.4 / 125.8 / 125.8).
        # Do not "strengthen" it without regenerating the dissertation numbers.
        uA0, uB0 = u_tr(a, t0), u_tr(b, t0)
        def nj(x):
            a1 = R.AB.copy(); a2 = R.AC.copy(); a1[b] += x[0]; a2[b] += x[1]
            o = F.solve(a1, a2, **kw)
            return -((u_tr(a, o) - uA0) + (u_tr(b, o) - uB0))
        f = min(nj(x) for x in WARM_STARTS + [np.array([145.7, 112.6])])
        es = cma.CMAEvolutionStrategy([0., 0.], 40., {"seed": 42, "verbose": -9,
                                                      "maxfevals": 200})
        while not es.stop():
            xs = es.ask(); fs = [nj(x) for x in xs]; es.tell(xs, fs)
            f = min(f, min(fs))
        return max(0.0, -f)
    fc = deal(R.names.index("LOW-CARBON FRONTIER"), R.names.index("CHINA"))
    ui = deal(R.names.index("UNITED STATES"), R.names.index("INDIA"))
    rows.append((name, round(nash, 2), f"{100*worst/cp0:.1f}%", round(fc, 1),
                 round(ui, 2)))
print(pd.DataFrame(rows, columns=["design", "endog NashConv", "max obstruction",
                                  "frontier-China", "US-India"]).to_string(index=False))

# ---- C: CMA+portfolio hybrid vs random search, equal budget
# Hybrid: ~8 portfolio + 40 random + 850 CMA evals. Random baseline: 900 evals.
rng2 = np.random.default_rng(3); rows = []
for i, n in enumerate(R.names):
    regC, _, _ = F.oracle(i, R.AB, R.AC, {}, cma_budget=850, sigma0=12.0,
                          seed=42 + i)
    t = F.solve(R.AB, R.AC); u0 = F.u_peaked(i, t, R.AB, R.AC); bu = u0
    for _ in range(900):
        xx = rng2.uniform(-60, 60, 2)
        a1 = R.AB.copy(); a2 = R.AC.copy(); a1[i] += xx[0]; a2[i] += xx[1]
        bu = max(bu, F.u_peaked(i, F.solve(a1, a2), R.AB, R.AC))
    rows.append((n, round(regC, 3), round(max(0.0, bu - u0), 3)))
tc = pd.DataFrame(rows, columns=["actor", "hybrid regret", "random regret"])
print(tc.to_string(index=False))
print(f"NashConv: hybrid {tc.iloc[:,1].sum():.2f} vs random {tc.iloc[:,2].sum():.2f}")
