"""RQ2 step 1: single-actor best-response oracle and the fixed-coverage DSIC control.

Fixed-coverage game (validation regime, reading (a)):
  - coverage frozen at the truthful operating point c* = 0.8687 (model coverage);
  - transfer rate pair frozen at the truthful solution (T+ = 0.24, T- = 0.3417);
  - the weighted-quantile price selection stays LIVE: each actor's report reduces,
    at fixed (c*, T*), to a scalar willingness, and the mechanism selects the lower
    weighted (1-c*) quantile of the nine reported scalars. Membership = report >= p.

Deviation space: (delta_base, delta_cov) perturbation of the truthful curve, searched
by CMA-ES (seeded, budgeted). regret_i = best found utility - truthful utility.

Two utility definitions are evaluated side by side, deliberately:
  'peaked'   : u_i = -|p - truth_i|  (single-peaked public-outcome preferences; the
               Moulin domain under which the fixed-coverage DSIC theorem holds).
  'transfer' : u_i = net per-capita transfer from the mechanism's own accounting
               (contributors pay T+ * p * (e-ebar); beneficiaries receive
               T- * p * (ebar-e); non-members 0).
The theorem guarantees zero regret only on the 'peaked' domain. Running both makes
the utility-definition question empirical rather than rhetorical.
"""
import numpy as np
import pandas as pd
import cma

# ---------------------------------------------------------------- truthful setup
EBAR = 6.6

T_PLUS = 0.24            # truthful contributor rate, fraction of price
T_MINUS = 0.3417         # truthful beneficiary rate, fraction of price
SEED = 42            # nonzero: cma treats seed=0 as unseeded
CMA_SIGMA0 = 12.0
CMA_BUDGET = 900          # objective evaluations per actor, split over restarts

df = pd.read_csv("actors_baseline.csv")
df.loc[df["name"] == "CHINA", "alpha_base"] = 3.27   # adopted calibration
names = df["name"].tolist()
e = df["e"].to_numpy(float)
pop = df["pop_m"].to_numpy(float)
w = pop * e
w = w / w.sum()                                      # emission weights
ab = df["alpha_base"].to_numpy(float)
ac = df["alpha_cov"].to_numpy(float)
at = df["alpha_trf"].to_numpy(float)
contrib = e > EBAR

# c* is the exact emission mass of the truthful six-member coalition (the engine's
# printed 0.8687 is a rounding of this; hardcoding the rounded value puts the
# quantile threshold on the wrong side of the pivotal US by 5e-5 of weight).
TRUTHFUL_MEMBERS = ["CHINA", "UNITED STATES", "EUROPEAN UNION", "INDIA",
                    "ADV. CARBON-PRICED CONDITIONAL JOINERS", "LOW-CARBON FRONTIER"]
C_STAR = float(w[[names.index(n) for n in TRUTHFUL_MEMBERS]].sum())

tau = np.where(contrib, -T_PLUS * (e - EBAR), T_MINUS * (EBAR - e))  # fixed-T transfer arg


def scalar_report(i, db=0.0, dc=0.0):
    """Actor i's willingness at (c*, T*) under a (delta_base, delta_cov) perturbation."""
    return max(0.0, (ab[i] + db) + (ac[i] + dc) * C_STAR + at[i] * tau[i])


TRUTH = np.array([scalar_report(i) for i in range(len(names))])


def quantile_price(reports):
    """Highest price x such that actors reporting >= x hold coverage >= c*.

    This is the engine's convention for the lower weighted (1-c) quantile and is
    tie-safe at the exact coverage boundary (the from-below cumsum construction
    misassigns the pivot when non-member mass equals 1-c* to machine precision).
    """
    order = np.argsort(-reports, kind="stable")          # descending
    cum = np.cumsum(w[order])
    k = np.searchsorted(cum, C_STAR - 1e-12)
    return reports[order[k]]


def outcome(reports):
    p = quantile_price(reports)
    members = reports >= p - 1e-12
    return p, members


def utility(i, reports, kind):
    p, members = outcome(reports)
    if kind == "peaked":                      # public outcome, single-peaked at truth
        return -abs(p - TRUTH[i])
    if kind == "transfer":                    # mechanism accounting, EUR per capita
        if not members[i]:
            return 0.0
        if contrib[i]:
            return -T_PLUS * p * (e[i] - EBAR)
        return T_MINUS * p * (EBAR - e[i])
    raise ValueError(kind)


def best_response(i, kind, sigma0=CMA_SIGMA0, budget=CMA_BUDGET, seed=SEED,
                  n_restarts=3):
    """CMA-ES over (delta_base, delta_cov); returns (regret, best_delta, best_report).

    Uses an explicit ask/tell loop with restarts and self-tracked best-ever, because
    the truthful point can sit on a fitness plateau (any report above the pivot
    leaves the quantile unchanged) and cma's optimize() terminates early on flat
    fitness before reaching the slope.
    """
    reports = TRUTH.copy()
    u_truth = utility(i, reports, kind)

    def neg_u(x):
        r = reports.copy()
        r[i] = scalar_report(i, x[0], x[1])
        return -utility(i, r, kind)

    best_f, best_x = -u_truth, np.zeros(2)          # truthful is always feasible
    per = max(1, budget // n_restarts)
    for k in range(n_restarts):
        es = cma.CMAEvolutionStrategy([0.0, 0.0], sigma0 * (1.5 ** k),
                                      {"seed": seed + k, "verbose": -9,
                                       "maxfevals": per})
        while not es.stop():
            xs = es.ask()
            fs = [neg_u(x) for x in xs]
            es.tell(xs, fs)
            j = int(np.argmin(fs))
            if fs[j] < best_f:
                best_f, best_x = fs[j], np.array(xs[j])
    u_best = -best_f
    regret = max(0.0, u_best - u_truth)
    r_best = scalar_report(i, best_x[0], best_x[1])
    return regret, best_x, r_best, u_truth, u_best


if __name__ == "__main__":
    p0, m0 = outcome(TRUTH)
    print(f"truthful fixed-coverage point: p = {p0:.2f}, members = "
          f"{[n for n, m in zip(names, m0) if m]}")
    print(f"(sanity: should equal the operating price 26.70 with the six joiners)\n")
    for kind in ("peaked", "transfer"):
        print(f"--- utility = {kind}")
        rows = []
        for i, n in enumerate(names):
            regret, x, r, ut, ub = best_response(i, kind)
            rows.append((n, TRUTH[i], r, ut, ub, regret))
        t = pd.DataFrame(rows, columns=["actor", "truthful report", "best report",
                                        "u(truth)", "u(best)", "regret"])
        print(t.round(3).to_string(index=False))
        print(f"NashConv = {t['regret'].sum():.4f}\n")
