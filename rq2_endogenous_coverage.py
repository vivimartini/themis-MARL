"""RQ2 step 2: single-actor exploitability of the ENDOGENOUS-coverage mechanism.

Each candidate deviation (delta_base, delta_cov) by actor i is evaluated through the
full self-consistent solver: exhaustive enumeration of the 512 nine-actor coalitions,
transfer-rate fraction T on a grid, price = minimum member willingness, self-
consistency = no willing non-member, objective argmax c*p. A misreport can now move
the coalition, the coverage, the rate pair and the price -- the channel outside the
fixed-coverage DSIC theorem.

Utilities (as in step 1, now with moving targets):
  'peaked'   : u_i = -|p - p_i^true(c_realised, T_realised)|  (public outcome; the
               truthful target moves with the realised configuration).
  'transfer' : mechanism accounting at the realised outcome, EUR per capita
               (budget-balanced here, unlike the frozen-T control).

Oracle: plateau-robust restart CMA-ES (explicit ask/tell, best-ever tracked),
identical to the validated step-1 oracle. regret_i = max(0, u_best - u_truth).
"""
import numpy as np
import pandas as pd
import cma

EBAR = 6.6
SEED = 42            # nonzero: cma treats seed=0 as unseeded
SIGMA0 = 12.0
BUDGET = 900            # full-solver evaluations per actor, split over restarts
RESTARTS = 3

df = pd.read_csv("actors_baseline.csv")
df.loc[df["name"] == "CHINA", "alpha_base"] = 3.27
names = df["name"].tolist()
e = df["e"].to_numpy(float)
pop = df["pop_m"].to_numpy(float)
w = pop * e; w = w / w.sum()
AB = df["alpha_base"].to_numpy(float)
AC = df["alpha_cov"].to_numpy(float)
AT = df["alpha_trf"].to_numpy(float)
N = 9
contrib = e > EBAR

# ------------------------------------------------ vectorised self-consistent solver
masks = []
for m in range(1, 2 ** N):
    mem = np.array([(m >> j) & 1 for j in range(N)], bool)
    if mem.sum() < 2 or not (mem & contrib).any() or not (mem & ~contrib).any():
        continue
    masks.append(mem)
M = np.array(masks)
COV = (M * w).sum(1)
EX = (M * np.maximum(e - EBAR, 0) * pop).sum(1)
DE = (M * np.maximum(EBAR - e, 0) * pop).sum(1)
T_GRID = np.round(np.linspace(0, 1, 101), 4)[1:]
TMINUS = T_GRID[None, :] * EX[:, None] / DE[:, None]
TAU_C = -T_GRID[None, :, None] * np.maximum(e - EBAR, 0)[None, None, :]
TAU_B = TMINUS[:, :, None] * np.maximum(EBAR - e, 0)[None, None, :]
TAU = np.where(contrib[None, None, :], TAU_C, TAU_B)     # [K, T, N]
MB = M[:, None, :]
BIG = 1e9


def solve_full(ab, ac):
    """Endogenous-coverage operating point for a full report profile (ab, ac)."""
    prefs = np.maximum(0, ab[None, None, :] + ac[None, None, :] * COV[:, None, None]
                       + AT[None, None, :] * TAU)
    price = np.where(MB, prefs, BIG).min(2)
    nonmax = np.where(~MB, prefs, -BIG).max(2)
    ok = (price > 0) & (nonmax < price + 0.01)
    obj = np.where(ok, COV[:, None] * price, -1.0)
    k, t = np.unravel_index(np.argmax(obj), obj.shape)
    if obj[k, t] < 0:
        return None                                       # no self-consistent point
    return dict(p=float(price[k, t]), c=float(COV[k]),
                Tplus=float(T_GRID[t]), Tminus=float(TMINUS[k, t]),
                members=M[k].copy())


def true_willingness(i, c, Tplus, Tminus):
    tau = -Tplus * (e[i] - EBAR) if contrib[i] else Tminus * (EBAR - e[i])
    return max(0.0, AB[i] + AC[i] * c + AT[i] * tau)


def utility(i, out, kind):
    if out is None:
        return -1e6                                       # mechanism failure: avoid
    p, c, Tp, Tm = out["p"], out["c"], out["Tplus"], out["Tminus"]
    if kind == "peaked":
        return -abs(p - true_willingness(i, c, Tp, Tm))
    if kind == "transfer":
        if not out["members"][i]:
            return 0.0
        if contrib[i]:
            return -Tp * p * (e[i] - EBAR)
        return Tm * p * (EBAR - e[i])
    raise ValueError(kind)


TRUTHFUL = solve_full(AB, AC)


def best_response(i, kind, sigma0=SIGMA0, budget=BUDGET, seed=SEED,
                  n_restarts=RESTARTS):
    u_truth = utility(i, TRUTHFUL, kind)

    def neg_u(x):
        ab = AB.copy(); ac = AC.copy()
        ab[i] += x[0]; ac[i] += x[1]
        return -utility(i, solve_full(ab, ac), kind)

    best_f, best_x = -u_truth, np.zeros(2)
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
    return max(0.0, u_best - u_truth), best_x, u_truth, u_best


if __name__ == "__main__":
    t = TRUTHFUL
    print(f"truthful endogenous point: p = {t['p']:.2f}, c_model = {t['c']:.4f}, "
          f"T+ = {t['Tplus']:.2f}, T- = {t['Tminus']:.4f}")
    print(f"members: {[n for n, m in zip(names, t['members']) if m]}")
    print("(sanity: 26.70 / 0.8687 / 0.24 / 0.3417 with the six joiners)\n")
    for kind in ("peaked", "transfer"):
        print(f"--- utility = {kind} (endogenous coverage)")
        rows = []
        for i, n in enumerate(names):
            regret, x, ut, ub = best_response(i, kind)
            # describe the best deviation's realised outcome
            ab = AB.copy(); ac = AC.copy(); ab[i] += x[0]; ac[i] += x[1]
            o = solve_full(ab, ac)
            desc = (f"p={o['p']:.2f}, c={o['c']:.3f}" if o else "invalid")
            rows.append((n, round(ut, 3), round(ub, 3), round(regret, 3),
                         f"({x[0]:+.1f},{x[1]:+.1f})", desc))
        tb = pd.DataFrame(rows, columns=["actor", "u(truth)", "u(best)", "regret",
                                         "best (db,dc)", "realised outcome"])
        print(tb.to_string(index=False))
        print(f"NashConv = {tb['regret'].sum():.3f}\n")
