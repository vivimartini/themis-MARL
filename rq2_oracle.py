"""Shared plateau-robust CMA-ES search for RQ2 best-response oracles.

All RQ2 scripts that optimize over (delta_base, delta_cov) use this module so
budget splitting, restart widening, best-ever tracking, and seeding behave
identically. CMA-ES seed must be non-zero: the cma package treats seed=0 as
"unseeded" in some versions.
"""
import numpy as np
import cma

DEFAULT_SEED = 42
DEFAULT_SIGMA0 = 12.0
DEFAULT_RESTARTS = 3

# Warm-start points for deviation search (entry / exit-shrink / nudge directions).
WARM_STARTS = [np.array(v, float) for v in
               [(0, 0), (-40, 0), (-15, -15), (-10, -20), (+20, 0), (+15, +15),
                (-25, -5), (+25, -10), (-5, +20),
                (-40, -45), (-30, -35), (-35, -40)]]   # extra exit-shrink corners


def cma_minimize(obj, budget, sigma0=DEFAULT_SIGMA0, seed=DEFAULT_SEED,
                 n_restarts=DEFAULT_RESTARTS, x0=None, warm_starts=None):
    """Minimize obj(x) over R^2 with restarted CMA-ES and global best-ever tracking."""
    x0 = np.zeros(2, float) if x0 is None else np.asarray(x0, float)
    best_f, best_x = float(obj(x0)), x0.copy()
    for xs in warm_starts or ():
        xs = np.asarray(xs, float)
        fv = float(obj(xs))
        if fv < best_f:
            best_f, best_x = fv, xs.copy()
    per = max(1, budget // n_restarts)
    for k in range(n_restarts):
        es = cma.CMAEvolutionStrategy(
            list(best_x), sigma0 * (1.5 ** k),
            {"seed": int(seed + k), "verbose": -9, "maxfevals": per},
        )
        while not es.stop():
            xs = es.ask()
            fs = [float(obj(x)) for x in xs]
            es.tell(xs, fs)
            j = int(np.argmin(fs))
            if fs[j] < best_f:
                best_f, best_x = fs[j], np.asarray(xs[j], float)
    return best_f, best_x
