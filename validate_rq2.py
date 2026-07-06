"""Fast regression checks for RQ2 solvers and oracles. Run: python validate_rq2.py

Exits 0 if all checks pass. Does not run the full ex-interim loop (see
rq2_exinterim_guardrails.py for that).
"""
import sys
import numpy as np
import rq2_endogenous_coverage as R
import rq2_fast as F
import rq2_fixed_coverage_control as FC
from rq2_oracle import cma_minimize, DEFAULT_SEED

TOL = 1.0   # regret/NashConv tolerance (budget=300 is a smoke test, not full 900)
errors = []


def check(label, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {label}" + (f" — {detail}" if detail else ""))
    if not ok:
        errors.append(label)


print("=== validate_rq2 ===\n")

# --- solver agreement at point calibration
t = R.TRUTHFUL
tf = F.solve(R.AB, R.AC)
p0, _ = FC.outcome(FC.TRUTH)
check("endogenous price ≈ 26.70", abs(t["p"] - 26.70) < 0.05, f"p={t['p']:.4f}")
check("endogenous c ≈ 0.8687", abs(t["c"] - 0.8687) < 0.001, f"c={t['c']:.4f}")
check("fixed vs endogenous price", abs(p0 - t["p"]) < 1e-3)
check("coarse vs fine grid", abs(tf["p"] - t["p"]) < 1e-3 and abs(tf["c"] - t["c"]) < 1e-3)
check("T rates match control", abs(t["Tplus"] - FC.T_PLUS) < 0.01
      and abs(t["Tminus"] - FC.T_MINUS) < 0.01)

# --- DSIC control: zero peaked regret
regrets_pk = [FC.best_response(i, "peaked", budget=300)[0] for i in range(9)]
nc_fixed = sum(regrets_pk)
check("fixed-coverage peaked NashConv = 0", nc_fixed < 0.01, f"NashConv={nc_fixed:.4f}")

# --- endogenous peaked NashConv (reference ~19.6)
regrets = [R.best_response(i, "peaked", budget=300)[0] for i in range(9)]
nc_endog = sum(regrets)
check("endogenous peaked NashConv ≈ 19.6", abs(nc_endog - 19.6) < TOL, f"NashConv={nc_endog:.3f}")

# --- oracle reproducibility (hybrid, coarse grid)
r_a = [F.oracle(i, R.AB, R.AC, {}, seed=DEFAULT_SEED + i)[0] for i in range(9)]
r_b = [F.oracle(i, R.AB, R.AC, {}, seed=DEFAULT_SEED + i)[0] for i in range(9)]
check("hybrid oracle reproducible", np.allclose(r_a, r_b))

# --- shared CMA search smoke test
f1, x1 = cma_minimize(lambda x: (x[0] - 3) ** 2 + (x[1] + 1) ** 2, 200, seed=DEFAULT_SEED)
check("cma_minimize finds minimum", abs(f1) < 0.5, f"f={f1:.4f}, x={x1}")

# --- obstruction headline (China damage ~69%)
from rq2_obstruction_voteselling import obstruction
cp0 = t["c"] * t["p"]
f_cn, _, _ = obstruction(R.names.index("CHINA"), budget=300)
dmg_cn = 100 * (cp0 - f_cn) / cp0
check("China obstruction damage ≈ 69%", abs(dmg_cn - 68.9) < 2.0, f"{dmg_cn:.1f}%")

print()
if errors:
    print(f"{len(errors)} check(s) failed: {', '.join(errors)}")
    sys.exit(1)
print("All checks passed.")
