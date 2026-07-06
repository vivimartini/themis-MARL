"""RQ2 step 8: PSRO-lite empirical game with multi-population alpha-Rank.

Policy sets are seeded from the deviations DISCOVERED by the best-response oracle at
the point calibration (truthful for every actor; plus its attack where regret > 0).
The joint game is evaluated exactly through the Themis solver (peaked utility), then
analysed with multi-population alpha-Rank (Omidshafiei et al. 2019), which is well
defined for general-sum many-player games without assuming Nash existence. One
PSRO expansion round follows: each actor's oracle searches for a best response to
the alpha-Rank meta-distribution; improving policies are added and the game
re-solved. The whole pipeline runs under the baseline design and under the pool-cap
guardrail (the strongest single guardrail from the ablation).

Outputs: alpha-Rank mass on the all-truthful profile, top-ranked profiles, restricted
-game NashConv of truthful, and the baseline-vs-guarded comparison.
"""
import itertools
import numpy as np
import pandas as pd
import rq2_fast as F
import rq2_endogenous_coverage as R

M_POP = 50           # alpha-Rank population size
SEED = 42


def seed_policies(kw):
    """Truthful + discovered point-calibration attack (if profitable) per actor."""
    pis = []
    for i in range(9):
        reg, x, _ = F.oracle(i, R.AB, R.AC, kw, cma_budget=120, sigma0=12.0,
                             seed=SEED + i)
        pis.append([np.zeros(2)] + ([np.array(x)] if reg > 0.05 else []))
    return pis


def payoff_tensor(pis, kw):
    shapes = [len(p) for p in pis]
    U = np.zeros((int(np.prod(shapes)), 9))
    profiles = list(itertools.product(*[range(s) for s in shapes]))
    for idx, prof in enumerate(profiles):
        ab = R.AB.copy(); ac = R.AC.copy()
        for i, k in enumerate(prof):
            ab[i] += pis[i][k][0]; ac[i] += pis[i][k][1]
        out = F.solve(ab, ac, **kw)
        # each actor's peaked utility against ITS OWN truth (the calibration truth)
        for i in range(9):
            U[idx, i] = F.u_peaked(i, out, R.AB, R.AC)
    return profiles, U, shapes


def alpha_rank(profiles, U, shapes, alpha):
    """Multi-population alpha-Rank stationary distribution over joint profiles."""
    S = len(profiles)
    index = {p: j for j, p in enumerate(profiles)}
    T = np.zeros((S, S))
    for j, prof in enumerate(profiles):
        stay = 1.0
        for i in range(9):
            n_i = shapes[i]
            if n_i == 1:
                continue
            for tau in range(n_i):
                if tau == prof[i]:
                    continue
                q = list(prof); q[i] = tau
                jq = index[tuple(q)]
                d = U[jq, i] - U[j, i]
                if abs(d) < 1e-12:
                    rho = 1.0 / M_POP
                else:
                    x = alpha * d
                    if x > 0:
                        rho = (1.0 if x > 500 else 1 - np.exp(-x)) if M_POP * x > 500 \
                            else (1 - np.exp(-x)) / (1 - np.exp(-M_POP * x))
                    else:
                        rho = 0.0 if -M_POP * x > 500 \
                            else (1 - np.exp(-x)) / (1 - np.exp(-M_POP * x))
                pr = (1.0 / 9) * (1.0 / (n_i - 1)) * rho
                T[j, jq] += pr
                stay -= pr
        T[j, j] = stay
    vals, vecs = np.linalg.eig(T.T)
    k = np.argmin(np.abs(vals - 1.0))
    pi = np.real(vecs[:, k]); pi = np.abs(pi); pi /= pi.sum()
    return pi


def restricted_nashconv(profiles, U, shapes, base_idx):
    """Sum over actors of best unilateral improvement from the given profile."""
    index = {p: j for j, p in enumerate(profiles)}
    prof = profiles[base_idx]
    total = 0.0
    for i in range(9):
        best = U[base_idx, i]
        for tau in range(shapes[i]):
            q = list(prof); q[i] = tau
            best = max(best, U[index[tuple(q)], i])
        total += best - U[base_idx, i]
    return total


def expand_once(pis, kw, sigma):
    """One PSRO round: best response to the alpha-Rank meta-distribution."""
    profiles, U, shapes = sigma["profiles"], sigma["U"], sigma["shapes"]
    dist = sigma["pi"]
    top = np.argsort(-dist)[:8]                     # support approximation
    added = []
    for i in range(9):
        def exp_u(x):
            tot = 0.0
            for j in top:
                ab = R.AB.copy(); ac = R.AC.copy()
                for a, k in enumerate(profiles[j]):
                    ab[a] += pis[a][k][0]; ac[a] += pis[a][k][1]
                ab[i] = R.AB[i] + x[0]; ac[i] = R.AC[i] + x[1]
                tot += dist[j] * F.u_peaked(i, F.solve(ab, ac, **kw), R.AB, R.AC)
            return tot / dist[top].sum()
        base = max(exp_u(p) for p in pis[i])
        best_u, best_x = base, None
        rng = np.random.default_rng(SEED + 100 + i)
        for x in F.PORTFOLIO + [rng.uniform(-60, 60, 2) for _ in range(20)]:
            u = exp_u(x)
            if u > best_u + 0.05:
                best_u, best_x = u, np.array(x)
        if best_x is not None:
            pis[i].append(best_x)
            added.append(R.names[i])
    return pis, added


def run(kw, label):
    print(f"===== {label} =====")
    pis = seed_policies(kw)
    sizes = {R.names[i]: len(pis[i]) for i in range(9) if len(pis[i]) > 1}
    print(f"seeded attack policies: {sizes}")
    profiles, U, shapes = payoff_tensor(pis, kw)
    truthful_idx = profiles.index(tuple([0] * 9))
    print(f"restricted-game NashConv of all-truthful: "
          f"{restricted_nashconv(profiles, U, shapes, truthful_idx):.2f}")
    for alpha in (0.05, 0.5, 5.0):
        pi = alpha_rank(profiles, U, shapes, alpha)
        mass_truth = pi[truthful_idx]
        top = np.argsort(-pi)[:3]
        desc = "; ".join(
            f"{pi[j]:.2f} on {' ,'.join(R.names[i][:8]+':'+('T' if k==0 else 'A'+str(k)) for i,k in enumerate(profiles[j]) if k>0) or 'ALL-TRUTHFUL'}"
            for j in top)
        print(f"alpha={alpha:<5}: truthful mass {mass_truth:.3f} | top: {desc}")
    # one PSRO expansion round at the highest alpha
    pi = alpha_rank(profiles, U, shapes, 5.0)
    sigma = dict(profiles=profiles, U=U, shapes=shapes, pi=pi)
    pis, added = expand_once(pis, kw, sigma)
    if added:
        profiles, U, shapes = payoff_tensor(pis, kw)
        truthful_idx = profiles.index(tuple([0] * 9))
        pi = alpha_rank(profiles, U, shapes, 5.0)
        print(f"PSRO round 1 added BR policies for: {added}")
        print(f"after expansion: truthful mass {pi[truthful_idx]:.3f}, "
              f"restricted NashConv of truthful "
              f"{restricted_nashconv(profiles, U, shapes, truthful_idx):.2f}")
    else:
        print("PSRO round 1: no actor found an improving response to the "
              "meta-distribution (policy set is a fixed point at this budget)")
    print()


if __name__ == "__main__":
    run({}, "baseline design")
    run({"pool_cap": 150000.0}, "pool-cap guardrail (<= 150bn)")
