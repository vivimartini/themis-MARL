"""Dissertation figures for RQ1 (one) and RQ2 (six). All values are outputs of the
seeded scripts named in each figure's provenance comment; distributional data are
read from the saved arrays. Regenerate everything with: python3 figures_rq2.py
Requires: mc_p.npz (from mc_scenario_prior.py draws) and exA_final.npz (from the
ex-interim run in rq2_exinterim_guardrails.py).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150})
OUT = "/mnt/user-data/outputs/figures/"
import os; os.makedirs(OUT, exist_ok=True)

ACTORS = ["China", "US", "EU", "India", "Russia", "Indonesia",
          "Adv. joiners", "Frontier", "Rentiers"]
MODE_COLOR = {"entry": "#2a7fba", "exit-shrink": "#c0392b",
              "extortion": "#8e44ad", "nudge": "#7f8c8d", "none": "#bdc3c7"}

# ---- Fig 1 (RQ1): MC operating-price distribution [mc_scenario_prior.py, seed 0]
d = np.load("/home/claude/mc_p.npz")
fig, ax = plt.subplots(figsize=(5.6, 3.0))
ax.hist(d["p"], bins=60, color="#4a7c9b", alpha=0.85)
ax.axvline(26.70, color="k", lw=1.2, ls="--")
ax.text(27.1, ax.get_ylim()[1] * 0.92, "point estimate €26.70", fontsize=8)
ax.axvline(20.97, color="#c0392b", lw=1.2, ls=":")
ax.text(12.3, ax.get_ylim()[1] * 0.92, "published €20.97", fontsize=8,
        color="#c0392b")
ax.set_xlabel("operating price $p^*$ (EUR/tCO$_2$e)")
ax.set_ylabel("draws")
fig.tight_layout(); fig.savefig(OUT + "fig_mc_pstar_hist.pdf"); plt.close(fig)

# ---- Fig 2 (RQ2 headline): fixed vs endogenous NashConv
# [rq2_fixed_coverage_control.py / rq2_endogenous_coverage.py, seed 42, fine grid]
fig, ax = plt.subplots(figsize=(3.4, 3.0))
ax.bar(["fixed coverage\n(DSIC control)", "endogenous\ncoverage"],
       [0.0, 19.52], color=["#4a7c9b", "#c0392b"], width=0.55)
ax.set_ylabel("NashConv (peaked domain)")
ax.text(0, 0.4, "0.0000", ha="center", fontsize=9)
ax.text(1, 19.9, "19.52", ha="center", fontsize=9)
fig.tight_layout(); fig.savefig(OUT + "fig_nashconv_headline.pdf"); plt.close(fig)

# ---- Fig 3: regret geography at the point calibration (fine grid)
# [rq2_endogenous_coverage.py, seed 42]; colors = attack mode
REGRETS = {"Indonesia": (9.21, "entry"), "EU": (4.86, "exit-shrink"),
           "China": (3.51, "exit-shrink"), "Rentiers": (1.25, "entry"),
           "Adv. joiners": (0.65, "exit-shrink"), "India": (0.04, "nudge"),
           "US": (0.0, "none"), "Russia": (0.0, "none"), "Frontier": (0.0, "none")}
names = list(REGRETS); vals = [REGRETS[n][0] for n in names]
cols = [MODE_COLOR[REGRETS[n][1]] for n in names]
fig, ax = plt.subplots(figsize=(5.6, 3.0))
y = np.arange(len(names))[::-1]
ax.barh(y, vals, color=cols)
ax.set_yticks(y); ax.set_yticklabels(names)
ax.set_xlabel("best-response regret (peaked domain)")
for m, c in [("entry", MODE_COLOR["entry"]),
             ("exit-shrink", MODE_COLOR["exit-shrink"]),
             ("nudge", MODE_COLOR["nudge"])]:
    ax.bar(0, 0, color=c, label=m)
ax.legend(frameon=False, fontsize=8, loc="lower right")
ax.annotate("pivots (US, India): locked in", xy=(0.15, 2.6), fontsize=8,
            color="#555")
fig.tight_layout(); fig.savefig(OUT + "fig_regret_geography.pdf"); plt.close(fig)

# ---- Fig 4: ex-interim exploitability over the scenario prior
# [rq2_exinterim_guardrails.py, hybrid oracle, 120 worlds, coarse grid]
A = np.load("/home/claude/exA_final.npz", allow_pickle=True)
REG, MODE = A["REG"], A["MODE"]
fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.9),
                         gridspec_kw={"width_ratios": [1.1, 1.4]})
nc = REG.sum(1)
axes[0].hist(nc, bins=24, color="#4a7c9b", alpha=0.85)
axes[0].axvline(19.5, color="k", ls="--", lw=1.1)
axes[0].text(20.2, axes[0].get_ylim()[1] * 0.9, "point\ncalib.", fontsize=8)
axes[0].set_xlabel("NashConv per world"); axes[0].set_ylabel("worlds")
modes = ["entry", "exit-shrink", "extortion", "nudge"]
share = np.zeros((9, len(modes)))
for i in range(9):
    pos = REG[:, i] > 0.01
    for k, m in enumerate(modes):
        share[i, k] = 100 * np.mean([mm == m for mm in MODE[pos, i]]) * pos.mean() \
            if pos.any() else 0.0
left = np.zeros(9)
yy = np.arange(9)[::-1]
for k, m in enumerate(modes):
    axes[1].barh(yy, share[:, k], left=left, color=MODE_COLOR[m], label=m)
    left += share[:, k]
axes[1].set_yticks(yy); axes[1].set_yticklabels(ACTORS, fontsize=8)
axes[1].set_xlabel("% of worlds attacked, by mode")
axes[1].legend(frameon=False, fontsize=7, ncol=2)
fig.tight_layout(); fig.savefig(OUT + "fig_exinterim.pdf"); plt.close(fig)

# ---- Fig 5: guardrail ablation [rq2_exinterim_guardrails.py, coarse grid]
designs = ["baseline", "T$^-\\!\\leq\\!1$", "c$\\geq$0.5", "pool cap", "all three"]
nashv = [20.30, 16.92, 20.18, 13.07, 12.95]
obstr = [68.9, 68.9, 74.1, 68.9, 74.1]
collu = [949.6, 158.4, 158.4, 125.8, 125.8]
fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.6))
for ax, v, t, c in [(axes[0], nashv, "endog. NashConv", "#4a7c9b"),
                    (axes[1], obstr, "max obstruction (%)", "#c0392b"),
                    (axes[2], collu, "collusion surplus", "#8e44ad")]:
    ax.bar(range(5), v, color=c, width=0.6)
    ax.set_xticks(range(5)); ax.set_xticklabels(designs, rotation=45,
                                                ha="right", fontsize=7)
    ax.set_title(t, fontsize=9)
axes[2].set_yscale("log")
axes[1].axhline(68.9, color="#999", lw=0.7, ls=":")
fig.suptitle("no guardrail distorts the truthful operating point; "
             "the floor worsens obstruction; collusion is attenuated, not closed",
             fontsize=8, y=1.02)
fig.tight_layout(); fig.savefig(OUT + "fig_guardrails.pdf",
                                bbox_inches="tight"); plt.close(fig)

# ---- Fig 6: regime map on the (c, p) plane with c*p iso-contours
# regimes verified in rq2_endogenous_coverage / rq2_obstruction_voteselling
fig, ax = plt.subplots(figsize=(5.6, 3.6))
cc = np.linspace(0.05, 1.0, 300); 
for lev in [5, 10, 23.2, 40, 59.8]:
    ax.plot(cc, lev / cc, color="#ddd", lw=0.8, zorder=0)
    if lev / 1.0 < 160:
        ax.text(0.985, lev / 0.985, f"c·p={lev:g}", fontsize=6, color="#aaa",
                ha="right", va="bottom")
pts = [("truthful\n(26.70, c=0.87)", 0.869, 26.70, "#2c3e50", "o"),
       ("Indonesia entry\n(25.67, c=0.90)", 0.904, 25.67, MODE_COLOR["entry"], "^"),
       ("China exit-shrink\n(15.39, c=0.47)", 0.468, 15.39,
        MODE_COLOR["exit-shrink"], "v"),
       ("alpha-Rank sink\n(10.42, c=0.29)", 0.291, 10.42, "#c0392b", "s"),
       ("frontier-China collusion\n(142.2, c=0.42)", 0.418, 142.22,
        MODE_COLOR["extortion"], "D"),
       ("residual deal, all guardrails\n(55.7, c=0.51)", 0.505, 55.73,
        "#8e44ad", "d")]
for lab, c, p, col, mk in pts:
    ax.scatter([c], [p], color=col, marker=mk, s=40, zorder=3)
    ax.annotate(lab, (c, p), textcoords="offset points", xytext=(6, 4),
                fontsize=7, color=col)
ax.set_xlabel("modelled coverage c"); ax.set_ylabel("price p (EUR/tCO$_2$e)")
ax.set_xlim(0.05, 1.0); ax.set_ylim(0, 160)
fig.tight_layout(); fig.savefig(OUT + "fig_regime_map.pdf"); plt.close(fig)

# ---- Fig 7: alpha-Rank stationary mass [rq2_psro_lite.py, alpha=5, m=50]
fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.7), sharey=True)
for ax, title, tops in [
        (axes[0], "baseline",
         [("China+EU+Adv shrink", 0.25), ("+Indonesia", 0.25),
          ("+Russia", 0.25), ("all-truthful", 0.0)]),
        (axes[1], "pool-cap guardrail",
         [("China+EU+Adv+Indo", 0.50), ("+Russia", 0.50),
          ("all-truthful", 0.0), ("", 0.0)])]:
    labs = [t[0] for t in tops]; vals = [t[1] for t in tops]
    ax.bar(range(len(labs)), vals,
           color=["#c0392b", "#c0392b", "#c0392b", "#4a7c9b"])
    ax.set_xticks(range(len(labs)))
    ax.set_xticklabels(labs, rotation=40, ha="right", fontsize=7)
    ax.set_title(title, fontsize=9)
axes[0].set_ylabel("alpha-Rank stationary mass")
fig.suptitle("truthful reporting is not an attractor of the evolutionary dynamics "
             "in the restricted empirical game", fontsize=8, y=1.03)
fig.tight_layout(); fig.savefig(OUT + "fig_alpharank.pdf",
                                bbox_inches="tight"); plt.close(fig)

print("figures written to", OUT)
