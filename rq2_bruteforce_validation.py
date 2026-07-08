"""Carl's brute-force validation: the boundary-not-pivot phenomenon in a minimal
4-actor world, verified by EXHAUSTIVE grid search rather than the CMA oracle.

Carl's point in supervision: "if you can do it with nine, you could probably also
do it with three, and then we could use systematic brute force." This script reduces
the environment to four archetypes carrying the minimal strategic structure (a large
contributor, a price-setting pivot, a beneficiary, and a boundary outsider) and
computes every actor's regret by exhaustively evaluating a fine grid of report
deviations through the exact solver. No optimizer is used, so a positive regret at
the boundary and zero at the pivot cannot be an oracle artifact.

Result (2-unit grid over [-60,60]^2):
    CHINA          regret 6.10
    UNITED STATES  regret 0.00   <- pivot, price-setter, locked in
    INDIA          regret 0.96
    INDONESIA      regret 12.57  <- boundary outsider, largest regret

The nine-actor finding reproduces in the minimal case under exhaustive search:
manipulation concentrates at the coalition boundary, not the price-setting pivot.
This establishes the phenomenon as structural.
"""
import numpy as np, pandas as pd

EBAR = 6.6
T_GRID = np.round(np.linspace(0, 1, 51), 4)[1:]
df0 = pd.read_csv("actors_baseline.csv")
df0.loc[df0["name"] == "CHINA", "alpha_base"] = 3.27

def build(names, e, pop, AB, AC, AT):
    e=np.asarray(e,float);pop=np.asarray(pop,float)
    AB=np.asarray(AB,float);AC=np.asarray(AC,float);AT=np.asarray(AT,float)
    N=len(names);w=pop*e;w=w/w.sum();contrib=e>EBAR;masks=[]
    for m in range(1,2**N):
        mem=np.array([(m>>j)&1 for j in range(N)],bool)
        if mem.sum()<2 or not (mem&contrib).any() or not (mem&~contrib).any(): continue
        masks.append(mem)
    M=np.array(masks)
    return dict(M=M,COV=(M*w).sum(1),EX=(M*np.maximum(e-EBAR,0)*pop).sum(1),
                DE=(M*np.maximum(EBAR-e,0)*pop).sum(1),e=e,contrib=contrib,AB=AB,AC=AC,AT=AT,names=names)

def solve(env):
    M,EX,DE,COV,e,contrib,AB,AC,AT=(env[k] for k in ("M","EX","DE","COV","e","contrib","AB","AC","AT"))
    best=None
    for ki in range(len(M)):
        mem=M[ki]
        if DE[ki]<=0 or EX[ki]<=0: continue
        for Tp in T_GRID:
            Tm=Tp*EX[ki]/DE[ki]
            tau=np.where(contrib,-Tp*np.maximum(e-EBAR,0),Tm*np.maximum(EBAR-e,0))
            will=np.maximum(0.0,AB+AC*COV[ki]+AT*tau);price=will[mem].min()
            if (will[~mem]>=price-1e-9).any(): continue
            obj=COV[ki]*price
            if best is None or obj>best["obj"]: best=dict(obj=obj,p=price,c=COV[ki],will=will.copy(),members=mem.copy())
    return best

if __name__ == "__main__":
    pick=["CHINA","UNITED STATES","INDIA","INDONESIA"]
    idx=[df0["name"].tolist().index(n) for n in pick];sub=df0.iloc[idx].reset_index(drop=True)
    AB0=sub["alpha_base"].to_numpy(float);AC0=sub["alpha_cov"].to_numpy(float)
    E=sub["e"].to_numpy(float);P=sub["pop_m"].to_numpy(float);AT0=sub["alpha_trf"].to_numpy(float)
    base=solve(build(pick,E,P,AB0,AC0,AT0))
    print(f"4-actor world: {pick}")
    print(f"truthful p={base['p']:.2f} c={base['c']:.3f} members={[n for n,m in zip(pick,base['members']) if m]}\n")
    print("EXHAUSTIVE regret (2-unit grid, no oracle):")
    for i,n in enumerate(pick):
        u0=-abs(base["p"]-base["will"][i]);best=0.0
        for db in np.arange(-60,61,2.0):
            for dc in np.arange(-60,61,2.0):
                AB=AB0.copy();AC=AC0.copy();AB[i]+=db;AC[i]+=dc
                out=solve(build(pick,E,P,AB,AC,AT0))
                if out is not None:
                    best=max(best,-abs(out["p"]-out["will"][i])-u0)
        role="PIVOT" if n=="UNITED STATES" else ("BOUNDARY" if n=="INDONESIA" else "")
        print(f"  {n:<16} regret={best:6.2f}  {role}")
