import os
import glob
import numpy as np
import pandas as pd
from scipy import stats, optimize
import matplotlib.pyplot as plt
RNG = np.random.default_rng(7)

HDD_FILE   = "data/iad_all_hdd.csv"
PRICE_GLOB = "data/load_price_merged*.csv"
PRICE_ALT  = "data/transco_z5_basis.csv"


def winter_label(dt):
    if dt.month >= 11: return f"{dt.year}-{dt.year+1}"
    if dt.month <= 3:  return f"{dt.year-1}-{dt.year}"
    return None


def build_hdd(path):

    df = pd.read_csv(path); df["day"] = pd.to_datetime(df["day"])

    if "hdd" not in df.columns:
        df["hdd"] = (65 - (df["max_temp_f"] + df["min_temp_f"]) / 2).clip(lower=0)

    return df[["day","hdd"]].sort_values("day")


def build_price(path):

    df = pd.read_csv(path); df["day"] = pd.to_datetime(df["day"])

    pcol = next((c for c in ["price","basis","Z5_Basis","z5_basis","PX_LAST",
                             "spark","spark_spread","lmp","LMP","value"] if c in df.columns), None)

    if pcol is None:
        raise ValueError(f"No price column found in {path}. Columns: {list(df.columns)}")

    return df[["day",pcol]].rename(columns={pcol:"price"}).dropna().sort_values("day")


def load_data():

    hits = glob.glob(PRICE_GLOB)
    price_path = hits[0] if hits else PRICE_ALT

    return build_hdd(HDD_FILE), build_price(price_path), price_path


def build(hdd, price, months, recent_start=None, mvp="2024-06-01"):

    w = pd.merge(hdd, price, on="day").assign(season=lambda d: d["day"].map(winter_label))

    w = w[w["day"].dt.month.isin(months)].dropna(subset=["season"]).copy()
    w["hdd_c"] = w["hdd"] - w["hdd"].mean()
    w["hdd2"]  = np.clip(w["hdd_c"], 0, None)**2

    order = sorted(w["season"].unique())
    idx = w["season"].map({v:i for i,v in enumerate(order)})
    cut = (len(order)-3) if recent_start is None else order.index(recent_start)
    w["tight"] = (idx >= cut).astype(float)

    w["hdd_x_t"]  = w["hdd_c"]*w["tight"]
    w["hdd2_x_t"] = w["hdd2"]*w["tight"]
    w["mvp"] = (w["day"] >= pd.Timestamp(mvp)).astype(float)

    return w.sort_values("day").reset_index(drop=True)


def ols_hac(y, Xdf, lags=10):

    X = np.column_stack([np.ones(len(Xdf))] + [Xdf[c].values for c in Xdf.columns])
    y = np.asarray(y, float); n,k = X.shape

    XtXi = np.linalg.inv(X.T@X); beta = XtXi@X.T@y; u = y - X@beta
    Xu = X*u[:,None]; S = Xu.T@Xu

    for l in range(1,lags+1):

        wl = 1-l/(lags+1); G = Xu[l:].T@Xu[:-l]; S += wl*(G+G.T)

    V = XtXi@S@XtXi * n/(n-k); se = np.sqrt(np.diag(V))
    t = beta/se; p = 2*(1-stats.t.cdf(np.abs(t), n-k))
    names = ["const"]+list(Xdf.columns)
    r2 = 1-(u@u)/(((y-y.mean())**2).sum())

    print(f"{'var':<12}{'coef':>12}{'HAC se':>11}{'t':>8}{'p':>8}")

    out = {}
    for nm,b,s,tt,pp in zip(names,beta,se,t,p):

        out[nm]=dict(coef=b,se=s,t=tt,p=pp); print(f"{nm:<12}{b:>12.4f}{s:>11.4f}{tt:>8.2f}{pp:>8.3f}")

    out["_names"]=names; out["_resid"]=u; print(f"N={n}  R2={r2:.3f}"); return out


def qreg(y, Xdf, tau):

    X = np.column_stack([np.ones(len(Xdf))] + [Xdf[c].values for c in Xdf.columns])
    y = np.asarray(y,float); n,k = X.shape

    c = np.r_[np.zeros(2*k), tau*np.ones(n), (1-tau)*np.ones(n)]
    A = np.hstack([X, -X, -np.eye(n), np.eye(n)])
    r = optimize.linprog(c, A_eq=A, b_eq=y, bounds=[(0,None)]*(2*k+2*n), method="highs")
    beta = r.x[:k]-r.x[k:2*k]

    return dict(zip(["const"]+list(Xdf.columns), beta))


def predict(b, hdd_c, tight):

    h2 = max(hdd_c,0)**2
    g = lambda nm: b.get(nm,0.0)

    return (g("const")+g("hdd_c")*hdd_c+g("hdd2")*h2+g("tight")*tight
            +g("hdd_x_t")*hdd_c*tight+g("hdd2_x_t")*h2*tight)


def quick_ols(y, Xdf):
    X = np.column_stack([np.ones(len(Xdf))] + [Xdf[c].values for c in Xdf.columns])
    y = np.asarray(y, float); n,k = X.shape
    beta = np.linalg.lstsq(X, y, rcond=None)[0]; u = y - X@beta
    r2 = 1-(u@u)/(((y-y.mean())**2).sum())
    adj = 1-(1-r2)*(n-1)/(n-k)
    return dict(zip(["const"]+list(Xdf.columns), beta)), r2, adj, u


def validate(hdd, price, w, cols, months):
    
    print("\nVALIDATION\n" )
    lin_cols = [c for c in cols if c not in ("hdd2","hdd2_x_t")]
    
    _,_,adj_lin,_ = quick_ols(w["price"], w[lin_cols])
    _,_,adj_cvx,_ = quick_ols(w["price"], w[cols])
    
    print(f"1) fit: linear adjR2={adj_lin:.3f}  vs  convex adjR2={adj_cvx:.3f}  (convex should win)")
    print("2) breakpoint sweep (hdd2_x_t as 'recent era' start moves):")
    
    order = sorted(w["season"].unique())
    
    for start in order[-5:-1]:
        ws = build(hdd, price, months, recent_start=start)
        b,_,_,_ = quick_ols(ws["price"], ws[cols])
        print(f"     recent from {start}:  hdd2_x_t = {b['hdd2_x_t']:+.5f}")
        
    y = w["price"].values; X = w[cols]; n = len(w); bl = 20
    draws = []
    
    for _ in range(400):
        starts = RNG.integers(0, n-bl, size=n//bl)
        idx = np.concatenate([np.arange(s, s+bl) for s in starts])
        b,_,_,_ = quick_ols(y[idx], X.iloc[idx]); draws.append(b["hdd2_x_t"])
        
    lo,med,hi = np.percentile(draws, [5,50,95])
    verdict = "excludes 0 -> robust" if (lo>0 or hi<0) else "straddles 0 -> NOT robust"
    print(f"3) block-bootstrap hdd2_x_t: median {med:+.5f}, 90% CI [{lo:+.5f}, {hi:+.5f}]  ({verdict})")


def graphs(w, ols, cols, fname="model_diagnostics.png"):
    b = {nm:ols[nm]["coef"] for nm in ols["_names"]}
    lo, hi = w["tight"].min(), w["tight"].max()
    grid = np.linspace(w["hdd"].min(), w["hdd"].max(), 120); hc = grid - w["hdd"].mean()
    q95 = qreg(w["price"], w[cols], 0.95)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    base, rec = w[w.tight==lo], w[w.tight==hi]
    ax[0].scatter(base.hdd, base.price, s=8, alpha=.3, color="#888", label="base era")
    ax[0].scatter(rec.hdd, rec.price, s=9, alpha=.5, color="#d97b00", label="recent era")
    ax[0].plot(grid, [predict(b,x,lo) for x in hc], "--", color="#555", lw=2, label="fit: base")
    ax[0].plot(grid, [predict(b,x,hi) for x in hc], color="#c0392b", lw=2.3, label="fit: recent")
    ax[0].plot(grid, [predict(q95,x,hi) for x in hc], ":", color="#7d1fa8", lw=1.8, label="recent 95th pct")
    ax[0].set(xlabel="HDD", ylabel="price ($)", title="Convexity to cold: recent vs base"); ax[0].legend(fontsize=7)
    ax[1].scatter(w.hdd, ols["_resid"], s=8, alpha=.3, color="#2c6"); ax[1].axhline(0, color="k", lw=.8)
    ax[1].set(xlabel="HDD", ylabel="residual", title="Residuals vs HDD (want no smile)")
    taus = [0.5,0.9,0.95,0.99]
    curv = [qreg(w["price"], w[cols], t)["hdd2"] for t in taus]
    ax[2].plot(taus, curv, "o-", color="#c0392b"); ax[2].axhline(0, color="k", lw=.6)
    ax[2].set(xlabel="quantile", ylabel="curvature (hdd2)", title="Tail curvature by quantile")
    fig.tight_layout(); fig.savefig(fname, dpi=130); print(f"\nsaved {fname}"); plt.show()


if __name__ == "__main__":

    hdd, price, price_path = load_data()
    print(f"price series: {price_path}   ({len(price)} rows)\n")
    Xcols = ["hdd_c","hdd2","tight","hdd_x_t","hdd2_x_t"]

    for lbl, months in [("NOV-MAR",(11,12,1,2,3)), ("DEC-FEB core",(12,1,2))]:
        w = build(hdd, price, months)
        cols = Xcols + (["mvp"] if 0 < w["mvp"].mean() < 1 else [])
        print( f"\n{lbl}  N={len(w)}\n" )
        ols = ols_hac(w["price"], w[cols])

        print("\ncurvature by quantile (want hdd2 to RISE with tau = fat tail):")
        for tau in (0.5,0.9,0.95,0.99):
            print(f"  tau={tau:.2f}  hdd2={qreg(w['price'], w[cols], tau)['hdd2']:+.5f}")

        r = ols.get("hdd2_x_t")
        print(f"\nconvexity-steepening (hdd2 x tight) = {r['coef']:+.5f} "
              f"(HAC t={r['t']:.2f}, p={r['p']:.3f})")

        b = {nm:ols[nm]["coef"] for nm in ols["_names"]}
        lo, hi = w["tight"].min(), w["tight"].max()
        print("predicted price on real in-sample cold days, base -> recent tightness:")
        for q in (0.90,0.95,0.99):
            h = w["hdd"].quantile(q); hc = h-w["hdd"].mean()
            mb, mr = predict(b,hc,lo), predict(b,hc,hi)
            print(f"  {q:.0%} cold (~{h:.0f} HDD, ~{65-h:.0f}F): {mb:.2f} -> {mr:.2f}")
    validate(hdd, price, w, cols, months)
    graphs(w, ols, cols)