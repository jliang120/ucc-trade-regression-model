"""
Reserve-Margin / Cold-Weather Regression Model
================================================
Trade: Long Northern Virginia Winter Energy-Scarcity Convexity
Purpose: Test whether Dominion zone peak load has become MORE sensitive to cold
weather in recent winters (2023/24-2025/26) vs. prior winters (2015/16-2022/23).
This is the empirical backbone of the thesis: "the same size cold shock produces
a bigger stress response now than it did a few years ago, because data-center
load has eaten the grid's reserve margin cushion."

Data sources:
  - Weather: Iowa Environmental Mesonet (IEM), station IAD (Washington Dulles),
    daily max/min temperature, Nov 2015 - Aug 2026.
    https://mesonet.agron.iastate.edu/request/daily.phtml
  - Load: PJM Data Miner 2, "Hourly Load: Metered" feed, filtered to the DOM
    (Dominion) zone, Jan 2016 - Aug 2026.
    https://dataminer2.pjm.com/feed/hrl_load_metered

Model:
    Peak Load (MW) = b0 + b1*HDD + b2*RecentEra + b3*(HDD x RecentEra) + error

  - HDD (Heating Degree Days) = max(0, 65F - average daily temp). Standard
    measure of how cold a day is relative to a 65F comfort baseline.
  - RecentEra = 1 for winters 2023/24, 2024/25, 2025/26 (post data-center
    buildout acceleration per Dominion's own Q1/Q2 2026 disclosures); 0 for
    winters 2015/16 through 2022/23.
  - The coefficient of interest is b3 (the interaction term): a positive,
    significant b3 means cold sensitivity has INCREASED in recent winters,
    i.e., the same cold shock now moves load more than it used to.

NOTE ON LIMITATIONS:
  - This regresses HDD against LOAD (MW), not PRICE (Transco Z5 basis or PJM
    LMP). It demonstrates the underlying MECHANISM (demand response to cold has
    steepened) but does not, on its own, prove the PRICE response has steepened
    by the same or a larger margin. For a complete proof of the price-spike
    thesis, this same regression should be re-run with PJM day-ahead LMP price
    (Dominion zone) or Transco Zone 5 gas basis as the dependent variable in
    place of peak_load_mw.
  - "RecentEra" is a simple pre/post dummy, not a continuous reserve-margin
    series (PJM's actual daily reserve margin data was not available for this
    build). It is a reasonable proxy given Dominion's own disclosed timeline
    of data-center load acceleration, but a true reserve-margin series (PJM
    capacity minus load, by day) would be a stronger specification if
    obtainable.
"""

import pandas as pd
import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# 1. LOAD AND PREP WEATHER DATA (HDD)
# ---------------------------------------------------------------------------

def build_hdd(weather_csv_path):
    df = pd.read_csv(weather_csv_path)
    df['day'] = pd.to_datetime(df['day'])
    df['avg_temp_f'] = (df['max_temp_f'] + df['min_temp_f']) / 2
    df['hdd'] = (65 - df['avg_temp_f']).clip(lower=0)
    return df[['day', 'hdd', 'max_temp_f', 'min_temp_f']]


# ---------------------------------------------------------------------------
# 2. LOAD AND PREP PJM DOMINION LOAD DATA
# ---------------------------------------------------------------------------

def build_daily_load(load_csv_paths):
    dfs = [pd.read_csv(p) for p in load_csv_paths]
    load = pd.concat(dfs, ignore_index=True)
    load['datetime_beginning_ept'] = pd.to_datetime(
        load['datetime_beginning_ept'], format='mixed'
    )
    load['day'] = pd.to_datetime(load['datetime_beginning_ept'].dt.date)
    daily = load.groupby('day')['mw'].agg(['max', 'mean', 'min']).reset_index()
    daily.columns = ['day', 'peak_load_mw', 'avg_load_mw', 'min_load_mw']
    return daily.sort_values('day')


# ---------------------------------------------------------------------------
# 3. MERGE, LABEL WINTER SEASONS, BUILD RECENT-ERA DUMMY
# ---------------------------------------------------------------------------

def winter_season_label(dt):
    if dt.month >= 11:
        return f"{dt.year}-{dt.year + 1}"
    elif dt.month <= 3:
        return f"{dt.year - 1}-{dt.year}"
    return None


def build_regression_dataset(hdd_df, load_df, recent_era_start="2023-2024"):
    merged = pd.merge(hdd_df, load_df, on='day', how='inner')
    merged['winter_season'] = merged['day'].apply(winter_season_label)
    winter = merged[merged['day'].dt.month.isin([11, 12, 1, 2, 3])].copy()
    winter = winter.dropna(subset=['winter_season'])

    season_order = sorted(winter['winter_season'].unique())
    season_map = {s: i for i, s in enumerate(season_order)}
    winter['season_ord'] = winter['winter_season'].map(season_map)
    winter['recent_era'] = (
        winter['season_ord'] >= season_map[recent_era_start]
    ).astype(int)
    winter['hdd_x_recent'] = winter['hdd'] * winter['recent_era']
    return winter


# ---------------------------------------------------------------------------
# 4. OLS REGRESSION (manual implementation - no statsmodels dependency)
# ---------------------------------------------------------------------------

def ols_regression(y, X_df, verbose=True):
    """
    Simple multivariate OLS with classical (homoskedastic) standard errors.
    Returns a dict of coefficients plus R^2 / adjusted R^2.
    """
    X = np.column_stack([np.ones(len(X_df))] + [X_df[c].values for c in X_df.columns])
    y = y.values
    n, k = X.shape

    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    dof = n - k
    sigma2 = (resid @ resid) / dof
    XtX_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(sigma2 * XtX_inv))
    tvals = beta / se
    pvals = 2 * (1 - stats.t.cdf(np.abs(tvals), dof))

    ss_res = resid @ resid
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k)

    names = ['const'] + list(X_df.columns)
    result = {
        name: {'coef': b, 'se': s, 't': t, 'p': p}
        for name, b, s, t, p in zip(names, beta, se, tvals, pvals)
    }
    result['_r2'] = r2
    result['_adj_r2'] = adj_r2
    result['_n'] = n

    if verbose:
        print(f"{'variable':<18}{'coef':>14}{'std err':>12}{'t':>10}{'p>|t|':>10}")
        for name in names:
            r = result[name]
            print(f"{name:<18}{r['coef']:>14.3f}{r['se']:>12.3f}{r['t']:>10.2f}{r['p']:>10.4f}")
        print(f"\nN = {result['_n']}, R2 = {r2:.4f}, Adj R2 = {adj_r2:.4f}")

    return result


# ---------------------------------------------------------------------------
# 5. MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import glob

    # NOTE: data/iad_all_hdd.csv and data/dominion_daily_load.csv included in
    # this repo are ALREADY PROCESSED (HDD already calculated; load already
    # aggregated to daily peak/avg/min) so we load them directly rather than
    # re-running build_hdd()/build_daily_load() on raw source files. If you
    # are starting from raw IEM weather CSVs or raw PJM hourly CSVs instead,
    # use build_hdd() / build_daily_load() on those raw files first.
    hdd = pd.read_csv("data/iad_all_hdd.csv", parse_dates=["day"])[["day", "hdd"]]
    load = pd.read_csv("data/dominion_daily_load.csv", parse_dates=["day"])
    winter = build_regression_dataset(hdd, load, recent_era_start="2023-2024")

    print("=" * 70)
    print("MODEL: Peak Load (MW) ~ HDD + Recent Era + (HDD x Recent Era)")
    print(f"Recent Era = winters 2023/24 - 2025/26 | Base = winters 2015/16 - 2022/23")
    print("=" * 70)
    result = ols_regression(
        winter['peak_load_mw'],
        winter[['hdd', 'recent_era', 'hdd_x_recent']]
    )

    base_slope = result['hdd']['coef']
    recent_slope = base_slope + result['hdd_x_recent']['coef']
    pct_change = (recent_slope - base_slope) / abs(base_slope) * 100

    print("\n" + "-" * 70)
    print("INTERPRETATION")
    print("-" * 70)
    print(f"Cold-weather sensitivity (MW per 1 HDD), base-era winters:   {base_slope:,.1f}")
    print(f"Cold-weather sensitivity (MW per 1 HDD), recent-era winters: {recent_slope:,.1f}")
    print(f"Change in sensitivity: {pct_change:+.1f}%")
    print(f"Interaction term p-value: {result['hdd_x_recent']['p']:.4f} "
          f"({'significant' if result['hdd_x_recent']['p'] < 0.05 else 'not significant'} at 5% level)")

    winter.to_csv("data/merged_winter_data.csv", index=False)
    print("\nSaved merged dataset to merged_winter_data.csv")
