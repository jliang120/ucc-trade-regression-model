"""
Transco Zone 5 Basis vs. Cold Weather: Convexity Regression + Chart
=====================================================================
Trade: Long Northern Virginia Winter Energy-Scarcity Convexity

Purpose: Test and VISUALIZE whether Transco Z5 gas basis (vs. Henry Hub) has
become MORE sensitive to cold weather in recent winters (2023/24-2025/26) vs.
prior winters (2015/16-2022/23). This is the direct price-side proof of the
thesis: "a 1-standard-deviation cold day today produces a larger basis spike
than the same size cold day did a few years ago."

This mirrors reserve_margin_regression.py exactly, but swaps the dependent
variable from Dominion zone peak load (MW) to Transco Z5 basis ($/MMBtu vs
Henry Hub).

REQUIRED INPUT FILE (not yet available - awaiting Bloomberg export):
  /mnt/user-data/uploads/transco_z5_basis.csv
  Expected columns: a date column and a price/basis column. This script's
  loader function accepts common variations (e.g., "Date"/"date", and a
  price column named "PX_LAST", "basis", "value", or similar) - adjust
  the COLUMN NAME CONSTANTS below once the real file is in hand and its
  exact headers are known.

Until that file exists, this script runs on SYNTHETIC PLACEHOLDER DATA
(clearly labeled) so the full pipeline - regression, 1-sigma calculation,
and chart generation - can be validated end to end before real data arrives.
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# COLUMN NAME CONSTANTS - update once real Bloomberg CSV headers are known
# ---------------------------------------------------------------------------
DATE_COL_CANDIDATES = ['Date', 'date', 'DATE']
PRICE_COL_CANDIDATES = ['PX_LAST', 'basis', 'value', 'Value', 'Z5_Basis', 'price']


def load_z5_basis(csv_path):
    """Load Transco Z5 basis CSV, auto-detecting date/price columns."""
    df = pd.read_csv(csv_path)
    date_col = next((c for c in DATE_COL_CANDIDATES if c in df.columns), None)
    price_col = next((c for c in PRICE_COL_CANDIDATES if c in df.columns), None)
    if date_col is None or price_col is None:
        raise ValueError(
            f"Could not auto-detect columns. Found columns: {list(df.columns)}. "
            f"Update DATE_COL_CANDIDATES / PRICE_COL_CANDIDATES at the top of this "
            f"script to match your actual Bloomberg export headers."
        )
    out = df[[date_col, price_col]].copy()
    out.columns = ['day', 'z5_basis']
    out['day'] = pd.to_datetime(out['day'])
    return out.sort_values('day')


def make_synthetic_z5_basis(hdd_df):
    """
    PLACEHOLDER ONLY. Generates fake Z5 basis data that mimics a plausible
    relationship to HDD (with a steeper recent-era slope baked in), purely so
    the regression + chart code can be tested before real data is available.
    NOT FOR USE IN THE FINAL ONE-PAGER.
    """
    np.random.seed(7)
    df = hdd_df.copy()
    df['winter_season'] = df['day'].apply(winter_season_label)
    df = df.dropna(subset=['winter_season'])
    season_order = sorted(df['winter_season'].unique())
    season_map = {s: i for i, s in enumerate(season_order)}
    df['season_ord'] = df['winter_season'].map(season_map)
    df['recent_era'] = (df['season_ord'] >= season_map[list(season_map.keys())[-3]]).astype(int)

    base_slope = 0.015     # $/MMBtu per HDD, base era (placeholder)
    recent_slope = 0.028   # steeper in recent era (placeholder, ~87% increase)
    noise = np.random.normal(0, 0.15, len(df))
    df['z5_basis'] = (
        0.35
        + np.where(df['recent_era'] == 1, recent_slope, base_slope) * df['hdd']
        + noise
    )
    return df[['day', 'z5_basis']]


def winter_season_label(dt):
    if dt.month >= 11:
        return f"{dt.year}-{dt.year + 1}"
    elif dt.month <= 3:
        return f"{dt.year - 1}-{dt.year}"
    return None


def build_dataset(hdd_df, basis_df, recent_era_start=None):
    merged = pd.merge(hdd_df, basis_df, on='day', how='inner')
    merged['winter_season'] = merged['day'].apply(winter_season_label)
    winter = merged[merged['day'].dt.month.isin([11, 12, 1, 2, 3])].copy()
    winter = winter.dropna(subset=['winter_season'])

    season_order = sorted(winter['winter_season'].unique())
    season_map = {s: i for i, s in enumerate(season_order)}
    if recent_era_start is None:
        # default: most recent 3 winters vs. everything before
        recent_era_start = season_order[-3]
    winter['season_ord'] = winter['winter_season'].map(season_map)
    winter['recent_era'] = (winter['season_ord'] >= season_map[recent_era_start]).astype(int)
    winter['hdd_x_recent'] = winter['hdd'] * winter['recent_era']
    return winter, recent_era_start


def ols_regression(y, X_df, verbose=True):
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
    result = {name: {'coef': b, 'se': s, 't': t, 'p': p}
              for name, b, s, t, p in zip(names, beta, se, tvals, pvals)}
    result['_r2'] = r2
    result['_n'] = n
    if verbose:
        print(f"{'variable':<18}{'coef':>14}{'std err':>12}{'t':>10}{'p>|t|':>10}")
        for name in names:
            r = result[name]
            print(f"{name:<18}{r['coef']:>14.4f}{r['se']:>12.4f}{r['t']:>10.2f}{r['p']:>10.4f}")
        print(f"\nN = {result['_n']}, R2 = {r2:.4f}")
    return result


def one_sigma_comparison(winter_df, result):
    """Compute the '1-sigma cold day' headline comparison."""
    hdd_std = winter_df['hdd'].std()
    base_slope = result['hdd']['coef']
    recent_slope = base_slope + result['hdd_x_recent']['coef']
    base_move = base_slope * hdd_std
    recent_move = recent_slope * hdd_std
    pct_increase = (recent_move - base_move) / abs(base_move) * 100 if base_move != 0 else np.nan
    print(f"\n1-sigma HDD = {hdd_std:.2f}")
    print(f"Implied basis move per 1-sigma cold day, base era:   ${base_move:.3f}/MMBtu")
    print(f"Implied basis move per 1-sigma cold day, recent era: ${recent_move:.3f}/MMBtu")
    print(f"Increase: {pct_increase:+.1f}%")
    return hdd_std, base_move, recent_move, pct_increase


def make_chart(winter_df, result, recent_era_start, out_path, placeholder=True):
    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=150)

    base = winter_df[winter_df['recent_era'] == 0]
    recent = winter_df[winter_df['recent_era'] == 1]

    ax.scatter(base['hdd'], base['z5_basis'], s=10, alpha=0.35, color='#7a7a7a',
               label=f'Base era (pre-{recent_era_start})')
    ax.scatter(recent['hdd'], recent['z5_basis'], s=12, alpha=0.55, color='#d97b00',
               label=f'Recent era ({recent_era_start} onward)')

    x_range = np.linspace(winter_df['hdd'].min(), winter_df['hdd'].max(), 100)
    b0 = result['const']['coef']
    b1 = result['hdd']['coef']
    b2 = result['recent_era']['coef']
    b3 = result['hdd_x_recent']['coef']

    ax.plot(x_range, b0 + b1 * x_range, color='#555555', linewidth=2, linestyle='--',
            label='Fitted: base era')
    ax.plot(x_range, (b0 + b2) + (b1 + b3) * x_range, color='#c0392b', linewidth=2,
            label='Fitted: recent era (steeper)')

    ax.set_xlabel('Heating Degree Days (HDD)', fontsize=9)
    ax.set_ylabel('Transco Z5 Basis vs Henry Hub ($/MMBtu)', fontsize=9)
    ax.set_title('Z5 Basis Sensitivity to Cold: Recent vs. Base Era', fontsize=11)
    ax.legend(fontsize=7, loc='upper left')
    ax.tick_params(labelsize=8)

    if placeholder:
        ax.text(0.5, 0.5, "PLACEHOLDER - SYNTHETIC DATA\nNOT FOR SUBMISSION",
                transform=ax.transAxes, fontsize=16, color='red', alpha=0.35,
                ha='center', va='center', rotation=25)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"\nChart saved to {out_path}")


if __name__ == "__main__":
    import os

    hdd = pd.read_csv('data/iad_all_hdd.csv', parse_dates=['day'])

    REAL_DATA_PATH = 'data/transco_z5_basis.csv'
    USE_REAL_DATA = os.path.exists(REAL_DATA_PATH)

    if USE_REAL_DATA:
        print(">>> Real Transco Z5 basis file found - running on ACTUAL data.\n")
        basis = load_z5_basis(REAL_DATA_PATH)
    else:
        print(">>> No real Z5 basis file found yet - running on SYNTHETIC placeholder")
        print(">>> data so the pipeline can be validated. DO NOT use these numbers")
        print(">>> or this chart in the actual one-pager.\n")
        basis = make_synthetic_z5_basis(hdd)

    winter, recent_era_start = build_dataset(hdd, basis)
    print(f"Recent era defined as: {recent_era_start} onward\n")
    print("=" * 70)
    print("MODEL: Transco Z5 Basis ~ HDD + Recent Era + (HDD x Recent Era)")
    print("=" * 70)
    result = ols_regression(winter['z5_basis'], winter[['hdd', 'recent_era', 'hdd_x_recent']])

    one_sigma_comparison(winter, result)

    out_name = '/home/claude/z5_model/z5_basis_convexity_chart.png'
    make_chart(winter, result, recent_era_start, out_name, placeholder=not USE_REAL_DATA)
