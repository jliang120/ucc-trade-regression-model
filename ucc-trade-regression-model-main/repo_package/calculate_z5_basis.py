"""
Calculate Transco Z5 Basis from Two Raw S&P Global / SNL Spot Price Exports
=============================================================================
Basis = Transco Z5 spot price - Henry Hub spot price

Both input files are S&P Global Commodity Charting exports (source: SNL),
which have 15 metadata rows before the actual data starts, and column
headers like "SNL Natural Gas Transco Z 5 Spot Natural Gas Index (Price/Value)".
This script skips that metadata and finds the price column automatically.

INPUT FILES EXPECTED:
  data/transco_z5_spot.csv   - Transco Z5 spot export
  data/henry_hub_spot.csv    - Henry Hub spot export

OUTPUT:
  data/transco_z5_basis.csv  - columns: day, z5_basis
  (this is the exact file z5_convexity_model.py and mispricing_check.py
   already look for - no changes needed in those scripts once this runs)
"""

import pandas as pd

Z5_CSV = "data/transco_z5_spot.csv"
HH_CSV = "data/henry_hub_spot.csv"
OUTPUT_CSV = "data/transco_z5_basis.csv"


def load_spglobal_spot(csv_path, value_col_name):
    """
    Load an S&P Global Commodity Charting export. These files have ~15 rows
    of metadata before the real header row (which starts with 'Date').
    """
    df = pd.read_csv(csv_path, encoding='latin-1', skiprows=15, usecols=[0, 1])
    df.columns = ['day', value_col_name]
    df = df.dropna()
    df['day'] = pd.to_datetime(df['day'], errors='coerce')
    df = df.dropna(subset=['day'])
    df = df.sort_values('day').drop_duplicates(subset='day')
    print(f"Loaded {value_col_name}: {len(df)} rows, {df['day'].min().date()} to {df['day'].max().date()}")
    return df


def calculate_basis():
    z5 = load_spglobal_spot(Z5_CSV, 'z5_spot')
    hh = load_spglobal_spot(HH_CSV, 'hh_spot')

    merged = pd.merge(z5, hh, on='day', how='inner')
    print(f"\nMerged (dates in both series): {len(merged)} rows")
    print(f"Date range: {merged['day'].min().date()} to {merged['day'].max().date()}")

    merged['z5_basis'] = merged['z5_spot'] - merged['hh_spot']

    print(f"\nBasis summary:")
    print(merged['z5_basis'].describe())

    top_spikes = merged.nlargest(5, 'z5_basis')[['day', 'z5_basis']]
    print(f"\nTop 5 basis spikes (biggest cold-snap events in the data):")
    print(top_spikes.to_string(index=False))

    out = merged[['day', 'z5_basis']].copy()
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved to {OUTPUT_CSV}")
    print("z5_convexity_model.py and mispricing_check.py will now use this real data automatically.")

    return out


if __name__ == "__main__":
    calculate_basis()