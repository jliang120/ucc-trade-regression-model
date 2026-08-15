"""
Z5 Basis Fat-Tails Histogram
=============================
Trade: Long Northern Virginia Winter Energy-Scarcity Convexity

Purpose: Show that daily Z5 basis changes are NOT normally distributed -
most days are quiet (basis barely moves), but rare days are extreme
(e.g., the January 27, 2026 spike). This is the "fat tails" argument for
why options (not an outright/linear position) are the right structure:
options let you pay a small premium for exposure to the rare extreme days,
without being exposed to all the boring noise on quiet days.

Uses the excess kurtosis statistic to quantify this: a normal distribution
has kurtosis = 0. Anything much higher than 0 means "fatter tails than
normal" - i.e., extreme moves happen more often than a standard bell curve
would predict.

INPUT: data/transco_z5_basis.csv (columns: day, z5_basis) - the real,
calculated Z5 basis file produced by calculate_z5_basis.py.

OUTPUT: z5_fat_tails_histogram.png
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

INPUT_CSV = "data/transco_z5_basis.csv"
OUTPUT_CHART = "z5_fat_tails_histogram.png"


def load_basis_changes(csv_path):
    basis = pd.read_csv(csv_path, parse_dates=['day'])
    basis = basis.sort_values('day').reset_index(drop=True)
    basis['basis_chg'] = basis['z5_basis'].diff()
    return basis.dropna(subset=['basis_chg'])


def print_stats(basis):
    changes = basis['basis_chg']
    print("=" * 60)
    print("Z5 BASIS DAILY CHANGE STATISTICS")
    print("=" * 60)
    print(changes.describe())
    print()
    print(f"Skewness: {changes.skew():.2f}")
    print(f"Excess Kurtosis: {changes.kurtosis():.2f}  "
          f"(normal distribution = 0; higher = fatter tails)")
    print()

    biggest = basis.nlargest(5, 'basis_chg')[['day', 'basis_chg']]
    print("Top 5 single-day basis increases (real cold-snap events):")
    print(biggest.to_string(index=False))
    print()


def make_histogram(basis, out_path, x_range=(-15, 15)):
    changes = basis['basis_chg']
    mean, std = changes.mean(), changes.std()

    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=150)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    ax.hist(changes, bins=200, range=x_range, density=True, alpha=0.9,
            color='#1a1a1a', label='Actual Z5 Basis Daily Changes')

    x = np.linspace(x_range[0], x_range[1], 500)
    normal_curve = stats.norm.pdf(x, mean, std)
    ax.plot(x, normal_curve, color='#c0392b', linewidth=2.2, linestyle='--',
            label='Normal Distribution (same mean/std)')

    ax.set_xlabel('Daily Z5 Basis Change ($/MMBtu)', fontsize=11, color='#1a1a1a')
    ax.set_ylabel('Frequency (density)', fontsize=11, color='#1a1a1a')
    ax.set_title('Daily Z5 Basis Changes vs. a Normal Distribution',
                 fontsize=13, color='#1a1a1a')
    ax.legend(fontsize=9, frameon=True, facecolor='white', edgecolor='#cccccc')
    ax.tick_params(labelsize=9, colors='#1a1a1a')

    for spine in ax.spines.values():
        spine.set_color('#cccccc')

    max_change = changes.max()
    max_date = basis.loc[changes.idxmax(), 'day'].date()
    if max_change > x_range[1]:
        ax.text(0.98, 0.85,
                f'The largest single-day spike\n'
                f'({max_date}, +${max_change:.1f}/MMBtu)\n'
                f'is off this chart\'s scale.',
                transform=ax.transAxes, fontsize=8.5, ha='right', va='top',
                color='#555555',
                bbox=dict(boxstyle='round', facecolor='#f5f5f5', edgecolor='#cccccc', alpha=0.95, linewidth=1))

    fig.tight_layout()
    fig.savefig(out_path, facecolor='white')
    plt.close(fig)
    print(f"Chart saved to {out_path}")


if __name__ == "__main__":
    basis = load_basis_changes(INPUT_CSV)
    print_stats(basis)
    make_histogram(basis, OUTPUT_CHART)