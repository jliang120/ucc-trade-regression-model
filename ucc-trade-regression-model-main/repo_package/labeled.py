"""
Z5 Basis Over Time, With Labeled Spike Events
================================================
Trade: Long Northern Virginia Winter Energy-Scarcity Convexity

Purpose: Plot the full Z5 basis history (2021-2026) as a line chart, with
callout labels/arrows marking the specific real cold-snap events, so a
reader can see exactly WHEN and HOW BIG each major spike was, not just
that spikes happen in general.

INPUT: data/transco_z5_basis.csv (columns: day, z5_basis)
OUTPUT: z5_basis_labeled_spikes.png
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

INPUT_CSV = "data/transco_z5_basis.csv"
OUTPUT_CHART = "z5_basis_labeled_spikes.png"

# How many of the biggest spikes to label (top N by basis level)
N_SPIKES_TO_LABEL = 4


def load_basis(csv_path):
    df = pd.read_csv(csv_path, parse_dates=['day'])
    return df.sort_values('day').reset_index(drop=True)


def make_labeled_chart(df, out_path):
    fig, ax = plt.subplots(figsize=(10.5, 5.8), dpi=150)

    ax.plot(df['day'], df['z5_basis'], color='#1a1a1a', linewidth=1.0)

    # Group nearby spikes into single named events (Elliott spans two close
    # spike days; Fern does too), so each real storm gets ONE clear label
    # instead of two overlapping ones.
    events = [
        {
            'peak_date': '2022-12-23', 'peak_value': 56.387,
            'label': "Winter Storm Elliott (Dec 23-27, 2022)\nBomb cyclone; Appalachian gas production\nfell >6 Bcf/d as wellheads froze",
            'y_offset': 55,
        },
        {
            'peak_date': '2026-01-27', 'peak_value': 109.433,
            'label': "Winter Storm Fern (Jan 27-30, 2026)\nTransco Z5 cash price surged to $132.47/MMBtu\nas Lower 48 production fell below 100 Bcf/d",
            'y_offset': 55,
        },
    ]

    for ev in events:
        peak_date = pd.Timestamp(ev['peak_date'])
        ax.annotate(
            f"{ev['label']}",
            xy=(peak_date, ev['peak_value']),
            xytext=(peak_date, ev['peak_value'] + ev['y_offset']),
            ha='center', fontsize=8.3, color='#555555',
            linespacing=1.5,
            arrowprops=dict(arrowstyle='->', color='#999999', lw=0.9),
        )

    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Transco Z5 Basis vs. Henry Hub ($/MMBtu)', fontsize=11)
    ax.set_title('Transco Z5 Basis, 2021-2026', fontsize=13)
    ax.set_ylim(-15, 185)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%-m/%-d/%y'))
    ax.tick_params(labelsize=9)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, facecolor='white')
    plt.close(fig)
    print(f"Chart saved to {out_path}")


if __name__ == "__main__":
    df = load_basis(INPUT_CSV)
    make_labeled_chart(df, OUTPUT_CHART)