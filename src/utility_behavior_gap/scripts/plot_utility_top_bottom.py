#!/usr/bin/env python3
"""Plot top and bottom fitted utility examples for each actor-domain cell."""

from __future__ import annotations

import os
import textwrap

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import pandas as pd

from utility_behavior_gap.constants import ACTOR_LABEL_ORDER, DOMAIN_LABEL, DOMAINS
from utility_behavior_gap.paths import ANALYSIS, FIGURES


CSV_PATH = ANALYSIS / "utility_top_bottom_10_by_actor_domain.csv"
PNG_PATH = FIGURES / "utility_top_bottom_examples.png"
PDF_PATH = FIGURES / "utility_top_bottom_examples.pdf"


def compact(text: str, width: int = 62) -> str:
    return textwrap.shorten(" ".join(text.split()), width=width, placeholder="...")


def cell_text(cell: pd.DataFrame) -> str:
    lines = []
    for side, title in [("top", "Top"), ("bottom", "Bottom")]:
        sub = cell[cell["side"] == side].sort_values("rank_within_side").head(3)
        lines.append(title)
        for _, row in sub.iterrows():
            lines.append(f"{float(row['utility']):+.2f}  {compact(str(row['description']))}")
    return "\n".join(lines)


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    df = df[df["rank_within_side"] <= 3].copy()

    fig, axes = plt.subplots(
        nrows=len(ACTOR_LABEL_ORDER),
        ncols=len(DOMAINS),
        figsize=(20, 27),
        constrained_layout=True,
    )
    fig.suptitle("Examples of Actor-Specific Utility Rankings", fontsize=18, fontweight="bold")

    for row_idx, actor_label in enumerate(ACTOR_LABEL_ORDER):
        for col_idx, domain in enumerate(DOMAINS):
            ax = axes[row_idx][col_idx]
            ax.set_axis_off()
            cell = df[(df["actor_label"] == actor_label) & (df["domain"] == domain)]
            title = DOMAIN_LABEL[domain] if row_idx == 0 else ""
            if title:
                ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
            if col_idx == 0:
                ax.text(
                    -0.04,
                    0.5,
                    actor_label,
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    fontsize=11,
                    fontweight="bold",
                    rotation=90,
                )
            ax.text(
                0.02,
                0.98,
                cell_text(cell),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=6.6,
                linespacing=1.25,
                family="DejaVu Sans",
            )
            ax.add_patch(
                plt.Rectangle(
                    (0, 0),
                    1,
                    1,
                    transform=ax.transAxes,
                    fill=False,
                    edgecolor="#D8DEE9",
                    linewidth=0.8,
                )
            )

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG_PATH, dpi=220, bbox_inches="tight")
    fig.savefig(PDF_PATH, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {PNG_PATH}")
    print(f"Wrote {PDF_PATH}")


if __name__ == "__main__":
    main()
