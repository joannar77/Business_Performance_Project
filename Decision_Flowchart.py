#!/usr/bin/env python
# coding: utf-8

# DECISION FLOW FOR REVENUE VS. PROFIT MARGIN ANALYSIS

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# Add a decision (process) box
def add_box(ax, center_xy, text, w=0.36, h=0.10, fontsize=12, fc=None):
    cx, cy = center_xy
    x, y = cx - w/2, cy - h/2
    
    # Draw box
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        transform=ax.transAxes,
        facecolor=fc, linewidth=1.8
    )
    ax.add_patch(box)
    
    # Add text
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=fontsize, transform=ax.transAxes)
    
    # Anchors for arrows
    return dict(center=(cx, cy), top=(cx, cy + h/2), bottom=(cx, cy - h/2),
                left=(cx - w/2, cy), right=(cx + w/2, cy))

# Draw an arrow between two points
def arrow(ax, start, end, rad=0.0):
    ax.annotate("",
        xy=end, xytext=start,
        xycoords=ax.transAxes, textcoords=ax.transAxes,
        arrowprops=dict(arrowstyle="->", lw=1.6,
                        connectionstyle=f"arc3,rad={rad}"),
        clip_on=False
    )

# Set up figure
fig, ax = plt.subplots(figsize=(12.5, 10), dpi=160)
ax.axis("off")

# Main Column 
start = add_box(ax, (0.55, 0.92),
                r"$\bf{START}$",
                w=0.26, h=0.08, fontsize=13, fc="#BFD0E1")

axesbx = add_box(ax, (0.55, 0.75),
                 r"$\bf{Define\ Axes}$" + "\nX = Profit Margin [-3, 0.9]\nY = Revenue [419k, 8.84B] (log scale)",
                 w=0.33, h=0.10, fc="#F0F8FF")

thr = add_box(ax, (0.55, 0.55),
              r"$\bf{Set\ Thresholds\ (Medians)}$" + "\nPM = 0.27\nRevenue = $89,685,500",
              w=0.33, h=0.10, fc="#F0F8FF")

cls = add_box(ax, (0.55, 0.35),
              r"$\bf{Classify\ by\ Medians}$" + "\n"
              "• STARS: Rev ≥ 89.7M & PM ≥ 27%\n"
              "• VOLUME PLAYERS: Rev ≥ 89.7M but PM < 27% \n"
              "• NICHE WINNERS: Rev < 89.7M but PM ≥ 27%\n"
              "• STRUGGLERS: Rev < 89.7M & PM < 27%",
              w=0.33, h=0.10, fontsize=11, fc="#F0F8FF")

viz = add_box(ax, (0.55, 0.15),
              r"$\bf{Visualize}$" + "\n"
              "Revenue (logarithmic scale) vs Profit Margin\n"
              "Add median lines & quadrant labels",
              w=0.33, h=0.10, fc="#F0F8FF")

# Side Notes ("Special Case" and "Why log scale on Revenue")
outlier = add_box(ax, (0.15, 0.62),
                  r"$\bf{Special\ Case}$" + "\n"
                  "If Profit Margin < -3 (outliers):\n"
                  "• Annotate separately\n"
                  "• Retain for insight",
                  w=0.24, h=0.10, fc="#C9DAEB")

whylog = add_box(ax, (0.15, 0.44),
                 r"$\bf{Why\ log\ scale\ on\ Revenue?}$" + "\n"
                 "• Values span ~10^4\n"
                 "• Preserves relative distances\n"
                 "• Prevents small firms from\n"
                 "  vanishing next to $B+ firms",
                 w=0.24, h=0.10, fc="#C9DAEB")

# Arrows
arrow(ax, start["bottom"], axesbx["top"])
arrow(ax, axesbx["bottom"], thr["top"], rad=0.0)
arrow(ax, thr["bottom"], cls["top"], rad=0.0)
arrow(ax, cls["bottom"], viz["top"], rad=0.0)

# Side branches
arrow(ax, (thr["left"][0]-0.01, thr["left"][1]),
      (outlier["right"][0]+0.01, outlier["right"][1]), rad=0.00)

arrow(ax, (thr["left"][0]-0.01, thr["left"][1]-0.02),
      (whylog["right"][0]+0.01,  whylog["right"][1]+0.02), rad=0.00)

# Title, layout, and margins
ax.set_title(r"$\bf{Decision\ Flow: Revenue\ vs. Profit\ Margin\ Matrix}$", pad=16, fontsize=15)

# roomy margins so nothing clips
plt.subplots_adjust(left=0.04, right=0.98, top=0.93, bottom=0.05)

plt.show()





