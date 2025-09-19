#!/usr/bin/env python
# coding: utf-8
"""
Business Health Analysis
- Reads business_financials_data.csv
- Validates columns
- Cleans and dedupes
- Computes Debt-to-Income (DTI)
- Flags negative Debt-to-Equity companies
- Produces descriptive stats and charts
- Exports CSVs
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

CSV_PATH = "business_financials_data.csv"

REQUIRED_COLS = [
    "Business ID", "Business State", "Total Long-term Debt", "Total Equity",
    "Debt to Equity", "Total Liabilities", "Total Revenue", "Profit Margin"
]

def main():
    # Step 1 — Load & preview
    df = pd.read_csv(CSV_PATH)
    print(df.head())
    print("Columns:", df.columns.tolist())

    # Step 2 — Check required columns
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    for c in missing:
        print(c, "is missing!")
    if missing:
        raise ValueError(f"Execution stopped. Missing columns: {missing}")

    # Step 3 — Remove rows with missing values (across all columns)
    df_clean = df.dropna().copy()
    print(f"Rows after cleaning: {len(df_clean)}")

    # Step 4 — Duplicate detection
    dups = df_clean[df_clean.duplicated()]
    if not dups.empty:
        print("The following rows in the dataset could be duplicates. Please check the rows below:")
        print(dups)
        print("Remove duplicates manually if appropriate.")
        raise SystemExit("Execution stopped for review of duplicate rows.")
    else:
        print("No duplicate rows found.")

    # Step 5a — Compute DTI with divide-by-zero safeguard
    dti_val = []
    for _, row in df_clean.iterrows():
        if row["Total Revenue"] == 0:
            dti_val.append(0)
        else:
            dti_val.append(row["Total Long-term Debt"] / row["Total Revenue"])

    # Step 5b — Create DTI dataframe and preview
    dti_df = pd.DataFrame({
        "Business ID": df_clean["Business ID"].values,
        "DebtToIncome": dti_val
    })
    print(dti_df.head())

    # Step 5c — Concatenate DTI back to cleaned data
    df_combined = pd.concat([df_clean, dti_df["DebtToIncome"]], axis=1)
    print("Business-level dataframe with Debt-to-Income added:")
    print(df_combined[["Business ID", "Business State", "DebtToIncome"]].head())
    print("Debt-to-Income column added (via separate dataframe + concat):")
    print(df_combined.head())

    # Step 6 — Negative Debt-to-Equity
    neg_dte = df_combined[df_combined["Debt to Equity"] < 0]
    print("Businesses with NEGATIVE Debt-to-Equity ratios:")
    print(neg_dte)

    # Step 7 — State-level descriptive statistics
    metrics_cols = [
        "Total Long-term Debt", "Total Equity", "Total Liabilities",
        "Total Revenue", "Profit Margin", "DebtToIncome"
    ]
    agg_results = df_combined.groupby("Business State")[metrics_cols].agg(
        ["mean", "median", "min", "max"]
    )
    print("State-level descriptive statistics:")
    print(agg_results.head())

    # Step 8 — Export CSVs
    neg_dte.to_csv("negative_debt_to_equity_businesses.csv", index=False)
    df_combined.to_csv("business_level_with_DTI.csv", index=False)
    agg_results.to_csv("state_descriptive_stats.csv")
    print("Analysis CSV exports complete.")

    # ----- Step 9 — Visualizations -----
    # 9a — Prep plotting frame
    dfp = df_combined.rename(columns={
        'Business ID': 'Business_ID',
        'Business State': 'Business_State',
        'Profit Margin': 'Profit_Margin',
        'Debt to Equity': 'Debt_to_Equity',
        'DebtToIncome': 'Debt_to_Income'
    })
    plot_df = dfp[['Business_ID', 'Business_State', 'Debt_to_Income', 'Profit_Margin', 'Debt_to_Equity']].dropna()

    # 9b — Profit Margin vs DTI (with neg. D/E highlighted)
    fig, ax = plt.subplots(figsize=(12, 7), dpi=150, constrained_layout=True)
    ax.scatter(plot_df['Debt_to_Income'], plot_df['Profit_Margin'], s=5, alpha=0.80, label='All businesses')

    neg_mask = plot_df['Debt_to_Equity'] < 0
    if neg_mask.any():
        ax.scatter(plot_df.loc[neg_mask, 'Debt_to_Income'],
                   plot_df.loc[neg_mask, 'Profit_Margin'],
                   marker='x', s=80, linewidths=1.3, color='darkorange',
                   label='Negative Debt-to-Equity')

    ax.set_xlim(0, 1)
    ax.set_ylim(-1, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.tick_params(axis='both', labelsize=8)

    corr = plot_df['Debt_to_Income'].corr(plot_df['Profit_Margin'])
    ax.text(0.01, 0.02, f"Correlation (r) = {corr:.2f}", transform=ax.transAxes, fontsize=11)

    xy = plot_df[['Debt_to_Income','Profit_Margin']].to_numpy()
    if len(xy) > 1:
        x, y = xy[:, 0], xy[:, 1]
        m, b = np.polyfit(x, y, 1)
        xg = np.linspace(*ax.get_xlim(), 100)
        ax.plot(xg, m*xg + b, linewidth=1, label='Trend line')

    ax.set_xlabel('Debt-to-Income (LT Debt / Revenue)', fontsize=11)
    ax.set_ylabel('Profit Margin', fontsize=11)
    ax.set_title('Profit Margin vs. Debt-to-Income (zoomed to typical ranges)', fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
    plt.show()

    # 9c — Avg DTI by State (bar)
    avg_dti_by_state = (dfp.groupby('Business_State', dropna=False)['Debt_to_Income']
                           .mean().sort_values(ascending=False).reset_index()
                           .rename(columns={'Debt_to_Income': 'Avg_DTI'}))
    plt.figure(figsize=(12, 6))
    plt.bar(avg_dti_by_state['Business_State'], avg_dti_by_state['Avg_DTI'])
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Average Debt-to-Income')
    plt.xlabel('State')
    plt.title('Average Debt-to-Income by State', fontsize=16)
    ax2 = plt.gca()
    ax2.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    plt.tight_layout()
    plt.show()
    print("Top 5 states by Avg DTI:")
    print(avg_dti_by_state.head(5))
    print("\nBottom 5 states by Avg DTI:")
    print(avg_dti_by_state.tail(5))

    print("Analysis complete.")

if __name__ == "__main__":
    main()
