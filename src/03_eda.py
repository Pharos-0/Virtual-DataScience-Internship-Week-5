# %% [markdown]
# ## Part 3 - Exploratory data analysis
#
# Works on the cleaned file from Part 2. The questions are simple: how are the
# numeric variables distributed, how common is churn, which customer groups
# churn more often, and how strongly do the variables move together?

# %%
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from matplotlib.ticker import PercentFormatter, StrMethodFormatter
from scipy.stats.contingency import association

from config import RESULTS_DIR, TABLE_DIR
from data_utils import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, load_clean_data
from plot_style import (CHURN_COLOR, MUTED_INK, RETAINED_COLOR, SEQUENTIAL_CMAP,
                        apply_style, save_figure)

pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 200)
apply_style()

df = load_clean_data()
eda_summary = {}
print(df.shape)

# %% [markdown]
# ### 3.1 Target variable: how many customers churned?

# %% tags=["shot-target"]
churn_counts = df["Churn"].value_counts().reindex(["No", "Yes"])
churn_rate = df["churn_flag"].mean()
display(pd.DataFrame({"customers": churn_counts,
                      "percent": (churn_counts / len(df) * 100).round(2)}))
print(f"Overall churn rate: {churn_rate:.2%}")
eda_summary["churn_counts"] = churn_counts.to_dict()
eda_summary["churn_rate"] = round(float(churn_rate), 4)

# %% [markdown]
# ### 3.2 Numeric variables: central tendency and dispersion

# %% tags=["shot-numeric-stats"]
numeric_stats = df[NUMERIC_COLUMNS].agg(["count", "mean", "median", "std", "min", "max", "skew"]).T
numeric_stats["q1"] = df[NUMERIC_COLUMNS].quantile(0.25)
numeric_stats["q3"] = df[NUMERIC_COLUMNS].quantile(0.75)
numeric_stats["iqr"] = numeric_stats["q3"] - numeric_stats["q1"]
numeric_stats = numeric_stats[["count", "mean", "median", "std", "min", "q1", "q3", "iqr", "max", "skew"]]
display(numeric_stats.round(2))
numeric_stats.round(3).to_csv(TABLE_DIR / "numeric_descriptive_stats.csv")
eda_summary["numeric_stats"] = numeric_stats.round(3).to_dict("index")

# %%
# Context for the shapes seen in the histograms below
tenure_1 = int((df["tenure"] == 1).sum())
tenure_72 = int((df["tenure"] == 72).sum())
print("Customers with tenure = 1 month  :", tenure_1)
print("Customers with tenure = 72 months:", tenure_72)
charges_by_internet = df.groupby("InternetService", observed=True)["MonthlyCharges"].describe().round(2)
display(charges_by_internet)
charges_by_internet.to_csv(TABLE_DIR / "monthly_charges_by_internet_service.csv")
eda_summary["tenure_1_count"] = tenure_1
eda_summary["tenure_72_count"] = tenure_72
eda_summary["monthly_charges_by_internet"] = charges_by_internet.to_dict("index")

# %%
# The same statistics split by churn status
by_churn = df.groupby("Churn", observed=True)[NUMERIC_COLUMNS].agg(["mean", "median", "std"]).round(2)
display(by_churn)
by_churn.to_csv(TABLE_DIR / "numeric_stats_by_churn.csv")
eda_summary["numeric_by_churn"] = {
    status: {col: {"mean": float(by_churn.loc[status, (col, "mean")]),
                   "median": float(by_churn.loc[status, (col, "median")])}
             for col in NUMERIC_COLUMNS}
    for status in ["No", "Yes"]
}

# %%
fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
units = {"tenure": "months", "MonthlyCharges": "USD per month", "TotalCharges": "USD"}
bins = {"tenure": np.arange(0, 75, 3), "MonthlyCharges": 30, "TotalCharges": 30}
for ax, column in zip(axes, NUMERIC_COLUMNS):
    ax.hist(df[column], bins=bins[column], color=CHURN_COLOR, alpha=0.85,
            edgecolor="white", linewidth=0.6)
    median = df[column].median()
    ax.axvline(median, color="#222222", linewidth=1.2)
    ax.text(median, ax.get_ylim()[1] * 0.97, f" median {median:,.0f}", va="top",
            fontsize=8.5, color=MUTED_INK)
    ax.set_title(column)
    ax.set_xlabel(units[column])
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
axes[0].set_ylabel("Number of customers")
fig.suptitle("Distribution of the numeric variables (n = 7,043)", x=0.01, ha="left",
             fontsize=12, fontweight="bold")
fig.tight_layout()
save_figure(fig, "fig1_2_numeric_distributions.png")

# %%
fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
for ax, column in zip(axes, NUMERIC_COLUMNS):
    groups = [df.loc[df["Churn"] == s, column] for s in ["No", "Yes"]]
    box = ax.boxplot(groups, tick_labels=["Retained", "Churned"], widths=0.5,
                     patch_artist=True, medianprops=dict(color="#222222", linewidth=1.4),
                     flierprops=dict(marker="o", markersize=2.5, alpha=0.4))
    for patch, colour in zip(box["boxes"], [RETAINED_COLOR, CHURN_COLOR]):
        patch.set_facecolor(colour)
        patch.set_edgecolor("#444444")
    ax.set_title(column)
    ax.set_ylabel(units[column])
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
fig.suptitle("Numeric variables by churn status", x=0.01, ha="left",
             fontsize=12, fontweight="bold")
fig.tight_layout()
save_figure(fig, "fig1_3_numeric_by_churn_boxplots.png")

# %% [markdown]
# ### 3.3 Categorical variables: frequencies and churn rates

# %% tags=["shot-category-rates"]
predictor_categoricals = [c for c in CATEGORICAL_COLUMNS if c != "Churn"]
rows = []
for column in predictor_categoricals:
    grouped = df.groupby(column, observed=True)["churn_flag"].agg(["size", "mean"])
    for category, (n, rate) in grouped.iterrows():
        rows.append({"variable": column, "category": category, "customers": int(n),
                     "share_pct": round(n / len(df) * 100, 2),
                     "churn_rate_pct": round(rate * 100, 2)})
category_table = pd.DataFrame(rows)
category_table.to_csv(TABLE_DIR / "churn_rate_by_category.csv", index=False)
display(category_table[category_table["variable"].isin(
    ["Contract", "InternetService", "PaymentMethod", "TechSupport", "SeniorCitizen"])])

# %%
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), gridspec_kw={"width_ratios": [3, 3, 4]})
for ax, column in zip(axes, ["Contract", "InternetService", "PaymentMethod"]):
    subset = category_table[category_table["variable"] == column].sort_values("churn_rate_pct")
    positions = np.arange(len(subset))
    ax.axvline(churn_rate * 100, color="#222222", linewidth=0.9, zorder=1)
    ax.barh(positions, subset["churn_rate_pct"], height=0.55, color=CHURN_COLOR, zorder=2)
    for y, (rate, n) in enumerate(zip(subset["churn_rate_pct"], subset["customers"])):
        ax.text(rate + 1, y, f"{rate:.1f}%  (n={n:,})", va="center", fontsize=8.5, color=MUTED_INK,
                zorder=3, bbox=dict(facecolor="white", edgecolor="none", pad=1.0))
    ax.set_yticks(positions)
    ax.set_yticklabels(subset["category"])
    ax.set_xlim(0, 75)
    ax.xaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax.set_title(column)
    ax.grid(axis="y", visible=False)
axes[1].set_xlabel("Churn rate (share of customers who left)")
fig.suptitle("Churn rate by contract type, internet service and payment method "
             f"(vertical line = overall rate, {churn_rate:.1%})",
             x=0.01, ha="left", fontsize=12, fontweight="bold")
fig.tight_layout()
save_figure(fig, "fig1_4_churn_rate_by_category.png")

eda_summary["churn_rate_by_category"] = {
    column: category_table[category_table["variable"] == column]
    .set_index("category")[["customers", "churn_rate_pct"]].to_dict("index")
    for column in predictor_categoricals
}

# %% [markdown]
# ### 3.4 Churn across the customer lifecycle (tenure bands)
#
# The data is a single snapshot, so there is no calendar time series. Tenure
# is the closest thing to a time axis: it shows how churn differs between
# newer and longer-standing customers.

# %% tags=["shot-tenure-trend"]
tenure_bins = [-1, 12, 24, 36, 48, 60, 72]
tenure_labels = ["0-12", "13-24", "25-36", "37-48", "49-60", "61-72"]
df["tenure_band"] = pd.cut(df["tenure"], bins=tenure_bins, labels=tenure_labels)
tenure_trend = df.groupby("tenure_band", observed=True)["churn_flag"].agg(["size", "mean"])
tenure_trend.columns = ["customers", "churn_rate"]
tenure_trend["churn_rate_pct"] = (tenure_trend["churn_rate"] * 100).round(2)
display(tenure_trend[["customers", "churn_rate_pct"]])
tenure_trend.to_csv(TABLE_DIR / "churn_rate_by_tenure_band.csv")
eda_summary["tenure_band_churn"] = tenure_trend[["customers", "churn_rate_pct"]].to_dict("index")

# %% tags=["shot-trend-scatter"]
# Left: the lifecycle trend as a chart. Right: the relationship between tenure and
# accumulated charges, with churned customers drawn on top so they stay visible.
fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), gridspec_kw={"width_ratios": [1, 1.25]})
ax = axes[0]
positions = np.arange(len(tenure_trend))
ax.plot(positions, tenure_trend["churn_rate"] * 100, color=CHURN_COLOR, linewidth=2, marker="o",
        markersize=6, markeredgecolor="white", markeredgewidth=1.5)
for x_pos, rate in zip(positions, tenure_trend["churn_rate"]):
    ax.text(x_pos, rate * 100 + 2.5, f"{rate:.1%}", ha="center", fontsize=8.5, color=MUTED_INK)
ax.axhline(churn_rate * 100, color="#222222", linewidth=0.8)
ax.text(positions[-1], churn_rate * 100 + 1.5, f"overall {churn_rate:.1%}", ha="right", fontsize=8)
ax.set_xticks(positions)
ax.set_xticklabels(tenure_trend.index.astype(str))
ax.set_ylim(0, 60)
ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
ax.set_xlabel("Tenure band (months)")
ax.set_ylabel("Churn rate")
ax.set_title("Churn rate falls with tenure")
ax = axes[1]
for status, colour, label in (("No", RETAINED_COLOR, "Retained"), ("Yes", CHURN_COLOR, "Churned")):
    subset = df[df["Churn"] == status]
    ax.scatter(subset["tenure"], subset["TotalCharges"], s=5, color=colour, alpha=0.55, linewidths=0, label=label)
ax.set_xlabel("Tenure (months)")
ax.set_ylabel("TotalCharges (USD)")
ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
ax.set_title("Tenure vs total charges by churn status")
ax.legend(loc="upper left", markerscale=3)
fig.tight_layout()
save_figure(fig, "fig1_7_tenure_trend_and_scatter.png")

# %% [markdown]
# ### 3.5 Relationships: correlations and association with churn

# %% tags=["shot-correlation"]
corr_columns = NUMERIC_COLUMNS + ["churn_flag"]
pearson = df[corr_columns].corr(method="pearson")
spearman = df[corr_columns].corr(method="spearman")
display(pearson.round(3))
display(spearman.round(3))
pearson.round(4).to_csv(TABLE_DIR / "correlation_pearson.csv")
spearman.round(4).to_csv(TABLE_DIR / "correlation_spearman.csv")
eda_summary["pearson"] = pearson.round(4).to_dict()
eda_summary["spearman"] = spearman.round(4).to_dict()

# %%
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), layout="constrained")
for ax, matrix, name in zip(axes, [pearson, spearman], ["Pearson", "Spearman (rank)"]):
    image = ax.imshow(matrix.abs(), cmap=SEQUENTIAL_CMAP, vmin=0, vmax=1)
    ax.set_xticks(range(len(corr_columns)))
    ax.set_yticks(range(len(corr_columns)))
    ax.set_xticklabels(corr_columns, rotation=30, ha="right")
    ax.set_yticklabels(corr_columns)
    for i in range(len(corr_columns)):
        for j in range(len(corr_columns)):
            value = matrix.iloc[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if abs(value) > 0.55 else "#222222")
    ax.set_title(name)
    ax.grid(False)
axes[1].set_yticklabels([])
cbar = fig.colorbar(image, ax=axes, shrink=0.8)
cbar.set_label("Absolute correlation (colour); signed value printed in cell")
fig.suptitle("Correlation between numeric variables and churn (churn_flag: 1 = churned)",
             x=0.01, ha="left", fontsize=12, fontweight="bold")
save_figure(fig, "fig1_5_correlation_heatmap.png")

# %%
# Cramer's V: strength of association between each categorical variable and churn
# (0 = no association, 1 = perfect association)
cramers_v = {}
for column in predictor_categoricals:
    table = pd.crosstab(df[column], df["Churn"])
    cramers_v[column] = association(table.to_numpy(), method="cramer")
cramers_v = pd.Series(cramers_v).sort_values(ascending=False)
display(cramers_v.round(3).to_frame("cramers_v"))
cramers_v.round(4).to_csv(TABLE_DIR / "cramers_v_with_churn.csv", header=["cramers_v"])
eda_summary["cramers_v"] = cramers_v.round(4).to_dict()

# %%
fig, ax = plt.subplots(figsize=(8, 5))
ordered = cramers_v.sort_values()
ax.barh(ordered.index, ordered.values, height=0.6, color=CHURN_COLOR)
for y, value in enumerate(ordered.values):
    ax.text(value + 0.005, y, f"{value:.2f}", va="center", fontsize=8.5, color=MUTED_INK)
ax.set_xlabel("Cramer's V with Churn (0 = none, 1 = perfect)")
ax.set_xlim(0, 0.5)
ax.grid(axis="y", visible=False)
ax.set_title("Strength of association between each categorical variable and churn")
save_figure(fig, "fig1_6_cramers_v_association.png")

# %%
with open(RESULTS_DIR / "eda_summary.json", "w") as f:
    json.dump(eda_summary, f, indent=2, default=str)
print("Saved outputs/results/eda_summary.json")
