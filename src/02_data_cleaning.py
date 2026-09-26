# %% [markdown]
# ## Part 2 - Initial inspection, data quality assessment and cleaning
#
# The raw file is inspected before anything is changed. Each cleaning decision
# below is based on something found during inspection, and every change is
# logged so the before/after state can be reported.

# %%
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

from config import CLEAN_DATA_PATH, RAW_DATA_PATH, RESULTS_DIR, TABLE_DIR
from data_utils import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS
from plot_style import CHURN_COLOR, GRID_COLOR, MUTED_INK, apply_style, save_figure

pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 220)
apply_style()

df_raw = pd.read_csv(RAW_DATA_PATH)
cleaning_log = {"raw_shape": list(df_raw.shape)}

# %% [markdown]
# ### 2.1 First look at the raw data

# %% tags=["shot-inspection"]
print("Shape (rows, columns):", df_raw.shape)
# head() shown in three column blocks so all 21 columns stay readable
for block in (slice(0, 7), slice(7, 14), slice(14, 21)):
    display(df_raw.head().iloc[:, block])

# %%
for block in (slice(0, 7), slice(7, 14), slice(14, 21)):
    display(df_raw.tail().iloc[:, block])

# %%
print(df_raw.columns.tolist())

# %% tags=["shot-info"]
df_raw.info()

# %%
display(df_raw.dtypes.value_counts().rename("number_of_columns").to_frame())

# %% tags=["shot-describe"]
# Only three columns are numeric in the raw file - TotalCharges is missing
# from this summary because it was read as text.
display(df_raw.describe().round(2))

# %%
display(df_raw.describe(include="all").T)

# %% [markdown]
# ### 2.2 Unique values and category frequencies

# %%
unique_summary = pd.DataFrame({
    "dtype": df_raw.dtypes.astype(str),
    "n_unique": df_raw.nunique(),
    "sample_values": [", ".join(map(str, df_raw[c].unique()[:4])) for c in df_raw.columns],
})
display(unique_summary)
unique_summary.to_csv(TABLE_DIR / "unique_values_summary.csv")

# %%
key_categoricals = ["Churn", "Contract", "InternetService", "PaymentMethod"]
for column in key_categoricals:
    counts = df_raw[column].value_counts()
    display(pd.DataFrame({"count": counts,
                          "percent": (counts / len(df_raw) * 100).round(2)}))

# %% [markdown]
# ### 2.3 Missing values
#
# `isna()` only counts true NaN values. Text columns can also hide missing
# data as empty strings, so both are checked.

# %% tags=["shot-missing"]
def missing_summary(frame):
    """Missing count and percentage for every column."""
    return pd.DataFrame({
        "missing_count": frame.isna().sum(),
        "missing_pct": (frame.isna().mean() * 100).round(3),
    })


print("NaN cells in raw data:", int(df_raw.isna().sum().sum()))

text_columns = df_raw.select_dtypes(exclude="number").columns
blank_strings = {c: int(df_raw[c].str.strip().eq("").sum()) for c in text_columns}
blank_strings = {c: n for c, n in blank_strings.items() if n > 0}
print("Columns containing blank strings:", blank_strings)

# Which TotalCharges values cannot be read as numbers?
converted = pd.to_numeric(df_raw["TotalCharges"], errors="coerce")
print("Non-numeric TotalCharges values:", df_raw.loc[converted.isna(), "TotalCharges"].unique().tolist())

# %% tags=["shot-missing-2"]
# Convert TotalCharges to a number; the blank strings become NaN
df = df_raw.copy()
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].str.strip().replace("", np.nan))

missing_after_conversion = missing_summary(df)
display(missing_after_conversion[missing_after_conversion["missing_count"] > 0])

# %% tags=["shot-missing-pattern"]
# Is missingness related to anything? Compare with tenure.
zero_tenure = df["tenure"].eq(0)
display(pd.crosstab(zero_tenure.rename("tenure == 0"),
                    df["TotalCharges"].isna().rename("TotalCharges missing")))
missing_rows = df.loc[df["TotalCharges"].isna(),
                      ["customerID", "tenure", "Contract", "MonthlyCharges", "TotalCharges", "Churn"]]
display(missing_rows)

# %%
# Missingness map: every column, customers ordered by tenure (x axis)
order = df.sort_values("tenure", kind="stable").reset_index(drop=True)
fig, ax = plt.subplots(figsize=(9, 5.2))
for y_position, column in enumerate(order.columns):
    ax.axhline(y_position, color=GRID_COLOR, linewidth=0.6, zorder=0)
    missing_positions = np.flatnonzero(order[column].isna().to_numpy())
    ax.scatter(missing_positions, np.full(len(missing_positions), y_position),
               marker="|", s=90, linewidths=1.6, color=CHURN_COLOR, zorder=2)
ax.set_yticks(range(len(order.columns)))
ax.set_yticklabels(order.columns, fontsize=8)
ax.invert_yaxis()
ax.set_xlim(-150, len(order) + 50)
ax.grid(False)
ax.set_xlabel("Customers ordered by tenure (row position, 0 = shortest tenure)")
ax.set_title("Missing values after converting TotalCharges to numeric")
ax.annotate("11 missing TotalCharges values,\nall at tenure = 0 (rows 0-10)",
            xy=(10, list(order.columns).index("TotalCharges")),
            xytext=(900, list(order.columns).index("TotalCharges") - 3.5),
            fontsize=9, color=MUTED_INK,
            arrowprops=dict(arrowstyle="-", color=MUTED_INK, linewidth=0.8))
save_figure(fig, "fig1_1_missing_value_map.png")

# %% [markdown]
# **Treatment.** All 11 customers with a blank `TotalCharges` have `tenure = 0`
# and no other customer has `tenure = 0`, so the values are missing because
# these accounts have not completed a billing month yet. Their accumulated
# charges are therefore set to 0. Dropping the rows or filling with the median
# (about $1,400) were rejected - the median would give brand-new accounts a
# billing history they do not have.

# %% tags=["shot-cleaning"]
total_charges_before = df["TotalCharges"].describe()

fill_mask = df["TotalCharges"].isna() & zero_tenure
df.loc[fill_mask, "TotalCharges"] = 0.0

total_charges_after = df["TotalCharges"].describe()
impact = pd.DataFrame({"before (NaN excluded)": total_charges_before,
                       "after (filled with 0)": total_charges_after}).round(2)
display(impact)
print("Remaining missing values:", int(df.isna().sum().sum()))

# %%
# Save the before/after evidence and log the counts for the report
missing_before_after = pd.DataFrame({
    "raw_nan": df_raw.isna().sum(),
    "raw_blank_strings": pd.Series(blank_strings).reindex(df_raw.columns).fillna(0).astype(int),
    "after_type_conversion": missing_after_conversion["missing_count"],
    "after_treatment": df.isna().sum(),
})
missing_before_after.to_csv(TABLE_DIR / "missing_values_before_after.csv")
impact.to_csv(TABLE_DIR / "totalcharges_imputation_impact.csv")

cleaning_log["missing"] = {
    "raw_nan_cells": int(df_raw.isna().sum().sum()),
    "blank_string_columns": blank_strings,
    "missing_after_conversion": int(df["TotalCharges"].isna().sum() + fill_mask.sum()),
    "missing_rows_all_tenure_zero": bool(df_raw.loc[converted.isna(), "tenure"].eq(0).all()),
    "tenure_zero_rows": int(zero_tenure.sum()),
    "missing_rows_churn_values": df_raw.loc[converted.isna(), "Churn"].value_counts().to_dict(),
    "missing_rows_contract_values": df_raw.loc[converted.isna(), "Contract"].value_counts().to_dict(),
    "filled_with_zero": int(fill_mask.sum()),
    "missing_after_treatment": int(df.isna().sum().sum()),
    "totalcharges_mean_before": round(float(total_charges_before["mean"]), 2),
    "totalcharges_mean_after": round(float(total_charges_after["mean"]), 2),
    "totalcharges_median_before": round(float(total_charges_before["50%"]), 2),
    "totalcharges_median_after": round(float(total_charges_after["50%"]), 2),
}

# %% [markdown]
# ### 2.4 Duplicate records

# %% tags=["shot-duplicates"]
full_row_duplicates = int(df.duplicated().sum())
duplicate_ids = int(df["customerID"].duplicated().sum())
profile_duplicates = int(df.drop(columns="customerID").duplicated().sum())
rows_in_profile_groups = int(df.drop(columns="customerID").duplicated(keep=False).sum())

print("Fully duplicated rows            :", full_row_duplicates)
print("Duplicated customerID values     :", duplicate_ids)
print("Rows repeating another customer's")
print("  attributes when ID is ignored  :", profile_duplicates,
      f"(involving {rows_in_profile_groups} rows)")

# %%
# What do the same-profile rows look like? Distinct IDs, identical attributes.
attribute_columns = [c for c in df.columns if c != "customerID"]
profile_groups = df[df.duplicated(subset=attribute_columns, keep=False)]
display(profile_groups.sort_values(attribute_columns).head(6))
display(profile_groups[["tenure", "MonthlyCharges"]].describe().round(2))

cleaning_log["duplicates"] = {
    "full_row_duplicates": full_row_duplicates,
    "duplicate_customer_ids": duplicate_ids,
    "profile_duplicates_excluding_id": profile_duplicates,
    "rows_in_profile_duplicate_groups": rows_in_profile_groups,
    "profile_duplicates_max_tenure": int(profile_groups["tenure"].max()),
    "profile_duplicates_internet_service": profile_groups["InternetService"].value_counts().to_dict(),
    "rows_removed": 0,
}

# %% [markdown]
# No duplicate records exist: every row has its own `customerID` and no row is
# repeated in full. The 22 rows that match another row once the ID is ignored
# belong to different customers on very common, simple plans (short tenure,
# identical price), so they are kept.

# %% [markdown]
# ### 2.5 Data quality checks: labels, whitespace, ranges and consistency

# %% tags=["shot-quality"]
quality_checks = {}

# (a) Leading/trailing whitespace in text columns (after the TotalCharges fix)
text_columns = df.select_dtypes(exclude="number").columns
quality_checks["values_with_extra_whitespace"] = int(
    sum((df[c] != df[c].str.strip()).sum() for c in text_columns))

# (b) Labels that differ only by case or spacing (e.g. "Yes" vs "yes ")
case_conflicts = {c: int(df[c].nunique() - df[c].str.strip().str.lower().nunique())
                  for c in text_columns if c != "customerID"}
quality_checks["case_or_spacing_label_conflicts"] = int(sum(case_conflicts.values()))

# (c) Structural labels must agree with the parent service column
addon_columns = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                 "TechSupport", "StreamingTV", "StreamingMovies"]
no_internet = df["InternetService"].eq("No")
quality_checks["phone_label_inconsistencies"] = int(
    (df["PhoneService"].eq("No") != df["MultipleLines"].eq("No phone service")).sum())
quality_checks["internet_label_inconsistencies"] = int(sum(
    (no_internet != df[c].eq("No internet service")).sum() for c in addon_columns))

# (d) Impossible numeric values
quality_checks["negative_numeric_values"] = int((df[NUMERIC_COLUMNS] < 0).sum().sum())
quality_checks["non_positive_monthly_charges"] = int(df["MonthlyCharges"].le(0).sum())
quality_checks["totalcharges_below_monthly_when_tenure_ge_1"] = int(
    (df["tenure"].ge(1) & (df["TotalCharges"] < df["MonthlyCharges"] - 0.01)).sum())

display(pd.Series(quality_checks, name="count").to_frame())
display(df[NUMERIC_COLUMNS].agg(["min", "max"]).round(2))

# %%
# (e) Billing consistency: TotalCharges vs tenure x current MonthlyCharges.
# Differences are expected when prices changed during the customer's life.
billed = df[df["tenure"] > 0].copy()
billed["expected_total"] = billed["tenure"] * billed["MonthlyCharges"]
billed["ratio"] = billed["TotalCharges"] / billed["expected_total"]
display(billed["ratio"].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).round(3).to_frame())

deviation = (billed["ratio"] - 1).abs()
quality_checks["billing_deviation_over_10pct"] = int(deviation.gt(0.10).sum())
quality_checks["billing_deviation_over_25pct"] = int(deviation.gt(0.25).sum())
print("Customers whose TotalCharges differs from tenure x MonthlyCharges by >10%:",
      quality_checks["billing_deviation_over_10pct"])
print("... by >25%:", quality_checks["billing_deviation_over_25pct"])
display(billed.loc[deviation.nlargest(5).index,
                   ["customerID", "tenure", "MonthlyCharges", "TotalCharges", "expected_total", "ratio"]].round(2))

# %%
# (f) Outliers by the 1.5 x IQR rule - flagged, not removed
def iqr_outlier_count(series):
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    return int(((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum())

outlier_counts = {c: iqr_outlier_count(df[c]) for c in NUMERIC_COLUMNS}
print("IQR-rule outliers per numeric column:", outlier_counts)

cleaning_log["quality_checks"] = quality_checks
cleaning_log["quality_checks"]["iqr_outliers"] = outlier_counts
cleaning_log["quality_checks"]["billing_ratio_median"] = round(float(billed["ratio"].median()), 4)
cleaning_log["quality_checks"]["billing_ratio_p01"] = round(float(billed["ratio"].quantile(0.01)), 3)
cleaning_log["quality_checks"]["billing_ratio_p99"] = round(float(billed["ratio"].quantile(0.99)), 3)
cleaning_log["numeric_ranges"] = df[NUMERIC_COLUMNS].agg(["min", "max"]).round(2).to_dict()
pd.Series(quality_checks).to_csv(TABLE_DIR / "data_quality_checks.csv", header=["count"])

# %% [markdown]
# ### 2.6 Data type corrections

# %% tags=["shot-dtypes"]
dtypes_before = df_raw.dtypes.astype(str)
memory_before = df_raw.memory_usage(deep=True).sum()

# SeniorCitizen is a yes/no flag stored as 0/1; use the same labels as Partner etc.
df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

# Text columns with a small fixed set of values become pandas categories
for column in CATEGORICAL_COLUMNS:
    df[column] = df[column].astype("category")

# Numeric churn flag for correlations and modelling (1 = churned)
df["churn_flag"] = (df["Churn"] == "Yes").astype("int64")

dtype_changes = pd.DataFrame({"before": dtypes_before,
                              "after": df.dtypes.astype(str).reindex(dtypes_before.index)})
dtype_changes.loc["churn_flag"] = ["(new column)", str(df["churn_flag"].dtype)]
display(dtype_changes[dtype_changes["before"] != dtype_changes["after"]])
memory_after = df.memory_usage(deep=True).sum()
print(f"Memory: {memory_before / 1024:,.0f} KB -> {memory_after / 1024:,.0f} KB")

# %%
dtype_changes.to_csv(TABLE_DIR / "dtype_changes.csv")
cleaning_log["dtypes"] = {
    "changed_columns": dtype_changes[dtype_changes["before"] != dtype_changes["after"]].to_dict("index"),
    "memory_kb_before": round(memory_before / 1024, 1),
    "memory_kb_after": round(memory_after / 1024, 1),
}

# %% [markdown]
# ### 2.7 Save the cleaned dataset

# %% tags=["shot-clean-summary"]
df.to_csv(CLEAN_DATA_PATH, index=False)
cleaning_log["clean_shape"] = list(df.shape)
cleaning_log["rows_removed"] = int(df_raw.shape[0] - df.shape[0])

print("Raw shape  :", df_raw.shape)
print("Clean shape:", df.shape, "(churn_flag added)")
print("Missing values remaining:", int(df.isna().sum().sum()))
df.info()

with open(RESULTS_DIR / "cleaning_log.json", "w") as f:
    json.dump(cleaning_log, f, indent=2, default=str)
print("Saved data/processed/telco_churn_clean.csv and outputs/results/cleaning_log.json")
