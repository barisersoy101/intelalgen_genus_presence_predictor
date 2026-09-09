import re
import numpy as np
import pandas as pd
from kneed import KneeLocator
from helper_functions import extract_genus
from config.keys_and_column_names import IDENTIFIER_KEYS as identifier_keys


# Find Kneedle threshold per sample
def find_knee_threshold(shares):
    shares = np.sort(np.asarray(shares, dtype=float))[::-1]
    if len(shares) < 2:
        return None
    x = np.arange(1, len(shares) + 1)
    try:
        kneedle = KneeLocator(x, shares, curve='convex', direction='decreasing')
        knee = kneedle.knee
    except Exception:
        return None
    if knee is None or knee <= 0 or knee >= len(shares):
        return None
    return float(shares[int(knee) - 1])

# Process Genus
def process_genus(df):
    df = df.copy()
    df["genus"] = df["taxon"].apply(extract_genus)
    unique_before = df["genus"].nunique()
    df = df[~df["genus"].str.lower().str.contains("unbestimmte", na=False)].copy()
    unique_after = df["genus"].nunique()
    return df

# Compute total biovolume per sample and relative shares
def compute_biovolume_shares(df, identifier_keys):
    sample_totals = (
        df.groupby(identifier_keys, as_index=False)["biovolume_cm3_m3"]
          .sum()
          .rename(columns={"biovolume_cm3_m3": "total_biovolume"})
    )
    df = df.merge(sample_totals, on=identifier_keys, how="left")
    df["share"] = df["biovolume_cm3_m3"] / (df["total_biovolume"] + 1e-12)
    return df

# Apply Kneedle threshold per sample
def apply_knee_threshold(df, identifier_keys):
    cleaned_rows = []
    for key, group in df.groupby(identifier_keys):
        total_bv = group["total_biovolume"].iloc[0]
        threshold_share = find_knee_threshold(group["share"].values)
        group = group.copy()
        if threshold_share is not None:
            threshold_bv = threshold_share * total_bv
            cond = group["biovolume_cm3_m3"] >= threshold_bv
            group["above_threshold"] = cond.astype(int)
            group["bv_thresholded"] = np.where(cond, group["biovolume_cm3_m3"], 0.0)
            group["knee_threshold_share"] = threshold_share
            group["knee_threshold_biovolume"] = threshold_bv
        else:
            group["above_threshold"] = 1
            group["bv_thresholded"] = group["biovolume_cm3_m3"].astype(float)
            group["knee_threshold_share"] = np.nan
            group["knee_threshold_biovolume"] = np.nan
        cleaned_rows.append(group)
    return pd.concat(cleaned_rows, ignore_index=True)

# Pivot presence and biovolume
def pivot_presence_biovolume(df, identifier_keys):
    presence_df = df.pivot_table(
        index=identifier_keys,
        columns="genus",
        values="above_threshold",
        aggfunc="max",
        fill_value=0
    ).reset_index()
    presence_df.columns.name = None

    biovolume_df = df.pivot_table(
        index=identifier_keys,
        columns="genus",
        values="bv_thresholded",
        aggfunc="sum",
        fill_value=0.0
    ).reset_index()
    biovolume_df.columns.name = None

    threshold_info_df = df.drop_duplicates(subset=identifier_keys)[
        identifier_keys + ["knee_threshold_share", "knee_threshold_biovolume"]
    ]

    presence_with_thresholds_df = presence_df.merge(
        threshold_info_df, on=identifier_keys, how="left"
    )

    return presence_df, presence_with_thresholds_df, biovolume_df

def filter_rare_genera(presence_df, biovolume_df, identifier_keys):

    # initial genus columns
    genus_cols = [c for c in presence_df.columns if c not in identifier_keys]
    genus_counts = presence_df[genus_cols].sum(axis=0).sort_values(ascending=False)
    x = np.arange(1, len(genus_counts) + 1)
    y = np.sort(genus_counts.values)[::-1]
    kneedle = KneeLocator(x, y, curve='convex', direction='decreasing')
    knee = kneedle.knee
    threshold_occurrence = (
        y[int(knee) - 1] if knee is not None and knee > 0 else None
    )
    if threshold_occurrence is not None:
        rare_genera = genus_counts[genus_counts < threshold_occurrence].index.tolist()
        for df_ in [presence_df, biovolume_df]:
            drop_cols = [g for g in rare_genera if g in df_.columns]
            if drop_cols:
                df_.drop(columns=drop_cols, inplace=True)

    # recompute genus  after filtering
    genus_columns = [c for c in presence_df.columns if c not in identifier_keys]

    return presence_df, biovolume_df, genus_counts, genus_columns

