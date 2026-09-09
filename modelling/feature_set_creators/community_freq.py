import numpy as np
import pandas as pd


# Local Community Frequency
def fe_local_community_frequency(train_df, test_df, genus_columns, target_genus, *args, **kwargs):
    train_out, test_out = train_df.copy(), test_df.copy()
    site_key = "measurement_number_bio_chem"

    other_genera = [g for g in genus_columns if g != target_genus and g in train_out.columns]
    total_g = len(other_genera)

    # Calculate per-sample community richness ratio
    train_out["_sample_ratio"] = (train_out[other_genera].sum(axis=1) / total_g).astype(np.float64)
    test_out["_sample_ratio"] = (test_out[other_genera].sum(axis=1) / total_g).astype(np.float64)

    # Calculate site-level statistics
    site_sums = train_out.groupby(site_key)["_sample_ratio"].transform("sum")
    site_cnts = train_out.groupby(site_key)["_sample_ratio"].transform("count")

    global_mean_com = float(train_out["_sample_ratio"].mean())

    # --- Train (Leave-One-Out) ---
    denom_train = site_cnts - 1.0
    train_out["LCom"] = np.where(
        denom_train > 0,
        (site_sums - train_out["_sample_ratio"]) / np.maximum(denom_train, 1.0),
        global_mean_com
    ).astype(np.float64)

    # --- Test (Direct site average) ---
    site_sums_map = train_out.groupby(site_key)["_sample_ratio"].sum()
    site_cnts_map = train_out.groupby(site_key)["_sample_ratio"].count()

    test_sums = test_out[site_key].map(site_sums_map).fillna(0).values
    test_cnts = test_out[site_key].map(site_cnts_map).fillna(0).values

    test_out["LCom"] = np.where(
        test_cnts > 0,
        test_sums / np.maximum(test_cnts, 1.0),
        global_mean_com
    ).astype(np.float64)

    # Cleanup temporary columns
    train_out.drop(columns=["_sample_ratio"], inplace=True, errors="ignore")
    test_out.drop(columns=["_sample_ratio"], inplace=True, errors="ignore")

    return train_out, test_out


# Global Community Frequency
def fe_global_community_frequency(train_df, test_df, genus_columns, target_genus, *args, **kwargs):
    train_out, test_out = train_df.copy(), test_df.copy()

    bg_genera = [g for g in genus_columns if g != target_genus and g in train_out.columns]
    total_g_count = len(bg_genera)

    total_presences_global = float(train_out[bg_genera].sum().sum())
    n_samples_global = len(train_out)

    # --- Train (Leave-One-Out) ---
    row_presence_count = train_out[bg_genera].sum(axis=1).astype(np.float64)
    loo_denom = total_g_count * max(n_samples_global - 1, 1)

    train_out["GCom"] = ((total_presences_global - row_presence_count) / loo_denom).astype(np.float64)

    # --- Test (Global mean ratio) ---
    test_out["GCom"] = np.full(
        len(test_out),
        total_presences_global / (total_g_count * max(n_samples_global, 1)),
        dtype=np.float64
    )

    return train_out, test_out