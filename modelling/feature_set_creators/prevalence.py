import numpy as np
import pandas as pd


def fe_local_prevalence(train_df, test_df, target_genus, *args, **kwargs):
    train_out, test_out = train_df.copy(), test_df.copy()
    site_key = "measurement_number_bio_chem"

    # Site-level totals and counts
    site_sums = train_out.groupby(site_key)[target_genus].transform("sum")
    site_cnts = train_out.groupby(site_key)[target_genus].transform("count")

    global_mean = float(train_out[target_genus].mean())

    # --- Train (Leave-One-Out) ---
    denom_train = site_cnts - 1.0
    train_out["LP_" + str(target_genus)] = np.where(
        denom_train > 0,
        (site_sums - train_out[target_genus]) / np.maximum(denom_train, 1.0),
        global_mean
    ).astype(np.float64)

    # --- Test (Direct site average) ---
    site_sum_map = train_out.groupby(site_key)[target_genus].sum()
    site_cnt_map = train_out.groupby(site_key)[target_genus].count()

    test_sums = test_out[site_key].map(site_sum_map).fillna(0).values
    test_cnts = test_out[site_key].map(site_cnt_map).fillna(0).values

    test_out["LP_" + str(target_genus)] = np.where(
        test_cnts > 0,
        test_sums / np.maximum(test_cnts, 1.0),
        global_mean
    ).astype(np.float64)

    return train_out, test_out


def fe_global_prevalence(train_df, test_df, target_genus, *args, **kwargs):
    train_out, test_out = train_df.copy(), test_df.copy()

    total_sum = float(train_out[target_genus].sum())
    total_cnt = len(train_out)
    global_mean = float(train_out[target_genus].mean())

    # --- Train (Leave-One-Out) ---
    loo_denom = max(total_cnt - 1, 1)
    train_out["GP_" + str(target_genus)] = (
        (total_sum - train_out[target_genus]) / loo_denom
    ).astype(np.float64)

    # --- Test (Overall mean) ---
    test_out["GP_" + str(target_genus)] = np.full(
        len(test_out), global_mean, dtype=np.float64
    )

    return train_out, test_out