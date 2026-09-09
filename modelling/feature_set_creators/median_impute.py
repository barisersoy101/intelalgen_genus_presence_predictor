import numpy as np
import pandas as pd


def fe_impute_median(train_df, test_df, genus_columns, identifier_keys):
    """
    Imputes missing numeric features using site-level medians (grouping by identifier_keys).
    Falls back to global medians if site-level medians are missing/NaN.
    """
    train_out, test_out = train_df.copy(), test_df.copy()

    # Identify numeric non-genus, non-identifier columns to impute
    exclude_cols = set(genus_columns) | set(identifier_keys)
    impute_cols = [
        c for c in train_out.select_dtypes(include=[np.number]).columns
        if c not in exclude_cols
    ]

    if not impute_cols:
        return train_out, test_out

    # 1. Calculate site-level medians from train
    site_keys = [k for k in identifier_keys if k in train_out.columns]
    
    if site_keys:
        site_medians = train_out.groupby(site_keys)[impute_cols].median().reset_index()

        # Merge and fill site-level medians
        train_out = train_out.merge(site_medians, on=site_keys, how="left", suffixes=("", "_site_med"))
        test_out = test_out.merge(site_medians, on=site_keys, how="left", suffixes=("", "_site_med"))

        for col in impute_cols:
            med_col = f"{col}_site_med"
            if med_col in train_out.columns:
                train_out[col] = train_out[col].fillna(train_out[med_col])
                train_out.drop(columns=[med_col], inplace=True)
            if med_col in test_out.columns:
                test_out[col] = test_out[col].fillna(test_out[med_col])
                test_out.drop(columns=[med_col], inplace=True)

    # 2. Fallback: Fill remaining NaNs with global training medians
    global_medians = train_out[impute_cols].median()
    train_out[impute_cols] = train_out[impute_cols].fillna(global_medians)
    test_out[impute_cols] = test_out[impute_cols].fillna(global_medians)

    return train_out, test_out