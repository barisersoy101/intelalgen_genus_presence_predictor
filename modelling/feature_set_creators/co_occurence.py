import numpy as np
import pandas as pd


# Local Target Co-occurrence (LOO for train)
def fe_local_target_cooccurrence(train_df, test_df, genus_columns, target_genus, *args, **kwargs):
    train_out, test_out = train_df.copy(), test_df.copy()
    site_key = "measurement_number_bio_chem"
    genus_cols = [g for g in genus_columns if g in train_out.columns]

    global_dot = train_out[genus_cols].T @ train_out[genus_cols]
    global_n = len(train_out)
    site_counts = train_out.groupby(site_key).size()

    new_train, new_test = {}, {}

    for g_o in genus_cols:
        if g_o == target_genus:
            continue

        col = f"LCoO_{g_o}_with_{target_genus}"

        train_out["_prod"] = (train_out[g_o] * train_out[target_genus]).astype(np.float64)
        test_out["_prod"] = (test_out[g_o] * test_out[target_genus]).astype(np.float64) if g_o in test_out else 0.0

        site_sum_tr = train_out.groupby(site_key)["_prod"].transform("sum")
        site_cnt_tr = train_out.groupby(site_key)["_prod"].transform("count")

        global_fallback = float(global_dot.loc[g_o, target_genus] / max(global_n, 1))

        # Train LOO
        denom_train = site_cnt_tr - 1.0
        train_vals = np.where(
            denom_train > 0,
            (site_sum_tr - train_out["_prod"]) / np.maximum(denom_train, 1.0),
            global_fallback
        ).astype(np.float64)

        # Test
        site_sum_map = train_out.groupby(site_key)["_prod"].sum()
        test_sums = test_out[site_key].map(site_sum_map).fillna(0).values
        test_cnts = test_out[site_key].map(site_counts).fillna(0).values

        test_vals = np.where(
            test_cnts > 0,
            test_sums / np.maximum(test_cnts, 1.0),
            global_fallback
        ).astype(np.float64)

        new_train[col] = train_vals
        new_test[col] = test_vals

    train_out.drop(columns=["_prod"], inplace=True, errors="ignore")
    test_out.drop(columns=["_prod"], inplace=True, errors="ignore")

    train_out = pd.concat([train_out, pd.DataFrame(new_train, index=train_out.index)], axis=1)
    test_out = pd.concat([test_out, pd.DataFrame(new_test, index=test_out.index)], axis=1)

    return train_out, test_out


# Global Target Co-occurrence (LOO for train)
def fe_global_target_cooccurrence(train_df, test_df, genus_columns, target_genus, *args, **kwargs):
    train_out, test_out = train_df.copy(), test_df.copy()
    genus_cols = [g for g in genus_columns if g in train_out.columns]

    dot_sum = train_out[genus_cols].T @ train_out[genus_cols]
    N = len(train_out)

    new_train, new_test = {}, {}

    for g_o in genus_cols:
        if g_o == target_genus:
            continue

        col = f"GCoO_{g_o}_with_{target_genus}"
        global_dot = float(dot_sum.loc[g_o, target_genus])
        global_mean = global_dot / max(N, 1)

        new_test[col] = np.full(len(test_out), global_mean, dtype=np.float64)

        row_contribs = (train_out[g_o].values * train_out[target_genus].values).astype(np.float64)
        loo_n = N - 1

        if loo_n > 0:
            train_vals = ((global_dot - row_contribs) / loo_n).astype(np.float64)
        else:
            train_vals = np.full(N, global_mean, dtype=np.float64)

        new_train[col] = train_vals

    train_out = pd.concat([train_out, pd.DataFrame(new_train, index=train_out.index)], axis=1)
    test_out = pd.concat([test_out, pd.DataFrame(new_test, index=test_out.index)], axis=1)

    return train_out, test_out