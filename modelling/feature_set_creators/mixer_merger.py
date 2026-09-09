import itertools

# Relative imports from sibling modules within feature_set_creators
from .median_impute import fe_impute_median
from .month_dummies import fe_month_dummies
from .prevalence import (
    fe_local_prevalence,
    fe_global_prevalence,
)
from .community_freq import (
    fe_local_community_frequency,
    fe_global_community_frequency,
)
from .co_occurence import (
    fe_local_target_cooccurrence,
    fe_global_target_cooccurrence,
)


# Feature Mixer Helper for Training
def apply_feature_mix(train_df, test_df, feature_functions, genus_columns, identifier_keys, target_genus):
    """
    Applies a list of feature engineering functions to train and test DataFrames.
    Always applies month dummies and median imputation first, then dispatches 
    remaining functions with their required arguments.
    """
    train_out, test_out = train_df.copy(), test_df.copy()

    # ------------------------------------------------
    # Always run these first
    # ------------------------------------------------
    train_out, test_out = fe_impute_median(
        train_out,
        test_out,
        genus_columns,
        identifier_keys
    )
    
    # Check and print NA counts right after median imputation
    train_na_count = train_out.isna().sum().sum()
    test_na_count = test_out.isna().sum().sum()
    print(f"[Post-Imputation Check] Train NAs: {train_na_count} | Test NAs: {test_na_count}")

    train_out, test_out = fe_month_dummies(train_out, test_out)

    # ------------------------------------------------
    # Run remaining feature functions with explicit argument signatures
    # ------------------------------------------------
    for fn in feature_functions:
        if fn in [fe_month_dummies, fe_impute_median]:
            continue  # already applied

        elif fn == fe_local_prevalence:
            train_out, test_out = fn(train_out, test_out, target_genus, identifier_keys)

        elif fn == fe_global_prevalence:
            train_out, test_out = fn(train_out, test_out, target_genus, identifier_keys)

        elif fn == fe_local_community_frequency:
            train_out, test_out = fn(train_out, test_out, genus_columns, target_genus, identifier_keys)

        elif fn == fe_global_community_frequency:
            train_out, test_out = fn(train_out, test_out, genus_columns, target_genus, identifier_keys)

        elif fn == fe_local_target_cooccurrence:
            train_out, test_out = fn(train_out, test_out, genus_columns, target_genus, identifier_keys)

        elif fn == fe_global_target_cooccurrence:
            train_out, test_out = fn(train_out, test_out, genus_columns, target_genus, identifier_keys)

        else:
            train_out, test_out = fn(train_out, test_out)

    return train_out, test_out


# No-op feature function to keep chemistry columns
def fe_keep_chem(train_df, test_df):
    return train_df.copy(), test_df.copy()


# Mapping for feature-set names
feature_name_map = {
    fe_local_prevalence: "lp",
    fe_global_prevalence: "gp",
    fe_local_community_frequency: "lcom",
    fe_global_community_frequency: "gcom",
    fe_local_target_cooccurrence: "lcoo",
    fe_global_target_cooccurrence: "gcoo",
}


def generate_feature_combinations():
    FEATURE_PAIRS = [
        (fe_local_prevalence, fe_global_prevalence),
        (fe_local_community_frequency, fe_global_community_frequency),
        (fe_local_target_cooccurrence, fe_global_target_cooccurrence),
    ]

    # Month dummies always applied
    base = [fe_impute_median, fe_keep_chem]

    feature_sets = []

    def with_month(features):
        return [fe_month_dummies] + features

    # -------------------
    # Baseline
    # -------------------
    feature_sets.append(("mth_chem", with_month(base)))

    local_fns = [l for l, _ in FEATURE_PAIRS]

    # -------------------
    # Local-only combinations
    # -------------------
    for r in range(1, len(local_fns) + 1):
        for locals_combo in itertools.combinations(local_fns, r):
            name = "mth_chem_" + "_".join(feature_name_map[fn] for fn in locals_combo)
            feature_sets.append((name, with_month(base + list(locals_combo))))

    # -------------------
    # Local + Global combinations (paired)
    # -------------------
    for r in range(1, len(FEATURE_PAIRS) + 1):
        for locals_combo_pairs in itertools.combinations(FEATURE_PAIRS, r):
            locals_part = [l for l, _ in locals_combo_pairs]
            globals_part = [g for _, g in locals_combo_pairs]

            # iterate over all non-empty subsets of globals
            for k in range(1, len(globals_part) + 1):
                for globals_subset in itertools.combinations(globals_part, k):
                    name = (
                        "mth_chem_"
                        + "_".join(feature_name_map[fn] for fn in locals_part)
                        + "_"
                        + "_".join(feature_name_map[fn] for fn in globals_subset)
                    )
                    feature_sets.append((name, with_month(base + locals_part + list(globals_subset))))

    # Final output: list of tuples (name, feature_function_list)
    final_sets = [(f"set_{i+1}_{name}", fn_list) for i, (name, fn_list) in enumerate(feature_sets)]
    return final_sets