import os
import time
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from config.config import N_CORES
from helper_functions import detect_genus_columns,evaluate_model
from feature_set_creators.mixer_merger import (
    apply_feature_mix,
    generate_feature_combinations,
)


def evaluate_xgb_full_features_all_genus(
    base_df,
    full_feature_set,
    identifier_keys,
    n_splits=5,
    random_state=42,
    max_parallel_genus=N_CORES,
    save_models_dir="models",
    save_logs_dir="logs"
):
    set_name, functions = full_feature_set
    all_metrics = []
    
    # Dynamic detection of binary genus columns directly inside the function
    genus_cols = detect_genus_columns(base_df, identifier_keys)
    identifier_cols = set(identifier_keys + ["pn_number", "sampling_location", "Lake_ID_MeasNum", "date"])

    print(f"Evaluating feature set: {set_name} | Target Genera Count: {len(genus_cols)}")

    os.makedirs(save_models_dir, exist_ok=True)
    os.makedirs(save_logs_dir, exist_ok=True)
    cv_data_dir = os.path.join(save_models_dir, "cv_data")
    final_data_dir = os.path.join(save_models_dir, "final_data")
    os.makedirs(cv_data_dir, exist_ok=True)
    os.makedirs(final_data_dir, exist_ok=True)

    feature_cols_base = [c for c in base_df.columns if c not in genus_cols and c not in identifier_cols]
    df = base_df.dropna(subset=feature_cols_base + genus_cols, how="all").reset_index(drop=True)

    default_params = {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 4,
        "min_child_weight": 1,        # Allow splits on rare positives
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "gamma": 0.05,                # Less aggressive pruning
        "reg_lambda": 2.0,
        "reg_alpha": 0.5,
        "scale_pos_weight": 830,    
        "random_state": random_state,
        "eval_metric": "aucpr",       # PR-AUC is honest at extreme imbalance
        "tree_method": "hist",
        "verbosity": 0,
        "n_jobs": 1,
    }

    # -----------------------
    # CV PER GENUS
    # -----------------------
    def _process_cv_genus(genus):
        y = df[genus].values
        if y.sum() == 0 or y.sum() == len(y):
            return genus, []

        genus_cv_dir = os.path.join(cv_data_dir, genus)
        os.makedirs(genus_cv_dir, exist_ok=True)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        fold_metrics_list = []
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(df, y)):
            df_train_raw, df_val_raw = df.iloc[train_idx].copy(), df.iloc[val_idx].copy()
            df_train_fe, df_val_fe = apply_feature_mix(
                df_train_raw, df_val_raw, functions, genus_cols, identifier_keys, target_genus=genus
            )

            feature_cols = [c for c in df_train_fe.columns if c not in genus_cols and c not in identifier_cols]
            X_train, X_val = df_train_fe[feature_cols], df_val_fe[feature_cols]
            y_train, y_val = y[train_idx], y[val_idx]

            fold_dir = os.path.join(genus_cv_dir, f"fold_{fold_idx}")
            os.makedirs(fold_dir, exist_ok=True)

            X_train.to_parquet(os.path.join(fold_dir, "X_train.parquet"))
            X_val.to_parquet(os.path.join(fold_dir, "X_val.parquet"))
            np.save(os.path.join(fold_dir, "y_train.npy"), y_train)
            np.save(os.path.join(fold_dir, "y_val.npy"), y_val)

            n_pos, n_neg = y_train.sum(), len(y_train) - y_train.sum()
            params = default_params.copy()
            params["scale_pos_weight"] = n_neg / max(n_pos, 1)

            model = XGBClassifier(**params)
            model.fit(X_train, y_train, verbose=False)
            y_prob = model.predict_proba(X_val)[:, 1]
            np.save(os.path.join(fold_dir, "y_pred.npy"), y_prob)

            fold_metrics_list.append(evaluate_model(y_val, y_prob))

        return genus, fold_metrics_list

    cv_results = Parallel(n_jobs=max_parallel_genus, prefer="threads")(
        delayed(_process_cv_genus)(g) for g in tqdm(genus_cols, desc="CV Parallel")
    )

    per_genus_fold_metrics = {genus: metrics for genus, metrics in cv_results if metrics}

    # -----------------------
    # AGGREGATE RESULTS
    # -----------------------
    for genus, fold_metrics_list in per_genus_fold_metrics.items():
        roc_aucs = [m["roc_auc"] for m in fold_metrics_list if not np.isnan(m["roc_auc"])]
        pr_aucs = [m["pr_auc"] for m in fold_metrics_list if not np.isnan(m["pr_auc"])]

        all_metrics.append({
            "Feature_Set": set_name,
            "Genus": genus,
            "roc_auc_mean": np.mean(roc_aucs) if roc_aucs else np.nan,
            "roc_auc_std": np.std(roc_aucs) if roc_aucs else np.nan,
            "pr_auc_mean": np.mean(pr_aucs) if pr_aucs else np.nan,
            "pr_auc_std": np.std(pr_aucs) if pr_aucs else np.nan,
        })

    detailed_df = pd.DataFrame(all_metrics)
    summary_df = (
        detailed_df.groupby("Feature_Set")[["roc_auc_mean", "roc_auc_std", "pr_auc_mean", "pr_auc_std"]]
        .mean()
        .reset_index()
        if not detailed_df.empty else pd.DataFrame()
    )

    # -----------------------
    # FINAL TRAINING
    # -----------------------
    def _train_final_genus(genus):
        df_full_train, _ = apply_feature_mix(df, df, functions, genus_cols, identifier_keys, target_genus=genus)
        final_feature_cols = [c for c in df_full_train.columns if c not in genus_cols and c not in identifier_cols]
        y_full = df_full_train[genus].values

        if y_full.sum() == 0 or y_full.sum() == len(y_full):
            return genus, None

        genus_final_dir = os.path.join(final_data_dir, genus)
        os.makedirs(genus_final_dir, exist_ok=True)

        X_full = df_full_train[final_feature_cols]
        X_full.to_parquet(os.path.join(genus_final_dir, "X_full.parquet"))
        np.save(os.path.join(genus_final_dir, "y_full.npy"), y_full)

        params = default_params.copy()
        params["scale_pos_weight"] = (len(y_full) - y_full.sum()) / max(y_full.sum(), 1)

        model = XGBClassifier(**params)
        model.fit(X_full, y_full, verbose=False)
        model.save_model(os.path.join(save_models_dir, f"{set_name}_{genus}.json"))

        return genus, model

    final_results = Parallel(n_jobs=max_parallel_genus, prefer="threads")(
        delayed(_train_final_genus)(g) for g in tqdm(genus_cols, desc="Final Training Parallel")
    )

    trained_models = {g: m for g, m in final_results if m is not None}

    detailed_df.to_csv(os.path.join(save_logs_dir, f"{set_name}_detailed.csv"), index=False)
    summary_df.to_csv(os.path.join(save_logs_dir, f"{set_name}_summary.csv"), index=False)

    print("Full-feature-set evaluation complete.\n")
    return detailed_df, summary_df, trained_models


def run_feature_set_training(
    presence_with_chem_df,
    identifier_keys,
    base_save_dir,
    n_splits=5,
    random_state=42,
    max_parallel_genus=N_CORES,
    max_feature_sets=None,
):
    intel_dir = os.path.join(base_save_dir, "intel_algen_xgboost")
    os.makedirs(intel_dir, exist_ok=True)

    feature_sets_to_try = generate_feature_combinations()
    
    #Testing
    feature_sets_to_try = feature_sets_to_try[26:]
    
    
    feature_set_strings = []
    for name, fns in feature_sets_to_try:
        fn_names = [f.__name__ for f in fns]
        feature_set_strings.append(" + ".join(fn_names))

    print("\nFeature sets to evaluate (feature names only):\n")
    for s in feature_set_strings:
        print(s)

    if max_feature_sets is not None:
        feature_sets_to_try = feature_sets_to_try[:max_feature_sets]

    print(f"\nTotal feature sets: {len(feature_sets_to_try)}\n")
    master_record = {}
    start_time = time.time()

    for fs_idx, feature_set in enumerate(feature_sets_to_try, 1):
        set_name = feature_set[0]
        print(f"[{fs_idx}/{len(feature_sets_to_try)}] Evaluating feature set: {set_name}")

        set_dir = os.path.join(intel_dir, set_name)
        models_dir = os.path.join(set_dir, "models")
        logs_dir = os.path.join(set_dir, "logs")
        data_dir = os.path.join(models_dir, "transformed_data")

        detailed_df, summary_df, trained_models = evaluate_xgb_full_features_all_genus(
            base_df=presence_with_chem_df,
            full_feature_set=feature_set,
            identifier_keys=identifier_keys,
            n_splits=n_splits,
            random_state=random_state,
            max_parallel_genus=max_parallel_genus,
            save_models_dir=models_dir,
            save_logs_dir=logs_dir
        )

        master_record[set_name] = {
            "models_dir": models_dir,
            "logs_dir": logs_dir,
            "data_dir": data_dir,
            "feature_lists_dir": os.path.join(models_dir, "feature_lists"),
            "num_genera_trained": len(trained_models),
            "genera_trained": list(trained_models.keys())
        }
        del trained_models

    master_record_path = os.path.join(intel_dir, "training_master_record.json")
    with open(master_record_path, "w") as f:
        json.dump(master_record, f, indent=4)

    elapsed = time.time() - start_time
    print(f"\nAll feature sets evaluated. Total runtime: {elapsed/60:.2f} minutes")
    print(f"Master record saved at: {master_record_path}")

    return master_record