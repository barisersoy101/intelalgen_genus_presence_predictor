import os
import argparse

# --- 1. CONFIG IMPORTS ---
from config.config import (
    FILE_PATH_PHYTOPLANKTON,
    FILE_PATH_CHEMIE,
    BASE_SAVE_DIR,
)
from config.keys_and_column_names import (
    COLUMN_TRANSLATION_PHYTOPLANKTON,
    COLUMN_TRANSLATION_CHEMIE,
    MEASUREMENT_COLS,
    IDENTIFIER_KEYS,
    SITE_KEYS
)

# --- 2. PIPELINE MODULE IMPORTS ---
from data_loading.plankton_data import load_and_prepare_phytoplankton
from data_loading.chemie_data import load_and_prepare_chemistry
from kneedle_preprocess.kneedle_filters_and_process import (
    process_genus,
    compute_biovolume_shares,
    apply_knee_threshold,
    pivot_presence_biovolume,
    filter_rare_genera,
)
from feature_set_creators.mixer_merger import (
    apply_feature_mix,
    generate_feature_combinations,
)
from model_files import run_feature_set_training
from helper_functions import (
    merge_with_chemistry,
    normalize_keys)


def main(
    file_path_phytoplankton,
    file_path_chemie,
    column_translation_phytoplankton,
    column_translation_chemie,
    measurement_cols,
    identifier_keys,
    base_save_dir="./results",
    n_splits=5,
    random_state=42,
    max_parallel_genus=None,
):
    """Full pipeline execution function."""
    if max_parallel_genus is None:
        max_parallel_genus = os.cpu_count() or 4

    print("1. Loading Phytoplankton Data...")
    all_plankton_df, sample_dfs, stammdaten_plankton_df = load_and_prepare_phytoplankton(
        file_path_phytoplankton,
        column_translation_phytoplankton,
        identifier_keys,
    )

    print("2. Loading Chemistry Data...")
    chemie_wide_df, stammdaten_df, st_z_epi_df = load_and_prepare_chemistry(
        file_path_chemie,
        column_translation_chemie,
        measurement_cols,
        identifier_keys,
    )

    print("3. Processing Genus and Biovolume Thresholds...")
    all_plankton_df_cleaned = process_genus(all_plankton_df)
    plankton_with_shares = compute_biovolume_shares(all_plankton_df_cleaned, identifier_keys)
    plankton_thresholded = apply_knee_threshold(plankton_with_shares, identifier_keys)

    print("4. Pivoting & Filtering Rare Genera...")
    presence_df, presence_with_thresholds_df, biovolume_df = pivot_presence_biovolume(
        plankton_thresholded, identifier_keys
    )
    presence_df, biovolume_df, genus_counts, genus_columns = filter_rare_genera(
        presence_df, biovolume_df, identifier_keys
    )

    print("5. Merging Plankton with Chemistry...")
    presence_with_chem_df = merge_with_chemistry(presence_df, chemie_wide_df, identifier_keys=IDENTIFIER_KEYS)
    print(presence_with_chem_df)
    print("6. Executing Parallel XGBoost Pipeline...")
    master_record = run_feature_set_training(
        presence_with_chem_df,
        identifier_keys=IDENTIFIER_KEYS,
        base_save_dir=base_save_dir,
        n_splits=n_splits,
        random_state=random_state,
        max_parallel_genus=max_parallel_genus,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Phytoplankton Pipeline")
    parser.add_argument("--output_dir", type=str, default=BASE_SAVE_DIR, help="Directory to save models/logs")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of CV splits")
    parser.add_argument("--cores", type=int, default=4, help="Number of local CPU cores")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    master_record = main(
        file_path_phytoplankton=FILE_PATH_PHYTOPLANKTON,
        file_path_chemie=FILE_PATH_CHEMIE,
        column_translation_phytoplankton=COLUMN_TRANSLATION_PHYTOPLANKTON,
        column_translation_chemie=COLUMN_TRANSLATION_CHEMIE,
        measurement_cols=MEASUREMENT_COLS,
        identifier_keys=IDENTIFIER_KEYS,
        base_save_dir=args.output_dir,
        n_splits=args.n_splits,
        random_state=42,
        max_parallel_genus=args.cores,
    )

    print(f"\nExecution finished successfully! Outputs saved to: {args.output_dir}")