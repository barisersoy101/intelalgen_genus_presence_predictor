import pandas as pd
import numpy as np
import re

def load_and_prepare_chemistry(
    file_path_chemie,
    column_translation_chemie,
    measurement_cols,
    identifier_keys,
):

    # Load Excel sheets
    sheets_dict = pd.read_excel(file_path_chemie, sheet_name=None)

    # Extract metadata sheets
    stammdaten_df = sheets_dict.pop("Stammdaten", None)
    st_z_epi_df = sheets_dict.pop("ST+Z_epi", None)

    # Special sheet
    daten_seen_2005_2006_df = sheets_dict.pop(
        "Daten_Seen_2005-2006", None
    )

    # Combine remaining sheets
    df_dict = {
        re.sub(r"\W+", "_", name): df
        for name, df in sheets_dict.items()
    }

    combined_df = (
        pd.concat(df_dict.values(), ignore_index=True)
        if df_dict
        else pd.DataFrame()
    )

    if daten_seen_2005_2006_df is not None:
        combined_df = pd.concat(
            [combined_df, daten_seen_2005_2006_df],
            ignore_index=True,
        )

    df = combined_df.copy()

    # Rename columns
    df.rename(columns=column_translation_chemie, inplace=True)

    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
    )

    # Normalize key columns
    if "water_body" in df.columns:
        df["water_body"] = (
            df["water_body"]
            .astype(str)
            .str.strip()
            .str.lower()
            .str.replace(r"[^\w]+", "_", regex=True)
        )

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["date"] = (
            df["date"]
            .dt.to_period("M")
            .dt.to_timestamp()
        )

    if "measurement_number" in df.columns:
        df["measurement_number"] = pd.to_numeric(
            df["measurement_number"]
            .astype(str)
            .str.extract(r"(\d+)")[0],
            errors="coerce",
        ).astype("Int64")

    # Value cleaning
    def replace_less_than(value):
        if isinstance(value, str):
            val = value.strip().replace(",", ".").lower()

            if val in {"n.b.", "n.b", "nb", "n_b"}:
                return np.nan

            if val.startswith("<"):
                try:
                    return float(val[1:].strip()) / 2
                except ValueError:
                    return np.nan

            try:
                return float(val)
            except ValueError:
                return np.nan

        return value

    for col in measurement_cols:
        if col in df.columns:
            df[col] = df[col].map(replace_less_than)
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Merge phosphorus variants smartly
    tp_cols = [
        "total_phosphorus_mg_per_l",
        "total_phosphorus_fia_mg_per_l",
    ]

    if all(c in df.columns for c in tp_cols):
        
        def combine_phosphorus(row):
            vals = [row[c] for c in tp_cols]
            valid_vals = [v for v in vals if pd.notna(v) and v != 0]
            
            if not valid_vals:
                return 0  # if both missing/zero
            elif len(valid_vals) == 1:
                return valid_vals[0]  # only one non-zero value
            else:
                return sum(valid_vals)/len(valid_vals)  # average if both non-zero
        
        df["total_phosphorus_combined_mg_per_l"] = df.apply(combine_phosphorus, axis=1)
        
        df.drop(columns=tp_cols, inplace=True)


    # --- Combine silica measurements smartly ---
    silica_cols = [
        "dissolved_silica_mg_per_l",
        "dissolved_silica_fia_mg_per_l",
    ]

    if all(c in df.columns for c in silica_cols):
        
        def combine_silica(row):
            vals = [row[c] for c in silica_cols]
            # Filter out zeros and NaNs
            valid_vals = [v for v in vals if pd.notna(v) and v != 0]
            
            if not valid_vals:
                return 0  # if both missing/zero
            elif len(valid_vals) == 1:
                return valid_vals[0]  # only one non-zero value
            else:
                return sum(valid_vals)/len(valid_vals)  # average if both non-zero
        
        df["dissolved_silica_combined_mg_per_l"] = df.apply(combine_silica, axis=1)
        
        df.drop(columns=silica_cols, inplace=True)

    # Drop unwanted columns
    cols_to_drop = [
        "orthophosphate_fia_mg_per_l",
        "ammonium_fia_mg_per_l",
        "nitrate_mg_per_l",
        "orthophosphate_mg_per_l",
    ]

    df.drop(
        columns=[c for c in cols_to_drop if c in df.columns],
        inplace=True,
    )

    # Aggregate to wide format
    final_measurement_cols = [
        c for c in df.columns if c not in identifier_keys
    ]

    chemie_wide_df = (
        df[identifier_keys + final_measurement_cols]
        .groupby(identifier_keys, dropna=True)
        .mean(numeric_only=True)
        .reset_index()
    )

    return (
        chemie_wide_df,
        stammdaten_df,
        st_z_epi_df,
    )