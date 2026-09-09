import re
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


# Detects which columns are genus or not. This is also used in the feature calculations, shap and inside the model to detect features.
def detect_genus_columns(df, identifier_keys):
    exclude = set(identifier_keys)
    engineered_prefixes = ("LP_", "GP_", "LCoO_", "GCoO_")
    engineered_exact = ("LCom", "GCom")
    genus_cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if c.startswith(engineered_prefixes) or c in engineered_exact:
            continue
        if pd.api.types.is_numeric_dtype(df[c]) and df[c].dropna().isin([0, 1]).all():
            genus_cols.append(c)
    return genus_cols

# Extract genus names and clean unclassified entries
def extract_genus(taxon):
    if pd.isna(taxon):
        return None
    name = str(taxon).strip()
    name = re.sub(r'\s+', ' ', name)
    return name.split(" ")[0]

# EVALUATION HELPERS
def evaluate_model(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return {"roc_auc": np.nan, "pr_auc": np.nan}

    return {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob)
    }

def summarize_cv_results(results):
    if not results:
        return pd.DataFrame()
    df = pd.DataFrame(results)
    return df.describe().T

def merge_with_chemistry(plankton_df, chem_df, identifier_keys):
    # Normalize water bodies and dates
    plankton_df = normalize_keys(plankton_df, water_col="water_body", date_col="date")
    chem_df = normalize_keys(chem_df, water_col="water_body", date_col="date")

    # Ensure measurement_number is integer in both DataFrames
    for col in ["measurement_number", "measurement_number_bio_chem"]:
        if col in plankton_df.columns:
            plankton_df[col] = pd.to_numeric(plankton_df[col], errors="coerce").astype("Int64")
        if col in chem_df.columns:
            chem_df[col] = pd.to_numeric(chem_df[col], errors="coerce").astype("Int64")

    # Merge dynamically using identifier_keys
    merged_df = plankton_df.merge(
        chem_df,
        left_on=identifier_keys,
        right_on=identifier_keys,
        how="inner",
        indicator=True
    )

    # Drop duplicate columns from merge
    merged_df.drop(columns=["_merge"], inplace=True, errors="ignore")

    # Identify chemistry columns (numeric but not genus columns)
    genus_cols = detect_genus_columns(merged_df,identifier_keys)
    numeric_cols = merged_df.select_dtypes(include=[np.number]).columns.tolist()
    chem_cols = [c for c in numeric_cols if c not in genus_cols]

    # Drop rows where all chemistry measurements are missing
    merged_df = merged_df.dropna(subset=chem_cols, how="all").drop_duplicates()

    return merged_df


# Normalize water body names and convert dates to monthly timestamps
def normalize_keys(df, water_col="water_body", date_col="date"):
    df = df.copy()
    if water_col in df.columns:
        df[water_col] = (
            df[water_col].astype(str)
            .str.lower()
            .str.strip()
            .str.replace(r"[^\w]+", "_", regex=True)
        )
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df[date_col] = df[date_col].dt.to_period("M").dt.to_timestamp()
    return df

def get_genus_list(df, identifier_keys):
    """Returns a clean list of binary genus target columns."""
    return list(detect_genus_columns(df, identifier_keys))