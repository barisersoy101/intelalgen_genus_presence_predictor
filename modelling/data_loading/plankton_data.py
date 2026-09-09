import pandas as pd
import re

def load_and_prepare_phytoplankton(
    file_path_phytoplankton,
    column_translation_phytoplankton,
    identifier_keys,
):

    # Load Excel sheets
    sheets_dict = pd.read_excel(file_path_phytoplankton, sheet_name=None)

    # Separate Stammdaten if present
    stammdaten_plankton_df = sheets_dict.pop("Stammdaten", None)

    # Merge remaining sheets
    all_plankton_df = pd.concat(sheets_dict.values(), ignore_index=True)

    # Rename columns
    all_plankton_df.rename(
        columns=column_translation_phytoplankton,
        inplace=True,
    )

    # Cleaning helpers
    def clean_taxon_name(taxon):
        taxon = str(taxon)
        taxon = re.sub(r'\s*[\d<]+[\s–-]*[\d]*\s*µm', '', taxon)
        taxon = re.sub(r'\s+var\.\s*\w+', '', taxon)
        taxon = re.sub(r'[\/_].*', '', taxon)
        taxon = re.sub(r'\s*(Formenkreis|FK|Komplex)$', '', taxon, flags=re.IGNORECASE)
        taxon = re.sub(r'\s+', ' ', taxon).strip()
        return taxon

    # Date cleaning
    if "date" in all_plankton_df.columns:
        all_plankton_df["date"] = pd.to_datetime(
            all_plankton_df["date"],
            errors="coerce",
        )
        all_plankton_df["date"] = (
            all_plankton_df["date"]
            .dt.to_period("M")
            .dt.to_timestamp()
        )
        all_plankton_df.dropna(subset=["date"], inplace=True)

    # Numeric cleaning
    numeric_cols = ["biovolume_cm3_m3", "cell_count_ml"]

    for col in numeric_cols:
        if col in all_plankton_df.columns:
            all_plankton_df[col] = (
                pd.to_numeric(all_plankton_df[col], errors="coerce")
                .fillna(0)
            )

    # Taxon cleaning
    if "taxon" in all_plankton_df.columns:
        all_plankton_df["taxon"] = all_plankton_df["taxon"].apply(
            clean_taxon_name
        )

    # Split into samples
    sample_dfs = {}

    for keys, group_df in all_plankton_df.groupby(
        identifier_keys,
        dropna=False,
    ):
        lake, mnum, date = keys

        date_str = (
            date.strftime("%Y%m%d")
            if pd.notna(date)
            else "nodate"
        )

        safe_name = f"{lake}_{mnum}_{date_str}"
        safe_name = re.sub(r"[^0-9a-zA-Z_]+", "_", safe_name).lower()

        sample_dfs[safe_name] = group_df

    return all_plankton_df, sample_dfs, stammdaten_plankton_df