# Intelalgen Repo

An end-to-end machine learning pipeline for modeling, predicting, and interpreting phytoplankton/algae occurrence and dynamics using environmental chemistry and community features.

---

## Project Structure

```text
Intelalgen_repo/

├── analysis_and_explainability/
│   ├── dependence_plots/           # SHAP dependence plots grouped by metric (e.g., by_pr_auc)
│   │   └── by_pr_auc/              # Organism-specific plots (Anabaena, Ceratium, Oocystis, etc.)
│   ├── images/                     # Output visualisations (e.g., overall_cv_distribution_final.png)
│   ├── shap_results/               # SHAP output metrics grouped by_pr_auc and by_roc_auc
│   ├── summary_tables/             # Performance evaluation summaries (e.g., final_performance_comparison.csv)
│   ├── tables/                     # Detailed tabular outputs
│   ├── xgboost_importance_results/ # Feature importance outputs grouped by metric
│   └── shap and xgboost.ipynb      # Jupyter notebook for explainability and feature evaluation
│
└── modelling/
    ├── config/                     # Configuration settings, keys, and column mappings
    │   ├── config.py
    │   └── keys_and_column_names.py
    │
    ├── data_files/                 # Raw input data files (UFZ lakes phytoplankton & chemistry datasets)
    │
    ├── data_loading/               # Data ingestion scripts
    │   ├── chemie_data.py
    │   └── plankton_data.py
    │
    ├── feature_set_creators/       # Modules for feature engineering & transformations
    │   ├── co_occurence.py
    │   ├── community_freq.py
    │   ├── median_impute.py
    │   ├── mixer_merger.py
    │   ├── month_dummies.py
    │   └── prevalence.py
    │
    ├── kneedle_preprocess/         # Knee-point detection and preprocessing scripts
    │   └── kneedle_filters_and_process.py
    │
    ├── model_results/              # Output metrics and trained XGBoost model runs
    │   ├── best_models_summary/    # Summarized best models
    │   └── intel_algen_xgboost/    # Output sets for specific feature combinations
    │
    ├── helper_functions.py         # Utility functions for pipeline operations
    ├── main.py                     # Main execution entry point for training and evaluation
    └── model_files.py              # Model definitions and persistence routines
```

---

## Key Features & Pipeline Steps

### 1. Data Loading & Ingestion

The pipeline processes two primary data sources:

* **Phytoplankton data** — loaded and prepared by `data_loading/plankton_data.py`.
* **Water chemistry data** — loaded and prepared by `data_loading/chemie_data.py`.

The two datasets are subsequently aligned using the configured identifier keys.

### 2. Phytoplankton Data Preparation

`data_loading/plankton_data.py` provides `load_and_prepare_phytoplankton()`.

The function:

* Loads all sheets from the phytoplankton Excel workbook.
* Separates the `Stammdaten` metadata sheet when present.
* Concatenates the remaining sheets.
* Renames columns using the configured phytoplankton column translation.
* Converts dates to monthly timestamps.
* Removes rows with invalid or missing dates.
* Converts `biovolume_cm3_m3` and `cell_count_ml` to numeric values.
* Replaces missing numeric plankton values with `0`.
* Cleans taxon names by removing size ranges, varieties, suffixes such as `Formenkreis`, `FK`, and `Komplex`, and other trailing classification information.
* Groups observations into individual samples using the configured `identifier_keys`.

The function returns:

```text
all_plankton_df
sample_dfs
stammdaten_plankton_df
```

### 3. Water Chemistry Data Preparation

`data_loading/chemie_data.py` provides `load_and_prepare_chemistry()`.

The function:

* Loads all Excel sheets from the chemistry workbook.
* Separates metadata sheets such as `Stammdaten` and `ST+Z_epi`.
* Separates the `Daten_Seen_2005-2006` sheet.
* Concatenates the remaining chemistry sheets.
* Applies the configured column translations.
* Standardizes column names to lowercase underscore-separated names.
* Normalizes `water_body` names.
* Converts dates to monthly timestamps.
* Extracts numeric measurement numbers.
* Converts chemistry measurements to numeric values.
* Handles values reported using `<` notation by converting them to half of the reported detection limit.
* Treats values such as `n.b.`, `n.b`, `nb`, and `n_b` as missing values.
* Combines alternative total-phosphorus measurements into `total_phosphorus_combined_mg_per_l`.
* Combines alternative dissolved-silica measurements into `dissolved_silica_combined_mg_per_l`.
* Removes selected chemistry columns that are not retained.
* Aggregates chemistry measurements to the configured identifier level.

The function returns:

```text
chemie_wide_df
stammdaten_df
st_z_epi_df
```

### 4. Genus Processing and Kneedle-Based Thresholding

The `kneedle_preprocess/kneedle_filters_and_process.py` module processes phytoplankton taxa, calculates biovolume shares, applies Kneedle-based thresholds, creates genus-level presence and biovolume matrices, and filters rare genera.

#### Genus Processing

`process_genus()` extracts the genus name from the `taxon` column using the `extract_genus()` helper function.

After genus extraction, observations whose genus name contains `"unbestimmte"` are removed.

#### Biovolume Share Calculation

`compute_biovolume_shares()` calculates the total biovolume for each group defined by the configured `identifier_keys`.

For each observation, its relative contribution to the total biovolume is calculated as:

```text
share = biovolume_cm3_m3 / total_biovolume
```

A small constant (`1e-12`) is added to the denominator to avoid division by zero.

#### Kneedle-Based Thresholding

`find_knee_threshold()` sorts the biovolume shares within each group in descending order and applies the Kneedle algorithm using a convex, decreasing curve.

If Kneedle identifies a valid knee, the corresponding biovolume share is used as the threshold.

`apply_knee_threshold()` then converts this relative threshold into an absolute biovolume threshold:

```text
knee_threshold_biovolume = knee_threshold_share × total_biovolume
```

Each observation is then evaluated against this threshold.

If:

```text
biovolume_cm3_m3 >= knee_threshold_biovolume
```

the observation is assigned:

```text
above_threshold = 1
bv_thresholded = original biovolume
```

Otherwise:

```text
above_threshold = 0
bv_thresholded = 0
```

The calculated `knee_threshold_share` and `knee_threshold_biovolume` are retained for each group.

If Kneedle cannot identify a valid knee, the code does **not remove or threshold the observations**. Instead, all observations in that group are retained:

```text
above_threshold = 1
bv_thresholded = original biovolume
```

In this case, both threshold columns are set to `NaN`.

#### Presence and Biovolume Matrices

`pivot_presence_biovolume()` converts the thresholded long-format phytoplankton data into genus-level matrices.

It creates:

* **Presence matrix (`presence_df`)** — contains binary `above_threshold` values for each genus and identifier group. When multiple observations of the same genus occur within a group, the maximum value is used.
* **Biovolume matrix (`biovolume_df`)** — contains the summed `bv_thresholded` values for each genus and identifier group.
* **Threshold information (`presence_with_thresholds_df`)** — combines the presence matrix with the corresponding Kneedle threshold information.

#### Rare Genus Filtering

`filter_rare_genera()` performs a second, separate Kneedle-based procedure to identify rare genera.

First, the number of occurrences of each genus is calculated from the presence matrix:

```text
genus occurrence = sum of binary presence values across groups
```

The occurrence counts are sorted in descending order and Kneedle is applied to this distribution.

If a valid knee is detected, the occurrence count at the knee is used as `threshold_occurrence`.

Genera with:

```text
genus occurrence < threshold_occurrence
```

are considered rare and are removed from both the presence and biovolume matrices.

Finally, the remaining genus columns are returned as `genus_columns` and are used in the downstream feature engineering and modeling pipeline.


### 5. Data Integration

`helper_functions.py` provides `merge_with_chemistry()` to combine phytoplankton presence data with the prepared chemistry dataset.

Before merging:

* Water-body names are normalized.
* Dates are converted to monthly timestamps.
* Measurement identifiers are converted to nullable integer types.

The datasets are then merged using the configured `identifier_keys`.

After merging, numeric columns are classified into genus and chemistry features. Rows where all chemistry measurements are missing are removed, and duplicate rows are dropped.

### 6. Feature Engineering

The `feature_set_creators/` modules generate ecological, temporal, and imputation-based features.

#### Median Imputation

`median_impute.py` provides `fe_impute_median()`.

Missing numeric features are filled using:

1. **Site-level medians** calculated from the training data.
2. **Global training medians** as a fallback.

Genus and identifier columns are excluded from imputation.

This ensures that imputation statistics applied to the test set originate from the training data.

#### Monthly Features

`month_dummies.py` provides `fe_month_dummies()`.

The month is extracted from the observation date and converted into one-hot encoded features such as:

```text
month_1
month_2
...
month_12
```

Training and test dummy columns are aligned so both datasets have the same feature structure.

Month features are always included in generated feature sets.

#### Local and Global Prevalence

`prevalence.py` provides target-genus prevalence features.

**Local prevalence (`LP_<target_genus>`)**

Calculates the average target-genus value within the same `measurement_number_bio_chem` group.

Training values use leave-one-out calculations, while test values are calculated using corresponding training-group statistics. If no matching group is available, the global training mean is used.

**Global prevalence (`GP_<target_genus>`)**

Calculates the overall target-genus mean.

Training observations use leave-one-out values, while test observations receive the global training mean.

#### Local and Global Community Frequency

`community_freq.py` describes the overall frequency of non-target genera in the community.

**Local community frequency (`LCom`)**

For each sample, the proportion of non-target genera that are present is calculated. This ratio is then averaged within the corresponding `measurement_number_bio_chem` group.

Training observations use leave-one-out group statistics. Test observations use statistics calculated from the training data.

**Global community frequency (`GCom`)**

Calculates the global proportion of presences across all non-target genera and samples.

Training values use a leave-one-out calculation, while test observations receive the global training-set value.

#### Local and Global Target Co-occurrence

`co_occurence.py` calculates the relationship between the target genus and every other available genus.

**Local target co-occurrence (`LCoO_<other_genus>_with_<target_genus>`)**

Calculates the product between the target genus and another genus and aggregates this relationship within the `measurement_number_bio_chem` group.

Training observations use leave-one-out group statistics. Test observations use corresponding statistics derived from the training data.

**Global target co-occurrence (`GCoO_<other_genus>_with_<target_genus>`)**

Calculates the global mean product between the target genus and each other genus.

Training values use leave-one-out calculations, while test observations receive the global training-set value.

### 7. Feature Set Generation

`mixer_merger.py` coordinates the feature-engineering workflow and generates combinations of feature sets.

`apply_feature_mix()` applies feature transformations in the following order:

1. Median imputation.
2. Post-imputation missing-value check.
3. Month dummy generation.
4. Selected ecological feature functions.

The module also defines `fe_keep_chem()`, which acts as a no-op function to retain the original chemistry features.

The feature combinations are generated from three local/global feature pairs:

```text
Prevalence:
    Local  → lp
    Global → gp

Community frequency:
    Local  → lcom
    Global → gcom

Target co-occurrence:
    Local  → lcoo
    Global → gcoo
```

The baseline feature set is:

```text
mth_chem
```

Additional sets combine local features and local/global features in different configurations.

Feature sets are automatically named using:

```text
set_<number>_<feature_configuration>
```

For example:

```text
set_1_mth_chem
set_2_mth_chem_lp
set_3_mth_chem_lcom
...
```

This allows the modeling pipeline to systematically compare different feature configurations.

### 8. Helper Functions

`helper_functions.py` contains shared utilities used throughout the pipeline.

#### Genus Detection

`detect_genus_columns()` identifies numeric binary columns containing only `0` and `1`, while excluding:

* Identifier columns.
* `LP_` and `GP_` prevalence features.
* `LCoO_` and `GCoO_` co-occurrence features.
* `LCom` and `GCom` community-frequency features.

This provides a consistent method for distinguishing genus features from engineered and chemistry features.

#### Genus Extraction

`extract_genus()` extracts the first component of a cleaned taxon name as the genus name.

#### Model Evaluation

`evaluate_model()` calculates:

* ROC-AUC
* PR-AUC

If the target contains fewer than two unique classes, both metrics are returned as `NaN`.

#### Cross-Validation Summary

`summarize_cv_results()` converts cross-validation results into a DataFrame and calculates descriptive statistics using `DataFrame.describe()`.

#### Key Normalization

`normalize_keys()` standardizes water-body names and converts dates to monthly timestamps.

#### Genus List Retrieval

`get_genus_list()` returns the detected binary genus columns using `detect_genus_columns()`.

### 9. XGBoost Modeling & Validation

The final modeling stage is orchestrated by `modelling/main.py`.

The pipeline passes the merged phytoplankton and chemistry dataset to:

```python
run_feature_set_training()
```

from `model_files.py`.

The modeling stage evaluates the generated feature sets using cross-validation and stores the resulting model performance and trained model outputs.

Performance is evaluated using:

* **PR-AUC**
* **ROC-AUC**

The best configurations are summarized separately according to these metrics.


## Model Explainability and SHAP Analysis

This component provides post-hoc explainability for the trained XGBoost models. It identifies the best-performing model for each genus, extracts model-based feature importance, computes SHAP values and SHAP interactions, and generates visual and tabular outputs for interpreting model predictions.

### Overview

The explainability workflow is applied separately to models selected according to:

* **ROC-AUC**
* **PR-AUC**

For each evaluation metric, the best-performing feature set for each genus is selected from the model evaluation results. The selected models and their corresponding data are then used for the subsequent feature-importance and SHAP analyses.

### 1. Best Model Selection

The detailed model evaluation results are collected and combined to identify the best-performing model for each genus.

For every genus, two model selections are considered:

* The model with the highest **mean ROC-AUC**
* The model with the highest **mean PR-AUC**

The selected model summaries are stored in:

```text
best_models_summary/
├── best_by_roc_auc.csv
└── best_by_pr_auc.csv
```

Each record identifies the selected model and its corresponding feature set.

### 2. XGBoost Feature Importance

For each selected XGBoost model, native XGBoost feature importance is extracted using the following importance measures:

* `weight`
* `gain`
* `cover`
* `total_gain`
* `total_cover`

The importance values can additionally be normalized to allow comparisons between models.

The results are organized separately according to the model-selection criterion:

```text
xgboost_importance_results/
├── by_roc_auc/
│   └── <Genus>/
│       └── xgboost_importance_roc_auc.csv
└── by_pr_auc/
    └── <Genus>/
        └── xgboost_importance_pr_auc.csv
```

These outputs provide a model-internal view of the features used by the XGBoost models and their relative importance.

### 3. SHAP Feature Importance

SHAP (`TreeExplainer`) is used to quantify the contribution of individual features to XGBoost model predictions.

For each selected genus model, global SHAP statistics are calculated, including:

* **Mean absolute SHAP** — overall magnitude of feature contribution
* **Mean signed SHAP** — average direction of contribution
* **Mean positive SHAP** — average contribution toward higher model output
* **Mean negative SHAP** — average contribution toward lower model output

Both raw and normalized SHAP measures can be retained, with features ranked according to mean absolute SHAP values.

The resulting outputs are organized as:

```text
shap_results/
├── by_roc_auc/
│   └── <Genus>/
│       ├── shap_summary_roc_auc.csv
│       └── shap_interactions_roc_auc.csv
└── by_pr_auc/
    └── <Genus>/
        ├── shap_summary_pr_auc.csv
        └── shap_interactions_pr_auc.csv
```

### 4. SHAP Interaction Analysis

SHAP interaction values are used to investigate relationships between predictors.

For the selected important features, pairwise SHAP interaction strengths are calculated using the mean absolute interaction value:

```text
mean(|SHAP interaction value|)
```

Feature pairs are then ranked according to their interaction strength. This allows the strongest pairwise relationships between predictors to be examined for each genus-specific model.

### 5. SHAP Dependence Plots

SHAP dependence plots are generated to examine the relationship between feature values and their corresponding SHAP contributions.

For each selected genus, important features identified through mean absolute SHAP are used to generate dependence plots. The analysis can be performed separately for the models selected according to ROC-AUC and PR-AUC.

The plots are organized as:

```text
dependence_plots/
└── by_<metric>/
    └── <Genus>/
        └── <feature>_dependence.png
```

An additional overlay analysis can be used for selected genera, where the top continuous features are plotted together after normalization of their feature values to a 0–100 scale.

### 6. SHAP Value Heatmap and Network Analysis

The exported SHAP values are further analyzed to investigate the relationship between feature values and their corresponding SHAP contributions.

For each genus, the SHAP value export contains one Excel file with one sheet per selected feature. Positive and negative SHAP values are retained together in the same sheet, allowing the full range of model contributions to be analyzed consistently.

Spearman Correlation Analysis

For every feature–genus combination, a Spearman rank correlation is calculated between the original feature values and the absolute SHAP values:

Spearman ρ = corr(feature value, |SHAP value|)

The analysis is performed separately for every feature sheet. Only statistically significant relationships with:

p-value < 0.05

are retained for the subsequent analysis.

The resulting relationships are represented as directed feature–genus edges, where the correlation coefficient is used as the edge weight.

Feature Name Mapping

Feature names are mapped from their internal model representations to more interpretable names before constructing the final correlation matrix. This includes environmental variables, month indicators, and ecological community features such as LCom and GCom.

Clustered Heatmap

The significant Spearman correlations are converted into a feature × genus matrix. Missing relationships are assigned a correlation value of zero.

Hierarchical clustering is then applied to both dimensions of the matrix using Ward linkage:

Rows → feature clustering
Columns → genus clustering

The resulting clustered heatmap uses a diverging color scale centered at zero, allowing positive and negative associations to be visually distinguished.

The final figure is saved as:

clustered_heatmap_shap.png

This heatmap provides an overview of which environmental, temporal, and community-related variables show similar SHAP-response patterns across the modeled genera.

Network Analysis with Gephi

The same significant feature–genus relationships are also used to construct a network representation in Gephi.

Each significant correlation is represented as a weighted directed edge between a genus and a feature. The Spearman correlation coefficient (ρ) is retained as the edge weight, preserving both the strength and direction of the relationship.

The resulting network can be imported into Gephi for network visualization and analysis, enabling the identification of:

Highly connected features
Genera strongly associated with multiple features
Strong positive or negative relationships
Feature–genus network structure and clusters

The Gephi workflow therefore provides a complementary network-based representation of the SHAP-derived relationships, while the clustered heatmap provides a matrix-based view of the same statistical associations.

Explainability Analysis Workflow

The complete explainability workflow can therefore be summarized as:

Best XGBoost Models
        ↓
XGBoost Feature Importance
        ↓
SHAP Values
        ↓
Top Feature Selection
        ↓
SHAP Value Exports
        ↓
Spearman Correlation
        ↓
Significant Relationships (p < 0.05)
        ↓
      ┌───────────────┐
      ↓               ↓
Clustered         Gephi Network
Heatmap           Analysis

### Explainability Outputs

Overall, the explainability component provides several complementary views of the trained genus-specific XGBoost models:

1. **XGBoost feature importance** — model-internal importance measures.
2. **Global SHAP importance** — magnitude and direction of feature contributions.
3. **SHAP interactions** — pairwise relationships between important predictors.
4. **SHAP dependence plots** — relationship between feature values and SHAP contributions.
5. **Feature-range SHAP analysis** — changes in average SHAP contribution across feature-value ranges.
6. **SHAP value exports** — detailed sample-level SHAP values for downstream analysis.

Together, these outputs provide global and local perspectives on the factors contributing to the predictions of the genus-specific XGBoost models.

## Getting Started

```markdown
### Prerequisites

* Python 3.8+
* `requirements.txt` dependencies

### Installation

Clone the repository and install the required dependencies:

```bash
git clone <your-repository-url>
cd Intelalgen_repo
pip install -r requirements.txt
```

Place the required raw UFZ phytoplankton and chemistry datasets under:

```text
modelling/data_files/
```

The input file paths are configured through:

```text
modelling/config/config.py
```

### Configuration

The configuration modules are located in:

```text
modelling/config/
├── config.py
└── keys_and_column_names.py
```

`config.py` defines project paths, input file locations, and output settings.

`keys_and_column_names.py` contains the column translations, measurement columns, and identifier definitions used throughout the pipeline.

The configuration also supports HPC execution through the available CPU configuration.

---

## Running the Modeling Pipeline

Run the complete pipeline with:

```bash
python modelling/main.py
```

The command-line interface supports the following arguments:

| Argument       | Type  |         Default | Description                                       |
| -------------- | ----- | --------------: | ------------------------------------------------- |
| `--output_dir` | `str` | `BASE_SAVE_DIR` | Directory for models and logs                     |
| `--n_splits`   | `int` |             `5` | Number of cross-validation splits                 |
| `--cores`      | `int` |             `4` | Number of CPU cores for parallel genus processing |

Example:

```bash
python modelling/main.py \
    --output_dir ./results \
    --n_splits 5 \
    --cores 4
```

The output directory is created automatically if it does not exist.

---

## Running Explainability & Analysis

Open the analysis notebook with:

```bash
jupyter notebook "analysis_and_explainability/shap and xgboost.ipynb"
```

The notebook can be used to inspect SHAP-based explanations, XGBoost feature importance, performance comparisons, and generated visualizations.

---

## Configuration & Data Mapping

The configuration module (`modelling/config/`) centralizes the paths and dataset mappings required by the pipeline.

### Configuration Parameters (`config.py`)

Important configuration values include:

* `PROJECT_ROOT` / `DATA_DIR` — project and raw-data locations.
* `FILE_PATH_PHYTOPLANKTON` — path to the phytoplankton input dataset.
* `FILE_PATH_CHEMIE` — path to the chemistry input dataset.
* `BASE_SAVE_DIR` — default output directory for model results.
* `N_CORES` — number of processing cores available to the pipeline.

### Data Schema & Feature Mappings (`keys_and_column_names.py`)

The data mapping configuration defines the correspondence between the original UFZ dataset columns and the standardized feature names used by the pipeline.

Primary identifiers are used to align and aggregate biological and chemical observations.

Important configured groups include:

* **Identifier keys** — keys used to align, group, and aggregate observations.
* **Site keys** — site-level grouping information.
* **Measurement columns** — physical, chemical, nutrient, and pigment measurements.
* **Genus columns** — binary phytoplankton occurrence columns used as targets and community features.
* **Column translations** — mappings from the original dataset column names to the standardized internal schema.

---

## Pipeline Summary

The complete workflow is:

```text
UFZ Phytoplankton Data
          │
          ▼
   Phytoplankton Cleaning
          │
          ▼
      Genus Extraction
          │
          ▼
   Biovolume Share Calculation
          │
          ▼
     Kneedle Thresholding
          │
          ▼
 Presence / Biovolume Matrices
          │
          ▼
     Rare Genus Filtering
          │
          ├──────────────────┐
          │                  │
          ▼                  ▼
   Phytoplankton        Water Chemistry
          │                  │
          └────────┬─────────┘
                   ▼
           Data Integration
                   │
                   ▼
         Feature Engineering
                   │
                   ▼
      Multiple Feature Sets
                   │
                   ▼
        XGBoost Modeling
                   │
                   ▼
      Cross-Validation Metrics
                   │
          ┌────────┴────────┐
          ▼                 ▼
       PR-AUC            ROC-AUC
          │                 │
          └────────┬────────┘
                   ▼
        Model Results & SHAP
                   │
                   ▼
      Explainability Analysis
```
