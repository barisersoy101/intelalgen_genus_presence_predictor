import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data_files")

FILE_PATH_PHYTOPLANKTON = os.path.join(DATA_DIR, "Datenanfrage_UFZ_Seen_Phytoplankton-01-08-2022.xlsx")
FILE_PATH_CHEMIE = os.path.join(DATA_DIR, "Datenanfrage_UFZ_Seen-Chemie_01-08-2022.xlsx")

# --- MAKE SURE THIS IS DEFINED ---
BASE_SAVE_DIR = os.path.join(PROJECT_ROOT, "intel_algen_xgboost_results")
os.makedirs(BASE_SAVE_DIR, exist_ok=True)

N_CORES = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4))