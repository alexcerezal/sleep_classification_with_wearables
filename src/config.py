from pathlib import Path

RANDOM_STATE = 42
N_SUBJECTS = 30
EPOCHS_PER_SUBJECT = 600
TEST_SIZE = 0.2
SHAP_SAMPLE_SIZE = 1000

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"

DATA_PATH = DATA_DIR / "synthetic_sleep_features.csv"
MODEL_PATH = MODELS_DIR / "random_forest_sleep.pkl"
REPORT_PATH = RESULTS_DIR / "classification_report.txt"
METRICS_PATH = RESULTS_DIR / "classification_metrics.json"
CONFUSION_MATRIX_PATH = RESULTS_DIR / "confusion_matrix.png"
PERMUTATION_IMPORTANCE_PATH = RESULTS_DIR / "permutation_importance.csv"
SHAP_GLOBAL_PATH = RESULTS_DIR / "shap_summary_global.png"
SHAP_CLASS_TEMPLATE = "shap_summary_{class_name}.png"

CLASS_NAMES = ["Wake", "NREM", "REM"]

FEATURE_COLUMNS = [
    "acc_mean",
    "acc_std",
    "acc_energy",
    "acc_p95",
    "acc_peaks",
    "hr_mean",
    "hr_std",
    "hr_min",
    "hr_max",
    "ibi_mean",
    "ibi_std",
    "rmssd",
    "sdnn",
    "bvp_mean",
    "bvp_std",
    "bvp_amp",
    "bvp_energy",
    "temp_mean",
    "temp_std",
    "temp_slope",
    "eda_mean",
    "eda_std",
    "eda_peaks",
    "time_from_sleep_start",
    "epoch_norm",
]


def ensure_directories() -> None:
    for directory in (DATA_DIR, MODELS_DIR, RESULTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
