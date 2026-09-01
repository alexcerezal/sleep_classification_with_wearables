from __future__ import annotations

import numpy as np
import pandas as pd
import analisis_shap
import joblib
import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt

from src_sint.config import (
    CLASS_NAMES,
    DATA_PATH,
    FEATURE_COLUMNS,
    MODEL_PATH,
    RANDOM_STATE,
    SHAP_CLASS_TEMPLATE,
    SHAP_GLOBAL_PATH,
    SHAP_SAMPLE_SIZE,
    RESULTS_DIR,
    ensure_directories,
)
from src_sint.train_random_forest import load_dataset, split_dataset


def normalize_shap_values(shap_values: object, class_names: list[str]) -> dict[str, np.ndarray]:
    if isinstance(shap_values, list):
        return {class_names[idx]: np.asarray(values) for idx, values in enumerate(shap_values)}

    values = np.asarray(shap_values)
    if values.ndim != 3:
        raise ValueError(f"Unsupported SHAP output shape: {values.shape}")

    if values.shape[1] == len(FEATURE_COLUMNS) and values.shape[2] == len(class_names):
        return {class_names[idx]: values[:, :, idx] for idx in range(len(class_names))}

    if values.shape[1] == len(class_names) and values.shape[2] == len(FEATURE_COLUMNS):
        return {class_names[idx]: values[:, idx, :] for idx in range(len(class_names))}

    raise ValueError(f"Unsupported SHAP output shape: {values.shape}")


def save_global_importance(shap_by_class: dict[str, np.ndarray], X_sample: pd.DataFrame) -> None:
    stacked = np.stack([np.abs(values) for values in shap_by_class.values()], axis=0)
    global_mean_abs = stacked.mean(axis=(0, 1))
    importance_df = pd.DataFrame(
        {"feature": X_sample.columns, "mean_abs_shap": global_mean_abs}
    ).sort_values("mean_abs_shap", ascending=True)

    figure, axis = plt.subplots(figsize=(8, 8))
    axis.barh(importance_df["feature"], importance_df["mean_abs_shap"], color="#4C78A8")
    axis.set_xlabel("Mean |SHAP value| across classes")
    axis.set_title("Global SHAP importance")
    figure.tight_layout()
    figure.savefig(SHAP_GLOBAL_PATH, dpi=200)
    plt.close(figure)


def save_class_plots(shap_by_class: dict[str, np.ndarray], X_sample: pd.DataFrame) -> None:
    for class_name, class_values in shap_by_class.items():
        plt.figure(figsize=(10, 6))
        analisis_shap.summary_plot(class_values, X_sample, show=False)
        plt.title(f"SHAP summary for {class_name}")
        plt.tight_layout()
        output_path = RESULTS_DIR / SHAP_CLASS_TEMPLATE.format(class_name=class_name.lower())
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close()


def main() -> None:
    ensure_directories()
    payload = joblib.load(MODEL_PATH)
    model = payload["model"]

    df = load_dataset(DATA_PATH)
    _, X_test, _, _ = split_dataset(df)
    X_sample = X_test.sample(n=min(SHAP_SAMPLE_SIZE, len(X_test)), random_state=RANDOM_STATE)

    explainer = analisis_shap.TreeExplainer(model)
    raw_shap_values = explainer.shap_values(X_sample)
    shap_by_class = normalize_shap_values(raw_shap_values, CLASS_NAMES)

    save_global_importance(shap_by_class, X_sample)
    save_class_plots(shap_by_class, X_sample)

    print(f"Saved SHAP global plot to {SHAP_GLOBAL_PATH}")
    for class_name in CLASS_NAMES:
        print(
            f"Saved SHAP class plot to "
            f"{RESULTS_DIR / SHAP_CLASS_TEMPLATE.format(class_name=class_name.lower())}"
        )


if __name__ == "__main__":
    main()
