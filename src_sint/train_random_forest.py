from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupShuffleSplit

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from src_sint.config import (
    CONFUSION_MATRIX_PATH,
    DATA_PATH,
    FEATURE_COLUMNS,
    METRICS_PATH,
    MODEL_PATH,
    PERMUTATION_IMPORTANCE_PATH,
    RANDOM_STATE,
    REPORT_PATH,
    TEST_SIZE,
    ensure_directories,
)


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run src/generate_synthetic_data.py first."
        )
    return pd.read_csv(path)


def split_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = df[FEATURE_COLUMNS]
    y = df["label"]
    groups = df["subject_id"]

    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]


def build_model() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def save_confusion_matrix(y_true: pd.Series, y_pred: pd.Series) -> None:
    figure, axis = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(y_true, y_pred, labels=["Wake", "NREM", "REM"]),
        display_labels=["Wake", "NREM", "REM"],
    ).plot(ax=axis, cmap="Blues", colorbar=False)
    axis.set_title("Random Forest confusion matrix")
    figure.tight_layout()
    figure.savefig(CONFUSION_MATRIX_PATH, dpi=200)
    plt.close(figure)


def save_permutation_importance(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=10,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    importance_df = pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    importance_df.to_csv(PERMUTATION_IMPORTANCE_PATH, index=False)
    return importance_df


def main() -> None:
    ensure_directories()
    df = load_dataset()
    X_train, X_test, y_train, y_test = split_dataset(df)

    model = build_model()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "macro_f1": f1_score(y_test, y_pred, average="macro"),
    }

    report_text = classification_report(y_test, y_pred, digits=4)
    REPORT_PATH.write_text(report_text, encoding="utf-8")
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_confusion_matrix(y_test, y_pred)
    importance_df = save_permutation_importance(model, X_test, y_test)

    joblib.dump(
        {
            "model": model,
            "feature_columns": FEATURE_COLUMNS,
            "classes": model.classes_.tolist(),
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
        },
        MODEL_PATH,
    )

    print(f"Model saved to {MODEL_PATH}")
    print("Metrics:")
    for metric_name, value in metrics.items():
        print(f"  - {metric_name}: {value:.4f}")
    print("Top permutation importance features:")
    print(importance_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
