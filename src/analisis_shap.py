from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

def _normalize_multiclass_shap_values(
    shap_values_raw,
    n_samples,
    n_features,
    n_classes
):
    """
    Normaliza la salida de shap.TreeExplainer a una lista:
        shap_values_by_class[class_idx] = array de shape (n_samples, n_features)

    SHAP puede devolver diferentes formatos según versión/modelo:
    - list de arrays, uno por clase
    - array 3D con shape (n_samples, n_features, n_classes)
    - array 3D con shape (n_classes, n_samples, n_features)
    - array 2D en clasificación binaria
    """

    if isinstance(shap_values_raw, list):
        return shap_values_raw

    shap_values_raw = np.asarray(shap_values_raw)

    # Caso binario o regresión: array 2D
    if shap_values_raw.ndim == 2:
        return [shap_values_raw]

    # Caso: (n_samples, n_features, n_classes)
    if shap_values_raw.ndim == 3:
        if shap_values_raw.shape[0] == n_samples and shap_values_raw.shape[1] == n_features:
            return [
                shap_values_raw[:, :, class_idx]
                for class_idx in range(shap_values_raw.shape[2])
            ]

        # Caso: (n_classes, n_samples, n_features)
        if shap_values_raw.shape[0] == n_classes and shap_values_raw.shape[1] == n_samples:
            return [
                shap_values_raw[class_idx, :, :]
                for class_idx in range(shap_values_raw.shape[0])
            ]

    raise ValueError(
        "Formato de shap_values no reconocido. "
        f"Shape recibido: {shap_values_raw.shape}"
    )


def _mean_abs_shap_importance(shap_values, feature_names):
    """
    Calcula importancia media absoluta de SHAP por feature.
    """
    mean_abs_values = np.mean(np.abs(shap_values), axis=0)

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_values
    })

    importance_df = importance_df.sort_values(
        by="mean_abs_shap",
        ascending=False
    ).reset_index(drop=True)

    return importance_df


def _save_global_shap_importance(
    shap_values_by_class,
    feature_names,
    class_names,
    output_dir,
    prefix
):
    """
    Guarda importancia SHAP global agregada en multiclase.

    Para cada feature:
    - calcula mean(|SHAP|) por clase
    - calcula mean(|SHAP|) global promediando clases
    """

    class_importance_dfs = []

    for class_idx, class_name in enumerate(class_names):
        shap_class = shap_values_by_class[class_idx]

        df_class = _mean_abs_shap_importance(
            shap_values=shap_class,
            feature_names=feature_names
        )

        df_class = df_class.rename(
            columns={"mean_abs_shap": f"mean_abs_shap_{class_name}"}
        )

        class_importance_dfs.append(df_class)

    # Unir por feature
    importance_df = class_importance_dfs[0]

    for df_class in class_importance_dfs[1:]:
        importance_df = importance_df.merge(
            df_class,
            on="feature",
            how="outer"
        )

    shap_cols = [
        col for col in importance_df.columns
        if col.startswith("mean_abs_shap_")
    ]

    importance_df["mean_abs_shap_global"] = importance_df[shap_cols].mean(axis=1)

    importance_df = importance_df.sort_values(
        by="mean_abs_shap_global",
        ascending=False
    ).reset_index(drop=True)

    importance_df.to_csv(
        output_dir / f"{prefix}_shap_importance.csv",
        index=False
    )

    return importance_df


def _plot_global_multiclass_summary(
    shap_values_by_class,
    X_shap,
    class_names,
    output_dir,
    filename,
    max_display=20
):
    """
    Genera gráfico SHAP global para clasificación multiclase.

    En SHAP, para multiclase se puede pasar una lista de arrays,
    uno por clase.
    """

    plt.figure()

    shap.summary_plot(
        shap_values_by_class,
        X_shap,
        class_names=class_names,
        show=False,
        max_display=max_display
    )

    plt.tight_layout()
    plt.savefig(
        output_dir / filename,
        dpi=200,
        bbox_inches="tight"
    )
    plt.close()


def _safe_filename(name):
    """
    Convierte nombres de clases en nombres seguros para archivos.
    """
    name = str(name)
    name = name.replace("/", "_")
    name = name.replace("\\", "_")
    name = name.replace(" ", "_")
    name = name.replace(":", "_")
    return name

def run_tree_shap_analysis(
    model_pipeline,
    X,
    y_true,
    y_pred,
    feature_names,
    class_names=None,
    output_dir="shap_outputs",
    model_step_name=None,
    max_display=20,
):
    """
    Realiza análisis SHAP para modelos de árboles dentro de un Pipeline.

    Compatible con:
    - RandomForestClassifier
    - LGBMClassifier
    - Otros modelos tree-based compatibles con shap.TreeExplainer

    Parameters
    ----------
    model_pipeline : sklearn Pipeline o modelo entrenado
        Pipeline entrenado que contiene al menos:
        - un imputador opcional
        - un modelo tree-based final

    X : pd.DataFrame
        Datos de entrada originales, antes de imputación.

    y_true : array-like
        Etiquetas reales.

    y_pred : array-like
        Predicciones del modelo.

    feature_names : list[str]
        Lista de nombres de features usadas por el modelo.

    class_names : list[str], optional
        Nombres de clases en el orden usado por el modelo.
        Ejemplo: ["NREM", "REM", "Wake"]

    output_dir : str or Path
        Carpeta donde guardar gráficos y CSVs.

    model_step_name : str, optional
        Nombre del paso del modelo dentro del Pipeline.
        Si es None, se toma el último paso del Pipeline.

    max_display : int
        Número máximo de features a mostrar en los gráficos SHAP.

    Returns
    -------
    shap_results : dict
        Diccionario con valores SHAP, datos imputados e importancias.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X = X.copy()
    X = X[feature_names]

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # ------------------------------------------------------------
    # 1. Extraer modelo final e imputador si existen
    # ------------------------------------------------------------

    if hasattr(model_pipeline, "named_steps"):
        named_steps = model_pipeline.named_steps

        if model_step_name is None:
            model_step_name = list(named_steps.keys())[-1]

        model = named_steps[model_step_name]

        # Si hay imputador en el pipeline, transformamos X antes de SHAP
        if "imputer" in named_steps:
            X_transformed = named_steps["imputer"].transform(X)

            X_shap = pd.DataFrame(
                X_transformed,
                columns=feature_names,
                index=X.index
            )
        else:
            X_shap = X.copy()

    else:
        model = model_pipeline
        X_shap = X.copy()

    # ------------------------------------------------------------
    # 2. Determinar nombres de clases
    # ------------------------------------------------------------

    if class_names is None:
        if hasattr(model, "classes_"):
            class_names = list(model.classes_)
        else:
            class_names = sorted(np.unique(y_true).tolist())

    class_names = list(class_names)

    print("[INFO] Clases usadas en SHAP:")
    for i, c in enumerate(class_names):
        print(f"  Clase {i}: {c}")

    # ------------------------------------------------------------
    # 3. Calcular SHAP
    # ------------------------------------------------------------

    print("[INFO] Calculando valores SHAP...")

    explainer = shap.TreeExplainer(model)
    shap_values_raw = explainer.shap_values(X_shap)

    shap_values_by_class = _normalize_multiclass_shap_values(
        shap_values_raw=shap_values_raw,
        n_samples=X_shap.shape[0],
        n_features=X_shap.shape[1],
        n_classes=len(class_names)
    )

    # ------------------------------------------------------------
    # 4. Máscaras de aciertos y errores
    # ------------------------------------------------------------

    correct_mask = y_true == y_pred
    error_mask = y_true != y_pred

    print(f"[INFO] Muestras totales: {len(y_true)}")
    print(f"[INFO] Aciertos: {correct_mask.sum()}")
    print(f"[INFO] Errores: {error_mask.sum()}")

    # ------------------------------------------------------------
    # 5. Análisis global
    # ------------------------------------------------------------

    global_importance_df = _save_global_shap_importance(
        shap_values_by_class=shap_values_by_class,
        feature_names=feature_names,
        class_names=class_names,
        output_dir=output_dir,
        prefix="global_all_samples"
    )

    _plot_global_multiclass_summary(
        shap_values_by_class=shap_values_by_class,
        X_shap=X_shap,
        class_names=class_names,
        output_dir=output_dir,
        filename="shap_summary_global_all_samples.png",
        max_display=max_display
    )

    # ------------------------------------------------------------
    # 6. SHAP por clase
    # ------------------------------------------------------------

    per_class_importances = {}

    for class_idx, class_name in enumerate(class_names):
        safe_class_name = _safe_filename(str(class_name))

        shap_class = shap_values_by_class[class_idx]

        class_importance_df = _mean_abs_shap_importance(
            shap_values=shap_class,
            feature_names=feature_names
        )

        class_importance_df.to_csv(
            output_dir / f"shap_importance_class_{safe_class_name}.csv",
            index=False
        )

        per_class_importances[class_name] = class_importance_df

        plt.figure()
        shap.summary_plot(
            shap_class,
            X_shap,
            feature_names=feature_names,
            show=False,
            max_display=max_display
        )
        plt.title(f"SHAP summary - clase {class_name}")
        plt.tight_layout()
        plt.savefig(
            output_dir / f"shap_summary_class_{safe_class_name}.png",
            dpi=200,
            bbox_inches="tight"
        )
        plt.close()

    # ------------------------------------------------------------
    # 7. SHAP en aciertos
    # ------------------------------------------------------------

    correct_importance_df = None

    if correct_mask.sum() > 0:
        shap_correct = [
            shap_values_by_class[i][correct_mask]
            for i in range(len(class_names))
        ]

        X_correct = X_shap.loc[correct_mask]

        correct_importance_df = _save_global_shap_importance(
            shap_values_by_class=shap_correct,
            feature_names=feature_names,
            class_names=class_names,
            output_dir=output_dir,
            prefix="global_correct_predictions"
        )

        _plot_global_multiclass_summary(
            shap_values_by_class=shap_correct,
            X_shap=X_correct,
            class_names=class_names,
            output_dir=output_dir,
            filename="shap_summary_correct_predictions.png",
            max_display=max_display
        )
    else:
        print("[AVISO] No hay aciertos. Se omite SHAP de aciertos.")

    # ------------------------------------------------------------
    # 8. SHAP en errores
    # ------------------------------------------------------------

    error_importance_df = None

    if error_mask.sum() > 0:
        shap_errors = [
            shap_values_by_class[i][error_mask]
            for i in range(len(class_names))
        ]

        X_errors = X_shap.loc[error_mask]

        error_importance_df = _save_global_shap_importance(
            shap_values_by_class=shap_errors,
            feature_names=feature_names,
            class_names=class_names,
            output_dir=output_dir,
            prefix="global_error_predictions"
        )

        _plot_global_multiclass_summary(
            shap_values_by_class=shap_errors,
            X_shap=X_errors,
            class_names=class_names,
            output_dir=output_dir,
            filename="shap_summary_error_predictions.png",
            max_display=max_display
        )
    else:
        print("[AVISO] No hay errores. Se omite SHAP de errores.")

    # ------------------------------------------------------------
    # 9. Guardar resumen por muestra
    # ------------------------------------------------------------

    prediction_summary = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "correct": correct_mask
    }, index=X.index)

    prediction_summary.to_csv(
        output_dir / "shap_prediction_correct_error_summary.csv",
        index=True
    )

    print(f"[OK] Análisis SHAP guardado en: {output_dir}")

    shap_results = {
        "explainer": explainer,
        "X_shap": X_shap,
        "shap_values_by_class": shap_values_by_class,
        "global_importance_df": global_importance_df,
        "per_class_importances": per_class_importances,
        "correct_importance_df": correct_importance_df,
        "error_importance_df": error_importance_df,
        "prediction_summary": prediction_summary,
    }

    return shap_results