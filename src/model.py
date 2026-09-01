import os

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, GroupShuffleSplit, GridSearchCV, GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline 
from imblearn.pipeline import Pipeline 
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef, roc_auc_score
from analisis_shap import run_tree_shap_analysis

INPUT_DIR = r"C:\Proyectos_compartidos\TFG\sleep_classification_with_wearables\data\dreamt_epoch_features2.csv"
RESULTS_DIR = r"C:\Proyectos_compartidos\TFG\sleep_classification_with_wearables\results\definitivo"

#PARA OBTENER RESULTADOS REPRODUCIBLES
RANDOM_STATE = 100

def remove_multicollinear_features(
    df,
    features,
    subject_col="subject_id",
    threshold=0.90,
):
    """
    Elimina features redundantes por multicolinealidad.

    Inspirado en el procedimiento de DREAMT_FE:
    - calcular correlación en el dataframe proporcionado
    - usar triángulo superior de la matriz
    - eliminar columnas con correlación absoluta > threshold
    - conservar, de cada pareja correlacionada, la feature que aparece antes
      en la lista original de features

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con una fila por época.
    features : list[str]
        Lista inicial de features candidatas. El orden importa:
        ante dos features correlacionadas, se conserva la primera.
    subject_col : str
        Nombre de la columna de sujeto. Se mantiene por compatibilidad,
        aunque esta versión usa directamente el dataframe recibido.
    threshold : float
        Umbral de correlación absoluta para eliminar redundancia.

    Returns
    -------
    selected_features : list[str]
        Features conservadas.
    dropped_features : list[str]
        Features eliminadas.
    corr_matrix_abs : pd.DataFrame
        Matriz de correlación absoluta usada para decidir.
    high_corr_pairs_df : pd.DataFrame
        Tabla con parejas de features con |correlación| > threshold,
        indicando cuál se conserva y cuál se elimina.
    """

    # ------------------------------------------------------------
    # 1. Quedarse solo con features existentes
    # ------------------------------------------------------------

    original_features = list(features)
    features = [f for f in features if f in df.columns]

    missing_features = [f for f in original_features if f not in df.columns]

    if missing_features:
        print("[AVISO] Features no encontradas en el dataframe:")
        for f in missing_features:
            print(f"  - {f}")

    if len(features) == 0:
        raise ValueError("Ninguna feature de la lista existe en el dataframe.")

    train_df = df.copy()

    if train_df.empty:
        raise ValueError("El dataframe de entrada está vacío.")

    # ------------------------------------------------------------
    # 2. Matriz X
    # ------------------------------------------------------------

    X_train = train_df[features].copy()

    # Convertir a numérico por seguridad
    X_train = X_train.apply(pd.to_numeric, errors="coerce")

    # ------------------------------------------------------------
    # 3. Eliminar features constantes o completamente NaN
    # ------------------------------------------------------------

    nunique = X_train.nunique(dropna=True)
    valid_features = nunique[nunique > 1].index.tolist()

    removed_constant = [f for f in features if f not in valid_features]

    if removed_constant:
        print("[INFO] Features constantes o no válidas eliminadas:")
        for f in removed_constant:
            print(f"  - {f}")

    X_train = X_train[valid_features]

    if X_train.empty:
        raise ValueError(
            "Después de eliminar features constantes/no válidas, no queda ninguna feature."
        )

    # ------------------------------------------------------------
    # 4. Matriz de correlación
    # ------------------------------------------------------------

    corr_matrix = X_train.corr()
    corr_matrix_abs = corr_matrix.abs()

    # Triángulo superior para no duplicar pares
    upper_abs = corr_matrix_abs.where(
        np.triu(np.ones(corr_matrix_abs.shape), k=1).astype(bool)
    )

    # ------------------------------------------------------------
    # 5. Detectar parejas con alta correlación y decidir eliminación
    # ------------------------------------------------------------

    dropped_corr = []
    high_corr_pairs = []

    for feature_2 in upper_abs.columns:
        correlated_with_feature_2 = upper_abs.index[upper_abs[feature_2] > threshold].tolist()

        for feature_1 in correlated_with_feature_2:
            corr_value = corr_matrix.loc[feature_1, feature_2]
            abs_corr_value = abs(corr_value)

            # Criterio original:
            # feature_1 aparece antes en el triángulo superior,
            # feature_2 se elimina.
            kept_feature = feature_1
            dropped_feature = feature_2

            high_corr_pairs.append({
                "feature_1": feature_1,
                "feature_2": feature_2,
                "correlation": corr_value,
                "abs_correlation": abs_corr_value,
                "kept_feature": kept_feature,
                "dropped_feature": dropped_feature,
                "reason": (
                    f"|corr| = {abs_corr_value:.4f} > {threshold}; "
                    f"se conserva '{kept_feature}' por aparecer antes en la lista."
                )
            })

            if dropped_feature not in dropped_corr:
                dropped_corr.append(dropped_feature)

    selected_features = [
        f for f in valid_features
        if f not in dropped_corr
    ]

    dropped_features = removed_constant + dropped_corr

    high_corr_pairs_df = pd.DataFrame(high_corr_pairs)

    if not high_corr_pairs_df.empty:
        high_corr_pairs_df = high_corr_pairs_df.sort_values(
            by="abs_correlation",
            ascending=False
        ).reset_index(drop=True)

    # ------------------------------------------------------------
    # 6. Mostrar resumen
    # ------------------------------------------------------------

    print(f"[INFO] Features iniciales: {len(features)}")
    print(f"[INFO] Features eliminadas por constantes/NaN: {len(removed_constant)}")
    print(f"[INFO] Features eliminadas por correlación > {threshold}: {len(dropped_corr)}")
    print(f"[INFO] Features finales: {len(selected_features)}")

    if not high_corr_pairs_df.empty:
        print("\n[INFO] Parejas con alta correlación:")
        print("-" * 80)

        for _, row in high_corr_pairs_df.iterrows():
            print(
                f"{row['feature_1']}  <->  {row['feature_2']} "
                f"| corr = {row['correlation']:.4f} "
                f"| se conserva: {row['kept_feature']} "
                f"| se elimina: {row['dropped_feature']}"
            )
    else:
        print(f"[INFO] No se encontraron parejas con |correlación| > {threshold}.")

    return selected_features, dropped_features, corr_matrix_abs, high_corr_pairs_df
"""
def main()-> None:
    # Cargar features ya extraídas
    df = pd.read_csv(INPUT_DIR)

    # Columnas que no son features
    non_feature_cols = ["subject_id", "epoch_id", "etiqueta", "etiqueta_3_fases", "etiqueta_4_fases"] 

    # Selección de features por multicolinealidad
    features = ["BVP_mean",
        "BVP_median",
        "BVP_std",
        "BVP_range",
        "BVP_skewness",
        "BVP_kurtosis",
        "BVP_Hjorth_Mobility",
        "BVP_Hjorth_Complexity",
        "HRV_SDNN",
        "HRV_pNN50",
        "HRV_SD1SD2",
        "HRV_HFD",
        "HRV_KFD",
        "HRV_SampEn",

        "HR_mean",
        "HR_std",

        "IBI_mean",
        "IBI_std",

        "ACC_X_trimmed_mean",
        "ACC_X_trimmed_IQR",
        "ACC_Y_trimmed_mean",
        "ACC_Y_trimmed_IQR",
        "ACC_Z_trimmed_mean",
        "ACC_Z_trimmed_IQR",
        "ACC_X_MAD_trimmed_IQR",
        "ACC_INDEX",

        "SCR_Height_mean",
        "SCR_RiseTime_mean",
        "SCR_RiseTime_max",
        "SCR_RecoveryTime_mean",
        "SCR_RecoveryTime_max",

        "TEMP_mean",
        "TEMP_std"]

    X = df.drop(columns=non_feature_cols)
    X = X[features]  
    # Cambiamos el objetivo en función del número de clasesque queremos clasficar
    y = df["etiqueta_3_fases"] 
    groups = df["subject_id"]

    # Split por sujeto
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=100
    )

    train_idx, test_idx = next(splitter.split(X, y, groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    groups_train = groups.iloc[train_idx]

    param_grid = {
        "rf__n_estimators": [100, 300, 500],
        "rf__max_depth": [10, 20, None],
        "rf__min_samples_split": [2, 5, 10],
        "rf__min_samples_leaf": [1, 2, 4],
    }

    cv = GroupKFold(n_splits=5)

    # Modelo
    model = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(
            class_weight="balanced",
            random_state=100,
            n_jobs=-1
        ))
    ])

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        verbose=2, 
        random_state=100
    )

    # Entrenamiento
    grid.fit(X_train, y_train, groups=groups_train)

    # Evaluación
    y_pred = grid.predict(X_test)

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    importances = pd.Series(
        grid.best_estimator_.named_steps["rf"].feature_importances_,
        index=X_train.columns
    ).sort_values(ascending=False)

    print(importances.head(20))
"""

def build_rf_pipeline(best_params=None, random_state=100):
    """
    Construye un Pipeline con imputación por mediana y Random Forest.

    Parameters
    ----------
    best_params : dict or None
        Diccionario de hiperparámetros óptimos. Puede venir directamente
        de GridSearchCV, por ejemplo:
        {
            "rf__n_estimators": 300,
            "rf__max_depth": 20,
            ...
        }

    random_state : int
        Semilla de reproducibilidad.

    Returns
    -------
    model : Pipeline
        Pipeline con SimpleImputer + RandomForestClassifier.
    """

    model = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        #("smote", SMOTE(random_state=random_state)),
        #("adasyn", ADASYN(random_state=random_state)),
        ("rf", RandomForestClassifier(
            class_weight= "balanced", #"balanced_subsample",
            random_state=random_state,
            n_jobs=-1
        ))
    ])

    if best_params is not None:
        model.set_params(**best_params)

    model1 = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        #("smote", SMOTE(random_state=random_state)),
        ("adasyn", ADASYN(random_state=random_state)),
        ("lgbm", LGBMClassifier(
            objective="multiclass", #"binary",
            #n_estimators=300,
            #learning_rate=0.05,
            #num_leaves=31,
            #max_depth=-1,
            class_weight=None, #"balanced",
            random_state=random_state,
            n_jobs=-1,
            verbose=-1
        ))
    ])    

    model2 = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("smote", SMOTE(random_state=random_state)),
        ("xgb", XGBClassifier(
            objective="multiclass", #"binary:logistic",
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            random_state=random_state,
            n_jobs=-1,
        ))
    ])

    return model


def optimize_rf_hyperparameters(
    X_train,
    y_train,
    groups_train,
    param_grid,
    k=5,
    scoring="f1_macro",
    random_state=100,
):
    """
    Optimiza los hiperparámetros del Random Forest usando GridSearchCV
    con GroupKFold por sujeto.

    Parameters
    ----------
    X_train : pd.DataFrame
        Features de entrenamiento.

    y_train : pd.Series
        Etiquetas de entrenamiento.

    groups_train : pd.Series
        Identificador de sujeto para cada muestra de entrenamiento.

    param_grid : dict
        Grid de hiperparámetros del Random Forest.
        Las claves deben usar prefijo 'rf__' porque el modelo está dentro
        de un Pipeline.

    k : int
        Número de folds de GroupKFold.

    scoring : str
        Métrica usada para optimizar. Recomendado: 'f1_macro'.

    random_state : int
        Semilla de reproducibilidad del Random Forest.

    Returns
    -------
    best_params : dict
        Mejor configuración encontrada.

    best_score : float
        Mejor puntuación media en validación cruzada interna.

    grid : GridSearchCV
        Objeto GridSearchCV ya entrenado.
    """

    cv = GroupKFold(n_splits=k)

    model = build_rf_pipeline(
        best_params=None,
        random_state=random_state
    )

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        verbose=2,
        refit=True
    )

    grid.fit(X_train, y_train, groups=groups_train)

    best_params = grid.best_params_
    best_score = grid.best_score_

    print("\nMejores hiperparámetros encontrados:")
    print(best_params)

    print(f"\nMejor {scoring} medio en CV interna: {best_score:.4f}")

    return best_params, best_score, grid

def save_confusion_matrix(y_true: pd.Series, y_pred: pd.Series) -> None:
    figure, axis = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(y_true, y_pred, labels=["NREM", "REM", "W"]),
        display_labels=["NREM", "REM", "W"],
    ).plot(ax=axis, cmap="Blues", colorbar=False)
    #axis.set_title("Random Forest confusion matrix")
    figure.tight_layout()
    figure.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=200)
    plt.close(figure)

def evaluate_rf_with_group_cv(
    X,
    y,
    groups,
    best_params,
    cv_strategy="groupkfold",
    k=5,
    random_state=100,
):
    """
    Entrena y valida un Random Forest con una configuración fija usando
    validación cruzada por sujeto.

    Permite comparar fácilmente distintos valores de K o usar LOSO.

    Parameters
    ----------
    X : pd.DataFrame
        Features de entrada.

    y : pd.Series
        Etiquetas.

    groups : pd.Series
        Identificador de sujeto para cada época.

    best_params : dict
        Hiperparámetros óptimos del Random Forest.

    cv_strategy : str
        Estrategia de validación:
        - 'groupkfold': GroupKFold con k folds.
        - 'loso': Leave-One-Subject-Out, implementado como LeaveOneGroupOut.

    k : int
        Número de folds si cv_strategy='groupkfold'.

    random_state : int
        Semilla de reproducibilidad del Random Forest.

    Returns
    -------
    metrics_summary : dict
        Métricas medias y desviaciones típicas.

    fold_results : list[dict]
        Métricas individuales de cada fold.

    y_true_all : list
        Etiquetas reales acumuladas de todos los folds.

    y_pred_all : list
        Predicciones acumuladas de todos los folds.
    """

    if cv_strategy == "groupkfold":
        cv =  GroupKFold(n_splits=k) #StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
        split_iterator = cv.split(X, y, groups=groups) #cv.split(X, y)
        cv_name = f"GroupKFold_{k}"

    elif cv_strategy == "loso":
        cv = LeaveOneGroupOut()
        split_iterator = cv.split(X, y, groups=groups)
        cv_name = "LOSO"

    else:
        raise ValueError("cv_strategy debe ser 'groupkfold' o 'loso'.")

    fold_results = []
    y_true_all = []
    y_pred_all = []

    for fold_idx, (train_idx, test_idx) in enumerate(split_iterator, start=1):

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        groups_train = groups.iloc[train_idx]
        groups_test = groups.iloc[test_idx]

        #overlap = set(groups_train.unique()) & set(groups_test.unique())
        #assert len(overlap) == 0, "Hay sujetos compartidos entre train y test."

        model = build_rf_pipeline(
            best_params=best_params,
            random_state=random_state
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro")
        f1_weighted = f1_score(y_test, y_pred, average="weighted")
        mcc = matthews_corrcoef(y_test, y_pred)

        fold_result = {
            "fold": fold_idx,
            "cv_strategy": cv_name,
            "train_subjects": groups_train.nunique(),
            "test_subjects": groups_test.nunique(),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "mcc": mcc,
        }

        fold_results.append(fold_result)

        y_true_all.extend(y_test.tolist())
        y_pred_all.extend(y_pred.tolist())

        print("\n" + "=" * 70)
        print(f"Fold {fold_idx} - {cv_name}")
        print("=" * 70)
        print(f"Sujetos train: {groups_train.nunique()}")
        print(f"Sujetos test:  {groups_test.nunique()}")
        print(f"Accuracy:          {acc:.4f}")
        print(f"Balanced accuracy: {bal_acc:.4f}")
        print(f"F1 macro:          {f1_macro:.4f}")
        print(f"F1 weighted:       {f1_weighted:.4f}")
        print(f"MCC:               {mcc:.4f}")

    metrics_df = pd.DataFrame(fold_results)

    metrics_summary = {
        "cv_strategy": cv_name,
        "n_folds": len(fold_results),

        "accuracy_mean": metrics_df["accuracy"].mean(),
        "accuracy_std": metrics_df["accuracy"].std(),

        "balanced_accuracy_mean": metrics_df["balanced_accuracy"].mean(),
        "balanced_accuracy_std": metrics_df["balanced_accuracy"].std(),

        "f1_macro_mean": metrics_df["f1_macro"].mean(),
        "f1_macro_std": metrics_df["f1_macro"].std(),

        "f1_weighted_mean": metrics_df["f1_weighted"].mean(),
        "f1_weighted_std": metrics_df["f1_weighted"].std(),

        "mcc_mean": metrics_df["mcc"].mean(),
        "mcc_std": metrics_df["mcc"].std(),
    }

    print("\n" + "=" * 70)
    print(f"Resumen global - {cv_name}")
    print("=" * 70)
    for key, value in metrics_summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    print("\nMatriz de confusión global:")
    print(confusion_matrix(y_true_all, y_pred_all))

    save_confusion_matrix(y_true_all, y_pred_all)

    print("\nClassification report global:")
    print(classification_report(y_true_all, y_pred_all))

    return metrics_summary, fold_results, y_true_all, y_pred_all, model, X_test


def main():
    # ============================================================
    # 1. Cargar features ya extraídas
    # ============================================================

    df = pd.read_csv(INPUT_DIR)

    # Columnas que no son features
    non_feature_cols = [
        "subject_id",
        "epoch_id",
        "etiqueta",
        "etiqueta_3_fases",
        "etiqueta_binaria", 
    ]

    # ============================================================
    # 2. Selección de features por multicolinealidad
    # ============================================================

    features = [
        "BVP_mean",
        "BVP_median",
        "BVP_std",
        "BVP_range",
        "BVP_skewness",
        "BVP_kurtosis",
        "BVP_Hjorth_Mobility",
        "BVP_Hjorth_Complexity",
        "HRV_SDNN",
        "HRV_pNN50",
        "HRV_SD1SD2",
        "HRV_HFD",
        "HRV_KFD",
        "HRV_SampEn",

        "HR_mean",
        "HR_std",

        "IBI_mean",
        "IBI_std",
        
        "ACC_X_trimmed_mean",
        "ACC_X_trimmed_IQR",
        "ACC_Y_trimmed_mean",
        "ACC_Y_trimmed_IQR",
        "ACC_Z_trimmed_mean",
        "ACC_Z_trimmed_IQR",
        "ACC_X_MAD_trimmed_IQR",
        "ACC_INDEX",

        "SCR_Height_mean",
        "SCR_RiseTime_mean",
        "SCR_RiseTime_max",
        "SCR_RecoveryTime_mean",
        "SCR_RecoveryTime_max",

        "TEMP_mean",
        "TEMP_std"
    ]

    # Comprobación de features ausentes
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        raise ValueError(f"Estas features no existen en el CSV: {missing_features}")

    # ============================================================
    # 3. Definir X, y y grupos
    # ============================================================

    X = df.drop(columns=non_feature_cols)

    # Cambiar aquí el objetivo según el número de clases
    y = df["etiqueta_binaria"]

    #y_encoded = LabelEncoder().fit_transform(y)

    groups = df["subject_id"]

    print("\nDataset cargado")
    print(f"Número de muestras: {len(X)}")
    print(f"Número de features: {X.shape[1]}")
    print(f"Número de sujetos: {groups.nunique()}")

    print("\nDistribución de clases:")
    print(y.value_counts())

    # ============================================================
    # 4. Split inicial por sujeto para optimizar hiperparámetros
    # ============================================================
    
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=RANDOM_STATE
    )

    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    #y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]


    groups_train = groups.iloc[train_idx]
    groups_test = groups.iloc[test_idx]

    overlap = set(groups_train.unique()) & set(groups_test.unique())
    assert len(overlap) == 0, "Hay sujetos compartidos entre train y test."

    print("\nSplit inicial para optimización")
    print(f"Sujetos train: {groups_train.nunique()}")
    print(f"Sujetos test:  {groups_test.nunique()}")

    # ============================================================
    # Modelo final entrenado solo con sujetos de entrenamiento
    # ============================================================

    best_model = build_rf_pipeline(
        #best_params=best_params,
        random_state=RANDOM_STATE
    )

    best_model.fit(X_train, y_train)

    y_pred_test = best_model.predict(X_test)

    # 4. Evaluación artificial: balancear sintéticamente también el test
    # Primero se imputa X_test con el imputer ya ajustado en train
    #X_test_imputed = best_model.named_steps["imputer"].transform(X_test)

    # Se aplica SMOTE al test usando y_test
    #smote_test = ADASYN(random_state=RANDOM_STATE)
    #X_test, y_test = smote_test.fit_resample(X_test_imputed, y_test)

    # Se predice directamente con el RF, porque el test ya está imputado y balanceado
    #y_pred_test = best_model.named_steps["rf"].predict(X_test)

    print("\nMatriz de confusión hold-out:")
    print(confusion_matrix(y_test, y_pred_test))

    print("\nClassification report hold-out:")
    print(classification_report(y_test, y_pred_test))

    print("F1 macro hold-out:", f1_score(y_test, y_pred_test, average="macro"))
    print("MCC hold-out:", matthews_corrcoef(y_test, y_pred_test))
    """
    # ============================================================
    # SHAP solo sobre sujetos no vistos
    # ============================================================
    
    shap_results = run_tree_shap_analysis(
        model_pipeline=best_model,
        X=X_test,
        y_true=y_test,
        y_pred=y_pred_test,
        feature_names=X.columns.tolist(),
        class_names=None,
        output_dir="shap_outputs_holdout_subjects",
        model_step_name="rf",
        max_display=30
    )

    print("\nArchivos guardados:")
    print("- best_rf_params.json")
    print("- rf_groupkfold_5_results.csv")
    print("- rf_loso_results.csv")
    print("- rf_feature_importances.csv")
    
    # ============================================================
    # 5. Grid de hiperparámetros
    # ============================================================
    
    param_grid_rf = {
        "rf__n_estimators": [100, 300, 500],
        "rf__max_depth": [10, 20, None],
        "rf__min_samples_split": [2, 5, 10],
        "rf__min_samples_leaf": [1, 2, 4],
    }

    # ============================================================
    # 6. Optimizar hiperparámetros en entrenamiento
    # ============================================================

    best_params, best_score, grid = optimize_rf_hyperparameters(
        X_train=X_train,
        y_train=y_train,
        groups_train=groups_train,
        param_grid=param_grid_rf,
        k=5,
        scoring="f1_macro",
        random_state=RANDOM_STATE
    )

    pd.Series(best_params).to_json("best_rf_params.json", indent=4)
    
    # ============================================================
    # 7. Evaluación hold-out inicial con la mejor configuración
    # ============================================================

    print("\n" + "=" * 70)
    print("Evaluación hold-out sobre sujetos no vistos")
    print("=" * 70)

    

    best_model = build_rf_pipeline(
        best_params=best_params,
        random_state=RANDOM_STATE
    )

    best_model.fit(X_train, y_train)

    y_pred_test = best_model.predict(X_test)

    print("\nMatriz de confusión hold-out:")
    print(confusion_matrix(y_test, y_pred_test))

    print("\nClassification report hold-out:")
    print(classification_report(y_test, y_pred_test))

    print("F1 macro hold-out:", f1_score(y_test, y_pred_test, average="macro"))
    print("MCC hold-out:", matthews_corrcoef(y_test, y_pred_test))

    importances = pd.Series(
        best_model.named_steps["rf"].feature_importances_,
        index=X.columns
    ).sort_values(ascending=False)

    print("\nTop 20 features más importantes en hold-out:")
    print(importances.head(20))
    
    # ============================================================
    # 8. Evaluación con GroupKFold usando la configuración óptima
    # ============================================================

    best_params = {
    "rf__max_depth":10,
    "rf__min_samples_leaf":4,
    "rf__min_samples_split":10,
    "rf__n_estimators":500
    }
    
    metrics_k5, fold_results_k5, y_true_k5, y_pred_k5, best_model, x_test_k5 = evaluate_rf_with_group_cv(
        X=X,
        y=y,
        groups=groups,
        best_params=None, #best_params,
        cv_strategy="groupkfold",
        k=5,
        random_state=RANDOM_STATE
    )
    
    pd.DataFrame(fold_results_k5).to_csv(
        os.path.join(RESULTS_DIR, "rf_groupkfold_5_results.csv"),
        #"rf_groupkfold_5_results.csv",
        index=False
    )

    

    # ============================================================
    # 9. Evaluación LOSO usando la misma configuración óptima
    # ============================================================
    
    metrics_loso, fold_results_loso, y_true_loso, y_pred_loso = evaluate_rf_with_group_cv(
        X=X,
        y=y,
        groups=groups,
        best_params=best_params,
        cv_strategy="loso",
        random_state=RANDOM_STATE
    )

    # ============================================================
    # 10. Guardar resultados útiles
    # ============================================================

    #pd.Series(best_params).to_json("best_rf_params.json", indent=4)

    pd.DataFrame(fold_results_k5).to_csv(
        #os.path.join(RESULTS_DIR, "rf_groupkfold_5_results.csv"),
        #"rf_groupkfold_5_results.csv",
        index=False
    )
    
    pd.DataFrame(fold_results_loso).to_csv(
        #os.path.join(RESULTS_DIR, "rf_loso_results.csv"),
        #"rf_loso_results.csv",
        index=False
    )

    importances.to_csv(
        #os.path.join(RESULTS_DIR, "rf_feature_importances.csv"),
        #"rf_feature_importances.csv",
        header=["importance"]
    )"""

    
    


if __name__ == "__main__":
    main()    


