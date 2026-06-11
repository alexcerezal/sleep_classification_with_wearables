"""
feature_extraction.py

Primera versión del script de extracción de características para DREAMT.

Entrada:
- CSV ya preprocesado.

Salida:
- CSV agrupado en épocas de 30 segundos.
- De momento, cada época contiene:
    subject_id
    epoch_id
    etiqueta
    Features ACC:
    - ACC_X_trimmed_mean
    - ACC_X_trimmed_max
    - ACC_X_trimmed_IQR
    - ACC_Y_trimmed_mean
    - ACC_Y_trimmed_max
    - ACC_Y_trimmed_IQR
    - ACC_Z_trimmed_mean
    - ACC_Z_trimmed_max
    - ACC_Z_trimmed_IQR

    - ACC_X_MAD_trimmed_mean
    - ACC_X_MAD_trimmed_max
    - ACC_X_MAD_trimmed_IQR
    - ACC_Y_MAD_trimmed_mean
    - ACC_Y_MAD_trimmed_max
    - ACC_Y_MAD_trimmed_IQR
    - ACC_Z_MAD_trimmed_mean
    - ACC_Z_MAD_trimmed_max
    - ACC_Z_MAD_trimmed_IQR

    - ACC_INDEX

Preprocesamiento previo esperado:
- BVP ya filtrada.
- HR ya limpiada y normalizada.
- IBI ya limpiada.
- ACC ya filtrada.
- EDA ya preprocesada.

En esta primera versión NO se extraen todavía features estadísticas ni
frecuenciales. Solo se prepara la estructura por épocas, siguiendo el flujo
general de DREAMT_FE: pasar de señal continua/muestreada a ventanas de 30 s.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# CONFIGURACIÓN DEL USUARIO
# =============================================================================

INPUT_CSV = r"C:\Proyectos_compartidos\TFG\sleep_classification_with_wearables\data\S002_preprocessed.csv"
OUTPUT_DIR = r"C:\Proyectos_compartidos\TFG\sleep_classification_with_wearables\results\feature_extraction"

# Si el CSV ya tiene una columna con el identificador del sujeto, pon aquí su nombre.
# Si no existe, se usará el nombre del archivo como subject_id.
SUBJECT_ID_COLUMN = "subject_id"

# Cambia esto si tu columna de etapa de sueño tiene otro nombre.
# Posibles nombres habituales: "Sleep_Stage", "sleep_stage", "Stage", "label", "Label"
LABEL_COLUMN = "Sleep_Stage"

# Si el CSV tiene timestamps, se pueden usar para construir las épocas.
# Si no existe esta columna, se agrupará por número de muestras.
TIMESTAMP_COLUMN = "TIMESTAMP"

# Frecuencia base del CSV preprocesado.
# Si el dataframe está alineado a BVP, normalmente será 64 Hz.
FS = 64.0

# Duración estándar de las épocas PSG.
EPOCH_DURATION_SECONDS = 30

# Columnas ACC.
ACC_COLUMNS = ["ACC_X", "ACC_Y", "ACC_Z"]

# Columnas de eventos respiratorios que se eliminan.
RESPIRATORY_EVENT_COLUMNS = [
    "Obstructive_Apnea",
    "Central_Apnea",
    "Hypopnea",
    "Multiple_Events",
]


# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def get_subject_id(df: pd.DataFrame, input_csv: Path) -> str:
    """
    Obtiene el subject_id.

    Si existe SUBJECT_ID_COLUMN en el CSV, usa el primer valor no nulo.
    Si no existe, usa el nombre del archivo.
    """
    if SUBJECT_ID_COLUMN in df.columns:
        subject_values = df[SUBJECT_ID_COLUMN].dropna().unique()

        if len(subject_values) > 0:
            return str(subject_values[0])

    return input_csv.stem


def drop_respiratory_event_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina las columnas de eventos respiratorios si existen.

    No lanza error si alguna no está presente, porque puede depender de cómo
    se haya construido el CSV.
    """
    columns_to_drop = [
        column for column in RESPIRATORY_EVENT_COLUMNS
        if column in df.columns
    ]

    return df.drop(columns=columns_to_drop)


def infer_epoch_id_from_timestamp(df: pd.DataFrame) -> pd.Series:
    """
    Construye epoch_id usando la columna de timestamp.

    Cada época corresponde a una ventana de 30 segundos desde el inicio
    del registro.
    """
    timestamps = pd.to_numeric(df[TIMESTAMP_COLUMN], errors="coerce")

    timestamps = (
        timestamps
        .interpolate(method="linear", limit_direction="both")
        .ffill()
        .bfill()
    )

    relative_time = timestamps - timestamps.iloc[0]

    epoch_id = np.floor(
        relative_time / EPOCH_DURATION_SECONDS
    ).astype(int)

    return pd.Series(epoch_id, index=df.index, name="epoch_id")


def infer_epoch_id_from_sample_index(df: pd.DataFrame) -> pd.Series:
    """
    Construye epoch_id usando el índice de las muestras.

    Se asume que el CSV está muestreado a FS Hz. Para FS=64 Hz y épocas de
    30 segundos:

        muestras_por_epoca = 64 * 30 = 1920
    """
    samples_per_epoch = int(FS * EPOCH_DURATION_SECONDS)

    if samples_per_epoch <= 0:
        raise ValueError("samples_per_epoch debe ser mayor que 0.")

    epoch_id = np.arange(len(df)) // samples_per_epoch

    return pd.Series(epoch_id, index=df.index, name="epoch_id")


def add_epoch_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Añade la columna epoch_id.

    Prioridad:
    1. Si existe TIMESTAMP_COLUMN, usa tiempo real.
    2. Si no, usa número de muestra y FS.
    """
    df = df.copy()

    if TIMESTAMP_COLUMN in df.columns:
        df["epoch_id"] = infer_epoch_id_from_timestamp(df)
    else:
        df["epoch_id"] = infer_epoch_id_from_sample_index(df)

    return df


def get_epoch_label(labels: pd.Series):
    """
    Obtiene la etiqueta representativa de una época.

    Se usa la moda, es decir, la etiqueta más frecuente dentro de la ventana.
    Esto es robusto por si en una ventana aparece algún valor puntual distinto.

    Si hay empate, pandas devuelve las modas ordenadas y tomamos la primera.
    """
    labels = labels.dropna()

    if len(labels) == 0:
        return np.nan

    mode_values = labels.mode()

    if len(mode_values) == 0:
        return np.nan

    return mode_values.iloc[0]


def to_clean_numpy(series: pd.Series) -> np.ndarray:
    """
    Convierte una serie a array numérico, interpolando posibles NaN.
    """
    numeric = pd.to_numeric(series, errors="coerce")

    numeric = (
        numeric
        .interpolate(method="linear", limit_direction="both")
        .ffill()
        .bfill()
    )

    return numeric.to_numpy(dtype=float)


# =============================================================================
# FEATURES ACC SEGÚN DREAMT_FE
# =============================================================================

def acc_trimmed_summary(acc: np.ndarray) -> tuple[float, float, float]:
    """
    Calcula mean, max e IQR sobre la señal recortada entre cuantiles 10 y 90.

    DREAMT_FE usa:
        acc_filtered = acc[(acc > q10) & (acc < q90)]

    Si el recorte deja la señal vacía, se devuelven:
        mean(acc), max(acc), 0
    """
    acc = np.asarray(acc, dtype=float)
    acc = acc[np.isfinite(acc)]

    if len(acc) == 0:
        return np.nan, np.nan, np.nan

    q10 = np.quantile(acc, 0.10)
    q90 = np.quantile(acc, 0.90)

    acc_filtered = acc[(acc > q10) & (acc < q90)]

    if len(acc_filtered) == 0:
        return float(np.mean(acc)), float(np.max(acc)), 0.0

    trimmed_mean = float(np.mean(acc_filtered))
    trimmed_max = float(np.max(acc_filtered))

    # Nota: DREAMT_FE calcula Q3 sobre acc_filtered y Q1 sobre acc.
    # Aquí mantengo una versión coherente del IQR trimmed: Q3-Q1 sobre acc_filtered.
    trimmed_iqr = float(
        np.quantile(acc_filtered, 0.75)
        - np.quantile(acc_filtered, 0.25)
    )

    return trimmed_mean, trimmed_max, trimmed_iqr


def mad_trimmed_summary(
    acc: np.ndarray,
    segment_seconds: int = 30,
) -> tuple[float, float, float]:
    """
    Calcula mean, max e IQR de la señal MAD.

    En DREAMT_FE:
    - Se divide la época de 30 segundos en 6 subventanas de 5 segundos.
    - En cada subventana se calcula:
        MAD = mean(abs(x - mean(x)))
    - Después se calculan:
        mean(MADs), max(MADs), IQR(MADs)

    Se mantiene el chequeo trimmed inicial para detectar señales degeneradas.
    """
    acc = np.asarray(acc, dtype=float)
    acc = acc[np.isfinite(acc)]

    if len(acc) == 0:
        return np.nan, np.nan, np.nan

    q10 = np.quantile(acc, 0.10)
    q90 = np.quantile(acc, 0.90)

    acc_filtered = acc[(acc > q10) & (acc < q90)]

    if len(acc_filtered) == 0:
        return float(np.mean(acc)), float(np.max(acc)), 0.0

    num_splits = int(segment_seconds / 5)

    if num_splits <= 0:
        raise ValueError("segment_seconds debe ser al menos 5.")

    splits = np.array_split(acc, num_splits)

    mads = np.array(
        [
            np.mean(np.abs(split - np.mean(split)))
            for split in splits
            if len(split) > 0
        ],
        dtype=float,
    )

    if len(mads) == 0:
        return np.nan, np.nan, np.nan

    mad_mean = float(np.mean(mads))
    mad_max = float(np.max(mads))
    mad_iqr = float(np.quantile(mads, 0.75) - np.quantile(mads, 0.25))

    return mad_mean, mad_max, mad_iqr


def calculate_acc_index(
    acc_x: np.ndarray,
    acc_y: np.ndarray,
    acc_z: np.ndarray,
    fs: float,
) -> float:
    """
    Calcula ACC_INDEX siguiendo la idea de DREAMT_FE.

    DREAMT_FE:
    - Calcula magnitud triaxial:
        acc = sqrt(x^2 + y^2 + z^2)
    - Divide la época en bloques de 5 segundos.
    - Calcula std de cada bloque.
    - Suma las std de los 6 bloques de una época de 30 segundos.

    Aquí se generaliza a la FS configurada.
    """
    acc_x = np.asarray(acc_x, dtype=float)
    acc_y = np.asarray(acc_y, dtype=float)
    acc_z = np.asarray(acc_z, dtype=float)

    min_len = min(len(acc_x), len(acc_y), len(acc_z))

    if min_len == 0:
        return np.nan

    acc_x = acc_x[:min_len]
    acc_y = acc_y[:min_len]
    acc_z = acc_z[:min_len]

    acc_magnitude = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)

    samples_per_5s = int(fs * 5)

    if samples_per_5s <= 0:
        raise ValueError("samples_per_5s debe ser mayor que 0.")

    num_complete_periods = len(acc_magnitude) // samples_per_5s

    if num_complete_periods == 0:
        return float(np.std(acc_magnitude))

    trimmed_length = num_complete_periods * samples_per_5s

    reshaped_acc = acc_magnitude[:trimmed_length].reshape(
        num_complete_periods,
        samples_per_5s,
    )

    acc_stds = np.std(reshaped_acc, axis=1)

    acc_index = float(np.sum(acc_stds))

    return acc_index


def extract_acc_features_from_epoch(
    epoch_df: pd.DataFrame,
) -> dict[str, float]:
    """
    Extrae features ACC de una época de 30 segundos.
    """
    missing_acc_columns = [
        column for column in ACC_COLUMNS
        if column not in epoch_df.columns
    ]

    if missing_acc_columns:
        raise ValueError(
            f"Faltan columnas ACC en el CSV: {missing_acc_columns}. "
            f"Columnas disponibles: {list(epoch_df.columns)}"
        )

    acc_x = to_clean_numpy(epoch_df[ACC_COLUMNS[0]])
    acc_y = to_clean_numpy(epoch_df[ACC_COLUMNS[1]])
    acc_z = to_clean_numpy(epoch_df[ACC_COLUMNS[2]])

    features = {}

    axis_arrays = {
        "ACC_X": acc_x,
        "ACC_Y": acc_y,
        "ACC_Z": acc_z,
    }

    for axis_name, axis_values in axis_arrays.items():
        trimmed_mean, trimmed_max, trimmed_iqr = acc_trimmed_summary(axis_values)

        features[f"{axis_name}_trimmed_mean"] = trimmed_mean
        features[f"{axis_name}_trimmed_max"] = trimmed_max
        features[f"{axis_name}_trimmed_IQR"] = trimmed_iqr

    for axis_name, axis_values in axis_arrays.items():
        mad_mean, mad_max, mad_iqr = mad_trimmed_summary(
            axis_values,
            segment_seconds=EPOCH_DURATION_SECONDS,
        )

        features[f"{axis_name}_MAD_trimmed_mean"] = mad_mean
        features[f"{axis_name}_MAD_trimmed_max"] = mad_max
        features[f"{axis_name}_MAD_trimmed_IQR"] = mad_iqr

    features["ACC_INDEX"] = calculate_acc_index(
        acc_x=acc_x,
        acc_y=acc_y,
        acc_z=acc_z,
        fs=FS,
    )

    return features

# =============================================================================
# CONSTRUCCIÓN DEL DATAFRAME FINAL POR ÉPOCAS
# =============================================================================

def build_epoch_dataframe(
    df: pd.DataFrame,
    subject_id: str,
) -> pd.DataFrame:
    """
    Agrupa el dataframe en épocas de 30 segundos y construye la tabla final.

    Devuelve:
    - subject_id
    - epoch_id
    - etiqueta
    - features ACC
    """
    if LABEL_COLUMN not in df.columns:
        raise ValueError(
            f"No se ha encontrado la columna de etiqueta '{LABEL_COLUMN}'. "
            f"Columnas disponibles: {list(df.columns)}"
        )

    rows = []

    for epoch_id, epoch_data in df.groupby("epoch_id", sort=True):
        row = {
            "subject_id": subject_id,
            "epoch_id": int(epoch_id),
            "etiqueta": get_epoch_label(epoch_data[LABEL_COLUMN]),
        }

        row.update(
            extract_acc_features_from_epoch(epoch_data)
        )

        rows.append(row)

    epoch_df = pd.DataFrame(rows)

    ordered_columns = [
        "subject_id",
        "epoch_id",
        "etiqueta",
        "ACC_X_trimmed_mean",
        "ACC_X_trimmed_max",
        "ACC_X_trimmed_IQR",
        "ACC_Y_trimmed_mean",
        "ACC_Y_trimmed_max",
        "ACC_Y_trimmed_IQR",
        "ACC_Z_trimmed_mean",
        "ACC_Z_trimmed_max",
        "ACC_Z_trimmed_IQR",
        "ACC_X_MAD_trimmed_mean",
        "ACC_X_MAD_trimmed_max",
        "ACC_X_MAD_trimmed_IQR",
        "ACC_Y_MAD_trimmed_mean",
        "ACC_Y_MAD_trimmed_max",
        "ACC_Y_MAD_trimmed_IQR",
        "ACC_Z_MAD_trimmed_mean",
        "ACC_Z_MAD_trimmed_max",
        "ACC_Z_MAD_trimmed_IQR",
        "ACC_INDEX",
    ]

    epoch_df = epoch_df[ordered_columns]

    return epoch_df


def build_extraction_report(
    original_df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
    epoch_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Genera un informe de control.
    """
    dropped_columns = [
        column for column in RESPIRATORY_EVENT_COLUMNS
        if column in original_df.columns
    ]

    acc_feature_columns = [
        column for column in epoch_df.columns
        if column.startswith("ACC_")
    ]

    report = {
        "num_original_rows": len(original_df),
        "num_epochs": len(epoch_df),
        "epoch_duration_seconds": EPOCH_DURATION_SECONDS,
        "fs_assumed_if_no_timestamp": FS,
        "label_column_used": LABEL_COLUMN,
        "dropped_respiratory_columns": ", ".join(dropped_columns),
        "num_rows_after_dropping_columns": len(cleaned_df),
        "num_epochs_with_nan_label": int(epoch_df["etiqueta"].isna().sum()),
        "num_acc_features": len(acc_feature_columns),
        "num_epochs_with_any_nan_feature": int(
            epoch_df[acc_feature_columns].isna().any(axis=1).sum()
        ),
    }

    return pd.DataFrame([report])



# =============================================================================
# SCRIPT PRINCIPAL
# =============================================================================

def main() -> None:
    input_csv = Path(INPUT_CSV)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_csv.exists():
        raise FileNotFoundError(f"No existe el CSV de entrada: {input_csv}")

    df = pd.read_csv(input_csv)
    original_df = df.copy()

    subject_id = get_subject_id(df, input_csv)

    df = drop_respiratory_event_columns(df)

    df = add_epoch_id(df)

    epoch_df = build_epoch_dataframe(
        df=df,
        subject_id=subject_id,
    )

    stem = input_csv.stem

    output_csv = output_dir / f"{stem}_epochs_basic.csv"
    report_csv = output_dir / f"{stem}_feature_extraction_report.csv"

    epoch_df.to_csv(output_csv, index=False)

    report_df = build_extraction_report(
        original_df=original_df,
        cleaned_df=df,
        epoch_df=epoch_df,
    )

    report_df.to_csv(report_csv, index=False)

    print("Extracción básica de features completada.")
    print(f"CSV por épocas: {output_csv}")
    print(f"Informe:        {report_csv}")
    print()
    print("Primeras épocas:")
    print(epoch_df.head().to_string(index=False))
    print()
    print("Resumen:")
    print(report_df.to_string(index=False))


if __name__ == "__main__":
    main()