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


def build_epoch_dataframe(
    df: pd.DataFrame,
    subject_id: str,
) -> pd.DataFrame:
    """
    Agrupa el dataframe en épocas de 30 segundos y construye la tabla final.

    De momento, solo devuelve:
    - subject_id
    - epoch_id
    - etiqueta
    """
    if LABEL_COLUMN not in df.columns:
        raise ValueError(
            f"No se ha encontrado la columna de etiqueta '{LABEL_COLUMN}'. "
            f"Columnas disponibles: {list(df.columns)}"
        )

    grouped = df.groupby("epoch_id", sort=True)

    epoch_df = grouped[LABEL_COLUMN].apply(get_epoch_label).reset_index()

    epoch_df = epoch_df.rename(
        columns={
            LABEL_COLUMN: "etiqueta",
        }
    )

    epoch_df.insert(0, "subject_id", subject_id)

    epoch_df = epoch_df[
        [
            "subject_id",
            "epoch_id",
            "etiqueta",
        ]
    ]

    return epoch_df


def build_extraction_report(
    original_df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
    epoch_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Genera un pequeño informe de control.
    """
    dropped_columns = [
        column for column in RESPIRATORY_EVENT_COLUMNS
        if column in original_df.columns
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