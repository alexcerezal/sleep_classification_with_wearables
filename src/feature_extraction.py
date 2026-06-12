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

    Features BVP:
 
    -features estadísticas básicas:
    ├── BVP_mean
    ├── BVP_median
    ├── BVP_std
    ├── BVP_min 
    ├── BVP_max
    └── BVP_range

    -features estadísticas propias:
    ├── BVP_iqr
    ├── BVP_mad
    ├── BVP_rms
    ├── BVP_skewness
    └── BVP_kurtosis
 
    - features HRV con NeuroKit2:
    ├── HRV_SDNN
    ├── HRV_RMSSD
    ├── HRV_pNN50
    ├── HRV_SD1
    ├── HRV_SD2
    ├── HRV_SD1SD2
    ├── HRV_HFD
    ├── HRV_KFD
    └── HRV_SampEn

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

    Features EDA:
    - SCR_Height_mean
    - SCR_Height_max
    - SCR_Amplitude_mean
    - SCR_Amplitude_max
    - SCR_RiseTime_mean
    - SCR_RiseTime_max
    - SCR_RecoveryTime_mean
    - SCR_RecoveryTime_max

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
import neurokit2 as nk
import warnings
from neurokit2.misc import NeuroKitWarning
from scipy.stats import skew, kurtosis


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

BVP_COLUMN = "BVP"

# BVP de Empatica E4 en DREAMT suele estar a 64 Hz.
FS_BVP = 64.0

# Columnas ACC.
ACC_COLUMNS = ["ACC_X", "ACC_Y", "ACC_Z"]

EDA_COLUMN = "EDA"

TEMP_COLUMN = "TEMP"

# La EDA real de Empatica E4 está a 4 Hz.
# Si el CSV está alineado a 64 Hz, la EDA suele estar repetida/interpolada.
FS_EDA = 4.0

# Factor entre 64 Hz y 4 Hz.
EDA_REPEAT_FACTOR_TO_64HZ = 16

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
# FEATURES BVP / HRV
# =============================================================================

BVP_FEATURE_COLUMNS = [
    "BVP_mean",
    "BVP_median",
    "BVP_std",
    "BVP_min",
    "BVP_max",
    "BVP_range",
    "BVP_iqr",
    "BVP_mad",
    "BVP_rms",
    "BVP_skewness",
    "BVP_kurtosis",
    "HRV_SDNN",
    "HRV_RMSSD",
    "HRV_pNN50",
    "HRV_SD1",
    "HRV_SD2",
    "HRV_SD1SD2",
    "HRV_HFD",
    "HRV_KFD",
    "HRV_SampEn",
]


def extract_bvp_statistical_features(
    bvp_values: np.ndarray,
) -> dict[str, float]:
    """
    Extrae features estadísticas de BVP por época.

    Incluye:
    - Estadísticas básicas similares a DREAMT_FE:
        mean, median, std, min, max, range
    - Estadísticas añadidas propias:
        iqr, mad, rms, skewness, kurtosis
    """
    bvp_values = np.asarray(bvp_values, dtype=float)
    bvp_values = bvp_values[np.isfinite(bvp_values)]

    if len(bvp_values) == 0:
        return {
            "BVP_mean": np.nan,
            "BVP_median": np.nan,
            "BVP_std": np.nan,
            "BVP_min": np.nan,
            "BVP_max": np.nan,
            "BVP_range": np.nan,
            "BVP_iqr": np.nan,
            "BVP_mad": np.nan,
            "BVP_rms": np.nan,
            "BVP_skewness": np.nan,
            "BVP_kurtosis": np.nan,
        }

    bvp_mean = np.mean(bvp_values)
    bvp_median = np.median(bvp_values)
    bvp_std = np.std(bvp_values)
    bvp_min = np.min(bvp_values)
    bvp_max = np.max(bvp_values)

    q1 = np.quantile(bvp_values, 0.25)
    q3 = np.quantile(bvp_values, 0.75)

    return {
        "BVP_mean": float(bvp_mean),
        "BVP_median": float(bvp_median),
        "BVP_std": float(bvp_std),
        "BVP_min": float(bvp_min),
        "BVP_max": float(bvp_max),
        "BVP_range": float(bvp_max - bvp_min),
        "BVP_iqr": float(q3 - q1),
        "BVP_mad": float(np.mean(np.abs(bvp_values - bvp_mean))),
        "BVP_rms": float(np.sqrt(np.mean(bvp_values**2))),
        "BVP_skewness": float(skew(bvp_values, bias=False)) if len(bvp_values) > 2 else np.nan,
        "BVP_kurtosis": float(kurtosis(bvp_values, bias=False)) if len(bvp_values) > 3 else np.nan,
    }


def extract_hrv_features_from_bvp_with_neurokit(
    bvp_values: np.ndarray,
    sampling_rate: float,
) -> dict[str, float]:
    """
    Extrae features HRV desde BVP usando NeuroKit2.

    Estrategia:
    1. Limpiar la señal BVP de la época.
    2. Detectar picos PPG/BVP con NeuroKit2.
    3. Calcular métricas HRV temporales, no lineales y de Poincaré.
    4. Devolver únicamente las métricas seleccionadas.

    Features:
    - HRV_SDNN
    - HRV_RMSSD
    - HRV_pNN50
    - HRV_SD1
    - HRV_SD2
    - HRV_SD1SD2
    - HRV_HFD
    - HRV_KFD
    - HRV_SampEn
    """
    empty_features = {
        "HRV_SDNN": np.nan,
        "HRV_RMSSD": np.nan,
        "HRV_pNN50": np.nan,
        "HRV_SD1": np.nan,
        "HRV_SD2": np.nan,
        "HRV_SD1SD2": np.nan,
        "HRV_HFD": np.nan,
        "HRV_KFD": np.nan,
        "HRV_SampEn": np.nan,
    }

    bvp_values = np.asarray(bvp_values, dtype=float)
    bvp_values = bvp_values[np.isfinite(bvp_values)]

    if len(bvp_values) < int(5 * sampling_rate):
        return empty_features

    try:
        # La BVP ya viene preprocesada, pero NeuroKit2 necesita una señal válida
        # para detectar picos PPG.
        ppg_signals, ppg_info = nk.ppg_process(
            ppg_signal=bvp_values,
            sampling_rate=sampling_rate,
        )

        peaks = ppg_info.get("PPG_Peaks", None)

        if peaks is None or len(peaks) < 3:
            return empty_features

        hrv_time = nk.hrv_time(
            peaks,
            sampling_rate=sampling_rate,
            show=False,
        )

        hrv_nonlinear = nk.hrv_nonlinear(
            peaks,
            sampling_rate=sampling_rate,
            show=False,
        )

    except Exception:
        return empty_features

    def get_feature(source_df: pd.DataFrame, column_name: str) -> float:
        """
        Extrae una feature de un dataframe de NeuroKit2 de forma segura.
        """
        if column_name not in source_df.columns:
            return np.nan

        value = source_df[column_name].iloc[0]

        if pd.isna(value) or not np.isfinite(value):
            return np.nan

        return float(value)

    features = {
        "HRV_SDNN": get_feature(hrv_time, "HRV_SDNN"),
        "HRV_RMSSD": get_feature(hrv_time, "HRV_RMSSD"),
        "HRV_pNN50": get_feature(hrv_time, "HRV_pNN50"),

        "HRV_SD1": get_feature(hrv_nonlinear, "HRV_SD1"),
        "HRV_SD2": get_feature(hrv_nonlinear, "HRV_SD2"),
        "HRV_SD1SD2": get_feature(hrv_nonlinear, "HRV_SD1SD2"),

        "HRV_HFD": get_feature(hrv_nonlinear, "HRV_HFD"),
        "HRV_KFD": get_feature(hrv_nonlinear, "HRV_KFD"),
        "HRV_SampEn": get_feature(hrv_nonlinear, "HRV_SampEn"),
    }

    return features


def extract_bvp_features_from_epoch(
    epoch_df: pd.DataFrame,
) -> dict[str, float]:
    """
    Extrae todas las features BVP/HRV de una época de 30 segundos.

    Combina:
    - Estadísticas BVP.
    - HRV calculada desde BVP mediante NeuroKit2.
    """
    if BVP_COLUMN not in epoch_df.columns:
        raise ValueError(
            f"No se ha encontrado la columna BVP '{BVP_COLUMN}'. "
            f"Columnas disponibles: {list(epoch_df.columns)}"
        )

    bvp_values = to_clean_numpy(epoch_df[BVP_COLUMN])

    features = {}

    features.update(
        extract_bvp_statistical_features(bvp_values)
    )

    features.update(
        extract_hrv_features_from_bvp_with_neurokit(
            bvp_values=bvp_values,
            sampling_rate=FS_BVP,
        )
    )

    return features

# =============================================================================
# FEATURES ACC 
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
# FEATURES EDA 
# =============================================================================

SCR_NAMES = [
    "SCR_Height",
    "SCR_Amplitude",
    "SCR_RiseTime",
    "SCR_RecoveryTime",
]


def reduce_eda_epoch_to_4hz(eda_epoch: np.ndarray) -> np.ndarray:
    """
    Reduce una época EDA desde el dataframe alineado a 64 Hz hasta 4 Hz.

    DREAMT_FE trabaja con la EDA real de Empatica E4 a 4 Hz. Si el CSV está
    alineado a 64 Hz, se toma una muestra cada 16.
    """
    eda_epoch = np.asarray(eda_epoch, dtype=float)

    if len(eda_epoch) >= EDA_REPEAT_FACTOR_TO_64HZ:
        eda_4hz = eda_epoch[1::EDA_REPEAT_FACTOR_TO_64HZ]
    else:
        eda_4hz = eda_epoch

    eda_4hz = eda_4hz[np.isfinite(eda_4hz)]

    return eda_4hz

def extract_scr_dataframe_with_neurokit(eda_4hz: np.ndarray) -> pd.DataFrame:
    """
    Extrae las columnas SCR mediante NeuroKit2.

    Como la EDA de Empatica está a 4 Hz, NeuroKit2 lanza un warning indicando
    que omite su filtrado interno. En este pipeline no es problemático porque
    la EDA ya ha sido preprocesada antes del feature extraction.
    """
    eda_4hz = np.asarray(eda_4hz, dtype=float)
    eda_4hz = eda_4hz[np.isfinite(eda_4hz)]

    if len(eda_4hz) < 10:
        return pd.DataFrame(columns=SCR_NAMES)

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="EDA signal is sampled at very low frequency. Skipping filtering.",
                category=NeuroKitWarning,
            )

            signals, info = nk.eda_process(
                eda_signal=eda_4hz,
                sampling_rate=FS_EDA,
                method="neurokit",
            )

    except Exception:
        return pd.DataFrame(columns=SCR_NAMES)

    missing_scr_columns = [
        column for column in SCR_NAMES
        if column not in signals.columns
    ]

    if missing_scr_columns:
        return pd.DataFrame(columns=SCR_NAMES)

    scr_df = signals[SCR_NAMES].copy()

    for column in SCR_NAMES:
        scr_df[column] = pd.to_numeric(scr_df[column], errors="coerce")

    return scr_df


def summarize_scr_features(scr_df: pd.DataFrame) -> dict[str, float]:
    """
    Calcula mean y max para cada variable SCR.

    Se sigue la lógica de DREAMT_FE:
    para cada columna SCR se calculan estadísticas resumen por época.

    Features generadas:
        SCR_Height_mean
        SCR_Height_max
        SCR_Amplitude_mean
        SCR_Amplitude_max
        SCR_RiseTime_mean
        SCR_RiseTime_max
        SCR_RecoveryTime_mean
        SCR_RecoveryTime_max
    """
    features = {}

    for scr_name in SCR_NAMES:
        if scr_name not in scr_df.columns:
            features[f"{scr_name}_mean"] = np.nan
            features[f"{scr_name}_max"] = np.nan
            continue

        values = pd.to_numeric(scr_df[scr_name], errors="coerce")
        values = values.replace([np.inf, -np.inf], np.nan)

        if values.dropna().empty:
            features[f"{scr_name}_mean"] = np.nan
            features[f"{scr_name}_max"] = np.nan
        else:
            features[f"{scr_name}_mean"] = float(values.mean(skipna=True))
            features[f"{scr_name}_max"] = float(values.max(skipna=True))

    return features


def extract_eda_features_from_epoch(
    epoch_df: pd.DataFrame,
) -> dict[str, float]:
    """
    Extrae features EDA de una época de 30 segundos según DREAMT_FE.

    Procedimiento:
    1. Tomar la EDA de la época.
    2. Reducirla a 4 Hz si viene alineada a 64 Hz.
    3. Procesarla con NeuroKit2.
    4. Extraer:
        SCR_Height
        SCR_Amplitude
        SCR_RiseTime
        SCR_RecoveryTime
    5. Calcular mean y max de cada variable.
    """
    if EDA_COLUMN not in epoch_df.columns:
        raise ValueError(
            f"No se ha encontrado la columna EDA '{EDA_COLUMN}'. "
            f"Columnas disponibles: {list(epoch_df.columns)}"
        )

    eda_epoch = to_clean_numpy(epoch_df[EDA_COLUMN])
    eda_4hz = reduce_eda_epoch_to_4hz(eda_epoch)

    scr_df = extract_scr_dataframe_with_neurokit(eda_4hz)

    return summarize_scr_features(scr_df)


# =============================================================================
# FEATURES TEMP SEGÚN DREAMT_FE
# =============================================================================

def extract_temp_features_from_epoch(
    epoch_df: pd.DataFrame,
) -> dict[str, float]:
    """
    Extrae features de temperatura de una época de 30 segundos según DREAMT_FE.

    Features:
    - TEMP_mean
    - TEMP_median
    - TEMP_max
    - TEMP_min
    - TEMP_std

    La señal TEMP ya debe venir preprocesada o limpia desde el CSV de entrada.
    """
    if TEMP_COLUMN not in epoch_df.columns:
        raise ValueError(
            f"No se ha encontrado la columna TEMP '{TEMP_COLUMN}'. "
            f"Columnas disponibles: {list(epoch_df.columns)}"
        )

    temp_values = to_clean_numpy(epoch_df[TEMP_COLUMN])

    temp_values = temp_values[np.isfinite(temp_values)]

    if len(temp_values) == 0:
        return {
            "TEMP_mean": np.nan,
            "TEMP_median": np.nan,
            "TEMP_max": np.nan,
            "TEMP_min": np.nan,
            "TEMP_std": np.nan,
        }

    return {
        "TEMP_mean": float(np.mean(temp_values)),
        "TEMP_median": float(np.median(temp_values)),
        "TEMP_max": float(np.max(temp_values)),
        "TEMP_min": float(np.min(temp_values)),
        "TEMP_std": float(np.std(temp_values)),
    }

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
            extract_bvp_features_from_epoch(epoch_data)
        )

        row.update(
            extract_acc_features_from_epoch(epoch_data)
        )

        row.update(
            extract_eda_features_from_epoch(epoch_data)
        )

        row.update(
            extract_temp_features_from_epoch(epoch_data)
        )

        rows.append(row)

    epoch_df = pd.DataFrame(rows)

    ordered_columns = [
        "subject_id",
        "epoch_id",
        "etiqueta",
        "BVP_mean",
        "BVP_median",
        "BVP_std",
        "BVP_min",
        "BVP_max",
        "BVP_range",
        "BVP_iqr",
        "BVP_mad",
        "BVP_rms",
        "BVP_skewness",
        "BVP_kurtosis",
        "HRV_SDNN",
        "HRV_RMSSD",
        "HRV_pNN50",
        "HRV_SD1",
        "HRV_SD2",
        "HRV_SD1SD2",
        "HRV_HFD",
        "HRV_KFD",
        "HRV_SampEn",
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
        "SCR_Height_mean",
        "SCR_Height_max",
        "SCR_Amplitude_mean",
        "SCR_Amplitude_max",
        "SCR_RiseTime_mean",
        "SCR_RiseTime_max",
        "SCR_RecoveryTime_mean",
        "SCR_RecoveryTime_max",
        "TEMP_mean",
        "TEMP_median",
        "TEMP_max",
        "TEMP_min",
        "TEMP_std",
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

    bvp_feature_columns = [
        column for column in epoch_df.columns
        if column.startswith("BVP_") or column.startswith("HRV_")
    ]

    acc_feature_columns = [
        column for column in epoch_df.columns
        if column.startswith("ACC_")
    ]

    eda_feature_columns = [
        column for column in epoch_df.columns
        if column.startswith("SCR_")
    ]

    temp_feature_columns = [
        column for column in epoch_df.columns
        if column.startswith("TEMP_")
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
        "num_bvp_features": len(bvp_feature_columns),
        "num_epochs_with_any_nan_bvp_feature": int(
            epoch_df[bvp_feature_columns].isna().any(axis=1).sum()
        ) if len(bvp_feature_columns) > 0 else 0,
        "num_acc_features": len(acc_feature_columns),
        "num_eda_features": len(eda_feature_columns),
        "num_epochs_with_any_nan_eda_feature": int(
            epoch_df[eda_feature_columns].isna().any(axis=1).sum()
        ) if len(eda_feature_columns) > 0 else 0,
        "num_temp_features": len(temp_feature_columns),
        "num_epochs_with_any_nan_temp_feature": int(
            epoch_df[temp_feature_columns].isna().any(axis=1).sum()
        ) if len(temp_feature_columns) > 0 else 0,
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