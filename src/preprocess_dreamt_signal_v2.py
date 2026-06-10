"""
preprocess_dreamt_signals.py

Preprocesamiento de señales DREAMT basado en el flujo general de DREAMT_FE,
pero priorizando las decisiones de diseño propias del proyecto.

Funcionalidad actual:
- Lee un CSV correspondiente a un sujeto.
- Filtra BVP con Chebyshev tipo II, orden 4, banda 0.5-8 Hz.
- Preprocesa HR:
    1. Fuerza rango fisiológico 30-220 bpm.
    2. Elimina outliers.
    3. Interpola valores eliminados.
    4. Normaliza por sujeto mediante z-score.
- Preprocesa IBI:
    1. Fuerza rango fisiológico 0.3-2.0 s.
    2. Elimina outliers.
    3. Interpola valores eliminados.
- Preprocesa acelerometría:
    1. Filtra ACC_X, ACC_Y y ACC_Z con Butterworth pasabanda.
    2. Orden 3.
    3. Banda 3-10 Hz.
    4. Frecuencia de muestreo 32 Hz.
    5. Genera una gráfica original vs filtrada para el eje que más filtrado sufre.
- Guarda un nuevo CSV preprocesado.

Dependencias:
    pip install numpy pandas scipy
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import cheby2, sosfiltfilt, butter
import matplotlib.pyplot as plt



# =============================================================================
# CONFIGURACIÓN DEL USUARIO
# =============================================================================

INPUT_CSV = Path(r"C:\Proyectos_compartidos\TFG\sleep_classification_with_wearables\data\prueba.csv")
OUTPUT_DIR = Path(r"C:\Proyectos_compartidos\TFG\sleep_classification_with_wearables\results\prepo")


BVP_COLUMN = "BVP"
HR_COLUMN = "HR"
IBI_COLUMN = "IBI"

# En algunos CSV las columnas pueden venir como "ACC X" en vez de "ACC_X".
# Aquí puedes adaptar los nombres a tu CSV.
ACC_COLUMNS = ["ACC_X", "ACC_Y", "ACC_Z"]

# Si tu CSV usa espacios, cambia la línea anterior por:
# ACC_COLUMNS = ["ACC X", "ACC Y", "ACC Z"]

TIMESTAMP_COLUMN = "TIMESTAMP"

FS_BVP = 64.0

# Filtro BVP
BVP_LOWCUT = 0.5
BVP_HIGHCUT = 8.0
BVP_FILTER_ORDER = 4
BVP_STOPBAND_ATTENUATION_DB = 20.0

# Rango fisiológico HR
HR_MIN = 30.0
HR_MAX = 220.0

# Rango fisiológico IBI
IBI_MIN = 0.30
IBI_MAX = 2.00

# ACC
FS_ACC = 32.0
ACC_LOWCUT = 3.0
ACC_HIGHCUT = 10.0
ACC_FILTER_ORDER = 3

# Método de outliers
OUTLIER_METHOD = "iqr"  # opciones: "iqr" o "zscore"

# Parámetros IQR
IQR_FACTOR = 1.5

# Parámetros z-score
ZSCORE_THRESHOLD = 3.0

# Guardar columnas originales para trazabilidad
KEEP_RAW_COLUMNS = False


# =============================================================================
# FUNCIONES GENERALES
# =============================================================================

def interpolate_missing_values(signal: pd.Series) -> pd.Series:
    """
    Interpola valores ausentes de una señal.

    Primero convierte la señal a numérica, forzando a NaN los valores no válidos.
    Después interpola linealmente y rellena extremos.
    """
    clean_signal = pd.to_numeric(signal, errors="coerce")

    clean_signal = (
        clean_signal
        .interpolate(method="linear", limit_direction="both")
        .ffill()
        .bfill()
    )

    return clean_signal


def remove_outliers_iqr(signal: pd.Series, factor: float = 1.5) -> pd.Series:
    """
    Elimina outliers usando el criterio del rango intercuartílico.

    Un valor se considera outlier si está fuera de:

        [Q1 - factor * IQR, Q3 + factor * IQR]

    Los outliers no se eliminan como filas, sino que se sustituyen por NaN
    para poder interpolarlos después.
    """
    clean_signal = signal.copy()

    q1 = clean_signal.quantile(0.25)
    q3 = clean_signal.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr

    outlier_mask = (
        (clean_signal < lower_bound)
        | (clean_signal > upper_bound)
    )

    clean_signal[outlier_mask] = np.nan

    return clean_signal


def remove_outliers_zscore(
    signal: pd.Series,
    threshold: float = 3.0,
) -> pd.Series:
    """
    Elimina outliers usando z-score.

    Los valores con |z| > threshold se sustituyen por NaN.
    """
    clean_signal = signal.copy()

    mean = clean_signal.mean(skipna=True)
    std = clean_signal.std(skipna=True)

    if std == 0 or np.isnan(std):
        return clean_signal

    z_scores = (clean_signal - mean) / std

    outlier_mask = np.abs(z_scores) > threshold
    clean_signal[outlier_mask] = np.nan

    return clean_signal


def remove_outliers(
    signal: pd.Series,
    method: str = "iqr",
    iqr_factor: float = 1.5,
    zscore_threshold: float = 3.0,
) -> pd.Series:
    """
    Aplica el método de eliminación de outliers seleccionado.
    """
    if method == "iqr":
        return remove_outliers_iqr(signal, factor=iqr_factor)

    if method == "zscore":
        return remove_outliers_zscore(signal, threshold=zscore_threshold)

    raise ValueError(
        f"Método de outliers no reconocido: {method}. "
        "Usa 'iqr' o 'zscore'."
    )


def apply_valid_range(
    signal: pd.Series,
    min_value: float,
    max_value: float,
) -> pd.Series:
    """
    Sustituye por NaN los valores fuera de un rango fisiológico válido.
    """
    clean_signal = pd.to_numeric(signal, errors="coerce")

    invalid_mask = (
        (clean_signal < min_value)
        | (clean_signal > max_value)
    )

    clean_signal[invalid_mask] = np.nan

    return clean_signal


def zscore_normalize_per_subject(signal: pd.Series) -> pd.Series:
    """
    Normaliza una señal por sujeto usando z-score.

    Como cada CSV corresponde a un sujeto, la media y desviación típica se
    calculan directamente sobre la señal del CSV:

        x_norm = (x - media_sujeto) / std_sujeto
    """
    mean = signal.mean(skipna=True)
    std = signal.std(skipna=True)

    if std == 0 or np.isnan(std):
        print(
            "AVISO: la desviación típica de HR es 0 o NaN. "
            "Se devuelve HR centrada en 0."
        )
        return signal - mean

    return (signal - mean) / std


def build_time_axis(df: pd.DataFrame, fs: float) -> np.ndarray:
    """
    Construye un eje temporal en segundos para las gráficas.

    Si existe TIMESTAMP, lo usa y lo normaliza para empezar en 0.
    Si no existe, crea un eje temporal artificial.
    """
    if TIMESTAMP_COLUMN in df.columns:
        timestamps = pd.to_numeric(df[TIMESTAMP_COLUMN], errors="coerce")
        timestamps = (
            timestamps
            .interpolate(method="linear", limit_direction="both")
            .ffill()
            .bfill()
        )
        timestamps = timestamps.to_numpy(dtype=float)

        return timestamps - timestamps[0]

    return np.arange(len(df)) / fs

# =============================================================================
# PREPROCESAMIENTO BVP
# =============================================================================

def cheby2_bandpass_filter(
    signal: np.ndarray,
    fs: float,
    lowcut: float,
    highcut: float,
    order: int,
    rs: float,
) -> np.ndarray:
    """
    Aplica un filtro Chebyshev tipo II pasabanda a BVP.
    """
    nyquist = fs / 2.0

    if lowcut <= 0:
        raise ValueError("lowcut debe ser mayor que 0 Hz.")

    if highcut >= nyquist:
        raise ValueError(
            f"highcut debe ser menor que Nyquist. "
            f"fs={fs} Hz, Nyquist={nyquist} Hz."
        )

    if lowcut >= highcut:
        raise ValueError("lowcut debe ser menor que highcut.")

    sos = cheby2(
        N=order,
        rs=rs,
        Wn=[lowcut, highcut],
        btype="bandpass",
        fs=fs,
        output="sos",
    )

    return sosfiltfilt(sos, signal)


def preprocess_bvp(df: pd.DataFrame) -> pd.Series:
    """
    Preprocesa la columna BVP:
    - Conversión a numérico.
    - Interpolación de NaN.
    - Filtro Chebyshev tipo II pasabanda 0.5-8 Hz.
    """
    bvp_clean = interpolate_missing_values(df[BVP_COLUMN])

    bvp_filtered = cheby2_bandpass_filter(
        signal=bvp_clean.to_numpy(dtype=float),
        fs=FS_BVP,
        lowcut=BVP_LOWCUT,
        highcut=BVP_HIGHCUT,
        order=BVP_FILTER_ORDER,
        rs=BVP_STOPBAND_ATTENUATION_DB,
    )

    return pd.Series(bvp_filtered, index=df.index, name=BVP_COLUMN)


# =============================================================================
# PREPROCESAMIENTO HR
# =============================================================================

def preprocess_hr(df: pd.DataFrame) -> pd.Series:
    """
    Preprocesa HR:
    - Fuerza rango 30-220 bpm.
    - Elimina outliers.
    - Interpola valores eliminados.
    - Normaliza por sujeto mediante z-score.
    """
    hr = apply_valid_range(
        signal=df[HR_COLUMN],
        min_value=HR_MIN,
        max_value=HR_MAX,
    )

    hr = remove_outliers(
        signal=hr,
        method=OUTLIER_METHOD,
        iqr_factor=IQR_FACTOR,
        zscore_threshold=ZSCORE_THRESHOLD,
    )

    hr = interpolate_missing_values(hr)

    hr_normalized = zscore_normalize_per_subject(hr)

    return pd.Series(hr_normalized, index=df.index, name=HR_COLUMN)


# =============================================================================
# PREPROCESAMIENTO IBI
# =============================================================================

def preprocess_ibi(df: pd.DataFrame) -> pd.Series:
    """
    Preprocesa IBI:
    - Fuerza rango 0.3-2.0 s.
    - Elimina outliers.
    - Interpola valores eliminados.

    De momento no se normaliza IBI porque no lo has indicado como decisión de
    diseño. Se conserva en segundos.
    """
    ibi = apply_valid_range(
        signal=df[IBI_COLUMN],
        min_value=IBI_MIN,
        max_value=IBI_MAX,
    )

    ibi = remove_outliers(
        signal=ibi,
        method=OUTLIER_METHOD,
        iqr_factor=IQR_FACTOR,
        zscore_threshold=ZSCORE_THRESHOLD,
    )

    ibi = interpolate_missing_values(ibi)

    return pd.Series(ibi, index=df.index, name=IBI_COLUMN)


# =============================================================================
# PREPROCESAMIENTO ACC
# =============================================================================


def butter_bandpass_filter(
    signal: np.ndarray,
    fs: float,
    lowcut: float,
    highcut: float,
    order: int,
) -> np.ndarray:
    """
    Aplica un filtro Butterworth pasabanda.
    """
    nyquist = fs / 2.0

    if lowcut <= 0:
        raise ValueError("lowcut debe ser mayor que 0 Hz.")

    if highcut >= nyquist:
        raise ValueError(
            f"highcut debe ser menor que Nyquist. "
            f"fs={fs} Hz, Nyquist={nyquist} Hz."
        )

    if lowcut >= highcut:
        raise ValueError("lowcut debe ser menor que highcut.")

    sos = butter(
        N=order,
        Wn=[lowcut, highcut],
        btype="bandpass",
        fs=fs,
        output="sos",
    )

    return sosfiltfilt(sos, signal)


def preprocess_single_acc_axis(df: pd.DataFrame, axis_column: str) -> pd.Series:
    """
    Preprocesa un eje de acelerometría:
    - Interpolación de NaN.
    - Filtro Butterworth pasabanda 3-10 Hz, orden 3, fs=32 Hz.
    """
    acc_clean = interpolate_missing_values(df[axis_column])

    acc_filtered = butter_bandpass_filter(
        signal=acc_clean.to_numpy(dtype=float),
        fs=FS_ACC,
        lowcut=ACC_LOWCUT,
        highcut=ACC_HIGHCUT,
        order=ACC_FILTER_ORDER,
    )

    return pd.Series(acc_filtered, index=df.index, name=axis_column)


def preprocess_accelerometry(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float], str]:
    """
    Preprocesa los tres ejes de acelerometría.

    Returns
    -------
    df : pd.DataFrame
        DataFrame con los ejes ACC sustituidos por los valores filtrados.
    filtering_scores : dict[str, float]
        MAE entre señal original y filtrada para cada eje.
    most_filtered_axis : str
        Eje con mayor MAE original-filtrada.
    """
    filtering_scores = {}

    for axis_column in ACC_COLUMNS:
        original_axis = pd.to_numeric(df[axis_column], errors="coerce")

        filtered_axis = preprocess_single_acc_axis(
            df=df,
            axis_column=axis_column,
        )

        valid_mask = (
            original_axis.notna()
            & filtered_axis.notna()
        )

        mae_filtering = np.mean(
            np.abs(
                filtered_axis.loc[valid_mask].to_numpy(dtype=float)
                - original_axis.loc[valid_mask].to_numpy(dtype=float)
            )
        )

        filtering_scores[axis_column] = float(mae_filtering)

        df[axis_column] = filtered_axis

    most_filtered_axis = max(filtering_scores, key=filtering_scores.get)

    return df, filtering_scores, most_filtered_axis


def save_most_filtered_acc_plot(
    original_df: pd.DataFrame,
    processed_df: pd.DataFrame,
    most_filtered_axis: str,
    output_dir: Path,
) -> None:
    """
    Guarda una gráfica comparando señal original y filtrada para el eje ACC
    que más filtración ha sufrido.

    Importante:
    El filtro pasabanda Butterworth elimina la componente DC y centra la señal
    filtrada alrededor de 0. Por eso, para que la comparación visual sea útil,
    la gráfica no muestra directamente la escala absoluta original, sino una
    versión centrada y reescalada de ambas señales.

    El CSV sigue guardando la señal filtrada real, sin esta normalización visual.
    """
    time_seconds = build_time_axis(original_df, fs=FS_ACC)

    original_signal = pd.to_numeric(
        original_df[most_filtered_axis],
        errors="coerce",
    )
    original_signal = interpolate_missing_values(original_signal)

    filtered_signal = pd.to_numeric(
        processed_df[most_filtered_axis],
        errors="coerce",
    )
    filtered_signal = interpolate_missing_values(filtered_signal)

    # -------------------------------------------------------------------------
    # Normalización solo para visualización
    # -------------------------------------------------------------------------
    original_centered = original_signal - original_signal.mean()
    filtered_centered = filtered_signal - filtered_signal.mean()

    original_std = original_centered.std()
    filtered_std = filtered_centered.std()

    if original_std != 0 and not np.isnan(original_std):
        original_plot = original_centered / original_std
    else:
        original_plot = original_centered

    if filtered_std != 0 and not np.isnan(filtered_std):
        filtered_plot = filtered_centered / filtered_std
    else:
        filtered_plot = filtered_centered

    # -------------------------------------------------------------------------
    # Gráfica comparativa
    # -------------------------------------------------------------------------
    plt.figure(figsize=(14, 5))

    plt.plot(
        time_seconds,
        original_plot,
        label=f"{most_filtered_axis} original centrada y reescalada",
        alpha=0.75,
    )

    plt.plot(
        time_seconds,
        filtered_plot,
        label=f"{most_filtered_axis} filtrada centrada y reescalada",
        alpha=0.85,
    )

    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud normalizada para visualización")
    plt.title(
        f"Comparación acelerometría: {most_filtered_axis} original vs filtrada"
    )
    plt.legend()
    plt.tight_layout()

    output_path = output_dir / "comparacion_acc_eje_mas_filtrado.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    
# =============================================================================
# MÉTRICAS DE CONTROL
# =============================================================================

def count_invalid_range_values(
    signal: pd.Series,
    min_value: float,
    max_value: float,
) -> int:
    """
    Cuenta valores fuera de rango antes de preprocesar.
    """
    numeric_signal = pd.to_numeric(signal, errors="coerce")

    invalid_mask = (
        (numeric_signal < min_value)
        | (numeric_signal > max_value)
        | numeric_signal.isna()
    )

    return int(invalid_mask.sum())


def build_preprocessing_report(
    original_df: pd.DataFrame,
    processed_df: pd.DataFrame,
    acc_filtering_scores: dict[str, float],
    most_filtered_axis: str,
) -> pd.DataFrame:
    """
    Crea un pequeño informe con métricas básicas del preprocesamiento.
    """
    report = {}

    if BVP_COLUMN in original_df.columns:
        report["BVP_original_nan"] = int(
            pd.to_numeric(original_df[BVP_COLUMN], errors="coerce").isna().sum()
        )
        report["BVP_processed_nan"] = int(
            pd.to_numeric(processed_df[BVP_COLUMN], errors="coerce").isna().sum()
        )

    if HR_COLUMN in original_df.columns:
        report["HR_original_invalid_or_nan"] = count_invalid_range_values(
            original_df[HR_COLUMN],
            HR_MIN,
            HR_MAX,
        )
        report["HR_processed_nan"] = int(
            pd.to_numeric(processed_df[HR_COLUMN], errors="coerce").isna().sum()
        )
        report["HR_processed_mean"] = float(processed_df[HR_COLUMN].mean())
        report["HR_processed_std"] = float(processed_df[HR_COLUMN].std())

    if IBI_COLUMN in original_df.columns:
        report["IBI_original_invalid_or_nan"] = count_invalid_range_values(
            original_df[IBI_COLUMN],
            IBI_MIN,
            IBI_MAX,
        )
        report["IBI_processed_nan"] = int(
            pd.to_numeric(processed_df[IBI_COLUMN], errors="coerce").isna().sum()
        )
        report["IBI_processed_mean_seconds"] = float(processed_df[IBI_COLUMN].mean())
        report["IBI_processed_std_seconds"] = float(processed_df[IBI_COLUMN].std())

    for axis_column, score in acc_filtering_scores.items():
        report[f"{axis_column}_filtering_MAE"] = score

    report["ACC_most_filtered_axis"] = most_filtered_axis
    report["num_samples"] = len(original_df)

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

    required_columns = [BVP_COLUMN, HR_COLUMN, IBI_COLUMN, *ACC_COLUMNS]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Faltan columnas necesarias en el CSV: {missing_columns}. "
            f"Columnas disponibles: {list(df.columns)}"
        )

    if KEEP_RAW_COLUMNS:
        df[f"{BVP_COLUMN}_raw"] = df[BVP_COLUMN]
        df[f"{HR_COLUMN}_raw"] = df[HR_COLUMN]
        df[f"{IBI_COLUMN}_raw"] = df[IBI_COLUMN]

        for axis_column in ACC_COLUMNS:
            df[f"{axis_column}_raw"] = df[axis_column]

    df[BVP_COLUMN] = preprocess_bvp(df)
    df[HR_COLUMN] = preprocess_hr(df)
    df[IBI_COLUMN] = preprocess_ibi(df)

    df, acc_filtering_scores, most_filtered_axis = preprocess_accelerometry(df)

    save_most_filtered_acc_plot(
        original_df=original_df,
        processed_df=df,
        most_filtered_axis=most_filtered_axis,
        output_dir=output_dir,
    )

    stem = input_csv.stem

    output_csv = output_dir / f"{stem}_preprocessed.csv"
    report_csv = output_dir / f"{stem}_preprocessing_report.csv"

    df.to_csv(output_csv, index=False)

    report_df = build_preprocessing_report(
        original_df=original_df,
        processed_df=df,
        acc_filtering_scores=acc_filtering_scores,
        most_filtered_axis=most_filtered_axis,
    )

    report_df.to_csv(report_csv, index=False)

    print("Preprocesamiento completado.")
    print(f"CSV preprocesado: {output_csv}")
    print(f"Informe:          {report_csv}")
    print(f"Gráfica ACC:      {output_dir / 'comparacion_acc_eje_mas_filtrado.png'}")
    print()
    print("Resumen:")
    print(report_df.to_string(index=False))


if __name__ == "__main__":
    main()