"""
preprocess_dreamt_bvp_hr_ibi.py

Preprocesamiento inicial de DREAMT inspirado en DREAMT_FE, pero priorizando
las decisiones de diseño propias del proyecto.

Funcionalidad:
- Recibe un CSV de entrada definido por el usuario.
- Recibe una carpeta de salida definida por el usuario.
- Filtra la señal BVP con Chebyshev tipo II, orden 4, banda 0.5-8 Hz.
- Calcula IBI y HR a partir de la BVP filtrada.
- Compara HR/IBI calculados contra HR/IBI originales de DREAMT.
- Guarda CSVs y gráficos de comparación.

Dependencias:
    pip install numpy pandas scipy matplotlib
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import cheby2, sosfiltfilt, find_peaks_cwt


# =============================================================================
# CONFIGURACIÓN DEL USUARIO
# =============================================================================

INPUT_CSV = Path(r"C:\Proyectos_compartidos\TFG\sleep_classification_with_wearables\data\prueba.csv")
OUTPUT_DIR = Path(r"C:\Proyectos_compartidos\TFG\sleep_classification_with_wearables\results\prepo2")

BVP_COLUMN = "BVP"
HR_COLUMN = "HR"
IBI_COLUMN = "IBI"
TIMESTAMP_COLUMN = "TIMESTAMP"

FS = 64.0

LOWCUT = 0.5
HIGHCUT = 8.0
FILTER_ORDER = 4
STOPBAND_ATTENUATION_DB = 20.0

KEEP_RAW_COLUMNS = True

# Parámetros inspirados en DREAMT_FE para detección de picos.
PEAK_WIDTHS = np.arange(2, 32)
MIN_PEAK_DISTANCE_SAMPLES = 12

# Rango fisiológico razonable para limpiar IBIs extremos.
MIN_IBI_SECONDS = 0.30   # 200 bpm
MAX_IBI_SECONDS = 2.00   # 30 bpm


# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def interpolate_missing_values(signal: pd.Series) -> np.ndarray:
    """
    Convierte una señal a float e interpola valores ausentes.

    Los filtros digitales no deben aplicarse sobre NaN. Por eso se hace una
    interpolación lineal y se rellenan posibles NaN en extremos.
    """
    clean_signal = (
        signal.astype(float)
        .interpolate(method="linear", limit_direction="both")
        .ffill()
        .bfill()
    )

    return clean_signal.to_numpy(dtype=float)


def cheby2_bandpass_filter(
    signal: np.ndarray,
    fs: float,
    lowcut: float,
    highcut: float,
    order: int,
    rs: float,
) -> np.ndarray:
    """
    Aplica un filtro Chebyshev tipo II pasabanda.

    Parameters
    ----------
    signal : np.ndarray
        Señal de entrada.
    fs : float
        Frecuencia de muestreo en Hz.
    lowcut : float
        Frecuencia inferior del filtro en Hz.
    highcut : float
        Frecuencia superior del filtro en Hz.
    order : int
        Orden del filtro.
    rs : float
        Atenuación mínima en banda eliminada, en dB.

    Returns
    -------
    np.ndarray
        Señal filtrada.
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


def filter_close_peaks(peaks: np.ndarray, min_distance: int) -> np.ndarray:
    """
    Elimina picos demasiado cercanos entre sí.

    Es una adaptación directa de la idea usada en DREAMT_FE: detectar picos
    candidatos y después imponer una distancia mínima entre picos sucesivos.
    """
    if len(peaks) == 0:
        return np.array([], dtype=int)

    peaks = np.asarray(peaks, dtype=int)
    peaks = np.sort(peaks)

    filtered_peaks = [peaks[0]]

    for peak in peaks[1:]:
        if peak - filtered_peaks[-1] > min_distance:
            filtered_peaks.append(peak)

    return np.asarray(filtered_peaks, dtype=int)


def calculate_ibi_hr_from_bvp(
    filtered_bvp: np.ndarray,
    fs: float,
    peak_widths: np.ndarray,
    min_peak_distance_samples: int,
    min_ibi_seconds: float,
    max_ibi_seconds: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcula IBI y HR a partir de una señal BVP filtrada.

    Estrategia:
    1. Detectar picos en BVP.
    2. Calcular IBI como diferencia temporal entre picos consecutivos.
    3. Eliminar IBI fisiológicamente imposibles.
    4. Interpolar IBI para tener un valor por muestra.
    5. Calcular HR como 60 / IBI.

    Returns
    -------
    ibi_full : np.ndarray
        IBI interpolado, en segundos, con la misma longitud que BVP.
    hr_full : np.ndarray
        HR interpolado, en bpm, con la misma longitud que BVP.
    peaks : np.ndarray
        Índices de los picos detectados.
    """
    peaks = find_peaks_cwt(filtered_bvp, peak_widths, window_size=16)
    peaks = filter_close_peaks(peaks, min_peak_distance_samples)

    if len(peaks) < 2:
        raise ValueError(
            "No se han detectado suficientes picos para calcular IBI/HR."
        )

    ibi_values = np.diff(peaks) / fs
    ibi_times = peaks[1:] / fs

    valid_mask = (
        (ibi_values >= min_ibi_seconds)
        & (ibi_values <= max_ibi_seconds)
        & np.isfinite(ibi_values)
    )

    ibi_values = ibi_values[valid_mask]
    ibi_times = ibi_times[valid_mask]

    if len(ibi_values) < 2:
        raise ValueError(
            "Tras limpiar IBIs extremos, quedan muy pocos valores válidos."
        )

    full_times = np.arange(len(filtered_bvp)) / fs

    ibi_full = np.interp(
        full_times,
        ibi_times,
        ibi_values,
        left=ibi_values[0],
        right=ibi_values[-1],
    )

    hr_full = 60.0 / ibi_full

    return ibi_full, hr_full, peaks


def build_time_axis(df: pd.DataFrame, fs: float, timestamp_column: str) -> np.ndarray:
    """
    Construye eje temporal en segundos.

    Si existe TIMESTAMP, se usa y se normaliza para empezar en 0.
    Si no existe, se crea a partir de la frecuencia de muestreo.
    """
    if timestamp_column in df.columns:
        timestamps = pd.to_numeric(df[timestamp_column], errors="coerce")
        timestamps = timestamps.interpolate(limit_direction="both").ffill().bfill()
        timestamps = timestamps.to_numpy(dtype=float)

        return timestamps - timestamps[0]

    return np.arange(len(df)) / fs


def calculate_comparison_metrics(
    dreamt_signal: pd.Series,
    calculated_signal: np.ndarray,
    name: str,
) -> dict[str, float]:
    """
    Calcula métricas básicas de comparación entre DREAMT y señal calculada.
    """
    dreamt = pd.to_numeric(dreamt_signal, errors="coerce").to_numpy(dtype=float)
    calculated = np.asarray(calculated_signal, dtype=float)

    valid_mask = np.isfinite(dreamt) & np.isfinite(calculated)

    if valid_mask.sum() == 0:
        return {
            f"{name}_MAE": np.nan,
            f"{name}_RMSE": np.nan,
            f"{name}_MEAN_DIFF": np.nan,
            f"{name}_CORR": np.nan,
        }

    dreamt_valid = dreamt[valid_mask]
    calc_valid = calculated[valid_mask]

    diff = calc_valid - dreamt_valid

    if len(dreamt_valid) > 1:
        corr = np.corrcoef(dreamt_valid, calc_valid)[0, 1]
    else:
        corr = np.nan

    return {
        f"{name}_MAE": float(np.mean(np.abs(diff))),
        f"{name}_RMSE": float(np.sqrt(np.mean(diff**2))),
        f"{name}_MEAN_DIFF": float(np.mean(diff)),
        f"{name}_CORR": float(corr),
    }


def save_comparison_plots(
    df: pd.DataFrame,
    time_seconds: np.ndarray,
    output_dir: Path,
    hr_column: str,
    ibi_column: str,
) -> None:
    """
    Guarda gráficos de comparación entre señales originales y calculadas.
    """
    # -------------------------------------------------------------------------
    # HR DREAMT vs HR calculada
    # -------------------------------------------------------------------------
    if hr_column in df.columns:
        plt.figure(figsize=(14, 5))
        plt.plot(time_seconds, df[hr_column], label="HR DREAMT", alpha=0.8)
        plt.plot(time_seconds, df["HR_calc"], label="HR calculada desde BVP filtrada", alpha=0.8)
        plt.xlabel("Tiempo (s)")
        plt.ylabel("HR (bpm)")
        plt.title("Comparación HR: DREAMT vs calculada desde BVP filtrada")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "comparacion_HR.png", dpi=300)
        plt.close()

    # -------------------------------------------------------------------------
    # IBI DREAMT vs IBI calculada
    # -------------------------------------------------------------------------
    if ibi_column in df.columns:
        plt.figure(figsize=(14, 5))
        plt.plot(time_seconds, df[ibi_column], label="IBI DREAMT", alpha=0.8)
        plt.plot(time_seconds, df["IBI_calc"], label="IBI calculada desde BVP filtrada", alpha=0.8)
        plt.xlabel("Tiempo (s)")
        plt.ylabel("IBI (s)")
        plt.title("Comparación IBI: DREAMT vs calculada desde BVP filtrada")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "comparacion_IBI.png", dpi=300)
        plt.close()

    # -------------------------------------------------------------------------
    # Error absoluto HR
    # -------------------------------------------------------------------------
    if hr_column in df.columns:
        hr_original = pd.to_numeric(df[hr_column], errors="coerce")
        hr_error = np.abs(df["HR_calc"] - hr_original)

        plt.figure(figsize=(14, 5))
        plt.plot(time_seconds, hr_error, label="|HR_calc - HR_DREAMT|")
        plt.xlabel("Tiempo (s)")
        plt.ylabel("Error absoluto HR (bpm)")
        plt.title("Error absoluto de HR")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "error_absoluto_HR.png", dpi=300)
        plt.close()

    # -------------------------------------------------------------------------
    # Error absoluto IBI
    # -------------------------------------------------------------------------
    if ibi_column in df.columns:
        ibi_original = pd.to_numeric(df[ibi_column], errors="coerce")
        ibi_error = np.abs(df["IBI_calc"] - ibi_original)

        plt.figure(figsize=(14, 5))
        plt.plot(time_seconds, ibi_error, label="|IBI_calc - IBI_DREAMT|")
        plt.xlabel("Tiempo (s)")
        plt.ylabel("Error absoluto IBI (s)")
        plt.title("Error absoluto de IBI")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "error_absoluto_IBI.png", dpi=300)
        plt.close()


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

    if BVP_COLUMN not in df.columns:
        raise ValueError(
            f"No se ha encontrado la columna '{BVP_COLUMN}'. "
            f"Columnas disponibles: {list(df.columns)}"
        )

    if HR_COLUMN not in df.columns:
        print(
            f"AVISO: no se ha encontrado la columna '{HR_COLUMN}'. "
            "No se podrá comparar HR contra DREAMT."
        )

    if IBI_COLUMN not in df.columns:
        print(
            f"AVISO: no se ha encontrado la columna '{IBI_COLUMN}'. "
            "No se podrá comparar IBI contra DREAMT."
        )

    # -------------------------------------------------------------------------
    # Guardar columnas originales si se desea
    # -------------------------------------------------------------------------
    if KEEP_RAW_COLUMNS:
        df[f"{BVP_COLUMN}_raw"] = df[BVP_COLUMN]

        if HR_COLUMN in df.columns:
            df[f"{HR_COLUMN}_DREAMT"] = df[HR_COLUMN]

        if IBI_COLUMN in df.columns:
            df[f"{IBI_COLUMN}_DREAMT"] = df[IBI_COLUMN]

    # -------------------------------------------------------------------------
    # Filtrado BVP
    # -------------------------------------------------------------------------
    raw_bvp = interpolate_missing_values(df[BVP_COLUMN])

    filtered_bvp = cheby2_bandpass_filter(
        signal=raw_bvp,
        fs=FS,
        lowcut=LOWCUT,
        highcut=HIGHCUT,
        order=FILTER_ORDER,
        rs=STOPBAND_ATTENUATION_DB,
    )

    df[BVP_COLUMN] = filtered_bvp

    # -------------------------------------------------------------------------
    # Cálculo de IBI y HR desde BVP filtrada
    # -------------------------------------------------------------------------
    ibi_calc, hr_calc, peaks = calculate_ibi_hr_from_bvp(
        filtered_bvp=filtered_bvp,
        fs=FS,
        peak_widths=PEAK_WIDTHS,
        min_peak_distance_samples=MIN_PEAK_DISTANCE_SAMPLES,
        min_ibi_seconds=MIN_IBI_SECONDS,
        max_ibi_seconds=MAX_IBI_SECONDS,
    )

    df["IBI_calc"] = ibi_calc
    df["HR_calc"] = hr_calc

    # -------------------------------------------------------------------------
    # Eje temporal
    # -------------------------------------------------------------------------
    time_seconds = build_time_axis(
        df=df,
        fs=FS,
        timestamp_column=TIMESTAMP_COLUMN,
    )

    df["time_seconds"] = time_seconds

    # -------------------------------------------------------------------------
    # Métricas de comparación
    # -------------------------------------------------------------------------
    metrics = {}

    if HR_COLUMN in df.columns:
        metrics.update(
            calculate_comparison_metrics(
                dreamt_signal=df[HR_COLUMN],
                calculated_signal=df["HR_calc"],
                name="HR",
            )
        )

    if IBI_COLUMN in df.columns:
        metrics.update(
            calculate_comparison_metrics(
                dreamt_signal=df[IBI_COLUMN],
                calculated_signal=df["IBI_calc"],
                name="IBI",
            )
        )

    metrics["num_samples"] = len(df)
    metrics["num_detected_peaks"] = len(peaks)
    metrics["estimated_duration_seconds"] = len(df) / FS

    metrics_df = pd.DataFrame([metrics])

    # -------------------------------------------------------------------------
    # Guardado de archivos
    # -------------------------------------------------------------------------
    stem = input_csv.stem

    preprocessed_csv = output_dir / f"{stem}_bvp_cheby2_hr_ibi.csv"
    comparison_csv = output_dir / f"{stem}_comparison_hr_ibi.csv"
    metrics_csv = output_dir / f"{stem}_metrics_hr_ibi.csv"

    df.to_csv(preprocessed_csv, index=False)

    comparison_columns = ["time_seconds"]

    if HR_COLUMN in df.columns:
        comparison_columns.append(HR_COLUMN)

    comparison_columns.append("HR_calc")

    if IBI_COLUMN in df.columns:
        comparison_columns.append(IBI_COLUMN)

    comparison_columns.append("IBI_calc")

    df[comparison_columns].to_csv(comparison_csv, index=False)
    metrics_df.to_csv(metrics_csv, index=False)

    # -------------------------------------------------------------------------
    # Gráficos
    # -------------------------------------------------------------------------
    save_comparison_plots(
        df=df,
        time_seconds=time_seconds,
        output_dir=output_dir,
        hr_column=HR_COLUMN,
        ibi_column=IBI_COLUMN,
    )

    # Gráfico extra: BVP filtrada con picos detectados
    plt.figure(figsize=(14, 5))
    plt.plot(time_seconds, filtered_bvp, label="BVP filtrada")
    plt.scatter(
        time_seconds[peaks],
        filtered_bvp[peaks],
        s=10,
        label="Picos detectados",
    )
    plt.xlabel("Tiempo (s)")
    plt.ylabel("BVP filtrada")
    plt.title("BVP filtrada con picos detectados")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "bvp_filtrada_picos_detectados.png", dpi=300)
    plt.close()

    print("Preprocesamiento completado.")
    print(f"CSV preprocesado: {preprocessed_csv}")
    print(f"CSV comparación:   {comparison_csv}")
    print(f"Métricas:          {metrics_csv}")
    print(f"Gráficos guardados en: {output_dir}")
    print()
    print("Métricas:")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()