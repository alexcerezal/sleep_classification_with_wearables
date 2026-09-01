import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import cheby2, sosfiltfilt, find_peaks, butter
from scipy.interpolate import interp1d


# ============================================================
# Utilidades de seguridad
# ============================================================

def find_column(df, candidates):
    """
    Busca una columna por nombres candidatos ignorando mayúsculas/minúsculas.
    """
    lower_map = {c.lower(): c for c in df.columns}

    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    for col in df.columns:
        col_lower = col.lower()
        for candidate in candidates:
            if candidate.lower() in col_lower:
                return col

    return None


def ensure_numeric(series, name):
    """
    Convierte una columna a numérica y avisa si hay valores no convertibles.
    """
    numeric = pd.to_numeric(series, errors="coerce")
    n_bad = numeric.isna().sum()

    if n_bad > 0:
        print(f"[AVISO] Columna {name}: {n_bad} valores NaN/no numéricos.")

    return numeric.to_numpy(dtype=float)


def infer_time_axis(df, fs, time_col=None):
    """
    Crea eje temporal. Si hay columna temporal válida, la usa.
    Si no, asume muestreo uniforme a fs Hz.
    """
    if time_col is not None and time_col in df.columns:
        t = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=float)

        if np.isfinite(t).sum() < len(t) * 0.8:
            print("[AVISO] La columna temporal tiene demasiados valores inválidos. Uso índice/fs.")
            return np.arange(len(df)) / fs

        # Si parece timestamp grande, lo normalizamos a empezar en 0
        t = t - np.nanmin(t)

        # Seguridad: comprobar monotonicidad
        finite_t = t[np.isfinite(t)]
        if len(finite_t) < 2 or np.any(np.diff(finite_t) < 0):
            print("[AVISO] La columna temporal no es monótona. Uso índice/fs.")
            return np.arange(len(df)) / fs

        return t

    return np.arange(len(df)) / fs


def validate_bvp_signal(bvp, fs):
    """
    Comprueba que la señal BVP tenga longitud suficiente y valores válidos.
    """
    if bvp is None:
        raise ValueError("No se encontró columna BVP.")

    if len(bvp) < fs * 10:
        raise ValueError(
            f"La señal es demasiado corta: {len(bvp)} muestras. "
            f"Con fs={fs} Hz son solo {len(bvp)/fs:.2f} segundos."
        )

    finite_ratio = np.isfinite(bvp).mean()
    if finite_ratio < 0.8:
        raise ValueError(
            f"Demasiados valores inválidos en BVP: solo {finite_ratio:.1%} son finitos."
        )

    if np.nanstd(bvp) == 0:
        raise ValueError("La señal BVP parece constante. No se puede filtrar ni detectar picos.")

    return True


def interpolate_nans(x):
    """
    Interpola NaNs para poder filtrar. No inventa señal si casi todo está vacío.
    """
    x = np.asarray(x, dtype=float)
    idx = np.arange(len(x))
    valid = np.isfinite(x)

    if valid.sum() < 2:
        raise ValueError("No hay suficientes valores válidos para interpolar.")

    if valid.sum() < len(x):
        print(f"[INFO] Interpolando {len(x) - valid.sum()} valores inválidos en BVP.")

    x_interp = x.copy()
    x_interp[~valid] = np.interp(idx[~valid], idx[valid], x[valid])

    return x_interp


# ============================================================
# Filtrado Chebyshev II
# ============================================================

def filter_bvp_cheby2(
    bvp,
    fs=64.0,
    lowcut=0.5,
    highcut=8.0,
    order=4,
    rs=40.0
):
    """
    Aplica filtro pasabanda Chebyshev tipo II con fase cero.

    Parámetros:
    - lowcut: frecuencia mínima en Hz.
    - highcut: frecuencia máxima en Hz.
    - order: orden del filtro.
    - rs: atenuación mínima en banda eliminada, en dB.
    """
    if fs <= 0:
        raise ValueError("fs debe ser positivo.")

    nyquist = fs / 2.0

    if not (0 < lowcut < highcut < nyquist):
        raise ValueError(
            f"Frecuencias inválidas: lowcut={lowcut}, highcut={highcut}, "
            f"Nyquist={nyquist}. Debe cumplirse 0 < lowcut < highcut < Nyquist."
        )

    if order < 1:
        raise ValueError("El orden del filtro debe ser >= 1.")

    if rs <= 0:
        raise ValueError("rs debe ser positivo.")

    bvp_clean = interpolate_nans(bvp)

    sos = cheby2(
        N=order,
        rs=rs,
        Wn=[lowcut, highcut],
        btype="bandpass",
        fs=fs,
        output="sos"
    )

    # sosfiltfilt aplica fase cero, evitando desplazar los picos.
    bvp_filtered = sosfiltfilt(sos, bvp_clean)

    if not np.all(np.isfinite(bvp_filtered)):
        raise ValueError("El filtrado produjo valores no finitos.")

    return bvp_filtered


def filter_bvp_butterworth(
    bvp,
    fs=64.0,
    lowcut=0.5,
    highcut=8.0,
    order=4
):
    """
    Aplica filtro pasabanda Butterworth con fase cero.

    A diferencia de Chebyshev II, Butterworth no introduce ripple ni en la
    banda de paso ni en la banda de rechazo, pero su transición suele ser
    menos abrupta.
    """
    if fs <= 0:
        raise ValueError("fs debe ser positivo.")

    nyquist = fs / 2.0

    if not (0 < lowcut < highcut < nyquist):
        raise ValueError(
            f"Frecuencias inválidas: lowcut={lowcut}, highcut={highcut}, "
            f"Nyquist={nyquist}. Debe cumplirse 0 < lowcut < highcut < Nyquist."
        )

    if order < 1:
        raise ValueError("El orden del filtro debe ser >= 1.")

    bvp_clean = interpolate_nans(bvp)

    sos = butter(
        N=order,
        Wn=[lowcut, highcut],
        btype="bandpass",
        fs=fs,
        output="sos"
    )

    bvp_filtered = sosfiltfilt(sos, bvp_clean)

    if not np.all(np.isfinite(bvp_filtered)):
        raise ValueError("El filtrado Butterworth produjo valores no finitos.")

    return bvp_filtered

# ============================================================
# Detección de picos y cálculo de IBI/HR
# ============================================================

def detect_bvp_peaks(
    bvp_filtered,
    fs=64.0,
    min_hr_bpm=35,
    max_hr_bpm=220,
    prominence_factor=0.4
):
    """
    Detecta picos sistólicos aproximados en BVP filtrada.

    La distancia mínima entre picos se calcula desde max_hr_bpm.
    Por ejemplo, 220 bpm -> 60/220 s entre latidos.
    """
    if min_hr_bpm <= 0 or max_hr_bpm <= 0 or min_hr_bpm >= max_hr_bpm:
        raise ValueError("Rango HR inválido.")

    # Distancia mínima en muestras entre dos picos.
    min_distance_seconds = 60.0 / max_hr_bpm
    min_distance_samples = int(np.floor(min_distance_seconds * fs))

    if min_distance_samples < 1:
        min_distance_samples = 1

    # Prominence adaptativa basada en la variabilidad de la señal.
    signal_std = np.nanstd(bvp_filtered)
    prominence = prominence_factor * signal_std

    if prominence <= 0:
        raise ValueError("Prominencia inválida. La señal filtrada parece degenerada.")

    peaks, properties = find_peaks(
        bvp_filtered,
        distance=min_distance_samples,
        prominence=prominence
    )

    if len(peaks) < 3:
        warnings.warn(
            "Se han detectado muy pocos picos. Puede que el fragmento sea corto, "
            "ruidoso o que haya que ajustar prominence_factor / banda del filtro."
        )

    peak_times = peaks / fs

    return peaks, peak_times, properties


def compute_ibi_hr_from_peaks(
    peak_times,
    min_hr_bpm=35,
    max_hr_bpm=220
):
    """
    Calcula IBI y HR derivados de los tiempos de picos.
    Filtra valores fisiológicamente improbables.
    """
    peak_times = np.asarray(peak_times, dtype=float)

    if len(peak_times) < 2:
        return pd.DataFrame(columns=["time_s", "ibi_s", "ibi_ms", "hr_bpm", "valid"])

    ibi_s = np.diff(peak_times)
    ibi_ms = ibi_s * 1000.0
    hr_bpm = 60.0 / ibi_s

    # Tiempo asociado a cada intervalo: punto medio entre dos picos.
    interval_times = (peak_times[:-1] + peak_times[1:]) / 2.0

    valid = (
        np.isfinite(ibi_s)
        & np.isfinite(hr_bpm)
        & (hr_bpm >= min_hr_bpm)
        & (hr_bpm <= max_hr_bpm)
    )

    ibi_df = pd.DataFrame({
        "time_s": interval_times,
        "ibi_s": ibi_s,
        "ibi_ms": ibi_ms,
        "hr_bpm": hr_bpm,
        "valid": valid
    })

    n_invalid = (~valid).sum()
    if n_invalid > 0:
        print(f"[AVISO] {n_invalid} intervalos IBI/HR fuera del rango fisiológico.")

    return ibi_df


# ============================================================
# Comparación con HR e IBI del CSV
# ============================================================

def normalize_ibi_units(ibi_values):
    """
    Intenta inferir si el IBI del CSV está en segundos o milisegundos.
    Empatica suele guardar IBI en segundos, pero otros CSV pueden usar ms.
    """
    ibi_values = np.asarray(ibi_values, dtype=float)
    valid = ibi_values[np.isfinite(ibi_values) & (ibi_values > 0)]

    if len(valid) == 0:
        return ibi_values, "unknown"

    median_val = np.nanmedian(valid)

    # Si la mediana está por encima de 10, probablemente está en ms.
    if median_val > 10:
        return ibi_values / 1000.0, "ms"

    return ibi_values, "s"


def compare_series(reference_time, reference_values, derived_time, derived_values, name):
    """
    Interpola la serie derivada sobre los tiempos de referencia y calcula métricas.
    """
    reference_time = np.asarray(reference_time, dtype=float)
    reference_values = np.asarray(reference_values, dtype=float)
    derived_time = np.asarray(derived_time, dtype=float)
    derived_values = np.asarray(derived_values, dtype=float)

    mask_ref = np.isfinite(reference_time) & np.isfinite(reference_values)
    mask_der = np.isfinite(derived_time) & np.isfinite(derived_values)

    reference_time = reference_time[mask_ref]
    reference_values = reference_values[mask_ref]
    derived_time = derived_time[mask_der]
    derived_values = derived_values[mask_der]

    if len(reference_time) < 3 or len(derived_time) < 3:
        print(f"[AVISO] No hay suficientes datos para comparar {name}.")
        return None

    # Limitar a rango temporal común.
    t_min = max(np.min(reference_time), np.min(derived_time))
    t_max = min(np.max(reference_time), np.max(derived_time))

    common = (reference_time >= t_min) & (reference_time <= t_max)

    if common.sum() < 3:
        print(f"[AVISO] Poco solapamiento temporal para comparar {name}.")
        return None

    reference_time_common = reference_time[common]
    reference_values_common = reference_values[common]

    interpolator = interp1d(
        derived_time,
        derived_values,
        kind="linear",
        bounds_error=False,
        fill_value=np.nan
    )

    derived_interp = interpolator(reference_time_common)

    valid = np.isfinite(reference_values_common) & np.isfinite(derived_interp)

    if valid.sum() < 3:
        print(f"[AVISO] No hay suficientes puntos válidos tras interpolar {name}.")
        return None

    ref = reference_values_common[valid]
    der = derived_interp[valid]

    diff = der - ref

    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(diff ** 2))

    if np.std(ref) > 0 and np.std(der) > 0:
        corr = np.corrcoef(ref, der)[0, 1]
    else:
        corr = np.nan

    result = {
        "name": name,
        "n": len(ref),
        "mae": mae,
        "rmse": rmse,
        "corr": corr,
        "reference_time": reference_time_common[valid],
        "reference_values": ref,
        "derived_values": der
    }

    print(f"\nComparación {name}")
    print("-" * 40)
    print(f"N puntos comparados: {result['n']}")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"Corr: {corr:.4f}")

    return result


# ============================================================
# Gráficos
# ============================================================
def plot_bvp_original_vs_filtered(
    time_s,
    bvp_original,
    bvp_cheby2,
    bvp_butter,
    output_path,
    max_seconds=60
):
    """
    Grafica BVP original, BVP filtrada con Chebyshev II y BVP filtrada con Butterworth.
    Para legibilidad, muestra solo los primeros max_seconds.
    """
    time_s = np.asarray(time_s)

    mask = time_s <= min(max_seconds, np.nanmax(time_s))

    plt.figure(figsize=(14, 5))

    plt.plot(
        time_s[mask],
        bvp_original[mask],
        label="BVP original",
        linewidth=1,
        alpha=0.55
    )

    plt.plot(
        time_s[mask],
        bvp_cheby2[mask],
        label="BVP filtrada Chebyshev II",
        linewidth=1.2
    )

    plt.plot(
        time_s[mask],
        bvp_butter[mask],
        label="BVP filtrada Butterworth",
        linewidth=1.2,
        alpha=0.85
    )

    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud BVP")
    plt.title("Comparación BVP original vs Chebyshev II vs Butterworth")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"[OK] Gráfico BVP guardado en: {output_path}")


def plot_hr_comparison(hr_comparison, output_path):
    """
    Grafica HR del CSV frente a HR derivada desde BVP filtrada.
    """
    if hr_comparison is None:
        return

    t = hr_comparison["reference_time"]
    ref = hr_comparison["reference_values"]
    der = hr_comparison["derived_values"]

    plt.figure(figsize=(14, 5))
    plt.plot(t, ref, label="HR CSV", linewidth=1.2)
    plt.plot(t, der, label="HR derivada de BVP filtrada", linewidth=1.2, alpha=0.8)
    plt.xlabel("Tiempo (s)")
    plt.ylabel("HR (bpm)")
    plt.title("Comparación HR CSV vs HR derivada")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"[OK] Gráfico HR guardado en: {output_path}")


def plot_ibi_comparison(ibi_comparison, output_path):
    """
    Grafica IBI del CSV frente a IBI derivado desde BVP filtrada.
    """
    if ibi_comparison is None:
        return

    t = ibi_comparison["reference_time"]
    ref = ibi_comparison["reference_values"]
    der = ibi_comparison["derived_values"]

    plt.figure(figsize=(14, 5))
    plt.plot(t, ref, label="IBI CSV", linewidth=1.2)
    plt.plot(t, der, label="IBI derivado de BVP filtrada", linewidth=1.2, alpha=0.8)
    plt.xlabel("Tiempo (s)")
    plt.ylabel("IBI (s)")
    plt.title("Comparación IBI CSV vs IBI derivado")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"[OK] Gráfico IBI guardado en: {output_path}")


# ============================================================
# Pipeline principal
# ============================================================
def main():
    # ============================================================
    # CONFIGURACIÓN MANUAL
    # ============================================================

    csv_path = Path(r"C:\Proyectos_compartidos\TFG\sleep_classification_with_wearables\data\S002.csv")

    output_dir = Path(r"C:\Proyectos_compartidos\TFG\sleep_classification_with_wearables\results\cheb")

    # Frecuencia de muestreo de BVP en DREAMT carpeta 64Hz
    fs = 64.0

    # Si quieres forzar nombres exactos de columnas, escríbelos aquí.
    # Si los dejas en None, el programa intentará detectarlos automáticamente.
    bvp_col_manual = None
    hr_col_manual = None
    ibi_col_manual = None
    time_col_manual = None

    # Parámetros del filtro Chebyshev II
    lowcut = 0.5
    highcut = 8.0
    order = 4
    rs = 40.0

    # Rango fisiológico aceptado para HR/IBI
    min_hr = 35.0
    max_hr = 220.0

    # Parámetro para detección de picos
    prominence_factor = 0.4

    # Cuántos segundos mostrar en el gráfico BVP original vs filtrada
    plot_seconds = 60.0

    # ============================================================
    # PIPELINE
    # ============================================================

    output_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(f"No existe el archivo: {csv_path}")

    print(f"[INFO] Leyendo CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    if df.empty:
        raise ValueError("El CSV está vacío.")

    print(f"[INFO] Columnas encontradas: {list(df.columns)}")
    print(f"[INFO] Número de filas: {len(df)}")

    # Detección automática de columnas
    bvp_col = bvp_col_manual or find_column(
        df,
        ["BVP", "bvp", "BloodVolumePulse", "blood_volume_pulse"]
    )

    hr_col = hr_col_manual or find_column(
        df,
        ["HR", "hr", "heart_rate", "HeartRate"]
    )

    ibi_col = ibi_col_manual or find_column(
        df,
        ["IBI", "ibi", "interbeat", "inter_beat_interval", "RR", "rr"]
    )

    time_col = time_col_manual or find_column(
        df,
        ["time", "timestamp", "Time", "Timestamp", "seconds", "t"]
    )

    if bvp_col is None:
        raise ValueError(
            "No he podido detectar la columna BVP. "
            "Escribe su nombre exacto en bvp_col_manual."
        )

    print(f"[INFO] Columna BVP usada: {bvp_col}")
    print(f"[INFO] Columna HR usada: {hr_col}")
    print(f"[INFO] Columna IBI usada: {ibi_col}")
    print(f"[INFO] Columna temporal usada: {time_col}")

    bvp_original = ensure_numeric(df[bvp_col], bvp_col)
    validate_bvp_signal(bvp_original, fs)

    time_s = infer_time_axis(df, fs, time_col=time_col)

    # Filtrado Chebyshev II
    print("[INFO] Aplicando filtro Chebyshev II...")
    bvp_filtered_cheby2 = filter_bvp_cheby2(
        bvp=bvp_original,
        fs=fs,
        lowcut=lowcut,
        highcut=highcut,
        order=order,
        rs=rs
    )

    # Filtrado Butterworth
    print("[INFO] Aplicando filtro Butterworth...")
    bvp_filtered_butter = filter_bvp_butterworth(
        bvp=bvp_original,
        fs=fs,
        lowcut=lowcut,
        highcut=highcut,
        order=order
    )

    peaks, peak_times, peak_properties = detect_bvp_peaks(
        bvp_filtered=bvp_filtered_cheby2,
        fs=fs,
        min_hr_bpm=min_hr,
        max_hr_bpm=max_hr,
        prominence_factor=prominence_factor
    )

    print(f"[INFO] Picos detectados: {len(peaks)}")

    ibi_df = compute_ibi_hr_from_peaks(
        peak_times=peak_times,
        min_hr_bpm=min_hr,
        max_hr_bpm=max_hr
    )

    valid_ibi_df = ibi_df[ibi_df["valid"]].copy()

    print("\nResumen IBI/HR derivados")
    print("-" * 40)

    if len(valid_ibi_df) > 0:
        print(f"IBI medio: {valid_ibi_df['ibi_ms'].mean():.2f} ms")
        print(f"IBI std:   {valid_ibi_df['ibi_ms'].std():.2f} ms")
        print(f"HR medio:  {valid_ibi_df['hr_bpm'].mean():.2f} bpm")
        print(f"HR std:    {valid_ibi_df['hr_bpm'].std():.2f} bpm")
    else:
        print("[AVISO] No hay IBI válidos tras el filtrado fisiológico.")

    # Guardar CSV procesado
    processed_df = df.copy()
    processed_df["BVP_filtered_cheby2"] = bvp_filtered_cheby2
    processed_df["BVP_filtered_butterworth"] = bvp_filtered_butter

    processed_df["BVP_peak_detected"] = False
    valid_peak_indices = peaks[(peaks >= 0) & (peaks < len(processed_df))]
    processed_df.loc[valid_peak_indices, "BVP_peak_detected"] = True

    processed_csv = output_dir / f"{csv_path.stem}_bvp_filtered.csv"
    ibi_csv = output_dir / f"{csv_path.stem}_derived_ibi_hr.csv"

    processed_df.to_csv(processed_csv, index=False)
    ibi_df.to_csv(ibi_csv, index=False)

    print(f"[OK] CSV con BVP filtrada guardado en: {processed_csv}")
    print(f"[OK] CSV con IBI/HR derivados guardado en: {ibi_csv}")

    # Comparación con HR del CSV
    hr_comparison = None

    if hr_col is not None:
        hr_csv = ensure_numeric(df[hr_col], hr_col)

        hr_csv_clean = hr_csv.copy()
        hr_csv_clean[(hr_csv_clean < min_hr) | (hr_csv_clean > max_hr)] = np.nan

        hr_comparison = compare_series(
            reference_time=time_s,
            reference_values=hr_csv_clean,
            derived_time=valid_ibi_df["time_s"].to_numpy(),
            derived_values=valid_ibi_df["hr_bpm"].to_numpy(),
            name="HR en bpm"
        )
    else:
        print("[INFO] No se encontró columna HR en el CSV. Se omite comparación HR.")

    # Comparación con IBI del CSV
    ibi_comparison = None

    if ibi_col is not None:
        ibi_csv_values = ensure_numeric(df[ibi_col], ibi_col)
        ibi_csv_s, unit = normalize_ibi_units(ibi_csv_values)

        print(f"[INFO] Unidad inferida para IBI del CSV: {unit}")

        min_ibi_s = 60.0 / max_hr
        max_ibi_s = 60.0 / min_hr

        ibi_csv_clean = ibi_csv_s.copy()
        ibi_csv_clean[(ibi_csv_clean < min_ibi_s) | (ibi_csv_clean > max_ibi_s)] = np.nan

        ibi_comparison = compare_series(
            reference_time=time_s,
            reference_values=ibi_csv_clean,
            derived_time=valid_ibi_df["time_s"].to_numpy(),
            derived_values=valid_ibi_df["ibi_s"].to_numpy(),
            name="IBI en segundos"
        )
    else:
        print("[INFO] No se encontró columna IBI en el CSV. Se omite comparación IBI.")

    # Gráfico BVP original vs filtrada
    plot_bvp_original_vs_filtered(
        time_s=time_s,
        bvp_original=bvp_original,
        bvp_cheby2=bvp_filtered_cheby2,
        bvp_butter=bvp_filtered_butter,
        output_path=output_dir / f"{csv_path.stem}_bvp_original_vs_cheby2_vs_butterworth.png",
        max_seconds=plot_seconds
    )

    # Gráfico HR comparativo
    plot_hr_comparison(
        hr_comparison=hr_comparison,
        output_path=output_dir / f"{csv_path.stem}_hr_comparison.png"
    )

    # Gráfico IBI comparativo
    plot_ibi_comparison(
        ibi_comparison=ibi_comparison,
        output_path=output_dir / f"{csv_path.stem}_ibi_comparison.png"
    )

    print("\nProceso terminado correctamente.")

if __name__ == "__main__":
    main()