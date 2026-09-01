"""
preprocess_dreamt_bvp_ibi_hr.py

Script de preprocesamiento inicial para DREAMT 64Hz.

Objetivo:
    Leer un CSV completo de un paciente y generar otro CSV donde:
    - BVP se sustituye por BVP filtrada con Chebyshev tipo II.
    - IBI se recalcula siguiendo la lógica de DREAMT_FE.clean_IBI.
    - HR se recalcula a partir del IBI siguiendo la lógica de DREAMT_FE.clean_IBI.
    - Opcionalmente se guardan copias de BVP, IBI y HR originales.

Fundamentación principal:
    Este script está adaptado de DREAMT_FE:
    https://github.com/WillKeWang/DREAMT_FE

    Funciones de referencia:
    - feature_engineering.py / preprocess_BVP
    - feature_engineering.py / clean_IBI
    - feature_engineering.py / preprocess_ALL_SIGNALS

Notas:
    DREAMT_FE usa en preprocess_BVP:
        signal.cheby2(
            N=10,
            rs=40,
            Wn=[0.5, 15],
            btype="bandpass",
            fs=64,
            output="sos"
        )

    Aquí se mantiene la misma familia de filtro y estructura general,
    pero se usa N=4 porque en este TFG se ha decidido trabajar con
    Chebyshev tipo II de cuarto orden.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from scipy import signal
from scipy.signal import find_peaks_cwt


# ============================================================
# CONFIGURACIÓN MANUAL
# ============================================================

INPUT_CSV = Path(r"C:\Proyectos_compartidos\TFG\sleep_classification_with_wearables\data\S002.csv")
OUTPUT_CSV = Path(r"C:\Proyectos_compartidos\TFG\sleep_classification_with_wearables\results\prepo2\S002_prepo2.csv")

FS_BVP = 64

# Si tus columnas se llaman exactamente como en DREAMT_FE, déjalas así.
# Si tu CSV usa otros nombres, cámbialos aquí.
TIMESTAMP_COL = "TIMESTAMP"
BVP_COL = "BVP"
IBI_COL = "IBI"
HR_COL = "HR"

# Columnas de acelerometría.
# DREAMT_FE.clean_IBI usa ACC_X, ACC_Y, ACC_Z para detectar artefactos de movimiento.
ACC_X_COL = "ACC_X"
ACC_Y_COL = "ACC_Y"
ACC_Z_COL = "ACC_Z"

# Parámetros BVP.
# Adaptado de DREAMT_FE.preprocess_BVP:
# - DREAMT_FE: N=10, rs=40, Wn=[0.5, 15], fs=64
# - Este script: N=4 por decisión del TFG
BVP_LOWCUT = 0.5
BVP_HIGHCUT = 15
BVP_ORDER = 4
BVP_RS = 40

# Parámetros de detección de picos.
# Adaptado de DREAMT_FE.clean_IBI:
# widths = np.arange(2, 32)
# window_size = 16
# min_distance = 12
CWT_WIDTHS = np.arange(2, 32)
CWT_WINDOW_SIZE = 16
MIN_PEAK_DISTANCE = 12

# Si True, guarda columnas originales como BVP_raw, IBI_raw, HR_raw.
KEEP_ORIGINAL_COLUMNS = True


# ============================================================
# UTILIDADES
# ============================================================

def require_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    """
    Comprueba que existan las columnas necesarias.

    Esta función no viene de DREAMT_FE; es una verificación de seguridad
    añadida para evitar errores silenciosos al trabajar con CSVs parciales.
    """
    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(
            "Faltan columnas necesarias en el CSV: "
            + ", ".join(missing)
        )


def to_numeric_array(df: pd.DataFrame, col: str) -> np.ndarray:
    """
    Convierte una columna a numérica.

    Esta función es una envoltura defensiva. DREAMT_FE asume que las columnas
    ya llegan limpias; aquí añadimos conversión explícita para trabajar con
    CSVs exportados o manipulados.
    """
    values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)

    n_nan = np.isnan(values).sum()
    if n_nan > 0:
        print(f"[AVISO] {col}: {n_nan} valores no numéricos/NaN.")

    return values


def interpolate_nans(signal_array: np.ndarray) -> np.ndarray:
    """
    Interpola NaNs antes del filtrado.

    DREAMT_FE no interpola NaNs explícitamente en preprocess_BVP.
    Esta protección se añade porque scipy.signal.sosfilt no maneja bien
    señales con NaNs.
    """
    x = np.asarray(signal_array, dtype=float)

    if np.isfinite(x).sum() == len(x):
        return x

    valid = np.isfinite(x)

    if valid.sum() == 0:
        raise ValueError("La señal no contiene ningún valor válido.")

    if valid.sum() == 1:
        print("[AVISO] Solo hay un valor válido. Se rellena toda la señal con ese valor.")
        return np.full_like(x, x[valid][0], dtype=float)

    idx = np.arange(len(x))
    x_interp = x.copy()
    x_interp[~valid] = np.interp(idx[~valid], idx[valid], x[valid])

    return x_interp


def validate_bvp(bvp: np.ndarray, fs: int = 64) -> None:
    """
    Verificaciones mínimas antes de filtrar BVP.

    Añadidas como seguridad adicional respecto al código original DREAMT_FE.
    """
    if len(bvp) < fs * 10:
        raise ValueError(
            f"La señal BVP es demasiado corta: {len(bvp)} muestras "
            f"({len(bvp) / fs:.2f} segundos)."
        )

    finite_ratio = np.isfinite(bvp).mean()

    if finite_ratio < 0.8:
        raise ValueError(
            f"La BVP tiene demasiados valores inválidos: "
            f"{finite_ratio:.2%} de valores finitos."
        )

    if np.nanstd(bvp) == 0:
        raise ValueError("La BVP parece constante. No tiene sentido filtrarla.")


# ============================================================
# BVP — ADAPTADO DE DREAMT_FE.preprocess_BVP
# ============================================================

def preprocess_bvp_dreamt_fe_style(
    bvp: np.ndarray,
    fs: int = FS_BVP,
    lowcut: float = BVP_LOWCUT,
    highcut: float = BVP_HIGHCUT,
    order: int = BVP_ORDER,
    rs: float = BVP_RS,
) -> np.ndarray:
    """
    Preprocesa BVP con un filtro Chebyshev tipo II.

    Basado en DREAMT_FE.feature_engineering.preprocess_BVP.

    Código original de referencia en DREAMT_FE:
        def preprocess_BVP(bvp):
            low = 0.5
            high = 15
            sos = signal.cheby2(
                N=10,
                rs=40,
                Wn=[low, high],
                btype="bandpass",
                fs=64,
                output="sos"
            )
            bvp_filtered = signal.sosfilt(sos, bvp)
            return bvp_filtered

    Adaptación:
        - Se usa N=4 en vez de N=10.
        - Se añaden verificaciones de seguridad.
        - Se interpolan NaNs antes del filtrado.
        - Se usa signal.sosfilt, igual que DREAMT_FE.
    """
    bvp = np.asarray(bvp, dtype=float)
    validate_bvp(bvp, fs=fs)

    nyquist = fs / 2

    if not (0 < lowcut < highcut < nyquist):
        raise ValueError(
            f"Frecuencias inválidas para BVP: lowcut={lowcut}, "
            f"highcut={highcut}, Nyquist={nyquist}."
        )

    bvp_clean = interpolate_nans(bvp)

    sos = signal.cheby2(
        N=order,
        rs=rs,
        Wn=[lowcut, highcut],
        btype="bandpass",
        fs=fs,
        output="sos",
    )

    bvp_filtered = signal.sosfilt(sos, bvp_clean)

    if not np.all(np.isfinite(bvp_filtered)):
        raise ValueError("El filtrado BVP produjo valores no finitos.")

    return bvp_filtered


# ============================================================
# IBI / HR — ADAPTADO DE DREAMT_FE.clean_IBI
# ============================================================

def filter_close_peaks_dreamt_fe_style(
    peaks: np.ndarray,
    min_distance: int = MIN_PEAK_DISTANCE,
) -> np.ndarray:
    """
    Filtra picos demasiado cercanos.

    Adaptado literalmente en lógica de DREAMT_FE.clean_IBI:
        def filter_close_peaks(peaks, min_distance):
            filtered_peaks = [peaks[0]]
            for peak in peaks[1:]:
                if peak - filtered_peaks[-1] > min_distance:
                    filtered_peaks.append(peak)
            return np.array(filtered_peaks)
    """
    peaks = np.asarray(peaks)

    if len(peaks) == 0:
        return np.array([], dtype=int)

    filtered_peaks = [peaks[0]]

    for peak in peaks[1:]:
        if peak - filtered_peaks[-1] > min_distance:
            filtered_peaks.append(peak)

    return np.array(filtered_peaks, dtype=int)


def calculate_hr_from_ibi_chunk_dreamt_fe_style(chunk: np.ndarray) -> np.ndarray:
    """
    Calcula HR para un bloque de IBI.

    Adaptado de DREAMT_FE.clean_IBI.process_chunk:
        mean = np.mean(chunk)
        inverse_mean = 60 / (mean + 1e-10)
        return np.full(chunk.shape, inverse_mean)
    """
    mean_ibi = np.mean(chunk)
    hr = 60 / (mean_ibi + 1e-10)

    return np.full(chunk.shape, hr)


def detect_motion_artifact_dreamt_fe_style(
    df: pd.DataFrame,
    window_seconds: int = 10,
    freq: int = FS_BVP,
) -> np.ndarray:
    """
    Detecta artefactos de movimiento a partir de ACC_X, ACC_Y y ACC_Z.

    Adaptado de DREAMT_FE.clean_IBI.detect_motion_artifact.

    DREAMT_FE marca artefacto si, dentro de una ventana, hay suficientes
    diferencias absolutas mayores que 5 en cualquiera de los ejes.
    """
    required_acc_cols = [ACC_X_COL, ACC_Y_COL, ACC_Z_COL]
    missing_acc = [col for col in required_acc_cols if col not in df.columns]

    if missing_acc:
        print(
            "[AVISO] No se puede calcular motion_artifact porque faltan: "
            + ", ".join(missing_acc)
        )
        return np.full(df.shape[0], np.nan)

    window = window_seconds * freq

    acc_x = to_numeric_array(df, ACC_X_COL)
    acc_y = to_numeric_array(df, ACC_Y_COL)
    acc_z = to_numeric_array(df, ACC_Z_COL)

    acc_x_diff = np.diff(acc_x)
    acc_y_diff = np.diff(acc_y)
    acc_z_diff = np.diff(acc_z)

    motion_artifact = np.zeros((acc_x.shape[0],))

    for i in range(1, acc_x.shape[0]):
        acc_x_diff_window = acc_x_diff[i:(i + window)]
        acc_y_diff_window = acc_y_diff[i:(i + window)]
        acc_z_diff_window = acc_z_diff[i:(i + window)]

        if (
            np.sum(np.abs(acc_x_diff_window) > 5) >= 5
            or np.sum(np.abs(acc_y_diff_window) > 5) >= 5
            or np.sum(np.abs(acc_z_diff_window) > 5) >= 5
        ):
            motion_artifact[i] = 1

    return motion_artifact


def clean_ibi_hr_dreamt_fe_style(
    df: pd.DataFrame,
    freq: int = FS_BVP,
) -> pd.DataFrame:
    """
    Recalcula IBI y HR a partir de BVP siguiendo DREAMT_FE.clean_IBI.

    Lógica de DREAMT_FE:
        1. Toma df.BVP.
        2. Detecta picos con find_peaks_cwt.
        3. Filtra picos cercanos con min_distance=12.
        4. Calcula IBI como diferencia entre picos.
        5. Repite cada valor de IBI hasta el siguiente pico para obtener
           una serie con longitud comparable a BVP.
        6. Divide por freq para pasar de muestras a segundos.
        7. Calcula HR por bloques de 64 muestras, es decir, 1 segundo.
        8. Guarda df["IBI"] y df["HR"].

    Esta función mantiene esa estructura.
    """
    if BVP_COL not in df.columns:
        raise ValueError(f"No existe la columna {BVP_COL} para recalcular IBI/HR.")

    bvp = to_numeric_array(df, BVP_COL)
    segment_length = bvp.shape[0]

    print("[INFO] Detectando picos BVP con find_peaks_cwt, como en DREAMT_FE.clean_IBI...")

    peaks = find_peaks_cwt(
        bvp,
        CWT_WIDTHS,
        window_size=CWT_WINDOW_SIZE,
    )

    peaks = np.asarray(peaks, dtype=int)
    peaks = filter_close_peaks_dreamt_fe_style(
        peaks,
        min_distance=MIN_PEAK_DISTANCE,
    )

    if len(peaks) < 2:
        raise ValueError(
            "Se han detectado menos de 2 picos en BVP. "
            "No se puede recalcular IBI/HR."
        )

    print(f"[INFO] Picos detectados tras filtrado de cercanía: {len(peaks)}")

    # Adaptado de DREAMT_FE.clean_IBI.
    ibi_array = np.diff(peaks)
    ibi_array = np.insert(ibi_array, 0, peaks[0])

    repeat_counts = np.insert(
        ibi_array[1:],
        ibi_array.shape[0] - 1,
        segment_length - peaks[-1],
    )

    repeated_array = np.repeat(ibi_array, repeat_counts)

    pad_width = segment_length - len(repeated_array)

    if pad_width < 0:
        raise ValueError(
            "La longitud final de IBI es mayor que la longitud de la señal. "
            "Esto reproduce la protección de DREAMT_FE.clean_IBI."
        )

    padded_array = np.pad(
        repeated_array,
        (pad_width, 0),
        mode="constant",
        constant_values=0,
    )

    cleaned_ibi_array = padded_array / freq

    # Artefacto de movimiento, si existen columnas ACC.
    motion_artifact_array = detect_motion_artifact_dreamt_fe_style(df, freq=freq)

    # HR por bloques de 64 muestras, igual que DREAMT_FE.
    chunk_size = freq

    hr_array = np.concatenate(
        [
            calculate_hr_from_ibi_chunk_dreamt_fe_style(
                cleaned_ibi_array[i:(i + chunk_size)]
            )
            for i in range(0, len(cleaned_ibi_array), chunk_size)
        ]
    )

    # Protección por si el último chunk deja una longitud distinta.
    if len(hr_array) > segment_length:
        hr_array = hr_array[:segment_length]
    elif len(hr_array) < segment_length:
        hr_array = np.pad(
            hr_array,
            (0, segment_length - len(hr_array)),
            mode="edge",
        )

    df[IBI_COL] = cleaned_ibi_array
    df[HR_COL] = hr_array
    df["motion_artifact"] = motion_artifact_array

    return df


# ============================================================
# PIPELINE PRINCIPAL
# ============================================================

def preprocess_patient_csv(
    input_csv: Path,
    output_csv: Path,
) -> pd.DataFrame:
    """
    Lee un CSV de paciente y genera un CSV preprocesado.

    Basado en la lógica general de DREAMT_FE.preprocess_ALL_SIGNALS:
        - parte de un dataframe E4 agregado,
        - preprocesa BVP,
        - conserva HR/TEMP/TIMESTAMP,
        - asigna BVP filtrada,
        - llama a clean_IBI para actualizar IBI y HR.

    En esta versión inicial nos centramos solo en:
        BVP, IBI y HR.
    """
    if not input_csv.exists():
        raise FileNotFoundError(f"No existe el CSV de entrada: {input_csv}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Leyendo CSV: {input_csv}")
    df = pd.read_csv(input_csv)

    if df.empty:
        raise ValueError("El CSV está vacío.")

    require_columns(df, [BVP_COL])

    df_out = df.copy()

    if KEEP_ORIGINAL_COLUMNS:
        df_out[f"{BVP_COL}_raw"] = df_out[BVP_COL]

        if IBI_COL in df_out.columns:
            df_out[f"{IBI_COL}_raw"] = df_out[IBI_COL]

        if HR_COL in df_out.columns:
            df_out[f"{HR_COL}_raw"] = df_out[HR_COL]

    # ------------------------------------------------------------
    # 1. BVP filtrada con Chebyshev II
    # ------------------------------------------------------------

    print("[INFO] Preprocesando BVP según DREAMT_FE.preprocess_BVP adaptado...")

    bvp_raw = to_numeric_array(df_out, BVP_COL)

    bvp_filtered = preprocess_bvp_dreamt_fe_style(
        bvp=bvp_raw,
        fs=FS_BVP,
        lowcut=BVP_LOWCUT,
        highcut=BVP_HIGHCUT,
        order=BVP_ORDER,
        rs=BVP_RS,
    )

    df_out[BVP_COL] = bvp_filtered

    print("[OK] BVP sustituida por BVP filtrada.")

    # ------------------------------------------------------------
    # 2. IBI y HR recalculados desde BVP
    # ------------------------------------------------------------

    print("[INFO] Recalculando IBI y HR según DREAMT_FE.clean_IBI adaptado...")

    df_out = clean_ibi_hr_dreamt_fe_style(
        df_out,
        freq=FS_BVP,
    )

    print("[OK] IBI y HR sustituidos por señales recalculadas desde BVP filtrada.")

    # ------------------------------------------------------------
    # 3. Guardado
    # ------------------------------------------------------------

    df_out.to_csv(output_csv, index=False)

    print(f"[OK] CSV preprocesado guardado en: {output_csv}")

    return df_out


def main():
    preprocess_patient_csv(
        input_csv=INPUT_CSV,
        output_csv=OUTPUT_CSV,
    )


if __name__ == "__main__":
    main()