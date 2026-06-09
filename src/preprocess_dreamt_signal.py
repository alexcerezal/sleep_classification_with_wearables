from pathlib import Path

import numpy as np
import pandas as pd

from scipy.signal import cheby2, sosfiltfilt, medfilt


# ============================================================
# CONFIGURACIÓN MANUAL
# ============================================================

INPUT_CSV = Path(r"C:\Proyectos_compartidos\TFG\sleep_classification_with_wearables\data\prueba.csv")

OUTPUT_CSV = Path(r"C:\Proyectos_compartidos\TFG\sleep_classification_with_wearables\results\prepo\prueba_preprocessed.csv")

# Frecuencia de muestreo de BVP en DREAMT 64 Hz
FS_BVP = 64.0

# Columnas manuales.
# Si las dejas en None, el script intenta detectarlas.
BVP_COL_MANUAL = None
HR_COL_MANUAL = None
IBI_COL_MANUAL = None

# Parámetros BVP
BVP_LOWCUT = 0.5
BVP_HIGHCUT = 4.0
BVP_ORDER = 4
BVP_RS = 40.0

# Rangos fisiológicos para limpieza
HR_MIN = 35.0
HR_MAX = 180.0

# IBI compatible con HR_MIN-HR_MAX
# IBI = 60 / HR
IBI_MIN_S = 60.0 / HR_MAX
IBI_MAX_S = 60.0 / HR_MIN

# Si True, guarda copia de columnas originales antes de sustituirlas
KEEP_ORIGINAL_COLUMNS = False


# ============================================================
# UTILIDADES GENERALES
# ============================================================

def find_column(df, candidates):
    """
    Busca una columna ignorando mayúsculas/minúsculas.
    Primero busca coincidencia exacta, luego coincidencia parcial.
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


def to_numeric_array(series, col_name):
    """
    Convierte una columna a array numérico.
    Los valores no convertibles pasan a NaN.
    """
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)

    n_nan = np.isnan(values).sum()
    if n_nan > 0:
        print(f"[AVISO] {col_name}: {n_nan} valores NaN/no numéricos.")

    return values


def interpolate_nans_1d(x):
    """
    Interpola NaNs en una señal 1D.
    Si hay NaNs al principio o al final, se rellenan con el primer/último valor válido.
    """
    x = np.asarray(x, dtype=float)
    idx = np.arange(len(x))

    valid = np.isfinite(x)

    if valid.sum() == 0:
        raise ValueError("La señal no tiene ningún valor válido.")

    if valid.sum() == 1:
        return np.full_like(x, x[valid][0], dtype=float)

    x_interp = x.copy()
    x_interp[~valid] = np.interp(idx[~valid], idx[valid], x[valid])

    return x_interp


def hampel_filter(values, window_size=7, n_sigmas=3.0):
    """
    Filtro Hampel simple para detectar outliers.
    Sustituye outliers por la mediana local.

    Es útil para HR e IBI porque elimina saltos puntuales no fisiológicos.
    """
    x = np.asarray(values, dtype=float).copy()

    if len(x) < window_size:
        return x

    half_window = window_size // 2

    for i in range(len(x)):
        start = max(0, i - half_window)
        end = min(len(x), i + half_window + 1)

        window = x[start:end]
        window = window[np.isfinite(window)]

        if len(window) < 3 or not np.isfinite(x[i]):
            continue

        median = np.median(window)
        mad = np.median(np.abs(window - median))

        if mad == 0:
            continue

        robust_z = 0.6745 * (x[i] - median) / mad

        if np.abs(robust_z) > n_sigmas:
            x[i] = median

    return x


# ============================================================
# PREPROCESAMIENTO BVP
# ============================================================

def validate_bvp(bvp, fs):
    """
    Comprobaciones básicas antes de filtrar BVP.
    """
    if len(bvp) < int(fs * 10):
        raise ValueError(
            f"La BVP es demasiado corta: {len(bvp)} muestras "
            f"({len(bvp) / fs:.2f} segundos)."
        )

    finite_ratio = np.isfinite(bvp).mean()

    if finite_ratio < 0.8:
        raise ValueError(
            f"Demasiados valores inválidos en BVP: solo {finite_ratio:.1%} son finitos."
        )

    if np.nanstd(bvp) == 0:
        raise ValueError("La BVP parece constante. No se puede filtrar correctamente.")


def filter_bvp_cheby2(
    bvp,
    fs=64.0,
    lowcut=0.5,
    highcut=4.0,
    order=4,
    rs=40.0
):
    """
    Filtra BVP con un filtro pasabanda Chebyshev tipo II.

    - Tipo: pasabanda
    - Orden: 4
    - Fase cero mediante sosfiltfilt
    """
    nyquist = fs / 2.0

    if not (0 < lowcut < highcut < nyquist):
        raise ValueError(
            f"Frecuencias inválidas: lowcut={lowcut}, highcut={highcut}, "
            f"Nyquist={nyquist}."
        )

    validate_bvp(bvp, fs)

    bvp_clean = interpolate_nans_1d(bvp)

    sos = cheby2(
        N=order,
        rs=rs,
        Wn=[lowcut, highcut],
        btype="bandpass",
        fs=fs,
        output="sos"
    )

    bvp_filtered = sosfiltfilt(sos, bvp_clean)

    if not np.all(np.isfinite(bvp_filtered)):
        raise ValueError("El filtrado de BVP produjo valores no finitos.")

    return bvp_filtered


# ============================================================
# PREPROCESAMIENTO HR
# ============================================================

def preprocess_hr(hr, hr_min=35.0, hr_max=180.0):
    """
    Limpieza básica de HR.

    Pasos:
    1. Convierte valores fuera de rango fisiológico a NaN.
    2. Interpola huecos.
    3. Aplica filtro Hampel para saltos puntuales.
    4. Aplica mediana suave para estabilizar.

    Nota:
    HR no se trata como si estuviera a 64 Hz. Se filtra sobre su propia secuencia
    de valores válidos/alineados en el CSV.
    """
    hr = np.asarray(hr, dtype=float).copy()

    invalid = (
        ~np.isfinite(hr)
        | (hr < hr_min)
        | (hr > hr_max)
    )

    n_invalid = invalid.sum()
    if n_invalid > 0:
        print(f"[INFO] HR: {n_invalid} valores fuera de rango convertidos a NaN.")

    hr[invalid] = np.nan

    if np.isfinite(hr).sum() == 0:
        print("[AVISO] HR no tiene valores válidos. Se deja como NaN.")
        return hr

    hr_interp = interpolate_nans_1d(hr)

    hr_hampel = hampel_filter(
        hr_interp,
        window_size=7,
        n_sigmas=3.0
    )

    # Kernel impar. Si la serie es muy corta, se omite.
    if len(hr_hampel) >= 5:
        hr_filtered = medfilt(hr_hampel, kernel_size=5)
    else:
        hr_filtered = hr_hampel

    return hr_filtered


# ============================================================
# PREPROCESAMIENTO IBI
# ============================================================

def infer_ibi_unit(ibi):
    """
    Intenta inferir si el IBI está en segundos o milisegundos.

    Empatica suele trabajar con IBI en segundos, pero algunos CSV pueden
    guardarlo en milisegundos.
    """
    valid = ibi[np.isfinite(ibi) & (ibi > 0)]

    if len(valid) == 0:
        return "unknown"

    median_val = np.nanmedian(valid)

    if median_val > 10:
        return "ms"

    return "s"


def preprocess_ibi(
    ibi,
    ibi_min_s=IBI_MIN_S,
    ibi_max_s=IBI_MAX_S
):
    """
    Limpieza básica de IBI.

    Pasos:
    1. Detecta si está en segundos o milisegundos.
    2. Convierte temporalmente a segundos.
    3. Elimina valores fuera de rango fisiológico.
    4. Interpola huecos.
    5. Aplica Hampel para outliers.
    6. Devuelve en la unidad original.

    Nota:
    IBI no es realmente una señal a 64 Hz. Puede ser una serie irregular o una
    columna alineada/rellenada. Aquí solo limpiamos sus valores, no asumimos 64 Hz.
    """
    ibi = np.asarray(ibi, dtype=float).copy()

    unit = infer_ibi_unit(ibi)
    print(f"[INFO] Unidad inferida para IBI: {unit}")

    if unit == "ms":
        ibi_s = ibi / 1000.0
    else:
        ibi_s = ibi.copy()

    invalid = (
        ~np.isfinite(ibi_s)
        | (ibi_s < ibi_min_s)
        | (ibi_s > ibi_max_s)
    )

    n_invalid = invalid.sum()
    if n_invalid > 0:
        print(f"[INFO] IBI: {n_invalid} valores fuera de rango convertidos a NaN.")

    ibi_s[invalid] = np.nan

    if np.isfinite(ibi_s).sum() == 0:
        print("[AVISO] IBI no tiene valores válidos. Se deja como NaN.")
        return ibi

    ibi_s_interp = interpolate_nans_1d(ibi_s)

    ibi_s_hampel = hampel_filter(
        ibi_s_interp,
        window_size=7,
        n_sigmas=3.0
    )

    if len(ibi_s_hampel) >= 5:
        ibi_s_filtered = medfilt(ibi_s_hampel, kernel_size=5)
    else:
        ibi_s_filtered = ibi_s_hampel

    if unit == "ms":
        return ibi_s_filtered * 1000.0

    return ibi_s_filtered


# ============================================================
# MAIN
# ============================================================

def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"No existe el CSV de entrada: {INPUT_CSV}")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Leyendo CSV: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)

    if df.empty:
        raise ValueError("El CSV está vacío.")

    print(f"[INFO] Filas: {len(df)}")
    print(f"[INFO] Columnas: {list(df.columns)}")

    bvp_col = BVP_COL_MANUAL or find_column(
        df,
        ["BVP", "bvp", "blood_volume_pulse", "BloodVolumePulse"]
    )

    hr_col = HR_COL_MANUAL or find_column(
        df,
        ["HR", "hr", "heart_rate", "HeartRate"]
    )

    ibi_col = IBI_COL_MANUAL or find_column(
        df,
        ["IBI", "ibi", "interbeat", "inter_beat_interval", "RR", "rr"]
    )

    print(f"[INFO] Columna BVP detectada: {bvp_col}")
    print(f"[INFO] Columna HR detectada: {hr_col}")
    print(f"[INFO] Columna IBI detectada: {ibi_col}")

    if bvp_col is None:
        raise ValueError(
            "No se ha encontrado columna BVP. "
            "Especifica el nombre manualmente en BVP_COL_MANUAL."
        )

    df_out = df.copy()

    # ------------------------------------------------------------
    # BVP
    # ------------------------------------------------------------

    print("[INFO] Preprocesando BVP...")

    bvp_raw = to_numeric_array(df[bvp_col], bvp_col)

    bvp_filtered = filter_bvp_cheby2(
        bvp=bvp_raw,
        fs=FS_BVP,
        lowcut=BVP_LOWCUT,
        highcut=BVP_HIGHCUT,
        order=BVP_ORDER,
        rs=BVP_RS
    )

    if KEEP_ORIGINAL_COLUMNS:
        df_out[f"{bvp_col}_raw"] = df_out[bvp_col]

    df_out[bvp_col] = bvp_filtered

    print("[OK] BVP sustituida por BVP filtrada Chebyshev II.")

    # ------------------------------------------------------------
    # HR
    # ------------------------------------------------------------

    if hr_col is not None:
        print("[INFO] Preprocesando HR...")

        hr_raw = to_numeric_array(df[hr_col], hr_col)

        hr_filtered = preprocess_hr(
            hr=hr_raw,
            hr_min=HR_MIN,
            hr_max=HR_MAX
        )

        if KEEP_ORIGINAL_COLUMNS:
            df_out[f"{hr_col}_raw"] = df_out[hr_col]

        df_out[hr_col] = hr_filtered

        print("[OK] HR sustituida por HR limpia.")
    else:
        print("[AVISO] No se encontró columna HR. Se omite.")

    # ------------------------------------------------------------
    # IBI
    # ------------------------------------------------------------

    if ibi_col is not None:
        print("[INFO] Preprocesando IBI...")

        ibi_raw = to_numeric_array(df[ibi_col], ibi_col)

        ibi_filtered = preprocess_ibi(
            ibi=ibi_raw,
            ibi_min_s=IBI_MIN_S,
            ibi_max_s=IBI_MAX_S
        )

        if KEEP_ORIGINAL_COLUMNS:
            df_out[f"{ibi_col}_raw"] = df_out[ibi_col]

        df_out[ibi_col] = ibi_filtered

        print("[OK] IBI sustituida por IBI limpio.")
    else:
        print("[AVISO] No se encontró columna IBI. Se omite.")

    # ------------------------------------------------------------
    # Guardado
    # ------------------------------------------------------------

    df_out.to_csv(OUTPUT_CSV, index=False)

    print(f"[OK] CSV preprocesado guardado en: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()