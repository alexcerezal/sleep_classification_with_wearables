import os
import glob
import re
import numpy as np
import pandas as pd

from scipy.signal import find_peaks, peak_widths, welch
from scipy.interpolate import interp1d


# ============================================================
# CONFIGURACIÓN
# ============================================================

# Carpeta con un CSV preprocesado por paciente.
SIGNALS_FOLDER_PATH = r"C:\Proyectos_compartidos\TFG\sleep_classification_with_wearables\data\preprocesados"

# CSV global con todas las features por época de todos los pacientes.
EPOCH_FEATURES_CSV_PATH = r"C:\Proyectos_compartidos\TFG\sleep_classification_with_wearables\data\dreamt_epoch_features.csv"

# CSV de salida.
OUTPUT_CSV_PATH = r"C:\Proyectos_compartidos\TFG\sleep_classification_with_wearables\data\probatura.csv"

# Columnas.
SUBJECT_COL = "subject_id"
BVP_COL = "BVP"
IBI_COL = "IBI"

# Parámetros DREAMT 64 Hz.
BVP_FS = 64
EPOCH_SECONDS = 30

# HRV frecuencial.
HRV_WINDOW_SECONDS = 5 * 60
HRV_WINDOW_MODE = "past"  # "past" o "centered"
HRV_INTERP_FS = 4.0

# Rango fisiológico de IBI en segundos.
IBI_MIN = 0.30
IBI_MAX = 2.00


# ============================================================
# MAPEO ENTRE subject_id Y CSV DE SEÑALES
# ============================================================

def normalize_subject_id(value):
    """
    Normaliza subject_id para comparar nombres.
    Ejemplo:
    1        -> "1"
    "001"    -> "1"
    "S001"   -> "1"
    "sub-01" -> "1"

    Si tus subject_id son strings más complejos, modifica esta función.
    """
    value = str(value)

    numbers = re.findall(r"\d+", value)

    if numbers:
        return str(int(numbers[-1]))

    return value.lower().strip()


def build_signal_file_index(signals_folder):
    """
    Crea un diccionario:
        subject_id_normalizado -> ruta_csv

    Busca cualquier .csv dentro de la carpeta.
    """
    csv_files = sorted(glob.glob(os.path.join(signals_folder, "*.csv")))

    if not csv_files:
        raise FileNotFoundError(f"No se encontraron CSV en: {signals_folder}")

    file_index = {}

    for path in csv_files:
        filename = os.path.basename(path)
        stem = os.path.splitext(filename)[0]

        subject_key = normalize_subject_id(stem)

        if subject_key in file_index:
            print(
                f"[AVISO] Hay más de un CSV que parece corresponder al sujeto {subject_key}. "
                f"Se conserva el primero:\n"
                f"  - {file_index[subject_key]}\n"
                f"  - ignorado: {path}"
            )
            continue

        file_index[subject_key] = path

    return file_index


def find_signal_file_for_subject(subject_id, file_index):
    """
    Busca el CSV de señales correspondiente a un subject_id.
    """
    subject_key = normalize_subject_id(subject_id)

    if subject_key in file_index:
        return file_index[subject_key]

    return None


# ============================================================
# UTILIDADES
# ============================================================

def safe_mean(x):
    x = np.asarray(x, dtype=float)
    return np.nan if x.size == 0 else float(np.nanmean(x))


def safe_std(x):
    x = np.asarray(x, dtype=float)
    return np.nan if x.size <= 1 else float(np.nanstd(x, ddof=1))


def bandpower(freqs, psd, fmin, fmax):
    mask = (freqs >= fmin) & (freqs < fmax)

    if np.sum(mask) < 2:
        return np.nan

    return float(np.trapezoid(psd[mask], freqs[mask]))


def compress_repeated_ibi_values(ibi_values, tolerance=1e-4):
    """
    Si IBI está repetido muchas veces porque está alineado con la señal a 64 Hz,
    elimina repeticiones consecutivas casi idénticas.
    """
    ibi_values = np.asarray(ibi_values, dtype=float)
    ibi_values = ibi_values[np.isfinite(ibi_values)]
    ibi_values = ibi_values[(ibi_values >= IBI_MIN) & (ibi_values <= IBI_MAX)]

    if ibi_values.size <= 1:
        return ibi_values

    keep = np.ones(len(ibi_values), dtype=bool)
    keep[1:] = np.abs(np.diff(ibi_values)) > tolerance

    return ibi_values[keep]


# ============================================================
# FEATURES MORFOLÓGICAS DE BVP / PPG
# ============================================================

def empty_bvp_morphology_features():
    return {
        "PPG_peak_count": np.nan,
        "PPG_signal_valid_ratio": np.nan,
        "PPG_valid_peak_ratio": np.nan,
        "PPG_mean_peak_amplitude": np.nan,
        "PPG_std_peak_amplitude": np.nan,
        "PPG_mean_pulse_amplitude": np.nan,
        "PPG_std_pulse_amplitude": np.nan,
        "PPG_mean_pulse_width": np.nan,
        "PPG_std_pulse_width": np.nan,
        "PPG_mean_rise_time": np.nan,
        "PPG_std_rise_time": np.nan,
        "PPG_mean_decay_time": np.nan,
        "PPG_std_decay_time": np.nan,
        "PPG_mean_pulse_area": np.nan,
        "PPG_std_pulse_area": np.nan,
        "PPG_mean_cycle_duration": np.nan,
        "PPG_std_cycle_duration": np.nan,
    }


def compute_bvp_morphology_features(bvp_segment, fs=BVP_FS):
    result = empty_bvp_morphology_features()

    bvp = np.asarray(bvp_segment, dtype=float)

    if bvp.size == 0:
        return result

    valid_mask = np.isfinite(bvp)
    result["PPG_signal_valid_ratio"] = float(np.mean(valid_mask))

    if np.sum(valid_mask) < fs * 5:
        return result

    idx = np.arange(len(bvp))

    if np.any(~valid_mask):
        bvp = np.interp(idx, idx[valid_mask], bvp[valid_mask])

    bvp_std = np.nanstd(bvp)

    if not np.isfinite(bvp_std) or bvp_std == 0:
        return result

    bvp_norm = (bvp - np.nanmean(bvp)) / bvp_std

    min_distance_samples = int(fs * 60 / 220)

    peaks, _ = find_peaks(
        bvp_norm,
        distance=max(min_distance_samples, 1),
        prominence=0.25
    )

    valleys, _ = find_peaks(
        -bvp_norm,
        distance=max(min_distance_samples, 1),
        prominence=0.15
    )

    result["PPG_peak_count"] = int(len(peaks))

    if len(peaks) == 0:
        return result

    peak_amplitudes = bvp[peaks]
    result["PPG_mean_peak_amplitude"] = safe_mean(peak_amplitudes)
    result["PPG_std_peak_amplitude"] = safe_std(peak_amplitudes)

    try:
        widths_samples = peak_widths(bvp_norm, peaks, rel_height=0.5)[0]
        widths_seconds = widths_samples / fs

        result["PPG_mean_pulse_width"] = safe_mean(widths_seconds)
        result["PPG_std_pulse_width"] = safe_std(widths_seconds)

    except Exception:
        pass

    pulse_amplitudes = []
    rise_times = []
    decay_times = []
    pulse_areas = []
    valid_peak_count = 0

    for peak in peaks:
        prev_valleys = valleys[valleys < peak]
        next_valleys = valleys[valleys > peak]

        if len(prev_valleys) == 0 or len(next_valleys) == 0:
            continue

        left_valley = prev_valleys[-1]
        right_valley = next_valleys[0]

        if right_valley <= left_valley:
            continue

        valid_peak_count += 1

        baseline = min(bvp[left_valley], bvp[right_valley])
        pulse_amp = bvp[peak] - baseline

        rise_time = (peak - left_valley) / fs
        decay_time = (right_valley - peak) / fs

        pulse = bvp[left_valley:right_valley + 1]
        pulse_area = np.trapezoid(pulse - baseline, dx=1 / fs)

        pulse_amplitudes.append(pulse_amp)
        rise_times.append(rise_time)
        decay_times.append(decay_time)
        pulse_areas.append(pulse_area)

    result["PPG_valid_peak_ratio"] = (
        float(valid_peak_count / len(peaks)) if len(peaks) > 0 else np.nan
    )

    result["PPG_mean_pulse_amplitude"] = safe_mean(pulse_amplitudes)
    result["PPG_std_pulse_amplitude"] = safe_std(pulse_amplitudes)
    result["PPG_mean_rise_time"] = safe_mean(rise_times)
    result["PPG_std_rise_time"] = safe_std(rise_times)
    result["PPG_mean_decay_time"] = safe_mean(decay_times)
    result["PPG_std_decay_time"] = safe_std(decay_times)
    result["PPG_mean_pulse_area"] = safe_mean(pulse_areas)
    result["PPG_std_pulse_area"] = safe_std(pulse_areas)

    if len(peaks) >= 2:
        cycle_durations = np.diff(peaks) / fs
        result["PPG_mean_cycle_duration"] = safe_mean(cycle_durations)
        result["PPG_std_cycle_duration"] = safe_std(cycle_durations)

    return result


# ============================================================
# FEATURES FRECUENCIALES DE HRV
# ============================================================

def empty_hrv_frequency_features():
    return {
        "HRV_VLF_power_5min": np.nan,
        "HRV_LF_power_5min": np.nan,
        "HRV_HF_power_5min": np.nan,
        "HRV_total_power_5min": np.nan,
        "HRV_LF_HF_ratio_5min": np.nan,
        "HRV_LF_norm_5min": np.nan,
        "HRV_HF_norm_5min": np.nan,
        "HRV_VLF_LF_ratio_5min": np.nan,
        "HRV_VLF_HF_ratio_5min": np.nan,
        "HRV_LF_total_ratio_5min": np.nan,
        "HRV_HF_total_ratio_5min": np.nan,
        "HRV_peak_LF_5min": np.nan,
        "HRV_peak_HF_5min": np.nan,
        "HRV_IBI_count_5min": np.nan,
        "HRV_IBI_mean_5min": np.nan,
        "HRV_IBI_std_5min": np.nan,
    }


def compute_hrv_frequency_features_from_ibi(ibi_values, interp_fs=HRV_INTERP_FS):
    result = empty_hrv_frequency_features()

    ibi = compress_repeated_ibi_values(ibi_values)

    result["HRV_IBI_count_5min"] = int(len(ibi)) if len(ibi) > 0 else 0
    result["HRV_IBI_mean_5min"] = safe_mean(ibi)
    result["HRV_IBI_std_5min"] = safe_std(ibi)

    if len(ibi) < 20:
        return result

    beat_times = np.cumsum(ibi)
    beat_times = beat_times - beat_times[0]

    duration = beat_times[-1]

    if duration < 60:
        return result

    t_interp = np.arange(0, duration, 1 / interp_fs)

    if len(t_interp) < 32:
        return result

    try:
        f_interp = interp1d(
            beat_times,
            ibi,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate"
        )

        ibi_interp = f_interp(t_interp)
        ibi_interp = ibi_interp - np.nanmean(ibi_interp)

        nperseg = min(256, len(ibi_interp))

        freqs, psd = welch(
            ibi_interp,
            fs=interp_fs,
            nperseg=nperseg,
            detrend="constant"
        )

    except Exception:
        return result

    vlf = bandpower(freqs, psd, 0.0033, 0.04)
    lf = bandpower(freqs, psd, 0.04, 0.15)
    hf = bandpower(freqs, psd, 0.15, 0.40)
    total = bandpower(freqs, psd, 0.0033, 0.40)

    result["HRV_VLF_power_5min"] = vlf
    result["HRV_LF_power_5min"] = lf
    result["HRV_HF_power_5min"] = hf
    result["HRV_total_power_5min"] = total

    if np.isfinite(lf) and np.isfinite(hf) and hf > 0:
        result["HRV_LF_HF_ratio_5min"] = float(lf / hf)

    if np.isfinite(vlf) and np.isfinite(lf) and lf > 0:
        result["HRV_VLF_LF_ratio_5min"] = float(vlf / lf)

    if np.isfinite(vlf) and np.isfinite(hf) and hf > 0:
        result["HRV_VLF_HF_ratio_5min"] = float(vlf / hf)

    if np.isfinite(total) and total > 0:
        if np.isfinite(lf):
            result["HRV_LF_total_ratio_5min"] = float(lf / total)
        if np.isfinite(hf):
            result["HRV_HF_total_ratio_5min"] = float(hf / total)

    denom = lf + hf if np.isfinite(lf) and np.isfinite(hf) else np.nan

    if np.isfinite(denom) and denom > 0:
        result["HRV_LF_norm_5min"] = float(lf / denom)
        result["HRV_HF_norm_5min"] = float(hf / denom)

    lf_mask = (freqs >= 0.04) & (freqs < 0.15)
    hf_mask = (freqs >= 0.15) & (freqs < 0.40)

    if np.any(lf_mask):
        result["HRV_peak_LF_5min"] = float(freqs[lf_mask][np.argmax(psd[lf_mask])])

    if np.any(hf_mask):
        result["HRV_peak_HF_5min"] = float(freqs[hf_mask][np.argmax(psd[hf_mask])])

    return result


# ============================================================
# FEATURES CIRCADIANAS
# ============================================================

def add_circadian_features(features_df):
    df = features_df.copy()

    if SUBJECT_COL not in df.columns:
        raise ValueError(f"El CSV de features debe contener la columna {SUBJECT_COL}.")

    df["epoch_idx_in_recording"] = df.groupby(SUBJECT_COL).cumcount()
    df["n_epochs_recording"] = df.groupby(SUBJECT_COL)["epoch_idx_in_recording"].transform("count")

    df["epoch_idx_norm"] = (
        df["epoch_idx_in_recording"] /
        np.maximum(df["n_epochs_recording"] - 1, 1)
    )

    df["time_from_start_min"] = df["epoch_idx_in_recording"] * (EPOCH_SECONDS / 60)
    df["time_from_start_hours"] = df["time_from_start_min"] / 60

    df["sin_time_night"] = np.sin(2 * np.pi * df["epoch_idx_norm"])
    df["cos_time_night"] = np.cos(2 * np.pi * df["epoch_idx_norm"])

    df["is_first_third"] = (df["epoch_idx_norm"] < 1 / 3).astype(int)

    df["is_second_third"] = (
        (df["epoch_idx_norm"] >= 1 / 3) &
        (df["epoch_idx_norm"] < 2 / 3)
    ).astype(int)

    df["is_last_third"] = (df["epoch_idx_norm"] >= 2 / 3).astype(int)

    df["estimated_sleep_cycle"] = np.floor(df["time_from_start_min"] / 90).astype(int) + 1
    df["cycle_position_norm"] = (df["time_from_start_min"] % 90) / 90

    df["sin_sleep_cycle"] = np.sin(2 * np.pi * df["cycle_position_norm"])
    df["cos_sleep_cycle"] = np.cos(2 * np.pi * df["cycle_position_norm"])

    return df


# ============================================================
# SEGMENTACIÓN DE SEÑALES POR ÉPOCAS
# ============================================================

def get_epoch_signal_segment(signal_df, epoch_idx, signal_col, fs=BVP_FS):
    samples_per_epoch = int(EPOCH_SECONDS * fs)

    start = epoch_idx * samples_per_epoch
    end = start + samples_per_epoch

    if signal_col not in signal_df.columns:
        return np.array([])

    return signal_df[signal_col].iloc[start:end].to_numpy(dtype=float)


def get_hrv_window_signal_segment(signal_df, epoch_idx, signal_col, fs=BVP_FS):
    samples_per_epoch = int(EPOCH_SECONDS * fs)
    hrv_window_samples = int(HRV_WINDOW_SECONDS * fs)

    epoch_start = epoch_idx * samples_per_epoch
    epoch_end = epoch_start + samples_per_epoch

    if HRV_WINDOW_MODE == "past":
        start = max(0, epoch_end - hrv_window_samples)
        end = epoch_end

    elif HRV_WINDOW_MODE == "centered":
        epoch_center = epoch_start + samples_per_epoch // 2
        start = max(0, epoch_center - hrv_window_samples // 2)
        end = epoch_center + hrv_window_samples // 2

    else:
        raise ValueError("HRV_WINDOW_MODE debe ser 'past' o 'centered'.")

    if signal_col not in signal_df.columns:
        return np.array([])

    return signal_df[signal_col].iloc[start:end].to_numpy(dtype=float)


# ============================================================
# CÁLCULO POR PACIENTE
# ============================================================

def compute_signal_based_features_for_subject(subject_features, subject_signals):
    """
    Calcula las nuevas features para un paciente.

    subject_features:
        DataFrame con las épocas de ese paciente.

    subject_signals:
        DataFrame con las señales preprocesadas de ese paciente.
    """
    bvp_rows = []
    hrv_rows = []

    n_epochs = len(subject_features)

    expected_min_samples = n_epochs * EPOCH_SECONDS * BVP_FS

    if len(subject_signals) < expected_min_samples:
        print(
            f"[AVISO] La señal tiene menos muestras de las esperadas. "
            f"Épocas: {n_epochs}, muestras esperadas aprox.: {expected_min_samples}, "
            f"muestras disponibles: {len(subject_signals)}"
        )

    for local_epoch_idx in range(n_epochs):
        bvp_segment = get_epoch_signal_segment(
            subject_signals,
            local_epoch_idx,
            BVP_COL,
            fs=BVP_FS
        )

        ibi_window = get_hrv_window_signal_segment(
            subject_signals,
            local_epoch_idx,
            IBI_COL,
            fs=BVP_FS
        )

        bvp_rows.append(
            compute_bvp_morphology_features(
                bvp_segment,
                fs=BVP_FS
            )
        )

        hrv_rows.append(
            compute_hrv_frequency_features_from_ibi(
                ibi_window,
                interp_fs=HRV_INTERP_FS
            )
        )

    bvp_df = pd.DataFrame(bvp_rows, index=subject_features.index)
    hrv_df = pd.DataFrame(hrv_rows, index=subject_features.index)

    return pd.concat([bvp_df, hrv_df], axis=1)


def add_signal_based_features_from_folder(features_df, signals_folder):
    file_index = build_signal_file_index(signals_folder)

    all_new_features = []

    for subject_id, subject_features in features_df.groupby(SUBJECT_COL, sort=False):
        signal_file = find_signal_file_for_subject(subject_id, file_index)

        print(f"\nProcesando sujeto {subject_id}")

        if signal_file is None:
            print(
                f"[AVISO] No se encontró CSV de señales para subject_id={subject_id}. "
                f"Se rellenan las nuevas features con NaN."
            )

            empty_cols = list(empty_bvp_morphology_features().keys()) + list(empty_hrv_frequency_features().keys())

            nan_df = pd.DataFrame(
                np.nan,
                index=subject_features.index,
                columns=empty_cols
            )

            all_new_features.append(nan_df)
            continue

        print(f"CSV de señales: {signal_file}")

        subject_signals = pd.read_csv(signal_file)

        subject_signals = subject_signals.reset_index(drop=True)

        new_features_subject = compute_signal_based_features_for_subject(
            subject_features=subject_features,
            subject_signals=subject_signals
        )

        all_new_features.append(new_features_subject)

    new_features_df = pd.concat(all_new_features, axis=0).sort_index()

    enriched_df = pd.concat([features_df, new_features_df], axis=1)

    return enriched_df


def add_sleep_wake_label(
    input_csv_path,
    output_csv_path,
    label_col="etiqueta",
    new_col="etiqueta_sleep_wake"
):
    """
    Añade una columna binaria Wake/Sleep al CSV de features.

    Mapeo:
    - W, Wake -> Wake
    - N1, N2, N3, NREM, REM -> Sleep
    """

    df = pd.read_csv(input_csv_path)

    if label_col not in df.columns:
        raise ValueError(f"No existe la columna '{label_col}' en el CSV.")

    mapping = {
        "W": "Wake",
        "Wake": "Wake",
        "WAKE": "Wake",

        "N1": "Sleep",
        "N2": "Sleep",
        "N3": "Sleep",
        "NREM": "Sleep",
        "REM": "Sleep",
        "Sleep": "Sleep",
        "SLEEP": "Sleep",
    }

    df[new_col] = df[label_col].map(mapping)

    unknown_labels = df.loc[df[new_col].isna(), label_col].unique()

    if len(unknown_labels) > 0:
        print(f"[AVISO] Etiquetas no reconocidas: {unknown_labels}")

    if output_csv_path is not None:
        df.to_csv(output_csv_path, index=False)

    return df


# ============================================================
# MAIN
# ============================================================

def main():
    print("Cargando CSV global de features por época...")
    features_df = pd.read_csv(EPOCH_FEATURES_CSV_PATH)

    if SUBJECT_COL not in features_df.columns:
        raise ValueError(f"El CSV de features debe contener la columna {SUBJECT_COL}.")
    """
    print("Añadiendo features circadianas...")
    features_df = add_circadian_features(features_df)

    print("Calculando features desde señales por paciente...")
    enriched_df = add_signal_based_features_from_folder(
        features_df=features_df,
        signals_folder=SIGNALS_FOLDER_PATH
    )"""

    enriched_df = add_sleep_wake_label(
        input_csv_path=OUTPUT_CSV_PATH,
        output_csv_path=OUTPUT_CSV_PATH,
        label_col="etiqueta_3_fases"
    )

    #print(f"\nGuardando CSV enriquecido en: {OUTPUT_CSV_PATH}")
    #enriched_df.to_csv(OUTPUT_CSV_PATH, index=False)

    print("\nProceso terminado.")
    print(f"Filas finales: {len(enriched_df)}")
    print(f"Columnas finales: {len(enriched_df.columns)}")

    added_cols = [
        col for col in enriched_df.columns
        if col.startswith("PPG_")
        or col.endswith("_5min")
        or col in [
            "epoch_idx_in_recording",
            "n_epochs_recording",
            "epoch_idx_norm",
            "time_from_start_min",
            "time_from_start_hours",
            "sin_time_night",
            "cos_time_night",
            "is_first_third",
            "is_second_third",
            "is_last_third",
            "estimated_sleep_cycle",
            "cycle_position_norm",
            "sin_sleep_cycle",
            "cos_sleep_cycle",
        ]
    ]

    print("\nNuevas columnas añadidas:")
    for col in added_cols:
        print(f" - {col}")


if __name__ == "__main__":
    main()

