from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import (
    CLASS_NAMES,
    DATA_PATH,
    EPOCHS_PER_SUBJECT,
    N_SUBJECTS,
    RANDOM_STATE,
    ensure_directories,
)


@dataclass(frozen=True)
class SubjectEffects:
    movement_scale: float
    hr_offset: float
    hrv_scale: float
    temp_offset: float
    eda_scale: float
    bvp_scale: float
    wake_proneness: float
    rem_proneness: float


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=-1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum(axis=-1, keepdims=True)


def sample_label(epoch_norm: float, rng: np.random.Generator, effects: SubjectEffects) -> str:
    wake_boost = 2.0 * np.exp(-epoch_norm / 0.07) + 1.7 * np.exp(-(1.0 - epoch_norm) / 0.08)
    wake_micro = 0.28 * (np.sin(epoch_norm * 12 * np.pi) + 1.0)
    rem_trend = 0.42 + 1.15 * epoch_norm + 0.24 * np.sin((epoch_norm - 0.1) * 5 * np.pi)
    nrem_trend = 2.45 - 0.8 * epoch_norm - 0.28 * np.sin(epoch_norm * 4 * np.pi)

    logits = np.array(
        [
            wake_boost + wake_micro + effects.wake_proneness,
            nrem_trend,
            rem_trend + effects.rem_proneness,
        ]
    )
    probabilities = softmax(logits[None, :])[0]
    return rng.choice(CLASS_NAMES, p=probabilities)


def positive_normal(rng: np.random.Generator, mean: float, std: float, minimum: float = 0.0) -> float:
    return float(max(minimum, rng.normal(mean, std)))


def generate_epoch_features(
    label: str,
    epoch_index: int,
    epochs_per_subject: int,
    rng: np.random.Generator,
    effects: SubjectEffects,
) -> dict[str, float]:
    epoch_norm = epoch_index / max(epochs_per_subject - 1, 1)
    time_from_sleep_start = epoch_index * 0.5
    late_night_factor = 0.3 + epoch_norm

    if label == "Wake":
        acc_mean = positive_normal(rng, 0.72 * effects.movement_scale, 0.22)
        acc_std = positive_normal(rng, 0.45 * effects.movement_scale, 0.16)
        acc_energy = positive_normal(rng, 1.65 * effects.movement_scale, 0.5)
        acc_p95 = positive_normal(rng, 1.0 * effects.movement_scale, 0.24)
        acc_peaks = int(np.clip(np.rint(rng.normal(4.8 * effects.movement_scale, 1.9)), 0, None))

        hr_mean = positive_normal(rng, 70 + effects.hr_offset - 3.5 * epoch_norm, 5.5, minimum=42)
        hr_std = positive_normal(rng, 5.9 * effects.hrv_scale, 1.8)
        rmssd = positive_normal(rng, 37 * effects.hrv_scale, 8.5)
        sdnn = positive_normal(rng, 47 * effects.hrv_scale, 9.0)
        ibi_std = positive_normal(rng, 44 * effects.hrv_scale, 11.0)
        bvp_std = positive_normal(rng, 0.27 * effects.bvp_scale, 0.06)
        bvp_amp = positive_normal(rng, 1.05 * effects.bvp_scale, 0.18)
        bvp_energy = positive_normal(rng, 1.4 * effects.bvp_scale, 0.28)
        temp_mean = positive_normal(rng, 33.05 + effects.temp_offset, 0.26)
        temp_std = positive_normal(rng, 0.13, 0.05)
        temp_slope = rng.normal(-0.001, 0.02)
        eda_mean = positive_normal(rng, 1.28 * effects.eda_scale, 0.26)
        eda_std = positive_normal(rng, 0.35 * effects.eda_scale, 0.11)
        eda_peaks = int(np.clip(np.rint(rng.normal(2.8 * effects.eda_scale, 1.1)), 0, None))
    elif label == "NREM":
        acc_mean = positive_normal(rng, 0.18 * effects.movement_scale, 0.08)
        acc_std = positive_normal(rng, 0.14 * effects.movement_scale, 0.06)
        acc_energy = positive_normal(rng, 0.5 * effects.movement_scale, 0.18)
        acc_p95 = positive_normal(rng, 0.29 * effects.movement_scale, 0.1)
        acc_peaks = int(np.clip(np.rint(rng.normal(1.4 * effects.movement_scale, 0.9)), 0, None))

        hr_mean = positive_normal(rng, 59 + effects.hr_offset - 2.8 * epoch_norm, 4.6, minimum=38)
        hr_std = positive_normal(rng, 4.1 * effects.hrv_scale, 1.3)
        rmssd = positive_normal(rng, 31 * effects.hrv_scale, 7.0)
        sdnn = positive_normal(rng, 40 * effects.hrv_scale, 8.0)
        ibi_std = positive_normal(rng, 37 * effects.hrv_scale, 8.5)
        bvp_std = positive_normal(rng, 0.21 * effects.bvp_scale, 0.05)
        bvp_amp = positive_normal(rng, 0.91 * effects.bvp_scale, 0.16)
        bvp_energy = positive_normal(rng, 0.98 * effects.bvp_scale, 0.19)
        temp_mean = positive_normal(rng, 33.45 + effects.temp_offset + 0.15 * epoch_norm, 0.18)
        temp_std = positive_normal(rng, 0.11, 0.04)
        temp_slope = rng.normal(0.012 - 0.008 * epoch_norm, 0.01)
        eda_mean = positive_normal(rng, 0.98 * effects.eda_scale, 0.2)
        eda_std = positive_normal(rng, 0.25 * effects.eda_scale, 0.07)
        eda_peaks = int(np.clip(np.rint(rng.normal(1.0 * effects.eda_scale, 0.7)), 0, None))
    else:
        acc_mean = positive_normal(rng, 0.23 * effects.movement_scale, 0.09)
        acc_std = positive_normal(rng, 0.17 * effects.movement_scale, 0.07)
        acc_energy = positive_normal(rng, 0.6 * effects.movement_scale, 0.22)
        acc_p95 = positive_normal(rng, 0.35 * effects.movement_scale, 0.11)
        acc_peaks = int(np.clip(np.rint(rng.normal(1.8 * effects.movement_scale, 0.9)), 0, None))

        hr_mean = positive_normal(rng, 61 + effects.hr_offset - 0.8 * epoch_norm, 5.0, minimum=40)
        hr_std = positive_normal(rng, (4.8 + 0.8 * late_night_factor) * effects.hrv_scale, 1.4)
        rmssd = positive_normal(rng, (34 + 3.5 * late_night_factor) * effects.hrv_scale, 7.5)
        sdnn = positive_normal(rng, (42 + 4.0 * late_night_factor) * effects.hrv_scale, 8.0)
        ibi_std = positive_normal(rng, (41 + 5.0 * late_night_factor) * effects.hrv_scale, 9.0)
        bvp_std = positive_normal(rng, (0.24 + 0.02 * late_night_factor) * effects.bvp_scale, 0.05)
        bvp_amp = positive_normal(rng, 0.98 * effects.bvp_scale, 0.18)
        bvp_energy = positive_normal(rng, (1.08 + 0.1 * late_night_factor) * effects.bvp_scale, 0.2)
        temp_mean = positive_normal(rng, 33.36 + effects.temp_offset + 0.1 * epoch_norm, 0.2)
        temp_std = positive_normal(rng, 0.115, 0.04)
        temp_slope = rng.normal(0.005, 0.012)
        eda_mean = positive_normal(rng, 1.1 * effects.eda_scale, 0.2)
        eda_std = positive_normal(rng, 0.27 * effects.eda_scale, 0.07)
        eda_peaks = int(np.clip(np.rint(rng.normal(1.5 * effects.eda_scale, 0.8)), 0, None))

    ibi_mean = positive_normal(rng, 60000.0 / hr_mean, 28.0, minimum=300)
    hr_min = positive_normal(rng, hr_mean - abs(rng.normal(4.8, 1.7)), 2.0, minimum=35)
    hr_max = positive_normal(rng, hr_mean + abs(rng.normal(5.4, 2.2)), 2.5, minimum=40)
    bvp_mean = rng.normal(0.0, 0.08)

    latent_motion = positive_normal(
        rng,
        0.22 * effects.movement_scale + 0.05 * np.sin(epoch_norm * 6 * np.pi),
        0.09,
    )
    latent_hrv = positive_normal(rng, 38 * effects.hrv_scale + 4 * epoch_norm, 6.5)
    latent_bvp = positive_normal(rng, 0.22 * effects.bvp_scale, 0.05)

    blend = rng.uniform(0.18, 0.34)
    acc_mean = (1 - blend) * acc_mean + blend * latent_motion
    acc_std = (1 - blend) * acc_std + blend * latent_motion
    acc_energy = (1 - blend) * acc_energy + blend * (latent_motion * 2.0)
    acc_p95 = (1 - blend) * acc_p95 + blend * (latent_motion * 1.5)
    hr_std = (1 - blend) * hr_std + blend * (latent_hrv / 8.0)
    rmssd = (1 - blend) * rmssd + blend * latent_hrv
    sdnn = (1 - blend) * sdnn + blend * (latent_hrv + 6.0)
    ibi_std = (1 - blend) * ibi_std + blend * (latent_hrv + 4.0)
    bvp_std = (1 - blend) * bvp_std + blend * latent_bvp
    bvp_energy = (1 - blend) * bvp_energy + blend * (latent_bvp * 4.0)

    return {
        "acc_mean": acc_mean,
        "acc_std": acc_std,
        "acc_energy": acc_energy,
        "acc_p95": acc_p95,
        "acc_peaks": acc_peaks,
        "hr_mean": hr_mean,
        "hr_std": hr_std,
        "hr_min": hr_min,
        "hr_max": hr_max,
        "ibi_mean": ibi_mean,
        "ibi_std": ibi_std,
        "rmssd": rmssd,
        "sdnn": sdnn,
        "bvp_mean": bvp_mean,
        "bvp_std": bvp_std,
        "bvp_amp": bvp_amp,
        "bvp_energy": bvp_energy,
        "temp_mean": temp_mean,
        "temp_std": temp_std,
        "temp_slope": temp_slope,
        "eda_mean": eda_mean,
        "eda_std": eda_std,
        "eda_peaks": eda_peaks,
        "time_from_sleep_start": time_from_sleep_start,
        "epoch_norm": epoch_norm,
    }


def subject_effects(rng: np.random.Generator) -> SubjectEffects:
    return SubjectEffects(
        movement_scale=float(rng.normal(1.0, 0.12)),
        hr_offset=float(rng.normal(0.0, 3.0)),
        hrv_scale=float(rng.normal(1.0, 0.1)),
        temp_offset=float(rng.normal(0.0, 0.2)),
        eda_scale=float(rng.normal(1.0, 0.12)),
        bvp_scale=float(rng.normal(1.0, 0.1)),
        wake_proneness=float(rng.normal(0.0, 0.25)),
        rem_proneness=float(rng.normal(0.0, 0.2)),
    )


def generate_dataset(
    n_subjects: int = N_SUBJECTS,
    epochs_per_subject: int = EPOCHS_PER_SUBJECT,
    seed: int = RANDOM_STATE,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records: list[dict[str, float | int | str]] = []

    for subject_num in range(1, n_subjects + 1):
        effects = subject_effects(rng)
        for epoch_index in range(epochs_per_subject):
            epoch_norm = epoch_index / max(epochs_per_subject - 1, 1)
            label = sample_label(epoch_norm, rng, effects)
            features = generate_epoch_features(label, epoch_index, epochs_per_subject, rng, effects)
            records.append(
                {
                    "subject_id": f"S{subject_num:02d}",
                    "epoch": epoch_index,
                    **features,
                    "label": label,
                }
            )

    return pd.DataFrame.from_records(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a synthetic sleep-stage dataset.")
    parser.add_argument("--n-subjects", type=int, default=N_SUBJECTS)
    parser.add_argument("--epochs-per-subject", type=int, default=EPOCHS_PER_SUBJECT)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_directories()
    dataset = generate_dataset(
        n_subjects=args.n_subjects,
        epochs_per_subject=args.epochs_per_subject,
        seed=args.seed,
    )
    dataset.to_csv(DATA_PATH, index=False)

    class_distribution = dataset["label"].value_counts(normalize=True).sort_index()
    print(f"Synthetic dataset saved to {DATA_PATH}")
    print(f"Shape: {dataset.shape}")
    print("Class distribution:")
    for label, ratio in class_distribution.items():
        print(f"  - {label}: {ratio:.3f}")


if __name__ == "__main__":
    main()
