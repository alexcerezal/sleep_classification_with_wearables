# Literature-informed preprocessing plan

## Scope

This note summarizes the practical preprocessing decisions derived from the local papers in `references/` and the official dataset descriptions. The goal is not to reproduce any paper exactly, but to choose a defensible pipeline for supervised sleep-stage classification from wearable signals, using PSG-derived labels as ground truth.

## Key references reviewed

Local references:

- `references/accelerest.pdf`: wrist accelerometry sleep staging across TBI, DREAMT, STAGES, Newcastle, Health and SleepAccel.
- `references/mamba.pdf`: Mamba-based sleep staging with multimodal wearable signals, including PPG and accelerometry.
- `references/CrossFusionSleepNet.pdf`: multimodal sleep staging with 30 s PSG epochs, AASM mapping and time-frequency preprocessing.
- `references/Interpretable_feature-based_machine_learning_for_a.pdf`: PPG feature-based staging.
- `references/DNN_Sleep_Stage_Classification_and_Sleep_Apnea_Detection_Using_Wrist-Worn_Consumer_Sleep_Technologies.pdf`: transfer learning from wearable signals.
- `references/Programar CNN-Mamba-2 en Windows (CPU).pdf` and `references/resumenIA.pdf`: project-specific synthesis notes.

Primary dataset sources:

- DREAMT, PhysioNet: https://physionet.org/content/dreamt/
- Motion and heart rate from a wrist-worn wearable and labeled sleep from polysomnography, PhysioNet: https://physionet.org/content/sleep-accel/1.0.0/
- Newcastle polysomnography and accelerometer data, Zenodo: https://zenodo.org/records/1160410

## Dataset reality check

The four datasets are not modality-equivalent:

| Dataset | Wearable device | Usable wearable signals for this project | PSG labels | Main consequence |
| --- | --- | --- | --- | --- |
| DREAMT | Empatica E4 | ACC x/y/z, BVP/PPG, HR/IBI, EDA, TEMP | Yes | Best fit for ACC + raw PPG supervised training. |
| Motion and Heart Rate / SleepAccel | Apple Watch | ACC and HR derived from PPG, not raw PPG | Yes | Useful for ACC + PPG-derived cardio signal, but not raw PPG morphology. |
| Newcastle | GENEActiv | ACC x/y/z only, left and right wrists | Yes | Useful for ACC-only training/evaluation/domain robustness. |
| ODC-TBI / TBI cohort | Actiwatch/Actiwatch Spectrum in AcceleRest references | ACC x/y/z only in the sleep-staging references reviewed | Yes | Useful for ACC-only training/evaluation; likely not raw PPG. |

Implication: a strict "ACC + raw PPG only" model can be trained mainly on DREAMT unless we add more raw-PPG datasets. If we want to exploit all four mentioned datasets, the model and data schema should support missing modalities.

## Recommended strategy

Use a canonical multimodal schema with modality masks:

- `acc`: always expected when the dataset supports it.
- `ppg`: raw PPG/BVP when available.
- `hr`: PPG-derived heart rate when raw PPG is unavailable but HR exists.
- `modality_mask`: flags which modalities are present for each epoch.

This lets us support three training regimes:

- ACC-only baseline across DREAMT, SleepAccel, Newcastle and TBI.
- ACC + cardio model using raw PPG for DREAMT and HR for SleepAccel.
- ACC + raw PPG final model trained on DREAMT and any later dataset with raw PPG.

The final architecture can still be CNN + Mamba: modality-specific CNN encoders convert raw epoch signals into embeddings, then Mamba models temporal context across epochs.

## Label preprocessing

Adopt 30-second PSG epochs as the canonical label unit.

Map labels to five classes:

- `W -> 0`
- `N1 -> 1`
- `N2 -> 2`
- `N3 -> 3`
- `R` or `REM -> 4`

Compatibility rules:

- `N4 -> N3`, following AASM-style consolidation used in modern sleep staging papers.
- `MOVEMENT`, `UNKNOWN`, missing labels, artefact labels and ambiguous annotations should be excluded from supervised training.
- Keep both `label_5class` and `label_3class`.
- Three-class mapping: `W`, `NREM`, `REM`.
- Optional four-class mapping for comparison: `W`, `light` = `N1 + N2`, `deep` = `N3`, `REM`.

The five-class task is the main target, but reporting three-class results is advisable because wearable-only models often struggle with `N1` and sometimes with `N3`.

## Signal preprocessing

### Accelerometry

Recommended canonical representation:

- Convert units to `g`.
- Resample to 30 Hz after anti-alias filtering.
- Store 30 s epochs as shape `(900, 3)`.
- Keep axis order as supplied after unit normalization.
- Add optional derived channels only outside the strict input contract: vector magnitude, ENMO, activity counts, or estimated respiratory/pulse components.

Rationale:

- AcceleRest resamples heterogeneous wrist accelerometry to 30 Hz and uses 30-second patches.
- 30 Hz is enough for sleep-stage movement patterns and reduces compute.
- It handles DREAMT ACC at 32 Hz, Newcastle at 85.7 Hz, TBI near 100 Hz and SleepAccel variable sampling.

Quality checks:

- Mark gaps larger than 1 s.
- Require at least 80% valid samples per epoch for exploratory processing; tighten to 90% for final experiments if enough data remains.
- Drop recordings with very short usable duration. AcceleRest used a 4-hour contiguous segment rule for SleepAccel; that is a good initial rule for variable-sampling data.
- Keep left and right wrist Newcastle recordings as separate recordings but ensure the same subject never crosses train/validation/test.

### PPG/BVP

Recommended canonical representation:

- Use raw BVP/PPG when available.
- Prefer native or source-aligned rate where possible.
- For DREAMT, use `data_100Hz` if PSG channel alignment is needed for verification; otherwise use `data_64Hz` for wearable-only raw BVP fidelity.
- Store a target PPG rate separately from ACC, initially `64 Hz` for raw BVP.
- Store 30 s epochs as shape `(1920, 1)` at 64 Hz.

Filtering and quality:

- Apply a light band-pass filter around the physiological pulse band, initially `0.5-5 Hz`.
- Compute a simple PPG quality flag before model training.
- Start with coverage and amplitude sanity checks; later add pulse-template or peak-consistency SQI if needed.
- Do not interpolate long PPG gaps into plausible-looking physiology.

### Heart rate

SleepAccel provides HR derived from Apple Watch PPG rather than raw PPG. Treat it as a separate modality:

- Resample HR to 1 Hz or align by timestamp to each 30 s epoch.
- Store summary features per epoch and optionally a 30-point sequence.
- Do not call HR "raw PPG" in the memory.

## Time alignment

All extractors should output epoch-level rows aligned to PSG labels.

Rules:

- Use PSG hypnogram timestamps as the authoritative label grid.
- Align wearable data by timestamps when available.
- Track clock drift corrections explicitly in metadata.
- DREAMT deserves a special check because AcceleRest reports a clock drift between accelerometry and PSG that they corrected by resampling the PSG and hypnogram.
- For datasets with separate files per signal, produce an alignment report per recording.

## Splits and evaluation

Mandatory:

- Split by subject, never by epoch.
- Keep all recordings from the same subject in the same split.
- For Newcastle, keep left and right wrist recordings for one subject in the same split.

Recommended initial protocol:

- DREAMT: subject-level train/validation/test or train/test matching literature when useful.
- SleepAccel: use as external evaluation if the final model can consume HR instead of raw PPG.
- Newcastle: ACC-only external evaluation.
- TBI: large ACC-only training/evaluation source if accessible.

Metrics:

- Accuracy.
- Macro F1.
- Cohen's kappa.
- Per-class precision, recall and F1.
- Confusion matrix.

Macro F1 and kappa should be emphasized in the memory because class imbalance is substantial and N1 is usually difficult.

## Proposed processed-data layout

```text
data/
  raw/
    dreamt/
    sleep_accel/
    psg_newcastle/
    odc_tbi/
  interim/
    <dataset>/
  processed/
    epochs.parquet
    signals/
      <dataset>/
        <recording_id>.npz
    reports/
      dataset_summary.json
      class_balance.csv
      alignment_report.csv
      split_summary.csv
```

The global `epochs.parquet` should index all usable epochs and point to signal arrays by `signal_path`, `recording_id` and epoch offsets.

## Recommended implementation order

1. Freeze the canonical schema.
2. Convert the current DREAMT notebook into a reusable extractor.
3. Produce DREAMT summary reports and use them to validate the schema.
4. Implement SleepAccel with ACC + HR support.
5. Implement Newcastle as ACC-only with paired-wrist subject handling.
6. Clarify ODC-TBI access and exact file format before implementation.
7. Add a dataset-composition report to guide model decisions.

## Working decision for the next coding block

Implement a dataset-adapter layer:

- `src/data/datasets/dreamt.py`
- `src/data/datasets/sleep_accel.py`
- `src/data/datasets/newcastle.py`
- `src/data/datasets/odc_tbi.py`
- `src/data/schema.py`
- `src/data/preprocess.py`

Each adapter should read raw files and emit the same canonical epoch table plus signal arrays. That keeps the model code independent from the quirks of PhysioNet, Zenodo or future datasets.
