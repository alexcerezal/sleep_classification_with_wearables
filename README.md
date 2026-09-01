# Critical and Interpretable Analysis of Sleep Stage Classification Using Wearable Device Signals

This project studies supervised sleep-stage classification using physiological signals collected by wearable devices. Its current workflow is focused on the DREAMT dataset and uses polysomnography (PSG) sleep-stage annotations as ground truth. Wearable signals such as blood-volume pulse (BVP), inter-beat interval (IBI), heart rate (HR), triaxial acceleration (ACC), electrodermal activity (EDA), and temperature are processed into features that can be used to train and interpret machine-learning classifiers.

The complete academic and technical documentation is available in [doc.pdf](doc/doc.pdf). This README provides only a functional overview of the principal source files.

## Processing workflow

The main data flow is:

1. `preprocess_dreamt_signal.py` cleans a subject's raw DREAMT signal CSV.
2. `feature_extraction.py` divides the cleaned signals into 30-second epochs and produces one feature row per epoch.
3. `model_pipeline.py` prepares the feature table, trains a classifier, and evaluates its predictions.
4. `analisis_shap.py` can explain a trained tree-based model with SHAP values.

Input, output, and result paths are currently set through constants near the beginning of the scripts and should be adjusted before execution.

## `src/preprocess_dreamt_signal.py`

This module preprocesses one DREAMT subject CSV, with its signal-cleaning logic adapted from DREAMT_FE.

It validates and interpolates the BVP signal, then applies a Chebyshev type II band-pass filter. Peaks are detected in the filtered BVP signal and peaks that are too close together are discarded. The remaining peaks are used to reconstruct sample-aligned IBI values and derive HR in one-second blocks. When acceleration columns are present, the script also marks likely motion artifacts from abrupt changes in the three ACC axes.

The output is a new CSV containing the filtered BVP and recalculated IBI and HR columns. If `KEEP_ORIGINAL_COLUMNS` is enabled, the original BVP, IBI, and HR values are retained in columns with a `_raw` suffix.

## `src/feature_extraction.py`

This module converts all preprocessed subject CSVs in the configured input directory into a single epoch-level feature table.

Records are split into standard 30-second epochs using timestamps when available, or the sample index and configured sampling frequency otherwise. The sleep-stage label for each epoch is selected by majority vote and normalized. In addition to the original stage, the output includes three-class labels (`W`, `REM`, and `NREM`) and four-class labels (`W`, `REM`, `Light_Sleep`, and `Deep_Sleep`).

For every epoch, the module calculates:

- BVP statistics, distribution measures, Hjorth mobility and complexity, and HRV measures obtained with NeuroKit2.
- Basic HR and IBI statistics.
- Robust summaries of each acceleration axis, summaries of their median absolute deviations, and a combined activity index.
- EDA skin-conductance-response summaries, including height, amplitude, rise time, and recovery time.
- Basic temperature statistics.

It writes the combined feature CSV, a per-subject extraction report, and—when applicable—a CSV listing files that could not be processed.

## `src/model_pipeline.py`

This module contains the training and evaluation workflow for the epoch feature table. It supports three-class sleep staging or binary sleep/wake classification and can build either a Random Forest or LightGBM pipeline.

The module includes utilities to remove constant and highly correlated features, compare class-balancing strategies (no balancing, class weights, SMOTE, or ADASYN), tune hyperparameters with subject-aware `GroupKFold`, and evaluate with grouped cross-validation or leave-one-subject-out validation. Its evaluation output includes a confusion matrix, classification report, accuracy, balanced accuracy, macro and weighted F1 scores, and Matthews correlation coefficient.

In the current `main()` configuration, three-class classification and Random Forest are selected. Missing values are median-imputed, ADASYN is applied to the complete dataset, and a stratified 80/20 split is used for the final reported evaluation. The alternative balance-selection, hyperparameter-search, grouped-validation, and SHAP sections are present but commented out. This distinction is important when interpreting the current results: the module provides subject-aware validation helpers, but the active final block does not use a subject-separated test set.

## `src/analisis_shap.py`

This module provides post-hoc interpretation for trained tree-based classifiers, including Random Forest and LightGBM models. It accepts either a fitted model or a fitted pipeline, applies the pipeline's imputer when present, and uses `shap.TreeExplainer` to calculate feature contributions.

Because SHAP output shapes differ across models and library versions, the module normalizes binary and multiclass outputs into a common per-class format. It then generates:

- Global mean absolute SHAP feature importance.
- Feature importance tables and summary plots for each class.
- Separate global analyses for correct and incorrect predictions.
- A CSV identifying each sample's true label, predicted label, and correctness.

All tables and plots are saved in the output directory passed to `run_tree_shap_analysis()`.

## Extended documentation

For the project motivation, methodology, dataset details, experimental design, results, and full discussion, see [the complete project report](doc/doc.pdf).
