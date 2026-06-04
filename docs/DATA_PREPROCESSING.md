# Plan de extraccion y preprocesado

## Objetivo del primer bloque

Construir un pipeline que lea varias bases de datos heterogeneas y produzca una representacion comun para entrenar un modelo supervisado de clasificacion de sueno con wearable como entrada y PSG como ground truth.

La revision detallada de papers y fuentes primarias esta en [LITERATURE_PREPROCESSING_REVIEW.md](LITERATURE_PREPROCESSING_REVIEW.md). Este documento queda como plan operativo.

## Senales objetivo

Entradas del modelo:

- `acc_x`, `acc_y`, `acc_z`: accelerometria triaxial.
- `ppg` o `bvp`: senal optica cruda cuando exista.
- `hr`: frecuencia cardiaca derivada de PPG cuando el dataset no proporcione PPG crudo.
- `modality_mask`: indicadores de disponibilidad de cada modalidad.

Etiqueta:

- `sleep_stage`: etapa del sueno anotada desde PSG.

Metadatos minimos:

- `dataset`.
- `subject_id`.
- `recording_id`.
- `epoch_start`.
- `epoch_seconds`.
- `sampling_rate_acc`.
- `sampling_rate_ppg`.
- `label_source`.

## Schema comun propuesto

Cada dataset se normalizara a epocas de 30 segundos. La salida procesada deberia separar datos y metadatos:

- `data/processed/epochs.parquet`: indice global de epocas y metadatos.
- `data/processed/signals/<dataset>/<recording_id>.npz`: arrays de senal por registro.
- `data/processed/reports/`: resumenes de cobertura, clases, alineacion y splits.

Columnas iniciales para `epochs.parquet`:

- `dataset`.
- `subject_id`.
- `recording_id`.
- `epoch_index`.
- `epoch_start_seconds`.
- `label_raw`.
- `label_5class`.
- `label_3class`.
- `has_acc`.
- `has_ppg`.
- `has_hr`.
- `coverage_acc`.
- `coverage_ppg`.
- `coverage_hr`.
- `quality_flag`.
- `signal_path`.

## Mapeo inicial de clases

Cinco clases:

- `W`: wake.
- `N1`: non-REM 1.
- `N2`: non-REM 2.
- `N3`: non-REM 3.
- `R`: REM.

Tres clases:

- `W`: wake.
- `NREM`: N1, N2, N3.
- `REM`: R.

Labels a revisar segun dataset:

- `N4` deberia mapearse normalmente a `N3` si aparece en anotaciones antiguas.
- `UNKNOWN`, `MOVEMENT`, `?` o equivalentes deben quedar excluidos o marcados como no entrenables.

## Decisiones tecnicas pendientes

Frecuencia objetivo:

- ACC: remuestrear a 30 Hz tras filtrado anti-alias.
- PPG/BVP crudo: mantener inicialmente a 64 Hz para DREAMT y registrar la frecuencia por epoch.
- HR derivada de PPG: alinear a 1 Hz o resumir por epoca.

Normalizacion:

- Por registro: robusta ante diferencias entre dispositivos.
- Por sujeto: util si hay varias noches por sujeto.
- Global por dataset: mas simple, pero mas sensible a dominios.

Ventanas:

- Epoca PSG de 30 s como unidad basica.
- Contexto secuencial de varias epocas para el modelo principal.
- Etiqueta central si se usan ventanas multi-epoca.

Splits:

- Siempre por sujeto, no por epoca.
- Mantener test externo por dataset si los tamanos lo permiten.
- Evitar que una misma persona aparezca en train y validation.

## Orden de trabajo propuesto

1. Inventariar DREAMT con el notebook actual.
2. Convertir DREAMT en extractor scriptable.
3. Definir el schema comun con DREAMT.
4. Repetir inventario para cada dataset restante.
5. Implementar extractores dataset por dataset.
6. Generar reporte comparativo.
7. Cerrar decisiones de frecuencia, labels y splits.

Orden recomendado tras la revision:

1. DREAMT: ACC + BVP/PPG crudo.
2. SleepAccel: ACC + HR derivada de PPG.
3. Newcastle: ACC-only con control de sujeto y muneca izquierda/derecha.
4. ODC-TBI: ACC-only, pendiente de confirmar acceso y estructura exacta.

## Validaciones minimas

- Numero de sujetos y registros por dataset.
- Duracion total usable.
- Conteo de epocas por clase.
- Porcentaje de epocas con ACC y PPG disponibles.
- Distribucion de frecuencias de muestreo.
- Checks de alineacion temporal entre senal y etiqueta.
- Ausencia de solapamiento de sujetos entre splits.

## Vinculo con la memoria

Cada extractor debe producir informacion reutilizable para la memoria:

- Tabla de caracteristicas del dataset.
- Figura o tabla de balance de clases.
- Diagrama del pipeline de preprocesado.
- Justificacion de criterios de exclusion.
