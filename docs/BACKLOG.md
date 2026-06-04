# Backlog tipo issues

Este backlog funciona como lista local de issues mientras decidimos si queremos replicarlo en GitHub Issues.

## Milestone M0. Repo y entorno

- [ ] ISSUE-001: Crear entorno Python reproducible.
- [ ] ISSUE-002: Definir estructura de carpetas para `src/`, `configs/`, `notebooks/` y scripts.
- [ ] ISSUE-003: Crear `requirements.txt` o `pyproject.toml`.
- [ ] ISSUE-004: Anadir comandos basicos de ejecucion y validacion.

## Milestone M1. Inventario de datasets

- [ ] ISSUE-005: Documentar estructura local esperada de DREAMT.
- [ ] ISSUE-006: Documentar estructura local esperada de psg-newcastle-db.
- [ ] ISSUE-007: Documentar estructura local esperada de odc-tbi.
- [ ] ISSUE-008: Documentar estructura local esperada de Motion and Heart Rate from Wearable Devices with PSG.
- [ ] ISSUE-009: Crear tabla comparativa de senales, labels, frecuencias y sujetos.

## Milestone M2. Extractores por dataset

- [ ] ISSUE-010: Convertir el notebook DREAMT en extractor reproducible.
- [ ] ISSUE-011: Implementar extractor psg-newcastle-db.
- [ ] ISSUE-012: Implementar extractor odc-tbi.
- [ ] ISSUE-013: Implementar extractor Motion and Heart Rate with PSG.
- [ ] ISSUE-014: Validar alineacion entre senales wearable y labels PSG.

## Milestone M3. Esquema comun de datos

- [ ] ISSUE-015: Definir schema comun de epocas.
- [ ] ISSUE-016: Definir mapeo de labels 5 clases y 3 clases.
- [ ] ISSUE-017: Definir formato de salida procesada.
- [ ] ISSUE-018: Crear reporte de cobertura y balance de clases.
- [ ] ISSUE-034: Definir `modality_mask` para ACC, PPG crudo y HR derivada de PPG.
- [ ] ISSUE-035: Separar semanticamente PPG/BVP crudo de HR derivada en schema, memoria y modelo.

## Milestone M4. Preprocesado y splits

- [ ] ISSUE-019: Definir frecuencia objetivo de ACC y PPG/BVP.
- [ ] ISSUE-020: Implementar normalizacion.
- [ ] ISSUE-021: Implementar filtros de calidad y missing data.
- [ ] ISSUE-022: Crear splits por sujeto y por dataset.
- [ ] ISSUE-023: Generar dataset de entrenamiento final.
- [ ] ISSUE-036: Validar drift y alineacion temporal en DREAMT.
- [ ] ISSUE-037: Implementar regla de segmentos continuos para SleepAccel.
- [ ] ISSUE-038: Garantizar que las dos munecas de Newcastle no crucen splits.

## Milestone M5. Modelado

- [ ] ISSUE-024: Implementar baseline mayoritario.
- [ ] ISSUE-025: Implementar baseline de features clasicas.
- [ ] ISSUE-026: Implementar CNN temporal simple.
- [ ] ISSUE-027: Implementar CNN + Mamba.
- [ ] ISSUE-028: Evaluar modelos con protocolo comun.

## Milestone M6. Memoria

- [ ] ISSUE-029: Redactar introduccion y objetivos.
- [ ] ISSUE-030: Redactar seccion de datasets.
- [ ] ISSUE-031: Redactar metodologia de preprocesado.
- [ ] ISSUE-032: Redactar metodologia experimental.
- [ ] ISSUE-033: Redactar resultados, discusion y conclusiones.
